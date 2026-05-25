import cv2
import time
import numpy as np
from multiprocessing import Process, Queue
import os
import psutil
import onnxruntime as ort

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ========================================================================================
# --- VULCAN CONFIGURATION (v1.1 FINAL) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "crowd_video.mp4")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "p2pnet.onnx")
PERFORMANCE_RESIZE_FACTOR = 0.9
BLUR_THRESHOLD = 100.0
TARGET_ANALYSIS_FPS = 0.5 # <-- NEW: Analyze 1 frame every 2 seconds (1 / 2.0 = 0.5 FPS)

# --- Chaos Score Tiers ---
NORMAL_COUNT = 50
MAX_SAFE_COUNT = 800
SURGE_TIME_WINDOW_SECONDS = 10
CRITICAL_SURGE_COUNT = 50
MAX_TRACKING_DISTANCE_PIXELS = 50

# --- Chaos Score Weights ---
W_DENSITY = 0.5
W_FLOW = 0.2
W_SURGE = 0.3

# --- System ---
PROCESS_AFFINITY = {'ingestion': [0], 'ai_worker': [1, 2, 3], 'ui': [0]}
# ========================================================================================

def check_blur(frame, threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def calculate_flow_norm(prev_points, current_points):
    from scipy.spatial.distance import cdist
    if prev_points is None or len(prev_points) < 2 or len(current_points) < 2: return 0.0
    distances = cdist(current_points, prev_points)
    prev_indices = np.argmin(distances, axis=1)
    min_distances = distances[np.arange(len(current_points)), prev_indices]
    tracked_mask = min_distances < MAX_TRACKING_DISTANCE_PIXELS
    if np.sum(tracked_mask) < 2: return 50.0
    vectors = current_points[tracked_mask] - prev_points[prev_indices[tracked_mask]]
    avg_vector = np.mean(vectors, axis=0)
    avg_vector_norm = np.linalg.norm(avg_vector)
    if avg_vector_norm == 0: return 100.0
    vector_norms = np.linalg.norm(vectors, axis=1)
    non_zero_mask = vector_norms > 0
    if not np.any(non_zero_mask): return 100.0
    similarities = np.zeros(len(vectors))
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_sim = np.dot(vectors[non_zero_mask], avg_vector) / (vector_norms[non_zero_mask] * avg_vector_norm)
    similarities[non_zero_mask] = cos_sim
    avg_similarity = np.mean(similarities[non_zero_mask])
    return max(0, min(100, (1 - avg_similarity) * 50))

# --- PIPELINE PROCESSES ---
def ingestion_process(video_path, frame_queue, result_queue):
    p = psutil.Process(os.getpid())
    try: p.cpu_affinity(PROCESS_AFFINITY['ingestion'])
    except: pass
    
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"[Ingestion] Error: Could not open video file.")
        frame_queue.put("STOP"); result_queue.put("STOP")
        return

    # --- NEW: Frame Skipping Logic ---
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    if video_fps == 0: video_fps = 30 # Default if FPS not available
    frames_to_skip = int(video_fps / TARGET_ANALYSIS_FPS)
    frame_counter = 0
    print(f"[Ingestion] Process started. Video FPS: {video_fps:.1f}. Analyzing 1 frame every {frames_to_skip} frames.")

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret: break

            # This is the core of true frame skipping
            if frame_counter % frames_to_skip == 0:
                if not check_blur(frame, BLUR_THRESHOLD):
                    timestamp_ms = video_capture.get(cv2.CAP_PROP_POS_MSEC)
                    if PERFORMANCE_RESIZE_FACTOR < 1.0:
                        h, w, _ = frame.shape
                        frame = cv2.resize(frame, (int(w * PERFORMANCE_RESIZE_FACTOR), int(h * PERFORMANCE_RESIZE_FACTOR)))
                    
                    data_packet = {'frame': frame, 'frame_num': frame_counter, 'timestamp_s': timestamp_ms / 1000.0}
                    try: frame_queue.get_nowait()
                    except: pass
                    frame_queue.put(data_packet)
            
            frame_counter += 1
    finally:
        frame_queue.put("STOP")
        result_queue.put("STOP")
        video_capture.release()
        print("[Ingestion] Process finished.")

def ai_worker_process_onnx(frame_queue, result_queue):
    # This function remains the same
    p = psutil.Process(os.getpid())
    try: p.cpu_affinity(PROCESS_AFFINITY['ai_worker'])
    except: pass
    print(f"[AI Worker] Process started on CPU {p.cpu_affinity()}.")
    print("[AI Worker] Initializing ONNX Runtime session...")
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = ort_session.get_inputs()[0].name
    _, _, h, w = ort_session.get_inputs()[0].shape
    previous_points = None
    historical_counts = []
    print("[AI Worker] Waiting for frames...")
    while True:
        data_packet = frame_queue.get()
        if isinstance(data_packet, str) and data_packet == "STOP":
            result_queue.put("STOP")
            break
        start_time = time.time()
        frame = data_packet['frame']
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img_rgb, (w, h))
        input_tensor = resized_image.astype(np.float32) / 255.0
        input_tensor = (input_tensor - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        input_tensor = np.expand_dims(input_tensor.transpose(2, 0, 1), 0).astype(np.float32)
        pred_logits, pred_points = ort_session.run(None, {input_name: input_tensor})
        exp_logits = np.exp(pred_logits[0])
        scores = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True))[:, 1]
        points = pred_points[0]
        current_points = points[scores > 0.5]
        current_count = current_points.shape[0]
        d_norm = ((current_count - NORMAL_COUNT) / (MAX_SAFE_COUNT - NORMAL_COUNT)) * 100.0 if NORMAL_COUNT < current_count < MAX_SAFE_COUNT else (0.0 if current_count <= NORMAL_COUNT else 100.0)
        f_norm = calculate_flow_norm(previous_points, current_points)
        previous_points = current_points
        current_time = time.time()
        historical_counts.append({'time': current_time, 'count': current_count})
        while historical_counts and current_time - historical_counts[0]['time'] > SURGE_TIME_WINDOW_SECONDS:
            historical_counts.pop(0)
        surge = 0
        if len(historical_counts) > 1: surge = current_count - historical_counts[0]['count']
        s_norm = min(max(0, surge / CRITICAL_SURGE_COUNT * 100), 100)
        chaos_score = (W_DENSITY * d_norm) + (W_FLOW * f_norm) + (W_SURGE * s_norm)
        if chaos_score >= 75: status = "🔴 HIGH RISK 🔴"
        elif chaos_score >= 40: status = "🟡 MODERATE RISK 🟡"
        else: status = "🟢 SAFE 🟢"
        result = {'count': current_count, 'd_norm': d_norm, 'f_norm': f_norm, 's_norm': s_norm,
                  'score': chaos_score, 'status': status, 'latency': time.time() - start_time,
                  'frame_num': data_packet['frame_num'], 'timestamp_s': data_packet['timestamp_s']}
        try: result_queue.get_nowait()
        except: pass
        result_queue.put(result)
    print("[AI Worker] Process finished.")

def ui_process(result_queue):
    # This function remains the same
    print(f"[UI] Process started.")
    while True:
        try:
            result = result_queue.get(timeout=15)
            if isinstance(result, str) and result == "STOP": break
            print(f"\n--- Analysis [Frame: {result['frame_num']} | Time: {result['timestamp_s']:.1f}s] (Latency: {result['latency']:.2f}s) ---")
            print(f"Live Count: {result['count']} | D:{result['d_norm']:.1f} | F:{result['f_norm']:.1f} | S:{result['s_norm']:.1f}")            
            print(f"CHAOS SCORE: {result['score']:.2f} | STATUS: {result['status']}")
        except:
            print("[UI] No new data from AI Worker. Shutting down.")
            break
    print("[UI] Process finished.")

if __name__ == "__main__":
    # This block remains the same
    print("--- Initializing Vulcan ONNX System (v1.1) ---")
    frame_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=1)
    p_ingestion = Process(target=ingestion_process, args=(VIDEO_PATH, frame_queue, result_queue))
    p_ai = Process(target=ai_worker_process_onnx, args=(frame_queue, result_queue))
    p_ui = Process(target=ui_process, args=(result_queue,))
    try:
        p_ingestion.start(); p_ai.start(); p_ui.start()
        p_ingestion.join(); p_ai.join(); p_ui.join()
    except KeyboardInterrupt:
        print("\n[System] Keyboard interrupt detected. Shutting down processes.")
        p_ingestion.terminate(); p_ai.terminate(); p_ui.terminate()
    print("--- Vulcan System Shut Down ---")