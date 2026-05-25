import cv2
import time
import numpy as np
from multiprocessing import Process, Queue, Manager
import os
import psutil
import onnxruntime as ort
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import logging
from waitress import serve
from threading import Thread

# Suppress unnecessary logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ========================================================================================
# --- TRINETRA CONFIGURATION (v1.2) ---
VIDEO_PATH = "crowd_video.mp4"
ONNX_MODEL_PATH = "p2pnet.onnx"
PERFORMANCE_RESIZE_FACTOR = 0.75
BLUR_THRESHOLD = 100.0
TARGET_ANALYSIS_FPS = 0.5

NORMAL_COUNT = 50
MAX_SAFE_COUNT = 800
SURGE_TIME_WINDOW_SECONDS = 10
CRITICAL_SURGE_COUNT = 50
MAX_TRACKING_DISTANCE_PIXELS = 50

W_DENSITY = 0.5
W_FLOW = 0.2
W_SURGE = 0.3

PROCESS_AFFINITY = {'ingestion': [0], 'ai_worker': [1, 2, 3], 'api_server': [0]}
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
def ingestion_process(video_path, frame_queue):
    p = psutil.Process(os.getpid())
    try: p.cpu_affinity(PROCESS_AFFINITY['ingestion'])
    except: pass
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"[Ingestion] Error: Could not open video file.")
        frame_queue.put("STOP")
        return
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    if video_fps == 0: video_fps = 30
    frames_to_skip = int(video_fps / TARGET_ANALYSIS_FPS)
    frame_counter = 0
    print(f"[Ingestion] Process started. Analyzing 1 frame every {frames_to_skip} frames.")
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            if frame_counter % frames_to_skip == 0:
                if not check_blur(frame, BLUR_THRESHOLD):
                    if PERFORMANCE_RESIZE_FACTOR < 1.0:
                        h, w, _ = frame.shape
                        frame = cv2.resize(frame, (int(w * PERFORMANCE_RESIZE_FACTOR), int(h * PERFORMANCE_RESIZE_FACTOR)))
                    data_packet = {'frame': frame}
                    try: frame_queue.get_nowait()
                    except: pass
                    frame_queue.put(data_packet)
            frame_counter += 1
    finally:
        frame_queue.put("STOP")
        video_capture.release()
        print("[Ingestion] Process finished.")

def ai_worker_process_onnx(frame_queue, result_queue):
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
        surge = current_count - historical_counts[0]['count'] if len(historical_counts) > 1 else 0
        s_norm = min(max(0, surge / CRITICAL_SURGE_COUNT * 100), 100)
        chaos_score = (W_DENSITY * d_norm) + (W_FLOW * f_norm) + (W_SURGE * s_norm)
        status = "🔴 HIGH RISK 🔴" if chaos_score >= 75 else ("🟡 MODERATE RISK 🟡" if chaos_score >= 40 else "🟢 SAFE 🟢")
        result = {'liveCount': current_count, 'density': d_norm, 'flow': f_norm, 'surge': s_norm,
                  'chaosScore': chaos_score, 'status': status, 'latency': time.time() - start_time}
        try: result_queue.get_nowait()
        except: pass
        result_queue.put(result)
    print("[AI Worker] Process finished.")

def socket_server_process(result_queue, shared_alert_dict, shared_volunteers_dict):
    p = psutil.Process(os.getpid())
    try: p.cpu_affinity(PROCESS_AFFINITY['api_server'])
    except: pass
    
    app = Flask(__name__)
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*")

    active_alert = shared_alert_dict
    connected_volunteers = shared_volunteers_dict

    def broadcast_loop():
        while True:
            result = result_queue.get()
            if isinstance(result, str) and result == "STOP":
                print("[API Server] STOP signal received. Shutting down.")
                socketio.stop()
                break
            result['activeVolunteerCount'] = len(connected_volunteers)
            socketio.emit('update_data', result)
            
    @socketio.on('connect')
    def handle_connect():
        from flask import request
        print(f"[SocketIO] Client connected: {request.sid}")

    @socketio.on('disconnect')
    def handle_disconnect():
        from flask import request
        sid = request.sid
        if sid in connected_volunteers:
            volunteer_name = connected_volunteers[sid]
            del connected_volunteers[sid]
            socketio.emit('volunteer_count_update', {'count': len(connected_volunteers)})
            print(f"[SocketIO] Volunteer '{volunteer_name}' disconnected: {sid}. Active volunteers: {len(connected_volunteers)}")
        else:
            print(f"[SocketIO] Client disconnected: {request.sid}.")

    # --- MODIFIED: Now accepts a name during registration ---
    @socketio.on('register_volunteer')
    def handle_register_volunteer(data):
        from flask import request
        sid = request.sid
        name = data.get('name', 'Unnamed Volunteer')
        connected_volunteers[sid] = name
        socketio.emit('volunteer_count_update', {'count': len(connected_volunteers)})
        print(f"[SocketIO] Volunteer '{name}' registered: {sid}. Total active: {len(connected_volunteers)}")
        socketio.emit('dispatch_alert', dict(active_alert), room=sid)

    @socketio.on('dispatch_event')
    def handle_dispatch(data):
        from flask import request
        sid = request.sid
        # Update the shared alert dictionary with the new data
        active_alert['active'] = data.get('active', False)
        active_alert['location'] = data.get('location')
        active_alert['chaosScore'] = data.get('chaosScore')
        active_alert['time'] = data.get('time')
        print(f"[SocketIO] Dispatch event received from Dashboard: Active = {active_alert['active']}")
        socketio.emit('dispatch_alert', dict(active_alert))

    print(f"[API Server] Process started on CPU {p.cpu_affinity()}.")
    broadcast_thread = Thread(target=broadcast_loop, daemon=True)
    broadcast_thread.start()
    
    print("[API Server] Socket.IO server starting on port 5000...")
    # Use socketio.run for proper WebSocket handling, which is better than waitress for this
    socketio.run(app, host='0.0.0.0', port=5000)
    print("[API Server] Process finished.")

if __name__ == "__main__":
    print("--- Initializing Trinetra Full-Stack WebSocket System ---")
    
    with Manager() as manager:
        frame_queue = manager.Queue(maxsize=1)
        result_queue = manager.Queue(maxsize=1)
        shared_alert = manager.dict({'active': False, 'location': None, 'chaosScore': 0, 'time': None})
        shared_volunteers = manager.dict()
        
        p_ingestion = Process(target=ingestion_process, args=(VIDEO_PATH, frame_queue))
        p_ai = Process(target=ai_worker_process_onnx, args=(frame_queue, result_queue))
        p_api = Process(target=socket_server_process, args=(result_queue, shared_alert, shared_volunteers))
        
        try:
            p_ingestion.start(); p_ai.start(); p_api.start()
            p_ingestion.join(); p_ai.join(); p_api.join()
        except KeyboardInterrupt:
            print("\n[System] Keyboard interrupt detected. Shutting down.")
            p_ingestion.terminate(); p_ai.terminate(); p_api.terminate()
        
        print("--- Trinetra System Shut Down ---")

