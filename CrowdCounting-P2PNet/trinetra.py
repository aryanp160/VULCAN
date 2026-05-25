import cv2
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as standard_transforms
from scipy.spatial.distance import cdist
from multiprocessing import Process, Queue
import os
import psutil

# --- FIX: The missing import is now added ---
from models import build_model

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# ========================================================================================
# --- VULCAN CONFIGURATION ---
VIDEO_PATH = "crowd_video.mp4"
PROCESSING_INTERVAL_SECONDS = 3.0
PERFORMANCE_RESIZE_FACTOR = 0.25
BLUR_THRESHOLD = 100.0
WEIGHT_PATH = './weights/SHTechA.pth'
DEVICE = 'cpu'
NORMAL_COUNT = 50
MAX_SAFE_COUNT = 800
MAX_TRACKING_DISTANCE_PIXELS = 50
W_DENSITY = 0.6
W_FLOW = 0.4

# --- CPU CORE AFFINITY SETTINGS ---
PROCESS_AFFINITY = {
    'ingestion': [0],
    'ai_worker': [1, 2, 3]
}
# --- END CONFIGURATION ---
# ========================================================================================

def load_ai_model():
    """Loads the P2PNet model AND applies dynamic quantization."""
    print("[AI Worker] Loading PyTorch model...")
    class MockArgs: backbone = 'vgg16_bn'; row = 2; line = 2
    from torch import nn
    model = build_model(MockArgs())
    device = torch.device(DEVICE)
    checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print("[AI Worker] Applying dynamic quantization to the model...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )
    print("[AI Worker] Quantization complete. Model is now in high-performance mode.")
    return quantized_model, device

def get_data_from_frame(frame, model, device):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    width, height = img_pil.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_pil = img_pil.resize((new_width, new_height), Image.ANTIALIAS)
    img_tensor = transform(img_pil)
    samples = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]
    threshold = 0.5
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy()
    return points.shape[0], points

def check_blur(frame, threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def calculate_flow_norm(prev_points, current_points):
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

# --- 2-PROCESS ARCHITECTURE with CPU PINNING ---

def ingestion_process(video_path, frame_queue):
    """Process 1: Grabs frames at a set interval."""
    p = psutil.Process(os.getpid())
    try:
        p.cpu_affinity(PROCESS_AFFINITY['ingestion'])
        print(f"[Ingestion] Process started on CPU {p.cpu_affinity()}. Grabbing one frame every {PROCESSING_INTERVAL_SECONDS} seconds.")
    except:
        print("[Ingestion] Could not set CPU affinity. Process started.")

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"[Ingestion] Error: Could not open video file.")
        frame_queue.put("STOP")
        return
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            frame_queue.put("STOP")
            break
        
        try:
            frame_queue.get_nowait()
        except:
            pass
        frame_queue.put(frame)
        time.sleep(PROCESSING_INTERVAL_SECONDS)

    video_capture.release()
    print("[Ingestion] Process finished.")

def ai_worker_process(frame_queue):
    """Process 2: Runs AI and prints results to the terminal."""
    p = psutil.Process(os.getpid())
    try:
        p.cpu_affinity(PROCESS_AFFINITY['ai_worker'])
        import torch

# Limit / expand PyTorch threading to match affinity
        torch.set_num_threads(len(p.cpu_affinity()))
        print(f"[AI Worker] Torch set to use {torch.get_num_threads()} threads.")

        print(f"[AI Worker] Process started on CPU {p.cpu_affinity()}.")
    except:
        print("[AI Worker] Could not set CPU affinity. Process started.")

    model, device = load_ai_model()
    previous_points = None
    
    print("[AI Worker] Waiting for frames...")
    while True:
        frame = frame_queue.get()
        if isinstance(frame, str) and frame == "STOP":
            break
            
        start_time = time.time()
        
        if PERFORMANCE_RESIZE_FACTOR < 1.0:
            h, w, _ = frame.shape
            frame = cv2.resize(frame, (int(w * PERFORMANCE_RESIZE_FACTOR), int(h * PERFORMANCE_RESIZE_FACTOR)))

        if not check_blur(frame, BLUR_THRESHOLD):
            current_count, current_points = get_data_from_frame(frame, model, device)
            
            if current_count < NORMAL_COUNT: d_norm = 0.0
            elif current_count > MAX_SAFE_COUNT: d_norm = 100.0
            else: d_norm = ((current_count - NORMAL_COUNT) / (MAX_SAFE_COUNT - NORMAL_COUNT)) * 100.0
            
            f_norm = calculate_flow_norm(previous_points, current_points)
            previous_points = current_points
            
            chaos_score = (W_DENSITY * d_norm) + (W_FLOW * f_norm)
            
            if chaos_score >= 75: status = "🔴 HIGH RISK 🔴"
            elif chaos_score >= 40: status = "🟡 MODERATE RISK 🟡"
            else: status = "🟢 SAFE 🟢"

            print(f"\n[AI Worker] --- Analysis Complete (took {time.time() - start_time:.2f}s) ---")
            print(f"Live Count: {current_count} | D_norm: {d_norm:.1f} | F_norm: {f_norm:.1f}")
            print(f"FINAL CHAOS SCORE: {chaos_score:.2f} | STATUS: {status}")
        else:
            print(f"\n[AI Worker] --- Frame Skipped (Too Blurry) ---")
            
    print("[AI Worker] Process finished.")

if __name__ == "__main__":
    print("--- Initializing Vulcan 2-Process System ---")
    frame_queue = Queue(maxsize=1)
    
    p1 = Process(target=ingestion_process, args=(VIDEO_PATH, frame_queue))
    p2 = Process(target=ai_worker_process, args=(frame_queue,))
    
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    
    print("done")