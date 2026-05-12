# 🧠 TRINETRA: AI-Powered Crowd Safety System


---

## 🔍 Overview

**Trinetra** is a full-stack, real-time AI prototype designed to prevent crowd-related tragedies.  
It analyzes live video feeds using an optimized deep learning model to understand complex crowd dynamics.  

The system calculates a multi-factor **Chaos Score** and broadcasts it to a command dashboard. When risk remains high, alerts are dispatched to a dedicated, mobile-first app for on-ground volunteer teams.

---

## 🚨 Problem & Solution

### ❗ Problem

Traditional crowd surveillance is **reactive**. Authorities detect danger only when it’s too late. Simple metrics like crowd density miss the dynamic risks of:

- 🚶‍♂️ **Flow:** Chaotic crowd movement  
- 🚀 **Surge:** Rapid influx of people  

These are **primary triggers** for stampedes.

### ✅ Solution: The Chaos Score

Trinetra calculates a proactive **Chaos Score** based on:

| Metric | Description |
|--------|-------------|
| **D (Density)** | How crowded is the space? |
| **F (Flow)**    | How chaotic is the movement? |
| **S (Surge)**   | How rapidly is the crowd growing? |

When the score remains high, Trinetra dispatches alerts to volunteers—**turning AI predictions into real-time action**.

### Performance Improvements
| Metric         | Before             | After          |
| -------------- | ------------------ | -------------- |
| Inference Time | ~40 sec/frame      | ~0.9 sec/frame |
| Backend Design | Single-threaded    | Multi-process  |
| Model Runtime  | PyTorch            | ONNX Runtime   |
| Alert Latency  | Delayed / Blocking | Near Real-Time |

---

## ⚙️ Logic & Workflow

Trinetra uses a **decoupled, multi-process architecture** to ensure real-time performance:

### 🧪 1. Data Collection (Ingestion)
- Reads frames from a video source (`crowd_video.mp4`)
- Applies pre-checks (blur detection)
- Adds frames to a shared queue

### 🤖 2. AI Processing (Worker Process)
- Pulls frames from queue
- Uses **ONNX Runtime** to analyze crowd
- Calculates D_norm, F_norm, S_norm → Chaos Score
- Passes results to output queue

### 🌐 3. API Server (WebSocket Server)
- Flask-SocketIO server broadcasts data via WebSockets
- Receives dispatch commands from dashboard
- Tracks volunteer status

### 🧑‍💻 4. Admin Dashboard
- React frontend displaying live metrics
- Calculates 10-second average of Chaos Score
- Sends dispatch command when high risk is sustained

### 📱 5. Volunteer App
- Mobile-first React app for on-ground teams
- Displays alerts in real-time with location and score

---

## 🧰 Tech Stack

### 🔙 Backend

| Technology      | Version    |
|-----------------|------------|
| Python          | 3.8        |
| ONNX Runtime    | 1.15.1     |
| Flask-SocketIO  | 5.3.6      |
| OpenCV          | 4.8.0      |
| NumPy / SciPy   | 1.24.3 / 1.10.1 |

### 🌐 Frontend

| Technology       | Version  |
|------------------|----------|
| React            | 18.2.0   |
| Socket.IO Client | 4.7.2    |
| Tailwind CSS     | 3.3.3    |
| Recharts         | 2.8.0    |

### 🧠 AI Model

| Model     | Format |
|-----------|--------|
| P2PNet    | ONNX   |

---

## 🚀 Setup & Installation Guide

### 🐍 Part 1: Python Backend Setup

1. **Install Anaconda / Miniconda**

2. **Create and activate the environment**

    ```bash
    conda create --name onnx_env python=3.8 -y
    conda activate onnx_env
    ```

3. **Install Python Dependencies**

    ```bash
    pip install onnx onnxruntime opencv-python numpy scipy psutil Flask Flask-Cors Flask-SocketIO python-socketio gevent-websocket waitress
    ```

4. **Export the AI Model (One-Time Step)**

    - Install legacy torch libraries needed for export:

      ```bash
      pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
      ```

    - Run the `export_to_onnx.py` script to generate the `p2pnet.onnx` file.

---

### 🌐 Part 2: React Frontend Setup

1. **Install Node.js (LTS version)**

2. **Install `serve` globally**

    ```bash
    npm install -g serve
    ```

3. **Set up the Main Dashboard**

    ```bash
    # Navigate to the trinetra-dashboard folder
    cd path/to/trinetra-dashboard
    npm install
    npm run build
    ```

4. **Set up the Volunteer App**

    ```bash
    # Navigate to the trinetra-volunteer-app folder
    cd path/to/trinetra-volunteer-app
    npm install
    npm run build
    ```

---

---

## 📜 Development & Version History

- **v1.3:** Full-Stack WebSocket Integration  
  - Two-way communication with volunteers  
  - Live tracking of connected volunteers  
  - Complete dispatch system from dashboard to volunteer app

- **v1.2:** ONNX Performance Migration  
  - Converted AI model to ONNX  
  - >30x reduction in inference latency (~1.05s)

- **v1.1:** Multi-Process Architecture & Quantization  
  - Parallel, non-blocking backend  
  - PyTorch Dynamic Quantization reduced latency to ~9–12 seconds

- **v1.0:** Core Engine & Initial Prototype  
  - Full D-F-S Chaos Score logic  
  - Integrated P2PNet AI model

- **v0.1:** Concept & Formula  
  - Defined initial Chaos Score formula

---

**Thank you for checking out Trinetra!**  
For questions or contributions, please open an issue or pull request.

---
