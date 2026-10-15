# 🚦 SmartTraffic AI — Intelligent Urban Traffic Management Dashboard

SmartTraffic AI is an edge-driven autonomous traffic signal management and urban flow analytics platform. Built with **Python**, **Flask-SocketIO**, **OpenCV**, and **YOLOv8**, the system dynamically computes lane density in real time, intelligently adjusts signal phases, prioritizes emergency vehicles, and provides live analytics through an Apple-inspired dashboard.

---

## ⚡ Core Capabilities

- **Real-Time Edge Computer Vision**: Multi-stream vehicle classification and bounding-box detection using YOLOv8 (Cars, Motorcycles, Buses, Heavy Trucks).
- **Class-Weighted Density Optimization**: Proportional priority weight calculations:
  $$W = \text{Cars} \times 1.0 + \text{Motorcycles} \times 0.8 + \text{Trucks} \times 1.5 + \text{Buses} \times 2.5$$
- **Adaptive Signal Phase Allocation**: Dynamically calculates green intervals with safe yellow transition clearance:
  $$\text{Green Time} = \text{Clamp}(\text{Min Green}, \text{Max Green}, \text{Weight} \times 1.2)$$
- **Emergency Priority Preemption**: Instant priority preemption corridor for emergency vehicles (optical strobe light detection + manual operator corridor override), safely holding conflicting lanes at Red.
- **Fixed vs. Adaptive Performance Benchmarking**: Side-by-side comparative simulation verifying real measured metrics (Vehicles Processed, Average Queue Length, Average Waiting Time) across identical pseudo-random seeds.
- **Zero-Latency WebSocket Telemetry**: Flask-SocketIO pushes real-time queue states, countdown timers, and decision rationales to a modern Apple-style responsive dashboard.
- **Interactive Settings Management**: Real-time configurable signal timings (Min Green, Max Green, Yellow Clearance, and YOLO Detection Confidence).

---

## 🏗 System Architecture

```text
       Camera Feeds / Uploaded Video Files
                        │
                        ▼
             YOLOv8 Real-Time Detector
      (Cars, Trucks, Buses, Motorcycles, Emergency)
                        │
                        ▼
            Dynamic Density Engine
      (Computes Per-Approach Weight Vector)
                        │
                        ▼
        Adaptive Signal Phase Allocator
    (Selects Heaviest Approach & Safe Transitions)
                        │
                        ▼
             Flask-SocketIO Gateway
                        │
                        ▼
     SmartTraffic AI Dashboard (Localhost:5000)
```

---

## 🚀 Quick Start Guide (Windows / VS Code)

### 1. Prerequisites
- Python 3.10 or 3.11 installed on your system.
- VS Code or your preferred IDE.

### 2. Launching with the Run Script
You can launch the application directly by double-clicking:
```bat
run_project.bat
```
This script automatically opens your default browser to `http://127.0.0.1:5000` and starts the backend server.

### 3. Manual Launch via VS Code Terminal
Open the project directory in VS Code and run:

```powershell
# Navigate to project folder
cd smart-traffic-management-main

# If using existing venv:
..\venv311\Scripts\activate

# Or create a new virtual environment:
python -m venv venv
venv\Scripts\activate
pip install Flask Flask-SocketIO Flask-Cors opencv-python numpy ultralytics

# Run the master server
python app.py
```

Open your browser and navigate directly to:
```text
http://127.0.0.1:5000
```

---

## 🖥 Dashboard Interface Overview

1. **Dashboard**:
   - **Traffic Video Panel**: Upload local traffic video (`.mp4`, `.avi`, `.mov`), load sample video in one click, play/pause controls, live YOLO bounding box stream.
   - **Traffic Statistics**: Total vehicles detected, traffic density level (Low/Medium/High), average queue length, active traffic signals.
   - **Four-Direction Status**: Compact cards for North, South, East, and West with vehicle counts, breakdown pills, queue estimates, and signal countdowns.
   - **Live Traffic Graph**: Rolling Chart.js time-series tracking traffic demand over time across all 4 directions.
   - **Adaptive Signal Controller**: Countdown timer ring, active phase indicator, next scheduled direction, and explicit timing decision explanations.
   - **Emergency Vehicle Alert**: Prominent alert displaying emergency vehicle type, detected direction, priority status, and signal action with manual override triggers.
   - **Fixed-Time vs Adaptive Comparison**: 60-second benchmarking measuring real vehicles processed, queue sizes, and waiting times.
2. **Traffic Monitoring**:
   - Multi-channel CCTV surveillance (2, 3, or 4 camera grid).
   - Real-time 60 FPS 4-way intersection aerial canvas simulator.
3. **Settings**:
   - Minimum green duration (3s - 30s).
   - Maximum green duration (15s - 120s).
   - Yellow transition duration (1s - 10s).
   - Vehicle detection confidence threshold (0.10 - 0.90).

---

## 📄 Operational License
Distributed under the MIT License. Developed for intelligent urban transportation research and operational control systems.
