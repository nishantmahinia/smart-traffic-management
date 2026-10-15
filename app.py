"""
SmartTraffic AI — Master Backend Server
Intelligent Autonomous Urban Traffic Signal Control & Computer Vision Telemetry
"""

import os
import sys
import threading
import time
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import random

print("[SYSTEM] Initializing SmartTraffic AI engines...")
sys.stdout.flush()

from sim_engine import SimEngine, ComparisonRunner
from yolo_engine import YoloEngine

print("[SYSTEM] Core analytics & vision modules loaded.")
sys.stdout.flush()

# ------------------------------------------------------------------ #
#  Application Configuration                                         #
# ------------------------------------------------------------------ #
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SMART_TRAFFIC_SECRET", "smart-traffic-ai-master-key-2026")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    ping_timeout=60, ping_interval=25, logger=False, engineio_logger=False)

SAMPLE_VIDEO_DIR = os.path.join(os.path.dirname(__file__), "sample_videos")

# Dynamically locate local YOLO model
LOCAL_MODEL = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
YOLO_MODEL_PATH = LOCAL_MODEL if os.path.exists(LOCAL_MODEL) else None

# ------------------------------------------------------------------ #
#  Engine Instances                                                    #
# ------------------------------------------------------------------ #
sim = SimEngine(socketio)
yolo_eng = YoloEngine(socketio, model_path=YOLO_MODEL_PATH)
comparison_runner = ComparisonRunner(socketio)

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov')


def get_random_sample_video():
    """Helper to check directory and return a random video path, or fallback to default."""
    if not os.path.exists(SAMPLE_VIDEO_DIR):
        return None
        
    videos = [
        f for f in os.listdir(SAMPLE_VIDEO_DIR) 
        if f.lower().endswith(VIDEO_EXTENSIONS)
    ]
    
    if videos:
        chosen_video = random.choice(videos)
        return os.path.join(SAMPLE_VIDEO_DIR, chosen_video)
        
    fallback_path = os.path.join(SAMPLE_VIDEO_DIR, "traffic_sample.mp4")
    return fallback_path if os.path.exists(fallback_path) else None


# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of real-time annotated YOLO frames."""
    def generate():
        while True:
            with yolo_eng.lock:
                frame_bytes = yolo_eng.current_frame
            if frame_bytes:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       frame_bytes + b"\r\n")
            else:
                time.sleep(0.05)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    """System health and AI engine status."""
    return jsonify({
        "status": "online",
        "yolo_ready": yolo_eng.model_ready,
        "is_analyzing": yolo_eng.is_analyzing,
        "is_paused": yolo_eng.is_paused,
        "n_lanes": yolo_eng.n_lanes,
        "active_videos": sum(1 for l in yolo_eng.lanes if l.loaded)
    })


@app.route("/upload_video/<int:lane_idx>", methods=["POST"])
def upload_video(lane_idx):
    """Handle video file upload for a specific lane (0-3)."""
    if "video" not in request.files:
        return jsonify({"error": "No file attached"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "Empty filename provided"}), 400
    
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        return jsonify({"error": f"Unsupported video format '{ext}'. Allowed: {', '.join(VIDEO_EXTENSIONS)}"}), 400

    os.makedirs(SAMPLE_VIDEO_DIR, exist_ok=True)
    save_path = os.path.join(SAMPLE_VIDEO_DIR, f"lane_{lane_idx}_uploaded{ext}")
    f.save(save_path)
    yolo_eng.set_lane_video(lane_idx, save_path)
    return jsonify({"status": "ok", "lane": lane_idx, "filename": f.filename})


@app.route("/api/upload_main_video", methods=["POST"])
def upload_main_video():
    """Upload traffic video from Dashboard."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        return jsonify({"error": f"Invalid format '{ext}'. Supported: {', '.join(VIDEO_EXTENSIONS)}"}), 400

    os.makedirs(SAMPLE_VIDEO_DIR, exist_ok=True)
    save_path = os.path.join(SAMPLE_VIDEO_DIR, f"main_analysis_video{ext}")
    f.save(save_path)
    
    # Mount to Lane 0 as primary video
    yolo_eng.set_lane_video(0, save_path)
    return jsonify({"status": "ok", "filename": f.filename, "lane": 0})


@app.route("/use_sample/<int:lane_idx>")
def use_sample(lane_idx):
    """Use a sample video for a specific lane."""
    sample_path = get_random_sample_video()
    if sample_path:
        yolo_eng.set_lane_video(lane_idx, sample_path)
        return jsonify({
            "status": "ok", 
            "lane": lane_idx, 
            "file_mounted": os.path.basename(sample_path)
        })
    return jsonify({"error": "No sample videos available. Please upload a video."}), 404


@app.route("/use_sample_all")
def use_sample_all():
    """Use sample videos for all active lanes."""
    for i in range(yolo_eng.n_lanes):
        sample_path = get_random_sample_video()
        if sample_path:
            yolo_eng.set_lane_video(i, sample_path)
    return jsonify({"status": "ok", "n_lanes": yolo_eng.n_lanes})


@app.route("/clear_lane/<int:lane_idx>")
def clear_lane(lane_idx):
    """Clear video feed for a specific lane."""
    yolo_eng.clear_lane_video(lane_idx)
    return jsonify({"status": "ok", "lane": lane_idx})


@app.route("/clear_all_lanes")
def clear_all_lanes():
    """Clear all lane videos."""
    for i in range(4):
        yolo_eng.clear_lane_video(i)
    return jsonify({"status": "ok"})


@app.route("/set_quads/<int:n>")
def set_quads(n):
    yolo_eng.set_n_lanes(n)
    return jsonify({"status": "ok", "n_lanes": n})


# ------------------------------------------------------------------ #
#  Analysis State Controls                                             #
# ------------------------------------------------------------------ #
@app.route("/api/analysis/start", methods=["POST"])
def analysis_start():
    yolo_eng.start_analysis()
    return jsonify({"status": "ok", "is_analyzing": True, "is_paused": False})


@app.route("/api/analysis/pause", methods=["POST"])
def analysis_pause():
    yolo_eng.pause_analysis()
    return jsonify({"status": "ok", "is_paused": True})


@app.route("/api/analysis/resume", methods=["POST"])
def analysis_resume():
    yolo_eng.resume_analysis()
    return jsonify({"status": "ok", "is_paused": False, "is_analyzing": True})


@app.route("/api/analysis/stop", methods=["POST"])
def analysis_stop():
    yolo_eng.stop_analysis()
    return jsonify({"status": "ok", "is_analyzing": False})


# ------------------------------------------------------------------ #
#  Settings Management                                                 #
# ------------------------------------------------------------------ #
@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        data = request.get_json() or {}
        min_green = data.get("min_green")
        max_green = data.get("max_green")
        yellow_duration = data.get("yellow_duration")
        conf_threshold = data.get("conf_threshold")

        # Validation
        errors = []
        if min_green is not None:
            try:
                min_green = int(min_green)
                if not (3 <= min_green <= 30):
                    errors.append("Minimum green duration must be between 3 and 30 seconds.")
            except (ValueError, TypeError):
                errors.append("Invalid minimum green value.")

        if max_green is not None:
            try:
                max_green = int(max_green)
                if not (15 <= max_green <= 120):
                    errors.append("Maximum green duration must be between 15 and 120 seconds.")
            except (ValueError, TypeError):
                errors.append("Invalid maximum green value.")

        if yellow_duration is not None:
            try:
                yellow_duration = int(yellow_duration)
                if not (1 <= yellow_duration <= 10):
                    errors.append("Yellow transition duration must be between 1 and 10 seconds.")
            except (ValueError, TypeError):
                errors.append("Invalid yellow transition duration.")

        if conf_threshold is not None:
            try:
                conf_threshold = float(conf_threshold)
                if not (0.1 <= conf_threshold <= 0.95):
                    errors.append("Confidence threshold must be between 0.10 and 0.95.")
            except (ValueError, TypeError):
                errors.append("Invalid confidence threshold.")

        if errors:
            return jsonify({"status": "error", "errors": errors}), 400

        # Apply settings to both engines
        yolo_eng.update_settings(
            min_green=min_green,
            max_green=max_green,
            yellow_duration=yellow_duration,
            conf_threshold=conf_threshold
        )
        sim.update_settings(
            min_green=min_green,
            max_green=max_green,
            yellow_duration=yellow_duration
        )

        return jsonify({
            "status": "ok",
            "message": "Settings successfully applied.",
            "settings": {
                "min_green": yolo_eng.min_green,
                "max_green": yolo_eng.max_green,
                "yellow_duration": yolo_eng.yellow_duration,
                "conf_threshold": yolo_eng.conf_threshold
            }
        })

    # GET request
    return jsonify({
        "status": "ok",
        "settings": {
            "min_green": yolo_eng.min_green,
            "max_green": yolo_eng.max_green,
            "yellow_duration": yolo_eng.yellow_duration,
            "conf_threshold": yolo_eng.conf_threshold
        }
    })


# ------------------------------------------------------------------ #
#  Comparison Benchmark API                                            #
# ------------------------------------------------------------------ #
@app.route("/api/benchmark/start", methods=["POST"])
def benchmark_start():
    success = comparison_runner.start_comparison()
    return jsonify({"status": "ok" if success else "already_running"})


@app.route("/api/benchmark/reset", methods=["POST"])
def benchmark_reset():
    comparison_runner.reset()
    return jsonify({"status": "ok"})


# ------------------------------------------------------------------ #
#  Emergency Preemption API                                            #
# ------------------------------------------------------------------ #
@app.route("/api/emergency/trigger", methods=["POST"])
def emergency_trigger():
    data = request.get_json() or {}
    lane = data.get("lane", 0)
    
    # Map lane identifier if string
    lane_map = {"N": 0, "E": 1, "S": 2, "W": 3, "North": 0, "East": 1, "South": 2, "West": 3}
    lane_idx = lane_map.get(lane, lane if isinstance(lane, int) else 0)
    
    sim_lane_code = ["N", "E", "S", "W"][lane_idx % 4]
    sim.trigger_emergency(lane=sim_lane_code)
    yolo_eng.force_emergency(lane=lane_idx)
    return jsonify({"status": "ok", "corridor_lane": lane_idx})


@app.route("/api/emergency/cancel", methods=["POST"])
def emergency_cancel():
    sim.cancel_emergency()
    yolo_eng.cancel_emergency()
    return jsonify({"status": "ok"})


# ------------------------------------------------------------------ #
#  SocketIO Events                                                     #
# ------------------------------------------------------------------ #
@socketio.on("connect")
def on_connect():
    print(f"[SOCKET] Client connected: {request.sid}")


@socketio.on("disconnect")
def on_disconnect():
    print(f"[SOCKET] Client disconnected: {request.sid}")


@socketio.on("toggle_emergency")
def on_toggle_emergency(data=None):
    lane = None
    if isinstance(data, dict):
        lane = data.get("lane")
    elif isinstance(data, str):
        lane = data
    sim.trigger_emergency(lane=lane)
    emit("sim_state", sim.tick(), broadcast=True)


@socketio.on("cancel_emergency")
def on_cancel_emergency():
    sim.cancel_emergency()
    emit("sim_state", sim.tick(), broadcast=True)


@socketio.on("yolo_set_quads")
def on_yolo_set_quads(data):
    n = data.get("n", 4)
    yolo_eng.set_n_lanes(n)


@socketio.on("yolo_emergency_override")
def on_yolo_emergency(data=None):
    lane = data.get("lane", 0) if isinstance(data, dict) else 0
    yolo_eng.force_emergency(lane=lane)


@socketio.on("start_comparison")
def on_start_comparison():
    comparison_runner.start_comparison()


# ------------------------------------------------------------------ #
#  Background Threads                                                  #
# ------------------------------------------------------------------ #
def start_sim_loop():
    time.sleep(1)
    sim.run_loop()


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("=" * 64)
    print("  🚦 SmartTraffic AI — Intelligent Urban Traffic Control Platform")
    print("  Local Control Center: http://localhost:5000")
    print("=" * 64)
    sys.stdout.flush()

    # Start simulation loop in background
    sim_thread = threading.Thread(target=start_sim_loop, daemon=True)
    sim_thread.start()

    # Start YOLO processing loop immediately
    yolo_eng.start()

    print("[SYSTEM] Starting web server on http://127.0.0.1:5000")
    sys.stdout.flush()
    socketio.run(app, host="0.0.0.0", port=5000, debug=False,
                 use_reloader=False, allow_unsafe_werkzeug=True)
