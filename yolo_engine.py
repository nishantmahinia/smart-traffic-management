"""
SmartTraffic AI — Per-Lane YOLOv8 Real-Time Vision & Density Engine

Multi-lane asynchronous video capture and real-time object detection.
Class-weighted vehicle density mapping and optical strobe / emergency light detection.
Stitches multi-camera feeds into a synchronized MJPEG telemetry stream.
"""

import cv2
import numpy as np
import threading
import time
import os
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# COCO vehicle classes & computational priority weightings
VEHICLE_CLASSES = {
    2: ("car",        1.0),
    3: ("motorcycle", 0.8),
    5: ("bus",        2.5),
    7: ("truck",      1.5),
}

# Per-lane modern accent colors (BGR)
LANE_COLORS = [
    (240, 145,  55),   # Lane 0 — Electric Sky
    (110, 215,  40),   # Lane 1 — Mint Emerald
    ( 50, 175, 245),   # Lane 2 — Sunset Amber
    (215,  85, 175),   # Lane 3 — Deep Orchid
]

LANE_DEFAULT_NAMES = ["North Approach", "East Approach", "South Approach", "West Approach"]

# Target frame size per lane tile in the output grid
TILE_W = 450
TILE_H = 300


def compute_weight(counts):
    return round(
        counts.get("car", 0)        * 1.0 +
        counts.get("motorcycle", 0) * 0.8 +
        counts.get("truck", 0)      * 1.5 +
        counts.get("bus", 0)        * 2.5,
        1
    )


def detect_emergency_color(frame):
    """Detect flashing red/blue (emergency strobe lights) via HSV color masking."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_r1 = cv2.inRange(hsv, np.array([0,   150, 150]), np.array([ 10, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([160, 150, 150]), np.array([180, 255, 255]))
    mask_b  = cv2.inRange(hsv, np.array([100, 150, 100]), np.array([130, 255, 255]))
    combined = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), mask_b)
    return cv2.countNonZero(combined) > 800


class LaneCapture:
    """Holds state for a single lane's video feed."""
    __slots__ = ["cap", "path", "name", "loaded", "frame", "counts",
                 "weight", "green_time", "light", "emergency", "lock"]

    def __init__(self, name):
        self.cap        = None
        self.path       = None
        self.name       = name
        self.loaded     = False
        self.frame      = None          # latest annotated BGR frame
        self.counts     = {"car": 0, "motorcycle": 0, "truck": 0, "bus": 0}
        self.weight     = 0.0
        self.green_time = 6
        self.light      = "red"
        self.emergency  = False
        self.lock       = threading.Lock()

    def set_video(self, path):
        with self.lock:
            if self.cap:
                self.cap.release()
            self.cap    = cv2.VideoCapture(path)
            self.path   = path
            self.loaded = self.cap.isOpened()

    def read_frame(self):
        """Read next frame, loop video at end. Returns BGR frame or None."""
        if not self.cap or not self.cap.isOpened():
            return None
        with self.lock:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.loaded = False

    def clear_video(self):
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.path = None
            self.loaded = False
            self.frame = None
            self.counts = {"car": 0, "motorcycle": 0, "truck": 0, "bus": 0}
            self.weight = 0.0
            self.green_time = 6
            self.light = "red"
            self.emergency = False


class YoloEngine:
    def __init__(self, socketio, model_path=None, n_lanes=4):
        self.socketio   = socketio
        self.model      = None
        self.model_path = model_path
        self.model_ready = False

        # Configurable signal timing & vision parameters
        self.min_green = 6
        self.max_green = 45
        self.yellow_duration = 3
        self.conf_threshold = 0.35

        self.n_lanes = n_lanes
        self.lanes   = [LaneCapture(LANE_DEFAULT_NAMES[i]) for i in range(4)]

        self.running       = False
        self.is_analyzing  = True
        self.is_paused     = False
        self.lock          = threading.Lock()
        self.current_frame = None      # stitched JPEG bytes for MJPEG
        self.active_lane   = 0
        self.next_lane     = 1
        self.phase_state   = "green"   # "green" or "yellow"
        self._cycle_timer  = 10
        self.emergency     = False
        self.emergency_lane = None
        self.emergency_type = "None"
        self.decision_reason = "Adaptive engine active — Initial green phase assigned to North Approach."
        self.frame_count   = 0

        # YOLO Frame skip cache
        self.cached_boxes  = [[] for _ in range(4)]
        self.cached_counts = [{"car": 0, "motorcycle": 0, "truck": 0, "bus": 0} for _ in range(4)]
        self.cached_emerg  = [False for _ in range(4)]

        # Start async model load
        threading.Thread(target=self._load_model_bg, daemon=True).start()

    def update_settings(self, min_green=None, max_green=None, yellow_duration=None, conf_threshold=None):
        with self.lock:
            if min_green is not None:
                self.min_green = max(3, int(min_green))
            if max_green is not None:
                self.max_green = max(self.min_green, int(max_green))
            if yellow_duration is not None:
                self.yellow_duration = max(1, int(yellow_duration))
            if conf_threshold is not None:
                self.conf_threshold = max(0.1, min(0.95, float(conf_threshold)))

    def compute_green_time(self, weight):
        return max(self.min_green, min(self.max_green, int(weight * 1.2)))

    # ------------------------------------------------------------------ #
    #  Model Loading (non-blocking)                                         #
    # ------------------------------------------------------------------ #
    def _load_model_bg(self):
        if not YOLO_AVAILABLE:
            print("[YOLO] ultralytics not available")
            return

        candidates = []
        if self.model_path and os.path.exists(self.model_path):
            candidates.append(self.model_path)

        local_dir = Path(__file__).resolve().parent
        for p in [
            local_dir / "yolov8n.pt",
            local_dir.parent / "yolov8n.pt",
            Path.cwd() / "yolov8n.pt",
            Path.home() / ".config" / "Ultralytics" / "yolov8n.pt",
        ]:
            if p.exists() and str(p) not in candidates:
                candidates.append(str(p))

        for path in candidates:
            try:
                print(f"[YOLO] Loading model: {path}")
                self.model = YOLO(path)
                self.model_ready = True
                print("[YOLO] ✓ Model ready")
                return
            except Exception as e:
                print(f"[YOLO] Load failed ({path}): {e}")

        try:
            print("[YOLO] Downloading yolov8n.pt ...")
            self.model = YOLO("yolov8n.pt")
            self.model_ready = True
            print("[YOLO] ✓ Downloaded and ready")
        except Exception as e:
            print(f"[YOLO] Could not load model: {e}")

    # ------------------------------------------------------------------ #
    #  Analysis Controls                                                   #
    # ------------------------------------------------------------------ #
    def start_analysis(self):
        with self.lock:
            self.is_analyzing = True
            self.is_paused = False
        if not self.running:
            self.start()

    def pause_analysis(self):
        with self.lock:
            self.is_paused = True

    def resume_analysis(self):
        with self.lock:
            self.is_paused = False
            self.is_analyzing = True

    def stop_analysis(self):
        with self.lock:
            self.is_analyzing = False
            self.is_paused = False

    # ------------------------------------------------------------------ #
    #  Lane Video Management                                               #
    # ------------------------------------------------------------------ #
    def set_lane_video(self, lane_idx, path):
        if 0 <= lane_idx < 4:
            self.lanes[lane_idx].set_video(path)
            print(f"[YOLO] Lane {lane_idx} ({self.lanes[lane_idx].name}) → {path}")
            self.is_analyzing = True
            self.is_paused = False
            if not self.running:
                self.start()

    def clear_lane_video(self, lane_idx):
        if 0 <= lane_idx < 4:
            self.lanes[lane_idx].clear_video()
            self.cached_boxes[lane_idx] = []
            self.cached_counts[lane_idx] = {"car": 0, "motorcycle": 0, "truck": 0, "bus": 0}
            self.cached_emerg[lane_idx] = False
            print(f"[YOLO] Lane {lane_idx} video cleared")

    def set_n_lanes(self, n):
        self.n_lanes = max(1, min(4, int(n)))
        for i in range(4):
            self.lanes[i].name = LANE_DEFAULT_NAMES[i]

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.run_loop, daemon=True).start()

    def stop(self):
        self.running = False
        for lane in self.lanes:
            lane.release()

    # ------------------------------------------------------------------ #
    #  Main Processing Loop                                                #
    # ------------------------------------------------------------------ #
    def run_loop(self):
        last_tick_time = time.time()
        while self.running:
            self.frame_count += 1
            lane_frames = []
            any_active = False

            # If paused, render previous frames with PAUSED banner
            if self.is_paused or not self.is_analyzing:
                for i in range(self.n_lanes):
                    lane = self.lanes[i]
                    if lane.frame is not None:
                        tile = lane.frame.copy()
                        # Add paused banner
                        cv2.rectangle(tile, (0, TILE_H - 30), (TILE_W, TILE_H), (15, 23, 42), -1)
                        status_txt = "⏸ ANALYSIS PAUSED" if self.is_paused else "⏹ ANALYSIS STOPPED"
                        cv2.putText(tile, status_txt, (TILE_W // 2 - 70, TILE_H - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                    else:
                        tile = self._placeholder_tile(i)
                    lane_frames.append(tile)

                grid = self._stitch_grid(lane_frames)
                _, jpeg = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self.lock:
                    self.current_frame = jpeg.tobytes()
                time.sleep(0.08)
                continue

            for i in range(self.n_lanes):
                lane = self.lanes[i]
                raw = lane.read_frame()

                if raw is None:
                    # Lane has no video yet — draw clean placeholder tile
                    tile = self._placeholder_tile(i)
                    lane.counts = {"car": 0, "motorcycle": 0, "truck": 0, "bus": 0}
                    lane.weight = 0.0
                    lane.green_time = self.min_green
                    lane.emergency = False
                    self.cached_boxes[i] = []
                    self.cached_counts[i] = lane.counts.copy()
                    self.cached_emerg[i] = False
                else:
                    any_active = True
                    # Resize for optimal processing balance
                    raw = cv2.resize(raw, (TILE_W, TILE_H))
                    # Run YOLO every 5 frames per lane, or if cache is empty
                    if self.frame_count % 5 == 0 or not self.cached_counts[i] or (lane.frame is None):
                        counts, boxes, emerg = self._detect(raw)
                        lane.counts    = counts
                        lane.weight    = compute_weight(counts)
                        lane.green_time = self.compute_green_time(lane.weight)
                        lane.emergency = emerg
                        # Cache
                        self.cached_boxes[i]  = boxes
                        self.cached_counts[i] = counts
                        self.cached_emerg[i]  = emerg
                    else:
                        counts = self.cached_counts[i]
                        boxes  = self.cached_boxes[i]
                        emerg  = self.cached_emerg[i]
                        lane.counts = counts
                        lane.weight = compute_weight(counts)
                        lane.green_time = self.compute_green_time(lane.weight)
                        lane.emergency = emerg

                    # Annotate tile with bounding boxes and sleek telemetry
                    tile = self._annotate_tile(raw, i, counts, boxes, emerg)
                    lane.frame = tile

                lane_frames.append(tile)

            # Timer tick once every real-time second
            now = time.time()
            if now - last_tick_time >= 1.0:
                self._update_cycle()
                last_tick_time = now

            # Apply light states on tiles (overlay border glow and corner lamp)
            annotated_tiles = []
            for i, tile in enumerate(lane_frames):
                if self.emergency:
                    light = "green" if i == self.active_lane else "red"
                elif i == self.active_lane:
                    light = self.phase_state  # "green" or "yellow"
                else:
                    light = "red"
                self.lanes[i].light = light
                annotated_tiles.append(self._apply_light_border(tile, i, light))

            # Stitch grid
            grid = self._stitch_grid(annotated_tiles)

            # JPEG encode
            _, jpeg = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 82])
            with self.lock:
                self.current_frame = jpeg.tobytes()

            # Push state every 5 frames
            if self.frame_count % 5 == 0:
                self.socketio.emit("yolo_state", self._build_state())

            if not any_active:
                time.sleep(0.08)
            else:
                time.sleep(0.015)

    # ------------------------------------------------------------------ #
    #  YOLO Detection                                                       #
    # ------------------------------------------------------------------ #
    def _detect(self, frame):
        counts  = {"car": 0, "motorcycle": 0, "truck": 0, "bus": 0}
        boxes   = []
        emerg   = detect_emergency_color(frame)

        if not self.model_ready or self.model is None:
            return counts, boxes, emerg

        results = self.model(frame, verbose=False, conf=self.conf_threshold, iou=0.45, device="cpu")[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            label, _ = VEHICLE_CLASSES[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            counts[label] += 1
            boxes.append({"box": (x1, y1, x2, y2), "label": label, "conf": conf})

        return counts, boxes, emerg

    # ------------------------------------------------------------------ #
    #  Frame Annotation & Overlays                                          #
    # ------------------------------------------------------------------ #
    def _annotate_tile(self, frame, lane_idx, counts, boxes, emerg):
        tile  = frame.copy()
        color = LANE_COLORS[lane_idx % len(LANE_COLORS)]
        name  = self.lanes[lane_idx].name

        # Draw crisp bounding boxes with label tags
        for b in boxes:
            x1, y1, x2, y2 = b["box"]
            cv2.rectangle(tile, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            lbl = f"{b['label'].capitalize()} {int(b['conf'] * 100)}%"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            ty = max(y1 - 6, th + 4)
            # Tag badge
            cv2.rectangle(tile, (x1, ty - th - 3), (x1 + tw + 6, ty + 3), (15, 23, 42), -1)
            cv2.rectangle(tile, (x1, ty - th - 3), (x1 + tw + 6, ty + 3), color, 1, lineType=cv2.LINE_AA)
            cv2.putText(tile, lbl, (x1 + 3, ty - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # Header overlay banner
        overlay = tile.copy()
        cv2.rectangle(overlay, (0, 0), (TILE_W, 44), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.90, tile, 0.10, 0, tile)
        cv2.line(tile, (0, 44), (TILE_W, 44), (51, 65, 85), 1, lineType=cv2.LINE_AA)
        cv2.line(tile, (0, 0), (TILE_W, 0), color, 3, lineType=cv2.LINE_AA)

        # Approach name
        cv2.putText(tile, name, (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (248, 250, 252), 1, lineType=cv2.LINE_AA)

        # Vehicle counts telemetry
        total = sum(counts.values())
        info  = f"Cars: {counts['car']} | Trucks: {counts['truck']} | Buses: {counts['bus']} | Moto: {counts['motorcycle']}  (Total: {total})"
        cv2.putText(tile, info, (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (148, 163, 184), 1, lineType=cv2.LINE_AA)

        # Emergency override banner
        if emerg:
            emerg_bar = tile.copy()
            cv2.rectangle(emerg_bar, (0, TILE_H - 34), (TILE_W, TILE_H), (20, 20, 210), -1)
            cv2.addWeighted(emerg_bar, 0.92, tile, 0.08, 0, tile)
            cv2.putText(tile, "EMERGENCY STROBE DETECTED — PRIORITY PHASE", (12, TILE_H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        return tile

    def _apply_light_border(self, tile, lane_idx, light):
        """Draw modern signal state border and traffic lamp indicator."""
        out   = tile.copy()
        bw    = 3
        # Signal colors: Emerald green, Amber/Yellow, Red
        if light == "green":
            color = (34, 197, 94)    # BGR for #10b981
        elif light == "yellow":
            color = (11, 158, 245)   # BGR for #f59e0b
        else:
            color = (68, 68, 239)    # BGR for #ef4444

        cv2.rectangle(out, (0, 0), (TILE_W - 1, TILE_H - 1), color, bw, lineType=cv2.LINE_AA)

        # Signal circle in top-right
        cx, cy = TILE_W - 22, 22
        cv2.circle(out, (cx, cy), 8, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 10, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # Weight & green time telemetry badge bottom-right
        lane    = self.lanes[lane_idx]
        label   = f"Weight {lane.weight} | Timer {lane.green_time}s"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        bx, by = TILE_W - lw - 16, TILE_H - 8
        cv2.rectangle(out, (bx - 4, by - lh - 4), (TILE_W - 4, TILE_H - 4), (15, 23, 42), -1)
        cv2.rectangle(out, (bx - 4, by - lh - 4), (TILE_W - 4, TILE_H - 4), (71, 85, 105), 1, lineType=cv2.LINE_AA)
        cv2.putText(out, label, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (226, 232, 240), 1, lineType=cv2.LINE_AA)

        return out

    def _placeholder_tile(self, lane_idx):
        """Clean modern placeholder tile when no video is loaded."""
        tile  = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
        tile[:] = (248, 250, 252) # Light gray background
        color = LANE_COLORS[lane_idx % len(LANE_COLORS)]
        name  = self.lanes[lane_idx].name

        # Subtle clean grid pattern
        for gx in range(0, TILE_W, 30):
            cv2.line(tile, (gx, 0), (gx, TILE_H), (241, 245, 249), 1)
        for gy in range(0, TILE_H, 30):
            cv2.line(tile, (0, gy), (TILE_W, gy), (241, 245, 249), 1)

        # Outer border
        cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), (226, 232, 240), 1, lineType=cv2.LINE_AA)
        cv2.line(tile, (0, 0), (TILE_W, 0), color, 4, lineType=cv2.LINE_AA)

        # Header bar
        cv2.rectangle(tile, (0, 0), (TILE_W, 40), (241, 245, 249), -1)
        cv2.line(tile, (0, 40), (TILE_W, 40), (226, 232, 240), 1, lineType=cv2.LINE_AA)
        cv2.putText(tile, f"CAMERA {lane_idx + 1}: {name.upper()}", (14, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 41, 59), 1, lineType=cv2.LINE_AA)

        # Center standby message
        cv2.putText(tile, "CAMERA STANDBY", (TILE_W // 2 - 80, TILE_H // 2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 116, 139), 1, lineType=cv2.LINE_AA)
        cv2.putText(tile, "Upload a video file or load sample stream",
                    (TILE_W // 2 - 130, TILE_H // 2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, lineType=cv2.LINE_AA)
        return tile

    # ------------------------------------------------------------------ #
    #  Grid Stitching                                                       #
    # ------------------------------------------------------------------ #
    def _stitch_grid(self, tiles):
        """Stitch tiles into a responsive grid."""
        n = len(tiles)
        if n == 0:
            return np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)

        if n == 1:
            return tiles[0]

        # Pad to even count for 2-column layout
        padded = list(tiles)
        while len(padded) % 2 != 0:
            padded.append(self._placeholder_tile(len(padded)))

        rows = []
        for i in range(0, len(padded), 2):
            sep = np.zeros((TILE_H, 2, 3), dtype=np.uint8) + 226  # thin light border
            rows.append(np.hstack([padded[i], sep, padded[i + 1]]))

        h_sep = np.zeros((2, rows[0].shape[1], 3), dtype=np.uint8) + 226
        grid  = rows[0]
        for row in rows[1:]:
            grid = np.vstack([grid, h_sep, row])

        return grid

    # ------------------------------------------------------------------ #
    #  Cycle Logic with Safe Yellow Clearance Transition                 #
    # ------------------------------------------------------------------ #
    def _update_cycle(self):
        self._cycle_timer -= 1
        if self._cycle_timer <= 0:
            # Check for emergency on any lane
            emerg_lanes = [i for i in range(self.n_lanes) if self.lanes[i].emergency]
            if emerg_lanes or self.emergency:
                if emerg_lanes:
                    self.active_lane = emerg_lanes[0]
                self.emergency    = True
                self.phase_state  = "green"
                self._cycle_timer = 15
                self.decision_reason = f"EMERGENCY OVERRIDE: Priority corridor assigned to {self.lanes[self.active_lane].name}."
            else:
                self.emergency = False
                weights = [self.lanes[i].weight for i in range(self.n_lanes)]
                
                # If currently green, transition to safe yellow before switching
                if self.phase_state == "green" and self.yellow_duration > 0:
                    self.phase_state = "yellow"
                    self._cycle_timer = self.yellow_duration
                    # Pick next lane
                    candidates = [i for i in range(self.n_lanes) if i != self.active_lane]
                    if candidates and any(weights[c] > 0 for c in candidates):
                        self.next_lane = max(candidates, key=lambda c: weights[c])
                    else:
                        self.next_lane = (self.active_lane + 1) % self.n_lanes
                    self.decision_reason = f"Yellow warning clearance ({self.yellow_duration}s) on {self.lanes[self.active_lane].name} before transitioning to {self.lanes[self.next_lane].name}."
                else:
                    # Switch to green for next lane
                    self.phase_state = "green"
                    self.active_lane = self.next_lane
                    # Pick prospective next lane
                    candidates = [i for i in range(self.n_lanes) if i != self.active_lane]
                    self.next_lane = (self.active_lane + 1) % self.n_lanes
                    if candidates and any(weights[c] > 0 for c in candidates):
                        self.next_lane = max(candidates, key=lambda c: weights[c])
                    
                    best_w = weights[self.active_lane] if self.active_lane < len(weights) else 0.0
                    self._cycle_timer = self.compute_green_time(best_w)
                    total_v = sum(self.lanes[self.active_lane].counts.values())
                    self.decision_reason = f"{self.lanes[self.active_lane].name} allocated Green ({self._cycle_timer}s): calculated demand weight of {best_w} with {total_v} vehicles."

    def force_emergency(self, lane=None):
        self.emergency = True
        self.emergency_type = "Manual Override"
        self.phase_state = "green"
        self._cycle_timer = 18
        if lane is not None and 0 <= lane < self.n_lanes:
            self.active_lane = lane
            self.emergency_lane = lane
        else:
            self.emergency_lane = self.active_lane
        self.decision_reason = f"MANUAL PRIORITY: Emergency corridor forced for {self.lanes[self.active_lane].name}."

    def cancel_emergency(self):
        self.emergency = False
        self.emergency_lane = None
        self.emergency_type = "None"
        self.phase_state = "green"
        weights = [self.lanes[i].weight for i in range(self.n_lanes)]
        self._cycle_timer = self.compute_green_time(weights[self.active_lane] if weights else 0.0)
        self.decision_reason = "Emergency override released. Resuming adaptive signal optimization."

    # ------------------------------------------------------------------ #
    #  State Payload                                                        #
    # ------------------------------------------------------------------ #
    def _build_state(self):
        quads = []
        total_vehicles = 0
        total_cars = 0
        total_trucks = 0
        total_buses = 0
        total_motos = 0

        for i in range(self.n_lanes):
            l = self.lanes[i]
            total_vehicles += sum(l.counts.values())
            total_cars += l.counts.get("car", 0)
            total_trucks += l.counts.get("truck", 0)
            total_buses += l.counts.get("bus", 0)
            total_motos += l.counts.get("motorcycle", 0)

            quads.append({
                "name":       l.name,
                "idx":        i,
                "loaded":     l.loaded,
                "counts":     l.counts.copy(),
                "weight":     l.weight,
                "green_time": l.green_time,
                "light":      l.light,
                "emergency":  l.emergency,
            })

        # Calculate density levels based on actual detection
        avg_queue = round(total_vehicles / max(1, self.n_lanes), 1)
        if total_vehicles < 6:
            density_level = "Low"
        elif total_vehicles < 16:
            density_level = "Medium"
        else:
            density_level = "High"

        return {
            "quads":            quads,
            "active_quad":      self.active_lane,
            "next_quad":        self.next_lane,
            "phase_state":      self.phase_state if not self.emergency else "emergency",
            "timer":            self._cycle_timer,
            "emergency":        self.emergency,
            "emergency_lane":   self.emergency_lane,
            "emergency_type":   self.emergency_type,
            "decision_reason":  self.decision_reason,
            "n_lanes":          self.n_lanes,
            "model_ready":      self.model_ready,
            "is_analyzing":     self.is_analyzing,
            "is_paused":        self.is_paused,
            "total_vehicles":   total_vehicles,
            "breakdown": {
                "car": total_cars,
                "truck": total_trucks,
                "bus": total_buses,
                "motorcycle": total_motos
            },
            "avg_queue_length": avg_queue,
            "traffic_density":  density_level,
            "settings": {
                "min_green": self.min_green,
                "max_green": self.max_green,
                "yellow_duration": self.yellow_duration,
                "conf_threshold": self.conf_threshold
            }
        }
