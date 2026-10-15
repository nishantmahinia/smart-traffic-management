"""
SmartTraffic AI — Intersection Simulation & Signal Controller
Multi-phase 4-way intersection simulation engine.
Weight-based adaptive phase timing with emergency priority preemption.
"""

import random
import time
import threading

LANES = ["N", "S", "E", "W"]
LANE_NAMES = {"N": "North", "S": "South", "E": "East", "W": "West"}


def compute_weight(lane_data):
    return round(
        lane_data["cars"] * 1.0 +
        lane_data["trucks"] * 1.5 +
        lane_data["buses"] * 2.5,
        1
    )


class SimEngine:
    def __init__(self, socketio=None, mode="dynamic", initial_seed=None):
        self.socketio = socketio
        self.lock = threading.Lock()
        self.running = False
        self.mode = mode  # "dynamic" or "fixed"
        
        # Signal timing configurations
        self.min_green = 6
        self.max_green = 30
        self.yellow_duration = 3
        
        if initial_seed is not None:
            self.rng = random.Random(initial_seed)
        else:
            self.rng = random.Random()

        # Per-lane vehicle queues
        self.lanes = {
            lane: {
                "cars":   self.rng.randint(2, 8),
                "trucks": self.rng.randint(0, 3),
                "buses":  self.rng.randint(0, 2)
            }
            for lane in LANES
        }

        self.active_lane = "N"
        self.next_lane = "E"
        self.phase_state = "green"  # "green" or "yellow"
        self.timer = self._get_timer_duration(self.active_lane)
        self.decision_reason = "Adaptive engine active — Initial green phase allocated to North approach."
        
        self.emergency = False
        self.emergency_lane = None   # which lane has the corridor open
        self.emergency_countdown = 0
        self.stats = {"total_vehicles": 0, "cycles": 0, "vehicles_passed": 0}
        self.weights = {}
        self._recalc_weights()
        
        # Ambulance tracking: {lane: count}
        self.ambulances = {lane: 0 for lane in LANES}
        self.AMBULANCE_SPAWN_RATE = 0.015  # 1.5% per tick per lane
        self.AMBULANCE_CORRIDOR_DURATION = 18  # seconds

    def update_settings(self, min_green=None, max_green=None, yellow_duration=None):
        with self.lock:
            if min_green is not None:
                self.min_green = max(3, int(min_green))
            if max_green is not None:
                self.max_green = max(self.min_green, int(max_green))
            if yellow_duration is not None:
                self.yellow_duration = max(1, int(yellow_duration))

    def _compute_timer(self, weight):
        return max(self.min_green, min(self.max_green, int(weight * 1.2)))

    def _get_timer_duration(self, lane):
        if self.mode == "fixed":
            return 30
        else:
            return self._compute_timer(compute_weight(self.lanes[lane]))

    def _recalc_weights(self):
        for lane in LANES:
            self.weights[lane] = compute_weight(self.lanes[lane])

    def _pick_next_lane(self):
        if self.mode == "fixed":
            # Round robin for fixed mode
            idx = LANES.index(self.active_lane)
            return LANES[(idx + 1) % len(LANES)]
        else:
            # Heaviest lane for dynamic mode (prefer different lane if available)
            candidates = [l for l in LANES if l != self.active_lane]
            if candidates:
                return max(candidates, key=lambda l: self.weights[l])
            return max(self.weights, key=lambda l: self.weights[l])

    def _arrival_tick(self):
        """Simulate vehicle arrivals and departures each tick."""
        for lane in LANES:
            # Random ambulance spawn in dynamic simulation mode
            if self.mode == "dynamic" and self.rng.random() < self.AMBULANCE_SPAWN_RATE:
                self.ambulances[lane] = self.ambulances.get(lane, 0) + 1
                # Auto-trigger emergency corridor for this lane
                if not self.emergency:
                    self._do_trigger_emergency(lane)

            # Arrivals on all lanes
            if self.rng.random() < 0.6:
                vtype = self.rng.choices(
                    ["cars", "trucks", "buses"], weights=[70, 20, 10]
                )[0]
                self.lanes[lane][vtype] = min(
                    self.lanes[lane][vtype] + self.rng.randint(1, 3), 20
                )
                self.stats["total_vehicles"] += 1

            # Departures: active green lane OR the emergency corridor lane (only during green)
            is_clearing = (lane == self.active_lane and self.phase_state == "green") or (self.emergency and lane == self.emergency_lane)
            if is_clearing:
                for vtype in ["cars", "trucks", "buses"]:
                    depart = self.rng.randint(1, 4)
                    actual_depart = min(self.lanes[lane][vtype], depart)
                    self.lanes[lane][vtype] = max(
                        0, self.lanes[lane][vtype] - depart
                    )
                    self.stats["vehicles_passed"] += actual_depart
                # Clear ambulances on the corridor/green lane
                if self.ambulances.get(lane, 0) > 0:
                    self.ambulances[lane] = 0

    def tick(self):
        with self.lock:
            if self.emergency:
                self._arrival_tick()
                self._recalc_weights()
                self.emergency_countdown -= 1
                self.decision_reason = f"EMERGENCY PREEMPTION: Safe green corridor opened for {LANE_NAMES.get(self.emergency_lane, 'Unknown')} approach ({self.emergency_countdown}s remaining)."
                if self.emergency_countdown <= 0:
                    self.emergency = False
                    self.emergency_lane = None
                    self.phase_state = "green"
                    self.timer = self._get_timer_duration(self.active_lane)
                    self.decision_reason = "Emergency corridor released. Resuming adaptive signal optimization."
            else:
                self.timer -= 1
                if self.timer <= 0:
                    if self.phase_state == "green" and self.mode == "dynamic" and self.yellow_duration > 0:
                        # Safe transition to Yellow
                        self.phase_state = "yellow"
                        self.timer = self.yellow_duration
                        self.next_lane = self._pick_next_lane()
                        self.decision_reason = f"Yellow warning clearance ({self.yellow_duration}s) on {LANE_NAMES[self.active_lane]} approach before switching to {LANE_NAMES[self.next_lane]}."
                    else:
                        # Switch to green on next lane
                        self._arrival_tick()
                        self._recalc_weights()
                        self.active_lane = self.next_lane if self.next_lane else self._pick_next_lane()
                        self.next_lane = self._pick_next_lane()
                        self.phase_state = "green"
                        self.timer = self._get_timer_duration(self.active_lane)
                        self.stats["cycles"] += 1
                        weight = self.weights.get(self.active_lane, 0.0)
                        q_count = sum(self.lanes[self.active_lane].values())
                        self.decision_reason = f"{LANE_NAMES[self.active_lane]} approach selected: highest queue weight ({weight}) with {q_count} vehicles -> Allocated {self.timer}s Green."
                else:
                    self._arrival_tick()
                    self._recalc_weights()

            # Build state payload
            state = {
                "lanes": {},
                "active_lane": self.active_lane,
                "next_lane": self.next_lane,
                "phase_state": self.phase_state if not self.emergency else "emergency",
                "timer": self.timer if not self.emergency else self.emergency_countdown,
                "emergency": self.emergency,
                "emergency_lane": self.emergency_lane,
                "decision_reason": self.decision_reason,
                "stats": self.stats.copy(),
                "weights": self.weights.copy()
            }
            for lane in LANES:
                w = self.weights[lane]
                if self.emergency:
                    light = "green" if lane == self.emergency_lane else "red"
                elif lane == self.active_lane:
                    light = self.phase_state  # "green" or "yellow"
                else:
                    light = "red"

                state["lanes"][lane] = {
                    **self.lanes[lane],
                    "ambulances": self.ambulances.get(lane, 0),
                    "weight": w,
                    "green_time": self._get_timer_duration(lane),
                    "name": LANE_NAMES[lane],
                    "light": light
                }

        return state

    def _do_trigger_emergency(self, lane=None):
        """Internal — must be called with lock held or before lock needed."""
        self.emergency = True
        self.emergency_countdown = self.AMBULANCE_CORRIDOR_DURATION
        self.phase_state = "green"
        self._recalc_weights()
        if lane in LANES:
            self.active_lane = lane
            self.emergency_lane = lane
        else:
            self.active_lane = self._pick_next_lane()
            self.emergency_lane = self.active_lane
        self.decision_reason = f"EMERGENCY OVERRIDE: Corridor assigned to {LANE_NAMES[self.emergency_lane]} approach."

    def trigger_emergency(self, lane=None):
        with self.lock:
            self._do_trigger_emergency(lane)

    def cancel_emergency(self):
        with self.lock:
            self.emergency = False
            self.emergency_lane = None
            self.emergency_countdown = 0
            self.phase_state = "green"
            self.timer = self._get_timer_duration(self.active_lane)
            self.decision_reason = "Emergency override cancelled by operator. Standard adaptive control restored."

    def run_loop(self):
        self.running = True
        while self.running:
            state = self.tick()
            if self.socketio:
                self.socketio.emit("sim_state", state)
            time.sleep(1.0)

    def stop(self):
        self.running = False


class ComparisonRunner:
    def __init__(self, socketio):
        self.socketio = socketio
        self.running = False
        self.lock = threading.Lock()
        self.last_results = None
        
    def start_comparison(self):
        with self.lock:
            if self.running:
                return False
            self.running = True
            
        seed = int(time.time() * 1000) % 1000000
        self.sim_fixed = SimEngine(mode="fixed", initial_seed=seed)
        self.sim_dynamic = SimEngine(mode="dynamic", initial_seed=seed)
        
        threading.Thread(target=self._run_loop, daemon=True).start()
        return True

    def reset(self):
        with self.lock:
            self.running = False
            self.last_results = None
        
    def _run_loop(self):
        ticks = 0
        max_ticks = 60
        cum_fixed_q = 0
        cum_dyn_q = 0

        while self.running and ticks < max_ticks:
            state_fixed = self.sim_fixed.tick()
            state_dynamic = self.sim_dynamic.tick()
            
            # Sum current queues across all 4 approaches
            fixed_q_now = sum(sum(self.sim_fixed.lanes[l].values()) for l in LANES)
            dyn_q_now = sum(sum(self.sim_dynamic.lanes[l].values()) for l in LANES)
            cum_fixed_q += fixed_q_now
            cum_dyn_q += dyn_q_now
            ticks += 1

            fixed_passed = state_fixed["stats"]["vehicles_passed"]
            dyn_passed = state_dynamic["stats"]["vehicles_passed"]

            # Compute real measured comparison metrics
            metrics = {
                "fixed": {
                    "vehicles_passed": fixed_passed,
                    "avg_queue_length": round(cum_fixed_q / (ticks * 4), 1),
                    "avg_waiting_time": round(cum_fixed_q / max(1, fixed_passed), 1),
                },
                "dynamic": {
                    "vehicles_passed": dyn_passed,
                    "avg_queue_length": round(cum_dyn_q / (ticks * 4), 1),
                    "avg_waiting_time": round(cum_dyn_q / max(1, dyn_passed), 1),
                }
            }
            
            payload = {
                "fixed": state_fixed,
                "dynamic": state_dynamic,
                "metrics": metrics,
                "progress": int((ticks / max_ticks) * 100),
                "remaining": max_ticks - ticks,
                "completed": False
            }
            
            if self.socketio:
                self.socketio.emit("comparison_state", payload)
            
            time.sleep(0.2)
            
        # Final payload
        if self.running:
            state_fixed = self.sim_fixed.tick()
            state_dynamic = self.sim_dynamic.tick()
            fixed_passed = state_fixed["stats"]["vehicles_passed"]
            dyn_passed = state_dynamic["stats"]["vehicles_passed"]
            
            metrics = {
                "fixed": {
                    "vehicles_passed": fixed_passed,
                    "avg_queue_length": round(cum_fixed_q / (max(1, ticks) * 4), 1),
                    "avg_waiting_time": round(cum_fixed_q / max(1, fixed_passed), 1),
                },
                "dynamic": {
                    "vehicles_passed": dyn_passed,
                    "avg_queue_length": round(cum_dyn_q / (max(1, ticks) * 4), 1),
                    "avg_waiting_time": round(cum_dyn_q / max(1, dyn_passed), 1),
                }
            }

            self.last_results = {
                "fixed": state_fixed,
                "dynamic": state_dynamic,
                "metrics": metrics,
                "progress": 100,
                "remaining": 0,
                "completed": True
            }

            if self.socketio:
                self.socketio.emit("comparison_state", self.last_results)
                
        self.running = False
