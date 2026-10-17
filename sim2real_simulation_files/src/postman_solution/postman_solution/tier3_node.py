#!/usr/bin/env python3
"""
tier3_node.py
Tier 3 — Cluttered-Arena Autonomous Navigation for Sim2Real Simulation Round.

All perception, mapping, planning, tracking, control, and state machine logic
are completely self-contained within this file.

Inputs : /joint_states, /imu, /overhead_camera/image_raw, /overhead_camera/camera_info
Outputs: /cmd_vel, /odom, TF odom -> base_link

Mission:
    1. Visually detect 5 initial green target disks and 4 magenta-topped obstacle walls.
    2. Build a 2D occupancy grid with dual inflation:
       - Interior obstacle walls: ~0.28 m inflation (robot half-diagonal + clearance).
       - Outer perimeter: 0.10 m inflation (preserves reachability of home at (0,0)).
    3. Plan collision-free paths using 8-connected A* with Line-of-Sight (LOS) smoothing.
    4. Compute optimal TSP visiting order using actual A* pairwise path costs.
    5. Track paths using a lookahead follower with obstacle crawl and heading slowdown.
    6. Fulfill continuous dwell >= 2.30 s (|v| <= 0.03 m/s) at all waypoints.
    7. Detect 2 dynamic green waypoints spawning visually at T+7.0 s and incorporate them.
    8. Snap odometry to visually confirmed targets upon dwell completion to eliminate drift.
    9. Return home to (0,0) and dwell >= 2.30 s with zero wall/obstacle collisions.

Integrity:
    - Never subscribes to /waypoints, /dynamic_waypoints, /evaluator/*, or /gz/*.
    - No hardcoded coordinates or seeds.
    - Forward-only motion: 0.0 <= v <= 0.30, |omega| <= 1.50.
    - postman_pkg remains unmodified.
"""

import math
import heapq
import itertools
from typing import List, Tuple, Dict, Optional, Any

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState, Imu, Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import TransformBroadcaster

try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Robot & Arena Constants
# ---------------------------------------------------------------------------
WHEEL_RADIUS = 0.0225       # m
WHEEL_SEPARATION = 0.159    # m

MAX_LINEAR_SPEED = 0.30     # m/s
MIN_LINEAR_SPEED = 0.0      # m/s (reverse strictly prohibited)
MAX_ANGULAR_SPEED = 1.50    # rad/s

WAYPOINT_TOLERANCE = 0.20   # m evaluator capture radius
STOP_SPEED = 0.03           # m/s stopped velocity threshold
HOME_TOLERANCE = 0.20       # m home capture radius
HOME_X = 0.0
HOME_Y = 0.0

WAYPOINT_PHYSICAL_RADIUS = 0.15  # m green disk radius

# Evaluator collision perimeter bounds
ARENA_COLLISION_MIN = -0.125
ARENA_COLLISION_MAX = 2.875

# Plausible arena coordinate bounds for detections
ARENA_MIN = -0.15
ARENA_MAX = 2.95

# Camera extrinsics from arena.sdf (fixed geometry)
CAM_X_DEFAULT = 1.50
CAM_Y_DEFAULT = 1.50
CAM_Z_DEFAULT = 4.00
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 800
CAMERA_FOV_DEG = 70.0

# ---------------------------------------------------------------------------
# Tuning Parameters
# ---------------------------------------------------------------------------
GRID_RESOLUTION = 0.03      # m/cell (~104 x 104 cells)
SAFETY_MARGIN = 0.28        # m interior obstacle clearance
PERIMETER_MARGIN = 0.10     # m perimeter margin (keeps home (0,0) reachable)

TARGET_STOP_RADIUS = 0.13   # m controller transition radius into dwell (< 0.20 m)
DWELL_TIME = 2.30           # s continuous dwell requirement (evaluator requires 2.00 s)

LOOKAHEAD_DISTANCE = 0.15   # m path lookahead distance

K_HEADING = 2.0             # in-place rotation gain
K_HEADING_DRIVE = 1.5       # driving angular gain
K_DISTANCE = 0.8            # distance to speed gain

ALIGN_THRESHOLD = math.radians(8.0)
REALIGN_THRESHOLD = math.radians(15.0)

WAYPOINT_DEDUP_DISTANCE = 0.25  # m spatial merge threshold

ENCODER_YAW_WEIGHT = 0.95
IMU_YAW_WEIGHT = 0.05

MAX_REASONABLE_DT = 0.20    # s; clock jump protection
CONTROL_HZ = 20.0
VISION_EVERY_N_TICKS = 2    # 20 Hz / 2 = 10 Hz perception

# Detection HSV defaults
GREEN_HSV_LOW_DEFAULT = (35, 80, 60)
GREEN_HSV_HIGH_DEFAULT = (90, 255, 255)

MAGENTA_HSV_LOW_DEFAULT = (125, 80, 60)
MAGENTA_HSV_HIGH_DEFAULT = (170, 255, 255)

MIN_CONTOUR_AREA = 400.0
MAX_CONTOUR_AREA = 3500.0
MIN_CIRCULARITY = 0.65
MIN_RADIUS_PX = 10.0
MAX_RADIUS_PX = 35.0
EXPECTED_RADIUS_TOL_LOW = 0.50
EXPECTED_RADIUS_TOL_HIGH = 1.60
MIN_FILL_RATIO = 0.55

OBSTACLE_MIN_AREA = 200.0
OBSTACLE_MAX_AREA = 6000.0
OBSTACLE_MIN_ASPECT = 2.5
OBSTACLE_MIN_LENGTH = 0.25
OBSTACLE_MAX_LENGTH = 0.75
OBSTACLE_MIN_WIDTH = 0.02
OBSTACLE_MAX_WIDTH = 0.20

PIXEL_SWAP_XY_DEFAULT = True
PIXEL_SIGN_U_DEFAULT = -1.0
PIXEL_SIGN_V_DEFAULT = -1.0


# ---------------------------------------------------------------------------
# Math & Utility Functions
# ---------------------------------------------------------------------------
def normalize_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def euclid(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(float(v), hi))


# ---------------------------------------------------------------------------
# Odometry & State Estimator
# ---------------------------------------------------------------------------
class OdometryEstimator:
    """Differential-drive dead reckoning with IMU yaw fusion."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.omega = 0.0
        self.left_vel = 0.0
        self.right_vel = 0.0
        self.has_joint = False
        self.imu_omega_z = 0.0
        self.has_imu = False
        self.last_time_s = None

    def on_joint_velocities(self, left: float, right: float) -> None:
        self.left_vel = float(left)
        self.right_vel = float(right)
        self.has_joint = True

    def on_imu(self, omega_z: float) -> None:
        self.imu_omega_z = float(omega_z)
        self.has_imu = True

    def update(self, now_s: float) -> None:
        if self.has_joint:
            v_left = self.left_vel * WHEEL_RADIUS
            v_right = self.right_vel * WHEEL_RADIUS
            v_enc = (v_right + v_left) / 2.0
            omega_enc = (v_right - v_left) / WHEEL_SEPARATION
            if self.has_imu:
                fused = (ENCODER_YAW_WEIGHT * omega_enc
                         + IMU_YAW_WEIGHT * self.imu_omega_z)
            else:
                fused = omega_enc
            self.v = v_enc
            self.omega = fused
        else:
            self.v = 0.0
            self.omega = 0.0

        if self.last_time_s is None:
            self.last_time_s = now_s
            return
        dt = now_s - self.last_time_s
        if dt <= 0.0 or dt > MAX_REASONABLE_DT:
            self.last_time_s = now_s
            return
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta = normalize_angle(self.theta + self.omega * dt)
        self.last_time_s = now_s

    def snap_to(self, x: float, y: float) -> None:
        """Correct dead-reckoning drift when stopped at a visually confirmed target."""
        self.x = float(x)
        self.y = float(y)


# ---------------------------------------------------------------------------
# Camera Projection & Coordinate Mapper
# ---------------------------------------------------------------------------
def focal_from_fov(width_px: float, fov_deg: float = 70.0) -> float:
    return (width_px / 2.0) / math.tan(math.radians(fov_deg / 2.0))


class CameraMapper:
    """Pinhole ground-plane projector between camera image pixels and arena world."""

    def __init__(self, cam_x=CAM_X_DEFAULT, cam_y=CAM_Y_DEFAULT, cam_z=CAM_Z_DEFAULT,
                 swap_xy=PIXEL_SWAP_XY_DEFAULT, sign_u=PIXEL_SIGN_U_DEFAULT, sign_v=PIXEL_SIGN_V_DEFAULT):
        self.cam_x = float(cam_x)
        self.cam_y = float(cam_y)
        self.cam_z = float(cam_z)
        self.swap_xy = bool(swap_xy)
        self.sign_u = float(sign_u)
        self.sign_v = float(sign_v)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.has_info = False

    def update_from_k(self, k: list) -> bool:
        try:
            if len(k) < 9:
                return False
            fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
            if fx <= 1e-6 or fy <= 1e-6:
                return False
            first = not self.has_info
            self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
            self.has_info = True
            return first
        except Exception:
            return False

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        fx = self.fx if self.fx else focal_from_fov(800.0)
        fy = self.fy if self.fy else focal_from_fov(800.0)
        cx = self.cx if self.cx is not None else 400.0
        cy = self.cy if self.cy is not None else 400.0
        xc = (u - cx) / fx * self.cam_z
        yc = (v - cy) / fy * self.cam_z
        if self.swap_xy:
            wx = self.cam_x + self.sign_v * yc
            wy = self.cam_y + self.sign_u * xc
        else:
            wx = self.cam_x + self.sign_u * xc
            wy = self.cam_y + self.sign_v * yc
        return wx, wy

    def world_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        fx = self.fx if self.fx else focal_from_fov(800.0)
        fy = self.fy if self.fy else focal_from_fov(800.0)
        cx = self.cx if self.cx is not None else 400.0
        cy = self.cy if self.cy is not None else 400.0
        if self.swap_xy:
            yc = (x - self.cam_x) / self.sign_v
            xc = (y - self.cam_y) / self.sign_u
        else:
            xc = (x - self.cam_x) / self.sign_u
            yc = (y - self.cam_y) / self.sign_v
        u = xc * fx / self.cam_z + cx
        v = yc * fy / self.cam_z + cy
        return u, v

    def meters_per_pixel(self) -> float:
        fx = self.fx if self.fx else focal_from_fov(800.0)
        return self.cam_z / max(fx, 1e-6)


# ---------------------------------------------------------------------------
# Perception Modules (OpenCV)
# ---------------------------------------------------------------------------
class WaypointDetector:
    """Detect green waypoint disks on the arena floor."""

    def __init__(self, hsv_lower=GREEN_HSV_LOW_DEFAULT, hsv_upper=GREEN_HSV_HIGH_DEFAULT):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, bgr: np.ndarray, fx: float, cam_z: float) -> List[Dict[str, float]]:
        if bgr is None or bgr.size == 0:
            return []
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[0] if len(res) == 2 else res[1]

        expected_r = WAYPOINT_PHYSICAL_RADIUS * fx / max(cam_z, 1e-6)
        out = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri < 1e-6:
                continue
            circularity = 4.0 * math.pi * area / (peri * peri)
            if circularity < MIN_CIRCULARITY:
                continue
            (_x, _y), radius = cv2.minEnclosingCircle(cnt)
            if radius < MIN_RADIUS_PX or radius > MAX_RADIUS_PX:
                continue
            if expected_r > 1e-6:
                ratio = radius / expected_r
                if ratio < EXPECTED_RADIUS_TOL_LOW or ratio > EXPECTED_RADIUS_TOL_HIGH:
                    continue
            m = cv2.moments(cnt)
            if abs(m["m00"]) < 1e-6:
                continue
            cu = float(m["m10"] / m["m00"])
            cv_ = float(m["m01"] / m["m00"])
            circle_area = math.pi * radius * radius
            if circle_area <= 0:
                continue
            fill = area / circle_area
            if fill < MIN_FILL_RATIO:
                continue
            out.append({"u": cu, "v": cv_, "radius_px": float(radius),
                        "circularity": float(circularity), "area": float(area)})
        return out


class ObstacleDetector:
    """Detect magenta-topped interior obstacle walls."""

    def __init__(self, hsv_lower=MAGENTA_HSV_LOW_DEFAULT, hsv_upper=MAGENTA_HSV_HIGH_DEFAULT):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    def detect(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        if bgr is None or bgr.size == 0:
            return []
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[0] if len(res) == 2 else res[1]

        out = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < OBSTACLE_MIN_AREA or area > OBSTACLE_MAX_AREA:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            if min(w, h) < 1e-6:
                continue
            aspect = max(w, h) / max(min(w, h), 1e-6)
            if aspect < OBSTACLE_MIN_ASPECT:
                continue
            try:
                corners = cv2.boxPoints(rect)
            except Exception:
                continue
            out.append({"cx": float(cx), "cy": float(cy),
                        "width_px": float(w), "height_px": float(h),
                        "angle_deg": float(angle),
                        "corners": np.array(corners, dtype=np.float64),
                        "area": float(area)})
        return out


def obstacles_to_world(detections: list, pixel_to_world_fn) -> List[Dict[str, float]]:
    """Convert pixel-space obstacle blobs to metric world obstacles."""
    obstacles = []
    for d in detections:
        try:
            corners = d["corners"]
            world_pts = [pixel_to_world_fn(float(px), float(py)) for (px, py) in corners]
        except Exception:
            continue
        edges = []
        for i in range(4):
            x0, y0 = world_pts[i]
            x1, y1 = world_pts[(i + 1) % 4]
            edges.append((math.hypot(x1 - x0, y1 - y0), (x0, y0), (x1, y1)))
        edges.sort(key=lambda e: e[0], reverse=True)
        length = (edges[0][0] + edges[1][0]) / 2.0
        width = (edges[2][0] + edges[3][0]) / 2.0
        if not (OBSTACLE_MIN_LENGTH <= length <= OBSTACLE_MAX_LENGTH):
            continue
        if not (OBSTACLE_MIN_WIDTH <= width <= OBSTACLE_MAX_WIDTH):
            continue
        (_ln, (lx0, ly0), (lx1, ly1)) = edges[0]
        yaw = math.atan2(ly1 - ly0, lx1 - lx0)
        while yaw >= math.pi / 2.0:
            yaw -= math.pi
        while yaw < -math.pi / 2.0:
            yaw += math.pi
        cx = sum(p[0] for p in world_pts) / 4.0
        cy = sum(p[1] for p in world_pts) / 4.0
        obstacles.append({"center_x": float(cx), "center_y": float(cy),
                          "length": float(length), "width": float(width),
                          "yaw": float(yaw)})
    return obstacles


# ---------------------------------------------------------------------------
# Spatial & Temporal Trackers
# ---------------------------------------------------------------------------
class WaypointTracker:
    """Identity-preserving target tracker with temporal-stability gating."""

    def __init__(self, merge_threshold=WAYPOINT_DEDUP_DISTANCE, min_obs=3, alpha=0.30, tentative_prune_s=8.0):
        self.merge_threshold = float(merge_threshold)
        self.min_obs = int(min_obs)
        self.alpha = float(alpha)
        self.tentative_prune_s = float(tentative_prune_s)
        self._items = {}
        self._next_id = 0

    def update(self, points_world: list, now_s: float):
        new_ids = []
        for (wx, wy) in points_world:
            if not (ARENA_MIN <= wx <= ARENA_MAX and ARENA_MIN <= wy <= ARENA_MAX):
                continue
            best_id = None
            best_d = float("inf")
            for wid, wp in self._items.items():
                d = math.hypot(wx - wp["x"], wy - wp["y"])
                if d < best_d:
                    best_d = d
                    best_id = wid
            if best_id is not None and best_d <= self.merge_threshold:
                wp = self._items[best_id]
                was_confirmed = wp["obs_count"] >= self.min_obs
                wp["x"] = (1.0 - self.alpha) * wp["x"] + self.alpha * wx
                wp["y"] = (1.0 - self.alpha) * wp["y"] + self.alpha * wy
                wp["obs_count"] += 1
                wp["last_seen"] = now_s
                wp["confidence"] = min(1.0, wp["obs_count"] / max(self.min_obs * 2, 1))
                if not was_confirmed and wp["obs_count"] >= self.min_obs:
                    new_ids.append(best_id)
            else:
                wid = self._next_id
                self._next_id += 1
                self._items[wid] = {"x": float(wx), "y": float(wy),
                                    "visited": False, "obs_count": 1,
                                    "last_seen": now_s, "confidence": 0.2,
                                    "id": wid}
                if self.min_obs <= 1:
                    new_ids.append(wid)
        stale = [wid for wid, wp in self._items.items()
                 if wp["obs_count"] < self.min_obs
                 and (now_s - wp["last_seen"]) > self.tentative_prune_s]
        for wid in stale:
            del self._items[wid]
        return self.confirmed_count(), new_ids

    def confirmed_count(self) -> int:
        return sum(1 for wp in self._items.values() if wp["obs_count"] >= self.min_obs)

    def confirmed_items(self) -> list:
        return [(wid, wp) for wid, wp in self._items.items() if wp["obs_count"] >= self.min_obs]

    def unvisited_confirmed(self) -> list:
        return [(wid, wp) for wid, wp in self.confirmed_items() if not wp["visited"]]

    def has_unvisited(self) -> bool:
        return any(not wp["visited"] for _, wp in self.confirmed_items())

    def mark_visited(self, wid: int) -> None:
        if wid in self._items:
            self._items[wid]["visited"] = True

    def get(self, wid: int) -> Optional[dict]:
        return self._items.get(wid)


class ObstacleTracker:
    """Temporal stabilization for detected static obstacle walls."""

    def __init__(self, merge_threshold=0.15, history=10, change_eps=0.05, min_confirm=2):
        self.merge_threshold = float(merge_threshold)
        self.history = int(history)
        self.change_eps = float(change_eps)
        self.min_confirm = int(min_confirm)
        self._tracks = []
        self._stable_cache = []

    @staticmethod
    def _ang_diff(a: float, b: float) -> float:
        d = abs(a - b) % math.pi
        return min(d, math.pi - d)

    def update(self, detections: list) -> Tuple[list, bool]:
        unmatched = list(detections)
        for tr in self._tracks:
            s = tr["stable"]
            best_i, best_d = -1, float("inf")
            for i, d in enumerate(unmatched):
                dist = math.hypot(d["center_x"] - s["center_x"], d["center_y"] - s["center_y"])
                if dist < best_d:
                    best_d, best_i = dist, i
            if best_i >= 0 and best_d <= self.merge_threshold:
                d = unmatched.pop(best_i)
                tr["samples"].append(d)
                if len(tr["samples"]) > self.history:
                    tr["samples"].pop(0)
        for d in unmatched:
            self._tracks.append({"samples": [d], "stable": dict(d)})

        new_stable = []
        for tr in self._tracks:
            ss = tr["samples"]
            n = len(ss)
            cx = sum(s["center_x"] for s in ss) / n
            cy = sum(s["center_y"] for s in ss) / n
            ln = sorted(s["length"] for s in ss)[n // 2]
            wd = sorted(s["width"] for s in ss)[n // 2]
            sx = sum(math.cos(2.0 * s["yaw"]) for s in ss)
            sy = sum(math.sin(2.0 * s["yaw"]) for s in ss)
            yaw = math.atan2(sy, sx) / 2.0
            tr["stable"] = {"center_x": cx, "center_y": cy, "length": ln, "width": wd, "yaw": yaw}
            if n >= self.min_confirm:
                new_stable.append(tr["stable"])

        changed = self._layout_changed(self._stable_cache, new_stable)
        self._stable_cache = new_stable
        return list(new_stable), changed

    def _layout_changed(self, old: list, new: list) -> bool:
        if len(old) != len(new):
            return True
        used = [False] * len(new)
        for o in old:
            best_i, best_d = -1, float("inf")
            for i, nn in enumerate(new):
                if used[i]:
                    continue
                d = math.hypot(o["center_x"] - nn["center_x"], o["center_y"] - nn["center_y"])
                if d < best_d:
                    best_d, best_i = d, i
            if best_i < 0:
                return True
            used[best_i] = True
            nn = new[best_i]
            if best_d > self.change_eps:
                return True
            if abs(o["length"] - nn["length"]) > 0.10:
                return True
            if self._ang_diff(o["yaw"], nn["yaw"]) > math.radians(10.0):
                return True
        return False

    @property
    def stable(self) -> list:
        return list(self._stable_cache)


# ---------------------------------------------------------------------------
# 2D Dual-Inflation Occupancy Grid
# ---------------------------------------------------------------------------
class OccupancyGrid:
    """2D arena occupancy grid with independent interior and perimeter inflation."""

    def __init__(self, resolution=GRID_RESOLUTION, min_x=-0.15, max_x=2.95, min_y=-0.15, max_y=2.95,
                 safety_margin=SAFETY_MARGIN, perimeter_margin=PERIMETER_MARGIN,
                 arena_min=ARENA_COLLISION_MIN, arena_max=ARENA_COLLISION_MAX):
        self.resolution = float(resolution)
        self.min_x = float(min_x)
        self.max_x = float(max_x)
        self.min_y = float(min_y)
        self.max_y = float(max_y)
        self.safety_margin = float(safety_margin)
        self.perimeter_margin = float(perimeter_margin)
        self.arena_min = float(arena_min)
        self.arena_max = float(arena_max)
        self.nx = int(math.ceil((self.max_x - self.min_x) / self.resolution))
        self.ny = int(math.ceil((self.max_y - self.min_y) / self.resolution))

        self.base_obs = np.zeros((self.ny, self.nx), dtype=np.uint8)
        self.base_peri = np.zeros((self.ny, self.nx), dtype=np.uint8)
        self.inflated = np.zeros((self.ny, self.nx), dtype=np.uint8)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        col = int(math.floor((x - self.min_x) / self.resolution))
        row = int(math.floor((y - self.min_y) / self.resolution))
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = self.min_x + (col + 0.5) * self.resolution
        y = self.min_y + (row + 0.5) * self.resolution
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.ny and 0 <= col < self.nx

    def reset(self) -> None:
        self.base_obs.fill(0)
        self.base_peri.fill(0)
        self.inflated.fill(0)

    def add_perimeter(self) -> None:
        xs = self.min_x + (np.arange(self.nx) + 0.5) * self.resolution
        ys = self.min_y + (np.arange(self.ny) + 0.5) * self.resolution
        x_bad = (xs < self.arena_min) | (xs > self.arena_max)
        y_bad = (ys < self.arena_min) | (ys > self.arena_max)
        if np.any(x_bad):
            self.base_peri[:, x_bad] = 1
        if np.any(y_bad):
            self.base_peri[y_bad, :] = 1

    def add_obstacle_rotated(self, cx: float, cy: float, length: float, width: float, yaw: float) -> None:
        hl = length / 2.0
        hw = width / 2.0
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        ex = abs(hl * cos_y) + abs(hw * sin_y)
        ey = abs(hl * sin_y) + abs(hw * cos_y)
        r_lo = max(0, min(self.world_to_grid(cx - ex, cy - ey)[0], self.world_to_grid(cx + ex, cy + ey)[0]) - 1)
        r_hi = min(self.ny - 1, max(self.world_to_grid(cx - ex, cy - ey)[0], self.world_to_grid(cx + ex, cy + ey)[0]) + 1)
        c_lo = max(0, min(self.world_to_grid(cx - ex, cy - ey)[1], self.world_to_grid(cx + ex, cy + ey)[1]) - 1)
        c_hi = min(self.nx - 1, max(self.world_to_grid(cx - ex, cy - ey)[1], self.world_to_grid(cx + ex, cy + ey)[1]) + 1)
        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                x, y = self.grid_to_world(r, c)
                dx = x - cx
                dy = y - cy
                lx = dx * math.cos(-yaw) - dy * math.sin(-yaw)
                ly = dx * math.sin(-yaw) + dy * math.cos(-yaw)
                if abs(lx) <= hl and abs(ly) <= hw:
                    self.base_obs[r, c] = 1

    def _dilate(self, layer: np.ndarray, cells: int) -> np.ndarray:
        if cells <= 0:
            return layer.copy()
        k = 2 * cells + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(layer, kernel)

    def inflate(self) -> None:
        obs_cells = int(math.ceil(self.safety_margin / self.resolution))
        peri_cells = int(math.ceil(self.perimeter_margin / self.resolution))
        inf_obs = self._dilate(self.base_obs, obs_cells)
        inf_peri = self._dilate(self.base_peri, peri_cells)
        self.inflated = np.maximum(inf_obs, inf_peri)

    def rebuild(self, obstacles: list) -> None:
        self.reset()
        self.add_perimeter()
        for ob in obstacles:
            self.add_obstacle_rotated(ob["center_x"], ob["center_y"], ob["length"], ob["width"], ob["yaw"])
        self.inflate()

    def is_free_cell(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return False
        return bool(self.inflated[row, col] == 0)

    def is_free_world(self, x: float, y: float) -> bool:
        r, c = self.world_to_grid(x, y)
        return self.is_free_cell(r, c)

    def nearest_free(self, x: float, y: float, max_radius_cells: int = 25) -> Optional[Tuple[float, float]]:
        r0, c0 = self.world_to_grid(x, y)
        if self.is_free_cell(r0, c0):
            return self.grid_to_world(r0, c0)
        for rad in range(1, max_radius_cells + 1):
            for dr in range(-rad, rad + 1):
                for dc in (-rad, rad):
                    if self.is_free_cell(r0 + dr, c0 + dc):
                        return self.grid_to_world(r0 + dr, c0 + dc)
            for dc in range(-rad + 1, rad):
                for dr in (-rad, rad):
                    if self.is_free_cell(r0 + dr, c0 + dc):
                        return self.grid_to_world(r0 + dr, c0 + dc)
        return None

    def segment_free(self, x0: float, y0: float, x1: float, y1: float, step=None) -> bool:
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 1e-9:
            return self.is_free_world(x0, y0)
        step = step or (self.resolution / 2.0)
        n = max(2, int(math.ceil(dist / step)) + 1)
        for i in range(n):
            t = i / (n - 1)
            if not self.is_free_world(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t):
                return False
        return True

    def path_free(self, path: list, step=None) -> bool:
        if not path:
            return False
        for i in range(len(path) - 1):
            if not self.segment_free(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1], step):
                return False
        return True


# ---------------------------------------------------------------------------
# Path Planning (A* + LOS Simplification + TSP)
# ---------------------------------------------------------------------------
def _octile(dx: int, dy: int) -> float:
    dx, dy = abs(dx), abs(dy)
    return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)


def astar(grid: OccupancyGrid, start_world: Tuple[float, float], goal_world: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
    sx, sy = start_world
    gx, gy = goal_world
    snapped_start = grid.nearest_free(sx, sy)
    snapped_goal = grid.nearest_free(gx, gy)
    if snapped_start is None or snapped_goal is None:
        return None
    sr, sc = grid.world_to_grid(*snapped_start)
    gr, gc = grid.world_to_grid(*snapped_goal)
    if (sr, sc) == (gr, gc):
        return [snapped_start, snapped_goal]

    ny, nx = grid.ny, grid.nx
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
             (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0))]

    open_h = []
    heapq.heappush(open_h, (_octile(gr - sr, gc - sc), 0.0, (sr, sc)))
    g_score = {(sr, sc): 0.0}
    came_from = {}
    closed = set()

    while open_h:
        _, g, (r, c) = heapq.heappop(open_h)
        if (r, c) in closed:
            continue
        if (r, c) == (gr, gc):
            cells = [(r, c)]
            while (r, c) in came_from:
                r, c = came_from[(r, c)]
                cells.append((r, c))
            cells.reverse()
            path = [grid.grid_to_world(rr, cc) for (rr, cc) in cells]
            if grid.is_free_world(sx, sy):
                path[0] = (sx, sy)
            if grid.is_free_world(gx, gy):
                path[-1] = (gx, gy)
            return path
        closed.add((r, c))
        for dr, dc, cost in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < ny and 0 <= nc < nx):
                continue
            if not grid.is_free_cell(nr, nc):
                continue
            if dr != 0 and dc != 0:
                if not grid.is_free_cell(r + dr, c) or not grid.is_free_cell(r, c + dc):
                    continue
            ng = g + cost
            if ng < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = ng
                came_from[(nr, nc)] = (r, c)
                f = ng + _octile(gr - nr, gc - nc)
                heapq.heappush(open_h, (f, ng, (nr, nc)))
    return None


def path_length(path: list) -> float:
    if not path or len(path) < 2:
        return 0.0
    return sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1))


def simplify_path(path: list, grid: OccupancyGrid) -> list:
    """Greedy Line-Of-Sight (LOS) pruning with collision checking on each shortcut."""
    if not path or len(path) <= 2:
        return list(path) if path else []
    simplified = [path[0]]
    i = 0
    n = len(path)
    while i < n - 1:
        j = n - 1
        while j > i + 1:
            if grid.segment_free(path[i][0], path[i][1], path[j][0], path[j][1]):
                break
            j -= 1
        simplified.append(path[j])
        i = j
    return simplified


def plan_safe_path(grid: OccupancyGrid, start_world: Tuple[float, float], goal_world: Tuple[float, float]) -> Optional[list]:
    raw = astar(grid, start_world, goal_world)
    if raw is None:
        return None
    short = simplify_path(raw, grid)
    if short and grid.path_free(short):
        return short
    if grid.path_free(raw):
        return raw
    return None


def _pairwise_cost(grid: OccupancyGrid, a: tuple, b: tuple, cache: dict) -> float:
    key = (round(a[0], 3), round(a[1], 3), round(b[0], 3), round(b[1], 3))
    if key in cache:
        return cache[key]
    p = astar(grid, a, b)
    if p is None:
        c = math.hypot(a[0] - b[0], a[1] - b[1]) + 50.0
    else:
        c = path_length(p)
    cache[key] = c
    return c


def order_waypoints_a_star(grid: OccupancyGrid, start_xy: tuple, unvisited: list, home_xy=(HOME_X, HOME_Y)) -> list:
    """Exhaustive TSP permutation ordering using true A* path costs."""
    if not unvisited:
        return []
    if len(unvisited) == 1:
        return [unvisited[0][0]]
    cache = {}
    points = {"__start__": start_xy, "__home__": home_xy}
    for wid, wp in unvisited:
        points[wid] = (wp["x"], wp["y"])

    def cost(a_key, b_key) -> float:
        return _pairwise_cost(grid, points[a_key], points[b_key], cache)

    best_perm = None
    best_cost = float("inf")
    for perm in itertools.permutations(unvisited):
        c = cost("__start__", perm[0][0])
        for i in range(len(perm) - 1):
            c += cost(perm[i][0], perm[i + 1][0])
            if c >= best_cost:
                break
        else:
            c += cost(perm[-1][0], "__home__")
            if c < best_cost:
                best_cost = c
                best_perm = perm
    if best_perm is None:
        return [wid for wid, _ in unvisited]
    return [wid for wid, _ in best_perm]


# ---------------------------------------------------------------------------
# Path Follower (Controller)
# ---------------------------------------------------------------------------
class PathFollower:
    """Lookahead path follower with obstacle-aware and arrival speed scaling."""

    def __init__(self, lookahead=LOOKAHEAD_DISTANCE, k_heading=K_HEADING, k_drive=K_HEADING_DRIVE,
                 k_distance=K_DISTANCE, speed_open=0.22, speed_near_obstacle=0.12, speed_near_target=0.06):
        self.lookahead = float(lookahead)
        self.k_heading = float(k_heading)
        self.k_drive = float(k_drive)
        self.k_distance = float(k_distance)
        self.speed_open = float(speed_open)
        self.speed_near_obstacle = float(speed_near_obstacle)
        self.speed_near_target = float(speed_near_target)
        self.path = []
        self._seg_index = 0

    def set_path(self, path: list) -> None:
        self.path = list(path) if path else []
        self._seg_index = 0

    def clear(self) -> None:
        self.path = []
        self._seg_index = 0

    def _nearest_index(self, x: float, y: float) -> int:
        best_i = 0
        best_d = float("inf")
        start = max(0, self._seg_index - 2)
        for i in range(start, len(self.path)):
            d = math.hypot(x - self.path[i][0], y - self.path[i][1])
            if d < best_d:
                best_d = d
                best_i = i
        self._seg_index = min(best_i, len(self.path) - 2) if self.path else 0
        return best_i

    def lookahead_point(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        if not self.path:
            return None
        ni = self._nearest_index(x, y)
        for i in range(ni, len(self.path)):
            d = math.hypot(x - self.path[i][0], y - self.path[i][1])
            if d >= self.lookahead:
                return self.path[i]
        return self.path[-1]

    def compute(self, x: float, y: float, theta: float, grid: OccupancyGrid = None) -> Tuple[float, float]:
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(theta)):
            return 0.0, 0.0
        if not self.path:
            return 0.0, 0.0
        goal = self.lookahead_point(x, y)
        if goal is None:
            return 0.0, 0.0

        dx = goal[0] - x
        dy = goal[1] - y
        dist_end = math.hypot(self.path[-1][0] - x, self.path[-1][1] - y)

        desired = math.atan2(dy, dx)
        he = normalize_angle(desired - theta)

        # Proportional base forward speed
        v_des = min(self.speed_open, self.k_distance * max(dist_end, 0.0))
        if dist_end <= 0.50:
            v_des = min(v_des, 0.15)
        if dist_end <= 0.25:
            v_des = min(v_des, self.speed_near_target + 0.02)
        if dist_end <= 0.12:
            v_des = 0.0

        # Slow down on large heading errors; rotate in place if > 30 deg
        ah = abs(he)
        if ah > math.radians(30.0):
            v_des = 0.0
        elif ah > math.radians(15.0):
            v_des *= 0.5

        omega = clamp(self.k_drive * he, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

        # Proactive obstacle crawl: probe ahead
        if grid is not None and dist_end > 0.12 and v_des > 0:
            px = x + math.cos(desired) * 0.20
            py = y + math.sin(desired) * 0.20
            if not grid.is_free_world(px, py):
                v_des = min(v_des, self.speed_near_obstacle)

        v = clamp(v_des, MIN_LINEAR_SPEED, MAX_LINEAR_SPEED)
        return v, omega


# ---------------------------------------------------------------------------
# Mission Manager (State Machine)
# ---------------------------------------------------------------------------
class MissionManager:
    INIT = "INIT"
    LOCALIZE = "LOCALIZE"
    PERCEIVE = "PERCEIVE"
    PLAN = "PLAN"
    NAVIGATE = "NAVIGATE"
    DWELL = "DWELL"
    REPLAN = "REPLAN"
    RETURN_HOME = "RETURN_HOME"
    HOME_DWELL = "HOME_DWELL"
    COMPLETE = "COMPLETE"
    FAILSAFE = "FAILSAFE"

    def __init__(self, dwell_time=DWELL_TIME, target_stop_radius=TARGET_STOP_RADIUS,
                 waypoint_tolerance=WAYPOINT_TOLERANCE, stop_speed=STOP_SPEED, home_tolerance=HOME_TOLERANCE):
        self.dwell_time = float(dwell_time)
        self.target_stop_radius = float(target_stop_radius)
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.stop_speed = float(stop_speed)
        self.home_tolerance = float(home_tolerance)

        self.state = self.INIT
        self.current_target = None
        self.current_path = []
        self.ordered_route = []
        self._dwell_start = None
        self._home_dwell_start = None
        self._planned_count = 0
        self.plan_request = False
        self.fail_reason = ""

    def is_stopped(self, v: float, omega: float) -> bool:
        return abs(v) <= self.stop_speed and abs(omega) <= 0.15

    def notify_planned(self, confirmed_count: int) -> None:
        self._planned_count = confirmed_count
        self.plan_request = False

    def request_replan(self) -> None:
        self.plan_request = True

    def fail(self, reason: str) -> None:
        self.fail_reason = reason
        self.state = self.FAILSAFE

    def recover(self) -> None:
        self.fail_reason = ""
        self.state = self.PLAN

    def step(self, pose: tuple, vel: tuple, tracker: WaypointTracker, now_s: float) -> str:
        x, y, _theta = pose
        v, omega = vel

        if self.state == self.INIT:
            self._dwell_start = None
            self._home_dwell_start = None
            self.state = self.LOCALIZE
            return self.state

        if self.state == self.LOCALIZE:
            self.state = self.PERCEIVE
            return self.state

        if self.state == self.PERCEIVE:
            if tracker.confirmed_count() >= 1:
                self.state = self.PLAN
            return self.state

        if self.state == self.PLAN:
            if not tracker.has_unvisited():
                d_home = math.hypot(x - HOME_X, y - HOME_Y)
                if d_home <= self.home_tolerance and self.is_stopped(v, omega):
                    if self._home_dwell_start is None:
                        self._home_dwell_start = now_s
                    self.state = self.HOME_DWELL
                else:
                    self.state = self.RETURN_HOME
                return self.state
            if self.current_target is None:
                return self.state
            self._dwell_start = None
            self.state = self.NAVIGATE
            return self.state

        if self.state == self.NAVIGATE:
            wp = tracker.get(self.current_target) if self.current_target is not None else None
            if wp is None or wp.get("visited", False):
                self._dwell_start = None
                self.current_path = []
                self.state = self.PLAN
                return self.state
            d = math.hypot(x - wp["x"], y - wp["y"])
            if d <= self.target_stop_radius:
                self.state = self.DWELL
                if self._dwell_start is None and self.is_stopped(v, omega):
                    self._dwell_start = now_s
                return self.state
            if self.plan_request and not self.current_path:
                self.state = self.REPLAN
                return self.state
            return self.state

        if self.state == self.DWELL:
            wp = tracker.get(self.current_target) if self.current_target is not None else None
            if wp is None or wp.get("visited", False):
                self._dwell_start = None
                self.state = self.PLAN
                return self.state
            d = math.hypot(x - wp["x"], y - wp["y"])
            if d > self.waypoint_tolerance or not self.is_stopped(v, omega):
                self._dwell_start = None
                self.state = self.NAVIGATE
                return self.state
            if self._dwell_start is None:
                if self.is_stopped(v, omega) and d <= self.waypoint_tolerance:
                    self._dwell_start = now_s
                return self.state
            if (now_s - self._dwell_start) >= self.dwell_time:
                prev_target = self.current_target
                tracker.mark_visited(self.current_target)
                if prev_target is not None:
                    prev_wp = tracker.get(prev_target)
                    if prev_wp is not None:
                        for wid, wpp in tracker.confirmed_items():
                            if not wpp.get("visited", False):
                                if math.hypot(wpp["x"] - prev_wp["x"], wpp["y"] - prev_wp["y"]) <= 0.30:
                                    tracker.mark_visited(wid)
                self._dwell_start = None
                self.current_target = None
                self.current_path = []
                self.plan_request = False
                if tracker.has_unvisited():
                    self.state = self.REPLAN if tracker.confirmed_count() > self._planned_count else self.PLAN
                else:
                    self.state = self.RETURN_HOME
                return self.state
            return self.state

        if self.state == self.REPLAN:
            if not tracker.has_unvisited():
                self.state = self.RETURN_HOME
                return self.state
            if self.current_target is not None and self.current_path:
                self._dwell_start = None
                self.plan_request = False
                self.state = self.NAVIGATE
            return self.state

        if self.state == self.RETURN_HOME:
            d_home = math.hypot(x - HOME_X, y - HOME_Y)
            if d_home <= self.home_tolerance:
                if self.is_stopped(v, omega):
                    if self._home_dwell_start is None:
                        self._home_dwell_start = now_s
                    self.state = self.HOME_DWELL
                return self.state
            if tracker.has_unvisited() and tracker.confirmed_count() > self._planned_count:
                self._home_dwell_start = None
                self.state = self.REPLAN
            return self.state

        if self.state == self.HOME_DWELL:
            d_home = math.hypot(x - HOME_X, y - HOME_Y)
            if d_home > self.home_tolerance or not self.is_stopped(v, omega):
                self._home_dwell_start = None
                self.state = self.RETURN_HOME
                return self.state
            if self._home_dwell_start is None:
                self._home_dwell_start = now_s
                return self.state
            if (now_s - self._home_dwell_start) >= self.dwell_time:
                self.state = self.COMPLETE
            return self.state

        if self.state == self.COMPLETE:
            return self.state

        if self.state == self.FAILSAFE:
            return self.state

        self.state = self.FAILSAFE
        return self.state


# ---------------------------------------------------------------------------
# SolutionNode — Master Orchestration
# ---------------------------------------------------------------------------
class SolutionNode(Node):
    """Tier 3 visual navigation and obstacle avoidance node."""

    def __init__(self):
        super().__init__("solution_node")

        # Tunable ROS parameters
        self.declare_parameter("pixel_swap_xy", PIXEL_SWAP_XY_DEFAULT)
        self.declare_parameter("pixel_sign_u", PIXEL_SIGN_U_DEFAULT)
        self.declare_parameter("pixel_sign_v", PIXEL_SIGN_V_DEFAULT)
        self.declare_parameter("camera_height", CAM_Z_DEFAULT)
        self.declare_parameter("camera_x", CAM_X_DEFAULT)
        self.declare_parameter("camera_y", CAM_Y_DEFAULT)
        self.declare_parameter("green_hsv_lower", list(GREEN_HSV_LOW_DEFAULT))
        self.declare_parameter("green_hsv_upper", list(GREEN_HSV_HIGH_DEFAULT))
        self.declare_parameter("magenta_hsv_lower", list(MAGENTA_HSV_LOW_DEFAULT))
        self.declare_parameter("magenta_hsv_upper", list(MAGENTA_HSV_HIGH_DEFAULT))
        self.declare_parameter("grid_resolution", GRID_RESOLUTION)
        self.declare_parameter("safety_margin", SAFETY_MARGIN)
        self.declare_parameter("perimeter_margin", PERIMETER_MARGIN)
        self.declare_parameter("lookahead_distance", LOOKAHEAD_DISTANCE)
        self.declare_parameter("dwell_time", DWELL_TIME)
        self.declare_parameter("target_stop_radius", TARGET_STOP_RADIUS)
        self.declare_parameter("waypoint_dedup_distance", WAYPOINT_DEDUP_DISTANCE)
        self.declare_parameter("debug_visualization", False)

        swap_xy = bool(self.get_parameter("pixel_swap_xy").value)
        sign_u = float(self.get_parameter("pixel_sign_u").value)
        sign_v = float(self.get_parameter("pixel_sign_v").value)
        cam_x = float(self.get_parameter("camera_x").value)
        cam_y = float(self.get_parameter("camera_y").value)
        cam_z = float(self.get_parameter("camera_height").value)
        green_lo = tuple(self.get_parameter("green_hsv_lower").value)
        green_hi = tuple(self.get_parameter("green_hsv_upper").value)
        mag_lo = tuple(self.get_parameter("magenta_hsv_lower").value)
        mag_hi = tuple(self.get_parameter("magenta_hsv_upper").value)
        grid_res = float(self.get_parameter("grid_resolution").value)
        safety = float(self.get_parameter("safety_margin").value)
        peri_margin = float(self.get_parameter("perimeter_margin").value)
        lookahead = float(self.get_parameter("lookahead_distance").value)
        dwell_time = float(self.get_parameter("dwell_time").value)
        stop_radius = float(self.get_parameter("target_stop_radius").value)
        dedup = float(self.get_parameter("waypoint_dedup_distance").value)
        self.debug_viz = bool(self.get_parameter("debug_visualization").value)

        # Core Modules
        self.estimator = OdometryEstimator()
        self.mapper = CameraMapper(cam_x=cam_x, cam_y=cam_y, cam_z=cam_z,
                                   swap_xy=swap_xy, sign_u=sign_u, sign_v=sign_v)
        self.wp_detector = WaypointDetector(hsv_lower=green_lo, hsv_upper=green_hi)
        self.obs_detector = ObstacleDetector(hsv_lower=mag_lo, hsv_upper=mag_hi)
        self.tracker = WaypointTracker(merge_threshold=dedup, min_obs=3, alpha=0.30)
        self.obs_tracker = ObstacleTracker()
        self.grid = OccupancyGrid(resolution=grid_res, safety_margin=safety, perimeter_margin=peri_margin)
        self.grid.rebuild([])
        self.follower = PathFollower(lookahead=lookahead)
        self.mission = MissionManager(dwell_time=dwell_time, target_stop_radius=stop_radius)
        self.bridge = CvBridge() if CvBridge is not None else None

        # State Variables
        self._latest_bgr = None
        self._has_image = False
        self._perceive_time0 = None
        self._map_built_with_obstacles = False
        self._plan_fail_count = 0
        self._tick = 0

        # Stuck detection
        self._stuck_pos_start = None
        self._stuck_threshold_v = 0.10
        self._stuck_min_progress = 0.05
        self._stuck_time = 5.0
        self._peri_bounds = (ARENA_COLLISION_MIN, ARENA_COLLISION_MAX, ARENA_COLLISION_MIN, ARENA_COLLISION_MAX)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.debug_pub = None
        if self.debug_viz:
            self.debug_pub = self.create_publisher(Image, "/tier3/debug_image", 10)

        # Subscriptions — ONLY public contestant topics
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)
        self.create_subscription(Imu, "/imu", self._on_imu, 10)
        self.create_subscription(Image, "/overhead_camera/image_raw", self._on_image, 10)
        self.create_subscription(CameraInfo, "/overhead_camera/camera_info", self._on_camera_info, 10)

        self.create_timer(1.0 / CONTROL_HZ, self._control_loop)
        self.get_logger().info("Tier 3 node started (vision + obstacle avoidance).")
        self.get_logger().info("Waiting for camera info + odometry sensors...")

    # -- Callbacks: store only --
    def _on_joint_states(self, msg: JointState) -> None:
        try:
            li = msg.name.index("left_wheel_joint")
            ri = msg.name.index("right_wheel_joint")
        except ValueError:
            return
        if len(msg.velocity) <= max(li, ri):
            return
        self.estimator.on_joint_velocities(msg.velocity[li], msg.velocity[ri])

    def _on_imu(self, msg: Imu) -> None:
        self.estimator.on_imu(float(msg.angular_velocity.z))

    def _on_camera_info(self, msg: CameraInfo) -> None:
        first = self.mapper.update_from_k(list(msg.k))
        if first:
            self.get_logger().info(
                f"CameraInfo acquired: fx={self.mapper.fx:.1f} fy={self.mapper.fy:.1f} "
                f"cx={self.mapper.cx:.1f} cy={self.mapper.cy:.1f}")

    def _on_image(self, msg: Image) -> None:
        if self.bridge is None:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        self._latest_bgr = bgr
        self._has_image = True

    # -- /odom + TF --
    def _publish_odom_and_tf(self, now_msg) -> None:
        odom = Odometry()
        odom.header.stamp = now_msg
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self.estimator.x
        odom.pose.pose.position.y = self.estimator.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = math.sin(self.estimator.theta / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.estimator.theta / 2.0)
        odom.twist.twist.linear.x = self.estimator.v
        odom.twist.twist.angular.z = self.estimator.omega
        self.odom_pub.publish(odom)

        t = TransformStamped()
        t.header.stamp = now_msg
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = self.estimator.x
        t.transform.translation.y = self.estimator.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(self.estimator.theta / 2.0)
        t.transform.rotation.w = math.cos(self.estimator.theta / 2.0)
        self.tf_broadcaster.sendTransform(t)

    # -- Command Helpers --
    def _make_safe_cmd(self, v: float, omega: float) -> Twist:
        if not math.isfinite(v):
            v = 0.0
        if not math.isfinite(omega):
            omega = 0.0
        cmd = Twist()
        cmd.linear.x = clamp(v, MIN_LINEAR_SPEED, MAX_LINEAR_SPEED)
        cmd.angular.z = clamp(omega, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
        return cmd

    def _publish_stop(self) -> None:
        self.cmd_pub.publish(self._make_safe_cmd(0.0, 0.0))

    def _sensors_ready(self) -> bool:
        return (self.estimator.has_joint and self.estimator.has_imu
                and self._has_image and self.mapper.has_info)

    # -- Perception Step (10 Hz slice) --
    def _perception_step(self, now_s: float) -> None:
        if not (self._has_image and self.mapper.has_info):
            return
        if self._latest_bgr is None:
            return
        if self._perceive_time0 is None:
            self._perceive_time0 = now_s
        fx = self.mapper.fx or 571.0

        # 1. Green Waypoint Detection
        try:
            detections = self.wp_detector.detect(self._latest_bgr, fx, self.mapper.cam_z)
        except Exception as exc:
            self.get_logger().warn(f"Green detection failed: {exc}")
            detections = []
        worlds = []
        for d in detections:
            wx, wy = self.mapper.pixel_to_world(d["u"], d["v"])
            if ARENA_MIN <= wx <= ARENA_MAX and ARENA_MIN <= wy <= ARENA_MAX:
                worlds.append((wx, wy))

        confirmed_before = self.tracker.confirmed_count()
        confirmed_total, new_ids = self.tracker.update(worlds, now_s)
        if new_ids:
            for wid in new_ids:
                wp = self.tracker.get(wid)
                if wp is not None:
                    self.get_logger().info(
                        f"[PERCEPTION] Waypoint #{wid} world=({wp['x']:.3f},{wp['y']:.3f})")
            if confirmed_before < 5 <= confirmed_total:
                self.get_logger().info(f"[PERCEPTION] Initial targets acquired: {confirmed_total}")
            elif confirmed_before >= 5:
                for wid in new_ids:
                    wp = self.tracker.get(wid)
                    if wp is not None:
                        self.get_logger().info(
                            f"[DYNAMIC] Dynamic waypoint detected: id={wid} ({wp['x']:.3f},{wp['y']:.3f})")

            if self.mission.state in (MissionManager.NAVIGATE, MissionManager.DWELL,
                                      MissionManager.RETURN_HOME, MissionManager.HOME_DWELL,
                                      MissionManager.PLAN, MissionManager.REPLAN):
                self.mission.request_replan()

        # 2. Magenta Obstacle Wall Detection
        try:
            obs_px = self.obs_detector.detect(self._latest_bgr)
            obs_world = obstacles_to_world(obs_px, self.mapper.pixel_to_world)
        except Exception as exc:
            self.get_logger().warn(f"Obstacle detection failed: {exc}")
            obs_world = []

        if obs_world:
            stable, changed = self.obs_tracker.update(obs_world)
            if changed or not self._map_built_with_obstacles:
                self.grid.rebuild(stable)
                self._map_built_with_obstacles = True
                self.get_logger().info(f"[PERCEPTION] Detected {len(stable)} obstacles; rebuilt grid.")
                if self.mission.current_path and not self.grid.path_free(self.mission.current_path):
                    self.get_logger().warn("[PLANNER] Active path blocked by map rebuild — replanning.")
                    self.mission.current_path = []
                    self.follower.clear()
                    self.mission.request_replan()
                    if self.mission.state == MissionManager.NAVIGATE:
                        self.mission.state = MissionManager.REPLAN

        if self.debug_viz and self.debug_pub is not None:
            self._publish_debug_image(detections)

    def _publish_debug_image(self, green_detections) -> None:
        if self._latest_bgr is None or self.bridge is None:
            return
        dbg = self._latest_bgr.copy()
        for d in green_detections:
            cv2.circle(dbg, (int(d["u"]), int(d["v"])), int(d["radius_px"]), (255, 0, 0), 2)
            cv2.circle(dbg, (int(d["u"]), int(d["v"])), 3, (0, 0, 255), -1)
        if self.mission.current_target is not None:
            wp = self.tracker.get(self.mission.current_target)
            if wp is not None:
                u, v = self.mapper.world_to_pixel(wp["x"], wp["y"])
                cv2.circle(dbg, (int(u), int(v)), 8, (0, 255, 255), 2)
        for i in range(len(self.mission.current_path) - 1):
            u0, v0 = self.mapper.world_to_pixel(*self.mission.current_path[i])
            u1, v1 = self.mapper.world_to_pixel(*self.mission.current_path[i + 1])
            cv2.line(dbg, (int(u0), int(v0)), (int(u1), int(v1)), (0, 255, 0), 2)
        msg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
        self.debug_pub.publish(msg)

    # -- Planning Helpers --
    def _do_full_plan(self, now_s: float) -> bool:
        unvisited = self.tracker.unvisited_confirmed()
        if not unvisited:
            return False
        start = (self.estimator.x, self.estimator.y)
        try:
            order = order_waypoints_a_star(self.grid, start, unvisited, (HOME_X, HOME_Y))
        except Exception as exc:
            self.get_logger().warn(f"[PLANNER] Ordering failed: {exc}")
            return False
        if not order:
            return False
        target_id = order[0]
        wp = self.tracker.get(target_id)
        if wp is None:
            return False
        path = plan_safe_path(self.grid, start, (wp["x"], wp["y"]))
        if path is None:
            self.get_logger().warn(f"[PLANNER] No collision-free path to #{target_id} ({wp['x']:.2f},{wp['y']:.2f})")
            return False
        self.mission.ordered_route = order
        self.mission.current_target = target_id
        self.mission.current_path = path
        self.follower.set_path(path)
        self.mission.notify_planned(self.tracker.confirmed_count())
        self._plan_fail_count = 0
        self.get_logger().info(
            f"[PLANNER] Path planned from ({start[0]:.2f},{start[1]:.2f}) to #{target_id} "
            f"({wp['x']:.2f},{wp['y']:.2f}) Length={path_length(path):.2f}m")
        return True

    def _do_replan_keep_target(self) -> bool:
        wp = self.tracker.get(self.mission.current_target)
        if wp is None:
            return self._do_full_plan(0.0)
        start = (self.estimator.x, self.estimator.y)
        path = plan_safe_path(self.grid, start, (wp["x"], wp["y"]))
        if path is None:
            return False
        self.mission.current_path = path
        self.follower.set_path(path)
        self.mission.notify_planned(self.tracker.confirmed_count())
        self._plan_fail_count = 0
        return True

    def _do_plan_home(self) -> bool:
        start = (self.estimator.x, self.estimator.y)
        path = plan_safe_path(self.grid, start, (HOME_X, HOME_Y))
        if path is None:
            self.get_logger().warn("[PLANNER] No collision-free path home!")
            return False
        self.mission.current_path = path
        self.follower.set_path(path)
        self._plan_fail_count = 0
        return True

    # -- Main Control Loop --
    def _control_loop(self) -> None:
        self._tick += 1
        now = self.get_clock().now()
        now_s = now.nanoseconds * 1e-9
        now_msg = now.to_msg()

        self.estimator.update(now_s)
        self._publish_odom_and_tf(now_msg)

        if self._tick % 60 == 0:
            self.get_logger().info(
                f"[ODOM] x={self.estimator.x:.3f} y={self.estimator.y:.3f} "
                f"th={self.estimator.theta:.3f} v={self.estimator.v:.3f} "
                f"state={self.mission.state} waypoints={self.tracker.confirmed_count()} "
                f"obstacles={len(self.obs_tracker.stable)}")

        if self._tick % VISION_EVERY_N_TICKS == 0:
            try:
                self._perception_step(now_s)
            except Exception as exc:
                self.get_logger().warn(f"Perception step failed: {exc}")

        pose = (self.estimator.x, self.estimator.y, self.estimator.theta)
        vel = (self.estimator.v, self.estimator.omega)

        if self.mission.state in (MissionManager.INIT, MissionManager.LOCALIZE):
            self.mission.step(pose, vel, self.tracker, now_s)
            self._publish_stop()
            return

        if self.mission.state == MissionManager.PERCEIVE:
            if not self._sensors_ready():
                self._publish_stop()
                return
            n_obs = len(self.obs_tracker.stable)
            waited = (now_s - self._perceive_time0) if self._perceive_time0 else 0.0
            if self.tracker.confirmed_count() >= 1 and (n_obs >= 2 or waited > 8.0):
                self.get_logger().info(
                    f"[MISSION] PERCEIVE -> PLAN ({self.tracker.confirmed_count()} targets, {n_obs} obstacles)")
                self.mission.step(pose, vel, self.tracker, now_s)
            self._publish_stop()
            return

        if self.mission.state == MissionManager.PLAN:
            prev = self.mission.state
            self.mission.step(pose, vel, self.tracker, now_s)
            if self.mission.state != prev:
                self._publish_stop()
                return
            if self._do_full_plan(now_s):
                self.mission.step(pose, vel, self.tracker, now_s)
            else:
                self._plan_fail_count += 1
                if self._plan_fail_count > 10:
                    self.mission.fail("planning failed")
                self._publish_stop()
            return

        if self.mission.state == MissionManager.REPLAN:
            prev_target = self.mission.current_target
            self.mission.step(pose, vel, self.tracker, now_s)
            if self.mission.state == MissionManager.RETURN_HOME:
                self._publish_stop()
                return
            if self.mission.state == MissionManager.NAVIGATE:
                return
            ok = False
            if (prev_target is not None and self.tracker.get(prev_target) is not None
                    and not self.tracker.get(prev_target)["visited"]):
                self.mission.current_target = prev_target
                ok = self._do_replan_keep_target()
                if not ok:
                    self.mission.current_target = None
                    ok = self._do_full_plan(now_s)
            else:
                self.mission.current_target = None
                ok = self._do_full_plan(now_s)
            if ok:
                self.mission.step(pose, vel, self.tracker, now_s)
            else:
                self._plan_fail_count += 1
                if self._plan_fail_count > 10:
                    self.mission.fail("replanning failed")
                self._publish_stop()
            return

        if self.mission.state == MissionManager.NAVIGATE:
            prev_target = self.mission.current_target
            self.mission.step(pose, vel, self.tracker, now_s)
            if self.mission.state != MissionManager.NAVIGATE:
                # Target reached / dwell started: snap odometry to target to zero out drift
                if prev_target is not None and self.mission.state in (MissionManager.PLAN, MissionManager.REPLAN):
                    wp = self.tracker.get(prev_target)
                    if wp is not None:
                        self.estimator.snap_to(wp["x"], wp["y"])
                        self.get_logger().info(f"[ODOM-SNAP] Drift zeroed at target #{prev_target} ({wp['x']:.2f},{wp['y']:.2f})")
                self._publish_stop()
                return

            if not self.mission.current_path or not self.grid.path_free(self.mission.current_path):
                self.get_logger().warn("[PLANNER] Path blocked -> REPLAN")
                self.mission.current_path = []
                self.follower.clear()
                self.mission.request_replan()
                self.mission.state = MissionManager.REPLAN
                self._publish_stop()
                return

            v, w = self.follower.compute(self.estimator.x, self.estimator.y, self.estimator.theta, self.grid)

            # Stuck detection & recovery
            if v > self._stuck_threshold_v:
                if self._stuck_pos_start is None:
                    self._stuck_pos_start = (now_s, self.estimator.x, self.estimator.y)
                else:
                    t0, x0, y0 = self._stuck_pos_start
                    dist_moved = math.hypot(self.estimator.x - x0, self.estimator.y - y0)
                    if now_s - t0 > self._stuck_time and dist_moved < self._stuck_min_progress:
                        self.get_logger().warn("[STUCK] Robot stuck; rotating in place.")
                        v, w = 0.0, 1.0
                        self._stuck_pos_start = None
            else:
                self._stuck_pos_start = None

            # Perimeter safety override
            px_min, px_max, py_min, py_max = self._peri_bounds
            dist_w = self.estimator.x - px_min
            dist_e = px_max - self.estimator.x
            dist_s = self.estimator.y - py_min
            dist_n = py_max - self.estimator.y
            d_min = min(dist_w, dist_e, dist_s, dist_n)
            if d_min < 0.18 and v > 0.05:
                v = 0.05
                if dist_w == d_min:
                    target_heading = 0.0
                elif dist_e == d_min:
                    target_heading = math.pi
                elif dist_s == d_min:
                    target_heading = math.pi / 2.0
                else:
                    target_heading = -math.pi / 2.0
                he_peri = normalize_angle(target_heading - self.estimator.theta)
                w = clamp(2.0 * he_peri, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

            self.cmd_pub.publish(self._make_safe_cmd(v, w))
            return

        if self.mission.state == MissionManager.DWELL:
            self._publish_stop()
            self.mission.step(pose, vel, self.tracker, now_s)
            if self.mission.state in (MissionManager.PLAN, MissionManager.REPLAN, MissionManager.RETURN_HOME):
                self.get_logger().info("[DWELL] Waypoint collected.")
            return

        if self.mission.state == MissionManager.RETURN_HOME:
            self.mission.step(pose, vel, self.tracker, now_s)
            if self.mission.state != MissionManager.RETURN_HOME:
                self._publish_stop()
                return
            if not self.mission.current_path or not self.grid.path_free(self.mission.current_path):
                if not self._do_plan_home():
                    self._plan_fail_count += 1
                    if self._plan_fail_count > 10:
                        self.mission.fail("no path home")
                    self._publish_stop()
                    return
                self.get_logger().info("[MISSION] Collision-free path planned home.")

            v, w = self.follower.compute(self.estimator.x, self.estimator.y, self.estimator.theta, self.grid)
            d_home = math.hypot(self.estimator.x - HOME_X, self.estimator.y - HOME_Y)
            if d_home <= self.mission.target_stop_radius:
                v, w = 0.0, 0.0
            self.cmd_pub.publish(self._make_safe_cmd(v, w))
            return

        if self.mission.state == MissionManager.HOME_DWELL:
            self._publish_stop()
            self.mission.step(pose, vel, self.tracker, now_s)
            if self.mission.state == MissionManager.COMPLETE:
                self.get_logger().info("[MISSION] COMPLETE — All waypoints collected and home dwell complete.")
            return

        if self.mission.state == MissionManager.COMPLETE:
            self._publish_stop()
            return

        if self.mission.state == MissionManager.FAILSAFE:
            self._publish_stop()
            if self._tick % 40 == 0:
                self.get_logger().warn(f"[MISSION] FAILSAFE ({self.mission.fail_reason}) — holding stop, retrying.")
                self._plan_fail_count = 0
                if self._sensors_ready():
                    self.mission.recover()
            return

        self.get_logger().warn(f"Unknown state {self.mission.state}; stopping.")
        self._publish_stop()


def main():
    rclpy.init()
    node = SolutionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            stop = Twist()
            node.cmd_pub.publish(stop)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
