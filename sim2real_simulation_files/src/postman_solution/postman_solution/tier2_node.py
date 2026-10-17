#!/usr/bin/env python3
"""
tier2_node.py
Tier 2 — Perception-Guided Navigation for Sim2Real Simulation Round.

Inputs : /joint_states, /imu, /overhead_camera/image_raw, /overhead_camera/camera_info
Outputs: /cmd_vel, /odom, TF odom -> base_link

Mission:
    1. Visually detect 5 initial green waypoint disks (overhead camera).
    2. Navigate to each, dwelling >= DWELL_TIME inside 0.20 m.
    3. Continuously perceive while navigating; detect 2 dynamic disks
       that visually appear at T+7.0 s after first motion.
    4. Visit all 7, return to (0,0), dwell, DONE.

Integrity:
    - No /waypoints, /dynamic_waypoints subscriptions (Tier-1-only, suppressed).
    - No /evaluator/*, no /gz/*, no hardcoded waypoint coordinates/seeds.
    - Forward-only motion: 0.0 <= v <= 0.30, |omega| <= 1.50.
    - postman_pkg remains unmodified.
"""

import math
import itertools
from typing import Tuple

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
# Organizer-defined constants
# ---------------------------------------------------------------------------
WHEEL_RADIUS = 0.0225       # m
WHEEL_SEPARATION = 0.159    # m

MAX_LINEAR_SPEED = 0.30     # m/s
MIN_LINEAR_SPEED = 0.0      # m/s (reverse prohibited)
MAX_ANGULAR_SPEED = 1.50    # rad/s

WAYPOINT_TOLERANCE = 0.20   # m capture radius (evaluator)
STOP_SPEED = 0.03           # m/s stopped threshold
HOME_TOLERANCE = 0.20       # m
HOME_X = 0.0
HOME_Y = 0.0

WAYPOINT_PHYSICAL_RADIUS = 0.15  # m green disk radius

# Camera extrinsics from arena.sdf (fixed geometry)
CAM_X_DEFAULT = 1.50
CAM_Y_DEFAULT = 1.50
CAM_Z_DEFAULT = 4.00

# Arena sanity bounds (evaluator: -0.125..2.875; waypoints 0.45..2.40)
ARENA_MIN = -0.15
ARENA_MAX = 2.95

# Perimeter safety (arena collision bounds from evaluator.py: -0.125..2.875)
PERIMETER_MIN_X, PERIMETER_MAX_X = -0.125, 2.875
PERIMETER_MIN_Y, PERIMETER_MAX_Y = -0.125, 2.875
PERIMETER_SAFETY_DIST = 0.10  # m; steer away when estimated pose is this close to a wall

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
K_ANGLE = 2.0               # in-place rotation gain
K_ANGLE_DRIVE = 1.5         # driving angular gain
K_DISTANCE = 0.8            # distance -> speed gain

ALIGN_THRESHOLD = math.radians(8.0)
REALIGN_THRESHOLD = math.radians(15.0)

BRAKE_DISTANCE = 0.20
STOP_APPROACH_DISTANCE = 0.10

ENCODER_YAW_WEIGHT = 0.95
IMU_YAW_WEIGHT = 0.05

MAX_REASONABLE_DT = 0.20    # s
CONTROL_HZ = 20.0
VISION_EVERY_N_TICKS = 2    # 20 Hz / 2 = 10 Hz perception

# Safety dwell with margin (2.30 s instead of bare 2.00 s)
DWELL_TIME = 2.30

# Conservative approach caps
SPEED_MID = 0.15            # 0.25 < d <= 0.50
SPEED_NEAR = 0.08           # 0.12 < d <= 0.25
# d <= 0.12 -> 0.0 (BRAKE/DWELL takes over)

# Vision: HSV green window
HSV_LOWER_DEFAULT = (35, 80, 60)
HSV_UPPER_DEFAULT = (90, 255, 255)

# Vision: contour acceptance
MIN_CONTOUR_AREA = 400.0
MAX_CONTOUR_AREA = 3500.0
MIN_CIRCULARITY = 0.65
MIN_RADIUS_PX = 10.0
MAX_RADIUS_PX = 35.0
EXPECTED_RADIUS_TOL_LOW = 0.50
EXPECTED_RADIUS_TOL_HIGH = 1.60

# Tracker
MERGE_THRESHOLD = 0.12      # m
MIN_OBSERVATIONS = 3        # frames before candidate is confirmed
SMOOTHING_ALPHA = 0.30      # weight of new observation (0..1)
TENTATIVE_PRUNE_S = 8.0     # drop unconfirmed candidates older than this

# Pixel->world default orientation
PIXEL_SWAP_XY_DEFAULT = True
PIXEL_SIGN_U_DEFAULT = -1.0
PIXEL_SIGN_V_DEFAULT = -1.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def euclid(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(float(v), hi))


def pixel_to_world(u: float, v: float, fx: float, fy: float,
                   cx: float, cy: float, cam_x: float, cam_y: float,
                   cam_z: float, swap_xy: bool,
                   sign_u: float, sign_v: float):
    """Pinhole ground-plane projection + axis mapping."""
    xc = (u - cx) / fx * cam_z
    yc = (v - cy) / fy * cam_z
    if swap_xy:
        wx = cam_x + sign_v * yc
        wy = cam_y + sign_u * xc
    else:
        wx = cam_x + sign_u * xc
        wy = cam_y + sign_v * yc
    return wx, wy


# ---------------------------------------------------------------------------
# OdometryEstimator
# ---------------------------------------------------------------------------
class OdometryEstimator:
    """Differential-drive dead reckoning with single coherent yaw fusion."""

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


# ---------------------------------------------------------------------------
# WaypointDetector — classical OpenCV only
# ---------------------------------------------------------------------------
class WaypointDetector:
    """Green-disk segmentation: HSV threshold -> morph -> contours -> filter."""

    def __init__(self, hsv_lower=HSV_LOWER_DEFAULT, hsv_upper=HSV_UPPER_DEFAULT):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, bgr: np.ndarray, fx: float, cam_z: float):
        """Return list of dicts: {u, v, radius_px, circularity, area}."""
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
            if fill < 0.55:
                continue
            out.append({"u": cu, "v": cv_, "radius_px": float(radius),
                        "circularity": float(circularity), "area": float(area)})
        return out


# ---------------------------------------------------------------------------
# WaypointTracker
# ---------------------------------------------------------------------------
class WaypointTracker:
    """Identity-preserving tracker with temporal-stability gating."""

    def __init__(self, merge_threshold=MERGE_THRESHOLD,
                 min_obs=MIN_OBSERVATIONS, alpha=SMOOTHING_ALPHA):
        self.merge_threshold = merge_threshold
        self.min_obs = min_obs
        self.alpha = alpha
        self._items = {}   # id -> dict(x,y,visited,obs_count,last_seen,confidence)
        self._next_id = 0

    def update(self, points_world, now_s: float):
        """Merge [(x,y), ...] detections. Returns (confirmed_total, new_ids)."""
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
                 and (now_s - wp["last_seen"]) > TENTATIVE_PRUNE_S]
        for wid in stale:
            del self._items[wid]
        return self.confirmed_count(), new_ids

    def confirmed_count(self) -> int:
        return sum(1 for wp in self._items.values()
                   if wp["obs_count"] >= self.min_obs)

    def confirmed_items(self):
        return [(wid, wp) for wid, wp in self._items.items()
                if wp["obs_count"] >= self.min_obs]

    def unvisited_confirmed(self):
        return [(wid, wp) for wid, wp in self.confirmed_items()
                if not wp["visited"]]

    def has_unvisited(self) -> bool:
        return any(not wp["visited"] for _, wp in self.confirmed_items())

    def mark_visited(self, wid) -> None:
        if wid in self._items:
            self._items[wid]["visited"] = True

    def get(self, wid):
        return self._items.get(wid)


# ---------------------------------------------------------------------------
# RoutePlanner — exact small TSP including home in the cost
# ---------------------------------------------------------------------------
def plan_route(start_xy, unvisited, home_xy=(HOME_X, HOME_Y)):
    """Brute-force shortest route: start -> perm(unvisited) -> home."""
    if not unvisited:
        return []
    best_perm = None
    best_cost = float("inf")
    for perm in itertools.permutations(unvisited):
        cost = euclid(start_xy[0], start_xy[1], perm[0][1]["x"], perm[0][1]["y"])
        for i in range(len(perm) - 1):
            cost += euclid(perm[i][1]["x"], perm[i][1]["y"],
                           perm[i + 1][1]["x"], perm[i + 1][1]["y"])
        cost += euclid(perm[-1][1]["x"], perm[-1][1]["y"], home_xy[0], home_xy[1])
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    return [wid for wid, _ in best_perm] if best_perm is not None else []


# ---------------------------------------------------------------------------
# SolutionNode — ROS wiring + mission state machine
# ---------------------------------------------------------------------------
class SolutionNode(Node):
    WAIT_FOR_SENSORS = "WAIT_FOR_SENSORS"
    DETECT_INITIAL_TARGETS = "DETECT_INITIAL_TARGETS"
    PLAN_ROUTE = "PLAN_ROUTE"
    ALIGN = "ALIGN"
    DRIVE = "DRIVE"
    BRAKE = "BRAKE"
    DWELL = "DWELL"
    REPLAN = "REPLAN"
    HOME = "HOME"
    HOME_DWELL = "HOME_DWELL"
    DONE = "DONE"

    def __init__(self):
        super().__init__("solution_node")

        # Tunable ROS parameters
        self.declare_parameter("pixel_swap_xy", PIXEL_SWAP_XY_DEFAULT)
        self.declare_parameter("pixel_sign_u", PIXEL_SIGN_U_DEFAULT)
        self.declare_parameter("pixel_sign_v", PIXEL_SIGN_V_DEFAULT)
        self.declare_parameter("camera_height", CAM_Z_DEFAULT)
        self.declare_parameter("camera_x", CAM_X_DEFAULT)
        self.declare_parameter("camera_y", CAM_Y_DEFAULT)
        self.declare_parameter("hsv_lower", list(HSV_LOWER_DEFAULT))
        self.declare_parameter("hsv_upper", list(HSV_UPPER_DEFAULT))
        self.declare_parameter("debug_visualization", False)
        self.declare_parameter("dwell_time", DWELL_TIME)

        self.swap_xy = bool(self.get_parameter("pixel_swap_xy").value)
        self.sign_u = float(self.get_parameter("pixel_sign_u").value)
        self.sign_v = float(self.get_parameter("pixel_sign_v").value)
        self.cam_x = float(self.get_parameter("camera_x").value)
        self.cam_y = float(self.get_parameter("camera_y").value)
        self.cam_z_param = float(self.get_parameter("camera_height").value)
        hsv_lo = list(self.get_parameter("hsv_lower").value)
        hsv_hi = list(self.get_parameter("hsv_upper").value)
        self.debug_viz = bool(self.get_parameter("debug_visualization").value)
        self.dwell_time = float(self.get_parameter("dwell_time").value)

        # Modules
        self.estimator = OdometryEstimator()
        self.detector = WaypointDetector(hsv_lower=tuple(hsv_lo),
                                         hsv_upper=tuple(hsv_hi))
        self.tracker = WaypointTracker()
        self.bridge = CvBridge() if CvBridge is not None else None

        # Sensor state (callbacks only store; timer decides)
        self._latest_bgr = None
        self._has_image = False
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None
        self._has_camera_info = False

        # Mission state
        self._state = self.WAIT_FOR_SENSORS
        self._route = []
        self._current_target = None
        self._dwell_start = None
        self._home_dwell_start = None
        self._planned_count = 0
        self._last_confirmed = 0
        self._tick = 0

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.debug_pub = None
        if self.debug_viz:
            self.debug_pub = self.create_publisher(Image, "/tier2/debug_image", 10)

        # Subscriptions — ONLY public contestant topics (suppressed topics NOT subscribed)
        self.create_subscription(JointState, "/joint_states",
                                 self._on_joint_states, 10)
        self.create_subscription(Imu, "/imu", self._on_imu, 10)
        self.create_subscription(Image, "/overhead_camera/image_raw",
                                 self._on_image, 10)
        self.create_subscription(CameraInfo, "/overhead_camera/camera_info",
                                 self._on_camera_info, 10)

        self.create_timer(1.0 / CONTROL_HZ, self._control_loop)
        self.get_logger().info("Tier 2 node started (perception-guided).")
        self.get_logger().info("Waiting for camera info + odometry sensors...")

    # -- Callbacks: store only ------------------------------------------------
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
        try:
            k = list(msg.k)
            if len(k) < 9:
                return
            fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
            if fx <= 1e-6 or fy <= 1e-6:
                return
            first = not self._has_camera_info
            self._fx, self._fy, self._cx, self._cy = fx, fy, cx, cy
            self._has_camera_info = True
            if first:
                self.get_logger().info(
                    f"CameraInfo acquired: fx={fx:.1f} fy={fy:.1f} "
                    f"cx={cx:.1f} cy={cy:.1f}")
        except Exception:
            return

    def _on_image(self, msg: Image) -> None:
        if self.bridge is None:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        self._latest_bgr = bgr
        self._has_image = True

    # -- /odom + TF -----------------------------------------------------------
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

    # -- Command helpers ------------------------------------------------------
    def _make_safe_cmd(self, v: float, omega: float) -> Twist:
        if not math.isfinite(v):
            v = 0.0
        if not math.isfinite(omega):
            omega = 0.0
        cmd = Twist()
        cmd.linear.x = clamp(v, MIN_LINEAR_SPEED, MAX_LINEAR_SPEED)
        cmd.angular.z = clamp(omega, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
        return cmd

    def _perimeter_safe_cmd(self, v: float, omega: float) -> Tuple[float, float]:
        """Override command if estimated pose is too close to a perimeter wall."""
        dist_west = self.estimator.x - PERIMETER_MIN_X
        dist_east = PERIMETER_MAX_X - self.estimator.x
        dist_south = self.estimator.y - PERIMETER_MIN_Y
        dist_north = PERIMETER_MAX_Y - self.estimator.y
        min_dist = min(dist_west, dist_east, dist_south, dist_north)

        if min_dist >= PERIMETER_SAFETY_DIST:
            return v, omega

        v = 0.0
        if dist_west == min_dist:
            target_heading = 0.0
        elif dist_east == min_dist:
            target_heading = math.pi
        elif dist_south == min_dist:
            target_heading = math.pi / 2.0
        else:
            target_heading = -math.pi / 2.0

        heading_error = normalize_angle(target_heading - self.estimator.theta)
        omega = clamp(K_ANGLE * heading_error, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
        return v, omega

    def _publish_stop(self) -> None:
        self.cmd_pub.publish(self._make_safe_cmd(0.0, 0.0))

    def _publish_rotation(self, heading_error: float) -> None:
        self.cmd_pub.publish(
            self._make_safe_cmd(0.0, K_ANGLE * heading_error))

    def _publish_drive(self, distance: float, heading_error: float) -> None:
        v_des = min(MAX_LINEAR_SPEED, K_DISTANCE * distance)
        v_des *= max(0.0, math.cos(heading_error))
        if distance <= 0.50:
            v_des = min(v_des, SPEED_MID)
        if distance <= 0.25:
            v_des = min(v_des, SPEED_NEAR)
        if distance <= 0.12:
            v_des = 0.0
        ah = abs(heading_error)
        if ah > math.radians(30.0):
            v_des = 0.0
        elif ah > math.radians(15.0):
            v_des *= 0.5
        omega = K_ANGLE_DRIVE * heading_error
        v_safe, omega_safe = self._perimeter_safe_cmd(v_des, omega)
        self.cmd_pub.publish(
            self._make_safe_cmd(v_safe, omega_safe))

    # -- Geometry / predicates -------------------------------------------------
    def _target_geometry(self, tx: float, ty: float):
        dx = tx - self.estimator.x
        dy = ty - self.estimator.y
        return math.hypot(dx, dy), normalize_angle(
            math.atan2(dy, dx) - self.estimator.theta)

    def _is_stopped(self) -> bool:
        return (abs(self.estimator.v) <= STOP_SPEED
                and abs(self.estimator.omega) <= 0.15)

    def _sensors_ready(self) -> bool:
        return (self.estimator.has_joint and self.estimator.has_imu
                and self._has_image and self._has_camera_info)

    # -- Perception step (10 Hz slice of the 20 Hz loop) ------------------------
    def _perception_step(self, now_s: float) -> None:
        if not (self._has_image and self._has_camera_info):
            return
        if self._latest_bgr is None:
            return
        detections = self.detector.detect(self._latest_bgr, self._fx,
                                           self.cam_z_param)
        worlds = []
        for d in detections:
            wx, wy = pixel_to_world(
                d["u"], d["v"], self._fx, self._fy, self._cx, self._cy,
                self.cam_x, self.cam_y, self.cam_z_param,
                self.swap_xy, self.sign_u, self.sign_v)
            if ARENA_MIN <= wx <= ARENA_MAX and ARENA_MIN <= wy <= ARENA_MAX:
                worlds.append((wx, wy))
            if self._tick % 200 == 0:
                self.get_logger().info(
                    f"Detected waypoint pixel=({d['u']:.0f},{d['v']:.0f}) "
                    f"world=({wx:.3f},{wy:.3f}) conf-r={d['radius_px']:.1f}px "
                    f"circ={d['circularity']:.2f}")
        confirmed_before = self.tracker.confirmed_count()
        confirmed_total, new_ids = self.tracker.update(worlds, now_s)
        if new_ids:
            for wid in new_ids:
                wp = self.tracker.get(wid)
                if wp is not None:
                    self.get_logger().info(
                        f"Detected waypoint #{wid} "
                        f"world=({wp['x']:.3f},{wp['y']:.3f}) "
                        f"confidence={wp['confidence']:.2f}")
            if confirmed_before < 5 <= confirmed_total:
                self.get_logger().info(
                    f"Initial targets acquired: {confirmed_total}")
            elif confirmed_before >= 5:
                for wid in new_ids:
                    wp = self.tracker.get(wid)
                    if wp is not None:
                        self.get_logger().info(
                            f"New waypoint detected dynamically: id={wid} "
                            f"({wp['x']:.3f},{wp['y']:.3f}) — "
                            f"{confirmed_total} targets available")
        self._last_confirmed = confirmed_total
        if self.debug_viz and self.debug_pub is not None:
            self._publish_debug_image(detections)

    def _publish_debug_image(self, detections) -> None:
        try:
            if self._latest_bgr is None:
                return
            dbg = self._latest_bgr.copy()
            for d in detections:
                cv2.circle(dbg, (int(d["u"]), int(d["v"])),
                           int(d["radius_px"]), (255, 0, 0), 2)
                cv2.circle(dbg, (int(d["u"]), int(d["v"])), 3, (0, 0, 255), -1)
            msg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
            self.debug_pub.publish(msg)
        except Exception:
            return

    # -- Route helper -----------------------------------------------------------
    def _replan_from_remaining(self) -> bool:
        unvisited = self.tracker.unvisited_confirmed()
        if not unvisited:
            return False
        start = (self.estimator.x, self.estimator.y)
        self._route = plan_route(start, unvisited)
        self._planned_count = self.tracker.confirmed_count()
        if self._route:
            names = []
            for wid in self._route:
                wp = self.tracker.get(wid)
                if wp is not None:
                    names.append(f"#{wid}({wp['x']:.2f},{wp['y']:.2f})")
            self.get_logger().info(
                "[PLAN] Route from "
                f"({start[0]:.2f},{start[1]:.2f}): "
                + " -> ".join(names) + " -> HOME")
            self._current_target = self._route[0]
            self._dwell_start = None
            return True
        return False

    # -- Main control loop ------------------------------------------------------
    def _control_loop(self) -> None:
        self._tick += 1
        now = self.get_clock().now()
        now_s = now.nanoseconds * 1e-9
        now_msg = now.to_msg()

        self.estimator.update(now_s)
        self._publish_odom_and_tf(now_msg)

        if self._tick % 60 == 0:
            self.get_logger().info(
                f"[EST] x={self.estimator.x:.3f} y={self.estimator.y:.3f} "
                f"th={self.estimator.theta:.3f} v={self.estimator.v:.3f} "
                f"w={self.estimator.omega:.3f} state={self._state} "
                f"confirmed={self.tracker.confirmed_count()}")

        # Perception runs at ~10 Hz; navigation decisions every tick.
        if self._tick % VISION_EVERY_N_TICKS == 0:
            try:
                self._perception_step(now_s)
            except Exception as exc:
                self.get_logger().warn(f"Perception step failed: {exc}")

        # Deferred dynamic replan: if new targets appeared while heading home, resume.
        if (self.tracker.confirmed_count() > self._planned_count
                and self._state == self.HOME):
            self.get_logger().info(
                f"7 targets available ({self.tracker.confirmed_count()} "
                "confirmed) — replanning remaining route.")
            self._home_dwell_start = None
            self._state = self.REPLAN
            self._publish_stop()
            return

        # ---------------- State machine ----------------
        if self._state == self.WAIT_FOR_SENSORS:
            if self._sensors_ready():
                self._state = self.DETECT_INITIAL_TARGETS
                self.get_logger().info(
                    "Sensors ready — scanning for initial green targets...")
            self._publish_stop()
            return

        if self._state == self.DETECT_INITIAL_TARGETS:
            if not self._sensors_ready():
                self._state = self.WAIT_FOR_SENSORS
                self._publish_stop()
                return
            if self.tracker.confirmed_count() >= 5:
                self._state = self.PLAN_ROUTE
            self._publish_stop()
            return

        if self._state == self.PLAN_ROUTE:
            if not self.tracker.has_unvisited():
                dist_home = math.hypot(self.estimator.x - HOME_X,
                                       self.estimator.y - HOME_Y)
                if dist_home <= HOME_TOLERANCE and self._is_stopped():
                    if self._home_dwell_start is None:
                        self._home_dwell_start = now_s
                    self._state = self.HOME_DWELL
                else:
                    self.get_logger().info("All targets visited — returning home.")
                    self._state = self.HOME
                self._publish_stop()
                return
            if self._replan_from_remaining():
                wp = self.tracker.get(self._current_target)
                if wp is not None:
                    self.get_logger().info(
                        f"Navigating to waypoint #{self._current_target} "
                        f"({wp['x']:.2f},{wp['y']:.2f})")
                self._state = self.ALIGN
            else:
                self._state = self.HOME
                self._publish_stop()
            return

        if self._state == self.REPLAN:
            if not self.tracker.has_unvisited():
                self._state = self.HOME
                self._publish_stop()
                return
            if self._replan_from_remaining():
                self._state = self.ALIGN
            else:
                self._state = self.HOME
            return

        if self._state in (self.ALIGN, self.DRIVE, self.BRAKE, self.DWELL):
            if (self._current_target is None
                    or self.tracker.get(self._current_target) is None
                    or self.tracker.get(self._current_target)["visited"]):
                self._dwell_start = None
                self._state = self.PLAN_ROUTE
                self._publish_stop()
                return
            wp = self.tracker.get(self._current_target)
            distance, he = self._target_geometry(wp["x"], wp["y"])

            if self._state == self.ALIGN:
                if distance <= STOP_APPROACH_DISTANCE:
                    self._state = self.BRAKE
                    self._publish_stop()
                    return
                if abs(he) <= ALIGN_THRESHOLD:
                    self._state = self.DRIVE
                    return
                self._publish_rotation(he)
                return

            if self._state == self.DRIVE:
                if distance <= BRAKE_DISTANCE:
                    self._state = self.BRAKE
                    self._publish_stop()
                    return
                if abs(he) > REALIGN_THRESHOLD:
                    self._state = self.ALIGN
                    self._publish_stop()
                    return
                self._publish_drive(distance, he)
                return

            if self._state == self.BRAKE:
                self._publish_stop()
                if distance <= WAYPOINT_TOLERANCE and self._is_stopped():
                    if self._dwell_start is None:
                        self._dwell_start = now_s
                        self.get_logger().info(
                            f"[DWELL] Started at #{self._current_target} "
                            f"({wp['x']:.2f},{wp['y']:.2f}) "
                            f"dist={distance:.3f}")
                    self._state = self.DWELL
                elif distance > WAYPOINT_TOLERANCE:
                    self._dwell_start = None
                    self._state = self.DRIVE
                return

            if self._state == self.DWELL:
                self._publish_stop()
                if distance > WAYPOINT_TOLERANCE or not self._is_stopped():
                    self._dwell_start = None
                    self._state = self.DRIVE
                    return
                elapsed = now_s - self._dwell_start if self._dwell_start else 0.0
                if elapsed >= self.dwell_time:
                    self.tracker.mark_visited(self._current_target)
                    self.get_logger().info(
                        f"[WP] Target #{self._current_target} collected "
                        f"({wp['x']:.2f},{wp['y']:.2f}) dwell={elapsed:.2f}s")
                    self.get_logger().info("Dwell complete.")
                    self._dwell_start = None
                    self._current_target = None
                    if self.tracker.has_unvisited():
                        if self.tracker.confirmed_count() > self._planned_count:
                            self.get_logger().info(
                                "Replanning with dynamically detected targets.")
                            self._state = self.REPLAN
                        else:
                            self._state = self.PLAN_ROUTE
                    else:
                        self._state = self.HOME
                        self.get_logger().info("Returning home.")
                return

        if self._state == self.HOME:
            distance, he = self._target_geometry(HOME_X, HOME_Y)
            if distance <= HOME_TOLERANCE:
                self._publish_stop()
                if self._is_stopped():
                    if self._home_dwell_start is None:
                        self._home_dwell_start = now_s
                        self.get_logger().info(
                            "[HOME] Entered home zone, dwell started.")
                    self._state = self.HOME_DWELL
                return
            if abs(he) <= ALIGN_THRESHOLD:
                self._publish_drive(distance, he)
            else:
                self._publish_rotation(he)
            return

        if self._state == self.HOME_DWELL:
            self._publish_stop()
            dist_home = math.hypot(self.estimator.x - HOME_X,
                                   self.estimator.y - HOME_Y)
            if dist_home > HOME_TOLERANCE or not self._is_stopped():
                self._home_dwell_start = None
                self._state = self.HOME
                return
            if self._home_dwell_start is None:
                self._home_dwell_start = now_s
                return
            if (now_s - self._home_dwell_start) >= self.dwell_time:
                self._state = self.DONE
                self.get_logger().info("Home dwell complete.")
                self.get_logger().info("Mission complete.")
            return

        if self._state == self.DONE:
            self._publish_stop()
            return

        self.get_logger().warn(f"Unknown state {self._state}; stopping.")
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
