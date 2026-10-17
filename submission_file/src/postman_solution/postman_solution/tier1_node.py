#!/usr/bin/env python3
"""
tier1_node.py
Tier 1 — Dead Reckoning Navigation for Sim2Real Simulation Round.

Data flow:
    /joint_states + /imu -> STATE ESTIMATOR -> internal x,y,theta,v,omega
                                                 -> /odom + TF odom->base_link
                                                 -> controller

    /waypoints + /dynamic_waypoints -> WAYPOINT MGR (dedupe) -> ROUTE PLANNER
                                     (TSP over remaining + home) -> STATE MACHINE
                                     -> /cmd_vel (forward-only, clamped)

Integrity:
    - Uses only allowed contestant topics (joint states, imu, waypoints,
      dynamic_waypoints, cmd_vel, odom + TF). No ground-truth topics, no internal sim topics,
      no navigation stack, no hardcoded waypoints.
    - Forward-only motion: 0.0 <= v <= 0.30, |omega| <= 1.50.
"""

import math
import itertools
from typing import Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseArray, TransformStamped
from tf2_ros import TransformBroadcaster


# ---------------------------------------------------------------------------
# Robot / mission constants (organizer-defined — do not tune these)
# ---------------------------------------------------------------------------
WHEEL_RADIUS = 0.0225       # m
WHEEL_SEPARATION = 0.159    # m

MAX_LINEAR_SPEED = 0.30     # m/s
MIN_LINEAR_SPEED = 0.0      # m/s (reverse prohibited)
MAX_ANGULAR_SPEED = 1.50    # rad/s

WAYPOINT_TOLERANCE = 0.20   # m capture radius
DWELL_TIME = 2.30           # s continuous dwell (2.00 evaluator + 0.30 margin)
STOP_SPEED = 0.03           # m/s stopped threshold
HOME_TOLERANCE = 0.20       # m evaluator tolerance
HOME_APPROACH_TOLERANCE = 0.08  # m controller stop margin (inside evaluator limit for drift margin)
HOME_X = 0.0
HOME_Y = 0.0

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
K_ANGLE = 2.0               # proportional gain for in-place rotation
K_ANGLE_DRIVE = 1.5         # proportional gain while driving
K_DISTANCE = 0.8            # proportional gain distance -> speed

ALIGN_THRESHOLD = math.radians(8.0)     # aligned enough to drive
REALIGN_THRESHOLD = math.radians(15.0)  # too misaligned -> re-align

BRAKE_DISTANCE = 0.20          # DRIVE -> BRAKE (capture radius)
STOP_APPROACH_DISTANCE = 0.10  # ALIGN -> BRAKE (very close, stop)

ENCODER_YAW_WEIGHT = 0.95
IMU_YAW_WEIGHT = 0.05

MAX_REASONABLE_DT = 0.20    # s; larger dt is discarded (clock jump)
CONTROL_HZ = 20.0

# Conservative approach speed caps
SPEED_FAR = 0.30      # distance > 0.50
SPEED_MID = 0.15      # 0.25 < distance <= 0.50
SPEED_NEAR = 0.08     # 0.12 < distance <= 0.25
# distance <= 0.12 -> command 0, let BRAKE/DWELL take over

# Perimeter safety (arena collision bounds from evaluator.py: -0.125..2.875)
PERIMETER_MIN_X, PERIMETER_MAX_X = -0.125, 2.875
PERIMETER_MIN_Y, PERIMETER_MAX_Y = -0.125, 2.875
PERIMETER_SAFETY_DIST = 0.10  # m; steer away when estimated pose is this close to a wall


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


class SolutionNode(Node):
    """Tier-1 dead-reckoning + waypoint mission node."""

    # Mission states
    WAIT_FOR_WAYPOINTS = "WAIT_FOR_WAYPOINTS"
    PLAN_ROUTE = "PLAN_ROUTE"
    ALIGN = "ALIGN"
    DRIVE = "DRIVE"
    BRAKE = "BRAKE"
    DWELL = "DWELL"
    HOME = "HOME"
    HOME_DWELL = "HOME_DWELL"
    FINISHED = "FINISHED"

    def __init__(self):
        super().__init__('solution_node')

        # -- Kinematic params --
        self.wheel_radius = WHEEL_RADIUS
        self.wheel_separation = WHEEL_SEPARATION
        self.min_linear_speed = MIN_LINEAR_SPEED
        self.max_linear_speed = MAX_LINEAR_SPEED
        self.max_angular_speed = MAX_ANGULAR_SPEED

        # -- State estimate --
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.omega = 0.0

        # Latest sensor readings (callbacks only store; timer integrates)
        self._left_wheel_vel = 0.0
        self._right_wheel_vel = 0.0
        self._has_joint = False
        self._imu_omega_z = 0.0
        self._has_imu = False
        self._last_time_s = None  # float seconds (sim time)

        # -- Waypoint manager: key=(round(x,4),round(y,4)) -> dict --
        # dict value: {"x": float, "y": float, "visited": bool, "id": int}
        self._waypoints = {}
        self._next_wp_id = 0
        self._route = []              # ordered list of keys (remaining, excl. visited)
        self._current_target = None    # key of active target
        self._route_dirty = False      # set by waypoint callbacks

        # -- Dwell tracking (sim-time seconds as float) --
        self._dwell_start = None
        self._home_dwell_start = None

        # -- Mission state --
        self._state = self.WAIT_FOR_WAYPOINTS

        # -- Publishers --
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # -- Sensor subscriptions (standard QoS) --
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10)
        self.create_subscription(Imu, '/imu', self._on_imu, 10)

        # -- Waypoint subscriptions (transient-local latched) --
        qos_latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(PoseArray, '/waypoints', self._on_waypoints, qos_latched)
        self.create_subscription(
            PoseArray, '/dynamic_waypoints', self._on_dynamic_waypoints, qos_latched
        )

        # -- Control / estimator timer (20 Hz) --
        self._tick = 0
        self.create_timer(1.0 / CONTROL_HZ, self._control_loop)
        self.get_logger().info('Tier 1 solution node initialized (dead-reckoning).')

    # =====================================================================
    # Sensor callbacks — store only, never integrate here
    # =====================================================================
    def _on_joint_states(self, msg: JointState) -> None:
        try:
            left_idx = msg.name.index('left_wheel_joint')
            right_idx = msg.name.index('right_wheel_joint')
        except ValueError:
            return
        if len(msg.velocity) <= max(left_idx, right_idx):
            return
        self._left_wheel_vel = float(msg.velocity[left_idx])
        self._right_wheel_vel = float(msg.velocity[right_idx])
        self._has_joint = True

    def _on_imu(self, msg: Imu) -> None:
        self._imu_omega_z = float(msg.angular_velocity.z)
        self._has_imu = True

    # =====================================================================
    # Waypoint manager
    # =====================================================================
    def _merge_poses(self, poses, source: str) -> bool:
        """Merge incoming poses into known set. Returns True if any new added."""
        added_any = False
        for p in poses:
            wx = float(p.position.x)
            wy = float(p.position.y)
            key = (round(wx, 4), round(wy, 4))
            if key not in self._waypoints:
                self._waypoints[key] = {
                    "x": wx,
                    "y": wy,
                    "visited": False,
                    "id": self._next_wp_id,
                }
                self._next_wp_id += 1
                added_any = True
                self.get_logger().info(
                    f'[WP] Added waypoint id={self._waypoints[key]["id"]} '
                    f'x={wx:.3f} y={wy:.3f} ({source})'
                )
        return added_any

    def _on_waypoints(self, msg: PoseArray) -> None:
        n_before = len(self._waypoints)
        added = self._merge_poses(msg.poses, '/waypoints')
        if added:
            self.get_logger().info(
                f'[WP] /waypoints merged: {n_before} -> {len(self._waypoints)} unique'
            )
            if self._has_unvisited():
                self._route_dirty = True

    def _on_dynamic_waypoints(self, msg: PoseArray) -> None:
        if not msg.poses:
            return
        self.get_logger().info(
            f'[DYNAMIC] Received {len(msg.poses)} dynamic waypoints'
        )
        added = self._merge_poses(msg.poses, '/dynamic_waypoints')
        if added:
            self.get_logger().info(
                '[PLAN] New targets merged — will replan remaining route '
                '(completed targets never revisited).'
            )
            self._route_dirty = True

    def _has_unvisited(self) -> bool:
        return any(not wp["visited"] for wp in self._waypoints.values())

    def _unvisited_items(self):
        return [(k, wp) for k, wp in self._waypoints.items() if not wp["visited"]]

    # =====================================================================
    # Route planner: brute-force TSP, remaining only
    # =====================================================================
    def _plan_remaining_route(self):
        """Return ordered list of waypoint keys: current pose -> perm -> home."""
        unvisited = self._unvisited_items()
        if not unvisited:
            return []
        start = (self.x, self.y)
        home = (HOME_X, HOME_Y)
        best_perm = None
        best_cost = float('inf')
        for perm in itertools.permutations(unvisited):
            cost = euclid(start[0], start[1], perm[0][1]["x"], perm[0][1]["y"])
            for i in range(len(perm) - 1):
                cost += euclid(
                    perm[i][1]["x"], perm[i][1]["y"],
                    perm[i + 1][1]["x"], perm[i + 1][1]["y"],
                )
            cost += euclid(perm[-1][1]["x"], perm[-1][1]["y"], home[0], home[1])
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
        route = [k for k, _ in best_perm] if best_perm is not None else []
        desc = []
        for k in route:
            wp = self._waypoints[k]
            desc.append(f'id={wp["id"]}({wp["x"]:.2f},{wp["y"]:.2f})')
        self.get_logger().info(
            f'[PLAN] Route from ({start[0]:.2f},{start[1]:.2f}): '
            + (' -> '.join(desc) if desc else '(empty)')
            + f' -> HOME (cost={best_cost:.2f}m)'
        )
        return route

    # =====================================================================
    # Estimator
    # =====================================================================
    def _update_estimate(self, now_s: float) -> None:
        if self._has_joint:
            v_left = self._left_wheel_vel * self.wheel_radius
            v_right = self._right_wheel_vel * self.wheel_radius
            v_enc = (v_right + v_left) / 2.0
            omega_enc = (v_right - v_left) / self.wheel_separation
            if self._has_imu:
                fused = (
                    ENCODER_YAW_WEIGHT * omega_enc
                    + IMU_YAW_WEIGHT * self._imu_omega_z
                )
            else:
                fused = omega_enc
            self.v = v_enc
            self.omega = fused
        else:
            self.v = 0.0
            self.omega = 0.0

        if self._last_time_s is None:
            self._last_time_s = now_s
            return
        dt = now_s - self._last_time_s
        if dt <= 0.0 or dt > MAX_REASONABLE_DT:
            self._last_time_s = now_s
            return
        # Differential-drive integration
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta = normalize_angle(self.theta + self.omega * dt)
        self._last_time_s = now_s

    # =====================================================================
    # /odom + TF
    # =====================================================================
    def _publish_odom_and_tf(self, now_msg) -> None:
        odom = Odometry()
        odom.header.stamp = now_msg
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = math.sin(self.theta / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.theta / 2.0)
        odom.twist.twist.linear.x = self.v
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = self.omega
        self.odom_pub.publish(odom)

        t = TransformStamped()
        t.header.stamp = now_msg
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(self.theta / 2.0)
        t.transform.rotation.w = math.cos(self.theta / 2.0)
        self.tf_broadcaster.sendTransform(t)

    # =====================================================================
    # Command helpers
    # =====================================================================
    def _make_safe_cmd(self, v: float, omega: float) -> Twist:
        # Non-finite guard: NaN/Inf can never reach /cmd_vel.
        if not math.isfinite(v):
            v = 0.0
        if not math.isfinite(omega):
            omega = 0.0
        v = clamp(v, MIN_LINEAR_SPEED, MAX_LINEAR_SPEED)
        omega = clamp(omega, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        return cmd

    def _perimeter_safe_cmd(self, v: float, omega: float) -> Tuple[float, float]:
        """Override command if estimated pose is too close to a perimeter wall."""
        dist_west = self.x - PERIMETER_MIN_X
        dist_east = PERIMETER_MAX_X - self.x
        dist_south = self.y - PERIMETER_MIN_Y
        dist_north = PERIMETER_MAX_Y - self.y
        min_dist = min(dist_west, dist_east, dist_south, dist_north)

        if min_dist >= PERIMETER_SAFETY_DIST:
            return v, omega

        # Too close to a wall: stop forward motion and steer inward.
        v = 0.0
        if dist_west == min_dist:
            target_heading = 0.0
        elif dist_east == min_dist:
            target_heading = math.pi
        elif dist_south == min_dist:
            target_heading = math.pi / 2.0
        else:
            target_heading = -math.pi / 2.0

        heading_error = normalize_angle(target_heading - self.theta)
        omega = clamp(K_ANGLE * heading_error, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
        return v, omega

    def _publish_stop(self) -> None:
        self.cmd_pub.publish(self._make_safe_cmd(0.0, 0.0))

    def _publish_rotation(self, heading_error: float) -> None:
        omega = clamp(K_ANGLE * heading_error, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
        self.cmd_pub.publish(self._make_safe_cmd(0.0, omega))

    def _publish_drive(self, distance: float, heading_error: float) -> None:
        # Distance-based speed with conservative caps near target.
        v_des = min(MAX_LINEAR_SPEED, K_DISTANCE * distance)
        if distance <= 0.50:
            v_des = min(v_des, SPEED_MID)
        if distance <= 0.25:
            v_des = min(v_des, SPEED_NEAR)
        if distance <= 0.12:
            v_des = 0.0
        # Heading slowdown: avoid wide arcs.
        ah = abs(heading_error)
        if ah > math.radians(30.0):
            v_des = 0.0
        elif ah > math.radians(15.0):
            v_des *= 0.5
        omega = clamp(K_ANGLE_DRIVE * heading_error, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
        v_safe, omega_safe = self._perimeter_safe_cmd(v_des, omega)
        self.cmd_pub.publish(self._make_safe_cmd(v_safe, omega_safe))

    # =====================================================================
    # Geometry / dwell predicates
    # =====================================================================
    def _target_geometry(self, tx: float, ty: float):
        dx = tx - self.x
        dy = ty - self.y
        distance = math.hypot(dx, dy)
        heading_error = normalize_angle(math.atan2(dy, dx) - self.theta)
        return distance, heading_error

    def _is_stopped(self) -> bool:
        return abs(self.v) <= STOP_SPEED and abs(self.omega) <= 0.15

    # =====================================================================
    # Main control loop
    # =====================================================================
    def _control_loop(self) -> None:
        self._tick += 1
        now = self.get_clock().now()
        now_s = now.nanoseconds * 1e-9
        now_msg = now.to_msg()

        # 1. Update estimator and publish odom + TF
        self._update_estimate(now_s)
        self._publish_odom_and_tf(now_msg)

        if self._tick % 60 == 0:
            self.get_logger().info(
                f'[EST] x={self.x:.3f} y={self.y:.3f} th={self.theta:.3f} '
                f'v={self.v:.3f} w={self.omega:.3f} state={self._state}'
            )

        # Handle pending replan requests.
        # DWELL / HOME_DWELL are not preempted: finish the dwell first.
        if self._route_dirty and self._state not in (self.DWELL, self.HOME_DWELL):
            if self._has_unvisited():
                if self._state in (self.HOME, self.FINISHED):
                    self._home_dwell_start = None
                if self._state not in (self.WAIT_FOR_WAYPOINTS, self.PLAN_ROUTE):
                    self.get_logger().info('[PLAN] Replanning remaining route.')
                    self._state = self.PLAN_ROUTE
            self._route_dirty = False

        # ---------------- State machine ----------------
        if self._state == self.WAIT_FOR_WAYPOINTS:
            if self._waypoints:
                self._state = self.PLAN_ROUTE
            self._publish_stop()
            return

        if self._state == self.PLAN_ROUTE:
            if not self._has_unvisited():
                dist_home = math.hypot(self.x - HOME_X, self.y - HOME_Y)
                if dist_home <= HOME_TOLERANCE and self._is_stopped():
                    self._home_dwell_start = (
                        now_s if self._home_dwell_start is None else self._home_dwell_start
                    )
                    self._state = self.HOME_DWELL
                else:
                    self._state = self.HOME
                self._publish_stop()
                return
            self._route = self._plan_remaining_route()
            if not self._route:
                self._state = self.HOME
                self._publish_stop()
                return
            self._current_target = self._route[0]
            self._dwell_start = None
            self._state = self.ALIGN
            return

        if self._state in (self.ALIGN, self.DRIVE, self.BRAKE, self.DWELL):
            if (
                self._current_target is None
                or self._current_target not in self._waypoints
                or self._waypoints[self._current_target]["visited"]
            ):
                self._dwell_start = None
                self._state = self.PLAN_ROUTE
                self._publish_stop()
                return
            wp = self._waypoints[self._current_target]
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
                            f'[DWELL] Started at id={wp["id"]} '
                            f'({wp["x"]:.2f},{wp["y"]:.2f}) dist={distance:.3f}'
                        )
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
                if elapsed >= DWELL_TIME:
                    wp["visited"] = True
                    self.get_logger().info(
                        f'[WP] Target id={wp["id"]} collected '
                        f'({wp["x"]:.2f},{wp["y"]:.2f}) dwell={elapsed:.2f}s'
                    )
                    self._dwell_start = None
                    self._current_target = None
                    self._route_dirty = False
                    if self._has_unvisited():
                        self._state = self.PLAN_ROUTE
                    else:
                        self._state = self.HOME
                return

        if self._state == self.HOME:
            distance, he = self._target_geometry(HOME_X, HOME_Y)
            if distance <= HOME_APPROACH_TOLERANCE:
                self._publish_stop()
                if self._is_stopped():
                    if self._home_dwell_start is None:
                        self._home_dwell_start = now_s
                        self.get_logger().info('[HOME] Entered home zone, dwell started.')
                    self._state = self.HOME_DWELL
                return
            if abs(he) <= ALIGN_THRESHOLD:
                self._publish_drive(distance, he)
            else:
                self._publish_rotation(he)
            return

        if self._state == self.HOME_DWELL:
            self._publish_stop()
            dist_home = math.hypot(self.x - HOME_X, self.y - HOME_Y)
            if dist_home > HOME_TOLERANCE or not self._is_stopped():
                self._home_dwell_start = None
                self._state = self.HOME
                return
            if self._home_dwell_start is None:
                self._home_dwell_start = now_s
                return
            if (now_s - self._home_dwell_start) >= DWELL_TIME:
                self._state = self.FINISHED
                self.get_logger().info('[MISSION] COMPLETE — all waypoints + home dwell done.')
            return

        if self._state == self.FINISHED:
            self._publish_stop()
            return

        self.get_logger().warn(f'Unknown state {self._state}; stopping.')
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


if __name__ == '__main__':
    main()
