#!/usr/bin/env python3
"""
evaluator.py
Automated Organizer Scoring & Verification Harness for Sim2Real Competition.

Monitors ground-truth simulation state to evaluate contestant runs:
1. Waypoints Visited: Checks if robot enters capture radius (0.25m grazing) AND
   dwells (stops with |v| <= 0.03 m/s) for at least 1.5 seconds.
2. Collisions: Detects contact with arena perimeter walls and Tier 3 obstacles.
3. Returned Home: Detects return to the 30x30 cm start zone at (0, 0) and stopping.
"""
import math
import sys
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Twist

# Arena & Geometry Constraints (meters)
ARENA_MIN_X, ARENA_MAX_X = -0.125, 2.875
ARENA_MIN_Y, ARENA_MAX_Y = -0.125, 2.875
WAYPOINT_TOLERANCE = 0.20    # Waypoint capture radius from center (meters)
DWELL_TIME_REQ = 2.00        # Minimum stop duration (seconds)
STOP_VEL_THRESH = 0.03       # Velocity threshold to consider "stopped" (m/s)
HOME_ZONE_THRESH = 0.20      # Distance from (0, 0) to consider "home"

# Tier 3 Static Obstacle Bounding Boxes [x, y, yaw, length, width]
OBSTACLES = [
    (1.1, 1.9, 0.785, 0.50, 0.06),
    (2.0, 1.1, 0.000, 0.50, 0.06),
    (0.9, 1.3, 1.571, 0.50, 0.06),
    (1.8, 2.2, 0.524, 0.50, 0.06),
]


class EvaluatorNode(Node):
    def __init__(self):
        super().__init__('evaluator_node')

        self.declare_parameter('tier', 1)
        # Parse tier from parameters or sys.argv (e.g. tier:=3 or --tier 3 or 3)
        self.tier = int(self.get_parameter('tier').value)
        args = sys.argv[1:]
        for i, arg in enumerate(args):
            if arg.startswith('tier:='):
                self.tier = int(arg.split(':=')[1])
            elif arg.startswith('--tier='):
                self.tier = int(arg.split('=')[1])
            elif arg in ('--tier', '-t') and i + 1 < len(args):
                self.tier = int(args[i + 1])
            elif arg in ('--tier1', '-tier1', 'tier1'):
                self.tier = 1
            elif arg in ('--tier2', '-tier2', 'tier2'):
                self.tier = 2
            elif arg in ('--tier3', '-tier3', 'tier3'):
                self.tier = 3
            elif arg in ('1', '2', '3'):
                self.tier = int(arg)

        # Obstacles are ONLY physically present in Tier 3 (arena_obstacles.sdf).
        # In Tiers 1 and 2, arena.sdf has ZERO interior obstacles.
        self.obstacles = OBSTACLES if self.tier == 3 else []

        self.waypoints = []              # [(x, y), ...]
        self.visited = set()             # Indices of visited waypoints
        self.dwell_start_time = {}       # {wp_idx: start_time}
        self.collisions = []             # List of collision log strings
        self.collision_detected = False

        self.total_path_length = 0.0
        self.last_pose = None
        self.run_started = False
        self.start_sim_time = None
        self.end_sim_time = None
        self.current_sim_time = None

        self.returned_home = False
        self.home_dwell_start = None

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(
            PoseArray, '/evaluator/ground_truth_waypoints', self._on_waypoints, qos
        )
        self.create_subscription(
            Odometry, '/evaluator/ground_truth_odom', self._on_odom, 20
        )
        self.create_subscription(
            Twist, '/cmd_vel', self._on_cmd_vel, 10
        )

        self.get_logger().info('====================================================')
        self.get_logger().info('Sim2Real Automated Evaluator Active (Simulation Round).')
        self.get_logger().info(f'Evaluator Mode: TIER {self.tier}')
        if self.tier == 3:
            self.get_logger().info('Obstacle Collision Detection: ENABLED (4 Tier 3 Obstacles)')
        else:
            self.get_logger().info('Obstacle Collision Detection: DISABLED (Tiers 1 & 2 have no obstacles)')
        self.get_logger().info('Timer Status: Waiting for solution node execution (/cmd_vel)...')
        self.get_logger().info('====================================================')

    def _on_cmd_vel(self, msg: Twist) -> None:
        """Trigger run start as soon as solution node begins issuing motion commands."""
        if not self.run_started:
            if abs(msg.linear.x) > 1e-4 or abs(msg.angular.z) > 1e-4:
                self.run_started = True
                self.start_sim_time = self.current_sim_time
                t_str = f'{self.start_sim_time:.2f}s' if self.start_sim_time else '0.00s'
                self.get_logger().info(
                    f'>>> [OFFICIAL RUN STARTED] Motion command received from solution node! '
                    f'Clock running from T_sim = {t_str}.'
                )

    def _on_waypoints(self, msg: PoseArray) -> None:
        pts = [(round(p.position.x, 3), round(p.position.y, 3)) for p in msg.poses]
        if len(pts) > len(self.waypoints):
            self.waypoints = pts
            self.get_logger().info(
                f'[EVALUATOR] Ground-truth targets registered: {len(self.waypoints)} waypoints.'
            )

    def _check_collision(self, x: float, y: float, yaw: float) -> None:
        # Physical robot bounding corners in base_link frame (meters)
        # Front bumper extends to +0.200m, rear to -0.010m, wheels to +/-0.0885m
        corners = [
            (-0.010, -0.0885),
            (-0.010,  0.0885),
            ( 0.200, -0.0885),
            ( 0.200,  0.0885),
        ]
        world_pts = [
            (
                x + cx * math.cos(yaw) - cy * math.sin(yaw),
                y + cx * math.sin(yaw) + cy * math.cos(yaw),
            )
            for cx, cy in corners
        ]

        # 1. Perimeter Wall Collision (applicable across all tiers)
        for wx, wy in world_pts:
            if (
                wx < ARENA_MIN_X
                or wx > ARENA_MAX_X
                or wy < ARENA_MIN_Y
                or wy > ARENA_MAX_Y
            ):
                if not self.collision_detected:
                    self.collision_detected = True
                    msg = f'Perimeter wall collision at ({wx:.2f}, {wy:.2f})'
                    self.collisions.append(msg)
                    self.get_logger().error(f'[COLLISION] {msg}')
                return

        # 2. Obstacle Collisions (only present in Tier 3)
        for idx, (ox, oy, oyaw, length, width) in enumerate(self.obstacles):
            half_l = length / 2.0
            half_w = width / 2.0
            for wx, wy in world_pts:
                dx = wx - ox
                dy = wy - oy
                local_x = dx * math.cos(-oyaw) - dy * math.sin(-oyaw)
                local_y = dx * math.sin(-oyaw) + dy * math.cos(-oyaw)

                if abs(local_x) <= half_l and abs(local_y) <= half_w:
                    if not self.collision_detected:
                        self.collision_detected = True
                        msg = f'Obstacle obs{idx} collision at ({wx:.2f}, {wy:.2f})'
                        self.collisions.append(msg)
                        self.get_logger().error(f'[COLLISION] {msg}')
                    return

        self.collision_detected = False

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        v = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        now_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        ori = msg.pose.pose.orientation
        siny = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cosy = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
        yaw = math.atan2(siny, cosy)

        self.current_sim_time = now_sec

        # Fallback movement trigger in case /cmd_vel arrived before evaluator subscribed
        if not self.run_started:
            if v > 0.02 or math.hypot(p.x, p.y) > 0.03:
                self.run_started = True
                self.start_sim_time = now_sec
                self.get_logger().info(
                    f'>>> [OFFICIAL RUN STARTED] Robot displacement detected! '
                    f'Clock running from T_sim = {now_sec:.2f}s.'
                )

        # Path Length accumulation (only during active run)
        if self.run_started and not self.returned_home:
            if self.last_pose is not None:
                step_dist = math.hypot(p.x - self.last_pose[0], p.y - self.last_pose[1])
                self.total_path_length += step_dist
            self.last_pose = (p.x, p.y)
        elif self.last_pose is None:
            self.last_pose = (p.x, p.y)

        # 1. Collision check (using oriented chassis envelope)
        self._check_collision(p.x, p.y, yaw)

        # 2. Waypoint Visit + Dwell Check
        for idx, (wx, wy) in enumerate(self.waypoints):
            if idx in self.visited:
                continue

            dist = math.hypot(p.x - wx, p.y - wy)
            if dist <= WAYPOINT_TOLERANCE:
                # Robot is within waypoint grazing footprint
                if v <= STOP_VEL_THRESH:
                    if idx not in self.dwell_start_time:
                        self.dwell_start_time[idx] = now_sec
                    else:
                        elapsed = now_sec - self.dwell_start_time[idx]
                        if elapsed >= DWELL_TIME_REQ:
                            self.visited.add(idx)
                            self.get_logger().info(
                                f'>>> [WAYPOINT {idx+1}/{len(self.waypoints)} COLLECTED] '
                                f'Target ({wx:.2f}, {wy:.2f}) reached! Dwell: {elapsed:.2f}s'
                            )
                else:
                    # Robot is moving too fast; reset dwell
                    self.dwell_start_time.pop(idx, None)
            else:
                self.dwell_start_time.pop(idx, None)

        # 3. Return to Home Check
        dist_home = math.hypot(p.x, p.y)
        if len(self.visited) > 0 and dist_home <= HOME_ZONE_THRESH:
            if v <= STOP_VEL_THRESH:
                if self.home_dwell_start is None:
                    self.home_dwell_start = now_sec
                elif (now_sec - self.home_dwell_start) >= DWELL_TIME_REQ:
                    if not self.returned_home:
                        self.returned_home = True
                        self.end_sim_time = now_sec
                        run_dur = (self.end_sim_time - self.start_sim_time) if self.start_sim_time else 0.0
                        self.get_logger().info(
                            f'>>> [HOME REACHED] Returned to origin (0,0) with error {dist_home*100:.1f} cm.'
                        )
                        self.get_logger().info(
                            f'>>> [RUN COMPLETED] Official run finished in {run_dur:.2f}s sim-time!'
                        )
            else:
                self.home_dwell_start = None
        else:
            self.home_dwell_start = None

    def print_report(self) -> None:
        if self.start_sim_time is not None:
            stop_time = self.end_sim_time if self.end_sim_time is not None else (self.current_sim_time or self.start_sim_time)
            elapsed = max(0.0, stop_time - self.start_sim_time)
        else:
            elapsed = 0.0
        total_wps = len(self.waypoints) if self.waypoints else 5
        visited_count = len(self.visited)
        final_err = math.hypot(self.last_pose[0], self.last_pose[1]) if self.last_pose else 0.0

        print('\n' + '=' * 60)
        print('     SIM2REAL OFFICIAL EVALUATION SCORECARD (SIM ROUND)')
        print('=' * 60)
        print(f'Total Waypoints Spawned : {total_wps}')
        print(f'Waypoints Collected     : {visited_count} / {total_wps}')
        print(f'Dwell Time Constraint   : >= 2.00s stop at each target')
        print(f'Collisions Detected     : {len(self.collisions)}')
        if self.collisions:
            for c in self.collisions[:5]:
                print(f'  - {c}')
        print(f'Returned to Start (0,0) : {"YES (PASS)" if self.returned_home else "NO (FAIL)"}')
        print(f'Final Position Error    : {final_err*100:.2f} cm')
        print(f'Total Odometry Distance : {self.total_path_length:.2f} m')
        print(f'Run Duration (Sim Time) : {elapsed:.2f} s')
        print('=' * 60 + '\n')


def main():
    rclpy.init()
    node = EvaluatorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.print_report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
