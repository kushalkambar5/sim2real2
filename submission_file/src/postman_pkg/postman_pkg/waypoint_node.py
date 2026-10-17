import math
import random
import subprocess
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseArray, Pose, Twist
from visualization_msgs.msg import Marker, MarkerArray

# Arena constraints (world frame, meters)
FRAME = 'odom'
ARENA_MIN = 0.45
ARENA_MAX = 2.40
MIN_SEP = 0.50
MIN_START_DIST = 0.50
# Tier 3 Static Obstacle Bounding Boxes [x, y, yaw, length, width]
TIER3_OBSTACLES = [
    (1.1, 1.9, 0.785, 0.50, 0.06),
    (2.0, 1.1, 0.000, 0.50, 0.06),
    (0.9, 1.3, 1.571, 0.50, 0.06),
    (1.8, 2.2, 0.524, 0.50, 0.06),
]
MIN_OBS_CLEARANCE = 0.45  # Clearance from obstacle faces (meters)


def _dist_to_box(x: float, y: float, ox: float, oy: float, oyaw: float, length: float, width: float) -> float:
    dx = x - ox
    dy = y - oy
    local_x = dx * math.cos(-oyaw) - dy * math.sin(-oyaw)
    local_y = dx * math.sin(-oyaw) + dy * math.cos(-oyaw)
    half_l = length / 2.0
    half_w = width / 2.0
    closest_x = max(-half_l, min(local_x, half_l))
    closest_y = max(-half_w, min(local_y, half_w))
    return math.hypot(local_x - closest_x, local_y - closest_y)
DYN_WP_SDF_TEMPLATE = """\
<sdf version='1.9'>
  <model name='{name}'>
    <static>true</static>
    <pose>{x:.4f} {y:.4f} 0.01 0 0 0</pose>
    <link name='l'>
      <visual name='v'>
        <geometry><cylinder><radius>0.15</radius><length>0.02</length></cylinder></geometry>
        <material>
          <ambient>0.05 0.8 0.05 1</ambient>
          <diffuse>0.05 0.8 0.05 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


class WaypointNode(Node):
    def __init__(self):
        super().__init__('waypoint_node')
        self.declare_parameter('tier', 1)
        self.declare_parameter('seed', 42)
        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('delay_tier1', 5.0)
        self.declare_parameter('delay_tier2_3', 7.0)
        self.declare_parameter('tolerance', 0.15)

        self.tier = int(self.get_parameter('tier').value)
        self.seed = int(self.get_parameter('seed').value)
        self.tol = float(self.get_parameter('tolerance').value)

        # Deterministic PRNG seeded per (base_seed, tier)
        effective_seed = self.seed * 10 + self.tier
        self.rng = random.Random(effective_seed)

        # Procedurally sample 5 initial static waypoints (or load override if file provided)
        wp_path = self.get_parameter('waypoints_file').value
        if wp_path:
            with open(wp_path) as f:
                self.static = [tuple(p) for p in yaml.safe_load(f)['waypoints']]
        else:
            self.static = self._sample_points(count=5, existing=[])

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # In Tier 1: coordinates published. In Tier 2 and 3: suppressed to force CV usage.
        self.pub_static = self.create_publisher(PoseArray, '/waypoints', qos)
        self.pub_dyn = self.create_publisher(PoseArray, '/dynamic_waypoints', qos)
        self.pub_mark = self.create_publisher(MarkerArray, '/waypoint_markers', qos)

        # Organizer-only ground truth waypoint topic for evaluator script
        self.pub_eval_wps = self.create_publisher(PoseArray, '/evaluator/ground_truth_waypoints', qos)
        self._publish_evaluator_waypoints(self.static)

        self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)

        self.armed = True
        self.timer = None
        self.dynamic_spawned = False
        self.static_markers = MarkerArray()

        # One-shot 1.0s timer to ensure Gazebo is ready to receive model creation requests
        self._init_timer = self.create_timer(1.0, self._setup_initial_waypoints)

    def _setup_initial_waypoints(self) -> None:
        self._init_timer.cancel()
        for idx, (x, y) in enumerate(self.static):
            self._spawn_in_gazebo(f'wp_static_{idx}', x, y)

        self._publish_evaluator_waypoints(self.static)

        if self.tier == 1:
            self._publish_static_tier1()
            self.create_timer(1.0, self._periodic_publish_tier1)
            self.get_logger().info(
                f'[Tier 1] Procedural static waypoints spawned & published on /waypoints (seed={self.seed}): {self.static}'
            )
        else:
            self.get_logger().info(
                f'[Tier {self.tier}] Procedural static waypoints spawned visually in Gazebo (seed={self.seed}): {self.static}. '
                f'Waypoints must be extracted from /overhead_camera/image_raw.'
            )

    def _periodic_publish_tier1(self) -> None:
        """Periodic heartbeat for late-joining participant nodes with volatile QoS."""
        if not self.dynamic_spawned and self.tier == 1:
            self._publish_static_tier1()

    def _marker(self, mid: int, x: float, y: float, rgb: tuple) -> Marker:
        m = Marker()
        m.header.frame_id = FRAME
        m.ns = 'waypoints'
        m.id = mid
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.001
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = 2.0 * self.tol
        m.scale.z = 0.002
        m.color.r, m.color.g, m.color.b = rgb
        m.color.a = 0.9
        return m

    def _publish_static_tier1(self) -> None:
        arr = PoseArray()
        arr.header.frame_id = FRAME
        self.static_markers = MarkerArray()
        for i, (x, y) in enumerate(self.static):
            p = Pose()
            p.position.x, p.position.y = float(x), float(y)
            p.orientation.w = 1.0
            arr.poses.append(p)
            self.static_markers.markers.append(self._marker(i, x, y, (0.05, 0.8, 0.05)))
        self.pub_static.publish(arr)
        self.pub_mark.publish(self.static_markers)

    def _on_cmd_vel(self, msg: Twist) -> None:
        if not self.armed:
            return
        if abs(msg.linear.x) < 1e-4 and abs(msg.angular.z) < 1e-4:
            return

        self.armed = False
        delay = (
            self.get_parameter('delay_tier1').value
            if self.tier == 1
            else self.get_parameter('delay_tier2_3').value
        )
        self.get_logger().info(
            f'Robot movement detected (/cmd_vel). Arming dynamic waypoint spawn in {delay:.1f} s.'
        )
        self.timer = self.create_timer(delay, self._fire_dynamic)

    def _sample_points(self, count: int, existing: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """
        Samples waypoints maximizing spatial dispersion across the arena.
        Employs Mitchell's best-candidate strategy: evaluates a pool of candidates
        and selects the point maximizing the minimum Euclidean distance to all existing
        waypoints and the robot start zone (0, 0).
        """
        results: list[tuple[float, float]] = []
        for _ in range(count):
            best_cand = None
            best_dist = -1.0
            for _ in range(500):
                x = round(self.rng.uniform(ARENA_MIN, ARENA_MAX), 3)
                y = round(self.rng.uniform(ARENA_MIN, ARENA_MAX), 3)
                if math.hypot(x, y) < MIN_START_DIST:
                    continue
                if self.tier == 3 and any(
                    _dist_to_box(x, y, ox, oy, oyaw, l, w) < MIN_OBS_CLEARANCE
                    for ox, oy, oyaw, l, w in TIER3_OBSTACLES
                ):
                    continue
                ref_pts = existing + results + [(0.0, 0.0)]
                min_d = min(math.hypot(x - px, y - py) for px, py in ref_pts)
                if min_d < MIN_SEP:
                    continue
                if min_d > best_dist:
                    best_dist = min_d
                    best_cand = (x, y)

            if best_cand is None:
                # Fallback to relax MIN_SEP if arena is constrained by obstacles
                for _ in range(500):
                    x = round(self.rng.uniform(ARENA_MIN, ARENA_MAX), 3)
                    y = round(self.rng.uniform(ARENA_MIN, ARENA_MAX), 3)
                    if math.hypot(x, y) < MIN_START_DIST:
                        continue
                    if self.tier == 3 and any(
                        _dist_to_box(x, y, ox, oy, oyaw, l, w) < MIN_OBS_CLEARANCE
                        for ox, oy, oyaw, l, w in TIER3_OBSTACLES
                    ):
                        continue
                    ref_pts = existing + results + [(0.0, 0.0)]
                    min_d = min(math.hypot(x - px, y - py) for px, py in ref_pts)
                    if min_d > best_dist:
                        best_dist = min_d
                        best_cand = (x, y)

            if best_cand is not None:
                results.append(best_cand)
            else:
                self.get_logger().error(f'Failed to sample point {len(results)+1}/{count}')
        return results

    def _spawn_in_gazebo(self, name: str, x: float, y: float) -> bool:
        sdf = (
            f'<sdf version="1.9">'
            f'<model name="{name}"><static>true</static>'
            f'<link name="l">'
            f'<visual name="v">'
            f'<geometry><cylinder><radius>0.15</radius><length>0.02</length></cylinder></geometry>'
            f'<material><ambient>0.05 0.8 0.05 1</ambient><diffuse>0.05 0.8 0.05 1</diffuse></material>'
            f'</visual>'
            f'</link>'
            f'</model>'
            f'</sdf>'
        )
        cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-world', 'arena',
            '-name', name,
            '-x', f'{x:.4f}',
            '-y', f'{y:.4f}',
            '-z', '0.01',
            '-string', sdf,
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and ('Entity creation successful' in res.stderr or 'Entity creation successful' in res.stdout):
                self.get_logger().info(f'Spawned {name} in Gazebo at ({x:.2f}, {y:.2f}).')
                return True
            self.get_logger().warn(f'Failed spawning {name} in Gazebo: {res.stderr or res.stdout}')
        except Exception as exc:
            self.get_logger().warn(f'Subprocess error spawning {name}: {exc}')
        return False

    def _fire_dynamic(self) -> None:
        self.timer.cancel()
        if self.dynamic_spawned:
            return
        self.dynamic_spawned = True

        dyn_points = self._sample_points(count=2, existing=self.static)
        self.get_logger().info(
            f'Spawning 2 dynamic waypoints: {dyn_points[0]} and {dyn_points[1]}.'
        )

        # 1. Spawn visual cylinders in Gazebo for camera detection
        for idx, (x, y) in enumerate(dyn_points):
            self._spawn_in_gazebo(f'wp_dyn_{idx}', x, y)

        # 2. Tier 1: Publish coordinates to ROS topics
        if self.tier == 1:
            arr_dyn = PoseArray()
            arr_dyn.header.frame_id = FRAME
            for idx, (x, y) in enumerate(dyn_points):
                p = Pose()
                p.position.x, p.position.y = float(x), float(y)
                p.orientation.w = 1.0
                arr_dyn.poses.append(p)
                self.static_markers.markers.append(
                    self._marker(100 + idx, x, y, (1.0, 0.5, 0.0))
                )

            self.pub_dyn.publish(arr_dyn)
            self.pub_mark.publish(self.static_markers)

            # Also publish combined 7-waypoint list to /waypoints
            all_arr = PoseArray()
            all_arr.header.frame_id = FRAME
            for x, y in self.static + dyn_points:
                p = Pose()
                p.position.x, p.position.y = float(x), float(y)
                p.orientation.w = 1.0
                all_arr.poses.append(p)
            self.pub_static.publish(all_arr)

            self.get_logger().info(
                '[Tier 1] Published 2 dynamic waypoints to /dynamic_waypoints and updated /waypoints.'
            )
        else:
            self.get_logger().info(
                f'[Tier {self.tier}] Dynamic waypoints spawned in Gazebo visually. Coordinates withheld.'
            )

        # Update evaluator with all 7 waypoints
        self._publish_evaluator_waypoints(self.static + dyn_points)

    def _publish_evaluator_waypoints(self, points: list[tuple[float, float]]) -> None:
        arr = PoseArray()
        arr.header.frame_id = FRAME
        for x, y in points:
            p = Pose()
            p.position.x, p.position.y = float(x), float(y)
            p.orientation.w = 1.0
            arr.poses.append(p)
        self.pub_eval_wps.publish(arr)


def main():
    rclpy.init()
    rclpy.spin(WaypointNode())


if __name__ == '__main__':
    main()
