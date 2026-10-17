#!/usr/bin/env python3
"""
tier3_node.py
Tier 3 Starter template node for Sim2Real Competition (Obstacle Avoidance & Perception).
NOTE: Coordinate topics /waypoints and /dynamic_waypoints are suppressed by the environment.
You must detect green waypoint disks and magenta-topped obstacles from /overhead_camera/image_raw via OpenCV.
"""
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState, Imu, Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseArray


class SolutionNode(Node):
    def __init__(self):
        super().__init__('solution_node')

        # Robot Physical Dimensions (Kinematic Constants)
        self.wheel_radius = 0.0225       # meters
        self.wheel_separation = 0.159    # meters

        # Actuator Velocity Caps (Enforced by physics simulation)
        self.min_linear_speed = 0.0      # m/s (reverse prohibited)
        self.max_linear_speed = 0.3      # m/s
        self.max_angular_speed = 1.5     # rad/s

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # Sensor Subscriptions
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 20)
        self.create_subscription(Imu, '/imu', self._on_imu, 20)
        self.create_subscription(Image, '/overhead_camera/image_raw', self._on_image, 10)
        self.create_subscription(CameraInfo, '/overhead_camera/camera_info', self._on_camera_info, 10)

        # Waypoint Subscriptions (Suppressed in Tier 3: coordinates will not be received)
        # Waypoints and obstacles must be detected visually via /overhead_camera/image_raw
        qos_latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(PoseArray, '/waypoints', self._on_waypoints, qos_latched)
        self.create_subscription(PoseArray, '/dynamic_waypoints', self._on_dynamic_waypoints, qos_latched)

        # Control Loop Timer (20 Hz)
        self.create_timer(0.05, self._control_loop)
        self.get_logger().info('Solution node initialized.')

    def _on_joint_states(self, msg: JointState) -> None:
        pass

    def _on_imu(self, msg: Imu) -> None:
        pass

    def _on_image(self, msg: Image) -> None:
        pass

    def _on_camera_info(self, msg: CameraInfo) -> None:
        pass

    def _on_waypoints(self, msg: PoseArray) -> None:
        pass

    def _on_dynamic_waypoints(self, msg: PoseArray) -> None:
        pass

    def _control_loop(self) -> None:
        cmd = Twist()
        # TODO: Implement your controller here
        cmd.linear.x = 0.0   # m/s
        cmd.angular.z = 0.0  # rad/s

        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = SolutionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
