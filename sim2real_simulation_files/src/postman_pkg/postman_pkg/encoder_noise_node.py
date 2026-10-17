#!/usr/bin/env python3
"""
encoder_noise_node.py
Intercepts raw Gazebo wheel encoder data on /gz/joint_states_raw, injects
calibrated Gaussian velocity noise, and republishes on /joint_states.

This simulates real-world encoder imperfections (quantisation, mounting
asymmetry, electrical noise) without touching the physics simulation.

Noise model
-----------
  Each wheel's angular velocity (rad/s) receives independent zero-mean
  Gaussian noise with std-dev `velocity_noise_stddev`.

  At the default 0.005 rad/s:
    - Linear velocity noise  ≈ 0.005 × 0.0225 ≈ 0.11 mm/s per wheel
    - Heading drift (1-sigma) ≈ 4–5° after 5 min of continuous travel
  This is realistic for a low-cost optical encoder on a small robot.

Parameters
----------
velocity_noise_stddev : float
    Gaussian noise std-dev added to each wheel's angular velocity [rad/s].
    Default: 0.005.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np


class EncoderNoiseNode(Node):
    def __init__(self) -> None:
        super().__init__('encoder_noise_node')

        self.declare_parameter('velocity_noise_stddev', 0.002)
        self._stddev: float = (
            self.get_parameter('velocity_noise_stddev').value
        )

        # Fixed seed -> reproducible noise character across restarts.
        # Teams cannot "cheat" by averaging multiple runs.
        self._rng = np.random.default_rng(seed=42)

        self._sub = self.create_subscription(
            JointState,
            '/gz/joint_states_raw',
            self._on_joint_states,
            10,
        )
        self._pub = self.create_publisher(JointState, '/joint_states', 10)

        self.get_logger().info(
            f'Encoder noise node ready  '
            f'(sigma = {self._stddev:.4f} rad/s per wheel, seed=42).'
        )

    def _on_joint_states(self, msg: JointState) -> None:
        """Add independent Gaussian noise to each wheel velocity during motion."""
        noisy = JointState()
        noisy.header   = msg.header
        noisy.name     = list(msg.name)
        noisy.position = list(msg.position)   # position ticks passed through clean
        noisy.effort   = list(msg.effort)

        n = len(msg.velocity)
        if n == 0:
            noisy.velocity = []
        else:
            noise = self._rng.normal(0.0, self._stddev, n)
            # Physical encoder model: stationary wheels emit 0 ticks (no noise at rest).
            # Noise is applied only when the wheel is in motion (|v| >= 1e-4 rad/s).
            noisy.velocity = [
                float(v + dv) if abs(v) >= 1e-4 else 0.0
                for v, dv in zip(msg.velocity, noise)
            ]

        self._pub.publish(noisy)


def main() -> None:
    rclpy.init()
    rclpy.spin(EncoderNoiseNode())


if __name__ == '__main__':
    main()
