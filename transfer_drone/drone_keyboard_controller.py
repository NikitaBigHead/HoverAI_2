#!/usr/bin/env python3
"""
Drone keyboard controller for Isaac Sim.

Controls:
  W / S      - Forward / Backward  (X axis)
  A / D      - Strafe Left / Right (Y axis)
  Z / X      - Altitude Up / Down  (Z axis)
  Q / E      - Yaw Left / Right

  Space      - Brake (zero velocity)
  R          - Reset to origin

Publishes:
  /drone_pose    (geometry_msgs/Pose)   - drone position & orientation
  /joint_command (sensor_msgs/JointState) - rotor velocities (visual)
"""

import math
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from pynput import keyboard


# ---------------------------------------------------------------------------
# Quadrotor rotor mixing (iris X-config)
#   joint0: front-right (CCW)   joint1: rear-left  (CCW)
#   joint2: front-left  (CW)    joint3: rear-right (CW)
#
# Mix columns: [throttle, pitch, roll, yaw]
#   pitch+ = nose up (backward)   roll+ = roll right
#   yaw+   = yaw right (CW body)
# ---------------------------------------------------------------------------
ROTOR_MIX = [
    [ 1, -1,  1, -1],  # joint0  front-right CCW
    [ 1,  1, -1, -1],  # joint1  rear-left   CCW
    [ 1, -1, -1,  1],  # joint2  front-left  CW
    [ 1,  1,  1,  1],  # joint3  rear-right  CW
]

BASE_ROTOR_SPEED = 50.0   # rad/s hover speed
ROTOR_DELTA      = 15.0   # rad/s authority per axis input

MOVE_STEP  = 0.05   # metres per tick
YAW_STEP   = 0.02   # radians per tick
RATE_HZ    = 50.0   # publish rate


class DroneKeyboardController(Node):

    def __init__(self):
        super().__init__('drone_keyboard_controller')

        self.pose_pub  = self.create_publisher(Pose,       '/drone_pose',    10)
        self.rotor_pub = self.create_publisher(JointState, '/joint_command', 10)

        # State
        self.x   = 0.0
        self.y   = 0.0
        self.z   = 1.5   # start 1.5 m above ground
        self.yaw = 0.0   # radians

        self._keys: set = set()
        self._lock = threading.Lock()

        self.timer = self.create_timer(1.0 / RATE_HZ, self._tick)

        # Start keyboard listener in background thread
        self._kb_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._kb_listener.start()

        self.get_logger().info('Drone keyboard controller started.')
        self.get_logger().info('  W/S=fwd/back  A/D=left/right  Z/X=up/down  Q/E=yaw  Space=brake  R=reset')

    # ------------------------------------------------------------------
    # Keyboard callbacks (run in listener thread)
    # ------------------------------------------------------------------

    def _on_press(self, key):
        try:
            with self._lock:
                self._keys.add(key.char.lower())
        except AttributeError:
            with self._lock:
                self._keys.add(key)

    def _on_release(self, key):
        try:
            with self._lock:
                self._keys.discard(key.char.lower())
        except AttributeError:
            with self._lock:
                self._keys.discard(key)

    def _pressed(self, *chars):
        with self._lock:
            return any(c in self._keys for c in chars)

    # ------------------------------------------------------------------
    # Control tick
    # ------------------------------------------------------------------

    def _tick(self):
        # --- handle reset / brake ---
        if self._pressed('r'):
            self.x = self.y = self.yaw = 0.0
            self.z = 1.5

        # --- translation in world frame rotated by current yaw ---
        fwd  = MOVE_STEP if self._pressed('w') else (-MOVE_STEP if self._pressed('s') else 0.0)
        side = MOVE_STEP if self._pressed('a') else (-MOVE_STEP if self._pressed('d') else 0.0)
        vert = MOVE_STEP if self._pressed('z') else (-MOVE_STEP if self._pressed('x') else 0.0)

        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        self.x   += cos_y * fwd - sin_y * side
        self.y   += sin_y * fwd + cos_y * side
        self.z   += vert
        self.z    = max(0.0, self.z)   # don't go underground

        # --- yaw ---
        if self._pressed('q'):
            self.yaw += YAW_STEP
        if self._pressed('e'):
            self.yaw -= YAW_STEP

        # --- keep yaw in [-π, π] ---
        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi

        self._publish_pose()
        self._publish_rotors(fwd, side, vert)

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def _publish_pose(self):
        msg = Pose()
        msg.position.x = self.x
        msg.position.y = self.y
        msg.position.z = self.z
        # yaw-only quaternion
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = math.sin(self.yaw * 0.5)
        msg.orientation.w = math.cos(self.yaw * 0.5)
        self.pose_pub.publish(msg)

    def _publish_rotors(self, fwd, side, vert):
        """
        Simple rotor mixing for visual spinning.
        throttle stays at hover speed; axes add differential.
        """
        throttle = 1.0
        pitch    = -fwd  / MOVE_STEP if MOVE_STEP else 0.0
        roll     = -side / MOVE_STEP if MOVE_STEP else 0.0
        yaw_in   = 0.0
        if self._pressed('q'):
            yaw_in = -1.0
        elif self._pressed('e'):
            yaw_in =  1.0

        velocities = []
        for mix in ROTOR_MIX:
            v = (BASE_ROTOR_SPEED
                 + mix[0] * throttle * 0          # throttle offset already in base
                 + mix[1] * pitch    * ROTOR_DELTA
                 + mix[2] * roll     * ROTOR_DELTA
                 + mix[3] * yaw_in   * ROTOR_DELTA)
            velocities.append(max(0.0, BASE_ROTOR_SPEED + v - BASE_ROTOR_SPEED))

        # Simpler: just spin all at base + scale by vertical input
        hover_boost = vert / MOVE_STEP * ROTOR_DELTA if MOVE_STEP else 0.0
        velocities = [
            max(0.0, BASE_ROTOR_SPEED
                + hover_boost
                + ROTOR_MIX[i][1] * pitch  * ROTOR_DELTA
                + ROTOR_MIX[i][2] * roll   * ROTOR_DELTA
                + ROTOR_MIX[i][3] * yaw_in * ROTOR_DELTA)
            for i in range(4)
        ]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = ['joint0', 'joint1', 'joint2', 'joint3']
        msg.velocity = velocities
        self.rotor_pub.publish(msg)

    # ------------------------------------------------------------------

    def destroy_node(self):
        self._kb_listener.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DroneKeyboardController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
