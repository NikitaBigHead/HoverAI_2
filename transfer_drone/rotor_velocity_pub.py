import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math


class RotorVelocityPublisher(Node):

    def __init__(self):
        super().__init__('rotor_velocity_publisher')

        self.publisher_ = self.create_publisher(
            JointState,
            '/joint_command',
            10
        )

        self.timer = self.create_timer(0.02, self.publish_velocity)  # 50 Hz

        self.joint_names = [
            'joint0',
            'joint1',
            'joint2',
            'joint3'
        ]

        self.rotor_velocity = 1.0  # rad/s
        self.angle_values = [0, math.pi/3, 2*math.pi/3, math.pi]  # radians (0, 60, 120, 180 degrees)
        self.current_angle_index = 0
        self.current_angle = 0.0  # Current rotation angle
        self.rotation_speed = 2.0  # rad/s for continuous rotation
        self.torque_value = 5.0  # N⋅m
        
        # Control mode timing
        self.start_time = self.get_clock().now()
        self.mode_duration = 5.0  # seconds per mode
        self.current_mode = 0  # 0: position, 1: velocity, 2: effort, 3: all

    def publish_velocity(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        # Calculate elapsed time and determine current mode
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.start_time).nanoseconds / 1e9
        self.current_mode = int(elapsed_time / self.mode_duration) % 4

        # Update continuous rotation angle
        dt = 0.02  # 50 Hz timer period
        self.current_angle += self.rotation_speed * dt
        
        # Keep angle in [-π, π] range to satisfy PhysX constraints
        if self.current_angle > math.pi:
            self.current_angle -= 2 * math.pi
        elif self.current_angle < -math.pi:
            self.current_angle += 2 * math.pi

        # Initialize empty arrays
        msg.position = []
        msg.velocity = []
        msg.effort = []

        # Mode 0: Position only (first 5 seconds)
        if self.current_mode == 0:
            base_angle = self.current_angle
            msg.position = [
                base_angle,
                base_angle + math.pi/2,
                base_angle + math.pi,
                base_angle + 3*math.pi/2
            ]
            # Ensure positions are within [-2π, 2π] range
            for i in range(len(msg.position)):
                if msg.position[i] > 2*math.pi:
                    msg.position[i] -= 2*math.pi
                elif msg.position[i] < -2*math.pi:
                    msg.position[i] += 2*math.pi

        # Mode 1: Velocity only (next 5 seconds)
        elif self.current_mode == 1:
            msg.velocity = [
                self.rotation_speed,
                self.rotation_speed,
                self.rotation_speed,
                self.rotation_speed
            ]

        # Mode 2: Effort only (next 5 seconds)
        elif self.current_mode == 2:
            msg.effort = [self.torque_value] * 4

        # Mode 3: All together (final 5 seconds)
        elif self.current_mode == 3:
            base_angle = self.current_angle
            msg.position = [
                base_angle,
                base_angle + math.pi/2,
                base_angle + math.pi,
                base_angle + 3*math.pi/2
            ]
            # Ensure positions are within [-2π, 2π] range
            for i in range(len(msg.position)):
                if msg.position[i] > 2*math.pi:
                    msg.position[i] -= 2*math.pi
                elif msg.position[i] < -2*math.pi:
                    msg.position[i] += 2*math.pi
            
            msg.velocity = [self.rotation_speed] * 4
            msg.effort = [self.torque_value] * 4

        # Log current mode
        mode_names = ["POSITION", "VELOCITY", "EFFORT", "ALL"]
        self.get_logger().info(f"Mode: {mode_names[self.current_mode]} - Time: {elapsed_time:.1f}s")

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RotorVelocityPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
