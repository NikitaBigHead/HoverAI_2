#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
import math


class DronePosePublisher(Node):

    def __init__(self):
        super().__init__('drone_pose_publisher')
        
        # Publisher for /drone_pose
        self.publisher_ = self.create_publisher(
            Pose,
            '/drone_pose',
            10
        )
        
        # Timer to publish at 10 Hz
        self.timer = self.create_timer(0.1, self.publish_pose)
        
        self.get_logger().info('Drone Pose Publisher Started')
        self.get_logger().info('Publishing simple static pose to: /drone_pose at 10 Hz')
        
    def publish_pose(self):
        """Publish simple static pose at 1 radian"""
        msg = Pose()
        
        # Simple static position
        msg.position.x = 1.0
        msg.position.y = 1.0
        msg.position.z = 1.0
        
        # Simple orientation at 1 radian (yaw)
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = math.sin(0.5)  # sin(1/2) for 1 radian yaw
        msg.orientation.w = math.cos(0.5)  # cos(1/2) for 1 radian yaw
        
        # Publish the message
        self.publisher_.publish(msg)
        
        # Log the pose data
        self.get_logger().info(
            f'Published Simple Pose - '
            f'Pos: [{msg.position.x:.1f}, {msg.position.y:.1f}, {msg.position.z:.1f}] '
            f'Quat: [{msg.orientation.x:.1f}, {msg.orientation.y:.1f}, {msg.orientation.z:.3f}, {msg.orientation.w:.3f}]'
        )
        
    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        # Roll (x), Pitch (y), Yaw (z)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        quat = Quaternion()
        quat.x = qx
        quat.y = qy
        quat.z = qz
        quat.w = qw
        
        return quat


def main(args=None):
    rclpy.init(args=args)
    node = DronePosePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
