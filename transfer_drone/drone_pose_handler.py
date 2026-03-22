#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose


class DronePoseHandler(Node):

    def __init__(self):
        super().__init__('drone_pose_handler')
        
        # Subscriber for /drone_pose
        self.subscription = self.create_subscription(
            Pose,
            '/drone_pose',
            self.pose_callback,
            10
        )
        
        # Publisher for /t/topic
        self.publisher_ = self.create_publisher(
            Pose,
            '/t/topic',
            10
        )
        
        self.get_logger().info('DronePoseHandler node initialized')
        self.get_logger().info('Subscribing to: /drone_pose')
        self.get_logger().info('Publishing to: /t/topic')
        
        # Timer to publish at 10 Hz
        self.timer = self.create_timer(0.1, self.publish_pose)
        self.current_pose = Pose()
        
    def pose_callback(self, msg):
        """Callback for receiving drone pose data"""
        self.current_pose = msg
        self.get_logger().info(f'Received drone pose: pos=[{msg.position.x:.3f}, {msg.position.y:.3f}, {msg.position.z:.3f}]')
        
    def publish_pose(self):
        """Publish the current pose to /t/topic"""
        self.publisher_.publish(self.current_pose)
        self.get_logger().debug(f'Published pose to /t/topic: pos=[{self.current_pose.position.x:.3f}, {self.current_pose.position.y:.3f}, {self.current_pose.position.z:.3f}]')


def main(args=None):
    rclpy.init(args=args)
    node = DronePoseHandler()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
