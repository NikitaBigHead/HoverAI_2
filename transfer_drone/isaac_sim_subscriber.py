#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry


class IsaacSimSubscriber(Node):

    def __init__(self):
        super().__init__('isaac_sim_subscriber')
        
        # Common Isaac Sim geometry topics
        self.pose_sub = self.create_subscription(
            Pose,
            '/drone_pose',
            self.pose_callback,
            10
        )
        
        self.twist_sub = self.create_subscription(
            Twist,
            '/drone_twist',
            self.twist_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/drone_imu',
            self.imu_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/drone_odometry',
            self.odometry_callback,
            10
        )
        
        self.get_logger().info('Isaac Sim Subscriber Node Started')
        self.get_logger().info('Listening for geometry_msgs from Isaac Sim...')
        
        # Store latest data
        self.latest_pose = Pose()
        self.latest_twist = Twist()
        self.latest_imu = Imu()
        self.latest_odom = Odometry()
        
    def pose_callback(self, msg):
        """Handle Pose messages from Isaac Sim"""
        self.latest_pose = msg
        self.get_logger().info(
            f'POSE - Pos: [{msg.position.x:.3f}, {msg.position.y:.3f}, {msg.position.z:.3f}] '
            f'Quat: [{msg.orientation.x:.3f}, {msg.orientation.y:.3f}, {msg.orientation.z:.3f}, {msg.orientation.w:.3f}]'
        )
        
    def twist_callback(self, msg):
        """Handle Twist messages from Isaac Sim"""
        self.latest_twist = msg
        self.get_logger().info(
            f'TWIST - Linear: [{msg.linear.x:.3f}, {msg.linear.y:.3f}, {msg.linear.z:.3f}] '
            f'Angular: [{msg.angular.x:.3f}, {msg.angular.y:.3f}, {msg.angular.z:.3f}]'
        )
        
    def imu_callback(self, msg):
        """Handle IMU messages from Isaac Sim"""
        self.latest_imu = msg
        self.get_logger().info(
            f'IMU - Accel: [{msg.linear_acceleration.x:.3f}, {msg.linear_acceleration.y:.3f}, {msg.linear_acceleration.z:.3f}] '
            f'Gyro: [{msg.angular_velocity.x:.3f}, {msg.angular_velocity.y:.3f}, {msg.angular_velocity.z:.3f}]'
        )
        
    def odometry_callback(self, msg):
        """Handle Odometry messages from Isaac Sim"""
        self.latest_odom = msg
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        vel = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        
        self.get_logger().info(
            f'ODOM - Pos: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}] '
            f'Vel: [{vel.x:.3f}, {vel.y:.3f}, {vel.z:.3f}]'
        )
        
    def get_latest_data(self):
        """Return all latest sensor data"""
        return {
            'pose': self.latest_pose,
            'twist': self.latest_twist,
            'imu': self.latest_imu,
            'odometry': self.latest_odom
        }


def main(args=None):
    rclpy.init(args=args)
    node = IsaacSimSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
