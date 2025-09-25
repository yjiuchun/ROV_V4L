#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EKF数据记录器
用于记录EKF系统的运行数据，便于后续分析
"""

import rospy
import numpy as np
import csv
import os
import time
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped
from std_msgs.msg import Header

class EKFDataLogger:
    def __init__(self):
        """初始化EKF数据记录器"""
        rospy.init_node('ekf_data_logger', anonymous=True)
        
        # 参数
        self.log_file = rospy.get_param('~log_file', 'ekf_data.csv')
        self.log_interval = rospy.get_param('~log_interval', 1.0)
        
        # 创建日志目录
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 数据存储
        self.data_buffer = []
        self.last_log_time = 0
        
        # 订阅话题
        self.setup_subscribers()
        
        # 初始化CSV文件
        self.init_csv_file()
        
        rospy.loginfo(f"EKF数据记录器已启动，日志文件: {self.log_file}")
    
    def setup_subscribers(self):
        """设置话题订阅"""
        # 订阅PnP原始结果
        self.pnp_sub = rospy.Subscriber(
            '/vision/estimated_pose', 
            PoseStamped, 
            self.pnp_callback
        )
        
        # 订阅EKF优化结果
        self.ekf_sub = rospy.Subscriber(
            '/vision/ekf_pose', 
            PoseStamped, 
            self.ekf_callback
        )
        
        # 订阅EKF协方差信息
        self.covariance_sub = rospy.Subscriber(
            '/vision/ekf_pose_covariance', 
            PoseWithCovarianceStamped, 
            self.covariance_callback
        )
        
        # 订阅速度估计
        self.velocity_sub = rospy.Subscriber(
            '/vision/estimated_velocity', 
            TwistStamped, 
            self.velocity_callback
        )
    
    def init_csv_file(self):
        """初始化CSV文件"""
        headers = [
            'timestamp',
            'pnp_x', 'pnp_y', 'pnp_z',
            'pnp_qx', 'pnp_qy', 'pnp_qz', 'pnp_qw',
            'ekf_x', 'ekf_y', 'ekf_z',
            'ekf_qx', 'ekf_qy', 'ekf_qz', 'ekf_qw',
            'ekf_vx', 'ekf_vy', 'ekf_vz',
            'pos_uncertainty',
            'position_error_x', 'position_error_y', 'position_error_z',
            'orientation_error'
        ]
        
        with open(self.log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
    
    def pnp_callback(self, msg):
        """PnP位姿回调"""
        current_time = time.time()
        
        # 存储PnP数据
        if not hasattr(self, 'pnp_data'):
            self.pnp_data = {}
        
        self.pnp_data = {
            'timestamp': current_time,
            'position': [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            'orientation': [msg.pose.orientation.x, msg.pose.orientation.y, 
                           msg.pose.orientation.z, msg.pose.orientation.w]
        }
        
        # 检查是否需要记录数据
        self.check_and_log()
    
    def ekf_callback(self, msg):
        """EKF位姿回调"""
        if not hasattr(self, 'ekf_data'):
            self.ekf_data = {}
        
        self.ekf_data.update({
            'position': [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            'orientation': [msg.pose.orientation.x, msg.pose.orientation.y, 
                           msg.pose.orientation.z, msg.pose.orientation.w]
        })
    
    def covariance_callback(self, msg):
        """协方差回调"""
        if not hasattr(self, 'covariance_data'):
            self.covariance_data = {}
        
        # 计算位置不确定性
        cov = msg.pose.covariance
        pos_uncertainty = np.sqrt(cov[0] + cov[7] + cov[14])
        
        self.covariance_data = {
            'pos_uncertainty': pos_uncertainty
        }
    
    def velocity_callback(self, msg):
        """速度估计回调"""
        if not hasattr(self, 'velocity_data'):
            self.velocity_data = {}
        
        self.velocity_data = {
            'velocity': [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
        }
    
    def check_and_log(self):
        """检查并记录数据"""
        current_time = time.time()
        
        # 检查是否有完整的数据
        if not (hasattr(self, 'pnp_data') and hasattr(self, 'ekf_data')):
            return
        
        # 检查记录间隔
        if current_time - self.last_log_time < self.log_interval:
            return
        
        # 记录数据
        self.log_data()
        self.last_log_time = current_time
    
    def log_data(self):
        """记录数据到CSV文件"""
        try:
            # 计算误差
            position_error = [0, 0, 0]
            orientation_error = 0
            
            if hasattr(self, 'pnp_data') and hasattr(self, 'ekf_data'):
                pnp_pos = self.pnp_data['position']
                ekf_pos = self.ekf_data['position']
                position_error = [abs(p - e) for p, e in zip(pnp_pos, ekf_pos)]
                
                # 计算四元数误差（简化处理）
                pnp_ori = self.pnp_data['orientation']
                ekf_ori = self.ekf_data['orientation']
                orientation_error = np.linalg.norm(np.array(pnp_ori) - np.array(ekf_ori))
            
            # 准备数据行
            row = [
                self.pnp_data.get('timestamp', 0),
                # PnP数据
                self.pnp_data.get('position', [0, 0, 0])[0],
                self.pnp_data.get('position', [0, 0, 0])[1],
                self.pnp_data.get('position', [0, 0, 0])[2],
                self.pnp_data.get('orientation', [0, 0, 0, 1])[0],
                self.pnp_data.get('orientation', [0, 0, 0, 1])[1],
                self.pnp_data.get('orientation', [0, 0, 0, 1])[2],
                self.pnp_data.get('orientation', [0, 0, 0, 1])[3],
                # EKF数据
                self.ekf_data.get('position', [0, 0, 0])[0],
                self.ekf_data.get('position', [0, 0, 0])[1],
                self.ekf_data.get('position', [0, 0, 0])[2],
                self.ekf_data.get('orientation', [0, 0, 0, 1])[0],
                self.ekf_data.get('orientation', [0, 0, 0, 1])[1],
                self.ekf_data.get('orientation', [0, 0, 0, 1])[2],
                self.ekf_data.get('orientation', [0, 0, 0, 1])[3],
                # 速度数据
                self.velocity_data.get('velocity', [0, 0, 0])[0],
                self.velocity_data.get('velocity', [0, 0, 0])[1],
                self.velocity_data.get('velocity', [0, 0, 0])[2],
                # 不确定性
                self.covariance_data.get('pos_uncertainty', 0),
                # 误差
                position_error[0],
                position_error[1],
                position_error[2],
                orientation_error
            ]
            
            # 写入CSV文件
            with open(self.log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
            
            rospy.logdebug(f"数据已记录: 时间={row[0]:.3f}, 位置误差={position_error}")
            
        except Exception as e:
            rospy.logerr(f"记录数据时出错: {e}")
    
    def run(self):
        """主运行循环"""
        rate = rospy.Rate(10)  # 10Hz
        
        rospy.loginfo("EKF数据记录器开始运行...")
        
        try:
            while not rospy.is_shutdown():
                rate.sleep()
                
        except KeyboardInterrupt:
            rospy.loginfo("数据记录器已停止")
            rospy.loginfo(f"数据已保存到: {self.log_file}")

def main():
    """主函数"""
    try:
        logger = EKFDataLogger()
        logger.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("程序被中断")
    except Exception as e:
        rospy.logerr(f"程序运行出错: {e}")

if __name__ == '__main__':
    main()

