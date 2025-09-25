#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import yaml
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs

class PnPSolver:
    def __init__(self):

        # 加载配置文件
        self.load_config()

        rospy.loginfo("高级PnP视觉定位系统已启动")
        rospy.loginfo("订阅话题: /camera/image_raw")
        rospy.loginfo("发布话题: /vision/estimated_pose")
        
    def load_config(self):
        """从配置文件加载参数"""
        config_path = os.path.join(os.path.dirname(__file__), '../config/pnp.yaml')
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # 加载相机内参
            cam_matrix = config['camera_matrix']
            self.camera_matrix = np.array([
                [cam_matrix['fx'], 0, cam_matrix['cx']],
                [0, cam_matrix['fy'], cam_matrix['cy']],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # 加载畸变系数
            self.dist_coeffs = np.array(config['distortion_coeffs'], dtype=np.float32)
            
            # 加载3D特征点
            self.object_points = np.array(config['object_points'], dtype=np.float32)
            
            rospy.loginfo("配置文件加载成功")
            
        except Exception as e:
            rospy.logerr(f"加载配置文件失败: {e}")
            rospy.logerr("使用默认参数")
            self.load_default_config()
    
    def load_default_config(self):
        """加载默认配置"""
        
        # 默认相机内参
        self.camera_matrix = np.array([
            [800, 0, 400],
            [0, 800, 300],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 默认畸变系数
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # 默认3D特征点
        self.object_points = np.array([
            [-0.15, 0.15, 0.0],
            [0.15, 0.15, 0.0],
            [0.15, -0.15, 0.0],
            [-0.15, -0.15, 0.0]
        ], dtype=np.float32)

    
    def solve_pnp(self, image_points):
        """使用PnP算法求解相机位姿"""

        try:
            # 确保有足够的点
            if len(image_points) < 4:
                rospy.logwarn("特征点数量不足，无法进行PnP求解")
                return None
            
            # 使用对应的3D点
            object_points = self.object_points[:len(image_points)]
            
            # 执行PnP算法
            success, rvec, tvec = cv2.solvePnP(
                object_points, 
                image_points, 
                self.camera_matrix, 
                self.dist_coeffs
            )
            return rvec, tvec

                
        except Exception as e:
            rospy.logerr(f"PnP求解出错: {e}")
            return None
    


if __name__ == '__main__':
        
    pass
