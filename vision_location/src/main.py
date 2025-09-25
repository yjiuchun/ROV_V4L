#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import yaml
import os
import sys

# 添加scripts目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(os.path.dirname(current_dir), 'scripts')
sys.path.insert(0, scripts_dir)

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped

# 导入scripts目录下的模块
from pnp_solver import PnPSolver
from pose_tf import PoseTF
from detector import Detector
# from ekf import ExtendedKalmanFilter




if __name__ == '__main__':
    rospy.init_node('main', anonymous=True)
    pose_pub = rospy.Publisher('/vision/estimated_pose', PoseStamped, queue_size=1)
    ekf_pose_pub = rospy.Publisher('/vision/ekf_pose', PoseStamped, queue_size=1)

    # 创建CV桥接器
    detector_bridge = CvBridge()
    
    # 初始化各个模块
    detector = Detector()
    pnp_solver = PnPSolver()
    pose_tf = PoseTF()
    
    rospy.loginfo("main node started")

    def image_callback(data):
        """图像回调函数"""
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = detector_bridge.imgmsg_to_cv2(data, "bgr8")
            
            # 提取特征点
            feature_points = detector.extract_feature_points(cv_image)
            
            # 检查是否检测到足够的特征点
            if feature_points is not None and len(feature_points) >= 4:
                # 使用PnP求解器求解位姿
                rvec, tvec = pnp_solver.solve_pnp(feature_points)
                
                if rvec is not None and tvec is not None:
                    # 使用位姿转换实例转换位姿
                    pose = pose_tf.pose_process(rvec, tvec)
                    if pose is not None:
                        pose_pub.publish(pose)
                        rospy.loginfo(f"发布位姿: ({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, {pose.pose.position.z:.3f})")
            else:
                rospy.logwarn("未检测到足够的特征点")
                
        except Exception as e:
            rospy.logerr(f"处理图像时出错: {e}")
    
    # 订阅图像话题
    rospy.Subscriber('/camera/image_raw', Image, image_callback)
    
    # 可选：启动EKF系统
    # ekf = ExtendedKalmanFilter()
    
    rospy.loginfo("等待图像数据...")
    rospy.spin()