#!/usr/bin/python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
import yaml
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pnp_solver import PnPSolver
from pose_tf import PoseTF




class Detector:
    def __init__(self,config_name='detector.yaml',folder='/home/yjc/Project/rov_ws/src/vision_location/',show_image=False):
        # 初始化默认参数
        self.color_ranges = {}
        self.min_area = 100
        self.circularity_threshold = 0.7
        self.kernel_size = 5
        self.bridge = CvBridge()
        self.show_image = show_image
        
        # 加载配置
        self.load_config(config_name,folder)


    def load_config(self,config_name,folder):
        """从配置文件加载参数"""
        config_path = os.path.join(folder, 'config/',config_name)
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # 加载颜色范围
            self.color_ranges = {}
            for color, ranges in config['color_ranges'].items():
                self.color_ranges[color] = []
                for range_config in ranges:
                    lower = np.array(range_config['lower'], dtype=np.uint8)
                    upper = np.array(range_config['upper'], dtype=np.uint8)
                    self.color_ranges[color].append((lower, upper))
            
            # 加载检测参数
            self.min_area = config['detection_params']['min_area']
            self.circularity_threshold = config['detection_params']['circularity_threshold']
            self.kernel_size = config['detection_params']['kernel_size']
            
            rospy.loginfo("配置文件加载成功")
            
        except Exception as e:
            rospy.logerr(f"加载配置文件失败: {e}")
            rospy.logerr("使用默认参数")
            self.load_default_config()
    
    def load_default_config(self):
        """加载默认配置"""
        # 默认颜色范围
        self.color_ranges = {
            'red': [
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                (np.array([170, 50, 50]), np.array([180, 255, 255]))
            ],
            'yellow': [(np.array([20, 50, 50]), np.array([30, 255, 255]))],
            'blue': [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))]
        }
        
        # 默认检测参数
        self.min_area = 100
        self.circularity_threshold = 0.7
        self.kernel_size = 5

    def extract_feature_points(self, image):
        """提取红黄蓝绿四个圆的中点作为特征点"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # image_copy = image.copy()

        feature_points = []
        colors = ['yellow', 'blue', 'green', 'red']
        
        for color in colors:
            # 创建颜色掩码
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for lower, upper in self.color_ranges[color]:
                color_mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, color_mask)
            
            # 形态学操作去除噪声
            kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 计算轮廓面积
                area = cv2.contourArea(largest_contour)
                
                # 过滤太小的轮廓
                if area > self.min_area:
                    # 计算轮廓的几何中心
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # 检查是否为圆形
                        if self.is_circular(largest_contour):
                            feature_points.append([cx, cy])
                            # rospy.loginfo(f"检测到{color}圆中心: ({cx}, {cy})")
                            
                            # 在图像上标记检测到的点
                            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
                            cv2.putText(image, color, (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if self.show_image:
            # 显示处理后的图像
            cv2.imshow("PnP Feature Detection", image)
            cv2.waitKey(1)
        return np.array(feature_points, dtype=np.float32), image

    def is_circular(self, contour):
        """检查轮廓是否为圆形"""
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return False
        
        # 计算圆形度 (4π*面积/周长²)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 圆形度接近1表示更接近圆形
        return 1


if __name__ == '__main__':
    rospy.init_node('detector_test_node', anonymous=True)
    
    # 创建检测器实例
    detector = Detector()
    # 创建PnP求解器实例
    pnp_solver = PnPSolver()
    # 创建位姿转换实例
    pose_tf = PoseTF()
    
    # 创建CV桥接器
    detector_bridge = CvBridge()

    def image_callback(data):
        """图像回调函数"""
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = detector_bridge.imgmsg_to_cv2(data, "bgr8")
            
            # 提取特征点
            feature_points = detector.extract_feature_points(cv_image)
            # 使用PnP求解器求解位姿
            rvec, tvec = pnp_solver.solve_pnp(feature_points)
            # 使用位姿转换实例转换位姿
            pose = pose_tf.pose_process(rvec, tvec)
            # if pose is not None:
            #     rospy.loginfo(f"PnP求解成功: 位置=({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, {pose.pose.position.z:.3f})")
            # else:
            #     rospy.logwarn("PnP求解失败")
                    
        except Exception as e:
            rospy.logerr(f"处理图像时出错: {e}")
    
    # 订阅图像话题
    rospy.Subscriber('/camera/image_raw', Image, image_callback)
    
    rospy.loginfo("检测器节点已启动")
    rospy.spin()