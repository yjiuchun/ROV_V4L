#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
基于照度的LED标识检测器

检测流程：
1. 图像灰度化
2. 二值化（基于阈值）
3. 定位LED标识（轮廓检测）
4. 根据LED标识掩码到原图，根据颜色分别获取四个区域中心点
"""

import rospy
import cv2
import numpy as np
import yaml
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class LightnessDetector:
    def __init__(self, config_name='colordetector.yaml', 
                 folder='/home/yjc/Project/rov_ws/src/vision4location/detection/config/',
                 show_image=False):
        """
        初始化基于照度的检测器
        
        参数:
            config_name: 配置文件名称
            folder: 配置文件文件夹路径
            show_image: 是否显示图像
        """
        # 初始化默认参数
        self.color_ranges = {}
        self.min_area = 100
        self.kernel_size = 5
        self.binary_threshold = 10  # 二值化阈值
        self.bridge = CvBridge()
        self.show_image = show_image
        
        # 加载配置
        self.load_config(config_name, folder)
    
    def load_config(self, config_name, folder):
        """从配置文件加载参数"""
        config_path = os.path.join(folder, config_name)
        print(f"加载配置文件: {config_path}")
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # 加载颜色范围（用于在原图上识别颜色）
            self.color_ranges = {}
            for color, ranges in config['color_ranges'].items():
                self.color_ranges[color] = []
                for range_config in ranges:
                    lower = np.array(range_config['lower'], dtype=np.uint8)
                    upper = np.array(range_config['upper'], dtype=np.uint8)
                    self.color_ranges[color].append((lower, upper))
            
            # 加载检测参数
            detection_params = config.get('detection_params', {})
            self.min_area = detection_params.get('min_area', 100)
            self.kernel_size = detection_params.get('kernel_size', 5)
            
            # 加载二值化阈值（如果配置中有）
            self.binary_threshold = detection_params.get('binary_threshold', 127)
            
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
        self.kernel_size = 5
        self.binary_threshold = 50
    
    def preprocess_image(self, image):
        """
        图像预处理：灰度化和二值化
        
        参数:
            image: 输入BGR图像
        
        返回:
            gray: 灰度图像
            binary: 二值化图像
        """
        # 1. 灰度化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. 二值化（使用固定阈值）
        # 如果LED是亮的，使用THRESH_BINARY；如果LED是暗的，使用THRESH_BINARY_INV
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # 可选：使用自适应阈值（如果光照不均匀）
        # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    #    cv2.THRESH_BINARY, 11, 2)
        
        return gray, binary
    
    def detect_led_regions(self, binary):
        """
        在二值化图像中检测LED区域
        
        参数:
            binary: 二值化图像
        
        返回:
            led_regions: LED区域列表，每个元素为(contour, bbox, center)
        """
        # 形态学操作去除噪声
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        led_regions = []
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤太小的轮廓
            if area > self.min_area:
                # 计算边界框
                x, y, w, h = cv2.boundingRect(contour)
                bbox = (x, y, w, h)
                
                # 计算轮廓中心
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center = (cx, cy)
                    
                    led_regions.append({
                        'contour': contour,
                        'bbox': bbox,
                        'center': center,
                        'area': area
                    })
        
        # 按面积排序，选择最大的4个区域（假设有4个LED）
        led_regions.sort(key=lambda x: x['area'], reverse=True)
        led_regions = led_regions[:4]
        
        return led_regions
    
    def identify_led_color(self, image, bbox):
        """
        在LED区域内识别颜色
        
        参数:
            image: 原始BGR图像
            bbox: 边界框 (x, y, w, h)
        
        返回:
            color: 识别到的颜色名称，如果无法识别则返回None
        """
        x, y, w, h = bbox
        
        # 提取LED区域
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return None
        
        # 转换为HSV颜色空间
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 对每种颜色进行匹配
        color_scores = {}
        for color, ranges in self.color_ranges.items():
            mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
            
            for lower, upper in ranges:
                color_mask = cv2.inRange(hsv_roi, lower, upper)
                mask = cv2.bitwise_or(mask, color_mask)
            
            # 计算匹配的像素数量
            match_count = cv2.countNonZero(mask)
            total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]
            
            if total_pixels > 0:
                color_scores[color] = match_count / total_pixels
        
        # 返回匹配度最高的颜色
        if color_scores:
            best_color = max(color_scores, key=color_scores.get)
            # 如果匹配度太低，返回None
            if color_scores[best_color] > 0.1:  # 至少10%的像素匹配
                return best_color
        
        return None
    
    def extract_feature_points(self, image):
        """
        提取红黄蓝绿四个LED的中心点作为特征点
        
        参数:
            image: 输入BGR图像
        
        返回:
            feature_points: 特征点数组，形状为(N, 2)，按颜色顺序：yellow, blue, green, red
            processed_image: 处理后的图像（用于可视化）
        """
        # 复制图像用于绘制
        processed_image = image.copy()
        
        # 1. 图像预处理：灰度化和二值化
        gray, binary = self.preprocess_image(image)
        
        # 2. 检测LED区域
        led_regions = self.detect_led_regions(binary)
        
        # 3. 识别每个LED的颜色并获取中心点
        color_to_point = {}  # 颜色 -> 中心点坐标
        
        for region in led_regions:
            bbox = region['bbox']
            center = region['center']
            
            # 识别LED颜色
            color = self.identify_led_color(image, bbox)
            
            if color and color not in color_to_point:
                color_to_point[color] = center
                
                # 在图像上标记
                cx, cy = center
                cv2.circle(processed_image, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(processed_image, color, (cx+10, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 4. 按指定顺序提取特征点：yellow, blue, green, red
        feature_points = []
        color_order = ['yellow', 'blue', 'green', 'red']
        
        for color in color_order:
            if color in color_to_point:
                feature_points.append(color_to_point[color])
            else:
                # 如果某个颜色未检测到，可以添加None或跳过
                rospy.logwarn(f"未检测到{color}颜色的LED")
        
        # 转换为numpy数组
        if feature_points:
            feature_points = np.array(feature_points, dtype=np.float32)
        else:
            feature_points = np.array([], dtype=np.float32).reshape(0, 2)
        
        # 可视化（可选）
        if self.show_image:
            # 显示灰度图
            cv2.imshow("Gray Image", gray)
            # 显示二值化图像
            cv2.imshow("Binary Image", binary)
            # 显示处理后的图像
            cv2.imshow("Lightness Detection", processed_image)
            cv2.waitKey(1)
        
        return feature_points, processed_image
    
    def extract_feature_points_with_mask(self, image):
        """
        使用掩码方法提取特征点（更精确的方法）
        
        参数:
            image: 输入BGR图像
        
        返回:
            feature_points: 特征点数组
            processed_image: 处理后的图像
        """
        processed_image = image.copy()
        
        # 1. 图像预处理
        gray, binary = self.preprocess_image(image)
        
        # 2. 检测LED区域
        led_regions = self.detect_led_regions(binary)
        
        # 3. 对每个LED区域，在原图上使用颜色掩码获取精确中心
        color_to_point = {}
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        for region in led_regions:
            bbox = region['bbox']
            x, y, w, h = bbox
            
            # 提取LED区域
            roi_hsv = hsv[y:y+h, x:x+w]
            roi_bgr = image[y:y+h, x:x+w]
            
            if roi_hsv.size == 0:
                continue
            
            # 对每种颜色进行匹配
            best_color = None
            best_match_ratio = 0
            
            for color, ranges in self.color_ranges.items():
                if color in color_to_point:  # 该颜色已经识别过了
                    continue
                
                mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
                
                for lower, upper in ranges:
                    color_mask = cv2.inRange(roi_hsv, lower, upper)
                    mask = cv2.bitwise_or(mask, color_mask)
                
                # 计算匹配度
                match_count = cv2.countNonZero(mask)
                total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]
                
                if total_pixels > 0:
                    match_ratio = match_count / total_pixels
                    if match_ratio > best_match_ratio and match_ratio > 0.1:
                        best_match_ratio = match_ratio
                        best_color = color
            
            # 如果识别到颜色，计算该颜色区域的精确中心
            if best_color:
                # 在整个图像上创建该颜色的掩码
                full_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                
                for lower, upper in self.color_ranges[best_color]:
                    color_mask = cv2.inRange(hsv, lower, upper)
                    full_mask = cv2.bitwise_or(full_mask, color_mask)
                
                # 形态学操作
                kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
                full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
                full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)
                
                # 在LED区域内查找该颜色的轮廓
                roi_mask = full_mask[y:y+h, x:x+w]
                contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # 找到最大的轮廓
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    
                    if area > self.min_area:
                        # 计算中心点（相对于整个图像）
                        M = cv2.moments(largest_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"]) + x
                            cy = int(M["m01"] / M["m00"]) + y
                            
                            color_to_point[best_color] = (cx, cy)
                            
                            # 在图像上标记
                            cv2.circle(processed_image, (cx, cy), 5, (0, 255, 0), -1)
                            cv2.putText(processed_image, best_color, (cx+10, cy), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 按指定顺序提取特征点
        feature_points = []
        color_order = ['yellow', 'blue', 'green', 'red']
        
        for color in color_order:
            if color in color_to_point:
                feature_points.append(color_to_point[color])
            else:
                rospy.logwarn(f"未检测到{color}颜色的LED")
        
        # 转换为numpy数组
        if feature_points:
            feature_points = np.array(feature_points, dtype=np.float32)
        else:
            feature_points = np.array([], dtype=np.float32).reshape(0, 2)
        
        # 可视化
        if self.show_image:
            cv2.imshow("Gray Image", gray)
            cv2.imshow("Binary Image", binary)
            cv2.imshow("Lightness Detection", processed_image)
            cv2.waitKey(1)
        
        return feature_points, processed_image


if __name__ == '__main__':
    rospy.init_node('lightness_detector_test_node', anonymous=True)
    
    # 创建检测器实例
    detector = LightnessDetector(show_image=True)
    
    detector_bridge = CvBridge()
    
    def image_callback(data):
        """图像回调函数"""
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = detector_bridge.imgmsg_to_cv2(data, "bgr8")
            
            # 提取特征点（使用掩码方法，更精确）
            feature_points, processed_image = detector.extract_feature_points_with_mask(cv_image)
            
            if len(feature_points) > 0:
                rospy.loginfo(f"检测到 {len(feature_points)} 个特征点")
                for i, point in enumerate(feature_points):
                    rospy.loginfo(f"  点{i+1}: ({point[0]:.1f}, {point[1]:.1f})")
            else:
                rospy.logwarn("未检测到特征点")
                    
        except Exception as e:
            rospy.logerr(f"处理图像时出错: {e}")
    
    # 订阅图像话题
    rospy.Subscriber('/camera/image_raw', Image, image_callback)
    
    rospy.loginfo("基于照度的检测器节点已启动")
    rospy.spin()

