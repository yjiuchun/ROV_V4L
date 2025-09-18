#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class FeatureDetectionTest:
    def __init__(self):
        rospy.init_node('feature_detection_test', anonymous=True)
        
        # 创建CV桥接器
        self.bridge = CvBridge()
        
        # 订阅相机图像话题
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        
        # 颜色范围定义 (HSV)
        self.color_ranges = {
            'red': [
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                (np.array([170, 50, 50]), np.array([180, 255, 255]))
            ],
            'yellow': [(np.array([20, 50, 50]), np.array([30, 255, 255]))],
            'blue': [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))]
        }
        
        rospy.loginfo("特征点检测测试节点已启动")
        rospy.loginfo("订阅话题: /camera/image_raw")
        rospy.loginfo("按 'q' 键退出")
        
    def image_callback(self, data):
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # 提取特征点
            feature_points = self.extract_feature_points(cv_image)
            
            # 显示结果
            cv2.imshow("Feature Detection Test", cv_image)
            
            # 等待按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rospy.loginfo("用户按下 'q' 键，退出程序")
                rospy.signal_shutdown("用户退出")
                
        except Exception as e:
            rospy.logerr(f"处理图像时出错: {e}")
    
    def extract_feature_points(self, image):
        """提取红黄蓝绿四个圆的中点作为特征点"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        feature_points = []
        colors = ['red', 'yellow', 'blue', 'green']
        
        for color in colors:
            # 创建颜色掩码
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for lower, upper in self.color_ranges[color]:
                color_mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, color_mask)
            
            # 形态学操作去除噪声
            kernel = np.ones((5, 5), np.uint8)
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
                if area > 100:  # 最小面积阈值
                    # 计算轮廓的几何中心
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # 检查是否为圆形
                        if self.is_circular(largest_contour):
                            feature_points.append([cx, cy])
                            rospy.loginfo(f"检测到{color}圆中心: ({cx}, {cy})")
                            
                            # 在图像上标记检测到的点
                            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
                            cv2.putText(image, color, (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        rospy.loginfo(f"总共检测到 {len(feature_points)} 个特征点")
        return feature_points
    
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
        return circularity > 0.7
    
    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("程序被用户中断")
        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        test = FeatureDetectionTest()
        test.run()
    except rospy.ROSInterruptException:
        pass
