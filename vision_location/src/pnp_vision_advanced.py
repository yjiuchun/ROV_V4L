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

class PnPVisionAdvanced:
    def __init__(self):
        """初始化高级PnP视觉定位类"""
        rospy.init_node('pnp_vision_advanced', anonymous=True)
        
        # 创建CV桥接器
        self.bridge = CvBridge()
        
        # 加载配置文件
        self.load_config()
        
        # 订阅相机图像话题
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        
        # 发布位置估计
        self.pose_pub = rospy.Publisher('/vision/estimated_pose', PoseStamped, queue_size=1)
        
        # TF相关
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        rospy.loginfo("高级PnP视觉定位系统已启动")
        rospy.loginfo("订阅话题: /camera/image_raw")
        rospy.loginfo("发布话题: /vision/estimated_pose")
        
    def load_config(self):
        """从配置文件加载参数"""
        config_path = os.path.join(os.path.dirname(__file__), '../config/color_detection.yaml')
        
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
    
    def image_callback(self, data):
        """图像回调函数"""
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # 提取特征点
            feature_points = self.extract_feature_points(cv_image)
            
            if len(feature_points) >= 4:
                # 执行PnP算法
                pose = self.solve_pnp(feature_points)
                if pose is not None:
                    self.publish_pose(pose, data.header)
                    
        except Exception as e:
            rospy.logerr(f"处理图像时出错: {e}")
    
    def extract_feature_points(self, image):
        """提取红黄蓝绿四个圆的中点作为特征点"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
                            rospy.loginfo(f"检测到{color}圆中心: ({cx}, {cy})")
                            
                            # 在图像上标记检测到的点
                            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
                            cv2.putText(image, color, (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 显示处理后的图像
        cv2.imshow("PnP Feature Detection", image)
        cv2.waitKey(1)
        
        return np.array(feature_points, dtype=np.float32)
    
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
        return circularity > self.circularity_threshold
    
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
            print(rvec,"======")
            if success:
                # 将旋转向量转换为旋转矩阵
                R, _ = cv2.Rodrigues(rvec)
                
                # 创建位姿消息
                pose = PoseStamped()
                pose.header.frame_id = "camera_link"
                pose.header.stamp = rospy.Time.now()
                
                # 设置位置
                pose.pose.position.x = tvec[0][0]
                pose.pose.position.y = tvec[1][0]
                pose.pose.position.z = tvec[2][0]
                
                # 设置姿态 (旋转矩阵转四元数)
                pose.pose.orientation = self.rotation_matrix_to_quaternion(R)
                
                rospy.loginfo(f"PnP求解成功: 位置=({tvec[0][0]:.3f}, {tvec[1][0]:.3f}, {tvec[2][0]:.3f})")
                return pose
            else:
                rospy.logwarn("PnP求解失败")
                return None
                
        except Exception as e:
            rospy.logerr(f"PnP求解出错: {e}")
            return None
    
    def rotation_matrix_to_quaternion(self, R):
        """将旋转矩阵转换为四元数"""
        from geometry_msgs.msg import Quaternion
        
        # 使用数学方法将旋转矩阵转换为四元数
        try:
            # 计算四元数
            trace = np.trace(R)
            
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
                qw = 0.25 * s
                qx = (R[2, 1] - R[1, 2]) / s
                qy = (R[0, 2] - R[2, 0]) / s
                qz = (R[1, 0] - R[0, 1]) / s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
            
            q = Quaternion()
            q.x = qx
            q.y = qy
            q.z = qz
            q.w = qw
            
            return q
            
        except Exception as e:
            rospy.logwarn(f"四元数转换出错: {e}，使用默认四元数")
            q = Quaternion()
            q.w = 1.0  # 默认单位四元数
            return q
    
    def publish_pose(self, pose, header):
        """发布位姿估计"""
        pose.header = header
        self.pose_pub.publish(pose)
    
    def run(self):
        """运行节点"""
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("程序被用户中断")
        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        pnp_vision = PnPVisionAdvanced()
        pnp_vision.run()
    except rospy.ROSInterruptException:
        pass
