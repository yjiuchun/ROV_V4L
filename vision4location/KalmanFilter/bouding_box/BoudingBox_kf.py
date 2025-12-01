#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
边界框卡尔曼滤波器 (Bounding Box Kalman Filter)

功能描述:
    使用卡尔曼滤波器对目标检测中的边界框进行平滑和预测，减少检测噪声，
    提高目标跟踪的稳定性和准确性。

流程：
    检测框==》卡尔曼预测==》预测框与检测框匹配==》卡尔曼更新==》跟踪框

变量：
    测量值:Z=[cx,cy,w,h]'T,cx,cy为检测框中心点坐标，w,h为检测框宽度和高度
    状态值:X=[cx,cy,w,h,vx,vy,vw,vh]T,cx,cy为检测框中心点坐标，w,h为检测框宽度和高度，vx,vy为检测框中心点速度，vw,vh为检测框宽度和高度速度
    观测矩阵H:Z=HX
        H=[1 0 0 0 0 0 0 0
           0 1 0 0 0 0 0 0
           0 0 1 0 0 0 0 0
           0 0 0 1 0 0 0 0]
    状态转移矩阵：假设变量符合线性变化规律，满足新状态量=旧状态量+变化量；变化量不变
        X(k)=FX(k-1)
        F=[1 0 0 0 1 0 0 0
           0 1 0 0 0 1 0 0
           0 0 1 0 0 0 1 0
           0 0 0 1 0 0 0 1
           0 0 0 0 1 0 0 0
           0 0 0 0 0 1 0 0
           0 0 0 0 0 0 1 0
           0 0 0 0 0 0 0 1
        ]
    过程噪声协方差矩阵Q:表示过程的噪声叠加，Q越大，噪声越大，预测越不准确
        Q=[q1 0 0 0 0 0 0 0
           0 q2 0 0 0 0 0 0
           0 0 q3 0 0 0 0 0
           0 0 0 q4 0 0 0 0
           0 0 0 0 q5 0 0 0
           0 0 0 0 0 q6 0 0
           0 0 0 0 0 0 q7 0
           0 0 0 0 0 0 0 q8]
    Q矩阵动态更新。
    观测噪声协方差矩阵R:表示观测的噪声叠加，R越大，噪声越大，更新越不准确
        R=[r1 0 0 0
           0 r2 0 0 
           0 0 r3 0
           0 0 0 r4]
    误差协方差矩阵P:表示预测的误差叠加，P越大，误差越大，预测越不准确
        P=[p1 0 0 0 0 0 0 0
           0 p2 0 0 0 0 0 0
           0 0 p3 0 0 0 0 0
           0 0 0 p4 0 0 0 0
           0 0 0 0 p5 0 0 0
           0 0 0 0 0 p6 0 0
           0 0 0 0 0 0 p7 0
           0 0 0 0 0 0 0 p8]
    P矩阵动态更新。
    卡尔曼预测：
        X(k)=FX(k-1)
        P(k)=FP(k-1)F^T+Q
    卡尔曼更新：
        K=P(k)H^T(HP(k)H^T+R)^-1
        X(k)=X(k)+K(Z(k)-HX(k))
        P(k)=(I-KH)P(k)
作者:Yangjiuchun
时间:2025-11-21
"""
import cv2

import numpy as np
import sys
import os
vision4location_detection_dir = '/home/yjc/Project/rov_ws/src/vision4location/detection'
if vision4location_detection_dir not in sys.path:
    sys.path.insert(0, vision4location_detection_dir)

class BoudingBox_kf:
    def __init__(self, dt=0.3):
        """
        初始化边界框卡尔曼滤波器
        
        参数:
            dt: 时间步长，默认为1.0
        """
        # 状态向量: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.zeros(8)
        self.x_pred = np.zeros(8)
        
        # 误差协方差矩阵 P (8x8)
        self.P = np.eye(8) * 0.1
        self.P_pred = np.eye(8) * 0.0
        
        # 过程噪声协方差矩阵 Q (8x8) - 对角矩阵
        self.Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1])
        
        # 观测噪声协方差矩阵 R (4x4) - 对角矩阵
        self.R = np.diag([0.1, 0.1, 0.1, 0.1]) * 1
        
        # 观测矩阵 H (4x8): Z = H * X
        # H矩阵从8维状态向量中提取4维观测值 [cx, cy, w, h]
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1.0  # cx
        self.H[1, 1] = 1.0  # cy
        self.H[2, 2] = 1.0  # w
        self.H[3, 3] = 1.0  # h
        
        # 状态转移矩阵 F (8x8): X(k) = F * X(k-1)
        # 位置 = 位置 + 速度 * dt
        # 速度保持不变
        self.F = np.eye(8)
        self.F[0, 4] = dt  # cx = cx + vx * dt
        self.F[1, 5] = dt  # cy = cy + vy * dt
        self.F[2, 6] = dt  # w = w + vw * dt
        self.F[3, 7] = dt  # h = h + vh * dt
        
        self.dt = dt
        self.initialized = False
    def predict(self):
        """卡尔曼预测"""
        self.x_pred = self.F @ self.x
        self.P_pred = self.F @ self.P @ self.F.T + self.Q
    def update(self, z):
        """卡尔曼更新"""
        self.K = self.P_pred @ self.H.T @ np.linalg.inv(self.H @ self.P_pred @ self.H.T + self.R)
        self.x = self.x_pred + self.K @ (z - self.H @ self.x_pred)
        self.P = (np.eye(8) - self.K @ self.H) @ self.P_pred
    
    def get_bbox(self):
        """
        获取当前边界框参数
        
        返回:
            cx, cy, w, h: 中心点坐标和宽高
        """
        cx = self.x[0]
        cy = self.x[1]
        w = self.x[2]
        h = self.x[3]
        return cx, cy, w, h
    
    def get_bbox_int(self):
        """
        获取当前边界框参数（整数格式，用于绘制和裁剪）
        
        返回:
            x1, y1, x2, y2: 左上角和右下角坐标
        """
        cx, cy, w, h = self.get_bbox()
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return x1, y1, x2, y2
    
    def crop_image(self, image):
        """
        根据卡尔曼滤波得到的矩形框裁剪图片
        
        参数:
            image: 输入图像 (numpy array)
        
        返回:
            cropped_image: 裁剪后的图像，如果边界框无效则返回None
        """
        x1, y1, x2, y2 = self.get_bbox_int()
        
        # 获取图像尺寸
        img_h, img_w = image.shape[:2]
        
        # 边界检查，确保裁剪区域在图像范围内
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        
        # 检查有效性
        if x2 <= x1 or y2 <= y1:
            return None
        
        # 裁剪图像
        cropped_image = image[y1:y2, x1:x2]
        return cropped_image
    
    def draw_bbox(self, image, color=(0, 255, 0), thickness=2):
        """
        在图像上绘制卡尔曼滤波得到的边界框
        
        参数:
            image: 输入图像
            color: 边界框颜色 (BGR格式)
            thickness: 线条粗细
        
        返回:
            image: 绘制了边界框的图像
        """
        x1, y1, x2, y2 = self.get_bbox_int()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        return image
    

if __name__ == '__main__':

    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    from ColorDetector import Detector


    rospy.init_node('bouding_box_kf_test_node', anonymous=True)
    
    # 创建检测器实例
    detector = Detector(show_image=False)
    bouding_box_kf = BoudingBox_kf()
    detector_bridge = CvBridge()

    def image_callback(data):
        """图像回调函数"""
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = detector_bridge.imgmsg_to_cv2(data, "bgr8")
            # cv2.imwrite('./left_image.jpg', cv_image)
            
            # 提取特征点
            feature_points,image,box = detector.extract_feature_points(cv_image)
            bouding_box_kf.update(box[0])
            bouding_box_kf.predict()

            cx = bouding_box_kf.x[0]
            cy = bouding_box_kf.x[1]
            w = bouding_box_kf.x[2]
            h = bouding_box_kf.x[3]
            vx = bouding_box_kf.x[4]
            vy = bouding_box_kf.x[5]
            vw = bouding_box_kf.x[6]
            vh = bouding_box_kf.x[7]
            # print(cx,cy,w,h,vx,vy,vw,vh)
            cv2.rectangle(image, (int(cx-w/2), int(cy-h/2)), (int(cx+w/2), int(cy+h/2)), (255, 0, 0), 2)
            cv2.imshow("Bouding Box", image)
            cv2.waitKey(1000)

                    
        except Exception as e:
            rospy.logerr(f"处理图像时出错: {e}")
    
    # 订阅图像话题
    rospy.Subscriber('/zed2/left_raw/image_raw_color', Image, image_callback)
    
    rospy.loginfo("检测器节点已启动")
    rospy.spin()

