#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
双目三角测量和定位模块
用于从左右图像的像素坐标计算3D坐标，并进行位姿估计
"""

import cv2
import numpy as np
import yaml
import os
import rospy


class StereoTriangulation:
    def __init__(self, config_path=None):
        """
        初始化双目三角测量器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        self.load_config(config_path)
        rospy.loginfo("双目三角测量模块已初始化")
    
    def load_config(self, config_path=None):
        """从配置文件加载相机标定参数"""
        if config_path is None:
            config_dir = os.path.dirname(os.path.dirname(__file__))
            config_path = os.path.join(config_dir, 'config', 'stereo_calibration.yaml')
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # 左相机内参
            left_cam = config['left_camera']
            self.left_camera_matrix = np.array([
                [left_cam['fx'], 0, left_cam['cx']],
                [0, left_cam['fy'], left_cam['cy']],
                [0, 0, 1]
            ], dtype=np.float32)
            self.left_dist_coeffs = np.array(left_cam['distortion_coeffs'], dtype=np.float32)
            
            # 右相机内参
            right_cam = config['right_camera']
            self.right_camera_matrix = np.array([
                [right_cam['fx'], 0, right_cam['cx']],
                [0, right_cam['fy'], right_cam['cy']],
                [0, 0, 1]
            ], dtype=np.float32)
            self.right_dist_coeffs = np.array(right_cam['distortion_coeffs'], dtype=np.float32)
            
            # 立体标定参数
            stereo = config['stereo']
            self.R = np.array(stereo['R'], dtype=np.float32)  # 旋转矩阵
            self.T = np.array(stereo['T'], dtype=np.float32)  # 平移向量
            self.E = np.array(stereo.get('E', [[1,0,0],[0,1,0],[0,0,1]]), dtype=np.float32)  # 本质矩阵
            self.F = np.array(stereo.get('F', [[1,0,0],[0,1,0],[0,0,1]]), dtype=np.float32)  # 基础矩阵
            
            # 计算投影矩阵
            self.left_proj_matrix = self.left_camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
            self.right_proj_matrix = self.right_camera_matrix @ np.hstack([self.R, self.T.reshape(3, 1)])
            
            rospy.loginfo("双目标定参数加载成功")
            rospy.loginfo(f"基线长度 (baseline): {np.linalg.norm(self.T):.4f} 米")
            
        except FileNotFoundError:
            rospy.logwarn(f"配置文件未找到: {config_path}")
            rospy.logwarn("使用默认标定参数")
            self.load_default_config()
        except Exception as e:
            rospy.logerr(f"加载配置文件失败: {e}")
            rospy.logwarn("使用默认标定参数")
            self.load_default_config()
    
    def load_default_config(self):
        """加载默认标定参数（示例参数，需要根据实际标定结果修改）"""
        # 默认左相机内参 (假设1280x720图像)
        self.left_camera_matrix = np.array([
            [476.7, 0, 640.5],
            [0, 476.7, 360.5],
            [0, 0, 1]
        ], dtype=np.float32)
        self.left_dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # 默认右相机内参
        self.right_camera_matrix = self.left_camera_matrix.copy()
        self.right_dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # 默认立体标定参数 (假设基线为0.12米)
        self.R = np.eye(3, dtype=np.float32)
        self.T = np.array([-0.12, 0.0, 0.0], dtype=np.float32)  # 基线长度12cm
        
        # 计算投影矩阵
        self.left_proj_matrix = self.left_camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.right_proj_matrix = self.right_camera_matrix @ np.hstack([self.R, self.T.reshape(3, 1)])
        
        rospy.logwarn("注意：使用的是默认标定参数，请使用实际标定结果替换")
    
    def undistort_points(self, points, camera_matrix, dist_coeffs):
        """
        对像素坐标进行畸变校正
        
        Args:
            points: 像素坐标，形状为 (N, 2) 或 (N, 1, 2)
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
        
        Returns:
            undistorted_points: 校正后的像素坐标
        """
        points_array = np.array(points, dtype=np.float32)
        
        # 转换为 (N, 1, 2) 格式
        if len(points_array.shape) == 2 and points_array.shape[1] == 2:
            points_array = points_array.reshape(-1, 1, 2)
        
        # 使用undistortPoints进行校正
        undistorted = cv2.undistortPoints(
            points_array,
            camera_matrix,
            dist_coeffs,
            P=camera_matrix  # 使用相同的相机矩阵作为输出
        )
        
        return undistorted
    
    def triangulate_points(self, left_points, right_points, use_undistort=True):
        """
        使用三角测量计算3D坐标
        
        Args:
            left_points: 左图像像素坐标，形状为 (N, 2) 或 (N, 1, 2) 的numpy数组
            right_points: 右图像像素坐标，形状为 (N, 2) 或 (N, 1, 2) 的numpy数组
            use_undistort: 是否进行畸变校正（默认True，推荐启用）
        
        Returns:
            points_3d: 3D坐标，形状为 (4, N) 的齐次坐标，需要除以最后一列得到 (X, Y, Z, 1)
                      或者返回 (N, 3) 的非齐次坐标
        """
        # 确保输入是正确的格式
        left_points = np.array(left_points, dtype=np.float32)
        right_points = np.array(right_points, dtype=np.float32)
        
        # 如果是 (N, 2) 格式，转换为 (N, 1, 2)
        if len(left_points.shape) == 2 and left_points.shape[1] == 2:
            left_points = left_points.reshape(-1, 1, 2)
        if len(right_points.shape) == 2 and right_points.shape[1] == 2:
            right_points = right_points.reshape(-1, 1, 2)
        
        if len(left_points) != len(right_points):
            rospy.logerr(f"左右图像点数不匹配: 左{len(left_points)}, 右{len(right_points)}")
            return None
        
        # 畸变校正（如果启用）
        if use_undistort:
            # 检查是否有非零畸变系数
            left_has_distortion = np.any(np.abs(self.left_dist_coeffs) > 1e-6)
            right_has_distortion = np.any(np.abs(self.right_dist_coeffs) > 1e-6)
            
            if left_has_distortion:
                left_points = self.undistort_points(
                    left_points, 
                    self.left_camera_matrix, 
                    self.left_dist_coeffs
                )
            if right_has_distortion:
                right_points = self.undistort_points(
                    right_points,
                    self.right_camera_matrix,
                    self.right_dist_coeffs
                )
        
        # 使用OpenCV的三角测量函数
        points_4d = cv2.triangulatePoints(
            self.left_proj_matrix,
            self.right_proj_matrix,
            left_points,
            right_points
        )
        
        # 转换为非齐次坐标 (N, 3)
        points_3d = points_4d[:3] / points_4d[3]  # 除以齐次坐标的w分量
        points_3d = points_3d.T  # 转置为 (N, 3)
        
        return points_3d
    
    def solve_pnp_from_3d(self, points_3d, object_points_3d):
        """
        使用已知的3D点和计算得到的3D点进行PnP求解
        
        Args:
            points_3d: 通过三角测量得到的3D点 (相机坐标系), 形状 (N, 3)
            object_points_3d: 世界坐标系中的3D点, 形状 (N, 3)
        
        Returns:
            success: 是否成功
            rvec: 旋转向量
            tvec: 平移向量
        """
        if len(points_3d) < 4:
            rospy.logwarn("点数不足，无法进行PnP求解")
            return False, None, None
        
        # 使用solvePnP求解位姿
        # points_3d是相机坐标系中的点，object_points_3d是世界坐标系中的点
        # 我们需要找到从世界坐标系到相机坐标系的变换
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points_3d.reshape(-1, 1, 3),
                points_3d.reshape(-1, 1, 3),  # 使用相机坐标系中的点作为"图像点"
                self.left_camera_matrix,
                self.left_dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                return True, rvec, tvec
            else:
                return False, None, None
        except Exception as e:
            rospy.logerr(f"PnP求解失败: {e}")
            return False, None, None
    
    def solve_pnp_from_image_points(self, left_image_points, right_image_points, object_points_3d):
        """
        完整的双目定位流程：三角测量 + PnP求解
        
        Args:
            left_image_points: 左图像像素坐标, 形状 (N, 2)
            right_image_points: 右图像像素坐标, 形状 (N, 2)
            object_points_3d: 世界坐标系中的3D点, 形状 (N, 3)
        
        Returns:
            success: 是否成功
            points_3d: 三角测量得到的3D点 (相机坐标系)
            rvec: 旋转向量
            tvec: 平移向量
        """
        # 步骤1: 三角测量
        points_3d = self.triangulate_points(left_image_points, right_image_points)
        
        if points_3d is None:
            return False, None, None, None
        
        # 步骤2: PnP求解（如果提供了世界坐标点）
        # 使用左图像的像素坐标和世界坐标系中的点进行PnP求解
        # 这样可以找到从世界坐标系到左相机坐标系的变换
        if object_points_3d is not None and len(object_points_3d) >= 4:
            try:
                # 确保输入格式正确
                object_pts = np.array(object_points_3d, dtype=np.float32).reshape(-1, 1, 3)
                left_img_pts = np.array(left_image_points, dtype=np.float32).reshape(-1, 1, 2)
                
                # 使用solvePnP求解位姿
                # object_pts: 世界坐标系中的3D点
                # left_img_pts: 左图像中的2D像素坐标
                # 结果：从世界坐标系到左相机坐标系的变换 (rvec, tvec)
                success, rvec, tvec = cv2.solvePnP(
                    object_pts,
                    left_img_pts,
                    self.left_camera_matrix,
                    self.left_dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    return True, points_3d, rvec, tvec
                else:
                    rospy.logwarn("PnP求解失败")
                    return False, points_3d, None, None
            except Exception as e:
                rospy.logerr(f"PnP求解出错: {e}")
                return False, points_3d, None, None
        
        # 如果不需要PnP，只返回3D点
        return True, points_3d, None, None
    
    def get_baseline(self):
        """获取基线长度（单位：米）"""
        return np.linalg.norm(self.T)
    
    def get_depth_from_disparity(self, disparity):
        """
        从视差计算深度
        
        Args:
            disparity: 视差值（像素）
        
        Returns:
            depth: 深度值（米）
        """
        baseline = self.get_baseline()
        fx = self.left_camera_matrix[0, 0]
        
        if disparity == 0:
            return float('inf')
        
        depth = (baseline * fx) / disparity
        return depth

