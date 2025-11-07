#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
双目三角测量测试脚本
用于测试三角测量功能，输入左右图像的像素坐标，输出3D坐标
"""

import sys
import os
import numpy as np

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入三角测量模块
try:
    import rospy
    rospy_available = True
except:
    rospy_available = False
    print("警告: ROS未安装，将使用静默模式")

if rospy_available:
    from stereo_triangulation import StereoTriangulation
else:
    # 如果ROS不可用，创建一个简化版本
    import cv2
    import yaml
    
    class StereoTriangulation:
        def __init__(self, config_path=None):
            self.load_config(config_path)
        
        def load_config(self, config_path=None):
            if config_path is None:
                config_dir = os.path.dirname(os.path.dirname(__file__))
                config_path = os.path.join(config_dir, 'config', 'stereo_calibration.yaml')
            
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            left_cam = config['left_camera']
            self.left_camera_matrix = np.array([
                [left_cam['fx'], 0, left_cam['cx']],
                [0, left_cam['fy'], left_cam['cy']],
                [0, 0, 1]
            ], dtype=np.float32)
            
            right_cam = config['right_camera']
            self.right_camera_matrix = np.array([
                [right_cam['fx'], 0, right_cam['cx']],
                [0, right_cam['fy'], right_cam['cy']],
                [0, 0, 1]
            ], dtype=np.float32)
            
            stereo = config['stereo']
            self.R = np.array(stereo['R'], dtype=np.float32)
            self.T = np.array(stereo['T'], dtype=np.float32)
            
            self.left_proj_matrix = self.left_camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
            self.right_proj_matrix = self.right_camera_matrix @ np.hstack([self.R, self.T.reshape(3, 1)])
        
        def triangulate_points(self, left_points, right_points):
            left_points = np.array(left_points, dtype=np.float32)
            right_points = np.array(right_points, dtype=np.float32)
            
            if len(left_points.shape) == 2 and left_points.shape[1] == 2:
                left_points = left_points.reshape(-1, 1, 2)
            if len(right_points.shape) == 2 and right_points.shape[1] == 2:
                right_points = right_points.reshape(-1, 1, 2)
            
            points_4d = cv2.triangulatePoints(
                self.left_proj_matrix,
                self.right_proj_matrix,
                left_points,
                right_points
            )
            
            points_3d = points_4d[:3] / points_4d[3]
            points_3d = points_3d.T
            
            return points_3d


def test_triangulation():
    """测试三角测量功能"""
    
    if rospy_available:
        rospy.init_node('test_triangulation', anonymous=True)
    
    print("=" * 60)
    print("双目三角测量测试")
    print("=" * 60)
    
    # 初始化三角测量器
    try:
        triangulator = StereoTriangulation()
        print("✓ 三角测量器初始化成功")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return
    
    # 示例：左右图像的像素坐标
    # 假设检测到4个配对的关键点
    print("\n输入数据：")
    left_points = np.array([
        [320.0, 240.0],   # 点1
        [480.0, 240.0],   # 点2
        [480.0, 360.0],   # 点3
        [320.0, 360.0]    # 点4
    ], dtype=np.float32)
    
    right_points = np.array([
        [300.0, 240.0],   # 点1（视差约20像素）
        [460.0, 240.0],   # 点2（视差约20像素）
        [460.0, 360.0],   # 点3（视差约20像素）
        [300.0, 360.0]    # 点4（视差约20像素）
    ], dtype=np.float32)
    
    print(f"左图像像素坐标 ({len(left_points)}个点):")
    for i, pt in enumerate(left_points):
        print(f"  点{i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    print(f"\n右图像像素坐标 ({len(right_points)}个点):")
    for i, pt in enumerate(right_points):
        print(f"  点{i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    # 执行三角测量
    print("\n执行三角测量...")
    try:
        points_3d = triangulator.triangulate_points(left_points, right_points)
        
        if points_3d is not None:
            print("✓ 三角测量成功！")
            print("\n3D坐标（相机坐标系，单位：米）：")
            print("-" * 60)
            print(f"{'点号':<6} {'X (m)':<12} {'Y (m)':<12} {'Z (m)':<12} {'深度':<12}")
            print("-" * 60)
            
            for i, pt in enumerate(points_3d):
                depth = np.sqrt(pt[0]**2 + pt[1]**2 + pt[2]**2)
                print(f"点{i+1:<5} {pt[0]:>11.4f} {pt[1]:>11.4f} {pt[2]:>11.4f} {depth:>11.4f}")
            
            print("-" * 60)
            
            # 计算视差
            print("\n视差分析：")
            disparities = left_points[:, 0] - right_points[:, 0]
            for i, disp in enumerate(disparities):
                depth_est = (triangulator.get_baseline() * triangulator.left_camera_matrix[0, 0]) / disp
                print(f"点{i+1}: 视差={disp:.1f}像素, 深度估计={depth_est:.4f}m, 实际深度={np.linalg.norm(points_3d[i]):.4f}m")
        
        else:
            print("✗ 三角测量失败")
    
    except Exception as e:
        print(f"✗ 三角测量出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    # 可以通过命令行参数输入实际的像素坐标
    if len(sys.argv) > 1:
        print("使用方法：")
        print("  python test_triangulation.py")
        print("\n或者直接修改脚本中的 left_points 和 right_points 数组")
    else:
        test_triangulation()


