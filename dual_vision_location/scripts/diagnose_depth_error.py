#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
深度估计误差诊断工具
用于分析为什么测量深度与实际深度不一致
"""

import numpy as np
import yaml
import os

def analyze_depth_error():
    """分析深度估计误差"""
    
    print("=" * 70)
    print("深度估计误差诊断工具")
    print("=" * 70)
    
    # 用户提供的数据
    measured_depth = 1.7  # 米 - 测量得到的深度
    real_depth = 2.0      # 米 - 实际深度
    
    print(f"\n已知信息：")
    print(f"  测量深度: {measured_depth:.2f} 米")
    print(f"  实际深度: {real_depth:.2f} 米")
    print(f"  误差: {abs(real_depth - measured_depth):.2f} 米 ({abs(real_depth - measured_depth)/real_depth*100:.1f}%)")
    
    # 读取当前配置
    config_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(config_dir, 'config', 'stereo_calibration.yaml')
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        baseline = np.linalg.norm(config['stereo']['T'])
        fx = config['left_camera']['fx']
        
        print(f"\n当前配置参数：")
        print(f"  基线长度: {baseline:.4f} 米")
        print(f"  焦距 fx: {fx:.2f} 像素")
        
        # 分析可能的原因
        print(f"\n" + "=" * 70)
        print("可能的原因分析：")
        print("=" * 70)
        
        # 1. 基线长度误差
        print(f"\n1. 基线长度可能不准确")
        print(f"   当前基线: {baseline:.4f} 米")
        
        # 如果其他参数正确，需要的基线长度
        # Z = (baseline * fx) / disparity
        # 如果disparity不变，real_Z / measured_Z = real_baseline / current_baseline
        required_baseline = baseline * (real_depth / measured_depth)
        print(f"   如果焦距和视差正确，需要的基线: {required_baseline:.4f} 米")
        print(f"   建议调整基线到: {required_baseline:.4f} 米")
        print(f"   或者，如果基线正确，误差可能是焦距或视差测量问题")
        
        # 2. 焦距误差
        print(f"\n2. 焦距可能不准确")
        print(f"   当前焦距: {fx:.2f} 像素")
        required_fx = fx * (real_depth / measured_depth)
        print(f"   如果基线正确，需要的焦距: {required_fx:.2f} 像素")
        
        # 3. 视差测量误差
        print(f"\n3. 视差测量可能不准确")
        print(f"   深度与视差成反比关系：Z = (baseline * fx) / disparity")
        print(f"   如果测量深度偏小，可能是视差测量偏大")
        print(f"   建议检查特征点匹配精度")
        
        # 4. 计算实际的视差
        print(f"\n4. 视差分析")
        # 假设使用当前参数，计算应该的视差
        disparity_measured = (baseline * fx) / measured_depth
        disparity_real = (baseline * fx) / real_depth
        print(f"   根据测量深度计算的视差: {disparity_measured:.2f} 像素")
        print(f"   根据实际深度计算的视差: {disparity_real:.2f} 像素")
        print(f"   视差差异: {disparity_measured - disparity_real:.2f} 像素")
        print(f"   (如果视差测量多 {disparity_measured - disparity_real:.2f} 像素，会导致深度偏小)")
        
        # 5. 畸变校正
        print(f"\n5. 畸变校正")
        dist_coeffs = config['left_camera']['distortion_coeffs']
        print(f"   畸变系数: {dist_coeffs}")
        print(f"   如果未对图像进行畸变校正，可能导致像素坐标偏差")
        print(f"   建议：在三角测量前先进行图像畸变校正")
        
        # 6. 建议的修正方案
        print(f"\n" + "=" * 70)
        print("建议的修正方案：")
        print("=" * 70)
        print(f"\n方案1: 调整基线长度")
        print(f"   将配置文件中的 T 从 [-{baseline:.4f}, 0.0, 0.0]")
        print(f"   改为 [-{required_baseline:.4f}, 0.0, 0.0]")
        
        print(f"\n方案2: 检查特征点匹配精度")
        print(f"   - 确保左右图像中的点正确配对")
        print(f"   - 检查y坐标是否对齐（应该相同）")
        print(f"   - 使用亚像素精度匹配")
        
        print(f"\n方案3: 进行图像畸变校正")
        print(f"   在三角测量前，使用 cv2.undistort() 校正图像")
        
        print(f"\n方案4: 重新标定相机")
        print(f"   使用实际的双目相机标定工具获取精确参数")
        
        # 7. 深度计算公式验证
        print(f"\n" + "=" * 70)
        print("深度计算公式验证：")
        print("=" * 70)
        print(f"\n公式: Z = (baseline × fx) / disparity")
        print(f"\n使用当前参数计算不同深度的视差：")
        test_depths = [1.0, 1.5, 2.0, 2.5, 3.0]
        print(f"{'深度(m)':<12} {'视差(像素)':<15} {'注释'}")
        print("-" * 50)
        for z in test_depths:
            d = (baseline * fx) / z
            note = "✓ 实际" if abs(z - real_depth) < 0.1 else "✗ 测量" if abs(z - measured_depth) < 0.1 else ""
            print(f"{z:<12.2f} {d:<15.2f} {note}")
        
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    analyze_depth_error()

