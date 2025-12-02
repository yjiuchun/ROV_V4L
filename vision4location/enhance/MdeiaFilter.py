#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中值滤波类

功能：
    - 对输入图像进行中值滤波处理
    - 去除图像中的椒盐噪声
    - 保持图像边缘信息

作者: 杨久春
时间: 2025-01-XX
"""

import cv2
import numpy as np
from typing import Union, Optional


class MediaFilter:
    """
    中值滤波类
    
    中值滤波是一种非线性滤波方法，能够有效去除椒盐噪声，
    同时保持图像边缘信息。
    """
    
    def __init__(self, kernel_size: int = 5):
        """
        初始化中值滤波器
        
        参数:
            kernel_size: 滤波核大小，必须是奇数（1, 3, 5, 7, 9等）
                        默认值为5
        """
        # 确保核大小为奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
            print(f"警告: 核大小必须是奇数，已自动调整为 {kernel_size}")
        
        self.kernel_size = kernel_size
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        对输入图像进行中值滤波
        
        参数:
            image: 输入图像，可以是灰度图或彩色图（BGR格式）
                  类型: numpy.ndarray，形状为 (H, W) 或 (H, W, 3)
        
        返回:
            filtered_image: 滤波后的图像，类型和形状与输入图像相同
        """
        if image is None:
            raise ValueError("输入图像不能为None")
        
        if len(image.shape) == 2:
            # 灰度图像
            filtered_image = cv2.medianBlur(image, self.kernel_size)
        elif len(image.shape) == 3:
            # 彩色图像（BGR格式）
            filtered_image = cv2.medianBlur(image, self.kernel_size)
        else:
            raise ValueError(f"不支持的图像维度: {image.shape}")
        
        return filtered_image
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        使类可以像函数一样调用
        
        参数:
            image: 输入图像
        
        返回:
            滤波后的图像
        """
        return self.filter(image)
    
    def set_kernel_size(self, kernel_size: int):
        """
        设置滤波核大小
        
        参数:
            kernel_size: 新的核大小（必须是奇数）
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
            print(f"警告: 核大小必须是奇数，已自动调整为 {kernel_size}")
        self.kernel_size = kernel_size
    
    def get_kernel_size(self) -> int:
        """
        获取当前滤波核大小
        
        返回:
            当前核大小
        """
        return self.kernel_size


# 使用示例
if __name__ == "__main__":
    # 创建中值滤波器，核大小为5
    median_filter = MediaFilter(kernel_size=5)
    
    # 读取测试图像
    test_image_path = "/home/yjc/Project/rov_ws/output_images/37.jpg"
    try:
        img = cv2.imread(test_image_path)
        if img is None:
            print(f"无法读取图像: {test_image_path}")
            print("请提供有效的图像路径进行测试")
        else:
            # 进行中值滤波
            filtered_img = median_filter.filter(img)
            
            # 显示结果（如果需要）
            # cv2.imshow("Original", img)
            # cv2.imshow("Filtered", filtered_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            print(f"图像滤波完成")
            print(f"原始图像形状: {img.shape}")
            print(f"滤波后图像形状: {filtered_img.shape}")
            print(f"使用核大小: {median_filter.get_kernel_size()}")
    except Exception as e:
        print(f"测试时出错: {e}")

