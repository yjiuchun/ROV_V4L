#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
暗通道先验去雾算法（Dark Channel Prior Dehazing）

功能：
    - 对输入图像进行去雾处理
    - 适用于有雾图像和 underwater 图像增强
    - 基于暗通道先验理论

作者: 杨久春
时间: 2025-01-XX

参考文献:
    Kaiming He, Jian Sun, and Xiaoou Tang. "Single image haze removal using dark channel prior."
    IEEE transactions on pattern analysis and machine intelligence, 33(12):2341-2353, 2010.
"""

import cv2
import numpy as np
from typing import Union, Optional, Tuple
from pathlib import Path


class DarkChannelPrior:
    """
    暗通道先验去雾类
    
    使用示例:
        dcp = DarkChannelPrior(patch_size=15, omega=0.95, t0=0.1)
        image = cv2.imread("hazy_image.jpg")
        dehazed_image = dcp.dehaze(image)
        cv2.imwrite("dehazed_image.jpg", dehazed_image)
    """
    
    def __init__(self, patch_size: int = 15, omega: float = 0.95, t0: float = 0.1):
        """
        初始化暗通道去雾算法
        
        参数:
            patch_size: 局部窗口大小，用于计算暗通道（默认15，必须是奇数）
            omega: 去雾强度参数，范围0-1，值越大去雾效果越强（默认0.95）
            t0: 透射率下限，防止过度去雾（默认0.1）
        """
        # 确保patch_size是奇数
        if patch_size % 2 == 0:
            patch_size += 1
            print(f"警告: patch_size必须是奇数，已自动调整为 {patch_size}")
        
        self.patch_size = patch_size
        self.omega = omega
        self.t0 = t0
    
    def _get_dark_channel(self, image: np.ndarray) -> np.ndarray:
        """
        计算图像的暗通道
        
        参数:
            image: 输入图像，BGR格式，形状为 (H, W, 3)
        
        返回:
            dark_channel: 暗通道图像，形状为 (H, W)
        """
        # 获取图像的最小通道值（RGB三通道中的最小值）
        min_channel = np.min(image, axis=2)
        
        # 对最小通道值进行最小值滤波（局部窗口内的最小值）
        kernel = np.ones((self.patch_size, self.patch_size), np.uint8)
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def _estimate_atmospheric_light(self, image: np.ndarray, 
                                   dark_channel: np.ndarray,
                                   top_percent: float = 0.1) -> Tuple[float, float, float]:
        """
        估计大气光值
        
        参数:
            image: 输入图像，BGR格式
            dark_channel: 暗通道图像
            top_percent: 选择暗通道值最大的前top_percent像素来估计大气光（默认0.1）
        
        返回:
            atmospheric_light: 大气光值，BGR三个通道的值 (B, G, R)
        """
        h, w = dark_channel.shape
        num_pixels = int(h * w * top_percent)
        
        # 将暗通道图像展平
        dark_channel_flat = dark_channel.reshape(-1)
        
        # 找到暗通道值最大的num_pixels个像素的索引
        indices = np.argpartition(dark_channel_flat, -num_pixels)[-num_pixels:]
        
        # 获取这些像素在原图像中的位置
        rows = indices // w
        cols = indices % w
        
        # 计算这些像素的RGB值的平均值作为大气光
        atmospheric_light = np.zeros(3, dtype=np.float32)
        for i in range(num_pixels):
            atmospheric_light += image[rows[i], cols[i]]
        
        atmospheric_light /= num_pixels
        
        return tuple(atmospheric_light.astype(np.uint8))
    
    def _estimate_transmission(self, image: np.ndarray, 
                              atmospheric_light: Tuple[float, float, float]) -> np.ndarray:
        """
        估计透射率
        
        参数:
            image: 输入图像，BGR格式
            atmospheric_light: 大气光值 (B, G, R)
        
        返回:
            transmission: 透射率图像，形状为 (H, W)，值范围0-1
        """
        # 将图像归一化（除以大气光值）
        normalized_image = image.astype(np.float32) / np.array(atmospheric_light, dtype=np.float32)
        
        # 计算归一化图像的暗通道
        dark_channel_norm = self._get_dark_channel(normalized_image)
        
        # 计算透射率：t = 1 - omega * dark_channel
        transmission = 1 - self.omega * dark_channel_norm
        
        # 限制透射率的下限，防止过度去雾
        transmission = np.maximum(transmission, self.t0)
        
        return transmission
    
    def _recover_scene_radiance(self, image: np.ndarray,
                               transmission: np.ndarray,
                               atmospheric_light: Tuple[float, float, float]) -> np.ndarray:
        """
        恢复场景辐射（去雾后的清晰图像）
        
        参数:
            image: 输入有雾图像，BGR格式
            transmission: 透射率图像
            atmospheric_light: 大气光值 (B, G, R)
        
        返回:
            recovered_image: 去雾后的清晰图像，BGR格式
        """
        # 将透射率扩展到三个通道
        transmission_3d = np.stack([transmission, transmission, transmission], axis=2)
        
        # 恢复公式: J = (I - A) / t + A
        # 其中 I 是有雾图像，A 是大气光，t 是透射率，J 是清晰图像
        image_float = image.astype(np.float32)
        atmospheric_light_array = np.array(atmospheric_light, dtype=np.float32)
        
        # 避免除零
        transmission_3d = np.maximum(transmission_3d, self.t0)
        
        # 恢复清晰图像
        recovered = (image_float - atmospheric_light_array) / transmission_3d + atmospheric_light_array
        
        # 限制像素值范围到0-255
        recovered = np.clip(recovered, 0, 255).astype(np.uint8)
        
        return recovered
    
    def _refine_transmission(self, transmission: np.ndarray, 
                            image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        使用引导滤波或软抠图优化透射率（可选）
        
        参数:
            transmission: 原始透射率图像
            image: 引导图像（可选，如果提供则使用引导滤波）
        
        返回:
            refined_transmission: 优化后的透射率图像
        """
        # 如果提供了引导图像，使用引导滤波
        if image is not None:
            # 转换为灰度图作为引导图像
            if len(image.shape) == 3:
                guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                guide = image
            
            # 使用引导滤波优化透射率
            # 注意：OpenCV的guidedFilter需要将图像归一化到0-1
            transmission_norm = transmission.astype(np.float32) / 255.0
            guide_norm = guide.astype(np.float32) / 255.0
            
            # 使用双边滤波作为替代（引导滤波的简化版本）
            refined = cv2.bilateralFilter(
                (transmission_norm * 255).astype(np.uint8), 
                d=9, 
                sigmaColor=75, 
                sigmaSpace=75
            ).astype(np.float32) / 255.0
            
            return refined
        else:
            # 如果没有引导图像，使用简单的高斯滤波
            transmission_uint8 = (transmission * 255).astype(np.uint8)
            refined = cv2.GaussianBlur(transmission_uint8, (5, 5), 0).astype(np.float32) / 255.0
            return refined
    
    def dehaze(self, image: Union[np.ndarray, str, Path],
               refine: bool = True) -> np.ndarray:
        """
        对输入图像进行去雾处理
        
        参数:
            image: 输入图像，可以是：
                  - numpy数组（BGR格式，OpenCV格式）
                  - 图像文件路径（字符串或Path对象）
            refine: 是否对透射率进行优化（默认True）
        
        返回:
            dehazed_image: 去雾后的图像，BGR格式
        """
        # 读取图像
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"无法读取图像: {image}")
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")
        
        # 确保图像是BGR格式
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError("输入图像必须是BGR格式的彩色图像")
        
        # 步骤1: 计算暗通道
        dark_channel = self._get_dark_channel(img)
        
        # 步骤2: 估计大气光值
        atmospheric_light = self._estimate_atmospheric_light(img, dark_channel)
        
        # 步骤3: 估计透射率
        transmission = self._estimate_transmission(img, atmospheric_light)
        
        # 步骤4: 优化透射率（可选）
        if refine:
            transmission = self._refine_transmission(transmission, img)
        
        # 步骤5: 恢复清晰图像
        dehazed_image = self._recover_scene_radiance(img, transmission, atmospheric_light)
        
        return dehazed_image
    
    def get_transmission_map(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """
        获取透射率图（用于可视化）
        
        参数:
            image: 输入图像
        
        返回:
            transmission_map: 透射率图像，值范围0-255
        """
        # 读取图像
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"无法读取图像: {image}")
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")
        
        # 计算透射率
        dark_channel = self._get_dark_channel(img)
        atmospheric_light = self._estimate_atmospheric_light(img, dark_channel)
        transmission = self._estimate_transmission(img, atmospheric_light)
        
        # 转换为0-255范围用于可视化
        transmission_map = (transmission * 255).astype(np.uint8)
        
        return transmission_map
    
    def get_dark_channel(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """
        获取暗通道图（用于可视化）
        
        参数:
            image: 输入图像
        
        返回:
            dark_channel: 暗通道图像，值范围0-255
        """
        # 读取图像
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"无法读取图像: {image}")
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")
        
        # 计算暗通道
        dark_channel = self._get_dark_channel(img)
        
        return dark_channel


def main():
    """示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='暗通道去雾算法示例')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图像路径（如果不指定，则显示图像）')
    parser.add_argument('--patch-size', type=int, default=15,
                       help='局部窗口大小（默认15）')
    parser.add_argument('--omega', type=float, default=0.95,
                       help='去雾强度参数，范围0-1（默认0.95）')
    parser.add_argument('--t0', type=float, default=0.1,
                       help='透射率下限（默认0.1）')
    parser.add_argument('--no-refine', action='store_true',
                       help='不对透射率进行优化')
    parser.add_argument('--show-transmission', action='store_true',
                       help='显示透射率图')
    parser.add_argument('--show-dark-channel', action='store_true',
                       help='显示暗通道图')
    
    args = parser.parse_args()
    
    # 创建去雾器
    dcp = DarkChannelPrior(
        patch_size=args.patch_size,
        omega=args.omega,
        t0=args.t0
    )
    
    # 进行去雾处理
    print("正在进行去雾处理...")
    dehazed_image = dcp.dehaze(args.input, refine=not args.no_refine)
    
    # 保存或显示结果
    if args.output:
        cv2.imwrite(args.output, dehazed_image)
        print(f"去雾结果已保存到: {args.output}")
    else:
        cv2.imshow("Dehazed Image", dehazed_image)
    
    # 显示透射率图
    if args.show_transmission:
        transmission_map = dcp.get_transmission_map(args.input)
        cv2.imshow("Transmission Map", transmission_map)
    
    # 显示暗通道图
    if args.show_dark_channel:
        dark_channel = dcp.get_dark_channel(args.input)
        cv2.imshow("Dark Channel", dark_channel)
    
    if not args.output:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

