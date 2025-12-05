#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
亮度峰值检测与3D可视化类

功能：
    - 计算灰度图直方图并检测峰值点
    - 返回峰值点与峰值强度
    - 3D亮度可视化
"""

import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib
# 设置非交互式后端，适用于SSH环境
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path
from typing import Union, Tuple, List, Optional


class GetLightnessPeak:
    """
    亮度峰值检测与3D可视化类
    
    功能：
        - 检测灰度图直方图的峰值点与峰值强度
        - 3D亮度可视化
    """
    
    def __init__(self, colormap: str = 'viridis'):
        """
        初始化类
        
        参数:
            colormap: 3D可视化的默认颜色映射
        """
        self.colormap = colormap
        self.gray_image = None
        self.hist = None
        self.peaks = None
        self.peak_intensities = None
        self.peak_counts = None
    
    def _load_image(self, image_or_path: Union[np.ndarray, str]) -> np.ndarray:
        """
        加载图片（内部方法）
        
        参数:
            image_or_path: 图片数组或图片路径
        
        返回:
            gray_image: 灰度图数组
        """
        if isinstance(image_or_path, str):
            # 如果是路径，读取图片
            if not os.path.exists(image_or_path):
                raise FileNotFoundError(f"图片文件不存在: {image_or_path}")
            image = cv2.imread(image_or_path)
            if image is None:
                raise ValueError(f"无法读取图片文件: {image_or_path}")
            # 转换为灰度图
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image
        elif isinstance(image_or_path, np.ndarray):
            # 如果已经是数组
            if len(image_or_path.shape) == 3:
                gray_image = cv2.cvtColor(image_or_path, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image_or_path.copy()
        else:
            raise TypeError(f"不支持的类型: {type(image_or_path)}")
        
        self.gray_image = gray_image
        return gray_image
    
    def get_lightness_peaks(self, 
                           image_or_path: Union[np.ndarray, str]) -> List[int]:
        """
        获取图像中最大值点的x坐标列表（核心函数）
        
        参数:
            image_or_path: 图片数组或图片路径
        
        返回:
            max_x_list: 最大值点的x坐标列表
        """
        # 加载图片
        gray_img = self._load_image(image_or_path)
        
        # 找到图像中的最大值
        max_value = gray_img.max()
        
        # 找到所有最大值的位置
        max_mask = gray_img == max_value
        
        # 获取所有最大值点的坐标 (y, x)
        max_positions = np.argwhere(max_mask)
        
        # 提取x坐标列表
        max_x_list = [int(x) for y, x in max_positions]
        
        # 保存结果（用于后续使用）
        self.max_value = max_value
        self.max_positions = max_positions
        self.max_x_list = max_x_list
        
        return max_x_list
    
    def visualize_3d_lightness(self,
                              image_or_path: Union[np.ndarray, str] = None,
                              show: bool = False,
                              save: bool = False,
                              save_path: Optional[str] = None,
                              downsample: int = 1,
                              colormap: Optional[str] = None,
                              title: Optional[str] = None,
                              mark_extrema: bool = True,
                              mode: str = 'surface') -> Optional[dict]:
        """
        显示3D强度图（函数2）
        
        参数:
            image_or_path: 图片数组或图片路径，如果为None则使用已加载的图片
            show: 是否显示图表窗口
            save: 是否保存图片
            save_path: 保存路径，如果为None且save=True，则自动生成
            downsample: 下采样因子（默认1，即不采样）
            colormap: 颜色映射（默认使用self.colormap）
            title: 图表标题
            mark_extrema: 是否标记极值点（默认True）
            mode: 可视化模式，'surface'（表面图）或'scatter'（散点图）
        
        返回:
            extrema_info: 极值点信息字典（如果mark_extrema为True）
        """
        # 加载图片（如果未提供或需要重新加载）
        if image_or_path is not None:
            gray_image = self._load_image(image_or_path)
        elif self.gray_image is not None:
            gray_image = self.gray_image
        else:
            raise ValueError("未提供图片，请传入image_or_path参数或先调用get_lightness_peaks")
        
        if colormap is None:
            colormap = self.colormap
        
        # 保存原始图像用于查找极值点
        original_image = gray_image.copy()
        
        # 下采样以减少数据点（提高性能）
        if downsample > 1:
            gray_image = gray_image[::downsample, ::downsample]
        
        height, width = gray_image.shape
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 查找并标记极值点
        extrema_info = None
        max_x_list = []
        max_y_list = []
        max_z_list = []
        min_x_list = []
        min_y_list = []
        min_z_list = []
        
        if mark_extrema:
            max_value = original_image.max()
            min_value = original_image.min()
            
            # 找到所有最大值和最小值的位置
            max_mask = original_image == max_value
            min_mask = original_image == min_value
            
            # 获取所有极值点的坐标 (y, x)
            max_positions = np.argwhere(max_mask)
            min_positions = np.argwhere(min_mask)
            
            extrema_info = {
                'max_value': max_value,
                'max_positions': max_positions.tolist(),
                'min_value': min_value,
                'min_positions': min_positions.tolist()
            }
            
            # 将原始坐标转换为下采样后的坐标
            for y_orig, x_orig in max_positions:
                x_down = x_orig // downsample
                y_down = y_orig // downsample
                if 0 <= x_down < width and 0 <= y_down < height:
                    max_x_list.append(x_down)
                    max_y_list.append(y_down)
                    max_z_list.append(gray_image[y_down, x_down])
            
            for y_orig, x_orig in min_positions:
                x_down = x_orig // downsample
                y_down = y_orig // downsample
                if 0 <= x_down < width and 0 <= y_down < height:
                    min_x_list.append(x_down)
                    min_y_list.append(y_down)
                    min_z_list.append(gray_image[y_down, x_down])
        
        # 根据模式选择可视化方式
        if mode == 'surface':
            # 创建坐标网格
            x = np.arange(0, width)
            y = np.arange(0, height)
            X, Y = np.meshgrid(x, y)
            Z = gray_image.astype(float)
            
            # 绘制3D表面图
            surf = ax.plot_surface(X, Y, Z,
                                  cmap=colormap,
                                  linewidth=0,
                                  antialiased=True,
                                  alpha=0.9)
            colorbar_obj = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='亮度值')
        else:  # scatter mode
            # 创建坐标和亮度值数组
            x = np.arange(0, width)
            y = np.arange(0, height)
            X, Y = np.meshgrid(x, y)
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            Z_flat = gray_image.flatten()
            
            # 绘制散点图
            scatter = ax.scatter(X_flat, Y_flat, Z_flat,
                                c=Z_flat,
                                cmap=colormap,
                                s=1,
                                alpha=0.6)
            colorbar_obj = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, label='亮度值')
        
        # 标记极值点
        if mark_extrema:
            # 标记最大值点（红色）
            if max_x_list:
                print(f"最大值点数量: {len(max_x_list)}")
                ax.scatter(max_x_list, max_y_list, max_z_list,
                          c='red', s=200, marker='^',
                          label=f'最大值点 (值={extrema_info["max_value"]})',
                          edgecolors='darkred', linewidths=2, zorder=10)
            
            # 标记最小值点（蓝色）
            if min_x_list:
                ax.scatter(min_x_list, min_y_list, min_z_list,
                          c='blue', s=200, marker='v',
                          label=f'最小值点 (值={extrema_info["min_value"]})',
                          edgecolors='darkblue', linewidths=2, zorder=10)
            
            # 添加图例
            if max_x_list or min_x_list:
                ax.legend(loc='upper left')
        
        # 设置标签和标题
        ax.set_xlabel('X (像素)', fontsize=12)
        ax.set_ylabel('Y (像素)', fontsize=12)
        ax.set_zlabel('亮度值', fontsize=12)
        
        if title is None:
            title = '3D Lightness Visualization'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 设置视角
        ax.view_init(elev=30, azim=45)
        
        # 保存图片
        if save:
            if save_path is None:
                # 自动生成保存路径
                if isinstance(image_or_path, str):
                    input_path = Path(image_or_path)
                    save_path = input_path.parent / f"{input_path.stem}_3d.png"
                else:
                    save_path = "lightness_3d.png"
            
            # 确保输出目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D可视化已保存到: {save_path}")
        
        # 显示图表
        if show:
            plt.show()
        else:
            plt.close()
        
        return extrema_info


if __name__ == "__main__":
    # 测试代码
    get_lightness_peak = GetLightnessPeak()
    
    # 测试1: 使用路径
    image_path = '/home/yjc/Project/rov_ws/src/vision4location/src/image_save/crop_img/spilt_cropimg/crop_img_40x42_1.jpg'
    
    # 获取最大值点的x坐标列表
    max_x_list = get_lightness_peak.get_lightness_peaks(image_path)
    print(f"检测到的最大值点x坐标列表: {max_x_list}")
    print(f"最大值点数量: {len(max_x_list)}")
    print(f"图像的最大亮度值: {get_lightness_peak.max_value}")
    
    # 3D可视化
    get_lightness_peak.visualize_3d_lightness(
        image_path,
        show=False,
        save=True,
        save_path='/home/yjc/Project/rov_ws/src/vision4location/src/image_save/crop_img/spilt_cropimg/crop_img_40x42_1_3d.png'
    )
    
    # 测试2: 使用图片数组
    image = cv2.imread(image_path)
    max_x_list2 = get_lightness_peak.get_lightness_peaks(image)
    print(f"使用图片数组检测到的最大值点x坐标列表: {max_x_list2}")
    print(f"最大值点数量: {len(max_x_list2)}")

