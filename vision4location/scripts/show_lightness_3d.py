#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将图片转为灰度图并绘制三维亮度可视化

功能：
    - 读取图片文件
    - 转换为灰度图
    - 根据亮度值绘制三维可视化（x, y坐标为位置，亮度值为高度）

作者: 杨久春
时间: 2025-01-XX
"""

import cv2
import numpy as np
import matplotlib
# 设置非交互式后端，适用于SSH环境
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from pathlib import Path


class Lightness3DVisualizer:
    """
    三维亮度可视化类
    
    功能：
        - 读取图片文件并转换为灰度图
        - 查找极值点
        - 绘制三维可视化（表面图和散点图）
    """
    
    def __init__(self, colormap: str = 'viridis'):
        """
        初始化可视化器
        
        参数:
            colormap: 默认颜色映射
        """
        self.colormap = colormap
        self.gray_image = None
        self.extrema_info = None
    
    def load_and_convert_to_grayscale(self, image_path: str) -> np.ndarray:
        """
        加载图片并转换为灰度图
        
        参数:
            image_path: 图片文件路径
        
        返回:
            gray_image: 灰度图数组
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片文件: {image_path}")
        
        # 转换为灰度图
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return self.gray_image
    
    def find_extrema_points(self, gray_image: np.ndarray = None) -> dict:
        """
        查找灰度图中的极值点（最大值和最小值点）
        
        参数:
            gray_image: 灰度图数组，如果为None则使用self.gray_image
    
        返回:
            extrema: 包含极值点信息的字典
                - max_value: 最大值
                - max_positions: 最大值点的坐标列表 [(y1, x1), (y2, x2), ...]
                - min_value: 最小值
                - min_positions: 最小值点的坐标列表 [(y1, x1), (y2, x2), ...]
        """
        if gray_image is None:
            if self.gray_image is None:
                raise ValueError("未加载图片，请先调用load_and_convert_to_grayscale或提供gray_image参数")
            gray_image = self.gray_image
        
        max_value = gray_image.max()
        min_value = gray_image.min()
        
        # 找到所有最大值和最小值的位置
        max_mask = gray_image == max_value
        min_mask = gray_image == min_value
        
        # 获取所有极值点的坐标 (y, x)
        max_positions = np.argwhere(max_mask)
        min_positions = np.argwhere(min_mask)
        
        self.extrema_info = {
            'max_value': max_value,
            'max_positions': max_positions.tolist(),  # 转换为列表，格式为 [(y, x), ...]
            'min_value': min_value,
            'min_positions': min_positions.tolist()   # 转换为列表，格式为 [(y, x), ...]
        }
        
        return self.extrema_info
    
    def visualize_3d_lightness(self, gray_image: np.ndarray = None,
                               downsample: int = 1,
                               colormap: str = None,
                               title: str = '3D Lightness Visualization',
                               save_path: str = None,
                               show: bool = True,
                               mark_extrema: bool = True):
        """
        绘制灰度图的三维可视化
        
        参数:
            gray_image: 灰度图数组，如果为None则使用self.gray_image
            downsample: 下采样因子，用于减少数据点（默认1，即不采样）
                        例如downsample=2表示每隔2个像素采样一次
            colormap: 颜色映射（默认使用self.colormap）
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
            show: 是否显示图表
            mark_extrema: 是否标记极值点（默认True）
        
        返回:
            extrema_info: 极值点信息字典（如果mark_extrema为True）
        """
        if gray_image is None:
            if self.gray_image is None:
                raise ValueError("未加载图片，请先调用load_and_convert_to_grayscale或提供gray_image参数")
            gray_image = self.gray_image
        
        if colormap is None:
            colormap = self.colormap
        
        # 保存原始图像用于查找极值点
        original_image = gray_image.copy()
        
        # 下采样以减少数据点（提高性能）
        if downsample > 1:
            gray_image = gray_image[::downsample, ::downsample]
        
        height, width = gray_image.shape
        
        # 创建坐标网格
        x = np.arange(0, width)
        y = np.arange(0, height)
        X, Y = np.meshgrid(x, y)
        
        # 亮度值作为Z轴
        Z = gray_image.astype(float)
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制3D表面图
        surf = ax.plot_surface(X, Y, Z, 
                              cmap=colormap,
                              linewidth=0,
                              antialiased=True,
                              alpha=0.9)
        
        # 查找并标记极值点
        extrema_info = None
        if mark_extrema:
            extrema = self.find_extrema_points(original_image)
            extrema_info = extrema
        
        # 将原始坐标转换为下采样后的坐标
        max_x_list = []
        max_y_list = []
        max_z_list = []
        min_x_list = []
        min_y_list = []
        min_z_list = []
        
        for y_orig, x_orig in extrema['max_positions']:
            # 计算下采样后的坐标
            x_down = x_orig // downsample
            y_down = y_orig // downsample
            # 确保坐标在范围内
            if 0 <= x_down < width and 0 <= y_down < height:
                max_x_list.append(x_down)
                max_y_list.append(y_down)
                max_z_list.append(gray_image[y_down, x_down])
        
        for y_orig, x_orig in extrema['min_positions']:
            # 计算下采样后的坐标
            x_down = x_orig // downsample
            y_down = y_orig // downsample
            # 确保坐标在范围内
            if 0 <= x_down < width and 0 <= y_down < height:
                min_x_list.append(x_down)
                min_y_list.append(y_down)
                min_z_list.append(gray_image[y_down, x_down])
        
        # 标记最大值点（红色）
        if max_x_list:
            ax.scatter(max_x_list, max_y_list, max_z_list,
                      c='red', s=200, marker='^', 
                      label=f'最大值点 (值={extrema["max_value"]})',
                      edgecolors='darkred', linewidths=2, zorder=10)
        
        # 标记最小值点（蓝色）
        if min_x_list:
            ax.scatter(min_x_list, min_y_list, min_z_list,
                      c='blue', s=200, marker='v',
                      label=f'最小值点 (值={extrema["min_value"]})',
                      edgecolors='darkblue', linewidths=2, zorder=10)
        
            # 添加图例
            if max_x_list or min_x_list:
                ax.legend(loc='upper left')
        
        # 设置标签和标题
        ax.set_xlabel('X (像素)', fontsize=12)
        ax.set_ylabel('Y (像素)', fontsize=12)
        ax.set_zlabel('亮度值', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='亮度值')
        
        # 设置视角
        ax.view_init(elev=30, azim=45)
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D可视化已保存到: {save_path}")
        
        # 显示图表
        if show:
            plt.show()
        else:
            plt.close()
        
        return extrema_info
    
    def visualize_3d_scatter(self, gray_image: np.ndarray = None,
                             downsample: int = 5,
                             colormap: str = None,
                             title: str = '3D Lightness Scatter',
                             save_path: str = None,
                             show: bool = True,
                             mark_extrema: bool = True):
        """
        使用散点图绘制灰度图的三维可视化（适合大图片）
        
        参数:
            gray_image: 灰度图数组，如果为None则使用self.gray_image
            downsample: 下采样因子（默认5）
            colormap: 颜色映射（默认使用self.colormap）
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
            show: 是否显示图表
            mark_extrema: 是否标记极值点（默认True）
        
        返回:
            extrema_info: 极值点信息字典（如果mark_extrema为True）
        """
        if gray_image is None:
            if self.gray_image is None:
                raise ValueError("未加载图片，请先调用load_and_convert_to_grayscale或提供gray_image参数")
            gray_image = self.gray_image
        
        if colormap is None:
            colormap = self.colormap
        
        # 保存原始图像用于查找极值点
        original_image = gray_image.copy()
        
        # 下采样
        if downsample > 1:
            gray_image = gray_image[::downsample, ::downsample]
        
        height, width = gray_image.shape
        
        # 创建坐标和亮度值数组
        x = np.arange(0, width)
        y = np.arange(0, height)
        X, Y = np.meshgrid(x, y)
        
        # 展平数组
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = gray_image.flatten()
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制散点图
        scatter = ax.scatter(X_flat, Y_flat, Z_flat,
                            c=Z_flat,
                            cmap=colormap,
                            s=1,
                            alpha=0.6)
        
        # 查找并标记极值点
        extrema_info = None
        if mark_extrema:
            extrema = self.find_extrema_points(original_image)
            extrema_info = extrema
        
        # 将原始坐标转换为下采样后的坐标
        max_x_list = []
        max_y_list = []
        max_z_list = []
        min_x_list = []
        min_y_list = []
        min_z_list = []
        
        for y_orig, x_orig in extrema['max_positions']:
            # 计算下采样后的坐标
            x_down = x_orig // downsample
            y_down = y_orig // downsample
            # 确保坐标在范围内
            if 0 <= x_down < width and 0 <= y_down < height:
                max_x_list.append(x_down)
                max_y_list.append(y_down)
                max_z_list.append(gray_image[y_down, x_down])
        
        for y_orig, x_orig in extrema['min_positions']:
            # 计算下采样后的坐标
            x_down = x_orig // downsample
            y_down = y_orig // downsample
            # 确保坐标在范围内
            if 0 <= x_down < width and 0 <= y_down < height:
                min_x_list.append(x_down)
                min_y_list.append(y_down)
                min_z_list.append(gray_image[y_down, x_down])
        
        # 标记最大值点（红色）
        if max_x_list:
            ax.scatter(max_x_list, max_y_list, max_z_list,
                      c='red', s=300, marker='^', 
                      label=f'最大值点 (值={extrema["max_value"]})',
                      edgecolors='darkred', linewidths=2, zorder=10)
        
        # 标记最小值点（蓝色）
        if min_x_list:
            ax.scatter(min_x_list, min_y_list, min_z_list,
                      c='blue', s=300, marker='v',
                      label=f'最小值点 (值={extrema["min_value"]})',
                      edgecolors='darkblue', linewidths=2, zorder=10)
        
            # 添加图例
            if max_x_list or min_x_list:
                ax.legend(loc='upper left')
        
        # 设置标签和标题
        ax.set_xlabel('X (像素)', fontsize=12)
        ax.set_ylabel('Y (像素)', fontsize=12)
        ax.set_zlabel('亮度值', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, label='亮度值')
        
        # 设置视角
        ax.view_init(elev=30, azim=45)
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D可视化已保存到: {save_path}")
        
        # 显示图表
        if show:
            plt.show()
        else:
            plt.close()
        
        return extrema_info


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='将图片转为灰度图并绘制三维亮度可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法：自动保存图片（默认行为，适用于SSH环境）
  python show_lightness_3d.py -i image.jpg
  # 输出文件：image_3d.png（自动生成）
  
  # 指定输出路径
  python show_lightness_3d.py -i image.jpg -o output.png
  
  # 显示图形窗口（本地图形环境）
  python show_lightness_3d.py -i image.jpg --show
  
  # 使用散点图模式（适合大图片）
  python show_lightness_3d.py -i image.jpg --mode scatter
  
  # 自定义下采样和颜色映射
  python show_lightness_3d.py -i image.jpg --downsample 3 --colormap hot
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='输入图片路径')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='输出图片路径（可选，如未指定则自动生成：输入文件名_3d_lightness.png）')
    parser.add_argument('--mode', type=str, default='surface',
                       choices=['surface', 'scatter'],
                       help='可视化模式：surface（表面图）或scatter（散点图，适合大图片）')
    parser.add_argument('--downsample', type=int, default=1,
                       help='下采样因子，用于减少数据点（默认1，即不采样）')
    parser.add_argument('--colormap', type=str, default='viridis',
                       help='颜色映射（默认viridis，可选：viridis, plasma, inferno, hot, cool, gray等）')
    parser.add_argument('--title', type=str, default=None,
                       help='图表标题（默认使用文件名）')
    parser.add_argument('--show', action='store_true',
                       help='显示图表窗口（默认不显示，仅保存图片，适用于SSH环境）')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示图表（仅保存），这是默认行为')
    parser.add_argument('--no-mark-extrema', action='store_true',
                       help='不标记极值点')
    
    args = parser.parse_args()
    
    try:
        # 创建可视化器实例
        visualizer = Lightness3DVisualizer(colormap=args.colormap)
        
        # 加载并转换图片
        print(f"正在加载图片: {args.input}")
        gray_image = visualizer.load_and_convert_to_grayscale(args.input)
        print(f"图片尺寸: {gray_image.shape[1]}x{gray_image.shape[0]}")
        print(f"亮度值范围: {gray_image.min()} - {gray_image.max()}")
        
        # 查找极值点
        extrema = visualizer.find_extrema_points()
        
        # 输出极值点信息
        print("\n" + "="*60)
        print("极值点信息:")
        print("="*60)
        print(f"最大值: {extrema['max_value']}")
        print(f"最大值点数量: {len(extrema['max_positions'])}")
        if len(extrema['max_positions']) > 0:
            print("最大值点坐标 (x, y):")
            for i, (y, x) in enumerate(extrema['max_positions'][:10], 1):  # 最多显示10个
                print(f"  {i}. ({x}, {y})")
            if len(extrema['max_positions']) > 10:
                print(f"  ... 还有 {len(extrema['max_positions']) - 10} 个最大值点")
        
        print(f"\n最小值: {extrema['min_value']}")
        print(f"最小值点数量: {len(extrema['min_positions'])}")
        if len(extrema['min_positions']) > 0:
            print("最小值点坐标 (x, y):")
            for i, (y, x) in enumerate(extrema['min_positions'][:10], 1):  # 最多显示10个
                print(f"  {i}. ({x}, {y})")
            if len(extrema['min_positions']) > 10:
                print(f"  ... 还有 {len(extrema['min_positions']) - 10} 个最小值点")
        print("="*60 + "\n")
        
        # 确定标题
        if args.title is None:
            title = f'3D Lightness Visualization: {Path(args.input).name}'
        else:
            title = args.title
        
        # 如果没有指定输出路径，自动生成一个
        output_path = args.output
        if output_path is None:
            input_path = Path(args.input)
            output_path = input_path.parent / f"{input_path.stem}_3d.png"
            print(f"未指定输出路径，将保存到: {output_path}")
        
        # 根据模式选择可视化方法
        mark_extrema = not args.no_mark_extrema
        extrema_info = None
        
        # 确定是否显示图形：默认不显示（适用于SSH环境），除非明确指定--show
        if args.show:
            show_graph = True
        elif args.no_show:
            show_graph = False
        else:
            # 默认不显示，除非有DISPLAY环境变量（本地图形环境）
            show_graph = os.getenv('DISPLAY') is not None
        
        if args.mode == 'surface':
            # 对于大图片，自动调整下采样
            if args.downsample == 1 and gray_image.size > 500000:
                auto_downsample = max(2, int(np.sqrt(gray_image.size / 500000)))
                print(f"图片较大，自动下采样因子: {auto_downsample}")
                extrema_info = visualizer.visualize_3d_lightness(
                    downsample=auto_downsample,
                    colormap=args.colormap,
                    title=title,
                    save_path=str(output_path),
                    show=show_graph,
                    mark_extrema=mark_extrema
                )
            else:
                extrema_info = visualizer.visualize_3d_lightness(
                    downsample=args.downsample,
                    colormap=args.colormap,
                    title=title,
                    save_path=str(output_path),
                    show=show_graph,
                    mark_extrema=mark_extrema
                )
        else:  # scatter mode
            # 散点图模式默认使用更大的下采样
            downsample = args.downsample if args.downsample > 1 else 5
            extrema_info = visualizer.visualize_3d_scatter(
                downsample=downsample,
                colormap=args.colormap,
                title=title,
                save_path=str(output_path),
                show=show_graph,
                mark_extrema=mark_extrema
            )
        
        print("\n✓ 可视化完成!")
        return 0
    
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

