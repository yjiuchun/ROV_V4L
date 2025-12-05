#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片分割脚本类

功能：
    - 读取图片文件
    - 将图片按照大小分成4份（2x2网格）
    - 保存分割后的图片到指定路径
    - 文件命名为原图片名称+_1,2,3,4

作者: 杨久春
时间: 2025-01-XX
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import os


class ImageSplitter:
    """
    图片分割类
    
    功能：
        - 将图片分成4份（2x2网格）
        - 保存分割后的图片
    """
    
    def __init__(self):
        """初始化图片分割器"""
        self.image = None
        self.image_path = None
        self.image_name = None
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        加载图片
        
        参数:
            image_path: 图片文件路径
        
        返回:
            image: 图片数组
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 读取图片
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"无法读取图片文件: {image_path}")
        
        self.image_path = image_path
        self.image_name = Path(image_path).stem  # 获取文件名（不含扩展名）
        
        return self.image
    
    def split_image(self, image: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        将图片分成4份（2x2网格）
        
        参数:
            image: 图片数组，如果为None则使用self.image
        
        返回:
            split_images: 包含4个子图片的列表，顺序为：
                [左上, 右上, 左下, 右下]
        """
        if image is None:
            if self.image is None:
                raise ValueError("未加载图片，请先调用load_image或提供image参数")
            image = self.image
        
        height, width = image.shape[:2]
        
        # 计算分割点（中间位置）
        mid_x = width // 2
        mid_y = height // 2
        
        # 分割图片：左上、右上、左下、右下
        top_left = image[0:mid_y, 0:mid_x]
        top_right = image[0:mid_y, mid_x:width]
        bottom_left = image[mid_y:height, 0:mid_x]
        bottom_right = image[mid_y:height, mid_x:width]
        
        split_images = [top_left, top_right, bottom_left, bottom_right]
        
        return split_images
    
    def save_split_images(self, 
                         output_dir: str,
                         image: Optional[np.ndarray] = None,
                         base_name: Optional[str] = None,
                         extension: Optional[str] = None) -> List[str]:
        """
        保存分割后的图片
        
        参数:
            output_dir: 输出目录路径
            image: 图片数组，如果为None则使用self.image
            base_name: 基础文件名，如果为None则使用self.image_name
            extension: 文件扩展名（如'jpg', 'png'），如果为None则从原文件获取
        
        返回:
            saved_paths: 保存的文件路径列表
        """
        if image is None:
            if self.image is None:
                raise ValueError("未加载图片，请先调用load_image或提供image参数")
            image = self.image
        
        if base_name is None:
            if self.image_name is None:
                base_name = "split_image"
            else:
                base_name = self.image_name
        
        if extension is None:
            if self.image_path:
                extension = Path(self.image_path).suffix[1:]  # 去掉点号
            else:
                extension = "jpg"
        
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        
        # 分割图片
        split_images = self.split_image(image)
        
        # 保存分割后的图片
        saved_paths = []
        for i, split_img in enumerate(split_images, 1):
            filename = f"{base_name}_{i}.{extension}"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, split_img)
            saved_paths.append(filepath)
            print(f"已保存: {filepath} (尺寸: {split_img.shape[1]}x{split_img.shape[0]})")
        
        return saved_paths
    
    def split_and_save(self, 
                      image_path: str,
                      output_dir: str) -> List[str]:
        """
        加载图片、分割并保存（一步完成）
        
        参数:
            image_path: 输入图片路径
            output_dir: 输出目录路径
        
        返回:
            saved_paths: 保存的文件路径列表
        """
        # 加载图片
        self.load_image(image_path)
        
        # 分割并保存
        saved_paths = self.save_split_images(output_dir)
        
        return saved_paths
    
    def composite_image(self, 
                       top_left: np.ndarray,
                       top_right: np.ndarray,
                       bottom_left: np.ndarray,
                       bottom_right: np.ndarray) -> np.ndarray:
        """
        将四个子图像拼接成完整的图像（2x2网格）
        
        参数:
            top_left: 左上角图像
            top_right: 右上角图像
            bottom_left: 左下角图像
            bottom_right: 右下角图像
        
        返回:
            composite: 拼接后的完整图像
        """
        # 检查输入图像是否有效
        if top_left is None or top_right is None or bottom_left is None or bottom_right is None:
            raise ValueError("所有四个输入图像不能为None")
        
        # 检查图像维度
        images = [top_left, top_right, bottom_left, bottom_right]
        for i, img in enumerate(images):
            if len(img.shape) < 2:
                raise ValueError(f"图像 {i} 维度无效")
        
        # 获取图像尺寸
        tl_h, tl_w = top_left.shape[:2]
        tr_h, tr_w = top_right.shape[:2]
        bl_h, bl_w = bottom_left.shape[:2]
        br_h, br_w = bottom_right.shape[:2]
        
        # 检查上下行的宽度是否匹配
        if tl_w != bl_w:
            raise ValueError(f"左上和左下图像宽度不匹配: {tl_w} vs {bl_w}")
        if tr_w != br_w:
            raise ValueError(f"右上和右下图像宽度不匹配: {tr_w} vs {br_w}")
        
        # 检查左右列的高度是否匹配
        if tl_h != tr_h:
            raise ValueError(f"左上和右上图像高度不匹配: {tl_h} vs {tr_h}")
        if bl_h != br_h:
            raise ValueError(f"左下和右下图像高度不匹配: {bl_h} vs {br_h}")
        
        # 检查图像通道数是否一致
        channels = [img.shape[2] if len(img.shape) == 3 else 1 for img in images]
        if len(set(channels)) > 1:
            # 如果通道数不一致，统一转换为灰度图或BGR
            max_channels = max(channels)
            if max_channels == 1:
                # 全部转为灰度图
                top_left = cv2.cvtColor(top_left, cv2.COLOR_BGR2GRAY) if len(top_left.shape) == 3 else top_left
                top_right = cv2.cvtColor(top_right, cv2.COLOR_BGR2GRAY) if len(top_right.shape) == 3 else top_right
                bottom_left = cv2.cvtColor(bottom_left, cv2.COLOR_BGR2GRAY) if len(bottom_left.shape) == 3 else bottom_left
                bottom_right = cv2.cvtColor(bottom_right, cv2.COLOR_BGR2GRAY) if len(bottom_right.shape) == 3 else bottom_right
            else:
                # 全部转为BGR
                if len(top_left.shape) == 2:
                    top_left = cv2.cvtColor(top_left, cv2.COLOR_GRAY2BGR)
                if len(top_right.shape) == 2:
                    top_right = cv2.cvtColor(top_right, cv2.COLOR_GRAY2BGR)
                if len(bottom_left.shape) == 2:
                    bottom_left = cv2.cvtColor(bottom_left, cv2.COLOR_GRAY2BGR)
                if len(bottom_right.shape) == 2:
                    bottom_right = cv2.cvtColor(bottom_right, cv2.COLOR_GRAY2BGR)
        
        # 水平拼接：上排和下排
        top_row = np.hstack([top_left, top_right])
        bottom_row = np.hstack([bottom_left, bottom_right])
        
        # 垂直拼接：上排和下排
        composite = np.vstack([top_row, bottom_row])
        
        return composite


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='将图片分成4份（2x2网格）并保存',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python img_spilt.py -i image.jpg -o output_dir
  
  # 指定输出目录
  python img_spilt.py -i image.jpg -o ./split_images
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='输入图片路径')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='输出目录路径')
    
    args = parser.parse_args()
    
    try:
        # 创建分割器实例
        splitter = ImageSplitter()
        
        # 分割并保存
        print(f"正在处理图片: {args.input}")
        saved_paths = splitter.split_and_save(args.input, args.output)
        
        print(f"\n✓ 完成! 共保存 {len(saved_paths)} 个文件:")
        for path in saved_paths:
            print(f"  - {path}")
        
        return 0
    
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

