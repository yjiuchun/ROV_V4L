#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将文件夹下的图像转换为MP4视频

功能：
    - 读取指定文件夹下的所有图像
    - 按文件名排序后转换为MP4视频
    - 支持自定义帧率、输出路径等参数

作者: 杨久春
时间: 2025-11-28
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from typing import List, Optional
import glob
import re


class ImageToVideoConverter:
    """
    图像转视频转换器
    
    使用示例:
        converter = ImageToVideoConverter()
        converter.convert_folder(
            input_folder="path/to/images",
            output_path="output.mp4",
            fps=30
        )
    """
    
    # 支持的图像格式
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    def __init__(self):
        """初始化转换器"""
        pass
    
    def _natural_sort_key(self, filepath: str) -> tuple:
        """
        生成自然排序的键值，支持按文件名中的数字排序
        
        参数:
            filepath: 文件路径
        
        返回:
            sort_key: 用于排序的元组
        """
        # 获取文件名（不含路径）
        filename = os.path.basename(filepath)
        # 将文件名分割为文本和数字部分
        # 例如: "image_13.jpg" -> ["image_", "13", ".jpg"]
        parts = re.split(r'(\d+)', filename)
        # 将数字部分转换为整数，文本部分保持原样
        sort_key = []
        for part in parts:
            if part.isdigit():
                sort_key.append(int(part))
            else:
                sort_key.append(part.lower())  # 不区分大小写
        return tuple(sort_key)
    
    def get_image_files(self, folder_path: str, 
                       extensions: Optional[List[str]] = None) -> List[str]:
        """
        获取文件夹中的所有图像文件
        
        参数:
            folder_path: 文件夹路径
            extensions: 图像扩展名列表（如果为None，使用默认支持的格式）
        
        返回:
            image_files: 图像文件路径列表（已排序）
        """
        if extensions is None:
            extensions = self.SUPPORTED_FORMATS
        
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"文件夹不存在: {folder_path}")
        
        if not folder.is_dir():
            raise ValueError(f"路径不是文件夹: {folder_path}")
        
        # 获取所有图像文件
        image_files = []
        for ext in extensions:
            # 不区分大小写
            pattern1 = str(folder / f"*{ext}")
            pattern2 = str(folder / f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern1))
            image_files.extend(glob.glob(pattern2))
        
        # 去重并按自然顺序排序（支持数字排序）
        image_files = list(set(image_files))
        image_files.sort(key=self._natural_sort_key)
        
        return image_files
    
    def get_image_size(self, image_path: str) -> tuple:
        """
        获取图像尺寸
        
        参数:
            image_path: 图像路径
        
        返回:
            (width, height): 图像尺寸
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        height, width = img.shape[:2]
        return (width, height)
    
    def convert_folder(self, input_folder: str,
                      output_path: str,
                      fps: float = 30.0,
                      codec: str = 'mp4v',
                      image_size: Optional[tuple] = None,
                      resize_mode: str = 'keep_aspect',
                      sort_by: str = 'name') -> bool:
        """
        将文件夹中的图像转换为视频
        
        参数:
            input_folder: 输入文件夹路径
            output_path: 输出视频路径
            fps: 帧率（默认30.0）
            codec: 视频编码器（默认'mp4v'，可选'XVID', 'H264', 'mp4v'等）
            image_size: 输出视频尺寸 (width, height)，如果为None则使用第一张图像的尺寸
            resize_mode: 图像缩放模式
                        - 'keep_aspect': 保持宽高比，添加黑边（默认）
                        - 'stretch': 拉伸填充
                        - 'crop': 裁剪填充
            sort_by: 排序方式
                    - 'name': 按文件名排序（默认）
                    - 'time': 按修改时间排序
        
        返回:
            success: 是否成功
        """
        # 获取所有图像文件
        print(f"正在扫描文件夹: {input_folder}")
        image_files = self.get_image_files(input_folder)
        
        if len(image_files) == 0:
            print(f"错误: 在文件夹 {input_folder} 中未找到图像文件")
            return False
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 按指定方式排序
        if sort_by == 'time':
            image_files.sort(key=lambda x: os.path.getmtime(x))
            print("按修改时间排序")
        else:
            # 按文件名自然排序（已由get_image_files完成，支持数字排序）
            print("按文件名自然排序（支持数字顺序）")
        
        # 读取第一张图像以确定尺寸
        first_image = cv2.imread(image_files[0])
        if first_image is None:
            print(f"错误: 无法读取第一张图像: {image_files[0]}")
            return False
        
        height, width = first_image.shape[:2]
        
        # 确定输出尺寸
        if image_size is None:
            output_width, output_height = width, height
        else:
            output_width, output_height = image_size
        
        print(f"输入图像尺寸: {width}x{height}")
        print(f"输出视频尺寸: {output_width}x{output_height}")
        print(f"帧率: {fps} fps")
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        if not out.isOpened():
            print(f"错误: 无法创建视频文件: {output_path}")
            return False
        
        # 处理每张图像
        success_count = 0
        for i, image_path in enumerate(image_files):
            img = cv2.imread(image_path)
            if img is None:
                print(f"警告: 无法读取图像 {image_path}，跳过")
                continue
            
            # 调整图像尺寸
            if image_size is not None:
                img = self._resize_image(img, output_width, output_height, resize_mode)
            
            # 写入视频
            out.write(img)
            success_count += 1
            
            # 显示进度
            if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                progress = (i + 1) / len(image_files) * 100
                print(f"进度: {i + 1}/{len(image_files)} ({progress:.1f}%)")
        
        # 释放资源
        out.release()
        
        print(f"\n转换完成!")
        print(f"成功处理: {success_count}/{len(image_files)} 张图像")
        print(f"输出视频: {output_path}")
        
        return True
    
    def _resize_image(self, img: np.ndarray, target_width: int, 
                     target_height: int, mode: str = 'keep_aspect') -> np.ndarray:
        """
        调整图像尺寸
        
        参数:
            img: 输入图像
            target_width: 目标宽度
            target_height: 目标高度
            mode: 缩放模式
        
        返回:
            resized_img: 调整后的图像
        """
        h, w = img.shape[:2]
        
        if mode == 'stretch':
            # 直接拉伸
            resized = cv2.resize(img, (target_width, target_height))
        
        elif mode == 'crop':
            # 裁剪模式：保持宽高比，裁剪多余部分
            scale = max(target_width / w, target_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(img, (new_w, new_h))
            
            # 裁剪中心部分
            start_x = (new_w - target_width) // 2
            start_y = (new_h - target_height) // 2
            resized = resized[start_y:start_y + target_height, 
                            start_x:start_x + target_width]
        
        else:  # keep_aspect
            # 保持宽高比，添加黑边
            scale = min(target_width / w, target_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(img, (new_w, new_h))
            
            # 创建黑色背景
            result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # 计算居中位置
            start_x = (target_width - new_w) // 2
            start_y = (target_height - new_h) // 2
            
            # 将调整后的图像放在中心
            result[start_y:start_y + new_h, start_x:start_x + new_w] = resized
            resized = result
        
        return resized
    
    def convert_with_custom_order(self, image_files: List[str],
                                  output_path: str,
                                  fps: float = 30.0,
                                  codec: str = 'mp4v',
                                  image_size: Optional[tuple] = None,
                                  resize_mode: str = 'keep_aspect') -> bool:
        """
        使用自定义图像列表转换为视频
        
        参数:
            image_files: 图像文件路径列表
            output_path: 输出视频路径
            fps: 帧率
            codec: 视频编码器
            image_size: 输出视频尺寸
            resize_mode: 图像缩放模式
        
        返回:
            success: 是否成功
        """
        if len(image_files) == 0:
            print("错误: 图像文件列表为空")
            return False
        
        # 读取第一张图像以确定尺寸
        first_image = cv2.imread(image_files[0])
        if first_image is None:
            print(f"错误: 无法读取第一张图像: {image_files[0]}")
            return False
        
        height, width = first_image.shape[:2]
        
        # 确定输出尺寸
        if image_size is None:
            output_width, output_height = width, height
        else:
            output_width, output_height = image_size
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        if not out.isOpened():
            print(f"错误: 无法创建视频文件: {output_path}")
            return False
        
        # 处理每张图像
        success_count = 0
        for i, image_path in enumerate(image_files):
            img = cv2.imread(image_path)
            if img is None:
                print(f"警告: 无法读取图像 {image_path}，跳过")
                continue
            
            # 调整图像尺寸
            if image_size is not None:
                img = self._resize_image(img, output_width, output_height, resize_mode)
            
            # 写入视频
            out.write(img)
            success_count += 1
            
            # 显示进度
            if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                progress = (i + 1) / len(image_files) * 100
                print(f"进度: {i + 1}/{len(image_files)} ({progress:.1f}%)")
        
        # 释放资源
        out.release()
        
        print(f"\n转换完成!")
        print(f"成功处理: {success_count}/{len(image_files)} 张图像")
        print(f"输出视频: {output_path}")
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='将文件夹下的图像转换为MP4视频',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python img2video.py -i /path/to/images -o output.mp4
  
  # 指定帧率
  python img2video.py -i /path/to/images -o output.mp4 --fps 24
  
  # 指定输出尺寸
  python img2video.py -i /path/to/images -o output.mp4 --width 1920 --height 1080
  
  # 使用H264编码器
  python img2video.py -i /path/to/images -o output.mp4 --codec H264
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='输入文件夹路径')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='输出视频路径（.mp4文件）')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='视频帧率（默认: 30.0）')
    parser.add_argument('--codec', type=str, default='mp4v',
                       choices=['mp4v', 'XVID', 'H264', 'avc1'],
                       help='视频编码器（默认: mp4v）')
    parser.add_argument('--width', type=int, default=None,
                       help='输出视频宽度（默认: 使用第一张图像的宽度）')
    parser.add_argument('--height', type=int, default=None,
                       help='输出视频高度（默认: 使用第一张图像的高度）')
    parser.add_argument('--resize-mode', type=str, default='keep_aspect',
                       choices=['keep_aspect', 'stretch', 'crop'],
                       help='图像缩放模式（默认: keep_aspect）')
    parser.add_argument('--sort-by', type=str, default='name',
                       choices=['name', 'time'],
                       help='图像排序方式（默认: name）')
    
    args = parser.parse_args()
    
    # 检查输入文件夹
    if not os.path.exists(args.input):
        print(f"错误: 输入文件夹不存在: {args.input}")
        return
    
    if not os.path.isdir(args.input):
        print(f"错误: 输入路径不是文件夹: {args.input}")
        return
    
    # 确定输出尺寸
    image_size = None
    if args.width is not None or args.height is not None:
        if args.width is None or args.height is None:
            print("错误: 必须同时指定宽度和高度")
            return
        image_size = (args.width, args.height)
    
    # 创建转换器
    converter = ImageToVideoConverter()
    
    # 执行转换
    try:
        success = converter.convert_folder(
            input_folder=args.input,
            output_path=args.output,
            fps=args.fps,
            codec=args.codec,
            image_size=image_size,
            resize_mode=args.resize_mode,
            sort_by=args.sort_by
        )
        
        if success:
            print("\n✓ 转换成功完成!")
        else:
            print("\n✗ 转换失败")
            return 1
    
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    # 示例用法：
    # python img2video.py -i /path/to/images -o output.mp4
    # python img2video.py -i /path/to/images -o output.mp4 --fps 24 --width 1920 --height 1080
    # python img2video.py -i /path/to/images -o output.mp4 --codec H264 --resize-mode keep_aspect
    
    exit(main())

