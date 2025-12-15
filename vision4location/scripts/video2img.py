#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将MP4视频转换为图片序列

功能：
    - 读取MP4视频文件
    - 按帧提取图片并保存
    - 支持自定义输出格式、帧间隔、输出目录等参数

作者: 杨久春
时间: 2025-01-XX
"""

import cv2
import os
import argparse
from pathlib import Path
from typing import Optional
import numpy as np


class VideoToImageConverter:
    """
    视频转图片转换器
    
    使用示例:
        converter = VideoToImageConverter()
        converter.convert_video(
            input_video="input.mp4",
            output_folder="output_images",
            image_format="jpg"
        )
    """
    
    # 支持的图像格式
    SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp']
    
    def __init__(self):
        """初始化转换器"""
        pass
    
    def get_video_info(self, video_path: str) -> dict:
        """
        获取视频信息
        
        参数:
            video_path: 视频文件路径
        
        返回:
            info: 包含视频信息的字典
                - fps: 帧率
                - frame_count: 总帧数
                - width: 宽度
                - height: 高度
                - duration: 时长（秒）
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration
        }
    
    def convert_video(self, input_video: str,
                     output_folder: str,
                     image_format: str = 'jpg',
                     frame_interval: int = 1,
                     start_frame: int = 0,
                     end_frame: Optional[int] = None,
                     prefix: str = 'frame',
                     zero_padding: int = 6,
                     resize: Optional[tuple] = None,
                     quality: int = 95) -> bool:
        """
        将视频转换为图片序列
        
        参数:
            input_video: 输入视频路径
            output_folder: 输出文件夹路径
            image_format: 输出图片格式（默认'jpg'，支持jpg, png, bmp等）
            frame_interval: 帧间隔，每隔多少帧提取一张（默认1，即提取所有帧）
            start_frame: 起始帧号（默认0）
            end_frame: 结束帧号（默认None，即到视频末尾）
            prefix: 输出文件名前缀（默认'frame'）
            zero_padding: 文件名中帧号的零填充位数（默认6，即frame_000001.jpg）
            resize: 输出图片尺寸 (width, height)，如果为None则保持原尺寸
            quality: 图片质量（仅对jpg格式有效，1-100，默认95）
        
        返回:
            success: 是否成功
        """
        # 检查输入文件
        if not os.path.exists(input_video):
            print(f"错误: 视频文件不存在: {input_video}")
            return False
        
        # 检查图片格式
        image_format = image_format.lower()
        if image_format not in self.SUPPORTED_FORMATS:
            print(f"错误: 不支持的图片格式: {image_format}")
            print(f"支持的格式: {', '.join(self.SUPPORTED_FORMATS)}")
            return False
        
        # 创建输出文件夹
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 打开视频
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件: {input_video}")
            return False
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频信息:")
        print(f"  文件: {input_video}")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps:.2f} fps")
        print(f"  总帧数: {total_frames}")
        print(f"  时长: {total_frames/fps:.2f} 秒")
        
        # 确定结束帧
        if end_frame is None:
            end_frame = total_frames
        else:
            end_frame = min(end_frame, total_frames)
        
        # 检查参数有效性
        if start_frame >= end_frame:
            print(f"错误: 起始帧({start_frame})必须小于结束帧({end_frame})")
            cap.release()
            return False
        
        if frame_interval < 1:
            print(f"错误: 帧间隔({frame_interval})必须大于等于1")
            cap.release()
            return False
        
        # 计算要提取的帧数
        frames_to_extract = list(range(start_frame, end_frame, frame_interval))
        total_to_extract = len(frames_to_extract)
        
        print(f"\n提取设置:")
        print(f"  起始帧: {start_frame}")
        print(f"  结束帧: {end_frame}")
        print(f"  帧间隔: {frame_interval}")
        print(f"  将提取: {total_to_extract} 张图片")
        if resize:
            print(f"  输出尺寸: {resize[0]}x{resize[1]}")
        print(f"  输出格式: {image_format.upper()}")
        print(f"  输出目录: {output_folder}\n")
        
        # 设置编码参数（仅对jpg有效）
        encode_params = []
        if image_format in ['jpg', 'jpeg']:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif image_format == 'png':
            # PNG压缩级别 0-9，9为最高压缩
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 11)]
        
        # 提取帧
        success_count = 0
        current_frame = 0
        frame_index = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检查是否在提取范围内
            if current_frame >= start_frame and current_frame < end_frame:
                if (current_frame - start_frame) % frame_interval == 0:
                    # 调整尺寸
                    if resize:
                        frame = cv2.resize(frame, resize)
                    
                    # 生成文件名
                    filename = f"{frame_index + 1}.{image_format}"
                    filepath = output_path / filename
                    
                    # 保存图片
                    if cv2.imwrite(str(filepath), frame, encode_params):
                        success_count += 1
                    else:
                        print(f"警告: 无法保存图片: {filepath}")
                    
                    frame_index += 1
                    
                    # 显示进度
                    if frame_index % 10 == 0 or frame_index == total_to_extract:
                        progress = frame_index / total_to_extract * 100
                        print(f"进度: {frame_index}/{total_to_extract} ({progress:.1f}%)")
            
            current_frame += 1
            
            # 如果已经超过结束帧，提前退出
            if current_frame >= end_frame:
                break
        
        # 释放资源
        cap.release()
        
        print(f"\n转换完成!")
        print(f"成功提取: {success_count}/{total_to_extract} 张图片")
        print(f"输出目录: {output_folder}")
        
        return True
    
    def extract_frame(self, video_path: str, frame_number: int,
                     output_path: str, resize: Optional[tuple] = None) -> bool:
        """
        提取视频中的指定帧
        
        参数:
            video_path: 视频文件路径
            frame_number: 帧号（从0开始）
            output_path: 输出图片路径
            resize: 输出图片尺寸 (width, height)，如果为None则保持原尺寸
        
        返回:
            success: 是否成功
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件: {video_path}")
            return False
        
        # 跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        if not ret:
            print(f"错误: 无法读取第 {frame_number} 帧")
            cap.release()
            return False
        
        # 调整尺寸
        if resize:
            frame = cv2.resize(frame, resize)
        
        # 保存图片
        success = cv2.imwrite(output_path, frame)
        
        cap.release()
        
        if success:
            print(f"成功提取第 {frame_number} 帧到: {output_path}")
        else:
            print(f"错误: 无法保存图片: {output_path}")
        
        return success


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='将MP4视频转换为图片序列',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法：提取所有帧
  python video2img.py -i input.mp4 -o output_images
  
  # 每隔10帧提取一张
  python video2img.py -i input.mp4 -o output_images --interval 10
  
  # 提取指定范围的帧
  python video2img.py -i input.mp4 -o output_images --start 100 --end 500
  
  # 指定输出格式和尺寸
  python video2img.py -i input.mp4 -o output_images --format png --width 1920 --height 1080
  
  # 提取单帧
  python video2img.py -i input.mp4 -o frame_100.jpg --frame 100
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='输入视频路径（.mp4文件）')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='输出路径（文件夹或单张图片路径）')
    parser.add_argument('--format', type=str, default='jpg',
                       choices=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'],
                       help='输出图片格式（默认: jpg）')
    parser.add_argument('--interval', type=int, default=1,
                       help='帧间隔，每隔多少帧提取一张（默认: 1，即提取所有帧）')
    parser.add_argument('--start', type=int, default=0,
                       help='起始帧号（默认: 0）')
    parser.add_argument('--end', type=int, default=None,
                       help='结束帧号（默认: None，即到视频末尾）')
    parser.add_argument('--prefix', type=str, default='frame',
                       help='输出文件名前缀（默认: frame）')
    parser.add_argument('--padding', type=int, default=6,
                       help='文件名中帧号的零填充位数（默认: 6）')
    parser.add_argument('--width', type=int, default=None,
                       help='输出图片宽度（默认: 保持原尺寸）')
    parser.add_argument('--height', type=int, default=None,
                       help='输出图片高度（默认: 保持原尺寸）')
    parser.add_argument('--quality', type=int, default=95,
                       help='图片质量，1-100（仅对jpg格式有效，默认: 95）')
    parser.add_argument('--frame', type=int, default=None,
                       help='提取单帧（指定帧号，如果设置此参数，将只提取该帧）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入视频文件不存在: {args.input}")
        return 1
    
    # 创建转换器
    converter = VideoToImageConverter()
    
    # 确定输出尺寸
    resize = None
    if args.width is not None or args.height is not None:
        if args.width is None or args.height is None:
            print("错误: 必须同时指定宽度和高度")
            return 1
        resize = (args.width, args.height)
    
    # 执行转换
    try:
        # 提取单帧模式
        if args.frame is not None:
            success = converter.extract_frame(
                video_path=args.input,
                frame_number=args.frame,
                output_path=args.output,
                resize=resize
            )
        else:
            # 批量提取模式
            success = converter.convert_video(
                input_video=args.input,
                output_folder=args.output,
                image_format=args.format,
                frame_interval=args.interval,
                start_frame=args.start,
                end_frame=args.end,
                prefix=args.prefix,
                zero_padding=args.padding,
                resize=resize,
                quality=args.quality
            )
        
        if success:
            print("\n✓ 转换成功完成!")
            return 0
        else:
            print("\n✗ 转换失败")
            return 1
    
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    # 示例用法：
    # python video2img.py -i input.mp4 -o output_images
    # python video2img.py -i input.mp4 -o output_images --interval 10 --format png
    # python video2img.py -i input.mp4 -o output_images --start 100 --end 500 --width 1920 --height 1080
    # python video2img.py -i input.mp4 -o frame_100.jpg --frame 100
    
    exit(main())

