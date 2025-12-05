#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灰度图峰值检测类

功能：
    - 计算灰度图直方图
    - 检测峰值点
    - 可视化并保存直方图
"""

import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from pathlib import Path


class GetGrayPeaks:
    def __init__(self):
        pass

    def get_gray_peaks(self, image, output_path, image_name=None, save_image=False):
        gray_img = image
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        hist_flat = hist.flatten()  # 将hist转换为1D数组

        # 找出极值对应的强度值
        max_count = np.max(hist)
        min_count = np.min(hist)
        max_intensity = np.argmax(hist)
        min_intensity = np.argmin(hist)

        # 检测所有峰值
        # height: 峰值的最小高度（设为最大值的5%以避免噪声）
        # distance: 峰值之间的最小距离（设为5，避免检测到相邻的微小波动）
        # prominence: 峰值的最小突出度（设为最大值的3%）
        min_height = max_count * 0.05
        min_distance = 5
        min_prominence = max_count * 0.03

        peaks, properties = find_peaks(hist_flat, 
                                    height=min_height,
                                    distance=min_distance,
                                    prominence=min_prominence)

        peak_intensities = peaks
        peak_counts = hist_flat[peaks]
        if save_image:
            # 绘制并保存直方图
            plt.figure(figsize=(10, 6))
            plt.plot(hist, color='black')
            plt.title('Grayscale Intensity Histogram', fontsize=14)
            plt.xlabel('Pixel Intensity Value', fontsize=12)
            plt.ylabel('Pixel Count', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 255])
            # 标注所有峰值点
            if len(peak_intensities) > 0:
                plt.plot(peak_intensities, peak_counts, 'ro', markersize=6, label=f'Peaks ({len(peak_intensities)} found)')
                # 标注每个峰值
                for intensity, count in zip(peak_intensities, peak_counts):
                    plt.annotate(f'I={intensity}\nC={int(count)}', 
                                xy=(intensity, count), 
                                xytext=(intensity + 15, count + max_count*0.05),
                                arrowprops=dict(arrowstyle='->', color='red', lw=1.0, alpha=0.7),
                                fontsize=8, color='red', weight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

            # 标注全局最大值点（用不同颜色区分）
            plt.plot(max_intensity, max_count, 'g*', markersize=12, label=f'Global Max: I={max_intensity}')

            # 标注最小值点（如果最小值不为0，也标注出来）
            if min_count > 0:
                plt.plot(min_intensity, min_count, 'bo', markersize=8, label=f'Min: Intensity={min_intensity}, Count={int(min_count)}')
                plt.annotate(f'Min: I={min_intensity}\nC={int(min_count)}', 
                            xy=(min_intensity, min_count), 
                            xytext=(min_intensity + 20, min_count + max_count*0.05),
                            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                            fontsize=10, color='blue', weight='bold')

            plt.legend(loc='upper right', fontsize=9)

            # 保存直方图
            # 确定图片名称
            if image_name is None:
                # 如果未提供图片名称，使用默认名称
                image_name = 'histogram'
            else:
                # 如果提供了完整路径，提取文件名（不含扩展名）
                image_name = Path(image_name).stem
            
            # 确保输出目录存在
            os.makedirs(output_path, exist_ok=True)
            
            # 生成保存路径：output_path + 图片名称 + _histogram.png
            hist_output_path = os.path.join(output_path, f'{image_name}_histogram.png')
            plt.savefig(hist_output_path, dpi=150, bbox_inches='tight')
            print(f"直方图已保存到: {hist_output_path}")

            plt.close()

            print("处理完成！")

if __name__ == "__main__":
    get_gray_peaks = GetGrayPeaks()
    image_path = '/home/yjc/Project/rov_ws/src/vision4location/src/image_save/crop_img/spilt_cropimg/crop_img_40x42_1.jpg'
    image = cv2.imread(image_path)
    get_gray_peaks.get_gray_peaks(image, 
                                  output_path='/home/yjc/Project/rov_ws/src/vision4location/src/image_save/crop_img/spilt_cropimg',
                                  image_name=image_path,
                                  save_image=True)