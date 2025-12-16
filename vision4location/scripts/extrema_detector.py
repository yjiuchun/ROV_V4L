#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极值点检测类

功能：
    - 输入彩色或灰度图像，输出灰度图的最大/最小值及其坐标
    - 支持直接传入 numpy 数组或文件路径

用法示例（代码）：
    detector = ExtremaDetector()
    extrema = detector.detect("image.jpg")
    print(extrema["max_value"], extrema["max_positions"])

命令行示例：
    python extrema_detector.py --input image.jpg
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Tuple


class ExtremaDetector:
    """灰度极值检测器。"""

    def __init__(self):
        pass

    def _to_gray(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """读取图像（若为路径）并转换为灰度图。"""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"无法读取图像: {image}")
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise TypeError(f"不支持的类型: {type(image)}")

        if img.ndim == 2:
            gray = img
        elif img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"不支持的图像维度: {img.shape}")
        return gray

    def detect(self, image: Union[str, Path, np.ndarray]) -> Dict[str, Union[int, Tuple[float, float]]]:
        """
        检测极值点。
        返回：
            {
                "max_value": int,
                "min_value": int,
                "max_positions": Tuple[float, float],  # (y, x) 均值坐标
                "min_positions": Tuple[float, float]   # (y, x) 均值坐标
            }
        """
        gray = self._to_gray(image)

        max_value = int(gray.max())
        min_value = int(gray.min())

        max_mask = gray == max_value
        min_mask = gray == min_value

        max_positions_array = np.argwhere(max_mask)  # (y, x)
        min_positions_array = np.argwhere(min_mask)

        # 计算坐标均值
        if len(max_positions_array) > 0:
            max_positions = tuple(np.mean(max_positions_array, axis=0).tolist())
        else:
            max_positions = (0.0, 0.0)

        if len(min_positions_array) > 0:
            min_positions = tuple(np.mean(min_positions_array, axis=0).tolist())
        else:
            min_positions = (0.0, 0.0)

        return {
            "max_value": max_value,
            "min_value": min_value,
            "max_positions": max_positions,
            "min_positions": min_positions,
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="检测灰度极值点")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入图像路径")
    parser.add_argument("--show", action="store_true", help="显示灰度图并标记极值点")
    parser.add_argument("--max-only", action="store_true", help="仅显示最大值点")
    parser.add_argument("--min-only", action="store_true", help="仅显示最小值点")
    args = parser.parse_args()

    detector = ExtremaDetector()
    extrema = detector.detect(args.input)

    print(f"最大值: {extrema['max_value']}, 位置: {extrema['max_positions']}")
    print(f"最小值: {extrema['min_value']}, 位置: {extrema['min_positions']}")

    if args.show:
        gray = detector._to_gray(args.input)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        draw_max = not args.min_only
        draw_min = not args.max_only

        if draw_max:
            y, x = extrema["max_positions"]
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)  # red for max
        if draw_min:
            y, x = extrema["min_positions"]
            cv2.circle(vis, (int(x), int(y)), 2, (255, 0, 0), -1)  # blue for min

        cv2.imshow("Extrema (red=max, blue=min)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

