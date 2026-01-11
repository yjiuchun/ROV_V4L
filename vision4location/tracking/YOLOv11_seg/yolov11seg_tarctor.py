#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11分割检测类

功能：
    - 传入图像进行实例分割
    - 返回分割掩码（mask）
    - 可视化分割结果，将mask和检测框绘制到图片上

作者: 杨久春
时间: 2025-01-XX
"""

import cv2
import numpy as np
from typing import Union, List, Tuple, Optional
from pathlib import Path
from ultralytics import YOLO
import torch
import os 

class YOLOv11SegDetector:
    """
    YOLOv11分割检测类
    
    使用示例:
        detector = YOLOv11SegDetector(model_path="yolo11n-seg.pt")
        image = cv2.imread("image.jpg")
        masks, results = detector.detect(image)
        vis_image = detector.visualize(image, masks, results)
        cv2.imwrite("result.jpg", vis_image)
    """
    
    def __init__(self, model_path: str = "yolo11n-seg.pt", 
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: Optional[str] = None,
                 imgsz: int = 640):
        """
        初始化YOLOv11分割检测器
        
        参数:
            model_path: 模型文件路径（.pt文件）
            conf_threshold: 置信度阈值（默认0.25）
            iou_threshold: IoU阈值，用于NMS（默认0.45）
            device: 设备类型，'cuda'或'cpu'，None表示自动选择
            imgsz: 输入图像尺寸（默认640）
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.imgsz = imgsz
        
        # 加载模型
        try:
            self.model = YOLO(model_path)
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            raise ValueError(f"无法加载模型 {model_path}: {e}")
    
    def detect(self, image: Union[np.ndarray, str, Path],
               conf: Optional[float] = None,
               iou: Optional[float] = None,
               classes: Optional[List[int]] = None) -> Tuple[List[np.ndarray], List[dict]]:
        """
        对输入图像进行分割检测
        
        参数:
            image: 输入图像，可以是：
                  - numpy数组（BGR格式，OpenCV格式）
                  - 图像文件路径（字符串或Path对象）
            conf: 置信度阈值（如果为None，使用初始化时的阈值）
            iou: IoU阈值（如果为None，使用初始化时的阈值）
            classes: 要检测的类别ID列表（None表示检测所有类别）
        
        返回:
            masks: 分割掩码列表，每个元素是一个numpy数组，形状为(H, W)，值为0或1
            results: 检测结果列表，每个元素包含：
                    - 'mask': 掩码数组 (H, W)
                    - 'confidence': 置信度（float）
                    - 'class_id': 类别ID（int）
                    - 'class_name': 类别名称（str）
                    - 'bbox': 边界框 [x1, y1, x2, y2]（左上角和右下角坐标）
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
        
        # 保存原始图像尺寸
        original_h, original_w = img.shape[:2]
        
        # 使用指定的阈值或默认阈值
        conf_thresh = conf if conf is not None else self.conf_threshold
        iou_thresh = iou if iou is not None else self.iou_threshold
        
        # 执行分割检测
        results = self.model.predict(
            img,
            conf=conf_thresh,
            iou=iou_thresh,
            classes=classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )
        
        # 解析检测结果
        masks_list = []
        detection_results = []
        
        for result in results:
            # 检查是否有分割结果
            if result.masks is None:
                # print("警告: 未检测到分割结果")
                return [], []
            
            boxes = result.boxes
            masks = result.masks
            
            # 遍历所有检测结果
            for i in range(len(boxes)):
                # 获取边界框坐标（xyxy格式）
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                
                # 获取置信度
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # 获取类别信息
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # 获取分割掩码
                # masks.data形状为 (num_objects, H, W)，需要转换为原始图像尺寸
                mask_tensor = masks.data[i]  # 形状为 (H, W)
                
                # 将mask转换为numpy数组
                if isinstance(mask_tensor, torch.Tensor):
                    mask_np = mask_tensor.cpu().numpy()
                else:
                    mask_np = np.array(mask_tensor)
                
                # 如果mask尺寸与原始图像不一致，需要调整
                mask_h, mask_w = mask_np.shape
                if mask_h != original_h or mask_w != original_w:
                    # 使用最近邻插值调整mask尺寸
                    mask_np = cv2.resize(mask_np.astype(np.uint8), 
                                        (original_w, original_h), 
                                        interpolation=cv2.INTER_NEAREST)
                
                # 将mask转换为二值图像（0或1）
                mask_binary = (mask_np > 0.5).astype(np.uint8)
                
                # 构建结果字典
                detection_result = {
                    'mask': mask_binary,          # 二值掩码 (H, W)
                    'confidence': confidence,     # 置信度
                    'class_id': class_id,        # 类别ID
                    'class_name': class_name,    # 类别名称
                    'bbox': box.tolist()         # 边界框 [x1, y1, x2, y2]
                }
                
                masks_list.append(mask_binary)
                detection_results.append(detection_result)
        
        return masks_list, detection_results
    
    def mask_to_quadrilateral(self, mask: np.ndarray, 
                              method: str = 'min_area_rect',
                              epsilon_ratio: float = 0.02) -> np.ndarray:
        """
        将掩码拟合为一个四边形，并返回四个点的坐标
        
        参数:
            mask: 二值掩码数组，形状为 (H, W)，值为0或1
            method: 拟合方法，可选：
                   - 'min_area_rect': 使用最小外接旋转矩形（默认）
                   - 'approx_poly': 使用轮廓近似多边形
                   - 'convex_hull': 使用凸包然后简化为四边形
            epsilon_ratio: 轮廓近似的精度比例（仅用于approx_poly方法，默认0.02）
        
        返回:
            corners: 四个点的坐标数组，形状为 (4, 2)，顺序为：
                    - [0]: 左上角 (x, y)
                    - [1]: 右上角 (x, y)
                    - [2]: 右下角 (x, y)
                    - [3]: 左下角 (x, y)
        """
        if mask is None or mask.size == 0:
            raise ValueError("掩码为空")
        
        # 确保掩码是二值图像
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        
        # 找到掩码的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            raise ValueError("掩码中没有有效像素")
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        if method == 'min_area_rect':
            # 方法1: 使用最小外接旋转矩形
            rect = cv2.minAreaRect(largest_contour)
            box_points = cv2.boxPoints(rect)  # 获取旋转矩形的四个角点
            corners = box_points.astype(np.float32)
            
            # 对点进行排序：左上、右上、右下、左下
            corners = self._sort_quadrilateral_points(corners)
            
        elif method == 'approx_poly':
            # 方法2: 使用轮廓近似多边形
            epsilon = epsilon_ratio * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # 如果近似后的点数不是4个，使用凸包
            if len(approx) != 4:
                hull = cv2.convexHull(largest_contour)
                # 尝试将凸包简化为4个点
                if len(hull) > 4:
                    # 使用更大的epsilon来简化
                    epsilon = epsilon_ratio * 2 * cv2.arcLength(hull, True)
                    approx = cv2.approxPolyDP(hull, epsilon, True)
                
                # 如果还是不是4个点，使用最小外接矩形
                if len(approx) != 4:
                    rect = cv2.minAreaRect(largest_contour)
                    box_points = cv2.boxPoints(rect)
                    corners = box_points.astype(np.float32)
                else:
                    corners = approx.reshape(-1, 2).astype(np.float32)
            else:
                corners = approx.reshape(-1, 2).astype(np.float32)
            
            # 对点进行排序
            corners = self._sort_quadrilateral_points(corners)
            
        elif method == 'convex_hull':
            # 方法3: 使用凸包然后简化为四边形
            hull = cv2.convexHull(largest_contour)
            
            # 如果凸包点数大于4，简化为4个点
            if len(hull) > 4:
                epsilon = epsilon_ratio * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                
                # 如果简化后还不是4个点，使用最小外接矩形
                if len(approx) != 4:
                    rect = cv2.minAreaRect(largest_contour)
                    box_points = cv2.boxPoints(rect)
                    corners = box_points.astype(np.float32)
                else:
                    corners = approx.reshape(-1, 2).astype(np.float32)
            else:
                # 如果凸包点数少于4，使用最小外接矩形
                if len(hull) < 4:
                    rect = cv2.minAreaRect(largest_contour)
                    box_points = cv2.boxPoints(rect)
                    corners = box_points.astype(np.float32)
                else:
                    corners = hull.reshape(-1, 2).astype(np.float32)
            
            # 对点进行排序
            corners = self._sort_quadrilateral_points(corners)
        else:
            raise ValueError(f"未知的拟合方法: {method}")
        
        return corners
    
    def _sort_quadrilateral_points(self, points: np.ndarray) -> np.ndarray:
        """
        将四边形的四个点按照顺序排序：左上、右上、右下、左下
        
        参数:
            points: 四个点的坐标数组，形状为 (4, 2)
        
        返回:
            sorted_points: 排序后的四个点坐标
        """
        # 按照y坐标排序，找到上边和下边的点
        y_sorted_indices = np.argsort(points[:, 1])
        top_points = points[y_sorted_indices[:2]]  # y坐标较小的两个点（上边）
        bottom_points = points[y_sorted_indices[2:]]  # y坐标较大的两个点（下边）
        
        # 在上边的点中，按x坐标排序：左、右
        top_x_sorted = top_points[np.argsort(top_points[:, 0])]
        top_left = top_x_sorted[0]
        top_right = top_x_sorted[1]
        
        # 在下边的点中，按x坐标排序：左、右
        bottom_x_sorted = bottom_points[np.argsort(bottom_points[:, 0])]
        bottom_left = bottom_x_sorted[0]
        bottom_right = bottom_x_sorted[1]
        
        # 组合为：左上、右上、右下、左下
        sorted_points = np.array([
            top_left,
            top_right,
            bottom_right,
            bottom_left
        ], dtype=np.float32)
        
        return sorted_points
    
    def mask_to_square(self, mask: np.ndarray, 
                      use_centroid: bool = True,
                      padding_ratio: float = 0.1) -> np.ndarray:
        """
        将掩码拟合为一个正方形，并返回四个点的坐标（保留向后兼容）
        
        参数:
            mask: 二值掩码数组，形状为 (H, W)，值为0或1
            use_centroid: 是否使用掩码的质心作为正方形中心（True），
                         否则使用边界框的中心（False）
            padding_ratio: 在原始尺寸基础上增加的填充比例（默认0.1，即10%）
        
        返回:
            corners: 四个点的坐标数组，形状为 (4, 2)，顺序为：
                    - [0]: 左上角 (x, y)
                    - [1]: 右上角 (x, y)
                    - [2]: 右下角 (x, y)
                    - [3]: 左下角 (x, y)
        """
        # 调用四边形拟合方法，然后转换为正方形
        corners = self.mask_to_quadrilateral(mask, method='min_area_rect')
        
        # 计算中心点和平均边长
        center = corners.mean(axis=0)
        distances = np.linalg.norm(corners - center, axis=1)
        avg_size = np.mean(distances) * 2
        
        # 添加填充
        square_size = avg_size * (1 + padding_ratio)
        half_size = square_size / 2.0
        
        # 如果使用质心，使用掩码的质心；否则使用四边形的中心
        if use_centroid:
            moments = cv2.moments(mask)
            if moments["m00"] != 0:
                center_x = moments["m10"] / moments["m00"]
                center_y = moments["m01"] / moments["m00"]
                center = np.array([center_x, center_y])
        
        # 计算正方形的四个角点
        corners = np.array([
            [center[0] - half_size, center[1] - half_size],  # 左上
            [center[0] + half_size, center[1] - half_size],  # 右上
            [center[0] + half_size, center[1] + half_size],  # 右下
            [center[0] - half_size, center[1] + half_size]   # 左下
        ], dtype=np.float32)
        
        return corners
    
    def masks_to_quadrilaterals(self, masks: List[np.ndarray],
                                method: str = 'min_area_rect',
                                epsilon_ratio: float = 0.02) -> List[np.ndarray]:
        """
        将多个掩码拟合为四边形，返回每个掩码对应的四个点坐标
        
        参数:
            masks: 掩码列表，每个元素是二值掩码数组
            method: 拟合方法，可选：'min_area_rect', 'approx_poly', 'convex_hull'
            epsilon_ratio: 轮廓近似的精度比例（仅用于approx_poly方法）
        
        返回:
            corners_list: 四个点坐标的列表，每个元素形状为 (4, 2)
        """
        corners_list = []
        for mask in masks:
            try:
                corners = self.mask_to_quadrilateral(mask, method, epsilon_ratio)
                corners_list.append(corners)
            except Exception as e:
                print(f"警告: 处理掩码时出错: {e}")
                # 返回一个默认的四边形（如果处理失败）
                corners_list.append(np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32))
        
        return corners_list
    
    def masks_to_squares(self, masks: List[np.ndarray],
                        use_centroid: bool = True,
                        padding_ratio: float = 0.1) -> List[np.ndarray]:
        """
        将多个掩码拟合为正方形，返回每个掩码对应的四个点坐标（保留向后兼容）
        
        参数:
            masks: 掩码列表，每个元素是二值掩码数组
            use_centroid: 是否使用掩码的质心作为正方形中心
            padding_ratio: 填充比例
        
        返回:
            corners_list: 四个点坐标的列表，每个元素形状为 (4, 2)
        """
        corners_list = []
        for mask in masks:
            try:
                corners = self.mask_to_square(mask, use_centroid, padding_ratio)
                corners_list.append(corners)
            except Exception as e:
                print(f"警告: 处理掩码时出错: {e}")
                # 返回一个默认的正方形（如果处理失败）
                corners_list.append(np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32))
        
        return corners_list
    
    def visualize(self, image: Union[np.ndarray, str, Path],
                 masks: Optional[List[np.ndarray]] = None,
                 results: Optional[List[dict]] = None,
                 conf: Optional[float] = None,
                 iou: Optional[float] = None,
                 show_labels: bool = False,
                 show_conf: bool = True,
                 show_boxes: bool = False,
                 show_masks: bool = True,
                 show_squares: bool = True,
                 mask_alpha: float = 0.5,
                 line_thickness: int = 2,
                 square_line_thickness: int = 2,
                 mask_colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
        """
        可视化分割结果
        
        参数:
            image: 输入图像
            masks: 掩码列表（如果为None，则先进行检测）
            results: 检测结果列表（如果为None，则先进行检测）
            conf: 置信度阈值
            iou: IoU阈值
            show_labels: 是否显示标签
            show_conf: 是否显示置信度
            show_boxes: 是否显示边界框
            show_masks: 是否显示分割掩码
            show_squares: 是否显示拟合的四边形
            mask_alpha: 掩码透明度（0.0-1.0）
            line_thickness: 边界框线条粗细
            square_line_thickness: 四边形线条粗细
            mask_colors: 掩码颜色列表，如果为None则自动生成
        
        返回:
            vis_image: 可视化后的图像
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
        
        # 如果没有提供检测结果，先进行检测
        if masks is None or results is None:
            masks, results = self.detect(image, conf=conf, iou=iou)
        
        # 如果没有检测结果，返回原图
        if len(results) == 0:
            return img
        
        # 生成掩码颜色（如果未提供）
        if mask_colors is None:
            mask_colors = self._generate_colors(len(masks))
        elif len(mask_colors) < len(masks):
            # 如果提供的颜色不够，补充生成
            additional_colors = self._generate_colors(len(masks) - len(mask_colors))
            mask_colors.extend(additional_colors)
        
        # 创建可视化图像
        vis_image = img.copy()
        
        # 绘制掩码
        if show_masks:
            for i, (mask, color) in enumerate(zip(masks, mask_colors)):
                # 创建彩色掩码
                mask_colored = np.zeros_like(vis_image)
                mask_colored[mask > 0] = color
                
                # 叠加掩码到图像上
                vis_image = cv2.addWeighted(vis_image, 1.0, mask_colored, mask_alpha, 0)
        
        # 绘制边界框和标签
        if show_boxes:
            for i, result in enumerate(results):
                bbox = result['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                confidence = result['confidence']
                class_name = result['class_name']
                
                # 获取对应的颜色
                color = mask_colors[i] if i < len(mask_colors) else (0, 255, 0)
                
                # 绘制边界框
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, line_thickness)
                
                # 绘制标签和置信度
                if show_labels or show_conf:
                    label = ""
                    if show_labels:
                        label += class_name
                    if show_conf:
                        if label:
                            label += f" {confidence:.2f}"
                        else:
                            label = f"{confidence:.2f}"
                    
                    # 计算文字大小
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    # 绘制文字背景
                    cv2.rectangle(vis_image, 
                                (x1, y1 - text_height - baseline - 5),
                                (x1 + text_width, y1), 
                                color, -1)
                    
                    # 绘制文字
                    cv2.putText(vis_image, label, (x1, y1 - baseline - 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 绘制拟合的四边形
        if show_squares and masks:
            quad_corners_list = self.masks_to_quadrilaterals(masks)
            for i, corners in enumerate(quad_corners_list):
                # 获取对应的颜色
                color = mask_colors[i] if i < len(mask_colors) else (255, 0, 255)  # 默认使用洋红色
                
                # 将角点转换为整数
                corners_int = corners.astype(int)
                
                # 绘制四边形（闭合多边形）
                cv2.polylines(vis_image, [corners_int], isClosed=True, 
                             color=color, thickness=square_line_thickness)
                
                # 可选：绘制四个角点
                for corner in corners_int:
                    cv2.circle(vis_image, tuple(corner), 3, color, -1)
        
        return vis_image
    
    def visualize_quadrilaterals(self, image: Union[np.ndarray, str, Path],
                                 masks: Optional[List[np.ndarray]] = None,
                                 results: Optional[List[dict]] = None,
                                 conf: Optional[float] = None,
                                 iou: Optional[float] = None,
                                 method: str = 'min_area_rect',
                                 epsilon_ratio: float = 0.02,
                                 line_thickness: int = 2,
                                 show_labels: bool = True) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        可视化拟合的四边形
        
        参数:
            image: 输入图像
            masks: 掩码列表（如果为None，则先进行检测）
            results: 检测结果列表（如果为None，则先进行检测）
            conf: 置信度阈值
            iou: IoU阈值
            method: 拟合方法，可选：'min_area_rect', 'approx_poly', 'convex_hull'
            epsilon_ratio: 轮廓近似的精度比例（仅用于approx_poly方法）
            line_thickness: 线条粗细
            show_labels: 是否显示标签
        
        返回:
            vis_image: 可视化后的图像
            quad_corners_list: 四边形四个点坐标的列表
        """
        # 如果没有提供检测结果，先进行检测
        if masks is None or results is None:
            masks, results = self.detect(image, conf=conf, iou=iou)
        
        # 读取图像
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"无法读取图像: {image}")
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")
        
        # 如果没有检测结果，返回原图
        if len(results) == 0:
            return img, []
        
        # 拟合四边形
        quad_corners_list = self.masks_to_quadrilaterals(masks, method, epsilon_ratio)
        
        # 生成颜色
        colors = self._generate_colors(len(masks))
        
        # 创建可视化图像
        vis_image = img.copy()
        
        # 绘制四边形
        for i, (corners, result, color) in enumerate(zip(quad_corners_list, results, colors)):
            # 将角点转换为整数
            corners_int = corners.astype(int)
            
            # 绘制四边形（闭合多边形）
            cv2.polylines(vis_image, [corners_int], isClosed=True, 
                         color=color, thickness=line_thickness)
            
            # 绘制四个角点
            for corner in corners_int:
                cv2.circle(vis_image, tuple(corner), 4, color, -1)
            
            # 显示标签
            if show_labels:
                # 使用四边形的左上角作为标签位置
                x, y = corners_int[0]
                label = f"{result['class_name']} {result['confidence']:.2f}"
                
                # 计算文字大小
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # 绘制文字背景
                cv2.rectangle(vis_image, 
                            (x, y - text_height - baseline - 5),
                            (x + text_width, y), 
                            color, -1)
                
                # 绘制文字
                cv2.putText(vis_image, label, (x, y - baseline - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return vis_image, quad_corners_list
    
    def visualize_squares(self, image: Union[np.ndarray, str, Path],
                         masks: Optional[List[np.ndarray]] = None,
                         results: Optional[List[dict]] = None,
                         conf: Optional[float] = None,
                         iou: Optional[float] = None,
                         use_centroid: bool = True,
                         padding_ratio: float = 0.1,
                         line_thickness: int = 2,
                         show_labels: bool = True) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        可视化拟合的正方形（保留向后兼容，实际调用visualize_quadrilaterals）
        
        参数:
            image: 输入图像
            masks: 掩码列表（如果为None，则先进行检测）
            results: 检测结果列表（如果为None，则先进行检测）
            conf: 置信度阈值
            iou: IoU阈值
            use_centroid: 是否使用掩码的质心作为正方形中心
            padding_ratio: 填充比例
            line_thickness: 线条粗细
            show_labels: 是否显示标签
        
        返回:
            vis_image: 可视化后的图像
            square_corners_list: 正方形四个点坐标的列表
        """
        # 调用四边形可视化方法，但使用正方形拟合
        vis_image, quad_corners = self.visualize_quadrilaterals(
            image, masks, results, conf, iou, 
            method='min_area_rect', epsilon_ratio=0.02,
            line_thickness=line_thickness, show_labels=show_labels
        )
        # 转换为正方形
        square_corners_list = []
        for mask in masks:
            try:
                corners = self.mask_to_square(mask, use_centroid, padding_ratio)
                square_corners_list.append(corners)
            except:
                square_corners_list.append(quad_corners[0] if quad_corners else np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32))
        return vis_image, square_corners_list
    
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """
        生成不同颜色的列表
        
        参数:
            num_colors: 需要生成的颜色数量
        
        返回:
            colors: 颜色列表，每个颜色为BGR格式的元组
        """
        colors = []
        np.random.seed(42)  # 固定随机种子，确保颜色一致
        
        for i in range(num_colors):
            # 生成HSV颜色空间中的颜色，确保颜色鲜艳
            hue = int(180 * i / num_colors)
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color_bgr)))
        
        return colors
    
    def get_class_names(self) -> dict:
        """
        获取类别名称字典
        
        返回:
            class_names: 类别ID到类别名称的映射字典
        """
        return self.model.names.copy()


def main():
    """示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv11分割检测示例')
    parser.add_argument('--model', type=str, 
                       default='/home/yjc/Project/rov_ws/src/vision4location/tracking/YOLOv11_seg/seg_second_train/led_sys_seg/weights/best.pt',
                       help='模型文件路径')
    parser.add_argument('--image', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--conf', type=float, default=0.45,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU阈值')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图像路径（如果不指定，则显示图像）')
    parser.add_argument('--mask-alpha', type=float, default=0.5,
                       help='掩码透明度')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = YOLOv11SegDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    video_path = "/home/yjc/Project/rov_ws/underwater_dataset/videos/first_capture/stereo_capture_left_20251212_131140.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("错误：无法打开视频文件！")
        exit()
    img_name_count = 0
    while cap.isOpened():
        # 读取单帧（ret=True表示读取成功，frame为帧数据（BGR格式））
        ret, frame = cap.read()
        if not ret:  # 读取完毕（或出错），退出循环
            print("视频读取完毕/出错")
            break

        masks, results = detector.detect(frame, conf=args.conf, iou=args.iou)
        vis_image = detector.visualize(frame, masks, results, mask_alpha=args.mask_alpha)

        if len(masks) > 0:  
            img_name_count += 1
            quad_corners = detector.mask_to_quadrilateral(masks[0], method='min_area_rect')
            cv2.drawContours(vis_image, [quad_corners.astype(int)], -1, (0, 0, 255), 2)
            # cv2.imshow("result", vis_image)
            cv2.imwrite(f"/home/yjc/Project/rov_ws/src/vision4location/tracking/YOLOv11_seg/test/{img_name_count}.jpg", vis_image)
        else:
            cv2.imshow("result", frame)
        cv2.waitKey(3)

    cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    # main()
    folder_path = "/home/yjc/Project/rov_ws/underwater_dataset/images/first_capture/right"  # Linux/macOS
    detector = YOLOv11SegDetector(
        model_path="/home/yjc/Project/rov_ws/src/vision4location/tracking/YOLOv11_seg/seg_second_train/led_sys_seg/weights/best.pt",
        conf_threshold=0.45,
        iou_threshold=0.45
    )
    for filename in os.listdir(folder_path):
        # 筛选 .jpg / .JPG（大小写兼容）
        if filename.lower().endswith(".jpg"):
            # 拼接完整路径
            full_path = os.path.join(folder_path, filename)
            img = cv2.imread(full_path)
            masks, results = detector.detect(img, conf=0.45, iou=0.45)
            vis_image = detector.visualize(img, masks, results, mask_alpha=0.5)
            cv2.imwrite(f"/home/yjc/Project/rov_ws/src/vision4location/tracking/YOLOv11_seg/right_seg/{filename}", vis_image)
