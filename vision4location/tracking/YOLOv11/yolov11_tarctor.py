#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11目标检测类

功能：
    - 传入图像进行目标检测
    - 输出检测框的四点像素坐标（左上、右上、右下、左下）
    - 输出置信度和类别信息

作者: 杨久春
时间: 2025-11-28
"""

import cv2
import numpy as np
from typing import Union, List, Tuple, Optional
from pathlib import Path
from ultralytics import YOLO
import sys
kf_dir = '/home/yjc/Project/rov_ws/src/vision4location/KalmanFilter/bouding_box'
if kf_dir not in sys.path:
    sys.path.insert(0, kf_dir)
from BoudingBox_kf import BoudingBox_kf

class YOLOv11Detector:
    """
    YOLOv11目标检测类
    
    使用示例:
        detector = YOLOv11Detector(model_path="yolo11n.pt")
        image = cv2.imread("image.jpg")
        results = detector.detect(image)
        for result in results:
            print(f"四点坐标: {result['corners']}")
            print(f"置信度: {result['confidence']}")
    """
    
    def __init__(self, model_path: str = "yolo11n.pt", 
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: Optional[str] = None,
                 imgsz: int = 640):
        """
        初始化YOLOv11检测器
        
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
        self.box=np.eye(1,4)
        
        # 加载模型
        try:
            self.model = YOLO(model_path)
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            raise ValueError(f"无法加载模型 {model_path}: {e}")
    
    def _xyxy_to_corners(self, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        """
        将xyxy格式（左上角和右下角坐标）转换为四点坐标
        
        参数:
            x1, y1: 左上角坐标
            x2, y2: 右下角坐标
        
        返回:
            corners: 四点坐标数组，形状为(4, 2)
                    顺序：左上、右上、右下、左下
        """
        corners = np.array([
            [x1, y1],  # 左上
            [x2, y1],  # 右上
            [x2, y2],  # 右下
            [x1, y2]   # 左下
        ], dtype=np.float32)
        return corners
    
    def detect(self, image: Union[np.ndarray, str, Path],
               conf: Optional[float] = None,
               iou: Optional[float] = None,
               classes: Optional[List[int]] = None) -> List[dict]:
        """
        对输入图像进行目标检测
        
        参数:
            image: 输入图像，可以是：
                  - numpy数组（BGR格式，OpenCV格式）
                  - 图像文件路径（字符串或Path对象）
            conf: 置信度阈值（如果为None，使用初始化时的阈值）
            iou: IoU阈值（如果为None，使用初始化时的阈值）
            classes: 要检测的类别ID列表（None表示检测所有类别）
        
        返回:
            results: 检测结果列表，每个元素包含：
                    - 'corners': 四点坐标数组，形状为(4, 2)，顺序：左上、右上、右下、左下
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
        
        # 使用指定的阈值或默认阈值
        conf_thresh = conf if conf is not None else self.conf_threshold
        iou_thresh = iou if iou is not None else self.iou_threshold
        
        # 执行检测
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
        detection_results = []
        
        for result in results:
            boxes = result.boxes
            
            # 遍历所有检测框
            for i in range(len(boxes)):
                # 获取边界框坐标（xyxy格式）
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = box
                
                # 转换为四点坐标
                corners = self._xyxy_to_corners(x1, y1, x2, y2)
                
                # 获取置信度
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # 获取类别信息
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # 构建结果字典
                detection_result = {
                    'corners': corners,           # 四点坐标 (4, 2)
                    'confidence': confidence,     # 置信度
                    'class_id': class_id,        # 类别ID
                    'class_name': class_name,    # 类别名称
                    'bbox': box.tolist()         # 边界框 [x1, y1, x2, y2]
                }
                bbox = detection_result['bbox']

                x1, y1, x2, y2 = bbox  # 解包：x1=左上角x，y1=左上角y，x2=右下角x，y2=右下角y

                # -------------- 步骤2：计算像素坐标的cx,cy,w,h --------------
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                self.box = np.array([cx, cy, w, h])
                detection_results.append(detection_result)
        
        return detection_results,self.box
    
    def detect_batch(self, images: List[Union[np.ndarray, str, Path]],
                    conf: Optional[float] = None,
                    iou: Optional[float] = None,
                    classes: Optional[List[int]] = None) -> List[List[dict]]:
        """
        批量检测图像
        
        参数:
            images: 图像列表
            conf: 置信度阈值
            iou: IoU阈值
            classes: 类别ID列表
        
        返回:
            results_list: 每个图像对应的检测结果列表
        """
        # 读取所有图像
        img_list = []
        for img in images:
            if isinstance(img, (str, Path)):
                img_array = cv2.imread(str(img))
                if img_array is None:
                    raise ValueError(f"无法读取图像: {img}")
                img_list.append(img_array)
            elif isinstance(img, np.ndarray):
                img_list.append(img.copy())
            else:
                raise TypeError(f"不支持的图像类型: {type(img)}")
        
        # 使用指定的阈值或默认阈值
        conf_thresh = conf if conf is not None else self.conf_threshold
        iou_thresh = iou if iou is not None else self.iou_threshold
        
        # 批量检测
        results = self.model.predict(
            img_list,
            conf=conf_thresh,
            iou=iou_thresh,
            classes=classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )
        
        # 解析所有结果
        all_results = []
        for result in results:
            boxes = result.boxes
            detection_results = []
            
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box
                corners = self._xyxy_to_corners(x1, y1, x2, y2)
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.model.names[class_id]
                
                detection_result = {
                    'corners': corners,
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'bbox': box.tolist()
                }
                detection_results.append(detection_result)
            
            all_results.append(detection_results)
        
        return all_results
    
    def visualize(self, image: Union[np.ndarray, str, Path],
                 detections: Optional[List[dict]] = None,
                 conf: Optional[float] = None,
                 iou: Optional[float] = None,
                 show_labels: bool = True,
                 show_conf: bool = True,
                 line_thickness: int = 2) -> np.ndarray:
        """
        可视化检测结果
        
        参数:
            image: 输入图像
            detections: 检测结果（如果为None，则先进行检测）
            conf: 置信度阈值
            iou: IoU阈值
            show_labels: 是否显示标签
            show_conf: 是否显示置信度
            line_thickness: 线条粗细
        
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
        if detections is None:
            detections = self.detect(image, conf=conf, iou=iou)
        
        # 绘制检测框
        for det in detections:
            corners = det['corners'].astype(int)
            confidence = det['confidence']
            class_name = det['class_name']
            
            # 绘制四点连线形成矩形
            pts = corners.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, 
                         color=(0, 255, 0), thickness=line_thickness)
            
            # 绘制标签和置信度
            if show_labels or show_conf:
                x1, y1 = int(corners[0][0]), int(corners[0][1])
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
                cv2.rectangle(img, (x1, y1 - text_height - baseline - 5),
                            (x1 + text_width, y1), (0, 255, 0), -1)
                
                # 绘制文字
                cv2.putText(img, label, (x1, y1 - baseline - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img
    
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

    
    parser = argparse.ArgumentParser(description='YOLOv11目标检测示例')
    parser.add_argument('--model', type=str, default='/home/yjc/Project/rov_ws/dataset/eight_led/sys_label/YOLODataset/firstTrain/led_ring/weights/best.pt',
                       help='模型文件路径')
    parser.add_argument('--image', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU阈值')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图像路径（可选）')
    parser.add_argument('--show', action='store_true',
                       help='显示结果图像')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = YOLOv11Detector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # 执行检测
    print(f"正在检测图像: {args.image}")
    results = detector.detect(args.image)
    
    print(f"\n检测到 {len(results)} 个目标:")
    for i, result in enumerate(results):
        print(f"\n目标 {i+1}:")
        print(f"  类别: {result['class_name']} (ID: {result['class_id']})")
        print(f"  置信度: {result['confidence']:.4f}")
        print(f"  边界框: {result['bbox']}")
        print(f"  四点坐标:")
        corners = result['corners']
        corner_names = ['左上', '右上', '右下', '左下']
        for j, (name, corner) in enumerate(zip(corner_names, corners)):
            print(f"    {name}: ({corner[0]:.2f}, {corner[1]:.2f})")
    
    # 可视化
    vis_image = detector.visualize(args.image, detections=results)
    
    # 保存结果
    if args.output:
        cv2.imwrite(args.output, vis_image)
        print(f"\n结果已保存到: {args.output}")
    
    # 显示结果
    if args.show:
        cv2.imshow('Detection Results', vis_image)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # 示例用法：
    # python yolov11_tarctor.py --image path/to/image.jpg
    # python yolov11_tarctor.py --image path/to/image.jpg --conf 0.5 --show
    # python yolov11_tarctor.py --image path/to/image.jpg --output result.jpg
    
    # main()
    import argparse
    self_lightness_dir = '/home/yjc/Project/rov_ws/src/vision4location/detection'
    if self_lightness_dir not in sys.path:
        sys.path.insert(0, self_lightness_dir)
    from self_lightness import SelfLightness
    parser = argparse.ArgumentParser(description='YOLOv11目标检测示例')
    parser.add_argument('--model', type=str, default='/home/yjc/Project/rov_ws/src/vision4location/tracking/YOLOv11/firsttrain_withledring/firsttrainledring/weights/best.pt',
                       help='模型文件路径')
    # parser.add_argument('--image', type=str, required=True,
    #                    help='输入图像路径')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU阈值')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图像路径（可选）')
    parser.add_argument('--show', action='store_true',
                       help='显示结果图像')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = YOLOv11Detector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    bouding_box_kf = BoudingBox_kf()
    # video_path = "/home/yjc/stereo_videos/960540/1L.mp4"
    video_path = "/home/yjc/stereo_videos/960540/2L.mp4"
    cap = cv2.VideoCapture(video_path)
    self_lightness = SelfLightness(show_image=True)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("错误：无法打开视频文件！")
        exit()
    while cap.isOpened():
        # 读取单帧（ret=True表示读取成功，frame为帧数据（BGR格式））
        ret, frame = cap.read()
        
        if not ret:  # 读取完毕（或出错），退出循环
            print("视频读取完毕/出错")
            break
        
        results,box = detector.detect(frame)
        bouding_box_kf.update(box)
        bouding_box_kf.predict()
        cx = bouding_box_kf.x[0]
        cy = bouding_box_kf.x[1]
        w = bouding_box_kf.x[2]
        h = bouding_box_kf.x[3]
        vx = bouding_box_kf.x[4]
        vy = bouding_box_kf.x[5]
        vw = bouding_box_kf.x[6]
        vh = bouding_box_kf.x[7]

        x1 = int(cx-w/2)
        y1 = int(cy-h/2)
        x2 = int(cx+w/2)
        y2 = int(cy+h/2)
        try:
            crop_img = frame[y1:y2, x1:x2]
            # cv2.imshow('crop_img', crop_img)
            # cv2.waitKey(300)
            # gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            feature_points, binary_img = self_lightness.extract_feature_points(crop_img)
            print(feature_points)
            # cv2.imshow('binary_img', binary_img)
            # cv2.waitKey(300)        
        except Exception as e:
            print(e)
            continue


        # crop_img = bouding_box_kf.crop_image(frame)
        # cv2.imshow('crop_img', crop_img)
        # cv2.waitKey(300)

        # print(cx,cy)
        # print(cx,cy,w,h,vx,vy,vw,vh)
        cv2.rectangle(frame, (int(cx-w/2), int(cy-h/2)), (int(cx+w/2), int(cy+h/2)), (255, 0, 0), 5)


        # print(box)
        # break
        # vis_image = detector.visualize(frame, detections=results)

        # cv2.imshow('results', vis_image)
        # cv2.waitKey(300)

    cap.release()
    cv2.destroyAllWindows()
