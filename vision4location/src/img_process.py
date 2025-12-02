# 修复conda环境中OpenCV的Qt插件问题（必须在导入cv2之前）
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
try:
    import opencv_qt_fix  # 自动修复Qt问题
except ImportError:
    # 如果找不到修复模块，直接设置环境变量
    if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
        plugin_path = os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
        if 'cv2/qt/plugins' in plugin_path:
            paths = plugin_path.split(os.pathsep)
            paths = [p for p in paths if 'cv2/qt/plugins' not in p]
            if paths:
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.pathsep.join(paths)
            else:
                system_qt_path = '/usr/lib/x86_64-linux-gnu/qt5/plugins'
                if os.path.exists(system_qt_path):
                    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = system_qt_path
                else:
                    del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

import cv2
import numpy as np
from typing import Union, List, Tuple, Optional
from pathlib import Path
from ultralytics import YOLO
import sys
import time

# YOLOv11目标检测路径
yolo_dir = '/home/yjc/Project/rov_ws/src/vision4location/tracking/YOLOv11'
if yolo_dir not in sys.path:
    sys.path.insert(0, yolo_dir)
# detection路径
detection_dir = '/home/yjc/Project/rov_ws/src/vision4location/detection'
if detection_dir not in sys.path:
    sys.path.insert(0, detection_dir)
# location路径
location_dir = '/home/yjc/Project/rov_ws/src/vision4location/location'
if location_dir not in sys.path:
    sys.path.insert(0, location_dir)
# KalmanFilter路径
kalmanfilter_dir = '/home/yjc/Project/rov_ws/src/vision4location/KalmanFilter'
if kalmanfilter_dir not in sys.path:
    sys.path.insert(0, kalmanfilter_dir)
enhance_dir = '/home/yjc/Project/rov_ws/src/vision4location/enhance'
if enhance_dir not in sys.path:
    sys.path.insert(0, enhance_dir)

from yolov11_tarctor import YOLOv11Detector
from self_lightness import SelfLightness
from MdeiaFilter import MediaFilter
class ImgProcess:
    def __init__(self,model_path="yolo11n.pt"):

        self.yolo_detector = YOLOv11Detector(model_path=model_path)     # YOLOv11目标检测
        self.self_lightness = SelfLightness(show_image=False)          # 照度检测
        self.media_filter = MediaFilter()                              # 图像滤波
        
    def GetRoI(self, img):
        start_time = time.time()
        results,box,x_offset = self.yolo_detector.detect(img)
        self.yolo_detector.visualize(img, detections=results)
        # print(results[0]['bbox'])
        x1,y1,x2,y2 = int(results[0]['bbox'][0]),int(results[0]['bbox'][1]),int(results[0]['bbox'][2]),int(results[0]['bbox'][3])
        crop_img = img[y1:y2, x1:x2]
        end_time = time.time()
        duration = end_time - start_time
        return box,x_offset,duration,crop_img,results
    def selfLightness(self,img):
        return self.self_lightness.get_histogram(img)
    def mediaFilter(self,img):
        return self.media_filter.filter(img)