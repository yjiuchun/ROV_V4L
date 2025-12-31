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

yoloseg_dir = '/home/yjc/Project/rov_ws/src/vision4location/tracking/YOLOv11_seg'
if yoloseg_dir not in sys.path:
    sys.path.insert(0, yoloseg_dir)
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
scripts_dir = '/home/yjc/Project/rov_ws/src/vision4location/scripts'
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
from extrema_detector import ExtremaDetector
from yolov11_tarctor import YOLOv11Detector
from yolov11seg_tarctor import YOLOv11SegDetector
from self_lightness import SelfLightness
from MdeiaFilter import MediaFilter
class ImgProcess:
    def __init__(self,yolo="yolo11n.pt",yoloseg="yolo11n-seg.pt"):

        self.yolo_detector = YOLOv11Detector(model_path=yolo)     # YOLOv11目标检测
        self.yolo_seg_detector = YOLOv11SegDetector(model_path=yoloseg) # YOLOv11分割检测
        self.self_lightness = SelfLightness(show_image=False)          # 照度检测
        self.media_filter = MediaFilter()                              # 图像滤波
        self.extrema_detector = ExtremaDetector()                      # 极值点检测
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
    def GetRoI_seg(self, img):
        masks, results = self.yolo_seg_detector.detect(img, conf=0.5, iou=0.5)
        vis_image = self.yolo_seg_detector.visualize(img, masks, results, mask_alpha=0.5)
        # cv2.imshow("vis_image", vis_image)
        # cv2.waitKey(0)
        quad_corners = self.yolo_seg_detector.mask_to_quadrilateral(masks[0], method='min_area_rect')
        cv2.drawContours(vis_image, [quad_corners.astype(int)], -1, (0, 0, 255), 2)
        # print(quad_corners)
        points = []
        for i in range(4):
            points.append((int(quad_corners[i][0]), int(quad_corners[i][1])))
        print(points)
        # cv2.imwrite("/home/yjc/Project/rov_ws/src/vision4location/src/image_save/seg/test/result.jpg", vis_image)
        return points, vis_image

    def selfLightness_split(self,img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        top_left_image, top_right_image, bottom_left_image, bottom_right_image = self.self_lightness.split_image(gray_img)
        top_left_binary_img = self.self_lightness.binary_image(top_left_image)
        top_right_binary_img = self.self_lightness.binary_image(top_right_image)
        bottom_left_binary_img = self.self_lightness.binary_image(bottom_left_image)
        bottom_right_binary_img = self.self_lightness.binary_image(bottom_right_image)
        composite_image = self.self_lightness.composite_image(top_left_binary_img, top_right_binary_img, bottom_left_binary_img, bottom_right_binary_img)
        return top_left_binary_img , top_right_binary_img , bottom_left_binary_img , bottom_right_binary_img , composite_image
    def selfLightness_notsplit(self,img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_img = self.self_lightness.binary_image(gray_img)
        return binary_img
    def mediaFilter(self,img):
        return self.media_filter.filter(img)

if __name__ == "__main__":
    img_process = ImgProcess(yoloseg="/home/yjc/Project/rov_ws/dataset/清水led双目/temp4train_simple/run/ledseg/weights/best.pt")
    folder_path = "/home/yjc/Project/rov_ws/dataset/清水led双目/images/left/left3"  # Linux/macOS
    img = cv2.imread("/home/yjc/Project/rov_ws/dataset/清水led双目/detected/leftwithout_redp/left2/490/image1.jpg")

    binary_img = img_process.self_lightness.binary_image(img)
    cv2.imshow("binary_img", binary_img)
    cv2.waitKey(0)



    # for filename in os.listdir(folder_path):
    #     # 筛选 .jpg / .JPG（大小写兼容）
    #     if filename.lower().endswith(".jpg"):
    #         # 拼接完整路径
    #         full_path = os.path.join(folder_path, filename)
    #         img = cv2.imread(full_path)
    #         dirname = filename.rsplit(".", 1)[0]
    #         dirpath = f"/home/yjc/Project/rov_ws/dataset/清水led双目/detected/left/left1/{dirname}"
    #         os.makedirs(dirpath, exist_ok=True)
    #         points = []
    #         points, vis_image = img_process.GetRoI_seg(img)
    #         if points == []:
    #             continue
    #         print(points)
    #         # cv2.imwrite("/home/yjc/Project/rov_ws/src/vision4location/src/image_save/seg/test/vis_image.jpg", vis_image)
    #         gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         roi_images = img_process.self_lightness.get_roi_image(gray_image, points)
    #         binary_images = []
    #         global_xy = []
    #         for i, roi_image in enumerate(roi_images): 
    #             extrema = img_process.extrema_detector.detect(roi_image)
    #             y, x = extrema["max_positions"]
    #             cv2.circle(roi_image, (int(x), int(y)), 2, (0, 0, 255), 3)  # red for max
    #             global_x = x - 15 + points[i][0]
    #             global_y = y - 15 + points[i][1]
    #             global_xy.append((global_x,global_y))
    #             binary_images.append(roi_image)
    #             cv2.circle(img, (int(global_x), int(global_y)), 2, (0, 0, 255), 2)
    #         # composite_image = img_process.self_lightness.combine_rois(points, binary_images, gray_image.shape, side_length_or_diameter=30)
    #         #     cv2.circle(img,(int(global)))
    #         # cv2.imwrite(f"{dirpath}/vis_image.jpg", vis_image)
    #         # cv2.imwrite(f"{dirpath}/composite_image.jpg", composite_image)
    #         cv2.imwrite(f"{dirpath}/image1.jpg", roi_images[0])
    #         cv2.imwrite(f"{dirpath}/image2.jpg", roi_images[1])
    #         cv2.imwrite(f"{dirpath}/image3.jpg", roi_images[2])
    #         cv2.imwrite(f"{dirpath}/image4.jpg", roi_images[3])
    #         cv2.imwrite(f"{dirpath}/image.jpg", img)
