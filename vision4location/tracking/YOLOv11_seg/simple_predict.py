from ultralytics import YOLO
import time
import cv2
# 加载训练好的模型
model = YOLO("/home/yjc/Project/rov_ws/dataset/清水led双目/temp4train_simple/run/ledseg/weights/best.pt")



results = model.predict(
    source="/home/yjc/Project/rov_ws/dataset/清水led双目/images/left/left1/1.jpg",  # 图片路径/文件夹/视频/摄像头（0）
    imgsz=640,
    conf=0.25,          # 置信度阈值
    iou=0.7,            # NMS的IoU阈值
    save=False,          # 保存预测结果图片/视频
    save_txt=False,      # 保存预测标签（分割坐标+类别）
    show=True          # 实时显示预测结果（桌面环境可用）
)

results = model.predict(
    source="/home/yjc/Project/rov_ws/dataset/清水led双目/images/left/left1/2.jpg",  # 图片路径/文件夹/视频/摄像头（0）
    imgsz=640,
    conf=0.25,          # 置信度阈值
    iou=0.7,            # NMS的IoU阈值
    save=False,          # 保存预测结果图片/视频
    save_txt=False,      # 保存预测标签（分割坐标+类别）
    show=False          # 实时显示预测结果（桌面环境可用）
)
