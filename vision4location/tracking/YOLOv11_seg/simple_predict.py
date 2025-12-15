from ultralytics import YOLO
import time
# 加载训练好的模型
model = YOLO("/home/yjc/Project/rov_ws/src/vision4location/tracking/YOLOv11_seg/yolov11_seg_train/led_sys_seg/weights/best.pt")

# 1. 图片推理
start_time = time.time()

results = model.predict(
    source="/home/yjc/Project/rov_ws/underwater_dataset/images/first_capture/right/64.jpg",  # 图片路径/文件夹/视频/摄像头（0）
    imgsz=640,
    conf=0.25,          # 置信度阈值
    iou=0.7,            # NMS的IoU阈值
    save=False,          # 保存预测结果图片/视频
    save_txt=False,      # 保存预测标签（分割坐标+类别）
    show=False          # 实时显示预测结果（桌面环境可用）
)

results = model.predict(
    source="/home/yjc/Project/rov_ws/underwater_dataset/images/first_capture/right/64.jpg",  # 图片路径/文件夹/视频/摄像头（0）
    imgsz=640,
    conf=0.25,          # 置信度阈值
    iou=0.7,            # NMS的IoU阈值
    save=False,          # 保存预测结果图片/视频
    save_txt=False,      # 保存预测标签（分割坐标+类别）
    show=False          # 实时显示预测结果（桌面环境可用）
)

results = model.predict(
    source="/home/yjc/Project/rov_ws/underwater_dataset/images/first_capture/right/64.jpg",  # 图片路径/文件夹/视频/摄像头（0）
    imgsz=640,
    conf=0.25,          # 置信度阈值
    iou=0.7,            # NMS的IoU阈值
    save=False,          # 保存预测结果图片/视频
    save_txt=False,      # 保存预测标签（分割坐标+类别）
    show=False          # 实时显示预测结果（桌面环境可用）
)

results = model.predict(
    source="/home/yjc/Project/rov_ws/underwater_dataset/images/first_capture/right/64.jpg",  # 图片路径/文件夹/视频/摄像头（0）
    imgsz=640,
    conf=0.25,          # 置信度阈值
    iou=0.7,            # NMS的IoU阈值
    save=False,          # 保存预测结果图片/视频
    save_txt=False,      # 保存预测标签（分割坐标+类别）
    show=False          # 实时显示预测结果（桌面环境可用）
)

results = model.predict(
    source="/home/yjc/Project/rov_ws/underwater_dataset/images/first_capture/right/64.jpg",  # 图片路径/文件夹/视频/摄像头（0）
    imgsz=640,
    conf=0.25,          # 置信度阈值
    iou=0.7,            # NMS的IoU阈值
    save=False,          # 保存预测结果图片/视频
    save_txt=False,      # 保存预测标签（分割坐标+类别）
    show=False          # 实时显示预测结果（桌面环境可用）
)

results = model.predict(
    source="/home/yjc/Project/rov_ws/underwater_dataset/images/first_capture/right/64.jpg",  # 图片路径/文件夹/视频/摄像头（0）
    imgsz=640,
    conf=0.25,          # 置信度阈值
    iou=0.7,            # NMS的IoU阈值
    save=False,          # 保存预测结果图片/视频
    save_txt=False,      # 保存预测标签（分割坐标+类别）
    show=False          # 实时显示预测结果（桌面环境可用）
)

results = model.predict(
    source="/home/yjc/Project/rov_ws/underwater_dataset/images/first_capture/right/64.jpg",  # 图片路径/文件夹/视频/摄像头（0）
    imgsz=640,
    conf=0.25,          # 置信度阈值
    iou=0.7,            # NMS的IoU阈值
    save=False,          # 保存预测结果图片/视频
    save_txt=False,      # 保存预测标签（分割坐标+类别）
    show=False          # 实时显示预测结果（桌面环境可用）
)

results = model.predict(
    source="/home/yjc/Project/rov_ws/underwater_dataset/images/first_capture/right/64.jpg",  # 图片路径/文件夹/视频/摄像头（0）
    imgsz=640,
    conf=0.25,          # 置信度阈值
    iou=0.7,            # NMS的IoU阈值
    save=False,          # 保存预测结果图片/视频
    save_txt=False,      # 保存预测标签（分割坐标+类别）
    show=False          # 实时显示预测结果（桌面环境可用）
)

results = model.predict(
    source="/home/yjc/Project/rov_ws/underwater_dataset/images/first_capture/right/65.jpg",  # 图片路径/文件夹/视频/摄像头（0）
    imgsz=640,
    conf=0.25,          # 置信度阈值
    iou=0.7,            # NMS的IoU阈值
    save=False,          # 保存预测结果图片/视频
    save_txt=False,      # 保存预测标签（分割坐标+类别）
    show=False          # 实时显示预测结果（桌面环境可用）
)

results = model.predict(
    source="/home/yjc/Project/rov_ws/underwater_dataset/images/first_capture/right/61.jpg",  # 图片路径/文件夹/视频/摄像头（0）
    imgsz=640,
    conf=0.25,          # 置信度阈值
    iou=0.7,            # NMS的IoU阈值
    save=False,          # 保存预测结果图片/视频
    save_txt=False,      # 保存预测标签（分割坐标+类别）
    show=False          # 实时显示预测结果（桌面环境可用）
)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
# 解析预测结果
for r in results:
    print(f"检测到的类别: {r.names}")
    print(f"分割掩码形状: {r.masks.data.shape}")  # 掩码张量
    print(f"目标框坐标: {r.boxes.xyxy}")  # 目标框（像素坐标）