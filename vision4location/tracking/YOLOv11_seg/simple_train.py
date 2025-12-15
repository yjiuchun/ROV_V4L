from ultralytics import YOLO

# 1. 加载预训练模型
model = YOLO("yolo11n-seg.pt")  # 也可加载自定义权重：YOLO("runs/segment/exp/weights/best.pt")

# 2. 开始训练
results = model.train(
    data="/home/yjc/Project/rov_ws/src/vision4location/tracking/YOLOv11_seg/dataset_1.yaml",  # 数据集配置文件路径
    epochs=100,           # 训练轮数
    batch=8,              # 批次大小（自动适配GPU显存，建议设为-1自动批量）
    imgsz=640,            # 输入图片尺寸
    device=0,             # 使用第0块GPU，CPU设为cpu
    lr0=0.01,             # 初始学习率
    patience=50,          # 早停阈值
    save=True,            # 保存权重
    save_period=10,       # 每10个epoch保存一次权重
    val=True,             # 训练中验证（默认开启）
    augment=True,         # 数据增强（默认开启）
    workers=4,            # 数据加载线程数
    project="/home/yjc/Project/rov_ws/src/vision4location/tracking/YOLOv11_seg/seg_second_train",  # 结果保存根目录
    name="led_sys_seg",           # 实验名称
    exist_ok=True         # 覆盖已有实验目录
)

# 3. 训练完成后评估
metrics = model.val()  # 验证集评估，返回mAP、IoU等指标
print(f"验证集mAP50: {metrics.box.map50}")  # 目标检测mAP50
print(f"验证集分割mAP50: {metrics.seg.map50}")  # 实例分割mAP50