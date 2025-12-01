yolo detect train model=yolov11n.pt data=/home/yjc/Project/rov_ws/output_images/dataset_ledring/YOLODataset/dataset.yaml epochs=100 batch=16 imgsz=640 device=0 cls_pw=1.0 obj_pw=2.0 conf=0.25 iou=0.45 save=True project=/home/yjc/Project/rov_ws/src/vision4location/tracking/YOLOv11/firsttrain_withledring name=single_class_exp


  yolo detect train \
  model=yolov11n.pt \       # 优先用轻量化模型（n/s，避免过拟合）
  data=/home/yjc/my_dataset.yaml \  # 单类别配置文件
  epochs=100 \              # 单类别可适当减少轮数（50-100足够）
  batch=16 \                # 按显存调整（8/16/32）
  imgsz=640 \               # 保持32倍数（480/640均可）
  device=0 \                # 显卡ID
  # 以下是单类别关键优化参数
  cls_pw=1.0 \              # 类别权重（单类别无需调整，默认1.0）
  obj_pw=2.0 \              # 目标置信度权重（单目标可适当提高，增强目标检测）
  conf=0.25 \               # 推理置信度阈值（单目标可设高一点，如0.3）
  iou=0.45 \                # NMS阈值（单目标无需调整）
  save=True \
  project=/home/yjc/Project/rov_ws/src/vision4location/tracking/YOLOv11/firsttrain_withledring \
  name=single_class_exp