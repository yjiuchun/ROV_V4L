# PnP 视觉定位系统使用说明

## 概述

本系统实现了基于 PnP (Perspective-n-Point) 算法的视觉定位功能，能够从 Gazebo 仿真环境中获取相机图像，提取红黄蓝绿四个圆的中点作为特征点，并计算相机的位姿。

## 文件结构

```
vision_location/
├── src/
│   ├── pnp_vision.py              # 基础 PnP 视觉定位类
│   └── pnp_vision_advanced.py     # 高级 PnP 视觉定位类（支持配置文件）
├── scripts/
│   └── test_feature_detection.py  # 特征点检测测试脚本
├── launch/
│   └── pnp_vision.launch          # PnP 系统启动文件
└── config/
    └── color_detection.yaml       # 颜色检测配置文件
```

## 主要功能

### 1. 图像获取
- 订阅 `/camera/image_raw` 话题获取 Gazebo 相机图像
- 使用 `cv_bridge` 将 ROS 图像消息转换为 OpenCV 格式

### 2. 特征点提取
- 使用 HSV 颜色空间进行颜色检测
- 支持检测红、黄、蓝、绿四种颜色的圆形特征
- 通过轮廓分析和圆形度检测确保特征点质量
- 提取每个圆形的几何中心作为特征点

### 3. PnP 位姿估计
- 使用 OpenCV 的 `cv2.solvePnP` 函数进行位姿求解
- 支持自定义 3D 特征点坐标和相机内参
- 将旋转矩阵转换为四元数表示

### 4. 结果发布
- 发布 `/vision/estimated_pose` 话题
- 位姿信息包含位置 (x, y, z) 和姿态 (四元数)

## 使用方法

### 1. 启动 Gazebo 仿真
```bash
cd /home/yjc/Project/rov_ws
source devel/setup.bash
roslaunch rov_description gazebo.launch
```

### 2. 启动 PnP 视觉定位系统

#### 方法一：使用启动文件
```bash
roslaunch vision_location pnp_vision.launch
```

#### 方法二：直接运行节点
```bash
# 基础版本
rosrun vision_location pnp_vision.py

# 高级版本（推荐）
rosrun vision_location pnp_vision_advanced.py
```

### 3. 测试特征点检测
```bash
rosrun vision_location test_feature_detection.py
```

## 配置参数

### 颜色检测参数 (config/color_detection.yaml)

```yaml
# HSV颜色范围定义
color_ranges:
  red:
    - lower: [0, 50, 50]
      upper: [10, 255, 255]
    - lower: [170, 50, 50]
      upper: [180, 255, 255]
  
  yellow:
    - lower: [20, 50, 50]
      upper: [30, 255, 255]
  
  blue:
    - lower: [100, 50, 50]
      upper: [130, 255, 255]
  
  green:
    - lower: [40, 50, 50]
      upper: [80, 255, 255]

# 检测参数
detection_params:
  min_area: 100          # 最小轮廓面积
  circularity_threshold: 0.7  # 圆形度阈值
  kernel_size: 5         # 形态学操作核大小
```

### 相机内参
```yaml
camera_matrix:
  fx: 800    # 焦距 x
  fy: 800    # 焦距 y
  cx: 400    # 主点 x
  cy: 300    # 主点 y
```

### 3D 特征点坐标
```yaml
object_points:
  - [-0.15, 0.15, 0.0]   # 黄色圆
  - [0.15, 0.15, 0.0]    # 蓝色圆
  - [0.15, -0.15, 0.0]   # 绿色圆
  - [-0.15, -0.15, 0.0]  # 红色圆
```

## 输出信息

### 控制台输出
```
[INFO] 配置文件加载成功
[INFO] 高级PnP视觉定位系统已启动
[INFO] 订阅话题: /camera/image_raw
[INFO] 发布话题: /vision/estimated_pose
[INFO] 检测到yellow圆中心: (364, 275)
[INFO] 检测到blue圆中心: (435, 275)
[INFO] 检测到green圆中心: (435, 347)
[INFO] 检测到red圆中心: (364, 347)
[INFO] PnP求解成功: 位置=(-0.000, -0.000, 0.000)
```

### 话题输出
- **输入话题**: `/camera/image_raw` (sensor_msgs/Image)
- **输出话题**: `/vision/estimated_pose` (geometry_msgs/PoseStamped)

## 调试和优化

### 1. 调整颜色检测参数
- 根据实际环境光照条件调整 HSV 颜色范围
- 修改 `min_area` 和 `circularity_threshold` 参数

### 2. 相机标定
- 使用实际相机标定结果更新 `camera_matrix` 和 `distortion_coeffs`
- 确保 3D 特征点坐标与实际物理位置一致

### 3. 性能优化
- 调整图像处理频率
- 优化形态学操作参数
- 使用多线程处理

## 故障排除

### 常见问题

1. **无法检测到特征点**
   - 检查颜色范围设置
   - 确认 Gazebo 环境中存在对应颜色的圆形
   - 调整最小面积阈值

2. **PnP 求解失败**
   - 确保检测到至少 4 个特征点
   - 检查 3D 特征点坐标是否正确
   - 验证相机内参设置

3. **位姿估计不准确**
   - 进行相机标定
   - 精确测量 3D 特征点位置
   - 检查特征点检测精度

## 扩展功能

### 1. 添加更多特征点
- 在配置文件中添加新的颜色定义
- 更新 3D 特征点坐标数组

### 2. 集成其他传感器
- 融合 IMU 数据提高位姿估计精度
- 结合激光雷达进行环境感知

### 3. 实时可视化
- 添加 RViz 可视化插件
- 显示检测到的特征点和估计位姿

## 依赖项

- ROS Noetic
- OpenCV 4.x
- cv_bridge
- sensor_msgs
- geometry_msgs
- tf2_ros
- PyYAML (用于配置文件解析)

## 版本信息

- 创建日期: 2024年1月
- 版本: 1.0
- 作者: AI Assistant
- 兼容性: ROS Noetic, Ubuntu 20.04
