# Vision Location 功能包

这个功能包用于从仿真环境中的相机获取图像，并进行视觉定位算法处理。

## 功能特性

- 订阅相机图像话题 `/camera/image_raw`
- 图像预处理和特征检测
- 位置估计和发布
- 图像保存功能
- 实时图像显示

## 文件结构

```
vision_location/
├── src/
│   ├── pnp_vision.py          # PnP视觉定位节点
│   └── pnp_vision_advanced.py # 高级PnP视觉定位节点
├── scripts/
│   ├── detector.py            # 特征检测器
│   ├── logplotter.py          # 简化日志绘图器
│   ├── pnp_solver.py          # PnP求解器
│   ├── pose_tf.py             # 位姿TF处理
│   └── test_logplotter.py     # 日志绘图器测试
├── config/
│   ├── color_detection.yaml   # 颜色检测配置
│   ├── detector.yaml          # 检测器配置
│   ├── logplotter.yaml        # 日志绘图器配置
│   └── pnp.yaml               # PnP配置
├── launch/
│   ├── logplotter.launch      # 日志绘图器启动文件
│   └── pnp_vision.launch      # PnP视觉定位启动文件
├── package.xml
├── CMakeLists.txt
└── README.md
```

## 使用方法

### 1. 编译工作空间

```bash
cd /home/yjc/Project/rov_ws
catkin_make
source devel/setup.bash
```

### 2. 启动仿真环境

```bash
# 启动 Gazebo 仿真
roslaunch rov_description gazebo.launch
```

### 3. 启动视觉定位

```bash
# 启动完整的视觉定位系统
roslaunch vision_location vision_location.launch

# 或者单独启动图像订阅节点
rosrun vision_location image_subscriber.py

# 或者单独启动视觉处理节点
rosrun vision_location vision_processor.py
```

### 4. 使用日志绘图器

```bash
# 启动日志绘图器 (获取camera_link位姿)
roslaunch vision_location logplotter.launch

# 或者直接运行
rosrun vision_location logplotter.py

# 测试日志绘图器
rosrun vision_location test_logplotter.py
```

### 5. 使用PnP视觉定位

```bash
# 启动PnP视觉定位系统
roslaunch vision_location pnp_vision.launch
```

## 日志绘图器 (LogPlotter)

### 功能说明
简化的日志绘图器专门用于获取和记录 `camera_link` 在世界坐标系中的位姿。

### 主要特性
- 实时获取 `camera_link` 相对于 `world` 坐标系的位姿
- 在控制台输出位姿信息 (位置和姿态)
- 自动保存数据到 NPZ 格式文件
- 可配置更新频率和打印间隔

### 输出示例
```
[0010] 时间: 1234.567s
     位置: (0.123456, 0.456789, 0.789012)
     姿态: (0.000000, 0.000000, 0.000000, 1.000000)
--------------------------------------------------
```

### 配置参数
- `source_frame`: 源坐标系 (默认: 'base_footprint')
- `target_frame`: 目标坐标系 (默认: 'camera_link')
- `update_rate`: 数据更新频率 (默认: 10 Hz)
- `print_interval`: 打印间隔 (默认: 每10次打印一次)

## 话题说明

### 订阅话题
- `/camera/image_raw` (sensor_msgs/Image): 相机原始图像
- `/camera/camera_info` (sensor_msgs/CameraInfo): 相机标定信息

### 发布话题
- `/vision/processed_image` (sensor_msgs/Image): 处理后的图像
- `/vision/estimated_pose` (geometry_msgs/PoseStamped): 估计的位置

## 图像保存

图像会自动保存到 `~/camera_images/` 目录下，文件名格式为 `camera_image_XXXX.jpg`。

## 依赖项

- rospy
- cv_bridge
- sensor_msgs
- std_msgs
- geometry_msgs
- tf2_ros
- tf2_geometry_msgs
- image_view

## 注意事项

1. 确保 Gazebo 仿真环境正在运行
2. 确保相机插件已正确配置
3. 图像保存功能需要足够的磁盘空间
4. 建议在启动前检查相机话题是否正常发布


