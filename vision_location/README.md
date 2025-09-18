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
│   ├── image_subscriber.py    # 图像订阅节点
│   └── vision_processor.py    # 视觉处理节点
├── scripts/
│   └── test_camera.py         # 相机测试脚本
├── launch/
│   └── vision_location.launch # 启动文件
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

### 4. 测试相机

```bash
# 运行相机测试脚本
rosrun vision_location test_camera.py
```

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


