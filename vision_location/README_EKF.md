# EKF视觉定位系统使用说明

## 概述

扩展卡尔曼滤波(EKF)视觉定位系统是一个用于优化PnP解算结果的工具，专门处理视觉标签移动时的定位误差。该系统通过状态估计和预测，提供更稳定和准确的位姿估计。

## 系统架构

### 状态向量设计
- **状态维度**: 10维
- **状态变量**: [x, y, z, vx, vy, vz, qx, qy, qz, qw]
  - 位置: (x, y, z)
  - 速度: (vx, vy, vz) 
  - 姿态: (qx, qy, qz, qw) 四元数

### 观测模型
- **观测维度**: 7维
- **观测变量**: [x, y, z, qx, qy, qz, qw]
  - 直接观测位置和姿态

### 运动模型
- **类型**: 匀速运动模型
- **状态转移**: x = x + vx * dt

## 功能特性

### 1. 状态估计
- 实时状态预测和更新
- 协方差矩阵管理
- 四元数归一化处理

### 2. 噪声处理
- 过程噪声建模
- 观测噪声建模
- 自适应噪声调整

### 3. 数据融合
- PnP解算结果作为观测输入
- 多传感器数据融合支持
- 异常值检测和处理

### 4. 输出接口
- 优化后的位姿估计
- 速度估计
- 不确定性量化
- TF变换发布

## 文件结构

```
vision_location/
├── scripts/
│   ├── ekf.py                 # 主EKF系统
│   ├── test_ekf.py           # 测试脚本
│   └── ekf_data_logger.py    # 数据记录器
├── config/
│   └── ekf_config.yaml       # 配置文件
├── launch/
│   └── ekf_visual_localization.launch  # 启动文件
└── README_EKF.md             # 使用说明
```

## 使用方法

### 1. 启动Gazebo仿真
```bash
cd /home/yjc/Project/rov_ws
source devel/setup.bash
roslaunch rov_description gazebo.launch
```

### 2. 启动PnP视觉定位
```bash
rosrun vision_location detector.py
```

### 3. 启动EKF系统

#### 方法一：使用启动文件
```bash
roslaunch vision_location ekf_visual_localization.launch
```

#### 方法二：直接运行节点
```bash
rosrun vision_location ekf.py
```

### 4. 启动测试和监控
```bash
# 启动性能测试
rosrun vision_location test_ekf.py

# 启动数据记录
rosrun vision_location ekf_data_logger.py
```

## 配置参数

### EKF核心参数
```yaml
ekf_params:
  process_noise_position: 0.01      # 位置过程噪声
  process_noise_velocity: 0.1       # 速度过程噪声
  process_noise_orientation: 0.01   # 姿态过程噪声
  observation_noise_position: 0.1   # 位置观测噪声
  observation_noise_orientation: 0.1 # 姿态观测噪声
  update_rate: 30.0                 # 更新频率
```

### 运动模型参数
```yaml
motion_model:
  type: "constant_velocity"         # 运动模型类型
  max_acceleration: 2.0            # 最大加速度
  max_velocity: 5.0                # 最大速度
```

### 观测模型参数
```yaml
observation_model:
  type: "pose"                     # 观测类型
  position_valid_range: [0.1, 10.0] # 位置有效范围
  outlier_detection:
    enabled: true                  # 异常值检测
    chi_square_threshold: 9.21     # 卡方阈值
```

## 话题接口

### 输入话题
- `/vision/estimated_pose` (PoseStamped): PnP解算结果
- `/imu/data` (可选): IMU数据
- `/odom` (可选): 里程计数据

### 输出话题
- `/vision/ekf_pose` (PoseStamped): EKF优化位姿
- `/vision/ekf_pose_covariance` (PoseWithCovarianceStamped): 带协方差的位姿
- `/vision/estimated_velocity` (TwistStamped): 速度估计
- `/vision/ekf_status` (可选): EKF状态信息

### TF变换
- `camera_link` → `target_link`: 目标相对于相机的位置

## 性能优化

### 1. 数值稳定性
- 四元数归一化
- 协方差矩阵正则化
- 条件数检查

### 2. 计算效率
- 矩阵运算优化
- 稀疏矩阵支持
- 并行处理选项

### 3. 自适应调整
- 噪声自适应
- 协方差膨胀
- 异常值处理

## 调试和监控

### 1. 状态监控
```bash
# 查看EKF状态
rostopic echo /vision/ekf_pose

# 查看速度估计
rostopic echo /vision/estimated_velocity

# 查看协方差信息
rostopic echo /vision/ekf_pose_covariance
```

### 2. 性能分析
```bash
# 启动测试脚本
rosrun vision_location test_ekf.py

# 查看实时性能曲线
# 程序会自动显示位置对比、速度估计、不确定性等图表
```

### 3. 数据记录
```bash
# 启动数据记录器
rosrun vision_location ekf_data_logger.py

# 数据会保存到CSV文件，包含：
# - 时间戳
# - PnP和EKF位姿对比
# - 速度估计
# - 不确定性
# - 误差分析
```

## 故障排除

### 常见问题

1. **EKF未初始化**
   - 检查PnP话题是否正常发布
   - 确认配置文件路径正确
   - 查看控制台错误信息

2. **位姿估计不稳定**
   - 调整过程噪声参数
   - 检查观测噪声设置
   - 验证运动模型参数

3. **计算性能问题**
   - 降低更新频率
   - 启用稀疏矩阵
   - 检查系统资源使用

### 调试命令

```bash
# 检查话题连接
rostopic list | grep vision

# 查看话题频率
rostopic hz /vision/estimated_pose
rostopic hz /vision/ekf_pose

# 检查TF变换
rosrun tf tf_echo camera_link target_link

# 查看节点状态
rosnode info /ekf_visual_localization
```

## 参数调优指南

### 1. 过程噪声调优
- **位置噪声**: 根据目标运动特性调整
- **速度噪声**: 影响速度估计的平滑性
- **姿态噪声**: 影响姿态估计的稳定性

### 2. 观测噪声调优
- **位置噪声**: 根据PnP解算精度设置
- **姿态噪声**: 根据姿态估计精度设置

### 3. 系统参数调优
- **更新频率**: 平衡精度和计算效率
- **异常值阈值**: 根据应用场景调整
- **协方差膨胀**: 提高系统鲁棒性

## 扩展功能

### 1. 多传感器融合
- 集成IMU数据
- 融合里程计信息
- 多相机系统支持

### 2. 高级运动模型
- 恒定加速度模型
- 恒定转弯模型
- 自定义运动模型

### 3. 自适应滤波
- 自适应噪声估计
- 模型参数在线调整
- 多模型滤波

## 依赖项

- ROS Noetic
- Python 3.x
- numpy
- matplotlib
- tf2_ros
- geometry_msgs
- sensor_msgs

## 安装依赖

```bash
# 安装Python依赖
pip3 install numpy matplotlib

# 安装ROS依赖
sudo apt-get install ros-noetic-tf2-ros ros-noetic-geometry-msgs
```

## 版本信息

- 创建日期: 2024年1月
- 版本: 1.0
- 作者: AI Assistant
- 兼容性: ROS Noetic, Ubuntu 20.04

## 参考文献

1. Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic robotics.
2. Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). Estimation with applications to tracking and navigation.
3. Julier, S. J., & Uhlmann, J. K. (2004). Unscented filtering and nonlinear estimation.


