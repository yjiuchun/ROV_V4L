# 简化日志绘图器使用说明

## 概述

`LogPlotter` 是一个简化的工具，专门用于获取和记录 `camera_link` 在世界坐标系中的位姿。它能够：

- 实时获取 `camera_link` 相对于 `world` 坐标系的位姿
- 在控制台输出位姿信息
- 保存位姿数据到文件
- 提供简单的数据记录功能

## 功能特性

### 1. 实时位姿监控
- 通过 TF 系统获取 `camera_link` 在世界坐标系中的位姿
- 支持位置 (x, y, z) 和姿态 (四元数) 的实时监控
- 自动处理 TF 变换异常

### 2. 控制台输出
- 定期在控制台显示位姿信息
- 包含时间戳、位置和姿态数据
- 可配置打印间隔

### 3. 数据管理
- 自动保存数据到文件
- 支持 NPZ 格式保存
- 包含完整的时间戳和位姿数据

## 使用方法

### 1. 启动 Gazebo 仿真
```bash
cd /home/yjc/Project/rov_ws
source devel/setup.bash
roslaunch rov_description gazebo.launch
```

### 2. 启动日志绘图器

#### 方法一：使用启动文件
```bash
roslaunch vision_location logplotter.launch
```

#### 方法二：直接运行节点
```bash
rosrun vision_location logplotter.py
```

### 3. 查看位姿数据
- 程序会在控制台输出位姿信息
- 显示位置 (x, y, z) 和姿态 (四元数)
- 数据会定期更新并保存到文件

## 配置参数

### 数据存储参数
```yaml
data_params:
  max_points: 1000          # 最大存储点数
  update_rate: 30           # 数据更新频率 (Hz)
  plot_update_rate: 10      # 绘图更新频率 (Hz)
```

### 绘图参数
```yaml
plot_params:
  figure_size: [12, 8]      # 图形大小 [宽, 高]
  line_width: 2             # 线条宽度
  grid_enabled: true        # 是否显示网格
  legend_enabled: true      # 是否显示图例
```

### TF 配置
```yaml
tf_config:
  source_frame: 'world'     # 源坐标系
  target_frame: 'camera_link'  # 目标坐标系
  timeout: 1.0              # TF 查找超时时间 (秒)
```

## 输出信息

### 控制台输出
```
[INFO] 日志绘图器已启动
[INFO] 正在监听 camera_link 在世界坐标系中的位姿...
[INFO] 开始显示实时位姿曲线...
[INFO] Camera Pose - Position: (0.123, 0.456, 0.789)
[INFO] Camera Pose - Orientation: (0.000, 0.000, 0.000, 1.000)
```

### 数据文件
- 自动保存为 `camera_pose_log_[timestamp].npz` 格式
- 包含时间戳、位置、姿态等完整数据

## 图形界面说明

### 1. 位置曲线图 (左上)
- X轴：时间 (秒)
- Y轴：位置 (米)
- 显示 X、Y、Z 三个方向的位置变化

### 2. 姿态曲线图 (右上)
- X轴：时间 (秒)
- Y轴：四元数值
- 显示四元数的 X、Y、Z、W 分量变化

### 3. 3D轨迹图 (左下)
- 显示相机在3D空间中的运动轨迹
- 红色点表示当前位置
- 蓝色线表示历史轨迹

### 4. 速度曲线图 (右下)
- X轴：时间 (秒)
- Y轴：速度 (米/秒)
- 显示 X、Y、Z 三个方向的速度变化

## 故障排除

### 常见问题

1. **无法获取 TF 变换**
   - 检查 Gazebo 是否正常运行
   - 确认 `camera_link` 和 `world` 坐标系存在
   - 检查 TF 树是否完整

2. **图形不显示**
   - 确认 matplotlib 已正确安装
   - 检查显示环境设置
   - 尝试设置 `export DISPLAY=:0`

3. **数据不更新**
   - 检查相机是否在运动
   - 确认 TF 变换是否正常发布
   - 查看控制台错误信息

### 调试命令

```bash
# 查看 TF 树
rosrun tf view_frames

# 检查特定 TF 变换
rosrun tf tf_echo world camera_link

# 查看所有 TF 帧
rostopic echo /tf
```

## 扩展功能

### 1. 自定义坐标系
修改 `tf_config` 中的 `source_frame` 和 `target_frame` 来监控其他坐标系关系。

### 2. 数据导出
支持导出为 CSV 或 JSON 格式，便于后续分析。

### 3. 实时分析
可以添加实时分析功能，如轨迹平滑度、速度统计等。

## 依赖项

- ROS Noetic
- Python 3.x
- matplotlib
- numpy
- tf2_ros
- geometry_msgs

## 安装依赖

```bash
# 安装 Python 依赖
pip3 install matplotlib numpy

# 安装 ROS 依赖
sudo apt-get install ros-noetic-tf2-ros ros-noetic-geometry-msgs
```

## 版本信息

- 创建日期: 2024年1月
- 版本: 1.0
- 作者: AI Assistant
- 兼容性: ROS Noetic, Ubuntu 20.04


