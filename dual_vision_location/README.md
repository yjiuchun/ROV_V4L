# 双目视觉定位系统

本系统用于使用双目相机（ZED2）进行3D定位和位姿估计。

## 功能概述

1. **三角测量（Triangulation）**：从左右图像的配对关键点像素坐标计算3D坐标
2. **位姿估计（Pose Estimation）**：如果提供了世界坐标系中的3D点坐标，可以使用PnP算法估计相机位姿

## 系统架构

- `Dual_detector.py`: 主检测器，负责图像接收、特征点检测和定位流程
- `stereo_triangulation.py`: 双目三角测量模块，负责3D坐标计算和PnP求解
- `config/stereo_calibration.yaml`: 双目相机标定参数配置文件

## 使用步骤

### 1. 配置相机标定参数

编辑 `config/stereo_calibration.yaml` 文件，填写您的双目相机标定结果：

```yaml
# 左相机内参
left_camera:
  fx: 476.7    # 焦距x (像素)
  fy: 476.7    # 焦距y (像素)
  cx: 640.5    # 主点x (像素)
  cy: 360.5    # 主点y (像素)
  distortion_coeffs: [0, 0, 0, 0]  # 畸变系数

# 右相机内参
right_camera:
  fx: 476.7
  fy: 476.7
  cx: 640.5
  cy: 360.5
  distortion_coeffs: [0, 0, 0, 0]

# 立体标定参数
stereo:
  R:  # 旋转矩阵（从右相机到左相机）
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  T: [-0.12, 0.0, 0.0]  # 平移向量（基线长度，单位：米）

# 世界坐标系中的3D点（可选，用于PnP位姿估计）
world_points:
  - [-0.15, 0.15, 0.0]
  - [0.15, 0.15, 0.0]
  - [0.15, -0.15, 0.0]
  - [-0.15, -0.15, 0.0]
```

**如何获取标定参数：**

1. 使用OpenCV的 `stereoCalibrate()` 函数进行双目标定
2. 或使用ROS的 `camera_calibration` 包进行标定
3. 对于ZED2相机，也可以从相机SDK获取标定参数

### 2. 获取配对的关键点坐标

系统已经集成了特征点检测功能。当检测到左右图像中都有4个或更多配对的关键点时，会自动进行三角测量。

关键点坐标格式：
- 左图像关键点：`[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]`
- 右图像关键点：`[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]`

**注意**：左右图像中的点必须一一对应，即第一个点对应第一个点，以此类推。

### 3. 运行系统

```bash
# 启动ROS节点
rosrun dual_vision_location Dual_detector.py
```

### 4. 查看结果

系统会发布以下ROS话题：

- `/stereo_vision/pose`: 位姿估计结果（PoseStamped消息，如果进行了PnP求解）
- `/stereo_vision/points_3d`: 3D点坐标（PointStamped消息）

可以使用以下命令查看：

```bash
# 查看位姿
rostopic echo /stereo_vision/pose

# 查看3D点
rostopic echo /stereo_vision/points_3d
```

## 工作流程

1. **接收图像**：从ROS话题订阅左右相机图像
2. **特征点检测**：在左右图像中分别检测特征点（RGB颜色圆）
3. **点配对**：确保左右图像的点一一对应
4. **三角测量**：使用双目视差计算3D坐标（相机坐标系）
5. **PnP求解**（可选）：如果提供了世界坐标点，计算相机位姿

## 代码示例

如果您想直接使用三角测量功能：

```python
from stereo_triangulation import StereoTriangulation
import numpy as np

# 初始化
triangulator = StereoTriangulation()

# 左右图像的像素坐标
left_points = np.array([[100, 200], [300, 200], [300, 400], [100, 400]], dtype=np.float32)
right_points = np.array([[80, 200], [280, 200], [280, 400], [80, 400]], dtype=np.float32)

# 执行三角测量
points_3d = triangulator.triangulate_points(left_points, right_points)

print("3D坐标：")
for i, pt in enumerate(points_3d):
    print(f"点{i+1}: X={pt[0]:.3f}m, Y={pt[1]:.3f}m, Z={pt[2]:.3f}m")
```

## 注意事项

1. **相机标定**：准确的标定参数对结果精度至关重要
2. **点配对**：确保左右图像中的点正确配对
3. **点数要求**：至少需要4个配对的关键点才能进行三角测量
4. **坐标系**：3D坐标是在左相机坐标系中（相机坐标系）
5. **基线长度**：双目相机的基线长度（T向量的模）影响深度估计精度

## 坐标系说明

- **图像坐标系**：左上角为原点，x向右，y向下，单位：像素
- **相机坐标系**（左相机）：相机光心为原点，x向右，y向下，z向前，单位：米
- **世界坐标系**：如果提供world_points，用于PnP求解的参考坐标系

## 故障排除

1. **标定参数未找到**：检查 `stereo_calibration.yaml` 文件路径和格式
2. **三角测量失败**：检查左右图像的点和数量是否匹配
3. **PnP求解失败**：检查world_points是否正确配置，点数和特征点是否一致


