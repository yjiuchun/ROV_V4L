#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
双目相机图像显示检测器
用于获取并显示zed2双目相机的左右图像
"""
import cv2
import numpy as np
import sys
import os
import time

# 添加vision_location的scripts目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
vision_location_scripts = '/home/yjc/Project/rov_ws/src/vision_location/scripts'
if vision_location_scripts not in sys.path:
    sys.path.insert(0, vision_location_scripts)

# 导入vision_location的Detector类
from detector import Detector

# 导入双目三角测量模块
sys.path.insert(0, current_dir)
from stereo_triangulation import StereoTriangulation


class DualCameraDetector:
    def __init__(self):
        """初始化双目相机检测器"""
        # 窗口名称
        self.window_name = "ZED2 Dual Camera View"
        self.left_detector = Detector(config_name='left_detector.yaml',folder=os.path.dirname(current_dir))
        self.right_detector = Detector(config_name='right_detector.yaml',folder=os.path.dirname(current_dir))
        self.left_feature_points = []
        self.right_feature_points = []
        self.save_image = True
        
        # 初始化双目三角测量器
        try:
            self.stereo_triangulator = StereoTriangulation()
            self.world_points = self.load_world_points()
        except Exception as e:
            print(f"警告: 无法初始化双目三角测量器: {e}")
            self.stereo_triangulator = None
            self.world_points = None



        # 帧率计算变量
        self.prev_time = time.time()
        self.fps = 0.0
        self.frame_count = 0
        self.fps_update_interval = 0.5  # 每0.5秒更新一次FPS

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 2560, 720)  # 两个1280x720并排显示
        except Exception as e:
            print("警告: 无法创建OpenCV窗口: {}".format(e))
            print("提示: 如果您在无显示器的环境中运行，请设置DISPLAY环境变量或使用X11转发")
            self.window_name = None
    
    def display_images(self, left_image, right_image):
        """
        显示双目图像
        
        Args:
            left_image: 左相机图像 (numpy array, BGR格式)
            right_image: 右相机图像 (numpy array, BGR格式)
        """
        if left_image is None or right_image is None:
            return
        
        if self.window_name is None:
            return
        
        try:
            # 计算帧率
            current_time = time.time()
            self.frame_count += 1
            
            # 每隔一定时间更新FPS
            if current_time - self.prev_time >= self.fps_update_interval:
                self.fps = self.frame_count / (current_time - self.prev_time)
                self.frame_count = 0
                self.prev_time = current_time
            
            # 在图像上添加标签
            left_labeled = left_image.copy()
            right_labeled = right_image.copy()
            
            # 添加文字标签
            cv2.putText(left_labeled, "Left Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(right_labeled, "Right Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 添加帧率显示
            fps_text = "FPS: {:.1f}".format(self.fps)
            cv2.putText(left_labeled, fps_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(right_labeled, fps_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 水平并排显示两个图像
            dual_view = np.hstack([left_labeled, right_labeled])
            
            # 显示图像
            cv2.imshow(self.window_name, dual_view)
            if self.save_image:
                cv2.imwrite('./left_image.jpg', left_labeled)
                cv2.imwrite('./right_image.jpg', right_labeled)
                # self.save_image = False
            
            cv2.waitKey(1)
            
        except cv2.error as e:
            print("OpenCV显示错误: {}".format(e))
        except Exception as e:
            print("显示图像时出错: {}".format(e))
            import traceback
            traceback.print_exc()
    
    def load_world_points(self):
        """
        从配置文件加载世界坐标系中的3D点
        
        Returns:
            world_points: 世界坐标系中的3D点，形状 (N, 3) 或 None
        """
        try:
            config_dir = os.path.dirname(os.path.dirname(__file__))
            config_path = os.path.join(config_dir, 'config', 'stereo_calibration.yaml')
            
            import yaml
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            if 'world_points' in config and config['world_points']:
                world_points = np.array(config['world_points'], dtype=np.float32)
                print(f"已加载 {len(world_points)} 个世界坐标点")
                return world_points
            else:
                print("配置文件中未找到世界坐标点，将只进行三角测量")
                return None
        except Exception as e:
            print(f"加载世界坐标点失败: {e}")
            return None
    
    def perform_triangulation(self, left_points, right_points):
        """
        对左右图像的关键点进行三角测量
        
        Args:
            left_points: 左图像像素坐标，列表格式 [[x1, y1], [x2, y2], ...]
            right_points: 右图像像素坐标，列表格式 [[x1, y1], [x2, y2], ...]
        
        Returns:
            points_3d: 3D坐标，形状 (N, 3)，或 None
        """
        if self.stereo_triangulator is None:
            print("双目三角测量器未初始化")
            return None
        
        if len(left_points) != len(right_points):
            print(f"左右图像点数不匹配: 左{len(left_points)}, 右{len(right_points)}")
            return None
        
        if len(left_points) < 4:
            print(f"点数不足，需要至少4个点，当前有{len(left_points)}个")
            return None
        
        # 转换为numpy数组
        left_pts = np.array(left_points, dtype=np.float32)
        right_pts = np.array(right_points, dtype=np.float32)
        
        # 执行三角测量
        points_3d = self.stereo_triangulator.triangulate_points(left_pts, right_pts)
        # print(points_3d[0], points_3d[1])
        
        return points_3d
    
    def perform_localization(self, left_points, right_points):
        """
        完整的双目定位流程：三角测量 + PnP求解（如果提供了世界坐标点）
        
        Args:
            left_points: 左图像像素坐标
            right_points: 右图像像素坐标
        
        Returns:
            points_3d: 3D坐标（相机坐标系）
            rvec: 旋转向量（如果进行了PnP求解）
            tvec: 平移向量（如果进行了PnP求解）
        """
        # 步骤1: 三角测量
        points_3d = self.perform_triangulation(left_points, right_points)
        
        if points_3d is None:
            return None, None, None
        
        # 步骤2: 如果有世界坐标点，进行PnP求解
        rvec, tvec = None, None
        if self.world_points is not None and len(self.world_points) >= 4 and 0:
            if len(points_3d) >= 4:
                success, _, rvec, tvec = self.stereo_triangulator.solve_pnp_from_image_points(
                    np.array(left_points, dtype=np.float32),
                    np.array(right_points, dtype=np.float32),
                    self.world_points[:len(points_3d)]  # 只使用匹配的点数
                )
                
                if success and rvec is not None:
                    print("PnP求解成功")
                else:
                    print("PnP求解失败")
            else:
                print(f"3D点数({len(points_3d)})不足，无法进行PnP求解")
        
        return points_3d, rvec, tvec
    
    def extract_feature_points(self, image):
        """
        提取图像中的关键点（RGB点）
        这是后续需要实现的功能
        
        Args:
            image: 输入图像 (numpy array, BGR格式)
            
        Returns:
            feature_points: 检测到的关键点列表
        """
        # TODO: 实现关键点检测逻辑
        feature_points = []
        return feature_points
    
    def cleanup(self):
        """清理资源"""
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped, PointStamped
    import cv2
    
    # 初始化ROS节点
    rospy.init_node('dual_camera_detector', anonymous=True)
    
    # CV桥接器
    bridge = CvBridge()
    
    # 使用列表存储图像，避免nonlocal作用域问题
    image_data = {
        'left': None,
        'right': None,
        'left_received': False,
        'right_received': False
    }
    
    # 图像话题（根据zed2.xacro配置）
    # 可以订阅原始图像或矫正图像
    left_topic = rospy.get_param('~left_image_topic', '/zed2/left_raw/image_raw_color')
    right_topic = rospy.get_param('~right_image_topic', '/zed2/right_raw/image_raw_color')
    
    rospy.loginfo("订阅左相机话题: {}".format(left_topic))
    rospy.loginfo("订阅右相机话题: {}".format(right_topic))
    
    # 创建检测器实例
    detector = DualCameraDetector()
    
    # 创建位姿发布器
    pose_pub = rospy.Publisher('/stereo_vision/pose', PoseStamped, queue_size=1)
    points_3d_pub = rospy.Publisher('/stereo_vision/points_3d', PointStamped, queue_size=1)
    
    def left_image_callback(msg):
        """左相机图像回调函数"""
        try:
            # 尝试使用cv_bridge转换，如果失败则使用numpy直接解析
            try:
                image_data['left'] = bridge.imgmsg_to_cv2(msg, "bgr8")
            except:
                # 使用numpy直接解析图像数据（绕过cv_bridge的系统库问题）
                if msg.encoding in ['rgb8', 'bgr8']:
                    image_data['left'] = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                    if msg.encoding == 'rgb8':
                        # 如果是rgb8，需要转换为bgr8
                        image_data['left'] = cv2.cvtColor(image_data['left'], cv2.COLOR_RGB2BGR)
                else:
                    rospy.logwarn("不支持的图像编码格式: {}".format(msg.encoding))
                    return
            
            if not image_data['left_received']:
                rospy.loginfo("成功接收左相机图像，尺寸: {}".format(image_data['left'].shape))
                image_data['left_received'] = True
        except Exception as e:
            rospy.logerr("处理左相机图像时出错: {}".format(e))
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def right_image_callback(msg):
        """右相机图像回调函数"""
        try:
            # 尝试使用cv_bridge转换，如果失败则使用numpy直接解析
            try:
                image_data['right'] = bridge.imgmsg_to_cv2(msg, "bgr8")
            except:
                # 使用numpy直接解析图像数据（绕过cv_bridge的系统库问题）
                if msg.encoding in ['rgb8', 'bgr8']:
                    image_data['right'] = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                    if msg.encoding == 'rgb8':
                        # 如果是rgb8，需要转换为bgr8
                        image_data['right'] = cv2.cvtColor(image_data['right'], cv2.COLOR_RGB2BGR)
                else:
                    rospy.logwarn("不支持的图像编码格式: {}".format(msg.encoding))
                    return
            
            if not image_data['right_received']:
                rospy.loginfo("成功接收右相机图像，尺寸: {}".format(image_data['right'].shape))
                image_data['right_received'] = True
        except Exception as e:
            rospy.logerr("处理右相机图像时出错: {}".format(e))
            import traceback
            rospy.logerr(traceback.format_exc())
    
    # 订阅左右相机图像话题
    rospy.Subscriber(left_topic, Image, left_image_callback)
    rospy.Subscriber(right_topic, Image, right_image_callback)
    
    rospy.loginfo("双目相机检测器已启动，等待图像...")
    rospy.loginfo("提示: 如果长时间没有图像，请检查话题名称是否正确")
    rospy.loginfo("可以使用 'rostopic list | grep zed2' 查看可用的zed2话题")
    
    # 计数器，用于定期输出状态信息
    loop_count = 0
    
    # 主循环：获取图像并传递给detector
    rate = rospy.Rate(30)  # 30Hz
    try:
        while not rospy.is_shutdown():
            # 当左右图像都准备好时，传递给detector显示
            if image_data['left'] is not None and image_data['right'] is not None:
                # time.sleep(0.1)
                detector.right_feature_points, image_data['right'] = detector.right_detector.extract_feature_points(image_data['right'])
                detector.left_feature_points, image_data['left'] = detector.left_detector.extract_feature_points(image_data['left'])
                # print(detector.left_feature_points)
                # print(detector.right_feature_points)
                # print(detector.left_feature_points[0][0]-detector.right_feature_points[0][0])
                # 如果检测到足够的特征点，进行三角测量和定位
                if len(detector.left_feature_points) >= 4 and len(detector.right_feature_points) >= 4:
                    # 确保左右图像的点一一对应
                    min_points = min(len(detector.left_feature_points), len(detector.right_feature_points))
                    left_pts = detector.left_feature_points[:min_points]
                    right_pts = detector.right_feature_points[:min_points]
                    
                    # 执行定位
                    points_3d, rvec, tvec = detector.perform_localization(left_pts, right_pts)
                    print(points_3d[0])
                    
                    if points_3d is not None:
                        # 打印3D坐标信息
                        rospy.loginfo_throttle(1.0, f"三角测量成功，获得 {len(points_3d)} 个3D点")
                        for i, pt in enumerate(points_3d):
                            rospy.loginfo_throttle(1.0, f"点{i+1}: ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}) 米")
                        
                        # 发布3D点（以第一个点为例，或可以发布所有点）
                        if len(points_3d) > 0:
                            point_msg = PointStamped()
                            point_msg.header.frame_id = "zed2_left_camera_optical_frame"
                            point_msg.header.stamp = rospy.Time.now()
                            point_msg.point.x = float(points_3d[0][0])
                            point_msg.point.y = float(points_3d[0][1])
                            point_msg.point.z = float(points_3d[0][2])
                            points_3d_pub.publish(point_msg)
                        
                        # 如果进行了PnP求解，发布位姿信息
                        if rvec is not None and tvec is not None:
                            # 将旋转向量转换为旋转矩阵
                            R, _ = cv2.Rodrigues(rvec)
                            
                            # 创建位姿消息
                            pose_msg = PoseStamped()
                            pose_msg.header.frame_id = "zed2_left_camera_optical_frame"
                            pose_msg.header.stamp = rospy.Time.now()
                            
                            # 设置位置（tvec是从世界坐标系到相机坐标系的平移）
                            pose_msg.pose.position.x = float(tvec[0][0])
                            pose_msg.pose.position.y = float(tvec[1][0])
                            pose_msg.pose.position.z = float(tvec[2][0])
                            
                            # 将旋转矩阵转换为四元数
                            trace = np.trace(R)
                            if trace > 0:
                                s = np.sqrt(trace + 1.0) * 2
                                w = 0.25 * s
                                x = (R[2, 1] - R[1, 2]) / s
                                y = (R[0, 2] - R[2, 0]) / s
                                z = (R[1, 0] - R[0, 1]) / s
                            elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                                w = (R[2, 1] - R[1, 2]) / s
                                x = 0.25 * s
                                y = (R[0, 1] + R[1, 0]) / s
                                z = (R[0, 2] + R[2, 0]) / s
                            elif R[1, 1] > R[2, 2]:
                                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                                w = (R[0, 2] - R[2, 0]) / s
                                x = (R[0, 1] + R[1, 0]) / s
                                y = 0.25 * s
                                z = (R[1, 2] + R[2, 1]) / s
                            else:
                                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                                w = (R[1, 0] - R[0, 1]) / s
                                x = (R[0, 2] + R[2, 0]) / s
                                y = (R[1, 2] + R[2, 1]) / s
                                z = 0.25 * s
                            
                            pose_msg.pose.orientation.w = float(w)
                            pose_msg.pose.orientation.x = float(x)
                            pose_msg.pose.orientation.y = float(y)
                            pose_msg.pose.orientation.z = float(z)
                            
                            pose_pub.publish(pose_msg)
                            rospy.loginfo_throttle(1.0, f"位姿估计成功: tvec=({tvec[0][0]:.3f}, {tvec[1][0]:.3f}, {tvec[2][0]:.3f})")
                
                detector.display_images(image_data['left'], image_data['right'])
                


            else:
                # 每300次循环（约10秒）输出一次状态
                loop_count += 1
                if loop_count % 300 == 0:
                    if image_data['left'] is None:
                        rospy.logwarn("仍在等待左相机图像...")
                    if image_data['right'] is None:
                        rospy.logwarn("仍在等待右相机图像...")
            rate.sleep()
    except KeyboardInterrupt:
        rospy.loginfo("正在关闭检测器...")
    finally:
        detector.cleanup()
