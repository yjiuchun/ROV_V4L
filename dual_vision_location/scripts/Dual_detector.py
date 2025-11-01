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


class DualCameraDetector:
    def __init__(self):
        """初始化双目相机检测器"""
        # 窗口名称
        self.window_name = "ZED2 Dual Camera View"
        self.left_detector = Detector(config_name='left_detector.yaml',folder=os.path.dirname(current_dir))
        self.right_detector = Detector(config_name='right_detector.yaml',folder=os.path.dirname(current_dir))

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
            cv2.waitKey(1)
            
        except cv2.error as e:
            print("OpenCV显示错误: {}".format(e))
        except Exception as e:
            print("显示图像时出错: {}".format(e))
            import traceback
            traceback.print_exc()
    
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
                image_data['left'] = detector.left_detector.extract_feature_points(image_data['left'])
                image_data['right'] = detector.right_detector.extract_feature_points(image_data['right'])
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
