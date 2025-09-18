#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def image_callback(data):
    try:
        # 创建CV桥接器
        bridge = CvBridge()
        
        # 将ROS图像消息转换为OpenCV格式
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        
        # 显示图像
        cv2.imshow("Camera Image", cv_image)
        
        # 等待按键，如果按下'q'则退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rospy.loginfo("用户按下 'q' 键，退出程序")
            rospy.signal_shutdown("用户退出")
            
    except Exception as e:
        rospy.logerr(f"处理图像时出错: {e}")

def main():
    rospy.init_node('system_opencv_viewer', anonymous=True)
    
    # 订阅相机图像话题
    rospy.Subscriber('/camera/image_raw', Image, image_callback)
    
    rospy.loginfo("系统OpenCV图像显示节点已启动")
    rospy.loginfo("订阅话题: /camera/image_raw")
    rospy.loginfo("按 'q' 键退出显示")
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("程序被用户中断")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
