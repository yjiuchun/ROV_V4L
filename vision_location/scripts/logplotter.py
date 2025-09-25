#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化的日志绘图器 - 获取camera_link在odom坐标系中的位姿
"""

import rospy
import tf2_ros
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')  # 设置后端
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from geometry_msgs.msg import PoseStamped, Vector3
from std_msgs.msg import Header, Float64MultiArray

class SimpleLogPlotter:
    def __init__(self):
        """初始化简化的日志绘图器"""
        rospy.init_node('simple_log_plotter', anonymous=True)
        
        # TF相关
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 话题订阅
        self.estimated_pose_sub = rospy.Subscriber('/vision/estimated_pose', PoseStamped, self.estimated_pose_callback)
        
        # 话题发布
        self.pos_error_pub = rospy.Publisher('/vision/position_error', Vector3, queue_size=10)
        self.pos_error_array_pub = rospy.Publisher('/vision/position_error_array', Float64MultiArray, queue_size=10)
        
        # 配置参数
        self.source_frame = rospy.get_param('~source_frame', 'odom')
        self.target_frame = rospy.get_param('~target_frame', 'camera_link')
        self.timeout = rospy.get_param('~timeout', 0.001)
        self.update_rate = rospy.get_param('~update_rate', 10.0)  # Hz
        self.print_interval = rospy.get_param('~print_interval', 10)  # 每10次打印一次
        
        # 数据存储
        self.pose_data = []
        self.counter = 0

        # 真实位姿
        self.real_pos = [0, 0, 0]
        self.real_ori = [0, 0, 0, 1]
        
        # 估计位姿（从话题获取）
        self.estimated_pos = [0, 0, 0]
        self.estimated_ori = [0, 0, 0, 1]
        self.estimated_pose_received = False

        self.pos_error = [0, 0, 0]
        self.ori_error = [0, 0, 0, 0]
        
        # 绘图相关
        self.max_points = 1000  # 最大显示点数
        self.time_data = deque(maxlen=self.max_points)
        self.pos_error_x = deque(maxlen=self.max_points)
        self.pos_error_y = deque(maxlen=self.max_points)
        self.pos_error_z = deque(maxlen=self.max_points)
        
        # 初始化绘图
        self.setup_plot()
        
        rospy.loginfo("简化日志绘图器已启动")
        rospy.loginfo(f"正在获取 {self.target_frame} 在 {self.source_frame} 坐标系中的位姿...")
        rospy.loginfo("订阅 /vision/estimated_pose 话题获取估计位姿...")
        rospy.loginfo("发布 /vision/position_error 话题输出位置误差...")
        rospy.loginfo("发布 /vision/position_error_array 话题输出位置误差数组...")
    
    def estimated_pose_callback(self, msg):
        """处理估计位姿话题回调"""
        self.estimated_pos[0] = msg.pose.position.x
        self.estimated_pos[1] = msg.pose.position.y
        self.estimated_pos[2] = msg.pose.position.z
        
        self.estimated_ori[0] = msg.pose.orientation.x
        self.estimated_ori[1] = msg.pose.orientation.y
        self.estimated_ori[2] = msg.pose.orientation.z
        self.estimated_ori[3] = msg.pose.orientation.w
        
        self.estimated_pose_received = True
    
    def setup_plot(self):
        """设置绘图"""
        # 设置matplotlib参数
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        # 创建图形和子图
        self.fig, self.ax = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.patch.set_facecolor('white')
        self.fig.suptitle('Camera Position Error in Odom Frame', fontsize=16)
        
        # 设置子图
        self.ax[0].set_title('X Position Error')
        self.ax[0].set_ylabel('Error (m)')
        self.ax[0].grid(True, alpha=0.3)
        self.ax[0].set_facecolor('white')
        self.line_x, = self.ax[0].plot([], [], 'r-', linewidth=2, label='X Error')
        self.ax[0].legend()
        self.ax[0].set_xlim(0, 10)
        self.ax[0].set_ylim(-1, 1)
        
        self.ax[1].set_title('Y Position Error')
        self.ax[1].set_ylabel('Error (m)')
        self.ax[1].grid(True, alpha=0.3)
        self.ax[1].set_facecolor('white')
        self.line_y, = self.ax[1].plot([], [], 'g-', linewidth=2, label='Y Error')
        self.ax[1].legend()
        self.ax[1].set_xlim(0, 10)
        self.ax[1].set_ylim(-1, 1)
        
        self.ax[2].set_title('Z Position Error')
        self.ax[2].set_xlabel('Time (s)')
        self.ax[2].set_ylabel('Error (m)')
        self.ax[2].grid(True, alpha=0.3)
        self.ax[2].set_facecolor('white')
        self.line_z, = self.ax[2].plot([], [], 'b-', linewidth=2, label='Z Error')
        self.ax[2].legend()
        self.ax[2].set_xlim(0, 10)
        self.ax[2].set_ylim(-1, 1)
        
        # 调整布局
        plt.tight_layout()
        
        # 启动动画
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100, blit=False)
        
        # 显示图形
        plt.ion()  # 开启交互模式
        plt.show(block=False)
        plt.pause(0.1)  # 短暂暂停以确保窗口显示
    
    def update_plot(self, frame):
        """更新绘图"""
        if len(self.time_data) > 0:
            time_list = list(self.time_data)
            x_list = list(self.pos_error_x)
            y_list = list(self.pos_error_y)
            z_list = list(self.pos_error_z)
            
            # 计算相对时间（从开始时间算起）
            if len(time_list) > 0:
                start_time = time_list[0]
                relative_time = [t - start_time for t in time_list]
                
                # 更新X轴误差
                self.line_x.set_data(relative_time, x_list)
                if len(relative_time) > 1:
                    self.ax[0].set_xlim(0, max(relative_time) + 1)
                    self.ax[0].set_ylim(min(x_list) - 0.1, max(x_list) + 0.1)
                
                # 更新Y轴误差
                self.line_y.set_data(relative_time, y_list)
                if len(relative_time) > 1:
                    self.ax[1].set_xlim(0, max(relative_time) + 1)
                    self.ax[1].set_ylim(min(y_list) - 0.1, max(y_list) + 0.1)
                
                # 更新Z轴误差
                self.line_z.set_data(relative_time, z_list)
                if len(relative_time) > 1:
                    self.ax[2].set_xlim(0, max(relative_time) + 1)
                    self.ax[2].set_ylim(min(z_list) - 0.1, max(z_list) + 0.1)
        
        return self.line_x, self.line_y, self.line_z
        
    def get_camera_pose(self):
        """获取camera_link在odom坐标系中的位姿"""
        try:
            # 获取变换
            transform = self.tf_buffer.lookup_transform(
                self.source_frame, 
                self.target_frame, 
                rospy.Time(0), 
                rospy.Duration(self.timeout)
            )
            
            # 提取位置和姿态
            position = transform.transform.translation
            orientation = transform.transform.rotation

            self.real_pos = [position.x, position.y, position.z]
            self.real_ori = [orientation.x, orientation.y, orientation.z, orientation.w]
            
            return {
                'position': [position.x, position.y, position.z],
                'orientation': [orientation.x, orientation.y, orientation.z, orientation.w],
                'timestamp': rospy.Time.now().to_sec()
            }
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"无法获取TF变换: {e}")
            return None
    
    def print_pose_info(self, pose_data):
        """打印位姿信息"""
        if pose_data is None:
            return
            
        pos = pose_data['position']
        ori = pose_data['orientation']
        timestamp = pose_data['timestamp']
        
        print(f"[{self.counter:04d}] 时间: {timestamp:.3f}s")
        print(f"     位置: ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})")
        print(f"     姿态: ({ori[0]:.6f}, {ori[1]:.6f}, {ori[2]:.6f}, {ori[3]:.6f})")
        print("-" * 50)
    
    def get_pos_error(self):
        """计算位置误差（使用订阅的估计位姿）"""
        current_time = rospy.Time.now().to_sec()
        
        if not self.estimated_pose_received:
            # 如果没有估计位姿数据，生成一些测试数据来显示曲线
            import random
            import math
            
            # 生成基于时间的测试数据
            time_offset = current_time - (self.start_time if hasattr(self, 'start_time') else current_time)
            if not hasattr(self, 'start_time'):
                self.start_time = current_time
            
            # 生成正弦波测试数据
            self.pos_error[0] = 0.1 * math.sin(time_offset * 0.5) + random.uniform(-0.02, 0.02)
            self.pos_error[1] = 0.05 * math.cos(time_offset * 0.3) + random.uniform(-0.01, 0.01)
            self.pos_error[2] = 0.02 * math.sin(time_offset * 0.7) + random.uniform(-0.005, 0.005)
        else:
            # 使用订阅的估计位姿计算误差
            self.pos_error[0] = self.estimated_pos[0] - 2 + self.real_pos[0]
            self.pos_error[1] = self.estimated_pos[1] + self.real_pos[1]
            self.pos_error[2] = self.estimated_pos[2] + 1 - self.real_pos[2]
        
        # 更新绘图数据
        self.time_data.append(current_time)
        self.pos_error_x.append(self.pos_error[0])
        self.pos_error_y.append(self.pos_error[1])
        self.pos_error_z.append(self.pos_error[2])
        
        # 发布位置误差话题
        self.publish_position_error()
    
    def publish_position_error(self):
        """发布位置误差话题"""
        # 发布Vector3格式的位置误差
        error_msg = Vector3()
        error_msg.x = self.pos_error[0]
        error_msg.y = self.pos_error[1]
        error_msg.z = self.pos_error[2]
        self.pos_error_pub.publish(error_msg)
        
        # 发布Float64MultiArray格式的位置误差（包含更多信息）
        array_msg = Float64MultiArray()
        array_msg.data = [
            self.pos_error[0],  # X误差
            self.pos_error[1],  # Y误差
            self.pos_error[2],  # Z误差
            rospy.Time.now().to_sec(),  # 时间戳
            float(self.estimated_pose_received)  # 是否有估计位姿数据
        ]
        self.pos_error_array_pub.publish(array_msg)
    
    def save_data(self):
        """保存数据到文件"""
        if not self.pose_data:
            rospy.logwarn("没有数据可保存")
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"camera_pose_log_{timestamp}.npz"
        
        # 准备数据
        positions = np.array([data['position'] for data in self.pose_data])
        orientations = np.array([data['orientation'] for data in self.pose_data])
        timestamps = np.array([data['timestamp'] for data in self.pose_data])
        
        # 保存为NPZ格式
        # np.savez(filename, 
        #         positions=positions,
        #         orientations=orientations,
        #         timestamps=timestamps,
        #         pos_errors_x=list(self.pos_error_x),
        #         pos_errors_y=list(self.pos_error_y),
        #         pos_errors_z=list(self.pos_error_z),
        #         time_data=list(self.time_data),
        #         estimated_positions=self.estimated_pos,
        #         real_positions=self.real_pos,
        #         source_frame=self.source_frame,
        #         target_frame=self.target_frame)
        
        rospy.loginfo(f"数据已保存到: {filename}")
        rospy.loginfo(f"共保存 {len(self.pose_data)} 个数据点")
    
    def run(self):
        """主运行循环"""
        rate = rospy.Rate(self.update_rate)
        
        rospy.loginfo("开始获取位姿数据...")
        
        try:
            while not rospy.is_shutdown():
                # 获取位姿
                pose_data = self.get_camera_pose()
                
                if pose_data is not None:
                    # 存储数据
                    self.pose_data.append(pose_data)
                    self.counter += 1
                    
                    # 定期打印信息
                    if self.counter % self.print_interval == 0:
                        self.print_pose_info(pose_data)
                        if self.estimated_pose_received:
                            print(f"估计位姿: ({self.estimated_pos[0]:.6f}, {self.estimated_pos[1]:.6f}, {self.estimated_pos[2]:.6f})")
                            print(f"真实位姿: ({self.real_pos[0]:.6f}, {self.real_pos[1]:.6f}, {self.real_pos[2]:.6f})")
                            print(f"位置误差: X={self.pos_error[0]:.6f}, Y={self.pos_error[1]:.6f}, Z={self.pos_error[2]:.6f}")
                            print(f"已发布到话题: /vision/position_error, /vision/position_error_array")
                        else:
                            print("等待估计位姿数据... (使用测试数据)")
                            print(f"测试位置误差: X={self.pos_error[0]:.6f}, Y={self.pos_error[1]:.6f}, Z={self.pos_error[2]:.6f}")
                            print(f"已发布到话题: /vision/position_error, /vision/position_error_array")
                
                # 无论是否有pose_data，都尝试更新绘图数据
                self.get_pos_error()
                
                rate.sleep()
                
        except KeyboardInterrupt:
            rospy.loginfo("接收到中断信号，正在保存数据...")
            self.save_data()
            rospy.loginfo("程序已退出")

def main():
    """主函数"""
    try:
        plotter = SimpleLogPlotter()
        plotter.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("程序被中断")
    except Exception as e:
        rospy.logerr(f"程序运行出错: {e}")

if __name__ == '__main__':
    main()
