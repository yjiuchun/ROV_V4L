#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
扩展卡尔曼滤波(EKF)视觉定位系统
用于优化PnP解算结果，处理视觉标签移动时的定位误差
"""

import rospy
import numpy as np
import yaml
import os
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped
from std_msgs.msg import Header
import tf2_ros

class ExtendedKalmanFilter:
    def __init__(self, config_file=None):
        """初始化扩展卡尔曼滤波器"""
        rospy.init_node('ekf_visual_localization', anonymous=True)
        
        # 加载配置
        self.load_config(config_file)
        
        # 状态向量: [x, y, z, vx, vy, vz, qx, qy, qz, qw] (10维)
        # 位置(x,y,z), 速度(vx,vy,vz), 四元数(qx,qy,qz,qw)
        self.state_dim = 10
        self.obs_dim = 7  # 观测维度: [x, y, z, qx, qy, qz, qw]
        
        # 初始化状态向量
        self.x = np.zeros(self.state_dim)
        self.x[6] = 1.0  # 四元数w分量为1
        
        # 初始化协方差矩阵
        self.P = np.eye(self.state_dim) * 0.1
        
        # 过程噪声协方差矩阵
        self.Q = np.eye(self.state_dim) * 0.01
        
        # 观测噪声协方差矩阵
        self.R = np.eye(self.obs_dim) * 0.1
        
        # 时间相关
        self.last_time = None
        self.dt = 0.0
        
        # ROS相关
        self.setup_ros()
        
        # 标志位
        self.initialized = False
        self.pose_received = False
        
        rospy.loginfo("EKF视觉定位系统已启动")
        rospy.loginfo(f"状态维度: {self.state_dim}, 观测维度: {self.obs_dim}")
    
    def load_config(self, config_file):
        """加载配置文件"""
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), '../config/ekf_config.yaml')
        
        try:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
            
            # 加载EKF参数
            ekf_params = config.get('ekf_params', {})
            self.process_noise_pos = ekf_params.get('process_noise_position', 0.01)
            self.process_noise_vel = ekf_params.get('process_noise_velocity', 0.1)
            self.process_noise_ori = ekf_params.get('process_noise_orientation', 0.01)
            
            self.obs_noise_pos = ekf_params.get('observation_noise_position', 0.1)
            self.obs_noise_ori = ekf_params.get('observation_noise_orientation', 0.1)
            
            # 初始化协方差
            self.init_cov_pos = ekf_params.get('initial_covariance_position', 1.0)
            self.init_cov_vel = ekf_params.get('initial_covariance_velocity', 1.0)
            self.init_cov_ori = ekf_params.get('initial_covariance_orientation', 1.0)
            
            # 更新频率
            self.update_rate = ekf_params.get('update_rate', 30.0)
            
            rospy.loginfo("EKF配置文件加载成功")
            
        except Exception as e:
            rospy.logwarn(f"加载配置文件失败: {e}，使用默认参数")
            self.set_default_config()
    
    def set_default_config(self):
        """设置默认配置"""
        self.process_noise_pos = 0.01
        self.process_noise_vel = 0.1
        self.process_noise_ori = 0.01
        
        self.obs_noise_pos = 0.1
        self.obs_noise_ori = 0.1
        
        self.init_cov_pos = 1.0
        self.init_cov_vel = 1.0
        self.init_cov_ori = 1.0
        
        self.update_rate = 30.0
    
    def setup_ros(self):
        """设置ROS话题和参数"""
        # 订阅PnP解算结果
        self.pnp_pose_sub = rospy.Subscriber(
            '/vision/estimated_pose', 
            PoseStamped, 
            self.pnp_pose_callback
        )
        
        # 发布EKF优化后的位姿
        self.ekf_pose_pub = rospy.Publisher(
            '/vision/ekf_pose', 
            PoseStamped, 
            queue_size=10
        )
        
        # 发布带协方差的位姿
        self.ekf_pose_cov_pub = rospy.Publisher(
            '/vision/ekf_pose_covariance', 
            PoseWithCovarianceStamped, 
            queue_size=10
        )
        
        # 发布速度估计
        self.velocity_pub = rospy.Publisher(
            '/vision/estimated_velocity', 
            TwistStamped, 
            queue_size=10
        )
        
        # TF相关
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # 参数
        self.source_frame = rospy.get_param('~source_frame', 'camera_link')
        self.target_frame = rospy.get_param('~target_frame', 'target_link')
    
    def pnp_pose_callback(self, msg):
        """PnP位姿回调函数"""
        try:
            # 提取位置和姿态
            position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            orientation = [msg.pose.orientation.x, msg.pose.orientation.y, 
                          msg.pose.orientation.z, msg.pose.orientation.w]
            
            # 构建观测向量
            z = np.array(position + orientation)
            
            # 计算时间间隔
            current_time = rospy.Time.now().to_sec()
            if self.last_time is not None:
                self.dt = current_time - self.last_time
            else:
                self.dt = 1.0 / self.update_rate
            
            self.last_time = current_time
            
            # 执行EKF更新
            self.update(z)
            
            # 发布结果
            self.publish_results()
            
            self.pose_received = True
            
        except Exception as e:
            rospy.logerr(f"处理PnP位姿时出错: {e}")
    
    def predict(self):
        """预测步骤"""
        if self.dt <= 0:
            return
        
        # 状态转移矩阵 F
        F = self.get_state_transition_matrix()
        
        # 预测状态
        self.x = F @ self.x
        
        # 预测协方差
        self.P = F @ self.P @ F.T + self.Q
        
        # 确保四元数归一化
        self.normalize_quaternion()
    
    def update(self, z):
        """更新步骤"""
        # 预测
        self.predict()
        
        # 观测模型
        h = self.observation_model()
        H = self.observation_jacobian()
        
        # 计算残差
        y = z - h
        
        # 计算卡尔曼增益
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 更新状态
        self.x = self.x + K @ y
        
        # 更新协方差
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P
        
        # 确保四元数归一化
        self.normalize_quaternion()
        
        # 标记为已初始化
        if not self.initialized:
            self.initialized = True
            rospy.loginfo("EKF已初始化")
    
    def get_state_transition_matrix(self):
        """获取状态转移矩阵"""
        F = np.eye(self.state_dim)
        
        # 位置更新: x = x + vx * dt
        F[0, 3] = self.dt  # x += vx * dt
        F[1, 4] = self.dt  # y += vy * dt
        F[2, 5] = self.dt  # z += vz * dt
        
        # 速度保持不变（假设匀速运动）
        # 姿态保持不变（假设无旋转）
        
        return F
    
    def observation_model(self):
        """观测模型 h(x)"""
        # 直接观测位置和姿态
        h = np.zeros(self.obs_dim)
        h[0:3] = self.x[0:3]  # 位置
        h[3:7] = self.x[6:10]  # 四元数
        
        return h
    
    def observation_jacobian(self):
        """观测雅可比矩阵 H"""
        H = np.zeros((self.obs_dim, self.state_dim))
        
        # 位置观测
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # z
        
        # 姿态观测
        H[3, 6] = 1.0  # qx
        H[4, 7] = 1.0  # qy
        H[5, 8] = 1.0  # qz
        H[6, 9] = 1.0  # qw
        
        return H
    
    def normalize_quaternion(self):
        """归一化四元数"""
        q_norm = np.linalg.norm(self.x[6:10])
        if q_norm > 1e-6:
            self.x[6:10] = self.x[6:10] / q_norm
    
    def update_noise_matrices(self):
        """更新噪声矩阵"""
        # 过程噪声矩阵
        self.Q = np.zeros((self.state_dim, self.state_dim))
        
        # 位置过程噪声
        self.Q[0, 0] = self.process_noise_pos * self.dt**2
        self.Q[1, 1] = self.process_noise_pos * self.dt**2
        self.Q[2, 2] = self.process_noise_pos * self.dt**2
        
        # 速度过程噪声
        self.Q[3, 3] = self.process_noise_vel * self.dt
        self.Q[4, 4] = self.process_noise_vel * self.dt
        self.Q[5, 5] = self.process_noise_vel * self.dt
        
        # 姿态过程噪声
        self.Q[6, 6] = self.process_noise_ori * self.dt
        self.Q[7, 7] = self.process_noise_ori * self.dt
        self.Q[8, 8] = self.process_noise_ori * self.dt
        self.Q[9, 9] = self.process_noise_ori * self.dt
        
        # 观测噪声矩阵
        self.R = np.zeros((self.obs_dim, self.obs_dim))
        
        # 位置观测噪声
        self.R[0, 0] = self.obs_noise_pos
        self.R[1, 1] = self.obs_noise_pos
        self.R[2, 2] = self.obs_noise_pos
        
        # 姿态观测噪声
        self.R[3, 3] = self.obs_noise_ori
        self.R[4, 4] = self.obs_noise_ori
        self.R[5, 5] = self.obs_noise_ori
        self.R[6, 6] = self.obs_noise_ori
    
    def publish_results(self):
        """发布EKF结果"""
        if not self.initialized:
            return
        
        # 更新噪声矩阵
        self.update_noise_matrices()
        
        # 发布优化后的位姿
        self.publish_ekf_pose()
        
        # 发布带协方差的位姿
        self.publish_ekf_pose_covariance()
        
        # 发布速度估计
        self.publish_velocity()
        
        # 发布TF变换
        self.publish_tf()
    
    def publish_ekf_pose(self):
        """发布EKF优化后的位姿"""
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.source_frame
        
        # 位置
        msg.pose.position.x = self.x[0]
        msg.pose.position.y = self.x[1]
        msg.pose.position.z = self.x[2]
        
        # 姿态
        msg.pose.orientation.x = self.x[6]
        msg.pose.orientation.y = self.x[7]
        msg.pose.orientation.z = self.x[8]
        msg.pose.orientation.w = self.x[9]
        
        self.ekf_pose_pub.publish(msg)
    
    def publish_ekf_pose_covariance(self):
        """发布带协方差的位姿"""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.source_frame
        
        # 位姿
        msg.pose.pose.position.x = self.x[0]
        msg.pose.pose.position.y = self.x[1]
        msg.pose.pose.position.z = self.x[2]
        msg.pose.pose.orientation.x = self.x[6]
        msg.pose.pose.orientation.y = self.x[7]
        msg.pose.pose.orientation.z = self.x[8]
        msg.pose.pose.orientation.w = self.x[9]
        
        # 协方差矩阵 (6x6: x,y,z,rx,ry,rz)
        cov_matrix = np.zeros(36)
        
        # 位置协方差
        cov_matrix[0] = self.P[0, 0]   # x-x
        cov_matrix[7] = self.P[1, 1]   # y-y
        cov_matrix[14] = self.P[2, 2]  # z-z
        
        # 姿态协方差 (简化处理)
        cov_matrix[21] = self.P[6, 6]  # rx-rx
        cov_matrix[28] = self.P[7, 7]  # ry-ry
        cov_matrix[35] = self.P[8, 8]  # rz-rz
        
        msg.pose.covariance = cov_matrix
        self.ekf_pose_cov_pub.publish(msg)
    
    def publish_velocity(self):
        """发布速度估计"""
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.source_frame
        
        # 线速度
        msg.twist.linear.x = self.x[3]
        msg.twist.linear.y = self.x[4]
        msg.twist.linear.z = self.x[5]
        
        # 角速度 (简化处理，设为0)
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        
        self.velocity_pub.publish(msg)
    
    def publish_tf(self):
        """发布TF变换"""
        try:
            from geometry_msgs.msg import TransformStamped
            
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = self.source_frame
            transform.child_frame_id = self.target_frame
            
            # 位置
            transform.transform.translation.x = self.x[0]
            transform.transform.translation.y = self.x[1]
            transform.transform.translation.z = self.x[2]
            
            # 姿态
            transform.transform.rotation.x = self.x[6]
            transform.transform.rotation.y = self.x[7]
            transform.transform.rotation.z = self.x[8]
            transform.transform.rotation.w = self.x[9]
            
            self.tf_broadcaster.sendTransform(transform)
            
        except Exception as e:
            rospy.logerr(f"发布TF变换失败: {e}")
    
    def get_state_info(self):
        """获取状态信息"""
        if not self.initialized:
            return "EKF未初始化"
        
        pos_uncertainty = np.sqrt(np.trace(self.P[0:3, 0:3]))
        vel_uncertainty = np.sqrt(np.trace(self.P[3:6, 3:6]))
        ori_uncertainty = np.sqrt(np.trace(self.P[6:10, 6:10]))
        
        info = f"""
EKF状态信息:
位置: ({self.x[0]:.3f}, {self.x[1]:.3f}, {self.x[2]:.3f})
速度: ({self.x[3]:.3f}, {self.x[4]:.3f}, {self.x[5]:.3f})
姿态: ({self.x[6]:.3f}, {self.x[7]:.3f}, {self.x[8]:.3f}, {self.x[9]:.3f})
位置不确定性: {pos_uncertainty:.3f}
速度不确定性: {vel_uncertainty:.3f}
姿态不确定性: {ori_uncertainty:.3f}
        """
        return info
    
    def run(self):
        """主运行循环"""
        rate = rospy.Rate(self.update_rate)
        
        rospy.loginfo("EKF视觉定位系统开始运行...")
        
        try:
            while not rospy.is_shutdown():
                # 如果没有接收到位姿数据，执行预测步骤
                if self.initialized and not self.pose_received:
                    self.predict()
                    self.publish_results()
                
                # 定期打印状态信息
                if self.initialized and rospy.Time.now().to_sec() % 5 < 0.1:
                    rospy.loginfo(self.get_state_info())
                
                self.pose_received = False
                rate.sleep()
                
        except KeyboardInterrupt:
            rospy.loginfo("EKF系统已停止")
        except Exception as e:
            rospy.logerr(f"EKF系统运行出错: {e}")

def main():
    """主函数"""
    try:
        ekf = ExtendedKalmanFilter()
        ekf.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("程序被中断")
    except Exception as e:
        rospy.logerr(f"程序运行出错: {e}")

if __name__ == '__main__':
    main()
