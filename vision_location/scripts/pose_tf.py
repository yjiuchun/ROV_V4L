from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
import tf2_geometry_msgs
import cv2
import rospy
import numpy as np


class PoseTF:
    def __init__(self):
        # rospy.init_node('pose_tf', anonymous=True)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.pose = PoseStamped()
        self.pose_pub = rospy.Publisher('/vision/ekf_pose', PoseStamped, queue_size=1)

        # TF 广播器
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

    
    def pose_process(self, rvec, tvec):

        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        
        # 创建位姿消息
        self.pose.header.frame_id = "camera_link"
        self.pose.header.stamp = rospy.Time.now()
        
        # 设置位置
        self.pose.pose.position.x = tvec[2][0]
        self.pose.pose.position.y = -tvec[0][0]
        self.pose.pose.position.z = tvec[1][0]
        
        # 设置姿态 (旋转矩阵转四元数)
        self.pose.pose.orientation = self.rotation_matrix_to_quaternion(R)
        
        # 发布位姿消息
        self.pose_pub.publish(self.pose)
        
        # 发布 TF 变换：camera_link -> target_link
        self.publish_target_tf(rvec, tvec)
        
        return self.pose
    
    def publish_target_tf(self, rvec, tvec):
        """发布 target_link 相对于 camera_link 的 TF 变换"""
        try:
            # 创建 TF 变换消息
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = "camera_link"
            transform.child_frame_id = "target_link"
            
            # 设置平移
            transform.transform.translation.x = tvec[2][0]
            transform.transform.translation.y = -tvec[0][0]
            transform.transform.translation.z = tvec[1][0]
            
            # 将旋转向量转换为四元数
            R, _ = cv2.Rodrigues(rvec)
            quaternion = self.rotation_matrix_to_quaternion(R)
            
            # 设置旋转
            transform.transform.rotation.x = quaternion.x
            transform.transform.rotation.y = quaternion.y
            transform.transform.rotation.z = quaternion.z
            transform.transform.rotation.w = quaternion.w
            
            # 发布 TF 变换
            self.tf_broadcaster.sendTransform(transform)
            
            rospy.loginfo(f"发布 TF 变换: camera_link -> target_link")
            rospy.loginfo(f"位置: ({tvec[0][0]:.3f}, {tvec[1][0]:.3f}, {tvec[2][0]:.3f})")
            
        except Exception as e:
            rospy.logerr(f"发布 TF 变换失败: {e}")
    

    def rotation_matrix_to_quaternion(self, R):
        """将旋转矩阵转换为四元数"""
        from geometry_msgs.msg import Quaternion
        
        # 使用数学方法将旋转矩阵转换为四元数
        try:
            # 计算四元数
            trace = np.trace(R)
            
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
                qw = 0.25 * s
                qx = (R[2, 1] - R[1, 2]) / s
                qy = (R[0, 2] - R[2, 0]) / s
                qz = (R[1, 0] - R[0, 1]) / s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
            
            q = Quaternion()
            q.x = qx
            q.y = qy
            q.z = qz
            q.w = qw
            
            return q
            
        except Exception as e:
            rospy.logwarn(f"四元数转换出错: {e}，使用默认四元数")
            q = Quaternion()
            q.w = 1.0  # 默认单位四元数
            return q