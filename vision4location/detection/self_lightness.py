# 修复conda环境中OpenCV的Qt插件问题（必须在导入cv2之前）
import os
import sys
# 尝试导入修复模块
try:
    # 添加src目录到路径
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import opencv_qt_fix
except ImportError:
    # 如果找不到修复模块，直接设置环境变量
    if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
        plugin_path = os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
        if 'cv2/qt/plugins' in plugin_path:
            paths = plugin_path.split(os.pathsep)
            paths = [p for p in paths if 'cv2/qt/plugins' not in p]
            if paths:
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.pathsep.join(paths)
            else:
                system_qt_path = '/usr/lib/x86_64-linux-gnu/qt5/plugins'
                if os.path.exists(system_qt_path):
                    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = system_qt_path
                else:
                    del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
scripts_dir = '/home/yjc/Project/rov_ws/src/vision4location/scripts'
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
from get_lightness_peak import GetLightnessPeak
from img_spilt import ImageSplitter

class SelfLightness:
    def __init__(self, config_name='self_lightness.yaml', folder='/home/yjc/Project/rov_ws/src/vision4location/detection/config/',show_image=False):
        self.min_area = 1
        self.kernel_size = 0
        self.show_image = show_image
        self.box=np.eye(4,4) # 4x4的单位矩阵
        self.img_size_factor = 0.02
        self.lightness_offset = 50
        # 加载配置
        self.load_config(config_name,folder)
        self.lightness_peak_detector = GetLightnessPeak()
        self.image_splitter = ImageSplitter()

    def load_config(self,config_name,folder):
        pass
    def get_binary_offset(self,image):
        height, width = image.shape[:2]
        print(height, width)
        return ((height+width) / 2 - self.lightness_offset) * self.img_size_factor
    
    def get_lightness_peak(self, image_or_path, output_path=None, save_image=False):
        """
        获取亮度峰值（返回图像的最大值强度）
        
        参数:
            image_or_path: 图片数组或图片路径
            output_path: 输出路径（用于保存直方图，可选）
            save_image: 是否保存直方图（已废弃，保留以兼容旧代码）
        
        返回:
            max_intensity: 图像的最大亮度强度值（0-255）
        """
        # 获取最大值点的x坐标列表（同时会保存max_value）
        max_x_list = self.lightness_peak_detector.get_lightness_peaks(image_or_path)
        
        # 返回图像的最大值强度
        return int(self.lightness_peak_detector.max_value) 
    def split_image(self,image):
        split_images = self.image_splitter.split_image(image)
        return split_images

    def binary_image(self,gray_img):

        self.threshold = self.get_lightness_peak(gray_img)-10+self.get_binary_offset(gray_img)
        # print(self.threshold)
        # 阈值分割
        _, binary_img = cv2.threshold(gray_img, self.threshold, 255, cv2.THRESH_BINARY)
        return binary_img
    def composite_image(self,top_left_binary_img, top_right_binary_img, bottom_left_binary_img, bottom_right_binary_img):
        composite_image = self.image_splitter.composite_image(top_left_binary_img, top_right_binary_img, bottom_left_binary_img, bottom_right_binary_img)
        return composite_image

    def extract_feature_points(self,image):
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.threshold = self.get_lightness_peak(gray_img)-self.get_binary_offset(gray_img)
        # print(self.threshold)
        # 阈值分割
        _, binary_img = cv2.threshold(gray_img, self.threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow("binary_img", binary_img)
        cv2.waitKey(0)
        # cv2.imwrite('/home/yjc/Project/binary_img.jpg', binary_img)

        # 形态学操作，去除噪声
        if self.kernel_size > 0:
            kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
            binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
            binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 存储圆形信息
        circles = []
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤小面积轮廓
            if area < self.min_area:
                continue
            
            # 计算轮廓的周长
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # 计算圆度 (4π*面积/周长²)，圆度接近1表示更接近圆形
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 过滤圆度较低的轮廓（圆度阈值可根据实际情况调整）
            if circularity < 0.1:
                continue
            
            # 拟合最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            circles.append({
                'center': center,
                'radius': radius,
                'area': area,
                'circularity': circularity,
                'contour': contour
            })
        
        # 按面积排序，选择最大的4个圆形
        circles.sort(key=lambda c: c['area'], reverse=True)
        circles = circles[:4]
        
        # 对点进行排序：面积最大的第一个，其余按顺时针顺序
        if len(circles) >= 2:
            # 第一个点是面积最大的（已经是第一个）
            first_circle = circles[0]
            first_center = np.array(first_circle['center'], dtype=np.float32)
            
            # 计算其余点相对于第一个点的角度（用于顺时针排序）
            remaining_circles = circles[1:]
            for circle in remaining_circles:
                center = np.array(circle['center'], dtype=np.float32)
                # 计算相对向量
                vec = center - first_center
                # 计算角度（atan2返回从x轴正方向逆时针的角度，范围[-π, π]）
                # 为了顺时针排序，我们使用 -atan2，这样角度从大到小就是顺时针
                angle = -np.arctan2(vec[1], vec[0])
                circle['angle'] = angle
            
            # 按角度从大到小排序（顺时针）
            remaining_circles.sort(key=lambda c: c['angle'], reverse=True)
            
            # 重新组合：第一个 + 按顺时针排序的其余点
            sorted_circles = [first_circle] + remaining_circles
        else:
            sorted_circles = circles
        
        # 提取中心点坐标（按排序后的顺序）
        center_points = []
        if len(sorted_circles) == 4:
            for i, circle in enumerate(sorted_circles):
                center_points.append(circle['center'])
                order = "第1个(面积最大)" if i == 0 else f"第{i+1}个(顺时针)"
                # print(f"{order} - 圆形中心点: {circle['center']}, 半径: {circle['radius']}, 面积: {circle['area']:.2f}, 圆度: {circle['circularity']:.3f}")
        else:
            print(f"警告: 只检测到 {len(sorted_circles)} 个圆形，期望4个")
            for i, circle in enumerate(sorted_circles):
                center_points.append(circle['center'])
                order = "第1个(面积最大)" if i == 0 else f"第{i+1}个(顺时针)"
                # print(f"{order} - 圆形中心点: {circle['center']}, 半径: {circle['radius']}, 面积: {circle['area']:.2f}, 圆度: {circle['circularity']:.3f}")
        
        # 转换为numpy数组格式（与其他detector保持一致）
        feature_points = np.array(center_points, dtype=np.float32) if len(center_points) > 0 else np.array([], dtype=np.float32).reshape(0, 2)
        centers = []
        # 可视化（可选）
        if self.show_image:
            result_img = image.copy()
            # 绘制所有通过过滤的轮廓（按排序后的顺序）
            for i, circle in enumerate(sorted_circles):
                contour = circle['contour']
                center = circle['center']
                radius = circle['radius']
                
                # 绘制轮廓（用不同颜色区分）
                color = (255, 0, 255) if i < 4 else (128, 128, 128)  # 前4个用紫色，其他用灰色
                cv2.drawContours(result_img, [contour], -1, color, 2)
                
                # 绘制最小外接圆
                cv2.circle(result_img, center, radius, (0, 255, 0), 2)
                
                # 绘制中心点
                cv2.circle(result_img, center, 3, (0, 0, 255), -1)
                
                # 标注中心点坐标和序号
                label = f"{i+1}:({center[0]},{center[1]})"
                cv2.putText(result_img, label, 
                           (center[0] + 10, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                centers.append(center)
            # cv2.imwrite('/home/yjc/Project/result_img.jpg', result_img)
            # cv2.imshow("Detected Circles", result_img)
            # cv2.waitKey(10)

        # cv2.imwrite("./binary_img.jpg", binary_img)
        # cv2.waitKey(0)
        
        return feature_points, binary_img,centers



if __name__ == '__main__':
    self_lightness = SelfLightness(show_image=True)
    image_path = '/home/yjc/Project/rov_ws/src/vision4location/src/image_save/crop_img/2/crop_img_152x144.jpg'
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    top_left_image, top_right_image, bottom_left_image, bottom_right_image = self_lightness.split_image(gray_image)
    top_left_binary_img = self_lightness.binary_image(top_left_image)
    top_right_binary_img = self_lightness.binary_image(top_right_image)
    bottom_left_binary_img = self_lightness.binary_image(bottom_left_image)
    bottom_right_binary_img = self_lightness.binary_image(bottom_right_image)
    composite_image = self_lightness.composite_image(top_left_binary_img, top_right_binary_img, bottom_left_binary_img, bottom_right_binary_img)
    cv2.imwrite("/home/yjc/Project/rov_ws/src/vision4location/src/image_save/crop_img/2/top_left_binary_img.jpg", top_left_binary_img)
    cv2.imwrite("/home/yjc/Project/rov_ws/src/vision4location/src/image_save/crop_img/2/top_right_binary_img.jpg", top_right_binary_img)
    cv2.imwrite("/home/yjc/Project/rov_ws/src/vision4location/src/image_save/crop_img/2/bottom_left_binary_img.jpg", bottom_left_binary_img)
    cv2.imwrite("/home/yjc/Project/rov_ws/src/vision4location/src/image_save/crop_img/2/bottom_right_binary_img.jpg", bottom_right_binary_img)
    cv2.imwrite("/home/yjc/Project/rov_ws/src/vision4location/src/image_save/crop_img/2/composite_image.jpg", composite_image)