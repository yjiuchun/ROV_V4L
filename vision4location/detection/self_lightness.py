import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os


class SelfLightness:
    def __init__(self, config_name='self_lightness.yaml', folder='/home/yjc/Project/rov_ws/src/vision4location/detection/config/',show_image=False):
        self.min_area = 10
        self.kernel_size = 0
        self.show_image = show_image
        self.box=np.eye(4,4) # 4x4的单位矩阵
        # 加载配置
        self.load_config(config_name,folder)

    def load_config(self,config_name,folder):
        pass
    def get_lightness_peak(self,image):
        # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = image
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        hist_flat = hist.flatten()  # 将hist转换为1D数组

        # 找出极值对应的强度值
        max_count = np.max(hist)
        min_count = np.min(hist)
        max_intensity = np.argmax(hist)
        min_intensity = np.argmin(hist)

        # 检测所有峰值
        # height: 峰值的最小高度（设为最大值的5%以避免噪声）
        # distance: 峰值之间的最小距离（设为5，避免检测到相邻的微小波动）
        # prominence: 峰值的最小突出度（设为最大值的3%）
        min_height = max_count * 0.05
        min_distance = 5
        min_prominence = max_count * 0.03

        peaks, properties = find_peaks(hist_flat, 
                                    height=min_height,
                                    distance=min_distance,
                                    prominence=min_prominence)

        peak_intensities = peaks
        peak_counts = hist_flat[peaks]
        if 0:
            # 绘制并保存直方图
            plt.figure(figsize=(10, 6))
            plt.plot(hist, color='black')
            plt.title('Grayscale Intensity Histogram', fontsize=14)
            plt.xlabel('Pixel Intensity Value', fontsize=12)
            plt.ylabel('Pixel Count', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 255])
            # 标注所有峰值点
            if len(peak_intensities) > 0:
                plt.plot(peak_intensities, peak_counts, 'ro', markersize=6, label=f'Peaks ({len(peak_intensities)} found)')
                # 标注每个峰值
                for intensity, count in zip(peak_intensities, peak_counts):
                    plt.annotate(f'I={intensity}\nC={int(count)}', 
                                xy=(intensity, count), 
                                xytext=(intensity + 15, count + max_count*0.05),
                                arrowprops=dict(arrowstyle='->', color='red', lw=1.0, alpha=0.7),
                                fontsize=8, color='red', weight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

            # 标注全局最大值点（用不同颜色区分）
            plt.plot(max_intensity, max_count, 'g*', markersize=12, label=f'Global Max: I={max_intensity}')

            # 标注最小值点（如果最小值不为0，也标注出来）
            if min_count > 0:
                plt.plot(min_intensity, min_count, 'bo', markersize=8, label=f'Min: Intensity={min_intensity}, Count={int(min_count)}')
                plt.annotate(f'Min: I={min_intensity}\nC={int(min_count)}', 
                            xy=(min_intensity, min_count), 
                            xytext=(min_intensity + 20, min_count + max_count*0.05),
                            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                            fontsize=10, color='blue', weight='bold')

            plt.legend(loc='upper right', fontsize=9)

            # 保存直方图
            hist_output_path = os.path.join('./', 'four_light_histogram.png')
            plt.savefig(hist_output_path, dpi=150, bbox_inches='tight')
            print(f"直方图已保存到: {hist_output_path}")

            plt.close()

            print("处理完成！")
        # print(peaks)
        return max(peaks) 

    def extract_feature_points(self,image):
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.threshold = self.get_lightness_peak(gray_img) + 50
        # print(self.threshold)
        # 阈值分割
        _, binary_img = cv2.threshold(gray_img, self.threshold, 255, cv2.THRESH_BINARY)

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
                print(f"{order} - 圆形中心点: {circle['center']}, 半径: {circle['radius']}, 面积: {circle['area']:.2f}, 圆度: {circle['circularity']:.3f}")
        else:
            print(f"警告: 只检测到 {len(sorted_circles)} 个圆形，期望4个")
            for i, circle in enumerate(sorted_circles):
                center_points.append(circle['center'])
                order = "第1个(面积最大)" if i == 0 else f"第{i+1}个(顺时针)"
                print(f"{order} - 圆形中心点: {circle['center']}, 半径: {circle['radius']}, 面积: {circle['area']:.2f}, 圆度: {circle['circularity']:.3f}")
        
        # 转换为numpy数组格式（与其他detector保持一致）
        feature_points = np.array(center_points, dtype=np.float32) if len(center_points) > 0 else np.array([], dtype=np.float32).reshape(0, 2)
        
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
            cv2.imshow("Detected Circles", result_img)
            cv2.waitKey(10)

        # cv2.imwrite("./binary_img.jpg", binary_img)
        # cv2.waitKey(0)
        
        return feature_points, binary_img


if __name__ == '__main__':
    self_lightness = SelfLightness(show_image=True)
    image_path = '/home/yjc/Project/rov_ws/demo/images/20251110-154306.jpg'
    image = cv2.imread(image_path)
    feature_points, binary_img = self_lightness.extract_feature_points(image)
    print(f"检测到的特征点坐标: {feature_points}")