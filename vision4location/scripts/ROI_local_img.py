import cv2
import numpy as np
from typing import List, Tuple, Union


class ROISegmenter:
    """
    ROI分割类，用于从图像中提取以指定点为中心的正方形或圆形区域
    
    功能：
    - 输入图像和四个点的像素坐标
    - 返回以这四个点为中心的正方形或圆形区域
    """
    
    def __init__(self, shape: str = 'square', size: Union[int, Tuple[int, int]] = 50):
        """
        初始化ROI分割器
        
        参数:
            shape: 区域形状，'square' 或 'circle'
            size: 区域大小
                - 对于正方形：可以是整数（边长）或元组(width, height)
                - 对于圆形：整数（半径）
        """
        self.shape = shape.lower()
        if self.shape not in ['square', 'circle']:
            raise ValueError("shape must be 'square' or 'circle'")
        
        self.size = size
    
    def extract_roi(self, image: np.ndarray, points: List[Tuple[int, int]], 
                    size: Union[int, Tuple[int, int]] = None,
                    side_length_or_diameter: Union[int, Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        从图像中提取以指定点为中心的ROI区域
        
        参数:
            image: 输入图像 (numpy array)
            points: 四个点的像素坐标列表，格式为 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            size: 区域大小，如果为None则使用初始化时设置的大小（已废弃，建议使用side_length_or_diameter）
                - 对于正方形：可以是整数（边长）或元组(width, height)
                - 对于圆形：整数（半径）
            side_length_or_diameter: 分割的边长或直径
                - 对于正方形：整数（边长）或元组(width, height)
                - 对于圆形：整数（直径）
                如果提供此参数，将优先使用此参数；如果为None，则使用size参数
        
        返回:
            roi_regions: 四个ROI区域的列表，每个区域是一个numpy数组
        """
        if len(points) != 4:
            raise ValueError("points must contain exactly 4 points")
        
        # 优先使用 side_length_or_diameter 参数
        if side_length_or_diameter is not None:
            if self.shape == 'square':
                # 正方形：直接使用边长
                actual_size = side_length_or_diameter
            else:  # circle
                # 圆形：将直径转换为半径
                if isinstance(side_length_or_diameter, tuple):
                    raise ValueError("圆形ROI的直径必须是整数，不能是元组")
                actual_size = side_length_or_diameter // 2
        elif size is not None:
            actual_size = size
        else:
            actual_size = self.size
        
        roi_regions = []
        h, w = image.shape[:2]
        
        for point in points:
            x, y = point
            
            if self.shape == 'square':
                roi = self._extract_square(image, x, y, actual_size, h, w)
            else:  # circle
                roi = self._extract_circle(image, x, y, actual_size, h, w)
            
            roi_regions.append(roi)
        
        return roi_regions
    
    def _extract_square(self, image: np.ndarray, center_x: int, center_y: int, 
                       size: Union[int, Tuple[int, int]], h: int, w: int) -> np.ndarray:
        """
        提取正方形ROI区域
        
        参数:
            image: 输入图像
            center_x: 中心点x坐标
            center_y: 中心点y坐标
            size: 正方形大小（整数或(width, height)元组）
            h: 图像高度
            w: 图像宽度
        
        返回:
            roi: 提取的ROI区域
        """
        if isinstance(size, tuple):
            half_width = size[0] // 2
            half_height = size[1] // 2
        else:
            half_width = size // 2
            half_height = size // 2
        
        # 计算边界，确保不超出图像范围
        x1 = max(0, center_x - half_width)
        y1 = max(0, center_y - half_height)
        x2 = min(w, center_x + half_width)
        y2 = min(h, center_y + half_height)
        
        # 提取ROI
        roi = image[y1:y2, x1:x2].copy()
        
        return roi
    
    def _extract_circle(self, image: np.ndarray, center_x: int, center_y: int, 
                       radius: int, h: int, w: int) -> np.ndarray:
        """
        提取圆形ROI区域
        
        参数:
            image: 输入图像
            center_x: 中心点x坐标
            center_y: 中心点y坐标
            radius: 圆形半径
            h: 图像高度
            w: 图像宽度
        
        返回:
            roi: 提取的ROI区域（带掩码的圆形区域）
        """
        # 计算边界框
        x1 = max(0, center_x - radius)
        y1 = max(0, center_y - radius)
        x2 = min(w, center_x + radius)
        y2 = min(h, center_y + radius)
        
        # 提取矩形区域
        roi_rect = image[y1:y2, x1:x2].copy()
        
        # 创建圆形掩码
        roi_h, roi_w = roi_rect.shape[:2]
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        
        # 计算在ROI矩形内的圆心坐标
        local_center_x = center_x - x1
        local_center_y = center_y - y1
        
        # 绘制圆形掩码
        cv2.circle(mask, (local_center_x, local_center_y), radius, 255, -1)
        
        # 应用掩码
        if len(roi_rect.shape) == 3:
            # 彩色图像
            mask_3d = mask[:, :, np.newaxis]
            roi = roi_rect * (mask_3d / 255.0)
            roi = roi.astype(roi_rect.dtype)
        else:
            # 灰度图像
            roi = roi_rect * (mask / 255.0)
            roi = roi.astype(roi_rect.dtype)
        
        return roi
    
    def extract_roi_with_mask(self, image: np.ndarray, points: List[Tuple[int, int]], 
                              size: Union[int, Tuple[int, int]] = None,
                              side_length_or_diameter: Union[int, Tuple[int, int]] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        提取ROI区域并返回掩码
        
        参数:
            image: 输入图像
            points: 四个点的像素坐标列表
            size: 区域大小（已废弃，建议使用side_length_or_diameter）
            side_length_or_diameter: 分割的边长或直径
                - 对于正方形：整数（边长）或元组(width, height)
                - 对于圆形：整数（直径）
                如果提供此参数，将优先使用此参数；如果为None，则使用size参数
        
        返回:
            roi_regions: ROI区域列表
            masks: 掩码列表
        """
        if len(points) != 4:
            raise ValueError("points must contain exactly 4 points")
        
        # 优先使用 side_length_or_diameter 参数
        if side_length_or_diameter is not None:
            if self.shape == 'square':
                # 正方形：直接使用边长
                actual_size = side_length_or_diameter
            else:  # circle
                # 圆形：将直径转换为半径
                if isinstance(side_length_or_diameter, tuple):
                    raise ValueError("圆形ROI的直径必须是整数，不能是元组")
                actual_size = side_length_or_diameter // 2
        elif size is not None:
            actual_size = size
        else:
            actual_size = self.size
        
        roi_regions = []
        masks = []
        h, w = image.shape[:2]
        
        for point in points:
            x, y = point
            
            if self.shape == 'square':
                roi, mask = self._extract_square_with_mask(image, x, y, actual_size, h, w)
            else:  # circle
                roi, mask = self._extract_circle_with_mask(image, x, y, actual_size, h, w)
            
            roi_regions.append(roi)
            masks.append(mask)
        
        return roi_regions, masks
    
    def _extract_square_with_mask(self, image: np.ndarray, center_x: int, center_y: int, 
                                 size: Union[int, Tuple[int, int]], h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        """提取正方形ROI区域和掩码"""
        if isinstance(size, tuple):
            half_width = size[0] // 2
            half_height = size[1] // 2
        else:
            half_width = size // 2
            half_height = size // 2
        
        x1 = max(0, center_x - half_width)
        y1 = max(0, center_y - half_height)
        x2 = min(w, center_x + half_width)
        y2 = min(h, center_y + half_height)
        
        roi = image[y1:y2, x1:x2].copy()
        mask = np.ones((roi.shape[0], roi.shape[1]), dtype=np.uint8) * 255
        
        return roi, mask
    
    def _extract_circle_with_mask(self, image: np.ndarray, center_x: int, center_y: int, 
                                 radius: int, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        """提取圆形ROI区域和掩码"""
        x1 = max(0, center_x - radius)
        y1 = max(0, center_y - radius)
        x2 = min(w, center_x + radius)
        y2 = min(h, center_y + radius)
        
        roi_rect = image[y1:y2, x1:x2].copy()
        roi_h, roi_w = roi_rect.shape[:2]
        
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        local_center_x = center_x - x1
        local_center_y = center_y - y1
        cv2.circle(mask, (local_center_x, local_center_y), radius, 255, -1)
        
        if len(roi_rect.shape) == 3:
            mask_3d = mask[:, :, np.newaxis]
            roi = roi_rect * (mask_3d / 255.0)
            roi = roi.astype(roi_rect.dtype)
        else:
            roi = roi_rect * (mask / 255.0)
            roi = roi.astype(roi_rect.dtype)
        
        return roi, mask
    
    def combine_rois(self, points: List[Tuple[int, int]], roi_regions: List[np.ndarray], 
                     image_shape: Tuple[int, int], 
                     side_length_or_diameter: Union[int, Tuple[int, int]] = None) -> np.ndarray:
        """
        将四个ROI组合回原图，ROI外的部分填充为黑色
        
        参数:
            points: 四个点的像素坐标列表，格式为 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            roi_regions: 四个ROI区域的列表
            image_shape: 原图片的尺寸，格式为 (height, width) 或 (height, width, channels)
            side_length_or_diameter: 分割的边长或直径（用于确定ROI的放置位置）
                - 对于正方形：整数（边长）或元组(width, height)
                - 对于圆形：整数（直径）
                如果为None，将根据ROI的实际大小自动推断
        
        返回:
            combined_image: 组合后的图像，ROI外的部分为黑色
        """
        if len(points) != 4:
            raise ValueError("points must contain exactly 4 points")
        
        if len(roi_regions) != 4:
            raise ValueError("roi_regions must contain exactly 4 ROI regions")
        
        # 确定图像通道数
        if len(image_shape) == 3:
            h, w, channels = image_shape
        else:
            h, w = image_shape
            # 根据第一个ROI确定通道数（优先使用ROI的通道数）
            if len(roi_regions[0].shape) == 3:
                channels = roi_regions[0].shape[2]
            else:
                channels = 1  # ROI是灰度图
        
        # 创建全黑图像（根据ROI的通道数创建，如果ROI是灰度图，组合图像也是灰度图）
        if channels == 1:
            combined_image = np.zeros((h, w), dtype=np.uint8)
        else:
            combined_image = np.zeros((h, w, channels), dtype=np.uint8)
        
        # 确定ROI尺寸
        if side_length_or_diameter is not None:
            if self.shape == 'square':
                if isinstance(side_length_or_diameter, tuple):
                    roi_width = side_length_or_diameter[0]
                    roi_height = side_length_or_diameter[1]
                else:
                    roi_width = side_length_or_diameter
                    roi_height = side_length_or_diameter
            else:  # circle
                if isinstance(side_length_or_diameter, tuple):
                    raise ValueError("圆形ROI的直径必须是整数，不能是元组")
                radius = side_length_or_diameter // 2
                roi_width = radius * 2
                roi_height = radius * 2
        else:
            # 根据ROI实际大小推断
            if self.shape == 'square':
                roi_height, roi_width = roi_regions[0].shape[:2]
            else:  # circle
                roi_height, roi_width = roi_regions[0].shape[:2]
                # 对于圆形，ROI是矩形框，需要找到实际半径
                # 可以通过查找非零像素来确定
                radius = min(roi_width, roi_height) // 2
        
        # 将每个ROI放回原位置
        for i, (point, roi) in enumerate(zip(points, roi_regions)):
            center_x, center_y = point
            
            if self.shape == 'square':
                self._place_square_roi(combined_image, roi, center_x, center_y, roi_width, roi_height, h, w)
            else:  # circle
                if side_length_or_diameter is not None:
                    radius = side_length_or_diameter // 2
                else:
                    # 从ROI推断半径（查找非零区域）
                    if len(roi.shape) == 3:
                        non_zero_mask = np.any(roi > 0, axis=2)
                    else:
                        non_zero_mask = roi > 0
                    # 找到非零区域的中心
                    y_coords, x_coords = np.where(non_zero_mask)
                    if len(y_coords) > 0:
                        roi_center_y = int(np.mean(y_coords))
                        roi_center_x = int(np.mean(x_coords))
                        # 计算最大半径
                        distances = np.sqrt((x_coords - roi_center_x)**2 + (y_coords - roi_center_y)**2)
                        radius = int(np.max(distances)) if len(distances) > 0 else min(roi_width, roi_height) // 2
                    else:
                        radius = min(roi_width, roi_height) // 2
                
                self._place_circle_roi(combined_image, roi, center_x, center_y, radius, h, w)
        
        return combined_image
    
    def _place_square_roi(self, combined_image: np.ndarray, roi: np.ndarray, 
                          center_x: int, center_y: int, roi_width: int, roi_height: int,
                          img_h: int, img_w: int):
        """将正方形ROI放置到组合图像中"""
        half_width = roi_width // 2
        half_height = roi_height // 2
        
        # 计算在图像中的位置
        x1 = max(0, center_x - half_width)
        y1 = max(0, center_y - half_height)
        x2 = min(img_w, center_x + half_width)
        y2 = min(img_h, center_y + half_height)
        
        # 计算ROI中需要放置的部分
        roi_h, roi_w = roi.shape[:2]
        
        # 计算ROI中的起始位置（如果ROI被裁剪了）
        roi_start_x = max(0, half_width - center_x)
        roi_start_y = max(0, half_height - center_y)
        roi_end_x = roi_start_x + (x2 - x1)
        roi_end_y = roi_start_y + (y2 - y1)
        
        # 确保不超出ROI边界
        roi_end_x = min(roi_end_x, roi_w)
        roi_end_y = min(roi_end_y, roi_h)
        
        # 提取ROI片段
        roi_patch = roi[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
        
        # 处理灰度ROI：如果ROI是灰度图但combined_image是彩色图，需要转换
        if len(roi_patch.shape) == 2 and len(combined_image.shape) == 3:
            # 将灰度ROI转换为3通道
            roi_patch = cv2.cvtColor(roi_patch, cv2.COLOR_GRAY2BGR)
        elif len(roi_patch.shape) == 3 and len(combined_image.shape) == 2:
            # 如果ROI是彩色但combined_image是灰度，转换为灰度
            roi_patch = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2GRAY)
        
        # 放置ROI
        combined_image[y1:y2, x1:x2] = roi_patch
    
    def _place_circle_roi(self, combined_image: np.ndarray, roi: np.ndarray,
                          center_x: int, center_y: int, radius: int,
                          img_h: int, img_w: int):
        """将圆形ROI放置到组合图像中"""
        # 计算边界框
        x1 = max(0, center_x - radius)
        y1 = max(0, center_y - radius)
        x2 = min(img_w, center_x + radius)
        y2 = min(img_h, center_y + radius)
        
        # 计算ROI中的位置
        roi_h, roi_w = roi.shape[:2]
        
        # 计算ROI中的起始位置
        roi_start_x = max(0, radius - center_x)
        roi_start_y = max(0, radius - center_y)
        roi_end_x = roi_start_x + (x2 - x1)
        roi_end_y = roi_start_y + (y2 - y1)
        
        # 确保不超出ROI边界
        roi_end_x = min(roi_end_x, roi_w)
        roi_end_y = min(roi_end_y, roi_h)
        
        # 提取要放置的ROI部分
        roi_patch = roi[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
        
        # 处理灰度ROI：如果ROI是灰度图但combined_image是彩色图，需要转换
        if len(roi_patch.shape) == 2 and len(combined_image.shape) == 3:
            # 将灰度ROI转换为3通道
            roi_patch = cv2.cvtColor(roi_patch, cv2.COLOR_GRAY2BGR)
        elif len(roi_patch.shape) == 3 and len(combined_image.shape) == 2:
            # 如果ROI是彩色但combined_image是灰度，转换为灰度
            roi_patch = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2GRAY)
        
        # 创建圆形掩码
        patch_h, patch_w = roi_patch.shape[:2]
        mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
        
        # 计算在patch中的圆心坐标
        local_center_x = center_x - x1
        local_center_y = center_y - y1
        
        # 绘制圆形掩码
        cv2.circle(mask, (local_center_x, local_center_y), radius, 255, -1)
        
        # 应用掩码并放置
        if len(combined_image.shape) == 3:
            mask_3d = mask[:, :, np.newaxis]
            mask_normalized = mask_3d / 255.0
            # 只更新掩码内的像素
            combined_image[y1:y2, x1:x2] = (
                combined_image[y1:y2, x1:x2] * (1 - mask_normalized) + 
                roi_patch * mask_normalized
            ).astype(combined_image.dtype)
        else:
            mask_normalized = mask / 255.0
            combined_image[y1:y2, x1:x2] = (
                combined_image[y1:y2, x1:x2] * (1 - mask_normalized) + 
                roi_patch * mask_normalized
            ).astype(combined_image.dtype)


# 示例使用代码
if __name__ == "__main__":
    # 创建ROI分割器实例
    # 正方形ROI，边长为100
    square_segmenter = ROISegmenter(shape='square', size=100)
    
    # 圆形ROI，半径为50
    circle_segmenter = ROISegmenter(shape='circle', size=50)
    
    # 读取测试图像（请替换为实际图像路径）
    image = cv2.imread('/home/yjc/Project/rov_ws/output_images/1.jpg')

    # 定义四个点的坐标
    points = [(234, 34), (235, 161), (362, 161), (362, 34)]
    
    # 提取正方形ROI - 使用边长参数（例如：边长为80）
    square_rois = square_segmenter.extract_roi(image, points, side_length_or_diameter=30)
    
    # 提取圆形ROI - 使用直径参数（例如：直径为100）
    circle_rois = circle_segmenter.extract_roi(image, points, side_length_or_diameter=100)
    
    # 保存结果
    for i, roi in enumerate(square_rois):
        cv2.imwrite(f'/home/yjc/Project/rov_ws/src/vision4location/scripts/image/square_roi_{i}.jpg', roi)
    
    for i, roi in enumerate(circle_rois):
        cv2.imwrite(f'/home/yjc/Project/rov_ws/src/vision4location/scripts/image/circle_roi_{i}.jpg', roi)
    
    # 组合ROI回原图 - 正方形ROI
    image_shape = image.shape  # (height, width, channels)
    combined_square = square_segmenter.combine_rois(
        points, square_rois, image_shape, side_length_or_diameter=30
    )
    cv2.imwrite('/home/yjc/Project/rov_ws/src/vision4location/scripts/image/combined_square.jpg', combined_square)
    
    # 组合ROI回原图 - 圆形ROI
    combined_circle = circle_segmenter.combine_rois(
        points, circle_rois, image_shape, side_length_or_diameter=100
    )
    cv2.imwrite('/home/yjc/Project/rov_ws/src/vision4location/scripts/image/combined_circle.jpg', combined_circle)

