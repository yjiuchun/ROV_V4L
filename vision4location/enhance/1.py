import cv2
import numpy as np

# img = cv2.imread("/home/yjc/Project/rov_ws/src/vision4location/enhance/images/DCP/J.png")
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow("gray_image", gray_image)
# cv2.imwrite("/home/yjc/Project/rov_ws/src/vision4location/enhance/images/DCP/gray_image.png", gray_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from DCP import DarkChannelPrior
import cv2

# 创建去雾器
dcp = DarkChannelPrior(patch_size=5, omega=0.15, t0=0.41)

# 方法1: 直接处理图像文件
dehazed_image = dcp.dehaze("/home/yjc/Project/rov_ws/underwater_dataset/images/first_capture/right/22.jpg")
cv2.imwrite("/home/yjc/Project/rov_ws/src/vision4location/enhance/images/DCP/dehazed_image.png", dehazed_image)

# # 方法2: 处理numpy数组
# image = cv2.imread("hazy_image.jpg")
# dehazed_image = dcp.dehaze(image)

# # 获取透射率图和暗通道图（用于分析）
# transmission_map = dcp.get_transmission_map(image)
# dark_channel = dcp.get_dark_channel(image)