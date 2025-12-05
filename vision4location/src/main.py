# 必须在导入任何其他模块之前修复OpenCV的Qt插件问题
import os
import sys

# 导入修复模块（必须在导入cv2之前）
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
try:
    import opencv_qt_fix  # 自动修复Qt问题
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
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

from img_process import ImgProcess
import cv2
if __name__ == "__main__":

    # 目标文件夹路径（替换为你的文件夹路径）
    folder_path = "/home/yjc/Project/rov_ws/output_images"  # Linux/macOS
    img_process = ImgProcess(model_path="/home/yjc/Project/rov_ws/src/vision4location/tracking/YOLOv11/firsttrain_withledring/firsttrainledring/weights/best.pt")
    # 遍历文件夹内所有文件
    for filename in os.listdir(folder_path):
        # 筛选 .jpg / .JPG（大小写兼容）
        if filename.lower().endswith(".jpg"):
            # 拼接完整路径
            full_path = os.path.join(folder_path, filename)
            dirname = filename.rsplit(".", 1)[0]
            dirpath = f"/home/yjc/Project/rov_ws/src/vision4location/src/image_save/crop_img/{dirname}"
            os.makedirs(dirpath, exist_ok=True)


            img = cv2.imread(full_path)
            box,x_offset,duration,crop_img,results  = img_process.GetRoI(img)
            top_left_binary_img , top_right_binary_img , bottom_left_binary_img , bottom_right_binary_img , composite_image = img_process.selfLightness(img_process.mediaFilter(crop_img))

            
            # 获取crop的尺寸
            crop_height, crop_width = crop_img.shape[:2]
            # 使用crop尺寸作为文件名
            filename = f"crop_img_{crop_width}x{crop_height}.jpg"
            cv2.imwrite(f"{dirpath}/top_left_binary_img.jpg", top_left_binary_img)
            cv2.imwrite(f"{dirpath}/top_right_binary_img.jpg", top_right_binary_img)
            cv2.imwrite(f"{dirpath}/bottom_left_binary_img.jpg", bottom_left_binary_img)
            cv2.imwrite(f"{dirpath}/bottom_right_binary_img.jpg", bottom_right_binary_img)
            cv2.imwrite(f"{dirpath}/composite_image.jpg", composite_image)
            cv2.imwrite(f"{dirpath}/{filename}", crop_img)
            # cv2.imwrite(f"/home/yjc/Project/rov_ws/src/vision4location/src/image_save/crop_img/2/{filename}", crop_img)
            # self_lightness = img_process.selfLightness(img_process.mediaFilter(crop_img))



    # vis_image = img_process.yolo_detector.visualize(img, detections=results)
    # cv2.imshow('results', vis_image)
    # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
    # cv2.imshow("img", img)
    # print(duration)
    # cv2.imwrite("./crop_img.jpg", crop_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()