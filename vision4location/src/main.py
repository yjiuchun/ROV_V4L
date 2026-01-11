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
import time
import numpy as np
if __name__ == "__main__":

    img_process = ImgProcess(yolo="/home/yjc/Project/rov_ws/src/vision4location/tracking/YOLOv11/firsttrain_withledring/firsttrainledring/weights/best.pt",yoloseg="/home/yjc/Project/rov_ws/dataset/清水led双目/temp4train_simple/run/ledseg/weights/best.pt")
    left_video_path = "/home/yjc/Project/rov_ws/dataset/清水led双目/videos/left/stereo_capture_left_20251225_101902.mp4"
    right_video_path = "/home/yjc/Project/rov_ws/dataset/清水led双目/videos/right/stereo_capture_right_20251225_101902.mp4"
    left_cap = cv2.VideoCapture(left_video_path)
    right_cap = cv2.VideoCapture(right_video_path)

    while left_cap.isOpened():
        left_ret, left_frame = left_cap.read()
        right_ret, right_frame = right_cap.read()
        start_time = time.time()
        # cv2.imshow("left_frame_original", left_frame)
        # cv2.waitKey(1)
        # continue
        if not left_ret or not right_ret:
            break
        # 分割获取四个点的坐标
        start_time_seg = time.time()
        points, vis_image_left = img_process.GetRoI_seg(left_frame)
        end_time_seg = time.time()
        duration_seg = end_time_seg - start_time_seg
        print(f"分割时间: {duration_seg} 秒")
        points_vaild = True
        # print(points)
        if points == []:
            # print("未检测到分割结果")
            continue
        for i in range(4):
            if points[i][0] < 0 or points[i][1] < 0:
                points_vaild = False
        if not points_vaild:
            continue

        # cv2.imwrite("/home/yjc/Project/rov_ws/src/vision4location/src/image_save/seg/test/vis_image.jpg", vis_image)
        gray_image = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_image_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        # 根据四个点获取四个roi图像
        roi_images = img_process.self_lightness.get_roi_image(gray_image, points)
        key_points_left = []
        key_points_right = []
        for i, roi_image in enumerate(roi_images): 
            # 照度自适应二值化
            binary_roi_image = img_process.self_lightness.binary_image(roi_image)
            # 极值点检测
            extrema = img_process.extrema_detector.detect(binary_roi_image)
            y, x = extrema["max_positions"]
            # cv2.circle(roi_image, (int(x), int(y)), 2, (0, 0, 255), 3)  # red for max
            global_x = x - 15 + points[i][0]
            global_y = y - 15 + points[i][1]
            key_points_left.append([global_x, global_y])
            cv2.circle(left_frame, (int(global_x), int(global_y)), 2, (0, 0, 255), 2)
        #单目PnP
        rvec, tvec = img_process.pnp_solver.solve_pnp(np.array(key_points_left))

        # 反解右图像关键点
        B = 0.12
        f = 492.3707
        z = int(tvec[2])
        du = int(B * f / (z+0.00001))
        # print(z)
        points_right = []
        for i in range(len(points)):
            points_right.append([points[i][0] - du, points[i][1]])
        roi_images_right = img_process.self_lightness.get_roi_image(gray_image_right, points_right)


        for i, roi_image in enumerate(roi_images_right):
            # 照度自适应二值化
            binary_roi_image = img_process.self_lightness.binary_image(roi_image)
            # 极值点检测
            extrema = img_process.extrema_detector.detect(binary_roi_image)
            y, x = extrema["max_positions"]
            # cv2.circle(roi_image, (int(x), int(y)), 2, (0, 0, 255), 3)  # red for max
            global_x = x - 15 + points[i][0]
            global_y = y - 15 + points[i][1]
            cv2.circle(right_frame, (int(global_x), int(global_y)), 2, (0, 0, 255), 2)
        cv2.imshow("left_frame", left_frame)
        cv2.imshow("right_frame", right_frame)
        end_time = time.time()
        duration = end_time - start_time
        print(f"处理时间: {duration} 秒")
        cv2.waitKey(1)
    left_cap.release()
    right_cap.release()
    cv2.destroyAllWindows()
    # 目标文件夹路径（替换为你的文件夹路径）
    # folder_path = "/home/yjc/Project/rov_ws/output_images"  # Linux/macOS
    # img_process = ImgProcess(yolo="/home/yjc/Project/rov_ws/src/vision4location/tracking/YOLOv11/firsttrain_withledring/firsttrainledring/weights/best.pt",yoloseg="/home/yjc/Project/rov_ws/src/vision4location/tracking/YOLOv11_seg/seg_second_train/led_sys_seg/weights/best.pt")
    # # 遍历文件夹内所有文件
    # for filename in os.listdir(folder_path):
    #     # 筛选 .jpg / .JPG（大小写兼容）
    #     if filename.lower().endswith(".jpg"):
    #         # 拼接完整路径
    #         full_path = os.path.join(folder_path, filename)
  


    #         img = cv2.imread(full_path)
    #         mode = "spilt"

    #         if mode == "spilt":
    #             box,x_offset,duration,crop_img,results  = img_process.GetRoI(img)
    #             top_left_binary_img , top_right_binary_img , bottom_left_binary_img , bottom_right_binary_img , composite_image = img_process.selfLightness_split(img_process.mediaFilter(crop_img))

    #             dirname = filename.rsplit(".", 1)[0]
    #             dirpath = f"/home/yjc/Project/rov_ws/src/vision4location/src/image_save/spilt_img/{dirname}"
    #             os.makedirs(dirpath, exist_ok=True)

    #             if dirname == '10':
    #                 tl_img,tr_img,bl_img,br_img = img_process.self_lightness.split_image(img_process.mediaFilter(crop_img))
    #                 cv2.imwrite(f"{dirpath}/tl_img.jpg", tl_img)
    #                 cv2.imwrite(f"{dirpath}/tr_img.jpg", tr_img)
    #                 cv2.imwrite(f"{dirpath}/bl_img.jpg", bl_img)
    #                 cv2.imwrite(f"{dirpath}/br_img.jpg", br_img)

    #             crop_height, crop_width = crop_img.shape[:2]
    #             # 使用crop尺寸作为文件名
    #             filename = f"crop_img_{crop_width}x{crop_height}.jpg"
    #             cv2.imwrite(f"{dirpath}/top_left_binary_img.jpg", top_left_binary_img)
    #             cv2.imwrite(f"{dirpath}/top_right_binary_img.jpg", top_right_binary_img)
    #             cv2.imwrite(f"{dirpath}/bottom_left_binary_img.jpg", bottom_left_binary_img)
    #             cv2.imwrite(f"{dirpath}/bottom_right_binary_img.jpg", bottom_right_binary_img)
    #             cv2.imwrite(f"{dirpath}/composite_image.jpg", composite_image)
    #             cv2.imwrite(f"{dirpath}/{filename}", crop_img)
    #         elif mode == "crop":
    #             dirname = filename.rsplit(".", 1)[0]
    #             dirpath = f"/home/yjc/Project/rov_ws/src/vision4location/src/image_save/crop_img/{dirname}"
    #             os.makedirs(dirpath, exist_ok=True)

    #             box,x_offset,duration,crop_img,results  = img_process.GetRoI(img)
    #             binary_img = img_process.selfLightness_notsplit(img_process.mediaFilter(crop_img))
    #             cv2.imwrite(f"{dirpath}/binary_img.jpg", binary_img)
    #             cv2.imwrite(f"{dirpath}/{filename}", crop_img)
    #         else:
    #             dirname = filename.rsplit(".", 1)[0]
    #             dirpath = f"/home/yjc/Project/rov_ws/src/vision4location/src/image_save/no_crop_img/{dirname}"
    #             os.makedirs(dirpath, exist_ok=True)

    #             binary_img = img_process.selfLightness_notsplit(img_process.mediaFilter(img))
    #             cv2.imwrite(f"{dirpath}/binary_img.jpg", binary_img)
    #         # cv2.imwrite(f"/home/yjc/Project/rov_ws/src/vision4location/src/image_save/crop_img/2/{filename}", crop_img)
    #         # self_lightness = img_process.selfLightness(img_process.mediaFilter(crop_img))
        



    # # vis_image = img_process.yolo_detector.visualize(img, detections=results)
    # # cv2.imshow('results', vis_image)
    # # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
    # # cv2.imshow("img", img)
    # # print(duration)
    # # cv2.imwrite("./crop_img.jpg", crop_img)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()