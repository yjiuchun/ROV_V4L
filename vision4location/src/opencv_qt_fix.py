"""
OpenCV Qt插件修复模块

在conda环境中，OpenCV的Qt插件可能缺少依赖导致无法加载。
此模块在导入cv2之前修复Qt环境变量。

使用方法：
    import opencv_qt_fix  # 必须在导入cv2之前
    import cv2
"""
import os
import sys

def fix_opencv_qt():
    """修复OpenCV的Qt插件问题"""
    # 移除OpenCV的Qt插件路径
    if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
        plugin_path = os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
        if 'cv2/qt/plugins' in plugin_path:
            paths = plugin_path.split(os.pathsep)
            paths = [p for p in paths if 'cv2/qt/plugins' not in p]
            if paths:
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.pathsep.join(paths)
            else:
                # 尝试使用系统Qt插件路径
                system_qt_path = '/usr/lib/x86_64-linux-gnu/qt5/plugins'
                if os.path.exists(system_qt_path):
                    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = system_qt_path
                else:
                    del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
    
    # 设置Qt平台为xcb
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

# 自动执行修复
fix_opencv_qt()

