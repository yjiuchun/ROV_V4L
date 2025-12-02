#!/bin/bash
# OpenCV Qt插件修复脚本
# 在conda环境中运行Python脚本前，先source此脚本

# 移除OpenCV的Qt插件路径
if [ -n "$QT_QPA_PLATFORM_PLUGIN_PATH" ]; then
    # 移除包含cv2/qt/plugins的路径
    export QT_QPA_PLATFORM_PLUGIN_PATH=$(echo "$QT_QPA_PLATFORM_PLUGIN_PATH" | tr ':' '\n' | grep -v 'cv2/qt/plugins' | tr '\n' ':' | sed 's/:$//')
    
    # 如果路径为空，尝试使用系统Qt插件路径
    if [ -z "$QT_QPA_PLATFORM_PLUGIN_PATH" ]; then
        if [ -d "/usr/lib/x86_64-linux-gnu/qt5/plugins" ]; then
            export QT_QPA_PLATFORM_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/qt5/plugins"
        else
            unset QT_QPA_PLATFORM_PLUGIN_PATH
        fi
    fi
fi

# 设置Qt平台为xcb
export QT_QPA_PLATFORM=xcb

echo "OpenCV Qt环境变量已设置"

