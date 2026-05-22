# Fisheye-Camera-Self-Calibration-Real-time-Undistortion-System
本系统是一个基于 PyQt6 和 OpenCV 开发的跨平台鱼眼相机视觉校准工具。系统通过捕获自然环境中的特征点运动轨迹（光流追踪），利用空间逆向自标定算法（基于极线约束优化），在无需依赖传统黑白棋盘格的情况下，实现鱼眼相机内参及畸变参数的自动解算，并提供超流畅的实时重映射（Remap）矫正画面输出。
