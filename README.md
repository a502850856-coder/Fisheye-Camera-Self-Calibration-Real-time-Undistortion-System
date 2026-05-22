

# Fisheye Camera Self-Calibration & Real-time Undistortion System

###  鱼眼相机自然特征自标定与实时高性能矫正系统
<img width="1117" height="904" alt="image" src="https://github.com/user-attachments/assets/dbd21012-4563-41dd-81d4-db3e5004bd8d" />


本系统是一个基于 **PyQt6** 和 **OpenCV** 开发的跨平台鱼眼相机视觉校准工具。系统通过捕获自然环境中的特征点运动轨迹（光流追踪），利用**空间逆向自标定算法（基于极线约束优化）**，在无需依赖传统黑白棋盘格的情况下，实现鱼眼相机内参及畸变参数的自动解算，并提供超流畅的实时重映射（Remap）矫正画面输出。

---

## ✨ 核心特性 (Key Features)

* **🛸 自然特征自标定 (Grid-free Self-Calibration)**：摆脱传统张正友标定法对物理棋盘格的依赖。手持相机在包含丰富线条的自然场景（如办公室、走廊）中缓慢晃动，系统即可通过算法逆向求解畸变参数。
* **🔌 多 USB 摄像头热切换 (Multi-Camera Hot-Swapping)**：
* 自动轮询底层硬件接口，智能检测并列表化所有挂载的 USB 视频输入设备。
* 支持在视频播放期间无缝热切换设备。
* 具备完善的底层硬件安全锁，防止并发刷新或热拔插导致的系统通信死锁。


* **⚡ 实时高性能矫正渲染 (High-Performance Real-time Undistortion)**：
* **优化前**：逐帧进行三角函数投影计算，导致高分辨率下界面严重卡顿、丢帧。
* **优化后（本项目）**：采用 **Map预烘焙机制**。仅在参数或 Balance（平衡系数）改变时计算一次高密度的坐标映射矩阵（`map1`, `map2`），日常渲染阶段仅执行底层 C++ 级优化的像素搬运（`cv2.remap`），稳稳跑满 30FPS+。


* **🎛 交互式拟合控制 (Interactive Fitting Control)**：提供可滑动调节的平衡系数（Balance Slider），实时裁剪或保留矫正后的边缘有效视场（FOV）。

---

## 🛠 技术栈 (Architecture & Tech Stack)

* **GUI 框架**：PyQt6 (提供信号槽事件驱动、多线程流控制、动态UI刷新)
* **计算机视觉**：OpenCV (ORB 特征检测、Lucas-Kanade 稀疏光流追踪、`cv2.fisheye` 畸变矩阵模型)
* **数值优化器**：SciPy (`scipy.optimize.minimize` - Nelder-Mead 单纯形下山法)
* **序列化**：Pickle (用于标定参数文件 `K` 和 `D` 的自动导出与覆写)

---

## 📐 算法原理拆解 (Algorithm Overview)

1. **特征追踪**：利用 ORB 算子初始化环境特征，随后通过 **LK 光流法 (Lucas-Kanade Optical Flow)** 持续追踪相邻帧间的运动轨迹，筛选出符合物理位移区间的优质特征序列。
2. **损失函数 (Loss Function)**：
将当前估计的鱼眼畸变系数 $\mathbf{D} [k_1, k_2, k_3, k_4]$ 对特征点进行逆向去畸变，利用去畸变后的归一化坐标计算 **基本矩阵 (Fundamental Matrix)**。基于对极几何中的**极线约束 (Epipolar Constraint)** 评估残差：

$$\text{Loss} = \sum |x_2^T F x_1|$$


3. **参数迭代**：优化器在多维空间中逆向迭代，使弯曲的线条在投影空间中回归至直线的几何物理大约束，收敛后即获得真实的相机畸变参数。

---

## 🚀 快速开始 (Quick Start)

### 1. 克隆仓库与依赖安装

```bash
git clone https://github.com/a502850856-coder/Fisheye Camera Self-Calibration & Real-time Undistortion System.git
cd Fisheye Camera Self-Calibration & Real-time Undistortion System
pip install PyQt6 opencv-python numpy scipy

```
PyQt6>=6.4.0
opencv-python>=4.7.0
numpy>=1.22.0
scipy>=1.10.0


### 2. 运行应用

```bash
python main.py

```

### 3. 操作向导

1. 在顶部的下拉菜单中选择你的 USB 鱼眼摄像头，点击 **【启动摄像头】**。
2. 点击 **【开始采集轨迹】**，手持摄像头面对富有纹理的环境（如门窗、地砖线条），**缓慢、平稳地做平移与旋转运动**。
3. 当特征运动样本集进度条满（达到 300 组）后，点击 **【💡 计算自标定参数】**。
4. 解算成功后，系统会弹出成功提示，**右侧画面将瞬间由弯曲拉直**，并在根目录下自动生成 `fisheye_calib.pkl` 校准文件。

---

## 📁 项目结构 (Project Structure)

```text
├── main.py                # 完整的 PyQt6 应用程序主入口
├── README.md              # 项目说明文档
└── fisheye_calib.pkl      # (运行后自动生成) 导出的相机内参和畸变参数文件

```

---

## 📝 许可证 (License)

本项目基于 [MIT License](https://www.google.com/search?q=LICENSE) 开源。欢迎提交 Issue 或 Pull Request 共同完善这个视觉项目！

---
