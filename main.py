import sys
import cv2
import numpy as np
import pickle
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QSlider, QHBoxLayout, QVBoxLayout, 
                             QProgressBar, QStatusBar, QComboBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from scipy.optimize import minimize

class SelfCalibFisheyeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("160° 鱼眼相机自然特征自标定系统 (多USB摄像头支持)")
        self.setGeometry(100, 100, 1100, 720)
        
        # 初始化相机状态与参数
        self.cap = None
        self.current_camera_index = -1  # 默认无设备
        self.balance = 0.2
        self.is_calibrating = False
        
        # 初始猜想值（在标定成功前，画面会基于这组基础值进行默认矫正）
        self.width, self.height = 1280, 720
        self.cx, self.cy = self.width / 2, self.height / 2
        self.fx = self.fy = 400.0 
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.D = np.array([[-0.02], [0.005], [-0.001], [0.0001]]) # 鱼眼畸变初值 [k1, k2, k3, k4]
        
        # 预烘焙的畸变映射表（用于实现高性能实时重映射矫正，避免每帧重复计算）
        self.map1 = None
        self.map2 = None
        
        # 特征追踪历史数据缓冲区
        self.tracked_sequences = [] 
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.prev_gray = None
        self.prev_pts = None
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 1. 顶部向导说明
        guide_lbl = QLabel(
            "<b>自标定操作指南：</b><br>"
            "1. 选择插在 USB 上的鱼眼摄像头，并点击【启动摄像头】。<br>"
            "2. 点击【开始采集轨迹】，手持摄像头对准含有丰富线条的环境（如走廊、办公室），<b>缓慢平稳地左右平移和旋转</b>。<br>"
            "3. 进度条满（采集到300组特征运动）后，点击【计算自标定参数】。<br>"
            "4. <b>解算成功后，右侧矫正画面将自动应用自标定算法生成的最新畸变参数。</b>"
        )
        guide_lbl.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px; color: #333;")
        main_layout.addWidget(guide_lbl)
        
        # 2. 设备选择控制栏
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("<b>选择 USB 摄像头:</b>"))
        
        self.camera_selector = QComboBox()
        self.camera_selector.currentIndexChanged.connect(self.on_camera_selection_changed)
        device_layout.addWidget(self.camera_selector, stretch=2)
        
        self.btn_refresh = QPushButton("🔄 刷新设备列表")
        self.btn_refresh.clicked.connect(self.refresh_camera_list)
        device_layout.addWidget(self.btn_refresh, stretch=1)
        
        main_layout.addLayout(device_layout)
        
        # 3. 图像显示区
        display_layout = QHBoxLayout()
        self.lbl_raw = QLabel("原始视频 (绿色光流轨迹)")
        self.lbl_raw.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_raw.setStyleSheet("border: 2px solid #555; background-color: #111; color: #fff;")
        self.lbl_raw.setFixedSize(512, 320)
        
        self.lbl_corrected = QLabel("实时自解算矫正画面")
        self.lbl_corrected.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_corrected.setStyleSheet("border: 2px solid #555; background-color: #111; color: #fff;")
        self.lbl_corrected.setFixedSize(512, 320)
        
        display_layout.addWidget(self.lbl_raw)
        display_layout.addWidget(self.lbl_corrected)
        main_layout.addLayout(display_layout)
        
        # 4. 样本采集进度条
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("特征运动样本集进度:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(300) 
        progress_layout.addWidget(self.progress_bar)
        main_layout.addLayout(progress_layout)

        # 5. 实时控制滑块
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("平衡系数 (Balance):"))
        self.balance_slider = QSlider(Qt.Orientation.Horizontal)
        self.balance_slider.setMinimum(0)
        self.balance_slider.setMaximum(100)
        self.balance_slider.setValue(20)
        self.slider_val_lbl = QLabel("0.20")
        self.balance_slider.valueChanged.connect(self.on_slider_changed)
        slider_layout.addWidget(self.balance_slider)
        slider_layout.addWidget(self.slider_val_lbl)
        main_layout.addLayout(slider_layout)
        
        # 6. 交互按钮
        btn_layout = QHBoxLayout()
        self.btn_camera = QPushButton("启动摄像头")
        self.btn_camera.clicked.connect(self.toggle_camera)
        
        self.btn_record = QPushButton("开始采集轨迹")
        self.btn_record.setCheckable(True)
        self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self.toggle_record)
        
        self.btn_optimize = QPushButton("💡 计算自标定参数")
        self.btn_optimize.setEnabled(False)
        self.btn_optimize.clicked.connect(self.run_self_calibration)
        
        btn_layout.addWidget(self.btn_camera)
        btn_layout.addWidget(self.btn_record)
        btn_layout.addWidget(self.btn_optimize)
        main_layout.addLayout(btn_layout)
        
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        
        # 初始化自动刷新
        self.refresh_camera_list()

    def update_undistort_maps(self):
        """核心性能优化：预先计算好畸变重映射矩阵(Map)，不再在process_frame里每帧重复计算"""
        if self.width > 0 and self.height > 0:
            try:
                new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    self.K, self.D, (self.width, self.height), np.eye(3), balance=self.balance
                )
                self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                    self.K, self.D, np.eye(3), new_K, (self.width, self.height), cv2.CV_16SC2
                )
            except Exception as e:
                self.status.showMessage(f"重映射矩阵计算失败: {str(e)}")

    def refresh_camera_list(self):
        old_idx = self.camera_selector.currentIndex()
        self.camera_selector.clear()
        self.camera_selector.addItem("请选择一个视频输入设备...", -1)
        
        valid_camera_count = 0
        for index in range(6):
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    self.camera_selector.addItem(f"USB 摄像头接口 [Device ID: {index}]", index)
                    valid_camera_count += 1
        
        if valid_camera_count == 0:
            self.status.showMessage("⚠️ 警告：系统未检测到任何插入的 USB 摄像头设备！")
            self.btn_camera.setEnabled(False)
        else:
            self.btn_camera.setEnabled(True)
            if old_idx > 0 and old_idx < self.camera_selector.count():
                self.camera_selector.setCurrentIndex(old_idx)
            else:
                self.camera_selector.setCurrentIndex(1)
            self.status.showMessage("USB 设备列表刷新完毕。")

    def on_camera_selection_changed(self, idx):
        hardware_id = self.camera_selector.currentData()
        if hardware_id is not None:
            self.current_camera_index = hardware_id
            if self.timer.isActive() and hardware_id != -1:
                self.toggle_camera()
                self.toggle_camera()

    def on_slider_changed(self, val):
        self.balance = val / 100.0
        self.slider_val_lbl.setText(f"{self.balance:.2f}")
        # Balance滑块改变时，需要实时更新重映射表
        self.update_undistort_maps()

    def toggle_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            if self.cap: 
                self.cap.release()
                self.cap = None
            self.btn_camera.setText("启动摄像头")
            self.btn_record.setEnabled(False)
            self.camera_selector.setEnabled(True)
            self.btn_refresh.setEnabled(True)
        else:
            if self.current_camera_index == -1:
                self.status.showMessage("❌ 请选择一个有效的 USB 设备后再启动！")
                return
                
            self.cap = cv2.VideoCapture(self.current_camera_index)
            if not self.cap.isOpened():
                self.status.showMessage(f"错误：无法打开 USB 摄像头 (ID: {self.current_camera_index})")
                return
            
            ret, frame = self.cap.read()
            if ret:
                self.height, self.width = frame.shape[:2]
                self.cx, self.cy = self.width / 2, self.height / 2
                # 根据真实分辨率动态重置基础内参阵
                self.K = np.array([[self.width*0.35, 0, self.cx], [0, self.width*0.35, self.cy], [0, 0, 1]])
                # 初始化畸变重映射图表
                self.update_undistort_maps()
            
            self.timer.start(33)
            self.btn_camera.setText("关闭摄像头")
            self.btn_record.setEnabled(True)
            self.camera_selector.setEnabled(False)
            self.btn_refresh.setEnabled(False)

    def toggle_record(self, checked):
        if checked:
            self.btn_record.setText("正在录制样本... 再次点击暂停")
            self.status.showMessage("请缓慢晃动镜头，捕获环境边缘线条的弯曲弧度...")
        else:
            self.btn_record.setText("继续采集轨迹")

    def process_frame(self):
        if self.cap is None: return
        ret, frame = self.cap.read()
        if not ret: return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_raw = frame.copy()
        
        # 光流轨迹采集部分
        if self.btn_record.isChecked() and len(self.tracked_sequences) < 300:
            if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) > 0:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)
                if next_pts is not None:
                    good_prev = self.prev_pts[status == 1]
                    good_next = next_pts[status == 1]
                    
                    for pt_p, pt_n in zip(good_prev, good_next):
                        dist = np.linalg.norm(pt_p - pt_n)
                        if 3.0 < dist < 50.0:
                            self.tracked_sequences.append((pt_p, pt_n))
                            cv2.line(display_raw, (int(pt_p[0]), int(pt_p[1])), (int(pt_n[0]), int(pt_n[1])), (0, 255, 0), 2)
                    
                    self.progress_bar.setValue(min(len(self.tracked_sequences), 300))
                    if len(self.tracked_sequences) >= 300:
                        self.btn_optimize.setEnabled(True)
                        self.btn_record.setChecked(False)
                        self.btn_record.setText("样本采集完毕")
                        self.status.showMessage("样本采集完毕！可以开始计算自标定。")
            
            if self.prev_pts is None or len(self.prev_pts) < 150:
                kp = self.orb.detect(gray, None)
                if kp:
                    self.prev_pts = np.array([k.pt for k in kp], dtype=np.float32).reshape(-1, 1, 2)
            else:
                self.prev_pts = good_next.reshape(-1, 1, 2) if 'good_next' in locals() else None
        else:
            kp = self.orb.detect(gray, None)
            display_raw = cv2.drawKeypoints(display_raw, kp, None, color=(0, 255, 255))

        self.prev_gray = gray.copy()
        
        # 高性能实时畸变矫正渲染部分
        if self.map1 is not None and self.map2 is not None:
            corrected_frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            corrected_frame = frame.copy() # 如果重映射表尚未初始化，展示原图
        
        self.render_to_label(self.lbl_raw, display_raw)
        self.render_to_label(self.lbl_corrected, corrected_frame)

    # --- 自标定核心优化求解器 ---
    def run_self_calibration(self):
        if len(self.tracked_sequences) < 100:
            self.status.showMessage("❌ 样本数据过少，无法进行稳定解算。")
            return

        self.status.showMessage("🚀 正在计算空间逆向自标定解算，请稍候...")
        self.btn_optimize.setEnabled(False)
        QApplication.processEvents() 
        
        pts_p = np.array([item[0] for item in self.tracked_sequences])
        pts_n = np.array([item[1] for item in self.tracked_sequences])
        
        # 损失函数：通过极线约束(Epipolar Constraint)评估当前畸变参数下的直线弯曲残差
        def fish_eye_self_calib_loss(d_params):
            test_D = np.array([[d_params[0]], [d_params[1]], [d_params[2]], [d_params[3]]])
            try:
                undist_p = cv2.fisheye.undistortPoints(pts_p.reshape(-1, 1, 2), self.K, test_D)
                undist_n = cv2.fisheye.undistortPoints(pts_n.reshape(-1, 1, 2), self.K, test_D)
                
                F, mask = cv2.findFundamentalMat(undist_p, undist_n, cv2.FM_RANSAC, 0.005, 0.99)
                if F is None or F.shape != (3, 3): return 99999.0
                
                pts1_h = np.hstack((undist_p.reshape(-1, 2), np.ones((len(undist_p), 1))))
                pts2_h = np.hstack((undist_n.reshape(-1, 2), np.ones((len(undist_n), 1))))
                lines2 = (F @ pts1_h.T).T
                lines2 /= np.linalg.norm(lines2[:, :2], axis=1, keepdims=True)
                err = np.abs(np.sum(pts2_h * lines2, axis=1))
                return np.mean(err[mask.ravel() == 1])
            except:
                return 99999.0

        # 以当前畸变参数作为优化初值
        initial_guess = [self.D[0][0], self.D[1][0], self.D[2][0], self.D[3][0]]
        res = minimize(fish_eye_self_calib_loss, initial_guess, method='Nelder-Mead', options={'maxiter': 150})
        
        if res.success:
            # 1. 将自标定解算出来的最新畸变参数写入类成员变量
            self.D = np.array([[res.x[0]], [res.x[1]], [res.x[2]], [res.x[3]]])
            
            # 2. 关键补全：自标定更新参数后，立即刷新畸变矫正映射表（画面立刻变直）
            self.update_undistort_maps()
            
            self.status.showMessage(f"✅ 自标定成功！新畸变参数已实时应用并导出。D: {res.x}")
            with open("fisheye_calib.pkl", "wb") as f:
                pickle.dump({"K": self.K, "D": self.D}, f)
        else:
            self.status.showMessage("❌ 解算收敛失败，请尝试在更平整、纹理更丰富的环境中重试。")
            
        self.btn_optimize.setEnabled(True)
        self.tracked_sequences = []
        self.progress_bar.setValue(0)

    def render_to_label(self, label, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        label.setPixmap(pix)

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened(): self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SelfCalibFisheyeApp()
    window.show()
    sys.exit(app.exec())
