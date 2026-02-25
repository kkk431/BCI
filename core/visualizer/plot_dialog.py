#!/usr/bin/env python3
"""
plot_dialog.py
独立绘图对话框 - 用于弹出式信号查看
复用signal_view的功能，但作为对话框
"""

import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
                             QGroupBox, QCheckBox, QMessageBox, QFileDialog,
                             QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import scipy.signal
from typing import Dict, List, Optional, Tuple, Any


class PlotDialog(QDialog):
    """
    独立绘图对话框
    用于快速查看信号片段
    """

    def __init__(self, data: np.ndarray, sampling_rate: float,
                 channel_names: List[str] = None,
                 title: str = "信号视图", parent=None):
        """
        初始化绘图对话框

        Args:
            data: 信号数据 (channels × samples) 或 (samples,)
            sampling_rate: 采样率 (Hz)
            channel_names: 通道名称列表
            title: 对话框标题
        """
        super().__init__(parent)

        # 处理数据
        if data.ndim == 1:
            self.data = data.reshape(1, -1)
        else:
            self.data = data

        self.sampling_rate = sampling_rate
        self.n_channels, self.n_samples = self.data.shape
        self.duration = self.n_samples / sampling_rate

        if channel_names is None:
            self.channel_names = [f"Ch{i + 1}" for i in range(self.n_channels)]
        else:
            self.channel_names = channel_names

        self.title = title

        # 当前显示状态
        self.current_page = 0
        self.page_duration = 5.0  # 默认每页5秒
        self.markers = []  # 事件标记

        # 设置对话框
        self.setWindowTitle(title)
        self.resize(1000, 700)
        self.setModal(False)  # 非模态，可以同时打开多个

        # 设置UI
        self.setup_ui()

        # 更新绘图
        self.update_plot()

    def setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)

        # ========== 顶部控制栏 ==========
        control_layout = QHBoxLayout()

        # 信息显示
        info_label = QLabel(f"{self.n_channels}通道 | {self.sampling_rate}Hz | {self.duration:.2f}秒")
        info_label.setFont(QFont("Microsoft YaHei", 9))
        control_layout.addWidget(info_label)

        control_layout.addStretch()

        # 翻页控制
        control_layout.addWidget(QLabel("页:"))
        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.setMaximum(max(1, int(np.ceil(self.duration / self.page_duration))))
        self.page_spin.setValue(1)
        self.page_spin.valueChanged.connect(self.on_page_changed)
        control_layout.addWidget(self.page_spin)

        control_layout.addWidget(QLabel("/"))
        self.total_pages_label = QLabel(str(self.page_spin.maximum()))
        control_layout.addWidget(self.total_pages_label)

        control_layout.addWidget(QLabel("  每页(秒):"))
        self.page_duration_spin = QDoubleSpinBox()
        self.page_duration_spin.setRange(1, 60)
        self.page_duration_spin.setValue(self.page_duration)
        self.page_duration_spin.setSingleStep(1)
        self.page_duration_spin.valueChanged.connect(self.on_page_duration_changed)
        control_layout.addWidget(self.page_duration_spin)

        # 翻页按钮
        prev_btn = QPushButton("◀")
        prev_btn.setMaximumWidth(30)
        prev_btn.clicked.connect(self.prev_page)
        control_layout.addWidget(prev_btn)

        next_btn = QPushButton("▶")
        next_btn.setMaximumWidth(30)
        next_btn.clicked.connect(self.next_page)
        control_layout.addWidget(next_btn)

        layout.addLayout(control_layout)

        # ========== 第二行：滤波控制 ==========
        filter_layout = QHBoxLayout()

        self.filter_check = QCheckBox("启用滤波")
        self.filter_check.toggled.connect(self.update_plot)
        filter_layout.addWidget(self.filter_check)

        filter_layout.addWidget(QLabel("低通(Hz):"))
        self.lowpass_spin = QDoubleSpinBox()
        self.lowpass_spin.setRange(0.1, 500)
        self.lowpass_spin.setValue(45)
        self.lowpass_spin.setEnabled(False)
        self.filter_check.toggled.connect(self.lowpass_spin.setEnabled)
        filter_layout.addWidget(self.lowpass_spin)

        filter_layout.addWidget(QLabel("高通(Hz):"))
        self.highpass_spin = QDoubleSpinBox()
        self.highpass_spin.setRange(0.01, 500)
        self.highpass_spin.setValue(0.5)
        self.highpass_spin.setEnabled(False)
        self.filter_check.toggled.connect(self.highpass_spin.setEnabled)
        filter_layout.addWidget(self.highpass_spin)

        filter_layout.addWidget(QLabel("陷波(Hz):"))
        self.notch_spin = QDoubleSpinBox()
        self.notch_spin.setRange(0, 100)
        self.notch_spin.setValue(50)
        self.notch_spin.setSpecialValueText("关闭")
        self.notch_spin.setEnabled(False)
        self.filter_check.toggled.connect(self.notch_spin.setEnabled)
        filter_layout.addWidget(self.notch_spin)

        filter_layout.addStretch()

        layout.addLayout(filter_layout)

        # ========== 绘图区域 ==========
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        # ========== 底部按钮 ==========
        btn_layout = QHBoxLayout()

        add_marker_btn = QPushButton("添加标记")
        add_marker_btn.clicked.connect(self.add_marker)
        btn_layout.addWidget(add_marker_btn)

        clear_marker_btn = QPushButton("清除标记")
        clear_marker_btn.clicked.connect(self.clear_markers)
        btn_layout.addWidget(clear_marker_btn)

        btn_layout.addStretch()

        save_btn = QPushButton("保存图像")
        save_btn.clicked.connect(self.save_plot)
        btn_layout.addWidget(save_btn)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def on_page_changed(self):
        """页码改变"""
        self.current_page = self.page_spin.value() - 1
        self.update_plot()

    def on_page_duration_changed(self):
        """每页时长改变"""
        self.page_duration = self.page_duration_spin.value()
        max_pages = max(1, int(np.ceil(self.duration / self.page_duration)))
        self.page_spin.setMaximum(max_pages)
        self.total_pages_label.setText(str(max_pages))
        self.update_plot()

    def prev_page(self):
        """上一页"""
        if self.current_page > 0:
            self.current_page -= 1
            self.page_spin.setValue(self.current_page + 1)

    def next_page(self):
        """下一页"""
        if self.current_page < self.page_spin.maximum() - 1:
            self.current_page += 1
            self.page_spin.setValue(self.current_page + 1)

    def apply_filter(self, data: np.ndarray) -> np.ndarray:
        """应用滤波"""
        if not self.filter_check.isChecked():
            return data

        filtered = data.copy()
        fs = self.sampling_rate

        # 低通滤波
        lowcut = self.lowpass_spin.value()
        if lowcut > 0 and lowcut < fs / 2:
            sos = scipy.signal.butter(4, lowcut, 'lowpass', fs=fs, output='sos')
            filtered = scipy.signal.sosfiltfilt(sos, filtered, axis=1)

        # 高通滤波
        highcut = self.highpass_spin.value()
        if highcut > 0:
            sos = scipy.signal.butter(4, highcut, 'highpass', fs=fs, output='sos')
            filtered = scipy.signal.sosfiltfilt(sos, filtered, axis=1)

        # 陷波滤波
        notch = self.notch_spin.value()
        if notch > 0 and notch < fs / 2:
            Q = 30
            b, a = scipy.signal.iirnotch(notch, Q, fs)
            filtered = scipy.signal.filtfilt(b, a, filtered, axis=1)

        return filtered

    def update_plot(self):
        """更新绘图"""
        self.figure.clear()

        # 计算当前页的时间范围
        t_start = self.current_page * self.page_duration
        t_end = min((self.current_page + 1) * self.page_duration, self.duration)

        start_idx = int(t_start * self.sampling_rate)
        end_idx = int(t_end * self.sampling_rate)

        # 提取数据
        data_segment = self.data[:, start_idx:end_idx]
        time = np.arange(start_idx, end_idx) / self.sampling_rate

        # 应用滤波
        data_filtered = self.apply_filter(data_segment)

        # 创建子图
        n_show = min(self.n_channels, 16)  # 最多显示16通道
        gs = self.figure.add_gridspec(n_show, 1, hspace=0.1)

        # 计算全局幅度范围
        y_min = np.min(data_filtered)
        y_max = np.max(data_filtered)
        margin = (y_max - y_min) * 0.1
        y_min -= margin
        y_max += margin

        # 绘制每个通道
        for i in range(n_show):
            ax = self.figure.add_subplot(gs[i, 0])

            # 绘制信号
            ax.plot(time, data_filtered[i], 'b-', linewidth=0.8)

            # 设置Y轴
            ax.set_ylabel(self.channel_names[i], fontsize=8)
            ax.set_ylim(y_min, y_max)

            # 隐藏X轴标签（除了最后一个）
            if i < n_show - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('时间 (秒)', fontsize=9)

            # 添加网格
            ax.grid(True, alpha=0.3)

            # 绘制标记
            for t, color, label in self.markers:
                if t_start <= t <= t_end:
                    ax.axvline(x=t, color=color, linestyle='-', linewidth=2)
                    ax.text(t, y_min + (y_max - y_min) * 0.1, label,
                            fontsize=8, color=color)

        self.figure.suptitle(f"{self.title} - 时间: {t_start:.2f} - {t_end:.2f} 秒", fontsize=12)

        self.canvas.draw()

    def add_marker(self):
        """添加标记（简化版）"""
        current_time = self.current_page * self.page_duration + self.page_duration / 2
        self.markers.append((current_time, 'red', f'M{len(self.markers) + 1}'))
        self.update_plot()

    def clear_markers(self):
        """清除标记"""
        self.markers.clear()
        self.update_plot()

    def save_plot(self):
        """保存图像"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "", "PNG图像 (*.png);;PDF文件 (*.pdf);;SVG图像 (*.svg)")

        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "保存成功", f"图像已保存到:\n{file_path}")
            except Exception as e:
                QMessageBox.warning(self, "保存失败", f"保存图像时出错:\n{str(e)}")


# 便捷函数：快速弹出绘图窗口
def quick_plot(data, sampling_rate=1000, channel_names=None, title="信号视图"):
    """
    快速弹出绘图窗口

    Args:
        data: 信号数据
        sampling_rate: 采样率
        channel_names: 通道名称
        title: 窗口标题
    """
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    dialog = PlotDialog(data, sampling_rate, channel_names, title)
    dialog.exec_()


# 测试代码
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    # 创建测试信号
    fs = 1000
    t = np.arange(0, 30, 1 / fs)
    data = np.array([np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t)) for _ in range(8)])

    app = QApplication(sys.argv)
    dialog = PlotDialog(data, fs, [f"Ch{i}" for i in range(8)], "测试信号")
    dialog.show()
    sys.exit(app.exec_())