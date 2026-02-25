#!/usr/bin/env python3
"""
signal_view.py
通用信号视图 - 支持所有模态的多通道信号可视化
基于四层数据格式: meta/signal/event/processed
"""

import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
                             QComboBox, QGroupBox, QGridLayout, QListWidget,
                             QListWidgetItem, QMessageBox, QFileDialog,
                             QSplitter, QCheckBox, QLineEdit, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette

import scipy.signal
from typing import Dict, List, Optional, Tuple, Any
import os
from datetime import datetime


class SignalView(QMainWindow):
    """
    通用信号视图类
    支持EEG/EMG/ECG/GSR/fNIRS/ET/RESP等多种模态
    """

    def __init__(self, data_dict: Dict[str, Any], modality: str = None, parent=None):
        """
        初始化信号视图

        Args:
            data_dict: 标准四层数据字典
            modality: 要显示的模态（None表示自动选择第一个）
        """
        super().__init__(parent)
        self.data_dict = data_dict
        self.modality = modality

        # 解析数据
        self._parse_data()

        # 当前显示状态
        self.current_page = 0
        self.zoom_level = 1.0
        self.markers = []  # 事件标记 [(time, color, label)]

        # 设置窗口
        self.setWindowTitle(f"信号视图 - {self.modality} - {self.subject_id}")
        self.resize(1200, 800)

        # 设置UI
        self.setup_ui()

        # 初始化绘图
        self.update_plot()

    def _parse_data(self):
        """解析数据字典"""
        # 获取元数据
        self.meta = self.data_dict.get("meta", {})
        self.subject_id = self.meta.get("subject_id", "unknown")
        self.session_id = self.meta.get("session_id", "unknown")
        self.task = self.meta.get("task", "unknown")

        # 获取信号数据
        signal_dict = self.data_dict.get("signal", {})

        # 确定要显示的模态
        if self.modality is None:
            # 自动选择第一个有数据的模态
            for mod in ['EEG', 'EMG', 'ECG', 'GSR', 'FNIRS', 'ET', 'RESP']:
                if mod in signal_dict:
                    self.modality = mod
                    break
            if self.modality is None and signal_dict:
                self.modality = list(signal_dict.keys())[0]

        if self.modality not in signal_dict:
            raise ValueError(f"模态 {self.modality} 不在数据中")

        signal_info = signal_dict[self.modality]

        # 信号数据
        self.data = signal_info.get("data")
        if self.data is None:
            raise ValueError("信号数据不存在")

        # 确保数据是2D (channels × samples)
        if self.data.ndim == 1:
            self.data = self.data.reshape(1, -1)

        self.n_channels, self.n_samples = self.data.shape
        self.sampling_rate = signal_info.get("sampling_rate", 1000)
        self.channel_names = signal_info.get("channel_names",
                                             [f"Ch{i + 1}" for i in range(self.n_channels)])
        self.unit = signal_info.get("unit", "unknown")
        self.signal_type = signal_info.get("signal_type", self.modality.lower())

        # 持续时间
        self.duration = self.n_samples / self.sampling_rate

        # 获取事件数据
        self.events = self.data_dict.get("event", {})
        self.event_times = self.events.get("event_time", [])
        self.event_labels = self.events.get("event_label", [])
        self.event_ids = self.events.get("event_id", [])

        # 获取预处理信息
        self.processed = self.data_dict.get("processed", {})
        self.preprocess_info = self.processed.get("eeg_preprocessing", {})

    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # ========== 顶部控制栏 ==========
        control_layout = QHBoxLayout()

        # 模态信息显示
        info_label = QLabel(f"{self.modality} | {self.n_channels}通道 | "
                            f"{self.sampling_rate}Hz | {self.duration:.2f}秒")
        info_label.setFont(QFont("Microsoft YaHei", 10))
        control_layout.addWidget(info_label)

        control_layout.addStretch()

        # 翻页控制
        control_layout.addWidget(QLabel("页:"))
        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.setMaximum(max(1, int(np.ceil(self.duration / 5))))
        self.page_spin.setValue(1)
        self.page_spin.valueChanged.connect(self.on_page_changed)
        control_layout.addWidget(self.page_spin)

        control_layout.addWidget(QLabel("/"))
        self.total_pages_label = QLabel(str(self.page_spin.maximum()))
        control_layout.addWidget(self.total_pages_label)

        control_layout.addWidget(QLabel("  每页(秒):"))
        self.page_duration_spin = QDoubleSpinBox()
        self.page_duration_spin.setRange(1, 60)
        self.page_duration_spin.setValue(5)
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

        main_layout.addLayout(control_layout)

        # ========== 第二行：滤波控制 ==========
        filter_layout = QHBoxLayout()

        filter_group = QGroupBox("滤波")
        filter_group.setLayout(QHBoxLayout())

        self.filter_check = QCheckBox("启用滤波")
        filter_group.layout().addWidget(self.filter_check)

        filter_group.layout().addWidget(QLabel("低通(Hz):"))
        self.lowpass_spin = QDoubleSpinBox()
        self.lowpass_spin.setRange(0.1, 500)
        self.lowpass_spin.setValue(45)
        self.lowpass_spin.setEnabled(False)
        filter_group.layout().addWidget(self.lowpass_spin)

        filter_group.layout().addWidget(QLabel("高通(Hz):"))
        self.highpass_spin = QDoubleSpinBox()
        self.highpass_spin.setRange(0.01, 500)
        self.highpass_spin.setValue(0.5)
        self.highpass_spin.setEnabled(False)
        filter_group.layout().addWidget(self.highpass_spin)

        filter_group.layout().addWidget(QLabel("陷波(Hz):"))
        self.notch_spin = QDoubleSpinBox()
        self.notch_spin.setRange(0, 100)
        self.notch_spin.setValue(50)
        self.notch_spin.setSpecialValueText("关闭")
        self.notch_spin.setEnabled(False)
        filter_group.layout().addWidget(self.notch_spin)

        self.filter_check.toggled.connect(self.on_filter_toggled)

        filter_layout.addWidget(filter_group)

        filter_layout.addStretch()

        # 幅度控制
        amp_group = QGroupBox("幅度范围")
        amp_group.setLayout(QHBoxLayout())

        amp_group.layout().addWidget(QLabel("最小:"))
        self.amp_min_spin = QDoubleSpinBox()
        self.amp_min_spin.setRange(-10000, 0)
        self.amp_min_spin.setValue(-200)
        self.amp_min_spin.setSpecialValueText("自动")
        self.amp_min_spin.valueChanged.connect(self.update_plot)
        amp_group.layout().addWidget(self.amp_min_spin)

        amp_group.layout().addWidget(QLabel("最大:"))
        self.amp_max_spin = QDoubleSpinBox()
        self.amp_max_spin.setRange(0, 10000)
        self.amp_max_spin.setValue(200)
        self.amp_max_spin.setSpecialValueText("自动")
        self.amp_max_spin.valueChanged.connect(self.update_plot)
        amp_group.layout().addWidget(self.amp_max_spin)

        self.auto_amp_check = QCheckBox("自动")
        self.auto_amp_check.setChecked(True)
        self.auto_amp_check.toggled.connect(self.on_auto_amp_toggled)
        amp_group.layout().addWidget(self.auto_amp_check)

        filter_layout.addWidget(amp_group)

        main_layout.addLayout(filter_layout)

        # ========== 中间：绘图区域 ==========
        plot_splitter = QSplitter(Qt.Horizontal)

        # 左侧：通道列表
        channel_widget = QWidget()
        channel_layout = QVBoxLayout(channel_widget)
        channel_layout.setContentsMargins(0, 0, 0, 0)

        channel_layout.addWidget(QLabel("通道选择:"))

        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QListWidget.MultiSelection)
        for i, name in enumerate(self.channel_names):
            item = QListWidgetItem(f"{i + 1:02d}. {name}")
            item.setData(Qt.UserRole, i)
            item.setSelected(True)  # 默认全选
            self.channel_list.addItem(item)

        self.channel_list.itemSelectionChanged.connect(self.update_plot)
        channel_layout.addWidget(self.channel_list)

        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self.select_all_channels)
        channel_layout.addWidget(select_all_btn)

        plot_splitter.addWidget(channel_widget)

        # 右侧：绘图区域
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # Matplotlib Figure
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        plot_splitter.addWidget(plot_widget)
        plot_splitter.setSizes([200, 800])

        main_layout.addWidget(plot_splitter, 1)

        # ========== 底部：事件标记区域 ==========
        event_widget = QWidget()
        event_layout = QHBoxLayout(event_widget)

        event_layout.addWidget(QLabel("事件标记:"))

        self.event_list = QListWidget()
        self.event_list.setMaximumHeight(80)
        for t, label in zip(self.event_times, self.event_labels):
            self.event_list.addItem(f"{t:.3f}s: {label}")

        event_layout.addWidget(self.event_list)

        add_marker_btn = QPushButton("添加标记")
        add_marker_btn.clicked.connect(self.add_marker_dialog)
        event_layout.addWidget(add_marker_btn)

        clear_markers_btn = QPushButton("清除标记")
        clear_markers_btn.clicked.connect(self.clear_markers)
        event_layout.addWidget(clear_markers_btn)

        save_markers_btn = QPushButton("保存标记")
        save_markers_btn.clicked.connect(self.save_markers)
        event_layout.addWidget(save_markers_btn)

        main_layout.addWidget(event_widget)

    def on_page_changed(self):
        """页码改变"""
        self.current_page = self.page_spin.value() - 1
        self.update_plot()

    def on_page_duration_changed(self):
        """每页时长改变"""
        page_duration = self.page_duration_spin.value()
        max_pages = max(1, int(np.ceil(self.duration / page_duration)))
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

    def on_filter_toggled(self, enabled):
        """滤波开关"""
        self.lowpass_spin.setEnabled(enabled)
        self.highpass_spin.setEnabled(enabled)
        self.notch_spin.setEnabled(enabled)
        self.update_plot()

    def on_auto_amp_toggled(self, auto):
        """自动幅度开关"""
        self.amp_min_spin.setEnabled(not auto)
        self.amp_max_spin.setEnabled(not auto)
        self.update_plot()

    def select_all_channels(self):
        """全选通道"""
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            item.setSelected(True)

    def get_selected_channels(self) -> List[int]:
        """获取选中的通道索引"""
        indices = []
        for item in self.channel_list.selectedItems():
            idx = item.data(Qt.UserRole)
            if idx is not None:
                indices.append(idx)
        return indices if indices else list(range(self.n_channels))

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

        # 获取选中的通道
        selected_channels = self.get_selected_channels()
        n_show = len(selected_channels)

        if n_show == 0:
            return

        # 计算当前页的时间范围
        page_duration = self.page_duration_spin.value()
        t_start = self.current_page * page_duration
        t_end = min((self.current_page + 1) * page_duration, self.duration)

        start_idx = int(t_start * self.sampling_rate)
        end_idx = int(t_end * self.sampling_rate)

        # 提取数据
        data_segment = self.data[:, start_idx:end_idx]
        time = np.arange(start_idx, end_idx) / self.sampling_rate

        # 应用滤波
        data_filtered = self.apply_filter(data_segment)

        # 创建子图
        gs = self.figure.add_gridspec(n_show, 1, hspace=0.1)

        # 计算幅度范围
        if self.auto_amp_check.isChecked():
            # 自动计算
            y_min = np.min(data_filtered[selected_channels])
            y_max = np.max(data_filtered[selected_channels])
            margin = (y_max - y_min) * 0.1
            y_min -= margin
            y_max += margin
        else:
            y_min = self.amp_min_spin.value()
            y_max = self.amp_max_spin.value()

        # 绘制每个通道
        for i, ch_idx in enumerate(selected_channels):
            ax = self.figure.add_subplot(gs[i, 0])

            # 绘制信号
            ax.plot(time, data_filtered[ch_idx], 'b-', linewidth=0.8)

            # 设置Y轴
            ax.set_ylabel(f"{self.channel_names[ch_idx]}\n({self.unit})", fontsize=8)
            ax.set_ylim(y_min, y_max)

            # 隐藏X轴标签（除了最后一个）
            if i < n_show - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('时间 (秒)', fontsize=9)

            # 添加网格
            ax.grid(True, alpha=0.3)

            # 绘制事件标记
            for t in self.event_times:
                if t_start <= t <= t_end:
                    ax.axvline(x=t, color='r', linestyle='--', linewidth=1, alpha=0.7)

            # 绘制用户标记
            for t, color, label in self.markers:
                if t_start <= t <= t_end:
                    ax.axvline(x=t, color=color, linestyle='-', linewidth=2)
                    ax.text(t, y_min + (y_max - y_min) * 0.1, label,
                            fontsize=8, color=color)

        self.figure.suptitle(f"{self.modality} - {self.subject_id} - {self.task}\n"
                             f"时间: {t_start:.2f} - {t_end:.2f} 秒",
                             fontsize=12)

        self.canvas.draw()

    def add_marker_dialog(self):
        """添加标记对话框"""
        # 简化实现：使用鼠标点击位置
        QMessageBox.information(self, "添加标记",
                                "在图上右键点击添加标记\n"
                                "左键起点（绿色），右键终点（红色）")

        # 实际实现需要canvas的pick事件
        # 这里先简单添加一个测试标记
        current_time = (self.current_page * self.page_duration_spin.value() +
                        self.page_duration_spin.value() / 2)
        self.markers.append((current_time, 'green', 'Marker'))
        self.update_plot()

    def clear_markers(self):
        """清除所有标记"""
        self.markers.clear()
        self.update_plot()

    def save_markers(self):
        """保存标记到事件字典"""
        if not self.markers:
            return

        # 更新数据字典的事件
        if "event" not in self.data_dict:
            self.data_dict["event"] = {
                "event_id": [],
                "event_label": [],
                "event_time": [],
                "duration": []
            }

        for t, color, label in self.markers:
            self.data_dict["event"]["event_time"].append(t)
            self.data_dict["event"]["event_label"].append(label)
            self.data_dict["event"]["event_id"].append(len(self.data_dict["event"]["event_id"]) + 1)
            self.data_dict["event"]["duration"].append(0)

        QMessageBox.information(self, "保存成功", f"已保存{len(self.markers)}个标记到数据字典")

    def keyPressEvent(self, event):
        """键盘事件"""
        if event.key() == Qt.Key_Left:
            self.prev_page()
        elif event.key() == Qt.Key_Right:
            self.next_page()
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            # 放大
            pass
        elif event.key() == Qt.Key_Minus:
            # 缩小
            pass
        else:
            super().keyPressEvent(event)


# 测试代码
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    # 创建测试数据
    fs = 1000
    t = np.arange(0, 30, 1 / fs)

    # 生成多模态测试信号
    data_dict = {
        "meta": {
            "subject_id": "test001",
            "session_id": "session1",
            "task": "rest",
            "modality": ["EEG", "EMG", "ECG"]
        },
        "signal": {
            "EEG": {
                "data": np.array([np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t))
                                  for _ in range(8)]),
                "sampling_rate": fs,
                "channel_names": [f"EEG_{i}" for i in range(8)],
                "unit": "uV",
                "signal_type": "eeg"
            },
            "EMG": {
                "data": np.array([np.random.randn(len(t)) * 100 for _ in range(4)]),
                "sampling_rate": fs,
                "channel_names": [f"EMG_{i}" for i in range(4)],
                "unit": "mV",
                "signal_type": "emg"
            }
        },
        "event": {
            "event_time": [5.0, 10.0, 15.0, 20.0],
            "event_label": ["start", "stim1", "stim2", "end"],
            "event_id": [1, 2, 2, 3],
            "duration": [0, 2, 2, 0]
        }
    }

    app = QApplication(sys.argv)
    view = SignalView(data_dict, modality="EEG")
    view.show()
    sys.exit(app.exec_())