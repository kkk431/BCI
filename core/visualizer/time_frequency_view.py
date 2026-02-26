# -*- coding: utf-8 -*-
"""
时频分析可视化模块
核心类：TimeFrequencyView
功能：
- STFT时频图（短时傅里叶变换）
- 通道选择下拉菜单
- 动态色标管理
- 时间-频率坐标系，功率用dB表示
"""

import sys
import matplotlib
import numpy as np
from scipy import signal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QSizePolicy, QFormLayout, QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 动态设置后端，不强制使用QtAgg
import matplotlib
current_backend = matplotlib.get_backend()
print(f"[time_frequency_view] 当前matplotlib后端: {current_backend}")

# 如果还没有后端被设置，且当前不是TkAgg，才尝试设置QtAgg
if current_backend in ['', 'agg'] and 'tk' not in current_backend.lower():
    try:
        matplotlib.use('QtAgg')
        print("[time_frequency_view] 已设置后端为 QtAgg")
    except:
        pass

# -------------------- 辅助函数--------------------
def short_time_Fourier_transform(data, segment_length=256, overlap=128, window='hamming'):
    """
    对多通道数据执行短时傅里叶变换
    返回格式：{'data': [(freqs, times, power), ...], 'ch_names': [...]}
    """
    # 假设 data 是字典，包含 'data' (n_channels x n_samples), 'srate', 'ch_names'
    raw_data = data['data']
    fs = data['srate']
    ch_names = data['ch_names']

    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)

    n_channels = raw_data.shape[0]
    results = []

    for ch in range(n_channels):
        f, t, Zxx = signal.stft(raw_data[ch], fs=fs, nperseg=segment_length,
                                 noverlap=overlap, window=window)
        power = np.abs(Zxx) ** 2
        # 为了符合原格式，将 f, t, power 放入元组
        results.append(([f], [t], [power]))  # 原代码期望嵌套一层列表

    return {
        'data': results,
        'ch_names': ch_names
    }


class TimeFrequencyView(QMainWindow):
    """时频分析查看器"""

    def __init__(self, stft_data, parent=None):
        """
        :param stft_data: STFT结果，格式为 {'data': [(freqs, times, power), ...], 'ch_names': [...]}
        """
        super().__init__(parent)
        self.data = stft_data
        self.current_colorbar = None
        self.setWindowTitle("Time Frequency Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()

    def init_ui(self):
        # 通道选择下拉框
        self.combo_channel = QComboBox()
        self.combo_channel.setFixedWidth(120)
        self.combo_channel.addItems(self.data['ch_names'])
        self.combo_channel.currentIndexChanged.connect(self.on_channel_changed)

        # 控制面板
        control_layout = QFormLayout()
        control_layout.addRow("Channel:", self.combo_channel)

        # 创建画布
        self.fig = Figure(figsize=(8, 6))
        self.axes = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 主布局
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.addLayout(control_layout)
        layout.addStretch(1)
        layout.addWidget(self.canvas)
        layout.addStretch(1)

        # 初始绘制
        self.plot(0)

    def plot(self, channel_idx):
        """绘制指定通道的时频图"""
        # 移除旧色标
        if self.current_colorbar is not None:
            self.current_colorbar.remove()
            self.current_colorbar = None
        self.axes.clear()

        # 获取数据
        freqs, times, power = self.data['data'][channel_idx]
        freqs = freqs[0]          # 去除额外嵌套
        times = times[0]
        power = power[0]

        # 转换为分贝
        power_db = 10 * np.log10(power + 1e-12)

        # 绘制伪彩图
        mesh = self.axes.pcolormesh(times, freqs, power_db, shading='gouraud')
        self.axes.set_ylabel('Frequency [Hz]')
        self.axes.set_xlabel('Time [sec]')
        self.axes.set_title(f'STFT of Channel {self.data["ch_names"][channel_idx]}')

        # 添加色标
        self.current_colorbar = self.fig.colorbar(mesh, ax=self.axes, label='Power/Frequency (dB/Hz)')
        self.canvas.draw()

    def on_channel_changed(self):
        idx = self.combo_channel.currentIndex()
        self.plot(idx)


if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_channels = 32
    sample_rate = 1000
    duration = 10
    n_samples = sample_rate * duration
    raw_data = np.random.randn(n_channels, n_samples)

    simulated_data = {
        'data': raw_data,
        'srate': sample_rate,
        'nchan': n_channels,
        'ch_names': [f'Ch{i}' for i in range(n_channels)],
        'events': [],
        'montage': 'standard_1020'
    }

    # 执行STFT
    stft_result = short_time_Fourier_transform(
        simulated_data,
        segment_length=256,
        overlap=128,
        window='hamming'
    )

    app = QApplication(sys.argv)
    viewer = TimeFrequencyView(stft_result)
    viewer.show()
    sys.exit(app.exec_())