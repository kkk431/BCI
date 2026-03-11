# -*- coding: utf-8 -*-
"""
时频分析可视化模块
核心类：TimeFrequencyView
功能：
- STFT时频图（短时傅里叶变换）
- 模态选择 + 通道选择下拉菜单
- 动态色标管理
- 时间-频率坐标系，功率用dB表示
"""

import sys
import matplotlib
import numpy as np
from scipy import signal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QSizePolicy, QComboBox, QHBoxLayout, QLabel
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 动态设置后端
current_backend = matplotlib.get_backend()
print(f"[time_frequency_view] 当前matplotlib后端: {current_backend}")

if current_backend in ['', 'agg'] and 'tk' not in current_backend.lower():
    try:
        matplotlib.use('QtAgg')
        print("[time_frequency_view] 已设置后端为 QtAgg")
    except:
        pass


# ========== 短时傅里叶变换函数 ==========
def short_time_Fourier_transform(data, segment_length=256, overlap=128, window='hamming'):
    """
    对多通道数据执行短时傅里叶变换
    返回格式：{'data': [(freqs, times, power), ...], 'ch_names': [...]}

    Args:
        data: 字典，包含 'data', 'srate', 'ch_names'
        segment_length: 窗口长度
        overlap: 重叠点数
        window: 窗口类型

    Returns:
        字典，包含 'data' 和 'ch_names'
    """
    # 确保 raw_data 是 numpy 数组
    raw_data = data['data']
    if isinstance(raw_data, list):
        raw_data = np.array(raw_data, dtype=np.float32)
        print(f"STFT: 将列表转换为numpy数组，形状: {raw_data.shape}")

    fs = data['srate']
    ch_names = data['ch_names']

    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)
        print(f"STFT: 将1D数据重塑为: {raw_data.shape}")

    n_channels = raw_data.shape[0]
    n_samples = raw_data.shape[1]
    results = []

    # 确保参数有效
    if segment_length > n_samples:
        segment_length = min(256, n_samples // 4)
        if segment_length < 32:
            segment_length = 32
        print(f"STFT: 调整 segment_length 为: {segment_length}")

    if overlap >= segment_length:
        overlap = segment_length // 2
        print(f"STFT: 调整 overlap 为: {overlap}")

    for ch in range(n_channels):
        try:
            f, t, Zxx = signal.stft(raw_data[ch], fs=fs, nperseg=segment_length,
                                    noverlap=overlap, window=window)
            power = np.abs(Zxx) ** 2
            results.append(([f], [t], [power]))
        except Exception as e:
            print(f"STFT: 通道 {ch} 处理失败: {e}")
            # 返回空数据
            f = np.array([0])
            t = np.array([0])
            power = np.array([[0]])
            results.append(([f], [t], [power]))

    return {
        'data': results,
        'ch_names': ch_names
    }


class TimeFrequencyView(QMainWindow):
    """时频分析查看器（支持多模态）"""

    def __init__(self, data_dict, modality=None, parent=None):
        """
        :param data_dict: 完整的四层数据字典
        :param modality: 初始模态
        """
        super().__init__(parent)
        self.data_dict = data_dict
        self.current_modality = modality
        self.current_channel_idx = 0
        self.current_colorbar = None
        self.stft_result = None

        # 获取所有可用模态
        self.available_modalities = list(data_dict.get("signal", {}).keys())
        if not self.available_modalities:
            raise ValueError("数据字典中没有信号模态")

        # 如果没有指定模态，使用第一个
        if self.current_modality is None:
            self.current_modality = self.available_modalities[0]

        # 加载当前模态的数据
        self.load_modality_data(self.current_modality)

        self.setWindowTitle(f"时频分析 - {self.current_modality}")
        self.setGeometry(100, 100, 900, 700)
        self.init_ui()
        self.update_stft()
        self.plot(0)

    def load_modality_data(self, modality):
        """加载指定模态的数据（修复列表转numpy问题）"""
        signal_info = self.data_dict["signal"][modality]

        # 获取原始数据
        raw_data = signal_info["data"]

        # 如果是列表，转换为numpy数组
        if isinstance(raw_data, list):
            try:
                self.raw_data = np.array(raw_data, dtype=np.float32)
                print(f"将列表转换为numpy数组，形状: {self.raw_data.shape}")
            except Exception as e:
                print(f"列表转换失败: {e}")
                # 如果转换失败，尝试创建模拟数据
                self.raw_data = np.random.randn(8, 1000)
                print("使用模拟数据")
        else:
            self.raw_data = np.array(raw_data)  # 确保是numpy数组

        self.sampling_rate = signal_info["sampling_rate"]

        # 确保数据是2D (channels × samples)
        if self.raw_data.ndim == 1:
            self.raw_data = self.raw_data.reshape(1, -1)
            print(f"将1D数据重塑为: {self.raw_data.shape}")
        elif self.raw_data.ndim == 2:
            # 2D数据：检查是否需要转置
            if self.raw_data.shape[0] > self.raw_data.shape[1]:
                # 如果通道数大于样本数，可能是 (samples, channels) 格式
                print(f"检测到数据可能为 (samples, channels) 格式，形状: {self.raw_data.shape}，进行转置")
                self.raw_data = self.raw_data.T
        elif self.raw_data.ndim > 2:
            # 更高维度：展平
            print(f"检测到{self.raw_data.ndim}D数据，尝试展平")
            self.raw_data = self.raw_data.reshape(self.raw_data.shape[0], -1)

        # 获取通道名称
        self.ch_names = signal_info.get("channel_names",
                                        [f"Ch{i}" for i in range(self.raw_data.shape[0])])

        print(f"加载模态 {modality}: {self.raw_data.shape[0]}通道, {self.sampling_rate}Hz")

    def init_ui(self):
        """初始化用户界面"""
        # 模态选择
        modality_layout = QHBoxLayout()
        modality_layout.addWidget(QLabel("模态:"))
        self.combo_modality = QComboBox()
        self.combo_modality.addItems(self.available_modalities)
        self.combo_modality.setCurrentText(self.current_modality)
        self.combo_modality.currentIndexChanged.connect(self.on_modality_changed)
        modality_layout.addWidget(self.combo_modality)
        modality_layout.addStretch()

        # 通道选择
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("通道:"))
        self.combo_channel = QComboBox()
        self.combo_channel.addItems(self.ch_names)
        self.combo_channel.currentIndexChanged.connect(self.on_channel_changed)
        channel_layout.addWidget(self.combo_channel)
        channel_layout.addStretch()

        # 控制面板
        control_layout = QVBoxLayout()
        control_layout.addLayout(modality_layout)
        control_layout.addLayout(channel_layout)

        # 创建画布
        self.fig = Figure(figsize=(10, 7))
        self.axes = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 主布局
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.addLayout(control_layout)
        layout.addWidget(self.canvas)

    def on_modality_changed(self, index):
        """模态切换"""
        new_modality = self.combo_modality.currentText()
        if new_modality != self.current_modality:
            print(f"切换模态: {self.current_modality} -> {new_modality}")
            self.current_modality = new_modality
            self.load_modality_data(new_modality)

            # 更新通道下拉框
            self.combo_channel.clear()
            self.combo_channel.addItems(self.ch_names)

            # 重新计算STFT
            self.update_stft()

            # 重新绘图
            self.plot(0)
            self.setWindowTitle(f"时频分析 - {self.current_modality}")

    def on_channel_changed(self, index):
        """通道切换"""
        self.plot(index)

    def update_stft(self):
        """更新STFT计算结果（带参数调整）"""
        # 确保 raw_data 是 numpy 数组
        if isinstance(self.raw_data, list):
            self.raw_data = np.array(self.raw_data, dtype=np.float32)
            print(f"update_stft: 将列表转换为numpy数组，形状: {self.raw_data.shape}")

        # 根据数据长度动态调整STFT参数
        n_samples = self.raw_data.shape[1]

        # 根据数据长度选择合适的窗口长度
        if n_samples < 256:
            segment_length = min(64, n_samples // 4)
            overlap = segment_length // 2
            print(f"数据长度较短 ({n_samples}), 使用 segment_length={segment_length}, overlap={overlap}")
        else:
            segment_length = 256
            overlap = 128

        stft_input = {
            'data': self.raw_data,
            'srate': self.sampling_rate,
            'ch_names': self.ch_names
        }

        # 调用STFT函数
        self.stft_result = short_time_Fourier_transform(
            stft_input,
            segment_length=segment_length,
            overlap=overlap
        )

    def plot(self, channel_idx):
        """绘制指定通道的时频图"""
        if self.stft_result is None:
            return

        # 移除旧色标
        if self.current_colorbar is not None:
            self.current_colorbar.remove()
            self.current_colorbar = None
        self.axes.clear()

        # 获取数据
        try:
            freqs, times, power = self.stft_result['data'][channel_idx]
            freqs = freqs[0]
            times = times[0]
            power = power[0]

            # 转换为分贝
            power_db = 10 * np.log10(power + 1e-12)

            # 绘制伪彩图
            mesh = self.axes.pcolormesh(times, freqs, power_db, shading='gouraud')
            self.axes.set_ylabel('频率 [Hz]')
            self.axes.set_xlabel('时间 [秒]')
            self.axes.set_title(f'{self.current_modality} - 通道 {self.ch_names[channel_idx]}')

            # 添加色标
            self.current_colorbar = self.fig.colorbar(mesh, ax=self.axes, label='功率/频率 (dB/Hz)')
        except Exception as e:
            print(f"绘图错误: {e}")
            self.axes.text(0.5, 0.5, f"绘图错误: {str(e)}",
                           ha='center', va='center', transform=self.axes.transAxes)

        self.canvas.draw()


if __name__ == "__main__":
    # 测试代码
    app = QApplication(sys.argv)

    # 创建测试数据
    fs = 1000
    t = np.arange(0, 10, 1 / fs)
    n_channels = 8
    data = np.random.randn(n_channels, len(t))

    test_data = {
        "signal": {
            "EEG": {
                "data": data,
                "sampling_rate": fs,
                "channel_names": [f"EEG_{i}" for i in range(n_channels)],
                "unit": "uV"
            },
            "EMG": {
                "data": data * 0.1,
                "sampling_rate": fs,
                "channel_names": [f"EMG_{i}" for i in range(n_channels)],
                "unit": "mV"
            }
        }
    }

    viewer = TimeFrequencyView(test_data, modality="EEG")
    viewer.show()
    sys.exit(app.exec_())