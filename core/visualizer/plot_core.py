# -*- coding: utf-8 -*-
"""
核心绘图模块
提供静态函数：
- plot_raw：绘制原始脑电波形（单/多通道）
- plot_eeg_psd：绘制功率谱密度拓扑图
- plot_raw_by_file：通过文件对话框打开文件并绘图
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
import matplotlib
import mne
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog, QWidget

from core.io.data_io import DataLoader

matplotlib.use('QtAgg')


# -------------------- 辅助归一化函数--------------------
def min_max_scaling_to_range(data, target_range=(-1, 1)):
    data = np.asarray(data)
    min_vals = data.min(axis=1, keepdims=True)
    max_vals = data.max(axis=1, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    scaled = (data - min_vals) / range_vals
    scaled = scaled * (target_range[1] - target_range[0]) + target_range[0]
    return scaled


def min_max_scaling_by_arrays(data, target_range=(-1, 1)):
    data = np.asarray(data)
    min_val = data.min()
    max_val = data.max()
    if max_val - min_val == 0:
        return np.zeros_like(data)
    scaled = (data - min_val) / (max_val - min_val)
    scaled = scaled * (target_range[1] - target_range[0]) + target_range[0]
    return scaled


def plot_raw_by_file(widget, path=None):
    """
    通过QT文件对话框读取文件并绘制原始脑电图数据
    :param widget: 父窗口（用于文件对话框）
    :param path: 可选路径，若提供则直接使用
    """
    if path is None:
        path, _ = QFileDialog.getOpenFileName(widget, "Open Data File", "",
                                              "All supported files (*.edf *.bdf *.gdf *.csv *.txt *.xlsx *.set *.vhdr *.mat *.npy *.npz)")
        if not path:
            return

    loader = DataLoader()
    try:
        data_dict = loader.load(path)
        # 从标准数据字典中提取原始数据和通道名
        # 假设第一个 modality 的信号是目标
        if 'signal' in data_dict and data_dict['signal']:
            mod = list(data_dict['signal'].keys())[0]
            signal_info = data_dict['signal'][mod]
            raw_data = signal_info['data']
            ch_names = signal_info.get('channel_names', [])
            plot_raw(data=raw_data, channel=ch_names)
        else:
            print("No signal data found in file.")
    except Exception as e:
        print(f"Error loading file: {e}")


def plot_raw(data, channel=None, sharey=False, line_color='black', linewidth=0.5,
             is_save=False, save_path=None):
    """
    绘制单通道或多通道原始脑电波形
    :param data: 1D 或 2D numpy数组
    :param channel: 通道名称列表
    :param sharey: 是否共享Y轴
    :param line_color: 线条颜色
    :param linewidth: 线宽
    :param is_save: 是否保存图片
    :param save_path: 保存路径
    """
    # 统一转为numpy数组
    data = np.array(data)
    dims = data.ndim

    if dims == 1:
        # 单通道
        length = data.shape[0]
        if channel is None:
            channel = ['channel']
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(data, color=line_color, linewidth=linewidth)
        ax.set_ylabel(f' {channel[0]}', rotation=0, ha='right')
        ax.tick_params(axis='both', which='both', bottom=False, top=False,
                       labelbottom=False, left=False, right=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_color('lightgrey')
        ax.set_xlim(left=-10, right=length)
        fig.subplots_adjust(hspace=0, wspace=0, bottom=0.02, left=0.1, top=0.98, right=0.98)

    elif dims == 2:
        # 多通道：形状 (n_channels, n_times)
        n_channels, n_times = data.shape
        if channel is None:
            channel = [str(i+1) for i in range(n_channels)]
        fig, axes = plt.subplots(n_channels, 1, figsize=(10, 6), sharex=True, sharey=sharey)
        if n_channels == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(data[i, :30000], color=line_color, linewidth=linewidth)  # 限制点数避免拥挤
            ax.set_ylabel(f' {channel[i]}', rotation=0, ha='right')
            ax.tick_params(axis='both', which='both', bottom=False, top=False,
                           labelbottom=False, left=False, right=False, labelleft=False)
            for spine in ax.spines.values():
                spine.set_color('lightgrey')
            ax.set_xlim(left=-10)

        # 底部子图显示刻度
        axes[-1].tick_params(axis='both', which='both', bottom=True, top=False,
                             left=False, right=False, labelbottom=True)
        fig.subplots_adjust(hspace=0, wspace=0, bottom=0.05, left=0.1, top=0.98, right=0.98)

    else:
        print(f"Unsupported data dimension: {dims}")
        return

    if is_save and save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()  # 或根据需要返回 figure


def plot_eeg_psd(data, is_relative=False, is_norm=True, title=''):
    """
    绘制脑电功率谱密度拓扑图
    :param data: 字典，包含 'data' (特征数组), 'ch_names', 'srate' 等
    :param is_relative: 相对功率归一化
    :param is_norm: 是否进行归一化
    :param title: 图表标题前缀
    """
    num_bands = np.array(data['data']).shape[1]
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'][:num_bands]

    fig, axes = plt.subplots(1, num_bands, figsize=(8, 5), sharex=True, sharey=True)
    if num_bands == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0, wspace=0.05, bottom=0.08, left=0.05, top=0.88, right=0.98)
    fig.suptitle(f"{title} EEG Power Spectral Density")

    # 准备MNE对象
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=data['ch_names'], sfreq=data.get('srate', 1000), ch_types='eeg')
    evoked = mne.EvokedArray(data=np.array(data['data']), info=info)
    evoked.set_montage(montage)

    # 数据处理
    plot_data = np.array(data['data']).T  # shape (bands, channels)
    if is_norm:
        if is_relative:
            norm_data = min_max_scaling_to_range(plot_data)
        else:
            norm_data = min_max_scaling_by_arrays(plot_data)
        vlim = (-1, 1)
    else:
        norm_data = plot_data
        vlim = (np.min(plot_data), np.max(plot_data))

    for i, (ax, name) in enumerate(zip(axes, band_names)):
        ax.set_title(name)
        mne.viz.plot_topomap(norm_data[i], evoked.info, axes=ax, show=False,
                             sensors=True, vlim=vlim)
    plt.show()


# 简单测试
if __name__ == "__main__":
    # 生成示例多通道数据
    fs = 1000
    t = np.linspace(0, 10, fs*10)
    data_2d = np.array([np.sin(2*np.pi*10*t) + 0.5*np.random.randn(len(t)) for _ in range(8)])
    plot_raw(data_2d, channel=[f'Ch{i}' for i in range(8)], sharey=False)