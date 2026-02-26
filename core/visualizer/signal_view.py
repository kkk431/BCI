#!/usr/bin/env python3
"""
signal_view.py
Tkinter版本 - 通用信号视图
支持所有模态的多通道信号可视化
基于四层数据格式: meta/signal/event/processed
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import scipy.signal
from typing import Dict, List, Optional, Tuple, Any
import os
from datetime import datetime


class SignalView(tk.Frame):
    """
    通用信号视图类 - Tkinter版本
    支持EEG/EMG/ECG/GSR/fNIRS/ET/RESP等多种模态
    """

    def __init__(self, parent, data_dict: Dict[str, Any], modality: str = None):
        """
        初始化信号视图

        Args:
            parent: 父窗口
            data_dict: 标准四层数据字典
            modality: 要显示的模态（None表示自动选择第一个）
        """
        super().__init__(parent)
        self.parent = parent
        self.data_dict = data_dict
        self.modality = modality

        # 解析数据
        self._parse_data()

        # 当前显示状态
        self.current_page = 0
        self.page_duration = 5.0  # 默认每页5秒
        self.zoom_level = 1.0
        self.markers = []  # 事件标记 [(time, color, label)]
        self.selected_channels = []  # 选中的通道

        # 创建matplotlib图形
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = None

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

    def setup_ui(self):
        """设置用户界面"""
        # 主布局
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ========== 顶部控制栏 ==========
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        # 左侧信息
        info_text = f"{self.modality} | {self.n_channels}通道 | {self.sampling_rate}Hz | {self.duration:.2f}秒"
        info_label = ttk.Label(control_frame, text=info_text, font=('微软雅黑', 10))
        info_label.pack(side=tk.LEFT, padx=5)

        # 右侧翻页控制
        page_frame = ttk.Frame(control_frame)
        page_frame.pack(side=tk.RIGHT)

        ttk.Label(page_frame, text="页:").pack(side=tk.LEFT)

        self.page_var = tk.StringVar(value="1")
        self.page_spin = ttk.Spinbox(page_frame, from_=1,
                                     to=max(1, int(np.ceil(self.duration / self.page_duration))),
                                     width=5, textvariable=self.page_var,
                                     command=self.on_page_changed)
        self.page_spin.pack(side=tk.LEFT, padx=2)

        ttk.Label(page_frame, text="/").pack(side=tk.LEFT)

        self.total_pages_label = ttk.Label(page_frame,
                                           text=str(max(1, int(np.ceil(self.duration / self.page_duration)))))
        self.total_pages_label.pack(side=tk.LEFT, padx=2)

        ttk.Label(page_frame, text="每页(秒):").pack(side=tk.LEFT, padx=(10, 2))

        self.page_duration_var = tk.StringVar(value="5")
        self.page_duration_spin = ttk.Spinbox(page_frame, from_=1, to=60, width=5,
                                              textvariable=self.page_duration_var,
                                              command=self.on_page_duration_changed)
        self.page_duration_spin.pack(side=tk.LEFT, padx=2)

        # 翻页按钮
        ttk.Button(page_frame, text="◀", width=3,
                   command=self.prev_page).pack(side=tk.LEFT, padx=2)
        ttk.Button(page_frame, text="▶", width=3,
                   command=self.next_page).pack(side=tk.LEFT, padx=2)

        # ========== 第二行：滤波控制 ==========
        filter_frame = ttk.LabelFrame(main_frame, text="滤波")
        filter_frame.pack(fill=tk.X, pady=5)

        self.filter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_frame, text="启用滤波",
                        variable=self.filter_var,
                        command=self.on_filter_toggled).pack(side=tk.LEFT, padx=5)

        ttk.Label(filter_frame, text="低通(Hz):").pack(side=tk.LEFT, padx=(10, 2))
        self.lowpass_var = tk.StringVar(value="45")
        lowpass_entry = ttk.Entry(filter_frame, textvariable=self.lowpass_var, width=8)
        lowpass_entry.pack(side=tk.LEFT, padx=2)
        lowpass_entry.bind('<Return>', lambda e: self.update_plot())

        ttk.Label(filter_frame, text="高通(Hz):").pack(side=tk.LEFT, padx=(10, 2))
        self.highpass_var = tk.StringVar(value="0.5")
        highpass_entry = ttk.Entry(filter_frame, textvariable=self.highpass_var, width=8)
        highpass_entry.pack(side=tk.LEFT, padx=2)
        highpass_entry.bind('<Return>', lambda e: self.update_plot())

        ttk.Label(filter_frame, text="陷波(Hz):").pack(side=tk.LEFT, padx=(10, 2))
        self.notch_var = tk.StringVar(value="50")
        notch_entry = ttk.Entry(filter_frame, textvariable=self.notch_var, width=8)
        notch_entry.pack(side=tk.LEFT, padx=2)
        notch_entry.bind('<Return>', lambda e: self.update_plot())

        # 初始禁用
        lowpass_entry.config(state='disabled')
        highpass_entry.config(state='disabled')
        notch_entry.config(state='disabled')

        # ========== 幅度控制 ==========
        amp_frame = ttk.LabelFrame(main_frame, text="幅度范围")
        amp_frame.pack(fill=tk.X, pady=5)

        ttk.Label(amp_frame, text="最小:").pack(side=tk.LEFT, padx=5)
        self.amp_min_var = tk.StringVar(value="-200")
        amp_min_entry = ttk.Entry(amp_frame, textvariable=self.amp_min_var, width=8)
        amp_min_entry.pack(side=tk.LEFT, padx=2)
        amp_min_entry.bind('<Return>', lambda e: self.update_plot())

        ttk.Label(amp_frame, text="最大:").pack(side=tk.LEFT, padx=(10, 2))
        self.amp_max_var = tk.StringVar(value="200")
        amp_max_entry = ttk.Entry(amp_frame, textvariable=self.amp_max_var, width=8)
        amp_max_entry.pack(side=tk.LEFT, padx=2)
        amp_max_entry.bind('<Return>', lambda e: self.update_plot())

        self.auto_amp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(amp_frame, text="自动",
                        variable=self.auto_amp_var,
                        command=self.on_auto_amp_toggled).pack(side=tk.LEFT, padx=10)

        # 初始禁用手动输入
        amp_min_entry.config(state='disabled')
        amp_max_entry.config(state='disabled')

        # ========== 中间：绘图区域 + 通道列表 ==========
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 左侧：通道列表
        channel_frame = ttk.LabelFrame(middle_frame, text="通道选择", width=150)
        channel_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        channel_frame.pack_propagate(False)

        # 通道列表
        self.channel_listbox = tk.Listbox(channel_frame, selectmode=tk.MULTIPLE,
                                          exportselection=False)
        self.channel_listbox.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        for i, name in enumerate(self.channel_names):
            self.channel_listbox.insert(tk.END, f"{i + 1:02d}. {name}")
            self.channel_listbox.selection_set(i)  # 默认全选
            self.selected_channels.append(i)

        self.channel_listbox.bind('<<ListboxSelect>>', self.on_channel_selected)

        # 全选按钮
        ttk.Button(channel_frame, text="全选",
                   command=self.select_all_channels).pack(fill=tk.X, padx=2, pady=2)

        # 右侧：绘图区域
        plot_frame = ttk.Frame(middle_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 创建matplotlib画布
        self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 工具栏
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # ========== 底部：事件标记区域 ==========
        event_frame = ttk.LabelFrame(main_frame, text="事件标记")
        event_frame.pack(fill=tk.X, pady=5)

        # 事件列表
        self.event_listbox = tk.Listbox(event_frame, height=3)
        self.event_listbox.pack(fill=tk.X, padx=2, pady=2)

        for t, label in zip(self.event_times, self.event_labels):
            self.event_listbox.insert(tk.END, f"{t:.3f}s: {label}")

        # 按钮
        btn_frame = ttk.Frame(event_frame)
        btn_frame.pack(fill=tk.X, pady=2)

        ttk.Button(btn_frame, text="添加标记",
                   command=self.add_marker_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="清除标记",
                   command=self.clear_markers).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="保存标记",
                   command=self.save_markers).pack(side=tk.LEFT, padx=2)

    def on_filter_toggled(self):
        """滤波开关"""
        enabled = self.filter_var.get()
        # 获取输入框并设置状态
        for child in self.winfo_children():
            if isinstance(child, ttk.LabelFrame) and child['text'] == '滤波':
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, ttk.Entry):
                        grandchild.config(state='normal' if enabled else 'disabled')
        self.update_plot()

    def on_auto_amp_toggled(self):
        """自动幅度开关"""
        auto = self.auto_amp_var.get()
        # 找到幅度输入框并设置状态
        for child in self.winfo_children():
            if isinstance(child, ttk.LabelFrame) and child['text'] == '幅度范围':
                entries = []
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, ttk.Entry):
                        entries.append(grandchild)
                if len(entries) >= 2:
                    entries[0].config(state='disabled' if auto else 'normal')
                    entries[1].config(state='disabled' if auto else 'normal')
        self.update_plot()

    def on_page_changed(self):
        """页码改变"""
        try:
            self.current_page = int(self.page_var.get()) - 1
            self.update_plot()
        except:
            pass

    def on_page_duration_changed(self):
        """每页时长改变"""
        try:
            self.page_duration = float(self.page_duration_var.get())
            max_pages = max(1, int(np.ceil(self.duration / self.page_duration)))
            self.page_spin.config(to=max_pages)
            self.total_pages_label.config(text=str(max_pages))
            self.update_plot()
        except:
            pass

    def on_channel_selected(self, event):
        """通道选择改变"""
        self.selected_channels = self.channel_listbox.curselection()
        self.update_plot()

    def prev_page(self):
        """上一页"""
        if self.current_page > 0:
            self.current_page -= 1
            self.page_var.set(str(self.current_page + 1))
            self.update_plot()

    def next_page(self):
        """下一页"""
        max_pages = max(1, int(np.ceil(self.duration / self.page_duration)))
        if self.current_page < max_pages - 1:
            self.current_page += 1
            self.page_var.set(str(self.current_page + 1))
            self.update_plot()

    def select_all_channels(self):
        """全选通道"""
        self.channel_listbox.selection_set(0, tk.END)
        self.selected_channels = list(range(self.n_channels))
        self.update_plot()

    def get_selected_channels(self) -> List[int]:
        """获取选中的通道索引"""
        return list(self.selected_channels) if self.selected_channels else list(range(self.n_channels))

    def apply_filter(self, data: np.ndarray) -> np.ndarray:
        """应用滤波"""
        if not self.filter_var.get():
            return data

        filtered = data.copy()
        fs = self.sampling_rate

        try:
            # 低通滤波
            lowcut = float(self.lowpass_var.get())
            if lowcut > 0 and lowcut < fs / 2:
                sos = scipy.signal.butter(4, lowcut, 'lowpass', fs=fs, output='sos')
                filtered = scipy.signal.sosfiltfilt(sos, filtered, axis=1)

            # 高通滤波
            highcut = float(self.highpass_var.get())
            if highcut > 0:
                sos = scipy.signal.butter(4, highcut, 'highpass', fs=fs, output='sos')
                filtered = scipy.signal.sosfiltfilt(sos, filtered, axis=1)

            # 陷波滤波
            notch = float(self.notch_var.get())
            if notch > 0 and notch < fs / 2:
                Q = 30
                b, a = scipy.signal.iirnotch(notch, Q, fs)
                filtered = scipy.signal.filtfilt(b, a, filtered, axis=1)
        except:
            pass

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
        gs = self.figure.add_gridspec(n_show, 1, hspace=0.1)

        # 计算幅度范围
        if self.auto_amp_var.get():
            # 自动计算
            y_min = np.min(data_filtered[selected_channels])
            y_max = np.max(data_filtered[selected_channels])
            margin = (y_max - y_min) * 0.1
            y_min -= margin
            y_max += margin
        else:
            try:
                y_min = float(self.amp_min_var.get())
                y_max = float(self.amp_max_var.get())
            except:
                y_min = -200
                y_max = 200

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
        # 简化实现
        current_time = (self.current_page * self.page_duration +
                        self.page_duration / 2)
        self.markers.append((current_time, 'green', f'Marker{len(self.markers) + 1}'))
        self.update_plot()
        messagebox.showinfo("添加标记", f"已添加标记 at {current_time:.2f}s")

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

        messagebox.showinfo("保存成功", f"已保存{len(self.markers)}个标记到数据字典")

    def destroy(self):
        """销毁时清理"""
        plt.close(self.figure)
        super().destroy()


class fNIRSView(SignalView):
    """
    fNIRS专用视图类
    继承自SignalView，增加HbO/HbR分离显示功能
    """

    def __init__(self, parent, data_dict: Dict[str, Any], modality: str = "fNIRS"):
        super().__init__(parent, data_dict, modality)

        # 添加fNIRS特有的控制
        self.setup_fnirs_controls()

    def setup_fnirs_controls(self):
        """设置fNIRS特有的控制"""
        # 找到滤波框架后面的位置插入
        for child in self.winfo_children():
            if isinstance(child, ttk.Frame):
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, ttk.LabelFrame) and grandchild['text'] == '滤波':
                        # 在滤波框架后面添加fNIRS控制
                        fnirs_frame = ttk.LabelFrame(child, text="fNIRS设置")
                        fnirs_frame.pack(fill=tk.X, pady=5, after=grandchild)

                        # HbO/HbR选择
                        self.hbo_var = tk.BooleanVar(value=True)
                        self.hbr_var = tk.BooleanVar(value=True)

                        ttk.Checkbutton(fnirs_frame, text="显示HbO",
                                        variable=self.hbo_var,
                                        command=self.update_plot).pack(side=tk.LEFT, padx=5)
                        ttk.Checkbutton(fnirs_frame, text="显示HbR",
                                        variable=self.hbr_var,
                                        command=self.update_plot).pack(side=tk.LEFT, padx=5)

                        # 通道配对信息
                        ttk.Label(fnirs_frame, text="通道配对:", font=('微软雅黑', 9)).pack(side=tk.LEFT, padx=(20, 2))
                        self.pair_info_var = tk.StringVar(value="自动")
                        ttk.Label(fnirs_frame, textvariable=self.pair_info_var).pack(side=tk.LEFT)
                        break
                break

    def _parse_data(self):
        """重写数据解析，处理fNIRS特有的数据格式"""
        super()._parse_data()

        # 检查是否有HbO/HbR分离数据
        if hasattr(self, 'data') and self.data is not None:
            # 假设数据格式：前一半通道是HbO，后一半是HbR
            n_channels = self.n_channels
            if n_channels % 2 == 0:
                self.n_hbo = n_channels // 2
                self.n_hbr = n_channels // 2
            else:
                self.n_hbo = (n_channels + 1) // 2
                self.n_hbr = n_channels // 2

    def get_selected_channels(self) -> List[int]:
        """重写通道选择，根据HbO/HbR设置筛选"""
        selected = super().get_selected_channels()

        if not hasattr(self, 'hbo_var'):
            return selected

        # 根据HbO/HbR设置过滤
        filtered = []
        for idx in selected:
            if idx < self.n_hbo and self.hbo_var.get():
                filtered.append(idx)
            elif idx >= self.n_hbo and self.hbr_var.get():
                filtered.append(idx)

        return filtered if filtered else selected

    def update_plot(self):
        """重写绘图方法，添加fNIRS特有的标记"""
        # 更新通道配对信息
        if hasattr(self, 'pair_info_var'):
            n_show = len(self.get_selected_channels())
            self.pair_info_var.set(f"显示{n_show}通道")

        super().update_plot()


# 在文件末尾添加测试代码
if __name__ == "__main__":
    import sys

    root = tk.Tk()
    root.title("fNIRS视图测试")
    root.geometry("1200x800")

    # 创建fNIRS测试数据
    fs = 100
    t = np.arange(0, 60, 1 / fs)

    # 模拟HbO和HbR数据
    hbo_data = np.array([0.5 * np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(len(t)) + 1.0
                         for _ in range(4)])
    hbr_data = np.array([0.3 * np.sin(2 * np.pi * 0.1 * t + 0.5) + 0.1 * np.random.randn(len(t)) + 0.5
                         for _ in range(4)])

    data_dict = {
        "meta": {
            "subject_id": "test001",
            "session_id": "session1",
            "task": "rest",
            "modality": ["fNIRS"]
        },
        "signal": {
            "fNIRS": {
                "data": np.vstack([hbo_data, hbr_data]),
                "sampling_rate": fs,
                "channel_names": [f"HbO_{i}" for i in range(4)] + [f"HbR_{i}" for i in range(4)],
                "unit": "μM"
            }
        }
    }

    view = fNIRSView(root, data_dict, modality="fNIRS")
    view.pack(fill=tk.BOTH, expand=True)

    root.mainloop()