#!/usr/bin/env python3
"""
plot_dialog.py
Tkinter版本 - 独立绘图对话框
用于弹出式信号查看
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


class PlotDialog(tk.Toplevel):
    """
    独立绘图对话框 - Tkinter版本
    用于快速查看信号片段
    """

    def __init__(self, parent, data: np.ndarray, sampling_rate: float,
                 channel_names: List[str] = None,
                 title: str = "信号视图"):
        """
        初始化绘图对话框

        Args:
            parent: 父窗口
            data: 信号数据 (channels × samples) 或 (samples,)
            sampling_rate: 采样率 (Hz)
            channel_names: 通道名称列表
            title: 对话框标题
        """
        super().__init__(parent)
        self.parent = parent
        self.title(title)
        self.geometry("1000x700")

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

        # 当前显示状态
        self.current_page = 0
        self.page_duration = 5.0  # 默认每页5秒
        self.markers = []  # 事件标记

        # 创建matplotlib图形
        self.figure = Figure(figsize=(10, 6), dpi=100)

        # 设置UI
        self.setup_ui()

        # 更新绘图
        self.update_plot()

        # 设置窗口关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        """设置用户界面"""
        # 主布局
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ========== 顶部控制栏 ==========
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        # 信息显示
        info_text = f"{self.n_channels}通道 | {self.sampling_rate}Hz | {self.duration:.2f}秒"
        info_label = ttk.Label(control_frame, text=info_text, font=('微软雅黑', 9))
        info_label.pack(side=tk.LEFT, padx=5)

        # 翻页控制
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

        # ========== 绘图区域 ==========
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 工具栏
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # ========== 底部按钮 ==========
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="添加标记",
                   command=self.add_marker).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="清除标记",
                   command=self.clear_markers).pack(side=tk.LEFT, padx=2)

        ttk.Button(btn_frame, text="保存图像",
                   command=self.save_plot).pack(side=tk.RIGHT, padx=2)
        ttk.Button(btn_frame, text="关闭",
                   command=self.on_close).pack(side=tk.RIGHT, padx=2)

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

        self.figure.suptitle(f"{self.title()} - 时间: {t_start:.2f} - {t_end:.2f} 秒", fontsize=12)

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
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG图像", "*.png"), ("PDF文件", "*.pdf"), ("SVG图像", "*.svg")]
        )

        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("保存成功", f"图像已保存到:\n{file_path}")
            except Exception as e:
                messagebox.showerror("保存失败", f"保存图像时出错:\n{str(e)}")

    def on_close(self):
        """关闭窗口"""
        plt.close(self.figure)
        self.destroy()


# 便捷函数：快速弹出绘图窗口
def quick_plot(parent, data, sampling_rate=1000, channel_names=None, title="信号视图"):
    """
    快速弹出绘图窗口

    Args:
        parent: 父窗口
        data: 信号数据
        sampling_rate: 采样率
        channel_names: 通道名称
        title: 窗口标题
    """
    dialog = PlotDialog(parent, data, sampling_rate, channel_names, title)
    dialog.grab_set()  # 模态对话框


# 测试代码
"""if __name__ == "__main__":
    import sys

    root = tk.Tk()
    root.title("绘图对话框测试")
    root.geometry("400x300")

    # 创建测试信号
    fs = 1000
    t = np.arange(0, 30, 1 / fs)
    data = np.array([np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t)) for _ in range(8)])


    def test_dialog():
        quick_plot(root, data, fs, [f"Ch{i}" for i in range(8)], "测试信号")


    ttk.Button(root, text="打开绘图对话框", command=test_dialog).pack(expand=True)

    root.mainloop()"""