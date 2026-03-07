# -*- coding: utf-8 -*-
"""
特征可视化模块 - 最佳布局版本
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import json
import os

# ==================== 全局字体设置（绝对避免□）====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def read_info(file_path):
    """从 JSON 文件读取传感器信息"""
    with open(file_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    return info


class PlotSettingsDialog(tk.Toplevel):
    """绘图设置对话框"""

    def __init__(self, parent, initial_settings=None):
        super().__init__(parent)
        self.parent = parent
        self.title("绘图设置")
        self.initial_settings = initial_settings or {}
        self.result = None

        self.geometry("450x350")
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)

        self._create_widgets()
        self.center_window()

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        ttk.Label(main_frame, text="标题:").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.title_entry = ttk.Entry(main_frame, width=30)
        self.title_entry.insert(0, self.initial_settings.get("title", ""))
        self.title_entry.grid(row=0, column=1, pady=8)

        # X轴标签
        ttk.Label(main_frame, text="X轴标签:").grid(row=1, column=0, sticky=tk.W, pady=8)
        self.xlabel_entry = ttk.Entry(main_frame, width=30)
        self.xlabel_entry.insert(0, self.initial_settings.get("xlabel", ""))
        self.xlabel_entry.grid(row=1, column=1, pady=8)

        # Y轴标签
        ttk.Label(main_frame, text="Y轴标签:").grid(row=2, column=0, sticky=tk.W, pady=8)
        self.ylabel_entry = ttk.Entry(main_frame, width=30)
        self.ylabel_entry.insert(0, self.initial_settings.get("ylabel", ""))
        self.ylabel_entry.grid(row=2, column=1, pady=8)

        # 宽度
        ttk.Label(main_frame, text="宽度(英寸):").grid(row=3, column=0, sticky=tk.W, pady=8)
        self.width_spin = ttk.Spinbox(main_frame, from_=4, to=20, width=10)
        self.width_spin.set(self.initial_settings.get("width", 10))
        self.width_spin.grid(row=3, column=1, sticky=tk.W, pady=8)

        # 高度
        ttk.Label(main_frame, text="高度(英寸):").grid(row=4, column=0, sticky=tk.W, pady=8)
        self.height_spin = ttk.Spinbox(main_frame, from_=4, to=15, width=10)
        self.height_spin.set(self.initial_settings.get("height", 7))
        self.height_spin.grid(row=4, column=1, sticky=tk.W, pady=8)

        # 按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=20)

        ttk.Button(btn_frame, text="保存", command=self.save, width=10).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="取消", command=self.destroy, width=10).pack(side=tk.LEFT, padx=10)

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def save(self):
        self.result = {
            "title": self.title_entry.get(),
            "xlabel": self.xlabel_entry.get(),
            "ylabel": self.ylabel_entry.get(),
            "width": int(self.width_spin.get()),
            "height": int(self.height_spin.get())
        }
        self.destroy()


# ---------- 画布基类 ----------
class BaseCanvas:
    """绘图画布基类"""

    def __init__(self, parent, width=10, height=7, dpi=100):
        self.parent = parent
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.ax = self.fig.add_subplot(111)

    def get_widget(self):
        return self.canvas.get_tk_widget()

    def draw(self):
        self.canvas.draw()


class CurveCanvas(BaseCanvas):
    """曲线图 - 横坐标是特征，不同颜色曲线代表不同通道"""

    def plot(self, data, selected_channels, selected_features, settings):
        # ===== 确保 settings 有默认值 =====
        if "title" not in settings or not settings["title"]:
            settings["title"] = "特征曲线图"  # 直接写死，不用 plot_type
        if "xlabel" not in settings or not settings["xlabel"]:
            settings["xlabel"] = "特征"
        if "ylabel" not in settings or not settings["ylabel"]:
            settings["ylabel"] = "幅值"

        self.ax.clear()
        select_data = self._get_selected_data(data, selected_channels, selected_features)

        x = np.arange(len(selected_features))
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_channels)))

        for i, ch in enumerate(selected_channels):
            values = [select_data[f][i] for f in selected_features]
            self.ax.plot(x, values, marker='o', linewidth=2, label=ch,
                         color=colors[i], markersize=8, markerfacecolor='white')

            # 标记数值
            for j, (xi, val) in enumerate(zip(x, values)):
                self.ax.annotate(f'{val:.1f}', (xi, val),
                                 textcoords="offset points", xytext=(0, 10),
                                 ha='center', fontsize=8)

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(selected_features, fontsize=9, rotation=30, ha='right')

        # ===== 强制设置标题和标签 =====
        title = settings.get("title", "特征曲线图")
        xlabel = settings.get("xlabel", "特征")
        ylabel = settings.get("ylabel", "幅值")

        print(f"设置标题: {title}, X轴: {xlabel}, Y轴: {ylabel}")  # 调试用

        self.ax.set_title(title, fontsize=14, pad=15)
        self.ax.set_xlabel(xlabel, fontsize=11)
        self.ax.set_ylabel(ylabel, fontsize=11)

        self.ax.legend(loc='best', fontsize=9, title="通道")
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.fig.tight_layout()
        self.draw()

    def _get_selected_data(self, feature_dict, channels, features):
        idx = [i for i, ch in enumerate(feature_dict['ch_names']) if ch in channels]
        out = {}
        for f in features:
            if f in feature_dict['feature']:
                out[f] = [feature_dict['feature'][f][i] for i in idx]
        return out


class BarCanvas(BaseCanvas):
    """柱状图 - 横坐标是特征，颜色代表通道"""

    def plot(self, data, selected_channels, selected_features, settings):
        # ===== 确保 settings 有默认值 =====
        if "title" not in settings or not settings["title"]:
            settings["title"] = "特征柱状图"  # 直接写死
        if "xlabel" not in settings or not settings["xlabel"]:
            settings["xlabel"] = "特征"
        if "ylabel" not in settings or not settings["ylabel"]:
            settings["ylabel"] = "幅值"

        self.ax.clear()
        select_data = self._get_selected_data(data, selected_channels, selected_features)

        x = np.arange(len(selected_features))
        width = 0.8 / len(selected_channels)

        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_channels)))

        for i, ch in enumerate(selected_channels):
            values = [select_data[f][i] for f in selected_features]
            offset = (i - len(selected_channels) / 2 + 0.5) * width
            bars = self.ax.bar(x + offset, values, width, label=ch,
                               color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.8)

            # 在柱子上显示数值
            for j, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                             f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(selected_features, fontsize=9, rotation=30, ha='right')

        # ===== 强制设置标题和标签 =====
        title = settings.get("title", "特征柱状图")
        xlabel = settings.get("xlabel", "特征")
        ylabel = settings.get("ylabel", "幅值")

        print(f"设置标题: {title}, X轴: {xlabel}, Y轴: {ylabel}")  # 调试用

        self.ax.set_title(title, fontsize=14, pad=15)
        self.ax.set_xlabel(xlabel, fontsize=11)
        self.ax.set_ylabel(ylabel, fontsize=11)

        self.ax.legend(loc='best', fontsize=9, title="通道")
        self.ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.fig.tight_layout()
        self.draw()

    def _get_selected_data(self, feature_dict, channels, features):
        idx = [i for i, ch in enumerate(feature_dict['ch_names']) if ch in channels]
        out = {}
        for f in features:
            if f in feature_dict['feature']:
                out[f] = [feature_dict['feature'][f][i] for i in idx]
        return out


class TableCanvas(ttk.Frame):
    """表格画布"""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.tree = None
        self._create_widgets()

    def _create_widgets(self):
        # 创建滚动条
        vsb = ttk.Scrollbar(self, orient="vertical")
        hsb = ttk.Scrollbar(self, orient="horizontal")

        # 创建表格
        self.tree = ttk.Treeview(self, show="headings",
                                 yscrollcommand=vsb.set,
                                 xscrollcommand=hsb.set,
                                 height=20)

        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        # 布局
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def plot(self, df):
        # 清空现有内容
        self.tree.delete(*self.tree.get_children())

        # 设置列
        self.tree["columns"] = list(df.columns)
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center")

        # 添加数据
        for _, row in df.iterrows():
            values = [f"{row[col]:.3f}" if isinstance(row[col], (int, float)) else str(row[col])[:30]
                      for col in df.columns]
            self.tree.insert("", tk.END, values=values)


class TopomapCanvas(BaseCanvas):
    """地形图画布"""

    def __init__(self, parent, width=10, height=7, dpi=100):
        super().__init__(parent, width, height, dpi)
        try:
            import mne
            self.mne_available = True
        except ImportError:
            self.mne_available = False

    def plot(self, data, info_dict, selected_channels, selected_features,
             is_relative=False, is_show_sensor=False):
        if not self.mne_available:
            self.ax.text(0.5, 0.5, "请安装 mne 以使用地形图功能",
                         ha='center', va='center', fontsize=14, color='red')
            self.draw()
            return

        import mne

        if not data:
            return

        num = len(selected_features)
        self.fig.clear()
        self.axes = self.fig.subplots(1, num, sharex=True, sharey=True)
        if num == 1:
            self.axes = [self.axes]

        select_data = self._get_selected_data(data, selected_channels, selected_features)
        chan_data = np.array([select_data[f] for f in selected_features])

        if is_relative:
            norm_data = (chan_data - chan_data.min(axis=1, keepdims=True)) / (
                        chan_data.max(axis=1, keepdims=True) - chan_data.min(axis=1, keepdims=True) + 1e-10)
            norm_data = norm_data * 2 - 1
        else:
            norm_data = (chan_data - chan_data.min()) / (chan_data.max() - chan_data.min() + 1e-10)
            norm_data = norm_data * 2 - 1

        if 'montage' in info_dict:
            montage = mne.channels.make_standard_montage(info_dict['montage'])
            info = mne.create_info(ch_names=selected_channels,
                                   sfreq=info_dict.get('srate', 1000),
                                   ch_types='eeg')
            evoked = mne.EvokedArray(data=chan_data.T, info=info)
            evoked.set_montage(montage)

            for i, (ax, psd) in enumerate(zip(self.axes, norm_data)):
                ax.clear()
                ax.set_title(selected_features[i], fontsize=11, pad=10)
                im = mne.viz.plot_topomap(
                    psd, evoked.info, axes=ax, show=False,
                    sensors=is_show_sensor, vlim=(-1, 1),
                    names=selected_channels if is_show_sensor else None,
                    cmap='RdBu_r'
                )
            self.fig.tight_layout()
            self.draw()

    def _get_selected_data(self, feature_dict, channels, features):
        idx = [i for i, ch in enumerate(feature_dict['ch_names']) if ch in channels]
        out = {}
        for f in features:
            if f in feature_dict['feature']:
                out[f] = [feature_dict['feature'][f][i] for i in idx]
        return out


# ---------- 主窗口 ----------
class FeatureView(tk.Toplevel):
    """特征可视化主窗口 - 最佳布局"""

    def __init__(self, parent, data, channels, features):
        super().__init__(parent)
        self.parent = parent
        self.data = data
        self.all_channels = channels
        self.all_features = features
        self.settings = {"width": 10, "height": 7, "title": "", "xlabel": "", "ylabel": ""}
        self.current_canvas = None
        self.info = None

        self.title("特征可视化")
        self.geometry("1400x850")
        self.minsize(800, 600)  # 设置最小尺寸

        # ===== 允许最小化和全屏化 =====
        self.resizable(True, True)  # 允许调整大小
        self.state('normal')  # 正常状态（不是最大化也不是最小化）

        # 添加窗口状态绑定
        self.bind("<F11>", lambda e: self.attributes('-fullscreen', not self.attributes('-fullscreen')))

        self._create_menu()
        self._create_layout()

       # self.transient(parent)
       # self.grab_set()

    def _create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="保存图像", command=self.save_plot, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.destroy, accelerator="Ctrl+Q")

        # 设置菜单
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="设置", menu=settings_menu)
        settings_menu.add_command(label="绘图设置", command=self.show_settings, accelerator="Ctrl+P")

        # 视图菜单
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图", menu=view_menu)
        view_menu.add_command(label="全选通道", command=self.select_all_channels)
        view_menu.add_command(label="全选特征", command=self.select_all_features)
        view_menu.add_separator()
        view_menu.add_command(label="最小化", command=self.iconify)  # 最小化
        view_menu.add_command(label="最大化", command=self.maximize)  # 最大化
        view_menu.add_command(label="全屏", command=self.toggle_fullscreen, accelerator="F11")  # 全屏
        view_menu.add_separator()
        view_menu.add_command(label="重置视图", command=self.reset_view)

        # 绑定快捷键
        self.bind_all("<Control-s>", lambda e: self.save_plot())
        self.bind_all("<Control-p>", lambda e: self.show_settings())
        self.bind_all("<Control-q>", lambda e: self.destroy())

    def _create_layout(self):
        """创建主布局 - 无空白紧凑布局"""
        # 主容器 - 使用PanedWindow支持拖动分割，去掉所有边距
        main_panel = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # ===== 左侧控制面板 =====
        left_frame = ttk.Frame(main_panel, width=300, relief=tk.FLAT)
        main_panel.add(left_frame, weight=1)

        self._create_left_panel(left_frame)

        # ===== 右侧图形面板 =====
        right_frame = ttk.Frame(main_panel, relief=tk.FLAT)
        main_panel.add(right_frame, weight=4)

        self._create_right_panel(right_frame)

    def _create_left_panel(self, parent):
        """创建左侧控制面板 - 优化布局填满整个面板"""

        # 使用网格布局让各部分按比例分配高度
        parent.grid_rowconfigure(0, weight=0)  # 标题
        parent.grid_rowconfigure(1, weight=2)  # 通道选择
        parent.grid_rowconfigure(2, weight=3)  # 特征选择
        parent.grid_rowconfigure(3, weight=0)  # 绘图类型
        parent.grid_rowconfigure(4, weight=0)  # 地形图选项
        parent.grid_rowconfigure(5, weight=0)  # 操作按钮
        parent.grid_columnconfigure(0, weight=1)

        # ===== 标题 =====
        title_label = tk.Label(parent, text="控制面板",
                               font=('微软雅黑', 14, 'bold'),
                               fg='#2c3e50', bg='#f5f5f5')
        title_label.grid(row=0, column=0, sticky="ew", pady=(5, 2))

        # ===== 通道选择区域 =====
        ch_frame = ttk.LabelFrame(parent, text="通道选择", padding=2)
        ch_frame.grid(row=1, column=0, sticky="nsew", padx=1, pady=1)

        ch_frame.grid_rowconfigure(0, weight=1)
        ch_frame.grid_columnconfigure(0, weight=1)

        # 通道列表
        self.ch_listbox = tk.Listbox(ch_frame, selectmode=tk.MULTIPLE,
                                     exportselection=False, bg='white',
                                     font=('微软雅黑', 9),
                                     relief=tk.FLAT, bd=1,
                                     highlightthickness=0)
        ch_scroll = ttk.Scrollbar(ch_frame, orient=tk.VERTICAL,
                                  command=self.ch_listbox.yview)
        self.ch_listbox.configure(yscrollcommand=ch_scroll.set)

        for ch in self.all_channels:
            self.ch_listbox.insert(tk.END, ch)
        self.ch_listbox.selection_set(0, tk.END)

        self.ch_listbox.grid(row=0, column=0, sticky="nsew", padx=(0, 1))
        ch_scroll.grid(row=0, column=1, sticky="ns")

        # 通道操作按钮
        ch_btn_frame = ttk.Frame(ch_frame)
        ch_btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=1)

        ttk.Button(ch_btn_frame, text="全选", width=4,
                   command=self.select_all_channels).pack(side=tk.LEFT, padx=1)
        ttk.Button(ch_btn_frame, text="清空", width=4,
                   command=self.clear_all_channels).pack(side=tk.LEFT, padx=1)
        ttk.Label(ch_btn_frame, text=f"共{len(self.all_channels)}个",
                  font=('微软雅黑', 8)).pack(side=tk.RIGHT, padx=2)

        # ===== 特征选择区域 =====
        feat_frame = ttk.LabelFrame(parent, text="特征选择", padding=2)
        feat_frame.grid(row=2, column=0, sticky="nsew", padx=1, pady=1)

        feat_frame.grid_rowconfigure(0, weight=1)
        feat_frame.grid_columnconfigure(0, weight=1)

        # 特征列表
        self.feat_listbox = tk.Listbox(feat_frame, selectmode=tk.MULTIPLE,
                                       exportselection=False, bg='white',
                                       font=('微软雅黑', 8),
                                       relief=tk.FLAT, bd=1,
                                       highlightthickness=0)
        feat_scroll = ttk.Scrollbar(feat_frame, orient=tk.VERTICAL,
                                    command=self.feat_listbox.yview)
        self.feat_listbox.configure(yscrollcommand=feat_scroll.set)

        for feat in self.all_features:
            self.feat_listbox.insert(tk.END, feat)
        self.feat_listbox.selection_set(0, tk.END)

        self.feat_listbox.grid(row=0, column=0, sticky="nsew", padx=(0, 1))
        feat_scroll.grid(row=0, column=1, sticky="ns")

        # 特征操作按钮
        feat_btn_frame = ttk.Frame(feat_frame)
        feat_btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=1)

        ttk.Button(feat_btn_frame, text="全选", width=4,
                   command=self.select_all_features).pack(side=tk.LEFT, padx=1)
        ttk.Button(feat_btn_frame, text="清空", width=4,
                   command=self.clear_all_features).pack(side=tk.LEFT, padx=1)
        ttk.Label(feat_btn_frame, text=f"共{len(self.all_features)}个",
                  font=('微软雅黑', 8)).pack(side=tk.RIGHT, padx=2)

        # ===== 绘图类型区域 =====
        type_frame = ttk.LabelFrame(parent, text="绘图类型", padding=2)
        type_frame.grid(row=3, column=0, sticky="ew", padx=1, pady=1)

        self.plot_type = tk.StringVar(value="曲线图")

        # 两行两列布局
        type_grid = ttk.Frame(type_frame)
        type_grid.pack(fill=tk.X, expand=True)

        # 第一行
        ttk.Radiobutton(type_grid, text="📈 曲线图", variable=self.plot_type,
                        value="曲线图").grid(row=0, column=0, sticky=tk.W, padx=2, pady=1)
        ttk.Radiobutton(type_grid, text="📊 柱状图", variable=self.plot_type,
                        value="柱状图").grid(row=0, column=1, sticky=tk.W, padx=2, pady=1)

        # 第二行
        ttk.Radiobutton(type_grid, text="📋 表格", variable=self.plot_type,
                        value="表格").grid(row=1, column=0, sticky=tk.W, padx=2, pady=1)
        ttk.Radiobutton(type_grid, text="🗺️ 地形图", variable=self.plot_type,
                        value="地形图").grid(row=1, column=1, sticky=tk.W, padx=2, pady=1)

        # 配置网格列权重
        type_grid.grid_columnconfigure(0, weight=1)
        type_grid.grid_columnconfigure(1, weight=1)

        # ===== 地形图选项区域 =====
        self.topo_frame = ttk.LabelFrame(parent, text="地形图设置", padding=2)
        self.topo_frame.grid(row=4, column=0, sticky="ew", padx=1, pady=1)

        # 信息文件选择
        info_row = ttk.Frame(self.topo_frame)
        info_row.pack(fill=tk.X, pady=1)

        ttk.Button(info_row, text="选择文件", width=6,
                   command=self.select_info).pack(side=tk.LEFT, padx=1)
        self.info_label = ttk.Label(info_row, text="未选择",
                                    font=('微软雅黑', 8), foreground='#666')
        self.info_label.pack(side=tk.LEFT, padx=2)

        # 地形图选项 - 并排显示
        opt_row = ttk.Frame(self.topo_frame)
        opt_row.pack(fill=tk.X, pady=1)

        self.cb_relative = tk.BooleanVar(value=False)
        self.cb_sensor = tk.BooleanVar(value=False)

        ttk.Checkbutton(opt_row, text="相对缩放",
                        variable=self.cb_relative).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Checkbutton(opt_row, text="显示传感器",
                        variable=self.cb_sensor).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # ===== 操作按钮区域 =====
        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=5, column=0, sticky="ew", padx=1, pady=2)

        # 生成图形按钮 - 最突出
        generate_btn = tk.Button(btn_frame, text="生成图形",
                                 bg="#27ae60", fg="white",
                                 font=('微软雅黑', 11, 'bold'),
                                 command=self.plot,
                                 relief=tk.FLAT)
        generate_btn.pack(fill=tk.X, pady=1)

        # 保存图像按钮
        save_btn = tk.Button(btn_frame, text="保存图像",
                             bg="#f39c12", fg="white",
                             font=('微软雅黑', 10, 'bold'),
                             command=self.save_plot,
                             relief=tk.FLAT)
        save_btn.pack(fill=tk.X, pady=1)

        # 设置按钮
        setting_btn = tk.Button(btn_frame, text="⚙️ 绘图设置",
                                bg="#3498db", fg="white",
                                font=('微软雅黑', 9),
                                command=self.show_settings,
                                relief=tk.FLAT)
        setting_btn.pack(fill=tk.X, pady=1)

    def _create_right_panel(self, parent):
        """创建右侧图形面板 - 保留坐标轴标题"""

        # 在创建任何图形前强制设置中文字体
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        # 方法1：通用字体设置
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # 方法2：直接指定字体路径（Windows系统）
        try:
            import os
            font_paths = [
                'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
                'C:/Windows/Fonts/simhei.ttf',  # 黑体
                'C:/Windows/Fonts/msyhbd.ttc',  # 微软雅黑粗体
            ]
            for font_path in font_paths:
                if os.path.exists(font_path):
                    mpl.font_manager.fontManager.addfont(font_path)
                    plt.rcParams['font.family'] = 'sans-serif'
                    plt.rcParams['font.sans-serif'] = [mpl.font_manager.FontProperties(fname=font_path).get_name()]
                    break
        except:
            pass

        # 创建画布容器 - 无边距
        self.canvas_container = ttk.Frame(parent, relief=tk.FLAT)
        self.canvas_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # 创建初始图形
        self.fig = Figure(figsize=(10, 7), dpi=100, facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.fig, self.canvas_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # 创建轴
        self.ax = self.fig.add_subplot(111)

        # 设置白色背景
        self.ax.set_facecolor('white')
        self.fig.patch.set_facecolor('white')

        # 保留坐标轴框架，但隐藏刻度
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(True)
        self.ax.spines['left'].set_visible(True)

        # 隐藏刻度标签但保留轴线
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # ===== 添加坐标轴标题和主标题 =====
        # 从设置中获取标题，如果没有则使用默认值
        title = self.settings.get("title", "")
        xlabel = self.settings.get("xlabel", "")
        ylabel = self.settings.get("ylabel", "")

        # 如果设置为空，显示默认提示
        if not title:
            title = "特征可视化"
        if not xlabel:
            xlabel = "特征"
        if not ylabel:
            ylabel = "幅值"

        # 设置标题和轴标签
        self.ax.set_title(title, fontsize=14, pad=15, fontfamily='Microsoft YaHei')
        self.ax.set_xlabel(xlabel, fontsize=11, fontfamily='Microsoft YaHei')
        self.ax.set_ylabel(ylabel, fontsize=11, fontfamily='Microsoft YaHei')

        # 在图形中心添加一个淡灰色的提示文字
        self.ax.text(0.5, 0.5, "请选择通道和特征后点击「生成图形」",
                     ha='center', va='center', fontsize=12,
                     transform=self.ax.transAxes,
                     color='#999999',
                     fontfamily='Microsoft YaHei',
                     bbox=dict(boxstyle="round,pad=0.5",
                               facecolor="#f8f8f8", alpha=0.8,
                               edgecolor='#dddddd'))

        self.canvas.draw()

        # 保存当前画布
        self.current_canvas = type('obj', (object,), {
            'fig': self.fig,
            'canvas': self.canvas,
            'ax': self.ax,
            'get_widget': lambda: self.canvas.get_tk_widget()
        })

    # ========== 辅助方法 ==========
    def select_all_channels(self):
        """全选通道"""
        self.ch_listbox.selection_set(0, tk.END)

    def clear_all_channels(self):
        """清空通道选择"""
        self.ch_listbox.selection_clear(0, tk.END)

    def select_all_features(self):
        """全选特征"""
        self.feat_listbox.selection_set(0, tk.END)

    def clear_all_features(self):
        """清空特征选择"""
        self.feat_listbox.selection_clear(0, tk.END)

    def reset_view(self):
        """重置视图"""
        self.select_all_channels()
        self.select_all_features()
        self.plot()

    def maximize(self):
        """最大化窗口"""
        self.state('zoomed')  # Windows
        # self.attributes('-zoomed', True)  # Linux 可能用这个

    def toggle_fullscreen(self, event=None):
        """切换全屏模式"""
        self.attributes('-fullscreen', not self.attributes('-fullscreen'))

    def select_info(self):
        """选择信息文件"""
        path = filedialog.askopenfilename(
            title="选择信息文件",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        if path:
            self.info = read_info(path)
            self.info_label.config(text=os.path.basename(path))

    def show_settings(self):
        """显示设置对话框"""
        dialog = PlotSettingsDialog(self, self.settings)
        self.wait_window(dialog)
        if dialog.result:
            self.settings.update(dialog.result)
            # 如果已经有图形，重新绘制
            if hasattr(self.current_canvas, 'plot'):
                self.plot()

    def get_selected_channels(self):
        """获取选中的通道"""
        indices = self.ch_listbox.curselection()
        return [self.all_channels[i] for i in indices]

    def get_selected_features(self):
        """获取选中的特征"""
        indices = self.feat_listbox.curselection()
        return [self.all_features[i] for i in indices]

    def plot(self):
        """生成图形"""
        selected_channels = self.get_selected_channels()
        selected_features = self.get_selected_features()

        if not selected_channels or not selected_features:
            messagebox.showwarning("警告", "请至少选择一个通道和一个特征")
            return

        plot_type = self.plot_type.get()

        # 清除旧画布
        for widget in self.canvas_container.winfo_children():
            widget.destroy()

        # 重新设置中文字体（确保）
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        try:
            if plot_type == "曲线图":
                canvas = CurveCanvas(self.canvas_container,
                                     width=self.settings["width"],
                                     height=self.settings["height"])
                canvas.plot(self.data, selected_channels,
                            selected_features, self.settings)

            elif plot_type == "柱状图":
                canvas = BarCanvas(self.canvas_container,
                                   width=self.settings["width"],
                                   height=self.settings["height"])
                canvas.plot(self.data, selected_channels,
                            selected_features, self.settings)

            elif plot_type == "表格":
                canvas = TableCanvas(self.canvas_container)
                df = self._feature_dict_to_df(self.data)
                canvas.plot(df)
                canvas.pack(fill=tk.BOTH, expand=True)
                self.current_canvas = canvas
                return

            elif plot_type == "地形图":
                if not self.info:
                    messagebox.showwarning("警告", "请先选择信息文件")
                    return
                canvas = TopomapCanvas(self.canvas_container,
                                       width=self.settings["width"],
                                       height=self.settings["height"])
                canvas.plot(self.data, self.info, selected_channels,
                            selected_features,
                            self.cb_relative.get(), self.cb_sensor.get())

            else:
                return

            canvas.get_widget().pack(fill=tk.BOTH, expand=True)

            # 添加工具栏
            toolbar_frame = ttk.Frame(self.canvas_container)
            toolbar_frame.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas.canvas, toolbar_frame)
            toolbar.update()

            self.current_canvas = canvas

        except Exception as e:
            messagebox.showerror("错误", f"绘图失败:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def save_plot(self):
        """保存图像"""
        if not hasattr(self.current_canvas, 'fig'):
            messagebox.showwarning("警告", "请先生成图形")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG图像", "*.png"), ("JPEG图像", "*.jpg"),
                       ("PDF文件", "*.pdf"), ("SVG图像", "*.svg")]
        )
        if path:
            dpi = simpledialog.askinteger("DPI", "请输入DPI (推荐300):",
                                          initialvalue=300, minvalue=50, maxvalue=600)
            if dpi:
                self.current_canvas.fig.savefig(path, dpi=dpi, bbox_inches='tight')
                messagebox.showinfo("成功", f"图像已保存到:\n{path}")

    def _feature_dict_to_df(self, fd):
        """特征字典转DataFrame"""
        data = {"通道": fd["ch_names"]}
        for k, v in fd["feature"].items():
            data[k] = v
        df = pd.DataFrame(data)
        return df


# ==================== 外部调用接口 ====================
def show_feature_view(parent, data_dict):
    """
    显示特征视图

    Args:
        parent: 父窗口
        data_dict: 数据字典
    """
    try:
        # 解析数据
        processed = data_dict.get("processed", {})
        feature_data = processed.get("features", {})

        if not feature_data:
            feature_data = data_dict.get("feature", {})

        if not feature_data:
            messagebox.showerror("错误", "数据中没有特征信息")
            return None

        # 获取通道列表
        channels = []
        if 'signal' in data_dict and data_dict['signal']:
            first_mod = list(data_dict['signal'].keys())[0]
            channels = data_dict['signal'][first_mod].get('channel_names', [])

        if not channels:
            messagebox.showerror("错误", "没有通道信息")
            return None

        # 获取特征列表
        feature_names_set = set()
        for key in feature_data.keys():
            if '_' in key:
                parts = key.split('_', 1)
                if len(parts) == 2:
                    feature_names_set.add(parts[1])

        feature_names = sorted(list(feature_names_set))

        if not feature_names:
            messagebox.showerror("错误", "无法识别特征格式")
            return None

        # 转换数据格式
        converted_features = {}
        for feat_name in feature_names:
            values = []
            for ch in channels:
                key = f"{ch}_{feat_name}"
                values.append(feature_data.get(key, 0.0))
            converted_features[feat_name] = values

        converted_data = {
            'ch_names': channels,
            'feature': converted_features
        }

        # 创建视图
        view = FeatureView(parent, converted_data, channels, feature_names)
        return view

    except Exception as e:
        messagebox.showerror("错误", f"打开特征视图失败:\n{str(e)}")
        import traceback
        traceback.print_exc()
        return None
