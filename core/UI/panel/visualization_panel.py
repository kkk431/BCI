#!/usr/bin/env python3
"""
visualization_panel.py
可视化集成面板 - 用于嵌入主界面的Visualization标签页
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import numpy as np
from datetime import datetime

# 导入功能模块
from core.visualizer.signal_view import SignalView, fNIRSView
from core.visualizer.stats_view import StatsView
from core.visualizer.bar_view import BarView
from core.visualizer.time_frequency_view import TimeFrequencyView
from core.visualizer.topography_view import TopographyView
from core.visualizer.feature_view import FeatureView

# 导入数据IO
try:
    from core.io.data_io import DataLoader
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from core.io.data_io import DataLoader


class ScrollableFrame(ttk.Frame):
    """可滚动的框架"""

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        # 创建画布
        self.canvas = tk.Canvas(self, highlightthickness=0)

        # 创建滚动条
        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)

        # 配置画布
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        # 创建可滚动的框架
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>",
                                   lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # 将框架添加到画布
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # 布局
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")

        # 配置网格权重
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # 绑定画布大小变化
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # 绑定鼠标滚轮
        self.bind_mousewheel()

    def on_canvas_configure(self, event):
        """画布大小变化时调整内部框架宽度"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def bind_mousewheel(self):
        """绑定鼠标滚轮事件"""
        def on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def on_shift_mousewheel(event):
            self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas.bind_all("<MouseWheel>", on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", on_shift_mousewheel)

    def destroy(self):
        """销毁时解绑事件"""
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Shift-MouseWheel>")
        super().destroy()


class ModernCard(tk.Frame):
    """现代风格卡片组件"""

    def __init__(self, parent, title, description, icon, color, command, **kwargs):
        super().__init__(parent, bg='white', **kwargs)

        self.command = command
        self.color = color

        # 配置卡片样式
        self.config(relief=tk.FLAT, highlightthickness=1,
                    highlightcolor='#e0e0e0', highlightbackground='#e0e0e0')

        # 图标和标题区域
        header = tk.Frame(self, bg='white')
        header.pack(fill=tk.X, padx=15, pady=(15, 5))

        # 图标
        icon_label = tk.Label(header, text=icon, font=('Segoe UI', 24),
                              bg='white', fg=color)
        icon_label.pack(side=tk.LEFT, padx=(0, 10))

        # 标题
        title_label = tk.Label(header, text=title, font=('微软雅黑', 14, 'bold'),
                               bg='white', fg='#333')
        title_label.pack(side=tk.LEFT)

        # 描述
        desc_label = tk.Label(self, text=description, font=('微软雅黑', 10),
                              bg='white', fg='#666', justify=tk.LEFT, wraplength=200)
        desc_label.pack(fill=tk.X, padx=15, pady=5)

        # 按钮
        btn_frame = tk.Frame(self, bg='white')
        btn_frame.pack(fill=tk.X, padx=15, pady=(10, 15))

        open_btn = tk.Button(btn_frame, text="打开", bg=color, fg='white',
                             font=('微软雅黑', 10, 'bold'), relief=tk.FLAT,
                             cursor='hand2', command=self.on_click)
        open_btn.pack(side=tk.RIGHT)

        # 悬停效果
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

        # 绑定所有子组件
        for child in [header, icon_label, title_label, desc_label, btn_frame, open_btn]:
            child.bind("<Enter>", self.on_enter)
            child.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        """鼠标进入"""
        self.config(highlightthickness=2, highlightcolor=self.color)
        self.config(relief=tk.RAISED)

    def on_leave(self, e):
        """鼠标离开"""
        self.config(highlightthickness=1, highlightcolor='#e0e0e0')
        self.config(relief=tk.FLAT)

    def on_click(self, e=None):
        """点击事件"""
        if self.command:
            self.command()


class ModernVisualizationPanel(ttk.Frame):
    """
    可视化集成面板 - 用于嵌入主界面的Visualization标签页
    包含6个功能模块卡片
    """

    # 配色方案
    COLORS = {
        'primary': '#2c3e50',
        'secondary': '#34495e',
        'accent': '#3498db',
        'success': '#27ae60',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'light': '#ecf0f1',
        'dark': '#2c3e50',
        'card1': '#3498db',   # 信号波形
        'card2': '#27ae60',   # 统计分析
        'card3': '#e74c3c',   # 柱状图
        'card4': '#9b59b6',   # 时频分析
        'card5': '#1abc9c',   # 地形图
        'card6': '#e67e22',   # 特征可视化
    }

    def __init__(self, parent, data_dict=None, file_path=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.data_dict = data_dict
        self.file_path = file_path
        self.data_loader = DataLoader()
        self.current_views = {}
        self.qt_app = None

        # 配置样式
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.style.configure('Modern.TLabel', font=('微软雅黑', 10))
        self.style.configure('Modern.TButton', font=('微软雅黑', 10), padding=5)
        self.style.configure('Modern.TLabelframe', font=('微软雅黑', 10, 'bold'))

        # 设置UI
        self.setup_ui()

        if file_path and not data_dict:
            self.load_file(file_path)

    def setup_ui(self):
        """设置用户界面"""
        # 主容器 - 使用网格布局
        self.grid_rowconfigure(0, weight=0)  # 数据面板
        self.grid_rowconfigure(1, weight=1)  # 功能卡片（可滚动）
        self.grid_columnconfigure(0, weight=1)

        # ========== 顶部数据面板 ==========
        self.create_data_panel().grid(row=0, column=0, sticky="ew", pady=10, padx=10)

        # ========== 可滚动的功能卡片区域 ==========
        self.scrollable_frame = ScrollableFrame(self)
        self.scrollable_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # 在可滚动框架中放置功能卡片
        self.create_function_grid(self.scrollable_frame.scrollable_frame)

    def create_data_panel(self):
        """创建数据面板"""
        panel = tk.Frame(self, bg='white', relief=tk.FLAT,
                         highlightthickness=1, highlightcolor='#e0e0e0')

        # 标题
        title_frame = tk.Frame(panel, bg='#ecf0f1', height=35)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title = tk.Label(title_frame, text="📁 数据管理", bg='#ecf0f1',
                         font=('微软雅黑', 11, 'bold'), fg='#2c3e50')
        title.pack(side=tk.LEFT, padx=15, pady=5)

        # 内容区域
        content = tk.Frame(panel, bg='white')
        content.pack(fill=tk.X, padx=15, pady=15)

        # 文件路径行
        path_frame = tk.Frame(content, bg='white')
        path_frame.pack(fill=tk.X, pady=5)

        tk.Label(path_frame, text="文件路径:", bg='white',
                 font=('微软雅黑', 10)).pack(side=tk.LEFT, padx=(0, 10))

        self.file_path_var = tk.StringVar()
        path_entry = tk.Entry(path_frame, textvariable=self.file_path_var,
                              font=('微软雅黑', 10), bg='#f9f9f9',
                              relief=tk.FLAT, highlightthickness=1,
                              highlightcolor='#ddd')
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        # 按钮行
        btn_frame = tk.Frame(content, bg='white')
        btn_frame.pack(fill=tk.X, pady=10)

        buttons = [
            ("📂 浏览文件", '#3498db', self.browse_file),
            ("📥 加载数据", '#27ae60', self.load_current_file),
            ("🎲 演示数据", '#f39c12', self.load_demo_data),
            ("🔄 重置", '#95a5a6', self.reset_data)
        ]

        for text, color, cmd in buttons:
            btn = tk.Button(btn_frame, text=text, bg=color, fg='white',
                            font=('微软雅黑', 9), relief=tk.FLAT,
                            cursor='hand2', command=cmd)
            btn.pack(side=tk.LEFT, padx=2, ipadx=10, ipady=3)

        # 数据信息卡片
        self.create_info_card(content)

        return panel

    def create_info_card(self, parent):
        """创建信息卡片"""
        info_container = tk.Frame(parent, bg='#f8f9fa', relief=tk.FLAT,
                                  highlightthickness=1, highlightcolor='#e0e0e0')
        info_container.pack(fill=tk.X, pady=10)

        tk.Label(info_container, text="📊 数据信息", bg='#f8f9fa',
                 font=('微软雅黑', 10, 'bold'), fg='#2c3e50').pack(anchor=tk.W, padx=10, pady=5)

        self.info_text = tk.Text(info_container, height=5, wrap=tk.WORD,
                                 font=('微软雅黑', 9), bg='#f8f9fa',
                                 relief=tk.FLAT, highlightthickness=0)
        self.info_text.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.info_text.config(state=tk.DISABLED)

    def create_function_grid(self, parent):
        """创建功能卡片网格 - 3列显示6个卡片"""
        # 配置网格列 - 3列
        for i in range(3):
            parent.columnconfigure(i, weight=1, uniform='col')

        # 6个功能卡片定义
        cards = [
            {
                'title': '信号波形',
                'desc': '多通道信号显示 · 实时滤波 · 翻页导航 · 事件标记 · fNIRS专用',
                'icon': '📈',
                'color': self.COLORS['card1'],
                'command': self.open_signal_view
            },
            {
                'title': '统计分析',
                'desc': '箱线图 · ROC曲线 · 混淆矩阵 · 3D传感器 · 显著性标记',
                'icon': '📊',
                'color': self.COLORS['card2'],
                'command': self.open_stats_view
            },
            {
                'title': '柱状图',
                'desc': 'Excel数据可视化 · 特征对比 · 数值标签 · 多子图',
                'icon': '📋',
                'color': self.COLORS['card3'],
                'command': self.open_bar_view
            },
            {
                'title': '时频分析',
                'desc': 'STFT时频图 · 通道选择 · 动态色标 · 功率谱',
                'icon': '⏱️',
                'color': self.COLORS['card4'],
                'command': self.open_time_frequency
            },
            {
                'title': '地形图',
                'desc': '多频带拓扑图 · 相对/绝对功率 · 坏通道排除 · 数值表格',
                'icon': '🗺️',
                'color': self.COLORS['card5'],
                'command': self.open_topography
            },
            {
                'title': '特征可视化',
                'desc': '曲线图 · 柱状图 · 表格 · 特征拓扑图 · 通道选择',
                'icon': '📐',
                'color': self.COLORS['card6'],
                'command': self.open_feature_view
            }
        ]

        # 创建卡片
        for i, card in enumerate(cards):
            row = i // 3
            col = i % 3

            card_widget = ModernCard(
                parent,
                title=card['title'],
                description=card['desc'],
                icon=card['icon'],
                color=card['color'],
                command=card['command']
            )
            card_widget.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')

        # 配置行权重
        for i in range(2):
            parent.rowconfigure(i, weight=1)

    # ========== 文件操作方法 ==========
    def browse_file(self):
        """浏览文件"""
        file_path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[
                ("所有支持文件", "*.edf *.bdf *.gdf *.mat *.csv *.xlsx *.xls *.json *.npy *.set *.vhdr"),
                ("EDF文件", "*.edf"),
                ("BDF文件", "*.bdf"),
                ("GDF文件", "*.gdf"),
                ("MAT文件", "*.mat"),
                ("NumPy文件", "*.npy"),
                ("JSON文件", "*.json"),
                ("CSV文件", "*.csv"),
                ("Excel文件", "*.xlsx *.xls"),
                ("EEGLAB文件", "*.set *.vhdr")
            ]
        )

        if file_path:
            self.file_path_var.set(file_path)

    def load_current_file(self):
        """加载当前文件"""
        file_path = self.file_path_var.get().strip()
        if not file_path:
            messagebox.showwarning("警告", "请先选择文件")
            return

        self.load_file(file_path)

    def load_file(self, file_path):
        """加载文件"""
        try:
            import numpy as np

            if file_path.endswith('.npy'):
                self.data_dict = np.load(file_path, allow_pickle=True).item()
            elif file_path.endswith('.json'):
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.data_dict = json.load(f)
            else:
                self.data_dict = self.data_loader.load(file_path)

            self.file_path = file_path
            self.update_file_info()
            messagebox.showinfo("成功", "文件加载成功！")

        except Exception as e:
            messagebox.showerror("错误", f"文件加载失败:\n{str(e)}")

    def load_demo_data(self):
        """加载演示数据"""
        fs = 1000
        t = np.arange(0, 30, 1 / fs)

        self.data_dict = {
            "meta": {
                "subject_id": "demo",
                "session_id": "session1",
                "task": "rest",
                "modality": ["EEG", "fNIRS"]
            },
            "signal": {
                "EEG": {
                    "data": np.array([np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t))
                                      for _ in range(8)]),
                    "sampling_rate": fs,
                    "channel_names": [f"EEG_{i}" for i in range(8)],
                    "unit": "uV"
                },
                "fNIRS": {
                    "data": np.array([np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(len(t))
                                      for _ in range(4)]),
                    "sampling_rate": fs,
                    "channel_names": [f"NIRS_{i}" for i in range(4)],
                    "unit": "uM"
                }
            },
            "feature": {
                "type": "eeg_psd",
                "ch_names": [f"EEG_{i}" for i in range(8)],
                "feature": {
                    "Delta": np.random.rand(8),
                    "Theta": np.random.rand(8),
                    "Alpha": np.random.rand(8),
                    "Beta": np.random.rand(8),
                    "Gamma": np.random.rand(8)
                }
            }
        }

        self.file_path = None
        self.update_file_info()
        messagebox.showinfo("成功", "演示数据加载成功！")

    def reset_data(self):
        """重置数据"""
        self.data_dict = None
        self.file_path = None
        self.file_path_var.set("")
        self.update_file_info()

    def update_file_info(self):
        """更新文件信息"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)

        if not self.data_dict:
            self.info_text.insert(tk.END, "请加载数据文件或使用演示数据")
        else:
            meta = self.data_dict.get("meta", {})
            subject = meta.get("subject_id", "unknown")
            task = meta.get("task", "unknown")
            modalities = meta.get("modality", [])

            info = f"🧑 被试: {subject}\n"
            info += f"📋 任务: {task}\n"
            info += f"🔬 模态: {', '.join(modalities)}\n"

            signal_dict = self.data_dict.get("signal", {})
            for mod, sig in signal_dict.items():
                if isinstance(sig, dict) and 'data' in sig:
                    data = sig['data']
                    n_ch = data.shape[0] if hasattr(data, 'shape') else 1
                    fs = sig.get('sampling_rate', '?')
                    info += f"📊 {mod}: {n_ch}通道, {fs}Hz\n"

            self.info_text.insert(tk.END, info)

        self.info_text.config(state=tk.DISABLED)

    # ========== 功能模块打开方法 ==========
    def open_signal_view(self):
        """打开信号视图"""
        if not self.data_dict:
            messagebox.showwarning("警告", "请先加载数据")
            return

        try:
            window = tk.Toplevel(self)
            window.title("信号波形视图")
            window.geometry("1200x800")

            meta = self.data_dict.get("meta", {})
            modalities = meta.get("modality", [])

            if "fNIRS" in modalities or "NIRS" in modalities:
                view = fNIRSView(window, self.data_dict)
            else:
                view = SignalView(window, self.data_dict)

            view.pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("错误", f"打开信号视图失败:\n{str(e)}")

    def open_stats_view(self):
        """打开统计视图"""
        if not self.data_dict:
            messagebox.showwarning("警告", "请先加载数据")
            return

        try:
            window = tk.Toplevel(self)
            window.title("统计分析视图")
            window.geometry("1100x800")

            view = StatsView(window, self.data_dict)
            view.pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("错误", f"打开统计视图失败:\n{str(e)}")

    def open_bar_view(self):
        """打开柱状图视图"""
        if not self.data_dict:
            messagebox.showwarning("警告", "请先加载数据")
            return

        try:
            window = tk.Toplevel(self)
            window.title("柱状图视图")
            window.geometry("1200x800")

            view = BarView(window, self.data_dict)
            view.pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("错误", f"打开柱状图视图失败:\n{str(e)}")

    def open_time_frequency(self):
        """打开时频分析视图"""
        if not self.data_dict:
            messagebox.showwarning("警告", "请先加载数据")
            return

        try:
            from core.visualizer.time_frequency_view import short_time_Fourier_transform

            signal_dict = self.data_dict.get("signal", {})
            if not signal_dict:
                messagebox.showwarning("警告", "数据中没有信号")
                return

            modality = list(signal_dict.keys())[0]
            signal_info = signal_dict[modality]

            raw_data = signal_info.get('data')
            fs = signal_info.get('sampling_rate', 1000)
            ch_names = signal_info.get('channel_names', [])

            stft_input = {
                'data': raw_data,
                'srate': fs,
                'ch_names': ch_names
            }

            stft_result = short_time_Fourier_transform(stft_input)

            self.ensure_pyqt_app()
            self.tf_window = TimeFrequencyView(stft_result)
            self.tf_window.show()

        except Exception as e:
            messagebox.showerror("错误", f"打开时频分析视图失败:\n{str(e)}")

    def open_topography(self):
        """打开地形图视图"""
        if not self.data_dict:
            messagebox.showwarning("警告", "请先加载数据")
            return

        try:
            feature_data = self.data_dict.get("feature")
            if not feature_data:
                self.prepare_topography_data()
                feature_data = self.data_dict.get("feature")

            if not feature_data:
                messagebox.showwarning("警告", "无法准备地形图数据")
                return

            self.ensure_pyqt_app()
            self.topo_window = TopographyView(feature_data)
            self.topo_window.show()

        except Exception as e:
            messagebox.showerror("错误", f"打开地形图视图失败:\n{str(e)}")

    def prepare_topography_data(self):
        """准备地形图数据"""
        standard_channels = ['Fz', 'Cz', 'Pz', 'Oz', 'F3', 'F4', 'C3', 'C4',
                             'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8']

        self.data_dict["feature"] = {
            'type': 'eeg_psd',
            'ch_names': standard_channels[:10],
            'feature': {
                'Delta': np.random.rand(10),
                'Theta': np.random.rand(10),
                'Alpha': np.random.rand(10),
                'Beta': np.random.rand(10),
                'Gamma': np.random.rand(10)
            }
        }

    def open_feature_view(self):
        """打开特征可视化视图"""
        if not self.data_dict:
            messagebox.showwarning("警告", "请先加载数据")
            return

        try:
            feature_data = self.data_dict.get("feature")
            if not feature_data:
                messagebox.showwarning("警告", "数据中没有特征信息")
                return

            # 从feature_data中提取channels和features
            channels = feature_data.get('ch_names', [])
            features = list(feature_data.get('feature', {}).keys())

            if not channels:
                messagebox.showwarning("警告", "没有通道信息")
                return

            if not features:
                messagebox.showwarning("警告", "没有特征信息")
                return

            self.ensure_pyqt_app()
            from core.visualizer.feature_view import FeatureView

            # 传入所有必需的参数
            self.feature_window = FeatureView(feature_data, channels, features)
            self.feature_window.show()

        except Exception as e:
            messagebox.showerror("错误", f"打开特征视图失败:\n{str(e)}")

    def ensure_pyqt_app(self):
        """确保PyQt应用存在"""
        try:
            from PyQt5.QtWidgets import QApplication
            if not QApplication.instance():
                self.qt_app = QApplication(sys.argv)
        except ImportError:
            pass


# 测试代码
if __name__ == "__main__":
    root = tk.Tk()
    root.title("可视化面板测试")
    root.geometry("1000x800")

    panel = ModernVisualizationPanel(root)
    panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    root.mainloop()