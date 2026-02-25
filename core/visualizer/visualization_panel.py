#!/usr/bin/env python3
"""
visualization_panel.py
高级版可视化集成面板 - 修复滚动问题
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import numpy as np
from datetime import datetime

# 导入可视化模块
from .signal_view import SignalView
from .stats_view import StatsView
from .bar_view import BarView
from .plot_dialog import quick_plot

# 导入数据IO
try:
    from core.io.data_io import DataLoader
except ImportError:
    import sys
    import os

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
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

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
        # 设置内部框架宽度与画布相同
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def bind_mousewheel(self):
        """绑定鼠标滚轮事件"""

        def on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def on_shift_mousewheel(event):
            self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

        # 绑定到自身和所有子组件
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
        self.config(relief=tk.FLAT, highlightthickness=1, highlightcolor='#e0e0e0', highlightbackground='#e0e0e0')

        # 图标和标题区域
        header = tk.Frame(self, bg='white')
        header.pack(fill=tk.X, padx=15, pady=(15, 5))

        # 图标
        icon_label = tk.Label(header, text=icon, font=('Segoe UI', 24), bg='white', fg=color)
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
    高级版可视化集成面板 - 支持滚动
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
        'card1': '#3498db',  # 信号波形
        'card2': '#27ae60',  # 统计分析
        'card3': '#e74c3c',  # 柱状图
        'card4': '#f39c12',  # 快速预览
    }

    def __init__(self, parent, data_dict=None, file_path=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.data_dict = data_dict
        self.file_path = file_path
        self.data_loader = DataLoader()
        self.current_views = {}

        # 配置样式
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # 自定义样式
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
        self.grid_rowconfigure(0, weight=0)  # 导航栏
        self.grid_rowconfigure(1, weight=1)  # 内容区域（可滚动）
        self.grid_rowconfigure(2, weight=0)  # 状态栏
        self.grid_columnconfigure(0, weight=1)

        # ========== 顶部导航栏 ==========
        self.create_navbar()

        # ========== 可滚动的内容区域 ==========
        self.scrollable_frame = ScrollableFrame(self)
        self.scrollable_frame.grid(row=1, column=0, sticky="nsew")

        # 在可滚动框架中放置内容
        self.create_content(self.scrollable_frame.scrollable_frame)

        # ========== 状态栏 ==========
        self.create_status_bar()

    def create_navbar(self):
        """创建导航栏"""
        navbar = tk.Frame(self, bg=self.COLORS['primary'], height=50)
        navbar.grid(row=0, column=0, sticky="ew")
        navbar.grid_propagate(False)

        # Logo
        logo = tk.Label(navbar, text="🧠 智融脑机", bg=self.COLORS['primary'],
                        fg='white', font=('微软雅黑', 14, 'bold'))
        logo.pack(side=tk.LEFT, padx=20, pady=10)

        # 时间显示
        self.time_var = tk.StringVar()
        self.update_time()
        time_label = tk.Label(navbar, textvariable=self.time_var,
                              bg=self.COLORS['primary'], fg='#bdc3c7',
                              font=('微软雅黑', 10))
        time_label.pack(side=tk.RIGHT, padx=20, pady=10)

    def update_time(self):
        """更新时间"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.time_var.set(f"📅 {now}")
        self.after(1000, self.update_time)

    def create_content(self, parent):
        """创建内容区域"""
        # 使用网格布局
        parent.grid_columnconfigure(0, weight=1)

        # ========== 欢迎区域 ==========
        welcome = tk.Frame(parent, bg='#f5f6fa')
        welcome.grid(row=0, column=0, sticky="ew", pady=(0, 20))

        title = tk.Label(welcome, text="数据可视化平台",
                         font=('微软雅黑', 24, 'bold'), bg='#f5f6fa', fg=self.COLORS['dark'])
        title.pack(anchor=tk.W)

        subtitle = tk.Label(welcome, text="多模态脑机接口数据可视化与分析",
                            font=('微软雅黑', 11), bg='#f5f6fa', fg='#7f8c8d')
        subtitle.pack(anchor=tk.W)

        # ========== 数据面板 ==========
        self.create_data_panel(parent).grid(row=1, column=0, sticky="ew", pady=(0, 20))

        # ========== 功能卡片网格 ==========
        self.create_function_grid(parent).grid(row=2, column=0, sticky="nsew", pady=(0, 20))

        # 配置最后一行权重
        parent.grid_rowconfigure(2, weight=1)

    def create_data_panel(self, parent):
        """创建数据面板"""
        # 数据面板容器
        panel = tk.Frame(parent, bg='white', relief=tk.FLAT,
                         highlightthickness=1, highlightcolor='#e0e0e0')

        # 标题
        title_frame = tk.Frame(panel, bg=self.COLORS['light'], height=35)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title = tk.Label(title_frame, text="📁 数据管理", bg=self.COLORS['light'],
                         font=('微软雅黑', 11, 'bold'), fg=self.COLORS['dark'])
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
            ("📂 浏览文件", self.COLORS['accent'], self.browse_file),
            ("📥 加载数据", self.COLORS['success'], self.load_current_file),
            ("🎲 演示数据", self.COLORS['warning'], self.load_demo_data),
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
        # 信息容器
        info_container = tk.Frame(parent, bg='#f8f9fa', relief=tk.FLAT,
                                  highlightthickness=1, highlightcolor='#e0e0e0')
        info_container.pack(fill=tk.X, pady=10)

        # 标题
        tk.Label(info_container, text="📊 数据信息", bg='#f8f9fa',
                 font=('微软雅黑', 10, 'bold'), fg=self.COLORS['dark']).pack(anchor=tk.W, padx=10, pady=5)

        # 信息内容
        self.info_text = tk.Text(info_container, height=5, wrap=tk.WORD,
                                 font=('微软雅黑', 9), bg='#f8f9fa',
                                 relief=tk.FLAT, highlightthickness=0)
        self.info_text.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.info_text.config(state=tk.DISABLED)

    def create_function_grid(self, parent):
        """创建功能卡片网格"""
        # 容器
        container = tk.Frame(parent, bg='#f5f6fa')

        # 标题
        title = tk.Label(container, text="🔧 可视化功能",
                         font=('微软雅黑', 16, 'bold'), bg='#f5f6fa', fg=self.COLORS['dark'])
        title.pack(anchor=tk.W, pady=(0, 15))

        # 卡片网格容器
        grid = tk.Frame(container, bg='#f5f6fa')
        grid.pack(fill=tk.BOTH, expand=True)

        # 配置网格列 - 2列，均匀分布
        for i in range(2):
            grid.columnconfigure(i, weight=1, uniform='col')

        # 功能卡片定义
        cards = [
            {
                'title': '信号波形',
                'desc': '多通道信号显示 · 实时滤波 · 翻页导航 · 事件标记',
                'icon': '📈',
                'color': self.COLORS['card1'],
                'command': self.open_signal_view
            },
            {
                'title': '统计分析',
                'desc': '箱线图 · ROC曲线 · 混淆矩阵 · 统计表格',
                'icon': '📊',
                'color': self.COLORS['card2'],
                'command': self.open_stats_view
            },
            {
                'title': '柱状图',
                'desc': '特征对比 · Excel数据可视化 · 数值标签',
                'icon': '📋',
                'color': self.COLORS['card3'],
                'command': self.open_bar_view
            },
            {
                'title': '快速预览',
                'desc': '独立绘图窗口 · 快速查看信号片段',
                'icon': '🔍',
                'color': self.COLORS['card4'],
                'command': self.open_quick_plot
            }
        ]

        # 创建卡片
        for i, card in enumerate(cards):
            row = i // 2
            col = i % 2

            card_widget = ModernCard(
                grid,
                title=card['title'],
                description=card['desc'],
                icon=card['icon'],
                color=card['color'],
                command=card['command']
            )
            card_widget.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')

        # 配置行权重，让卡片能够扩展
        for i in range(2):
            grid.rowconfigure(i, weight=1)

        return container

    def create_status_bar(self):
        """创建状态栏"""
        status_bar = tk.Frame(self, bg='#ecf0f1', height=30)
        status_bar.grid(row=2, column=0, sticky="ew")
        status_bar.grid_propagate(False)

        # 状态信息
        self.status_var = tk.StringVar(value="✨ 就绪")
        status_label = tk.Label(status_bar, textvariable=self.status_var,
                                bg='#ecf0f1', fg='#7f8c8d',
                                font=('微软雅黑', 9))
        status_label.pack(side=tk.LEFT, padx=15, pady=5)

        # 数据模态信息
        self.modal_info_var = tk.StringVar(value="")
        modal_label = tk.Label(status_bar, textvariable=self.modal_info_var,
                               bg='#ecf0f1', fg=self.COLORS['accent'],
                               font=('微软雅黑', 9, 'bold'))
        modal_label.pack(side=tk.RIGHT, padx=15, pady=5)

    def browse_file(self):
        """浏览文件"""
        file_path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[
                ("所有支持文件", "*.edf *.bdf *.gdf *.mat *.csv *.xlsx *.xls *.json *.npy"),
                ("EDF文件", "*.edf"),
                ("BDF文件", "*.bdf"),
                ("MAT文件", "*.mat"),
                ("NumPy文件", "*.npy"),
                ("JSON文件", "*.json"),
                ("CSV文件", "*.csv"),
                ("Excel文件", "*.xlsx *.xls")
            ]
        )

        if file_path:
            self.file_path_var.set(file_path)
            self.status_var.set(f"已选择文件: {os.path.basename(file_path)}")

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
            self.status_var.set("🔄 正在加载文件...")
            self.update()

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

            self.status_var.set(f"✅ 文件加载成功: {os.path.basename(file_path)}")
            messagebox.showinfo("成功", "文件加载成功！")

        except Exception as e:
            self.status_var.set("❌ 文件加载失败")
            messagebox.showerror("错误", f"文件加载失败:\n{str(e)}")

    def load_demo_data(self):
        """加载演示数据"""
        self.status_var.set("🔄 正在生成演示数据...")
        self.update()

        fs = 1000
        t = np.arange(0, 30, 1 / fs)

        self.data_dict = {
            "meta": {
                "subject_id": "demo",
                "session_id": "session1",
                "task": "rest",
                "modality": ["EEG"]
            },
            "signal": {
                "EEG": {
                    "data": np.array([np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t))
                                      for _ in range(8)]),
                    "sampling_rate": fs,
                    "channel_names": [f"EEG_{i}" for i in range(8)],
                    "unit": "uV"
                }
            },
            "event": {
                "event_time": [5.0, 10.0, 15.0],
                "event_label": ["stim1", "stim2", "stim3"],
                "event_id": [1, 2, 3],
                "duration": [1, 1, 1]
            }
        }

        self.file_path = None
        self.update_file_info()

        self.status_var.set("🎲 演示数据已加载")
        messagebox.showinfo("成功", "演示数据加载成功！")

    def reset_data(self):
        """重置数据"""
        self.data_dict = None
        self.file_path = None
        self.file_path_var.set("")
        self.update_file_info()
        self.status_var.set("🔄 数据已重置")

    def update_file_info(self):
        """更新文件信息"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)

        if not self.data_dict:
            self.info_text.insert(tk.END, "请加载数据文件或使用演示数据")
            self.modal_info_var.set("")
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
            self.modal_info_var.set(f"当前: {', '.join(modalities)}")

        self.info_text.config(state=tk.DISABLED)

    def open_signal_view(self):
        """打开信号视图"""
        if not self.data_dict:
            messagebox.showwarning("警告", "请先加载数据")
            return

        try:
            window = tk.Toplevel(self)
            window.title("信号波形视图")
            window.geometry("1200x800")

            from .signal_view import SignalView
            view = SignalView(window, self.data_dict)
            view.pack(fill=tk.BOTH, expand=True)

            self.current_views['signal'] = window
            self.status_var.set("📈 信号视图已打开")

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

            from .stats_view import StatsView
            view = StatsView(window, self.data_dict)
            view.pack(fill=tk.BOTH, expand=True)

            self.current_views['stats'] = window
            self.status_var.set("📊 统计视图已打开")

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

            from .bar_view import BarView
            view = BarView(window, self.data_dict)
            view.pack(fill=tk.BOTH, expand=True)

            self.current_views['bar'] = window
            self.status_var.set("📋 柱状图视图已打开")

        except Exception as e:
            messagebox.showerror("错误", f"打开柱状图视图失败:\n{str(e)}")

    def open_quick_plot(self):
        """打开快速预览"""
        if not self.data_dict:
            messagebox.showwarning("警告", "请先加载数据")
            return

        try:
            signal_dict = self.data_dict.get("signal", {})
            if not signal_dict:
                messagebox.showwarning("警告", "数据中没有信号")
                return

            modality = list(signal_dict.keys())[0]
            signal_info = signal_dict[modality]

            data = signal_info.get('data')
            fs = signal_info.get('sampling_rate', 1000)
            ch_names = signal_info.get('channel_names', [f"Ch{i}" for i in range(data.shape[0])])

            from .plot_dialog import quick_plot
            quick_plot(self, data, fs, ch_names, f"快速预览 - {modality}")

            self.status_var.set("🔍 快速预览已打开")

        except Exception as e:
            messagebox.showerror("错误", f"打开快速预览失败:\n{str(e)}")


# 测试代码
if __name__ == "__main__":
    root = tk.Tk()
    root.title("高级可视化面板 - 支持滚动")
    root.geometry("1000x800")

    # 设置根窗口样式
    root.configure(bg='#f5f6fa')

    panel = ModernVisualizationPanel(root)
    panel.pack(fill=tk.BOTH, expand=True)

    root.mainloop()