# -*- coding: utf-8 -*-
"""
统一特征可视化模块 - 自动适配所有模态
数据格式要求：
{
    "processed": {
        "features": {
            "fNIRS": {              # 或 "EMG", "EEG", "ECG"
                "time_domain": {
                    "mean": [ch1_val, ch2_val, ...],  # 多通道数组
                    "std": [ch1_val, ch2_val, ...],
                    ...
                },
                "freq_domain": {
                    "power": [ch1_val, ch2_val, ...],
                    ...
                },
                "wavelet": {
                    "cA4_energy": [ch1_val, ch2_val, ...],
                    ...
                },
                "hbo_hbr": {         # fNIRS特有
                    "hbo_mean": 4998.8,    # 全局标量
                    "hbr_mean": 4998.34,
                    ...
                }
            }
        }
    }
}
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import json
import os
from collections import defaultdict

# ==================== 全局字体设置 ====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FeatureView(tk.Toplevel):
    """统一特征可视化主窗口 - 自动适配所有模态"""

    # 各模态推荐的图表类型（可根据需要扩展）
    RECOMMENDED_CHARTS = {
        "fNIRS": [
            ("📈 曲线图 - 特征对比", "curve"),
            ("📊 柱状图 - 特征对比", "bar"),
            ("🔥 热力图 - 通道×特征", "heatmap"),
            ("🔄 HbO/HbR通道对比", "fnirs_dual"),
            ("📊 全局平均特征", "fnirs_global"),
            ("📋 数据表格", "table")
        ],
        "EMG": [
            ("📈 曲线图 - 特征对比", "curve"),
            ("📊 柱状图 - 特征对比", "bar"),
            ("🔥 热力图 - 通道×特征", "heatmap"),
            ("📊 功率谱特征", "emg_spectrum"),
            ("📋 数据表格", "table")
        ],
        "EEG": [
            ("📈 曲线图 - 特征对比", "curve"),
            ("📊 柱状图 - 特征对比", "bar"),
            ("🔥 热力图 - 通道×特征", "heatmap"),
            ("🗺️ 地形图 (需MNE)", "topomap"),
            ("📋 数据表格", "table")
        ],
        "ECG": [
            ("📈 曲线图 - 特征对比", "curve"),
            ("📊 柱状图 - 特征对比", "bar"),
            ("🔥 热力图 - 通道×特征", "heatmap"),
            ("📊 HRV频谱", "ecg_hrv_spectrum"),
            ("📋 数据表格", "table")
        ]
    }

    # 默认通用图表
    DEFAULT_CHARTS = [
        ("📈 曲线图", "curve"),
        ("📊 柱状图", "bar"),
        ("🔥 热力图", "heatmap"),
        ("📋 表格", "table")
    ]

    def __init__(self, parent, data_dict):
        super().__init__(parent)
        self.parent = parent
        self.data_dict = data_dict

        # ===== 解析统一格式的数据 =====
        self._parse_data()

        self.title(f"特征可视化 - {self.modality}")
        self.geometry("1400x850")
        self.minsize(800, 600)

        # 绘图设置
        self.settings = {
            "width": 10,
            "height": 7,
            "title": "",
            "xlabel": "特征",
            "ylabel": "幅值"
        }
        self.current_canvas = None
        self.chart_type = "curve"

        # 先创建菜单，再创建布局
        self._create_menu()
        self._create_layout()

        # 强制更新
        self.update()

    def _create_menu(self):
        """创建菜单栏"""
        print("=== 创建菜单栏 ===")

        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # 测试菜单
        test_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="测试", menu=test_menu)
        test_menu.add_command(label="测试菜单项", command=lambda: print("测试菜单点击"))

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="保存图像", command=self.save_plot, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.destroy)

        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="设置", menu=settings_menu)
        settings_menu.add_command(label="绘图设置", command=self.show_settings, accelerator="Ctrl+P")

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图", menu=view_menu)
        view_menu.add_command(label="全选通道", command=self.select_all_channels)
        view_menu.add_command(label="全选特征", command=self.select_all_features)
        view_menu.add_separator()
        view_menu.add_command(label="重置视图", command=self.reset_view)

        self.bind_all("<Control-s>", lambda e: self.save_plot())
        self.bind_all("<Control-p>", lambda e: self.show_settings())

        print("=== 菜单栏创建完成 ===")

    def show_settings(self):
        """显示绘图设置对话框"""
        dialog = PlotSettingsDialog(self, self.settings)
        self.wait_window(dialog)
        if dialog.result:
            self.settings.update(dialog.result)
            # 如果已经有图形，重新绘制
            if hasattr(self, 'current_canvas') and self.current_canvas:
                self.plot()

    def save_plot(self):
        """保存当前图形"""
        if not hasattr(self, 'current_canvas') or not self.current_canvas:
            messagebox.showwarning("警告", "没有可保存的图形，请先生成图形")
            return

        # 获取当前画布
        canvas = self.current_canvas

        # 如果是表格视图，不能保存为图片
        if isinstance(canvas, UnifiedTableCanvas):
            messagebox.showinfo("提示", "表格视图不支持保存为图片，请使用截图功能")
            return

        # 询问保存路径
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG 图片", "*.png"),
                ("JPEG 图片", "*.jpg"),
                ("PDF 文档", "*.pdf"),
                ("SVG 矢量图", "*.svg"),
                ("所有文件", "*.*")
            ],
            title="保存图形"
        )

        if not file_path:
            return

        try:
            # 保存图形
            dpi = simpledialog.askinteger("DPI", "请输入图片分辨率 (DPI):",
                                          initialvalue=300, minvalue=72, maxvalue=1200)
            if dpi:
                canvas.fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
                messagebox.showinfo("成功", f"图形已保存到:\n{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败:\n{str(e)}")

    def reset_view(self):
        """重置视图（全选所有通道和特征）"""
        self.select_all_channels()
        self.select_all_features()
        self.plot()

    def _parse_data(self):
        """解析统一格式的数据"""
        print("\n=== 开始解析数据 ===")

        # 获取特征数据
        processed = self.data_dict.get("processed", {})
        print(f"processed 类型: {type(processed)}")
        print(f"processed 键: {list(processed.keys())}")

        # 先尝试获取 features 字典
        features_dict = None

        if "features" in processed:
            features_dict = processed["features"]
            print("✅ 从 processed['features'] 获取数据")
            print(f"   features_dict 类型: {type(features_dict)}")
            if isinstance(features_dict, dict):
                print(f"   features_dict 键: {list(features_dict.keys())}")
                if features_dict:
                    print(f"   features_dict 非空，包含 {len(features_dict)} 个键")
                else:
                    print("   ⚠️ features_dict 是空字典！")
            else:
                print(f"   features_dict 不是字典，而是: {type(features_dict)}")

        if not features_dict and "feature" in processed:
            features_dict = processed["feature"]
            print("✅ 从 processed['feature'] 获取数据")
            if isinstance(features_dict, dict):
                print(f"   features_dict 键: {list(features_dict.keys())}")

        # 如果 processed 中没有，尝试根目录
        if not features_dict:
            print("尝试从根目录获取...")
            if "features" in self.data_dict:
                features_dict = self.data_dict["features"]
                print("✅ 从根目录['features'] 获取数据")
                if isinstance(features_dict, dict):
                    print(f"   features_dict 键: {list(features_dict.keys())}")
            elif "feature" in self.data_dict:
                features_dict = self.data_dict["feature"]
                print("✅ 从根目录['feature'] 获取数据")
                if isinstance(features_dict, dict):
                    print(f"   features_dict 键: {list(features_dict.keys())}")

        # 如果还是找不到，打印整个数据字典的结构
        if not features_dict:
            print("\n❌ 错误：找不到特征数据")
            print("=" * 50)
            print("数据字典完整结构:")
            self._print_dict_structure(self.data_dict)
            print("=" * 50)
            print(f"processed中的键: {list(processed.keys())}")
            print(f"根目录中的键: {list(self.data_dict.keys())}")
            raise ValueError("没有找到特征数据")

        print(f"\n✅ 成功获取 features_dict")
        print(f"   features_dict 类型: {type(features_dict)}")
        print(f"   features_dict 键: {list(features_dict.keys())}")

        # ===== 直接使用 features_dict，没有模态层 =====
        # 从 meta 中获取模态，如果不存在则使用 "UNKNOWN"
        meta_modality = self.data_dict.get("meta", {}).get("modality", ["UNKNOWN"])
        if isinstance(meta_modality, list) and meta_modality:
            self.modalities = meta_modality
        else:
            self.modalities = ["UNKNOWN"]
        self.modality = self.modalities[0]
        self.modality_features = features_dict  # 直接使用 features_dict 作为模态特征

        print(f"\n📊 解析结果:")
        print(f"   模态: {self.modality}")
        print(f"   特征类别: {list(self.modality_features.keys())}")

        # 获取通道信息
        self._get_channel_info()

        # 构建特征列表
        self._build_feature_list()

        print(f"\n=== 数据加载成功 ===")
        print(f"模态: {self.modality}")
        print(f"通道数: {self.n_channels}")
        print(f"通道名称: {self.channel_names}")
        print(f"特征类别: {list(self.modality_features.keys())}")
        print(f"特征总数: {len(self.feature_names)}")

        # 打印前10个特征名
        if self.feature_names:
            print(f"特征示例 (前10个): {self.feature_names[:10]}")

    def _print_dict_structure(self, d, indent=0, max_depth=3):
        """打印字典结构，帮助调试"""
        if indent > max_depth:
            return
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}: dict ({len(value)} 个键)")
                if indent < max_depth:
                    self._print_dict_structure(value, indent + 1, max_depth)
            elif isinstance(value, (list, tuple)):
                print(f"{prefix}{key}: {type(value).__name__} (长度: {len(value)})")
            elif isinstance(value, np.ndarray):
                print(f"{prefix}{key}: ndarray (形状: {value.shape})")
            else:
                print(f"{prefix}{key}: {type(value).__name__}")

    def _get_channel_info(self):
        """从数据中获取通道信息"""
        # 默认值
        self.n_channels = 1
        self.channel_names = ["Global"]

        # 从meta中获取
        meta = self.data_dict.get("meta", {})
        if "n_channels" in meta:
            self.n_channels = meta["n_channels"]
        if "channel_names" in meta:
            self.channel_names = meta["channel_names"]
        else:
            self.channel_names = [f"Ch{i + 1}" for i in range(self.n_channels)]

        # 从信号中获取（更准确）
        signal = self.data_dict.get("signal", {})
        for mod, sig_info in signal.items():
            if mod.upper() == self.modality.upper():
                if "data" in sig_info:
                    data = sig_info["data"]
                    if hasattr(data, "shape") and len(data.shape) > 1:
                        self.n_channels = data.shape[0]
                if "channel_names" in sig_info and sig_info["channel_names"]:
                    self.channel_names = sig_info["channel_names"]
                break

    def _build_feature_list(self):
        """构建特征列表 - 统一格式"""
        self.feature_categories = list(self.modality_features.keys())
        self.feature_map = {}  # {特征名: (类别, 是否全局, 示例值)}
        self.feature_values = defaultdict(list)  # {特征名: [通道值列表]}

        for category, features in self.modality_features.items():
            if not isinstance(features, dict):
                continue

            for feat_name, feat_value in features.items():
                # 判断是否为多通道特征
                is_multichannel = isinstance(feat_value, (list, tuple, np.ndarray)) and len(feat_value) > 1
                is_global = not is_multichannel

                # 生成显示用的特征名
                display_name = f"{category}.{feat_name}"

                self.feature_map[display_name] = {
                    "category": category,
                    "raw_name": feat_name,
                    "is_global": is_global,
                    "is_multichannel": is_multichannel
                }

                # 存储特征值
                if is_multichannel:
                    self.feature_values[display_name] = list(feat_value)
                else:
                    # 全局特征扩展到所有通道
                    self.feature_values[display_name] = [feat_value] * self.n_channels

        self.feature_names = sorted(list(self.feature_map.keys()))

    def _create_layout(self):
        """创建主布局 - 完全自适应，无空白"""

        # 使用PanedWindow让用户可以调整比例
        self.main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # 左侧控制面板 - 固定初始宽度
        self.left_frame = ttk.Frame(self.main_paned, relief=tk.FLAT, width=280)
        self.main_paned.add(self.left_frame, weight=1)

        # 右侧图形面板
        self.right_frame = ttk.Frame(self.main_paned, relief=tk.FLAT)
        self.main_paned.add(self.right_frame, weight=3)

        # 创建左右面板内容
        self._create_left_panel(self.left_frame)
        self._create_right_panel(self.right_frame)

        # 设置最小宽度并防止收缩
        self.left_frame.update_idletasks()
        self.left_frame.pack_propagate(False)
        self.left_frame.grid_propagate(False)

        # 绑定窗口大小调整事件
        self.bind('<Configure>', self._on_window_configure)

    def _on_window_configure(self, event):
        """窗口大小改变时的处理"""
        # 确保左侧面板的grid权重正确应用
        self.left_frame.update_idletasks()

    def _create_left_panel(self, parent):
        """创建左侧控制面板 - 完全填充，无空白，无需滚动"""

        # 直接使用grid布局，不用Canvas滚动
        parent.grid_rowconfigure(0, weight=0)  # 标题 - 固定
        parent.grid_rowconfigure(1, weight=0)  # 模态 - 固定
        parent.grid_rowconfigure(2, weight=1)  # 通道 - 可扩展
        parent.grid_rowconfigure(3, weight=1)  # 特征 - 可扩展
        parent.grid_rowconfigure(4, weight=0)  # 图表 - 固定
        parent.grid_rowconfigure(5, weight=0)  # 按钮 - 固定
        parent.grid_columnconfigure(0, weight=1)

        row_idx = 0

        # ===== 标题 =====
        title_label = tk.Label(parent, text=f"{self.modality} 控制面板",
                               font=('微软雅黑', 12, 'bold'), bg='#f0f0f0')
        title_label.grid(row=row_idx, column=0, sticky="ew", pady=2)
        row_idx += 1

        # ===== 模态选择 =====
        if len(self.modalities) > 1:
            modality_frame = ttk.LabelFrame(parent, text="模态", padding=2)
            modality_frame.grid(row=row_idx, column=0, sticky="ew", pady=2)

            self.modality_var = tk.StringVar(value=self.modality)
            modality_combo = ttk.Combobox(modality_frame, textvariable=self.modality_var,
                                          values=self.modalities, state="readonly")
            modality_combo.pack(fill=tk.X)
            modality_combo.bind('<<ComboboxSelected>>', self._on_modality_change)
            row_idx += 1

        # ===== 通道选择 =====
        ch_frame = ttk.LabelFrame(parent, text=f"通道 ({self.n_channels})", padding=2)
        ch_frame.grid(row=row_idx, column=0, sticky="nsew", pady=2)

        # 配置通道框架的grid权重
        ch_frame.grid_rowconfigure(0, weight=0)  # 搜索框 - 固定
        ch_frame.grid_rowconfigure(1, weight=1)  # 列表 - 可扩展
        ch_frame.grid_rowconfigure(2, weight=0)  # 按钮 - 固定
        ch_frame.grid_columnconfigure(0, weight=1)

        # 搜索框
        search_frame = ttk.Frame(ch_frame)
        search_frame.grid(row=0, column=0, sticky="ew", pady=1)
        ttk.Label(search_frame, text="🔍", width=2).pack(side=tk.LEFT)
        self.ch_search_var = tk.StringVar()
        self.ch_search_var.trace('w', lambda *args: self._filter_channels())
        ch_search = ttk.Entry(search_frame, textvariable=self.ch_search_var)
        ch_search.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 通道列表 - 使用grid让它填满
        list_container = ttk.Frame(ch_frame)
        list_container.grid(row=1, column=0, sticky="nsew", pady=1)
        list_container.grid_rowconfigure(0, weight=1)
        list_container.grid_columnconfigure(0, weight=1)

        self.ch_listbox = tk.Listbox(list_container, selectmode=tk.MULTIPLE,
                                     exportselection=False, font=('微软雅黑', 9))
        ch_scroll = ttk.Scrollbar(list_container, orient=tk.VERTICAL,
                                  command=self.ch_listbox.yview)
        self.ch_listbox.configure(yscrollcommand=ch_scroll.set)

        self.ch_listbox.grid(row=0, column=0, sticky="nsew")
        ch_scroll.grid(row=0, column=1, sticky="ns")

        # 填充通道
        self.all_channels_display = self.channel_names.copy()
        for ch in self.channel_names:
            self.ch_listbox.insert(tk.END, ch)
        self.ch_listbox.selection_set(0, tk.END)

        # 按钮行
        ch_btn_frame = ttk.Frame(ch_frame)
        ch_btn_frame.grid(row=2, column=0, sticky="ew", pady=1)
        ch_btn_frame.grid_columnconfigure(0, weight=1)
        ch_btn_frame.grid_columnconfigure(1, weight=1)

        ttk.Button(ch_btn_frame, text="全选", command=self.select_all_channels).grid(row=0, column=0, sticky="ew",
                                                                                     padx=1)
        ttk.Button(ch_btn_frame, text="清空", command=self.clear_all_channels).grid(row=0, column=1, sticky="ew",
                                                                                    padx=1)

        row_idx += 1

        # ===== 特征选择 =====
        feat_frame = ttk.LabelFrame(parent, text=f"特征 ({len(self.feature_names)})", padding=2)
        feat_frame.grid(row=row_idx, column=0, sticky="nsew", pady=2)

        # 配置特征框架的grid权重
        feat_frame.grid_rowconfigure(0, weight=0)  # 搜索框 - 固定
        feat_frame.grid_rowconfigure(1, weight=1)  # 列表 - 可扩展
        feat_frame.grid_rowconfigure(2, weight=0)  # 按钮 - 固定
        feat_frame.grid_columnconfigure(0, weight=1)

        # 搜索框
        search_frame2 = ttk.Frame(feat_frame)
        search_frame2.grid(row=0, column=0, sticky="ew", pady=1)
        ttk.Label(search_frame2, text="🔍", width=2).pack(side=tk.LEFT)
        self.feat_search_var = tk.StringVar()
        self.feat_search_var.trace('w', lambda *args: self._filter_features())
        feat_search = ttk.Entry(search_frame2, textvariable=self.feat_search_var)
        feat_search.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 特征列表
        list_container2 = ttk.Frame(feat_frame)
        list_container2.grid(row=1, column=0, sticky="nsew", pady=1)
        list_container2.grid_rowconfigure(0, weight=1)
        list_container2.grid_columnconfigure(0, weight=1)

        self.feat_listbox = tk.Listbox(list_container2, selectmode=tk.MULTIPLE,
                                       exportselection=False, font=('微软雅黑', 8))
        feat_scroll = ttk.Scrollbar(list_container2, orient=tk.VERTICAL,
                                    command=self.feat_listbox.yview)
        self.feat_listbox.configure(yscrollcommand=feat_scroll.set)

        self.feat_listbox.grid(row=0, column=0, sticky="nsew")
        feat_scroll.grid(row=0, column=1, sticky="ns")

        # 填充特征
        self.all_features_display = self.feature_names.copy()
        self.feat_listbox_display = []
        for feat in self.feature_names:
            if self.feature_map[feat]["is_global"]:
                display = f"🌐 {feat}"
            else:
                display = f"📊 {feat}"
            self.feat_listbox_display.append(display)
            self.feat_listbox.insert(tk.END, display)
        self.feat_listbox.selection_set(0, tk.END)

        # 按钮行
        feat_btn_frame = ttk.Frame(feat_frame)
        feat_btn_frame.grid(row=2, column=0, sticky="ew", pady=1)
        feat_btn_frame.grid_columnconfigure(0, weight=1)
        feat_btn_frame.grid_columnconfigure(1, weight=1)

        ttk.Button(feat_btn_frame, text="全选", command=self.select_all_features).grid(row=0, column=0, sticky="ew",
                                                                                       padx=1)
        ttk.Button(feat_btn_frame, text="清空", command=self.clear_all_features).grid(row=0, column=1, sticky="ew",
                                                                                      padx=1)

        row_idx += 1

        # ===== 图表类型 =====
        chart_frame = ttk.LabelFrame(parent, text="图表", padding=2)
        chart_frame.grid(row=row_idx, column=0, sticky="ew", pady=2)

        self.chart_type_var = tk.StringVar(value="curve")
        charts = self.RECOMMENDED_CHARTS.get(self.modality, self.DEFAULT_CHARTS)

        # 使用grid布局让图表选项填满
        for i, (label, value) in enumerate(charts):
            row = i // 2
            col = i % 2
            chart_frame.grid_columnconfigure(col, weight=1)

            # 简化标签显示
            short_label = label.split(' ')[1] if ' ' in label else label
            rb = ttk.Radiobutton(chart_frame, text=short_label,
                                 variable=self.chart_type_var,
                                 value=value, command=self.on_chart_type_changed)
            rb.grid(row=row, column=col, sticky=tk.W, padx=2, pady=1)

        row_idx += 1

        # ===== 操作按钮 =====
        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=row_idx, column=0, sticky="ew", pady=2)

        # 生成图形按钮 - 占满整行
        generate_btn = tk.Button(btn_frame, text="生成图形", bg="#27ae60", fg="white",
                                 font=('微软雅黑', 10, 'bold'), command=self.plot,
                                 relief=tk.FLAT)
        generate_btn.pack(fill=tk.X, pady=1)

        # 保存和设置按钮 - 并排
        sub_btn_frame = ttk.Frame(btn_frame)
        sub_btn_frame.pack(fill=tk.X, pady=1)
        sub_btn_frame.grid_columnconfigure(0, weight=1)
        sub_btn_frame.grid_columnconfigure(1, weight=1)

        save_btn = tk.Button(sub_btn_frame, text="保存", bg="#3498db", fg="white",
                             font=('微软雅黑', 9), command=self.save_plot,
                             relief=tk.FLAT)
        save_btn.grid(row=0, column=0, sticky="ew", padx=1)

        setting_btn = tk.Button(sub_btn_frame, text="设置", bg="#f39c12", fg="white",
                                font=('微软雅黑', 9), command=self.show_settings,
                                relief=tk.FLAT)
        setting_btn.grid(row=0, column=1, sticky="ew", padx=1)

    def _create_right_panel(self, parent):
        """创建右侧图形面板 - 完全填充，无空白"""

        # 使用grid布局填满整个父容器
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # 创建画布容器
        self.canvas_container = ttk.Frame(parent)
        self.canvas_container.grid(row=0, column=0, sticky="nsew")
        self.canvas_container.grid_rowconfigure(0, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)

        # 初始提示
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, self.canvas_container)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.ax = self.fig.add_subplot(111)
        self.ax.text(0.5, 0.5, f"请选择{self.modality}特征和图表类型后点击「生成图形」",
                     ha='center', va='center', fontsize=12, color='#999999')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    def _filter_channels(self):
        """过滤通道列表"""
        search = self.ch_search_var.get().lower()
        self.ch_listbox.delete(0, tk.END)

        for i, ch in enumerate(self.all_channels_display):
            # 创建显示的文本
            display_text = f"Ch{i + 1}" if ch.startswith('Channel') else ch
            if search in ch.lower() or search in display_text.lower():
                self.ch_listbox.insert(tk.END, display_text)

    def _filter_features(self):
        """过滤特征列表"""
        search = self.feat_search_var.get().lower()
        self.feat_listbox.delete(0, tk.END)

        for i, feat in enumerate(self.all_features_display):
            display = self.feat_listbox_display[i]
            if search in feat.lower() or search in display.lower():
                self.feat_listbox.insert(tk.END, display)

    def select_all_channels(self):
        self.ch_listbox.selection_set(0, tk.END)

    def clear_all_channels(self):
        self.ch_listbox.selection_clear(0, tk.END)

    def select_all_features(self):
        self.feat_listbox.selection_set(0, tk.END)

    def clear_all_features(self):
        self.feat_listbox.selection_clear(0, tk.END)

    def _on_modality_change(self, event=None):
        """切换模态"""
        new_modality = self.modality_var.get()
        if new_modality != self.modality:
            self.modality = new_modality
            self.modality_features = self.data_dict["processed"]["features"][new_modality]
            self._build_feature_list()
            self.title(f"特征可视化 - {self.modality}")
            # 刷新界面...

    def on_chart_type_changed(self):
        self.chart_type = self.chart_type_var.get()


    def get_selected_channels(self):
        """获取选中的通道名"""
        indices = self.ch_listbox.curselection()
        # 映射回原始通道名
        filtered = [self.all_channels_display[i] for i in indices]
        return [ch for ch in self.channel_names if ch in filtered]

    def get_selected_features(self):
        """获取选中的特征名"""
        indices = self.feat_listbox.curselection()
        return [self.all_features_display[i] for i in indices]

    def plot(self):
        """生成图形 - 统一绘图接口"""
        selected_channels = self.get_selected_channels()
        selected_features = self.get_selected_features()

        if not selected_channels or not selected_features:
            messagebox.showwarning("警告", "请至少选择一个通道和一个特征")
            return

        # 检查是否有全局特征
        global_features = [f for f in selected_features if self.feature_map[f]["is_global"]]
        if global_features and len(selected_channels) > 1:
            result = messagebox.askyesno("提示",
                                         f"您选择了 {len(global_features)} 个全局特征（如 {global_features[0]}），\n"
                                         "这些特征在所有通道上的值相同，可能导致图形重叠。\n"
                                         "是否继续？")
            if not result:
                return

        print(f"\n=== 开始绘图 ===")
        print(f"选中的通道: {selected_channels}")
        print(f"选中的特征: {selected_features}")
        print(f"全局特征: {global_features}")

        # 清除旧画布
        for widget in self.canvas_container.winfo_children():
            widget.destroy()

        chart_type = self.chart_type_var.get()
        print(f"图表类型: {chart_type}")

        try:
            # 创建对应图表
            if chart_type == "curve":
                canvas = UnifiedCurveCanvas(self.canvas_container, self)
                canvas.plot(selected_channels, selected_features)
                canvas.get_widget().pack(fill=tk.BOTH, expand=True)

            elif chart_type == "bar":
                canvas = UnifiedBarCanvas(self.canvas_container, self)
                canvas.plot(selected_channels, selected_features)
                canvas.get_widget().pack(fill=tk.BOTH, expand=True)

            elif chart_type == "heatmap":
                canvas = UnifiedHeatmapCanvas(self.canvas_container, self)
                canvas.plot(selected_channels, selected_features)
                canvas.get_widget().pack(fill=tk.BOTH, expand=True)

            elif chart_type == "table":
                canvas = UnifiedTableCanvas(self.canvas_container, self)
                canvas.plot(selected_channels, selected_features)

            elif chart_type == "fnirs_dual" and self.modality == "fNIRS":
                canvas = UnifiedFNIRSDualCanvas(self.canvas_container, self)
                canvas.plot(selected_channels, selected_features)
                canvas.get_widget().pack(fill=tk.BOTH, expand=True)

            elif chart_type == "fnirs_global" and self.modality == "fNIRS":
                canvas = UnifiedFNIRSGlobalCanvas(self.canvas_container, self)
                canvas.plot(selected_channels, selected_features)
                canvas.get_widget().pack(fill=tk.BOTH, expand=True)

            elif chart_type == "emg_spectrum" and self.modality == "EMG":
                canvas = UnifiedEMGSpectrumCanvas(self.canvas_container, self)
                canvas.plot(selected_channels, selected_features)
                canvas.get_widget().pack(fill=tk.BOTH, expand=True)

            elif chart_type == "ecg_hrv_spectrum" and self.modality == "ECG":
                canvas = UnifiedECGHRVCanvas(self.canvas_container, self)
                canvas.plot(selected_channels, selected_features)
                canvas.get_widget().pack(fill=tk.BOTH, expand=True)

            else:
                canvas = UnifiedCurveCanvas(self.canvas_container, self)
                canvas.plot(selected_channels, selected_features)
                canvas.get_widget().pack(fill=tk.BOTH, expand=True)

            # 添加工具栏（表格不需要）
            if chart_type != "table":
                toolbar_frame = ttk.Frame(self.canvas_container)
                toolbar_frame.pack(fill=tk.X)
                toolbar = NavigationToolbar2Tk(canvas.canvas, toolbar_frame)
                toolbar.update()

            self.current_canvas = canvas
            print("✅ 绘图完成")

        except Exception as e:
            messagebox.showerror("错误", f"绘图失败:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def save_plot(self):
        """保存当前图形"""
        if not hasattr(self, 'current_canvas') or not self.current_canvas:
            messagebox.showwarning("警告", "没有可保存的图形，请先生成图形")
            return

        # 获取当前画布
        canvas = self.current_canvas

        # 询问保存路径
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG 图片", "*.png"),
                ("JPEG 图片", "*.jpg"),
                ("PDF 文档", "*.pdf"),
                ("SVG 矢量图", "*.svg"),
                ("所有文件", "*.*")
            ],
            title="保存图形"
        )

        if not file_path:
            return

        try:
            # 保存图形
            dpi = simpledialog.askinteger("DPI", "请输入图片分辨率 (DPI):",
                                          initialvalue=300, minvalue=72, maxvalue=1200)
            if dpi:
                canvas.fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
                messagebox.showinfo("成功", f"图形已保存到:\n{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败:\n{str(e)}")


# ==================== 统一画布基类 ====================

class UnifiedBaseCanvas:
    """统一画布基类"""

    def __init__(self, parent, view):
        self.parent = parent
        self.view = view
        self.fig = Figure(figsize=(view.settings["width"], view.settings["height"]), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.ax = self.fig.add_subplot(111)

    def get_widget(self):
        return self.canvas.get_tk_widget()

    def draw(self):
        self.fig.tight_layout()
        self.canvas.draw()
        self.canvas.flush_events()  # 强制刷新事件


class UnifiedCurveCanvas(UnifiedBaseCanvas):
    """统一曲线图"""

    def plot(self, selected_channels, selected_features):
        self.ax.clear()

        print(f"\n=== UnifiedCurveCanvas.plot ===")
        print(f"selected_channels: {selected_channels}")
        print(f"selected_features: {selected_features}")

        x = np.arange(len(selected_features))
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_channels)))

        for i, ch in enumerate(selected_channels):
            ch_idx = self.view.channel_names.index(ch)
            print(f"通道 {ch} 的索引: {ch_idx}")

            values = []
            for feat in selected_features:
                if feat in self.view.feature_values:
                    vals = self.view.feature_values[feat]
                    if ch_idx < len(vals):
                        val = vals[ch_idx]
                        values.append(val)
                    else:
                        values.append(np.nan)
                else:
                    values.append(np.nan)

            print(f"通道 {ch} 的值列表: {values}")

            # 绘制曲线
            line = self.ax.plot(x, values, marker='o', linewidth=2, label=ch,
                                color=colors[i], markersize=8)[0]

            # 在每个数据点上显示数值
            for j, (x_val, y_val) in enumerate(zip(x, values)):
                if not np.isnan(y_val):
                    # 根据数值大小决定显示格式
                    if abs(y_val) < 0.01 or abs(y_val) > 1000:
                        text = f'{y_val:.2e}'
                    else:
                        text = f'{y_val:.2f}'

                    # 根据曲线位置调整文本位置
                    if i == 0:
                        offset = 5  # 第一条曲线上方
                    else:
                        offset = 5 + i * 3  # 其他曲线上方偏移

                    self.ax.annotate(text, (x_val, y_val),
                                     textcoords="offset points",
                                     xytext=(0, offset),
                                     ha='center', va='bottom',
                                     fontsize=7, color=colors[i],
                                     bbox=dict(boxstyle='round,pad=0.2',
                                               facecolor='white',
                                               alpha=0.7,
                                               edgecolor='none'))

        self.ax.set_xlabel(self.view.settings.get('xlabel', '特征'))
        self.ax.set_ylabel(self.view.settings.get('ylabel', '幅值'))
        self.ax.set_title(self.view.settings.get('title', f'{self.view.modality}特征曲线'))
        self.ax.set_xticks(x)
        self.ax.set_xticklabels([f.split('.')[-1] for f in selected_features], rotation=45, ha='right')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        # 强制刷新
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.tight_layout()
        self.canvas.draw()
        self.canvas.flush_events()

        print("✅ UnifiedCurveCanvas.plot 完成")


class UnifiedBarCanvas(UnifiedBaseCanvas):
    """统一柱状图"""

    def plot(self, selected_channels, selected_features):
        self.ax.clear()

        print(f"\n=== UnifiedBarCanvas.plot ===")
        print(f"selected_channels: {selected_channels}")
        print(f"selected_features: {selected_features}")

        x = np.arange(len(selected_features))
        width = 0.8 / len(selected_channels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_channels)))

        # 存储所有值用于确定y轴范围
        all_values = []

        for i, ch in enumerate(selected_channels):
            ch_idx = self.view.channel_names.index(ch)

            values = []
            for feat in selected_features:
                if feat in self.view.feature_values:
                    vals = self.view.feature_values[feat]
                    if ch_idx < len(vals):
                        val = vals[ch_idx]
                        values.append(val)
                        all_values.append(val)
                    else:
                        values.append(np.nan)
                else:
                    values.append(np.nan)

            offset = (i - len(selected_channels) / 2 + 0.5) * width
            bars = self.ax.bar(x + offset, values, width, label=ch,
                               color=colors[i], alpha=0.7)

            # 在每个柱子上显示数值
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    height = bar.get_height()
                    # 根据数值大小决定显示格式
                    if abs(val) < 0.01 or abs(val) > 1000:
                        text = f'{val:.2e}'
                    else:
                        text = f'{val:.2f}'

                    # 根据数值正负决定文本位置
                    if val >= 0:
                        va = 'bottom'
                        y_pos = height
                    else:
                        va = 'top'
                        y_pos = height

                    self.ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                                 text, ha='center', va=va,
                                 fontsize=8, rotation=0,
                                 bbox=dict(boxstyle='round,pad=0.1',
                                           facecolor='white',
                                           alpha=0.7,
                                           edgecolor='none'))

        self.ax.set_xlabel(self.view.settings.get('xlabel', '特征'))
        self.ax.set_ylabel(self.view.settings.get('ylabel', '幅值'))
        self.ax.set_title(self.view.settings.get('title', f'{self.view.modality}特征柱状图'))
        self.ax.set_xticks(x)
        self.ax.set_xticklabels([f.split('.')[-1] for f in selected_features], rotation=45, ha='right')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3, axis='y')

        # 根据数值范围自动调整y轴边距
        if all_values:
            y_min, y_max = min(all_values), max(all_values)
            y_range = y_max - y_min
            self.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.2 * y_range)

        # 强制刷新
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.tight_layout()
        self.canvas.draw()
        self.canvas.flush_events()

        print("✅ UnifiedBarCanvas.plot 完成")


class UnifiedHeatmapCanvas(UnifiedBaseCanvas):
    """统一热力图"""

    def plot(self, selected_channels, selected_features):
        self.ax.clear()

        print(f"\n=== UnifiedHeatmapCanvas.plot ===")
        print(f"selected_channels: {selected_channels}")
        print(f"selected_features: {selected_features}")

        # 构建矩阵
        n_channels = len(selected_channels)
        n_features = len(selected_features)
        matrix = np.zeros((n_channels, n_features))

        for i, ch in enumerate(selected_channels):
            ch_idx = self.view.channel_names.index(ch)
            for j, feat in enumerate(selected_features):
                if feat in self.view.feature_values:
                    vals = self.view.feature_values[feat]
                    if ch_idx < len(vals):
                        matrix[i, j] = vals[ch_idx]
                    else:
                        matrix[i, j] = np.nan
                else:
                    matrix[i, j] = np.nan

        im = self.ax.imshow(matrix, cmap='viridis', aspect='auto', interpolation='nearest')

        # 在每个单元格中显示数值
        for i in range(n_channels):
            for j in range(n_features):
                val = matrix[i, j]
                if not np.isnan(val):
                    # 根据数值大小决定显示格式
                    if abs(val) < 0.01 or abs(val) > 1000:
                        text = f'{val:.2e}'
                    else:
                        text = f'{val:.2f}'

                    # 根据背景颜色决定文字颜色
                    color = 'white' if matrix[i, j] > matrix.mean() else 'black'

                    self.ax.text(j, i, text,
                                 ha='center', va='center',
                                 fontsize=7, color=color)

        self.ax.set_xticks(range(n_features))
        self.ax.set_yticks(range(n_channels))
        self.ax.set_xticklabels([f.split('.')[-1] for f in selected_features], rotation=45, ha='right')
        self.ax.set_yticklabels(selected_channels)
        self.ax.set_xlabel(self.view.settings.get('xlabel', '特征'))
        self.ax.set_ylabel(self.view.settings.get('ylabel', '通道'))
        self.ax.set_title(self.view.settings.get('title', f'{self.view.modality}特征热力图'))
        self.fig.colorbar(im, ax=self.ax)

        # 强制刷新
        self.fig.tight_layout()
        self.canvas.draw()
        self.canvas.flush_events()

        print("✅ UnifiedHeatmapCanvas.plot 完成")


class UnifiedTableCanvas(UnifiedBaseCanvas):
    """统一表格视图"""

    def __init__(self, parent, view):
        super().__init__(parent, view)
        # 表格不使用matplotlib，使用Treeview
        self.fig.clear()
        self.canvas.get_tk_widget().destroy()

        self.tree_frame = ttk.Frame(parent)
        self.tree_frame.pack(fill=tk.BOTH, expand=True)

        # 创建滚动条
        vsb = ttk.Scrollbar(self.tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(self.tree_frame, orient="horizontal")

        self.tree = ttk.Treeview(self.tree_frame, show="headings",
                                 yscrollcommand=vsb.set,
                                 xscrollcommand=hsb.set,
                                 height=20)

        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        self.tree_frame.grid_rowconfigure(0, weight=1)
        self.tree_frame.grid_columnconfigure(0, weight=1)

    def plot(self, selected_channels, selected_features):
        # 清空
        self.tree.delete(*self.tree.get_children())

        # 设置列
        columns = ["通道"] + [f.split('.')[-1] for f in selected_features]
        self.tree["columns"] = columns
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")

        # 添加数据
        for ch in selected_channels:
            ch_idx = self.view.channel_names.index(ch)
            values = [ch]
            for feat in selected_features:
                vals = self.view.feature_values[feat]
                if ch_idx < len(vals):
                    val = vals[ch_idx]
                    if isinstance(val, (int, float)):
                        values.append(f"{val:.4f}")
                    else:
                        values.append(str(val)[:20])
                else:
                    values.append("N/A")
            self.tree.insert("", tk.END, values=values)


# ==================== 模态特定画布 ====================

class UnifiedFNIRSDualCanvas(UnifiedBaseCanvas):
    """fNIRS HbO/HbR双通道对比"""

    def plot(self, selected_channels, selected_features):
        self.ax.clear()

        # 分离HbO和HbR通道
        hbo_channels = [ch for i, ch in enumerate(selected_channels) if i % 2 == 0]
        hbr_channels = [ch for i, ch in enumerate(selected_channels) if i % 2 == 1]

        n_pairs = min(len(hbo_channels), len(hbr_channels))
        if n_pairs == 0:
            self.ax.text(0.5, 0.5, '请同时选择HbO和HbR通道',
                         ha='center', va='center', fontsize=12)
            self.draw()
            return

        # 选择特征
        display_features = selected_features[:8]
        x = np.arange(len(display_features))
        width = 0.35

        for i in range(n_pairs):
            hbo_idx = self.view.channel_names.index(hbo_channels[i])
            hbr_idx = self.view.channel_names.index(hbr_channels[i])

            hbo_values = []
            hbr_values = []
            for feat in display_features:
                vals = self.view.feature_values[feat]
                if hbo_idx < len(vals):
                    hbo_values.append(vals[hbo_idx])
                else:
                    hbo_values.append(np.nan)
                if hbr_idx < len(vals):
                    hbr_values.append(vals[hbr_idx])
                else:
                    hbr_values.append(np.nan)

            offset = i * width * 2
            self.ax.bar(x - width + offset, hbo_values, width,
                        label=f'{hbo_channels[i]}', color='red', alpha=0.7)
            self.ax.bar(x + offset, hbr_values, width,
                        label=f'{hbr_channels[i]}', color='blue', alpha=0.7)

        self.ax.set_xlabel('特征')
        self.ax.set_ylabel('浓度变化')
        self.ax.set_title('fNIRS HbO/HbR通道对比')
        self.ax.set_xticks(x)
        self.ax.set_xticklabels([f.split('.')[-1] for f in display_features], rotation=45, ha='right')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3, axis='y')
        self.draw()


class UnifiedFNIRSGlobalCanvas(UnifiedBaseCanvas):
    """fNIRS全局平均特征"""

    def plot(self, selected_channels, selected_features):
        self.ax.clear()

        # 筛选全局特征
        global_feats = {}
        for feat in selected_features:
            info = self.view.feature_map[feat]
            if info["is_global"] and any(p in feat for p in ['hbo_', 'hbr_', 'hbt_', 'diff_']):
                global_feats[feat] = self.view.feature_values[feat][0]

        if not global_feats:
            self.ax.text(0.5, 0.5, '未找到全局特征\n(如 hbo_mean, hbr_mean 等)',
                         ha='center', va='center', fontsize=12)
            self.draw()
            return

        # 分组
        hbo_dict = {k: v for k, v in global_feats.items() if 'hbo_' in k}
        hbr_dict = {k: v for k, v in global_feats.items() if 'hbr_' in k}
        hbt_dict = {k: v for k, v in global_feats.items() if 'hbt_' in k}
        diff_dict = {k: v for k, v in global_feats.items() if 'diff_' in k}

        x = np.arange(len(hbo_dict)) if hbo_dict else np.arange(1)
        width = 0.2
        colors = {'hbo': 'red', 'hbr': 'blue', 'hbt': 'green', 'diff': 'purple'}

        if hbo_dict:
            self.ax.bar(x - width * 1.5, list(hbo_dict.values()), width,
                        label='HbO', color=colors['hbo'], alpha=0.7)
        if hbr_dict:
            self.ax.bar(x - width / 2, list(hbr_dict.values()), width,
                        label='HbR', color=colors['hbr'], alpha=0.7)
        if hbt_dict:
            self.ax.bar(x + width / 2, list(hbt_dict.values()), width,
                        label='HbT', color=colors['hbt'], alpha=0.7)
        if diff_dict:
            self.ax.bar(x + width * 1.5, list(diff_dict.values()), width,
                        label='HbO-HbR', color=colors['diff'], alpha=0.7)

        # 设置标签
        if hbo_dict:
            labels = [k.replace('hbo_', '').split('.')[-1] for k in hbo_dict.keys()]
        elif hbr_dict:
            labels = [k.replace('hbr_', '').split('.')[-1] for k in hbr_dict.keys()]
        else:
            labels = [k.split('.')[-1] for k in global_feats.keys()]

        self.ax.set_xlabel('特征')
        self.ax.set_ylabel('浓度变化')
        self.ax.set_title('fNIRS全局平均特征对比')
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels, rotation=45, ha='right')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3, axis='y')
        self.draw()


class UnifiedEMGSpectrumCanvas(UnifiedBaseCanvas):
    """EMG功率谱特征"""

    def plot(self, selected_channels, selected_features):
        self.ax.clear()

        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_channels)))

        for i, ch in enumerate(selected_channels):
            ch_idx = self.view.channel_names.index(ch)

            # 提取频率和功率特征
            freqs = []
            powers = []
            for feat in selected_features:
                vals = self.view.feature_values[feat]
                if ch_idx < len(vals):
                    val = vals[ch_idx]
                    if 'freq' in feat.lower() and 'power' not in feat.lower():
                        freqs.append(val)
                    elif 'power' in feat.lower():
                        powers.append(val)

            if len(freqs) == len(powers) and len(freqs) > 0:
                self.ax.plot(freqs, powers, 'o-', linewidth=2, label=ch,
                             color=colors[i], markersize=6)

        self.ax.set_xlabel('频率 (Hz)')
        self.ax.set_ylabel('功率')
        self.ax.set_title('EMG功率谱特征')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.draw()


class UnifiedECGHRVCanvas(UnifiedBaseCanvas):
    """ECG HRV频谱特征"""

    def plot(self, selected_channels, selected_features):
        self.ax.clear()

        if not selected_channels:
            return

        ch = selected_channels[0]
        ch_idx = self.view.channel_names.index(ch)

        # 提取频带功率
        freq_bands = ['VLF', 'LF', 'HF']
        powers = []

        for band in freq_bands:
            found = False
            for feat in selected_features:
                if band.lower() in feat.lower() and 'power' in feat.lower():
                    vals = self.view.feature_values[feat]
                    if ch_idx < len(vals):
                        powers.append(vals[ch_idx])
                        found = True
                        break
            if not found:
                powers.append(0)

        x = range(len(freq_bands))
        self.ax.bar(x, powers, color=['blue', 'green', 'red'], alpha=0.7)
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(freq_bands)
        self.ax.set_xlabel('频带')
        self.ax.set_ylabel('功率')
        self.ax.set_title('HRV频谱')

        # 显示LF/HF比
        if len(powers) >= 3 and powers[2] > 0:
            lf_hf = powers[1] / powers[2]
            self.ax.text(0.5, 0.9, f'LF/HF = {lf_hf:.2f}',
                         transform=self.ax.transAxes, ha='center',
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        self.ax.grid(True, alpha=0.3, axis='y')
        self.draw()


# ==================== 绘图设置对话框 ====================

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

        self._create_widgets()

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

    def save(self):
        self.result = {
            "title": self.title_entry.get(),
            "xlabel": self.xlabel_entry.get(),
            "ylabel": self.ylabel_entry.get(),
            "width": int(self.width_spin.get()),
            "height": int(self.height_spin.get())
        }
        self.destroy()


# ==================== 外部调用接口 ====================

def show_feature_view(parent, data_dict):
    """
    统一特征可视化入口
    """
    try:
        view = FeatureView(parent, data_dict)  # 这里也用原名
        return view
    except Exception as e:
        messagebox.showerror("错误", f"打开特征视图失败:\n{str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 创建测试数据
    test_data = {
        "meta": {"modality": ["TEST"]},
        "processed": {
            "features": {
                "test_category": {
                    "test_feature": [1, 2, 3, 4]
                }
            }
        },
        "signal": {}
    }

    # 创建视图
    view = FeatureView(root, test_data)
    view.mainloop()