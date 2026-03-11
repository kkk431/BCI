#!/usr/bin/env python3
"""
bar_view.py
Tkinter版本 - 柱状图视图
支持特征对比和Excel数据可视化（支持多模态）
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import pandas as pd
import os
from typing import Dict, List, Optional, Tuple, Any


class BarView(tk.Frame):
    """
    柱状图视图类 - Tkinter版本
    支持特征对比和Excel数据可视化（支持多模态）
    """

    def __init__(self, parent, data_dict: Dict[str, Any] = None, excel_file: str = None, modality: str = None):
        """
        初始化柱状图视图

        Args:
            parent: 父窗口
            data_dict: 包含特征数据的数据字典
            excel_file: Excel文件路径
            modality: 初始模态
        """
        super().__init__(parent)
        self.parent = parent
        self.data_dict = data_dict
        self.excel_file = excel_file
        self.current_modality = modality
        self.df = None
        self.column_names = []

        # 获取所有可用模态
        self.available_modalities = []
        if data_dict and "signal" in data_dict:
            self.available_modalities = list(data_dict["signal"].keys())

        # 创建matplotlib图形
        self.figure = Figure(figsize=(10, 6), dpi=100)

        # 加载数据
        self._load_data()

        # 设置UI
        self.setup_ui()

        # 更新图表
        self.update_plot()

    def _load_data(self):
        """加载数据"""
        if self.excel_file and os.path.exists(self.excel_file):
            try:
                # 从Excel文件加载
                self.df = pd.read_excel(self.excel_file)
                self.column_names = self.df.columns.tolist()
                self.data_source = "excel"
                self.file_name = os.path.basename(self.excel_file)

                messagebox.showinfo("加载成功",
                                    f"已加载Excel文件:\n{self.excel_file}\n"
                                    f"共{len(self.df)}行，{len(self.column_names)}列")

            except Exception as e:
                messagebox.showerror("加载失败", f"无法加载Excel文件:\n{str(e)}")
                self._create_demo_data()

        elif self.data_dict:
            # 从数据字典的特征中加载
            self._load_from_data_dict()

        else:
            self._create_demo_data()

    def _load_from_data_dict(self):
        """从数据字典加载特征数据"""
        # 获取当前模态的通道
        if self.current_modality and self.current_modality in self.data_dict.get("signal", {}):
            signal_info = self.data_dict["signal"][self.current_modality]
            channels = signal_info.get("channel_names", [])
        else:
            # 如果没有指定模态或模态不存在，使用第一个
            signal_dict = self.data_dict.get("signal", {})
            if signal_dict:
                first_mod = list(signal_dict.keys())[0]
                signal_info = signal_dict[first_mod]
                channels = signal_info.get("channel_names", [])
                if self.current_modality is None:
                    self.current_modality = first_mod
            else:
                channels = []

        # 获取特征数据
        processed = self.data_dict.get("processed", {})
        features = processed.get("features", {})

        if features:
            # 提取特征名称
            feature_keys = list(features.keys())
            feature_names = set()

            for key in feature_keys:
                parts = key.split('_', 1)
                if len(parts) == 2:
                    feature_names.add(parts[1])

            feature_names = sorted(list(feature_names))

            # 构建DataFrame
            data = []
            for ch in channels:
                row = [ch]
                for feat in feature_names:
                    key = f"{ch}_{feat}"
                    row.append(features.get(key, np.nan))
                data.append(row)

            self.df = pd.DataFrame(data, columns=['channel'] + feature_names)
            self.column_names = self.df.columns.tolist()
            self.data_source = "features"

            messagebox.showinfo("加载成功",
                                f"已从数据字典加载特征\n"
                                f"模态: {self.current_modality}\n"
                                f"共{len(channels)}个通道，{len(feature_names)}个特征")
        else:
            self._create_demo_data()

    def _create_demo_data(self):
        """创建演示数据"""
        channels = ['Fz', 'Cz', 'Pz', 'Oz', 'F3', 'F4', 'C3', 'C4']
        features = ['mean', 'std', 'alpha_power', 'beta_power', 'theta_power']

        np.random.seed(42)
        data = []
        for ch in channels:
            row = [ch]
            for _ in features:
                row.append(np.random.rand() * 100)
            data.append(row)

        self.df = pd.DataFrame(data, columns=['channel'] + features)
        self.column_names = self.df.columns.tolist()
        self.data_source = "demo"

        messagebox.showinfo("演示模式", "使用演示数据进行展示")

    def setup_ui(self):
        """设置用户界面（添加模态选择）"""
        # 主布局
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ========== 顶部控制栏 ==========
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        # ===== 添加模态选择器 =====
        if len(self.available_modalities) > 1:
            ttk.Label(control_frame, text="模态:").pack(side=tk.LEFT, padx=5)
            self.modality_var = tk.StringVar(value=self.current_modality if self.current_modality else "")
            modality_combo = ttk.Combobox(control_frame, textvariable=self.modality_var,
                                          values=self.available_modalities, state="readonly", width=10)
            modality_combo.pack(side=tk.LEFT, padx=2)
            modality_combo.bind('<<ComboboxSelected>>', self.on_modality_changed)

        # 文件信息
        if hasattr(self, 'file_name'):
            info_text = f"文件: {self.file_name}"
        else:
            info_text = f"数据源: {getattr(self, 'data_source', '未知')} 模态: {self.current_modality}"

        info_label = ttk.Label(control_frame, text=info_text, font=('微软雅黑', 10))
        info_label.pack(side=tk.LEFT, padx=5)

        # 打开文件按钮
        ttk.Button(control_frame, text="打开Excel文件",
                   command=self.open_excel_file).pack(side=tk.RIGHT, padx=5)

        # 刷新按钮
        ttk.Button(control_frame, text="刷新",
                   command=self.update_plot).pack(side=tk.RIGHT, padx=5)

        # ========== 中间：数据选择 + 绘图 ==========
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 左侧控制面板
        control_panel = ttk.LabelFrame(middle_frame, text="设置", width=250)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        control_panel.pack_propagate(False)

        # X轴选择
        x_frame = ttk.LabelFrame(control_panel, text="X轴 (类别)")
        x_frame.pack(fill=tk.X, padx=5, pady=5)

        self.x_var = tk.StringVar()
        self.x_combo = ttk.Combobox(x_frame, textvariable=self.x_var,
                                    values=self.column_names, state="readonly")
        self.x_combo.pack(fill=tk.X, padx=5, pady=5)
        if self.column_names:
            self.x_combo.current(0)
        self.x_combo.bind('<<ComboboxSelected>>', self.on_x_changed)

        # Y轴选择
        y_frame = ttk.LabelFrame(control_panel, text="Y轴 (数值列)")
        y_frame.pack(fill=tk.X, padx=5, pady=5)

        self.y_var = tk.StringVar()
        self.y_combo = ttk.Combobox(y_frame, textvariable=self.y_var,
                                    values=[c for c in self.column_names if c != self.x_var.get()],
                                    state="readonly")
        self.y_combo.pack(fill=tk.X, padx=5, pady=5)
        if len(self.column_names) > 1:
            self.y_combo.current(0)
        self.y_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plot())

        # 图表设置
        plot_frame = ttk.LabelFrame(control_panel, text="图表设置")
        plot_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(plot_frame, text="标题:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.title_var = tk.StringVar(value="Feature Comparison Bar Chart")
        ttk.Entry(plot_frame, textvariable=self.title_var).grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)

        ttk.Label(plot_frame, text="X轴标签:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.xlabel_var = tk.StringVar(value="Channel")
        ttk.Entry(plot_frame, textvariable=self.xlabel_var).grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)

        ttk.Label(plot_frame, text="Y轴标签:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.ylabel_var = tk.StringVar(value="Value")
        ttk.Entry(plot_frame, textvariable=self.ylabel_var).grid(row=2, column=1, padx=5, pady=2, sticky=tk.EW)

        ttk.Label(plot_frame, text="柱子宽度:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.width_var = tk.StringVar(value="0.6")
        width_spin = ttk.Spinbox(plot_frame, from_=0.1, to=1.0, increment=0.1,
                                 textvariable=self.width_var, width=10)
        width_spin.grid(row=3, column=1, padx=5, pady=2, sticky=tk.W)
        width_spin.bind('<Return>', lambda e: self.update_plot())

        plot_frame.columnconfigure(1, weight=1)

        # 显示选项
        display_frame = ttk.LabelFrame(control_panel, text="显示选项")
        display_frame.pack(fill=tk.X, padx=5, pady=5)

        self.show_values_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="显示数值标签",
                        variable=self.show_values_var,
                        command=self.update_plot).pack(anchor=tk.W, padx=5, pady=2)

        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="显示网格",
                        variable=self.show_grid_var,
                        command=self.update_plot).pack(anchor=tk.W, padx=5, pady=2)

        self.rotate_x_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(display_frame, text="旋转X轴标签",
                        variable=self.rotate_x_var,
                        command=self.update_plot).pack(anchor=tk.W, padx=5, pady=2)

        # 右侧：笔记本（绘图 + 数据表格）
        right_notebook = ttk.Notebook(middle_frame)
        right_notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 绘图选项卡
        plot_tab = ttk.Frame(right_notebook)
        right_notebook.add(plot_tab, text="柱状图")

        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.figure, plot_tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 工具栏
        toolbar_frame = ttk.Frame(plot_tab)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # 数据表格选项卡
        table_tab = ttk.Frame(right_notebook)
        right_notebook.add(table_tab, text="数据表格")

        # 创建表格
        self.setup_data_table(table_tab)

    def on_modality_changed(self, event=None):
        """模态切换"""
        new_modality = self.modality_var.get()
        if new_modality != self.current_modality:
            print(f"切换模态: {self.current_modality} -> {new_modality}")
            self.current_modality = new_modality
            self._load_from_data_dict()
            self.update_plot()

    def setup_data_table(self, parent):
        """设置数据表格"""
        columns = self.column_names
        self.tree = ttk.Treeview(parent, columns=columns, show="headings", height=20)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.CENTER)

        v_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        self.populate_table()

    def populate_table(self):
        """填充数据表格"""
        for item in self.tree.get_children():
            self.tree.delete(item)

        if self.df is None:
            return

        for _, row in self.df.iterrows():
            values = []
            for col in self.column_names:
                val = row[col]
                if isinstance(val, (int, float)):
                    if abs(val) < 1000:
                        values.append(f"{val:.4f}")
                    else:
                        values.append(f"{val:.2e}")
                else:
                    values.append(str(val))
            self.tree.insert("", tk.END, values=values)

    def on_x_changed(self, event=None):
        """X轴列改变"""
        x_col = self.x_var.get()
        y_options = [col for col in self.column_names if col != x_col]
        self.y_combo['values'] = y_options
        if y_options:
            self.y_combo.current(0)
        self.update_plot()

    def open_excel_file(self):
        """打开Excel文件"""
        file_path = filedialog.askopenfilename(
            title="选择Excel文件",
            filetypes=[("Excel文件", "*.xlsx *.xls"), ("所有文件", "*.*")]
        )

        if file_path:
            self.excel_file = file_path
            self._load_data()

            self.x_combo['values'] = self.column_names
            if self.column_names:
                self.x_combo.current(0)
                self.on_x_changed()
            self.populate_table()
            self.update_plot()

    def update_plot(self, event=None):
        """更新柱状图"""
        self.figure.clear()

        x_col = self.x_var.get()
        y_col = self.y_var.get()

        if not x_col or not y_col or x_col == y_col or self.df is None:
            return

        x_categories = self.df[x_col].tolist()
        y_values = self.df[y_col].tolist()

        if not all(isinstance(v, (int, float)) for v in y_values):
            try:
                y_values = [float(v) if v else 0 for v in y_values]
            except:
                messagebox.showerror("错误", f"列 '{y_col}' 包含非数值数据")
                return

        ax = self.figure.add_subplot(111)
        x_pos = np.arange(len(x_categories))

        try:
            width = float(self.width_var.get())
        except:
            width = 0.6

        bars = ax.bar(x_pos, y_values, width=width,
                      color='steelblue', edgecolor='black', alpha=0.7)

        ax.set_xticks(x_pos)
        rotation = 45 if self.rotate_x_var.get() else 0
        ax.set_xticklabels(x_categories, rotation=rotation)

        ax.set_xlabel(self.xlabel_var.get(), fontsize=12)
        ax.set_ylabel(self.ylabel_var.get(), fontsize=12)
        ax.set_title(self.title_var.get(), fontsize=14)

        if self.show_grid_var.get():
            ax.grid(True, alpha=0.3, axis='y')

        if self.show_values_var.get():
            for i, (bar, val) in enumerate(zip(bars, y_values)):
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                offset = 3 if height >= 0 else -3
                ax.text(bar.get_x() + bar.get_width() / 2., height + offset,
                        f'{val:.2f}', ha='center', va=va, fontsize=8,
                        rotation=90 if len(x_categories) > 8 else 0)

        self.figure.tight_layout()
        self.canvas.draw()

    def save_plot(self):
        """保存图表"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG图像", "*.png"), ("PDF文件", "*.pdf"), ("SVG图像", "*.svg")]
        )

        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("保存成功", f"图表已保存到:\n{file_path}")
            except Exception as e:
                messagebox.showerror("保存失败", f"保存图表时出错:\n{str(e)}")

    def destroy(self):
        """销毁时清理"""
        plt.close(self.figure)
        super().destroy()


# 测试代码
if __name__ == "__main__":
    import sys

    root = tk.Tk()
    root.title("柱状图视图测试 - 支持多模态")
    root.geometry("1200x800")

    # 测试数据
    test_data = {
        "signal": {
            "EEG": {
                "channel_names": [f"EEG_{i}" for i in range(16)]
            },
            "EMG": {
                "channel_names": [f"EMG_{i}" for i in range(8)]
            }
        },
        "processed": {
            "features": {
                f"EEG_{i}_mean": np.random.rand() * 10 for i in range(16)
            }
        }
    }
    for i in range(16):
        test_data["processed"]["features"][f"EEG_{i}_std"] = np.random.rand() * 5
        test_data["processed"]["features"][f"EEG_{i}_alpha"] = np.random.rand() * 20

    view = BarView(root, data_dict=test_data, modality="EEG")
    view.pack(fill=tk.BOTH, expand=True)

    root.mainloop()