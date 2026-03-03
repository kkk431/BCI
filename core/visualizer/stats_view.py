#!/usr/bin/env python3
"""
stats_view.py
Tkinter版本 - 统计分析视图
箱线图/ROC曲线/混淆矩阵
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置 matplotlib 全局字体为英文字体
plt.rcParams['font.family'] = 'DejaVu Sans'  # 或 'Arial', 'Helvetica'
plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号


class StatsView(tk.Frame):
    """
    统计分析视图类 - Tkinter版本
    显示统计模块的输出结果
    """

    def __init__(self, parent, data_dict: Dict[str, Any] = None, stats_file: str = None):
        """
        初始化统计视图

        Args:
            parent: 父窗口
            data_dict: 包含统计结果的四层数据字典
            stats_file: 直接加载统计结果文件
        """
        super().__init__(parent)
        self.parent = parent
        self.data_dict = data_dict
        self.stats_file = stats_file
        self.stats_results = {}

        # 创建matplotlib图形
        self.boxplot_figure = Figure(figsize=(8, 6), dpi=100)
        self.roc_figure = Figure(figsize=(8, 6), dpi=100)
        self.cm_figure = Figure(figsize=(6, 6), dpi=100)

        # 加载数据
        self._load_data()

        # 设置UI
        self.setup_ui()

        # 更新显示
        self.update_plots()

    def _load_data(self):
        """加载统计数据"""
        if self.stats_file:
            # 从文件加载
            try:
                if self.stats_file.endswith('.json'):
                    with open(self.stats_file, 'r', encoding='utf-8') as f:
                        self.stats_results = json.load(f)
                elif self.stats_file.endswith(('.pkl', '.pickle')):
                    import pickle
                    with open(self.stats_file, 'rb') as f:
                        self.stats_results = pickle.load(f)
                elif self.stats_file.endswith('.csv'):
                    df = pd.read_csv(self.stats_file)
                    self.stats_results = df.to_dict(orient='list')
            except Exception as e:
                messagebox.showerror("加载失败", f"无法加载统计文件: {str(e)}")
                self.stats_results = self._create_demo_data()

        elif self.data_dict:
            # 从数据字典加载
            processed = self.data_dict.get("processed", {})
            self.stats_results = processed.get("stat_results", {})

        # 如果没有数据，创建演示数据
        if not self.stats_results:
            self.stats_results = self._create_demo_data()

    def _create_demo_data(self) -> Dict:
        """创建演示数据"""
        np.random.seed(42)

        # 箱线图数据 - 三组对比
        group1 = np.random.normal(10, 2, 30)
        group2 = np.random.normal(12, 2.5, 30)
        group3 = np.random.normal(9, 1.8, 30)

        # ROC数据
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = np.concatenate([np.random.normal(0.3, 0.2, 50),
                                  np.random.normal(0.7, 0.2, 50)])

        # 混淆矩阵
        y_pred = (y_score > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        return {
            "boxplot": {
                "data": [group1.tolist(), group2.tolist(), group3.tolist()],
                "labels": ["Left Hand", "Right Hand", "Foot"],
                "colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
                "title": "EEG Power by Motor Imagery Task"
            },
            "roc": {
                "y_true": y_true.tolist(),
                "y_score": y_score.tolist(),
                "title": "分类器性能 (左手 vs 右手)",
                "auc": 0.85
            },
            "confusion_matrix": {
                "matrix": cm.tolist(),
                "labels": ["左手", "右手"],
                "title": "混淆矩阵"
            },
            "statistics": {
                "t_test": {
                    "t_statistic": 3.24,
                    "p_value": 0.002,
                    "significant": True
                },
                "anova": {
                    "f_statistic": 5.67,
                    "p_value": 0.004,
                    "significant": True
                }
            }
        }

    def setup_ui(self):
        """设置用户界面"""
        # 主布局
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ========== 顶部控制栏 ==========
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="分析类型:").pack(side=tk.LEFT, padx=5)

        self.plot_type_var = tk.StringVar(value="箱线图")
        plot_type_combo = ttk.Combobox(control_frame, textvariable=self.plot_type_var,
                                       values=["箱线图", "ROC曲线", "混淆矩阵", "所有"],
                                       state="readonly", width=15)
        plot_type_combo.pack(side=tk.LEFT, padx=5)
        plot_type_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plots())

        # 显著性水平
        ttk.Label(control_frame, text="显著性水平 α:").pack(side=tk.LEFT, padx=(20, 5))

        self.alpha_var = tk.StringVar(value="0.05")
        alpha_spin = ttk.Spinbox(control_frame, from_=0.001, to=0.1, increment=0.005,
                                 textvariable=self.alpha_var, width=8)
        alpha_spin.pack(side=tk.LEFT, padx=5)
        alpha_spin.bind('<Return>', lambda e: self.update_plots())

        # 保存按钮
        ttk.Button(control_frame, text="保存图表",
                   command=self.save_plots).pack(side=tk.RIGHT, padx=5)

        # ========== 中间：选项卡 ==========
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # 箱线图选项卡
        self.boxplot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.boxplot_frame, text="箱线图")
        self.setup_boxplot_tab()

        # ROC曲线选项卡
        self.roc_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.roc_frame, text="ROC曲线")
        self.setup_roc_tab()

        # 混淆矩阵选项卡
        self.cm_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.cm_frame, text="混淆矩阵")
        self.setup_cm_tab()

        # 统计表格选项卡
        self.table_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.table_frame, text="统计结果")
        self.setup_stats_table_tab()

    def setup_boxplot_tab(self):
        """设置箱线图选项卡"""
        # 左侧控制面板
        control_frame = ttk.LabelFrame(self.boxplot_frame, text="显示选项", width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)

        # 大约在第180行附近
        self.show_points_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="显示数据点",
                        variable=self.show_points_var,
                        command=self.update_plots).pack(anchor=tk.W, padx=5, pady=5)

        self.show_stats_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="显示统计标记",
                        variable=self.show_stats_var,
                        command=self.update_plots).pack(anchor=tk.W, padx=5, pady=5)

        # 右侧绘图区域
        plot_frame = ttk.Frame(self.boxplot_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建画布
        self.boxplot_canvas = FigureCanvasTkAgg(self.boxplot_figure, plot_frame)
        self.boxplot_canvas.draw()
        self.boxplot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 工具栏
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        self.boxplot_toolbar = NavigationToolbar2Tk(self.boxplot_canvas, toolbar_frame)
        self.boxplot_toolbar.update()

    def setup_roc_tab(self):
        """设置ROC曲线选项卡"""
        # 左侧控制面板
        control_frame = ttk.LabelFrame(self.roc_frame, text="显示选项", width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)

        self.show_auc_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="显示AUC值",
                        variable=self.show_auc_var,
                        command=self.update_plots).pack(anchor=tk.W, padx=5, pady=5)

        self.show_diag_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="显示对角线",
                        variable=self.show_diag_var,
                        command=self.update_plots).pack(anchor=tk.W, padx=5, pady=5)

        # 右侧绘图区域
        plot_frame = ttk.Frame(self.roc_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建画布
        self.roc_canvas = FigureCanvasTkAgg(self.roc_figure, plot_frame)
        self.roc_canvas.draw()
        self.roc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 工具栏
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        self.roc_toolbar = NavigationToolbar2Tk(self.roc_canvas, toolbar_frame)
        self.roc_toolbar.update()

    def setup_cm_tab(self):
        """设置混淆矩阵选项卡"""
        # 左侧控制面板
        control_frame = ttk.LabelFrame(self.cm_frame, text="显示选项", width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)

        self.show_numbers_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="显示数值",
                        variable=self.show_numbers_var,
                        command=self.update_plots).pack(anchor=tk.W, padx=5, pady=5)

        self.show_percent_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="显示百分比",
                        variable=self.show_percent_var,
                        command=self.update_plots).pack(anchor=tk.W, padx=5, pady=5)

        ttk.Label(control_frame, text="颜色映射:").pack(anchor=tk.W, padx=5, pady=(10, 2))

        self.cmap_var = tk.StringVar(value="Blues")
        cmap_combo = ttk.Combobox(control_frame, textvariable=self.cmap_var,
                                  values=['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'viridis'],
                                  state="readonly", width=15)
        cmap_combo.pack(anchor=tk.W, padx=5, pady=2)
        cmap_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plots())

        # 右侧绘图区域
        plot_frame = ttk.Frame(self.cm_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建画布
        self.cm_canvas = FigureCanvasTkAgg(self.cm_figure, plot_frame)
        self.cm_canvas.draw()
        self.cm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 工具栏
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        self.cm_toolbar = NavigationToolbar2Tk(self.cm_canvas, toolbar_frame)
        self.cm_toolbar.update()

    def setup_stats_table_tab(self):
        """设置统计结果表格选项卡"""
        # 创建表格
        columns = ("统计量", "数值", "自由度", "p值", "显著性")
        self.stats_tree = ttk.Treeview(self.table_frame, columns=columns, show="headings", height=15)

        # 设置列标题
        for col in columns:
            self.stats_tree.heading(col, text=col)
            if col == "统计量":
                self.stats_tree.column(col, width=150)
            else:
                self.stats_tree.column(col, width=100)

        # 滚动条
        scrollbar = ttk.Scrollbar(self.table_frame, orient=tk.VERTICAL, command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=scrollbar.set)

        self.stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 填充表格
        self.populate_stats_table()

    def populate_stats_table(self):
        """填充统计结果表格"""
        # 清空现有内容
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)

        stats_data = self.stats_results.get("statistics", {})

        for test_name, test_result in stats_data.items():
            if isinstance(test_result, dict):
                if "t_statistic" in test_result:
                    values = (
                        test_name,
                        f"{test_result['t_statistic']:.4f}",
                        str(test_result.get('df', 'N/A')),
                        f"{test_result['p_value']:.4f}",
                        "★" if test_result.get('significant', False) else ""
                    )
                    self.stats_tree.insert("", tk.END, values=values)

                elif "f_statistic" in test_result:
                    df_str = f"{test_result.get('df1', 'N/A')}, {test_result.get('df2', 'N/A')}"
                    values = (
                        test_name,
                        f"{test_result['f_statistic']:.4f}",
                        df_str,
                        f"{test_result['p_value']:.4f}",
                        "★" if test_result.get('significant', False) else ""
                    )
                    self.stats_tree.insert("", tk.END, values=values)

    def update_plots(self):
        """更新所有图表"""
        plot_type = self.plot_type_var.get()

        if plot_type in ["箱线图", "所有"]:
            self.update_boxplot()
        if plot_type in ["ROC曲线", "所有"]:
            self.update_roc()
        if plot_type in ["混淆矩阵", "所有"]:
            self.update_cm()

        # 更新选项卡
        if plot_type == "箱线图":
            self.notebook.select(0)
        elif plot_type == "ROC曲线":
            self.notebook.select(1)
        elif plot_type == "混淆矩阵":
            self.notebook.select(2)

    def update_boxplot(self):
        """更新箱线图"""
        self.boxplot_figure.clear()
        ax = self.boxplot_figure.add_subplot(111)

        # 获取数据
        boxplot_data = self.stats_results.get("boxplot", {})
        data = boxplot_data.get("data", [])
        labels = boxplot_data.get("labels", [])
        colors = boxplot_data.get("colors", ["#1f77b4"] * len(data))
        title = boxplot_data.get("title", "箱线图")

        if not data:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', transform=ax.transAxes)
            self.boxplot_canvas.draw()
            return

        # 绘制箱线图
        bp = ax.boxplot(data, patch_artist=True, labels=labels, showfliers=False)

        # 设置颜色
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 显示数据点
        if self.show_points_var.get():
            for i, d in enumerate(data):
                x = np.random.normal(i + 1, 0.04, size=len(d))
                ax.plot(x, d, 'o', color='black', alpha=0.5, markersize=3)

        # 显示统计标记
        if self.show_stats_var.get() and len(data) >= 2:
            try:
                alpha = float(self.alpha_var.get())

                # 进行t检验
                for i in range(len(data)):
                    for j in range(i + 1, len(data)):
                        t_stat, p_val = stats.ttest_ind(data[i], data[j])

                        if p_val < alpha:
                            # 在两组之间添加显著性标记
                            y_max = max(np.max(data[i]), np.max(data[j]))
                            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                            y_pos = y_max + y_range * 0.1

                            x1, x2 = i + 1, j + 1
                            ax.plot([x1, x1, x2, x2], [y_pos, y_pos + y_range * 0.03,
                                                       y_pos + y_range * 0.03, y_pos],
                                    'k-', linewidth=1)

                            # 添加星号
                            stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                            ax.text((x1 + x2) / 2, y_pos + y_range * 0.04, stars,
                                    ha='center', va='bottom', fontsize=12, fontweight='bold', family='DejaVu Sans')
            except:
                pass

        ax.set_title(title, fontsize=14)
        ax.set_ylabel("值")
        ax.grid(True, alpha=0.3, axis='y')

        self.boxplot_figure.tight_layout()
        self.boxplot_canvas.draw()

    def update_roc(self):
        """更新ROC曲线"""
        self.roc_figure.clear()
        ax = self.roc_figure.add_subplot(111)

        # 获取数据
        roc_data = self.stats_results.get("roc", {})
        y_true = np.array(roc_data.get("y_true", []))
        y_score = np.array(roc_data.get("y_score", []))
        title = roc_data.get("title", "ROC曲线")

        if len(y_true) == 0 or len(y_score) == 0:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', transform=ax.transAxes)
            self.roc_canvas.draw()
            return

        # 计算ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # 绘制ROC曲线
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')

        # 显示对角线
        if self.show_diag_var.get():
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='随机分类器')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('假阳性率 (FPR)', fontsize=12)
        ax.set_ylabel('真阳性率 (TPR)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        self.roc_figure.tight_layout()
        self.roc_canvas.draw()

    def update_cm(self):
        """更新混淆矩阵"""
        self.cm_figure.clear()
        ax = self.cm_figure.add_subplot(111)

        # 获取数据
        cm_data = self.stats_results.get("confusion_matrix", {})
        cm = np.array(cm_data.get("matrix", []))
        labels = cm_data.get("labels", ["类别1", "类别2"])
        title = cm_data.get("title", "混淆矩阵")

        if cm.size == 0:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', transform=ax.transAxes)
            self.cm_canvas.draw()
            return

        # 绘制混淆矩阵
        cmap = plt.cm.get_cmap(self.cmap_var.get())
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

        # 添加颜色条
        self.cm_figure.colorbar(im, ax=ax)

        # 设置刻度
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # 添加数值
        if self.show_numbers_var.get() or self.show_percent_var.get():
            total = np.sum(cm)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if self.show_percent_var.get():
                        text = f"{cm[i, j]}\n({cm[i, j] / total * 100:.1f}%)"
                    else:
                        text = str(cm[i, j])

                    color = "white" if cm[i, j] > cm.max() / 2 else "black"
                    ax.text(j, i, text, ha="center", va="center", color=color)

        ax.set_xlabel('预测标签', fontsize=12)
        ax.set_ylabel('真实标签', fontsize=12)
        ax.set_title(title, fontsize=14)

        self.cm_figure.tight_layout()
        self.cm_canvas.draw()

    def save_plots(self):
        """保存图表"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG图像", "*.png"), ("PDF文件", "*.pdf"), ("SVG图像", "*.svg")]
        )

        if not file_path:
            return

        try:
            # 保存当前显示的图表
            current_tab = self.notebook.index(self.notebook.select())
            if current_tab == 0:
                self.boxplot_figure.savefig(file_path, dpi=300, bbox_inches='tight')
            elif current_tab == 1:
                self.roc_figure.savefig(file_path, dpi=300, bbox_inches='tight')
            elif current_tab == 2:
                self.cm_figure.savefig(file_path, dpi=300, bbox_inches='tight')

            messagebox.showinfo("保存成功", f"图表已保存到:\n{file_path}")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存图表时出错:\n{str(e)}")

    def destroy(self):
        """销毁时清理"""
        plt.close(self.boxplot_figure)
        plt.close(self.roc_figure)
        plt.close(self.cm_figure)
        super().destroy()


# 测试代码
if __name__ == "__main__":
    import sys

    root = tk.Tk()
    root.title("统计视图测试")
    root.geometry("1100x800")

    view = StatsView(root)
    view.pack(fill=tk.BOTH, expand=True)

    root.mainloop()