"""
statistical_analysis_view.py
真正的统计分析可视化视图 - 展示效应量、假设检验、信度分析等结果
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from scipy import stats


class StatisticalAnalysisView(tk.Frame):
    """
    统计分析结果可视化视图
    展示：效应量森林图、p值分布、信度分析、功效曲线等
    """

    def __init__(self, parent, stats_results=None):
        super().__init__(parent)
        self.parent = parent
        self.stats_results = stats_results or {}

        # 创建图形
        self.effect_size_figure = Figure(figsize=(8, 6), dpi=100)
        self.p_value_figure = Figure(figsize=(8, 6), dpi=100)
        self.reliability_figure = Figure(figsize=(8, 6), dpi=100)
        self.power_figure = Figure(figsize=(8, 6), dpi=100)

        self.setup_ui()
        self.update_all_plots()

    def setup_ui(self):
        """设置用户界面"""
        # 主布局
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 顶部控制栏
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="分析类型:").pack(side=tk.LEFT, padx=5)

        self.plot_type_var = tk.StringVar(value="效应量")
        plot_combo = ttk.Combobox(control_frame, textvariable=self.plot_type_var,
                                  values=["效应量", "p值分布", "信度分析", "功效分析", "元分析"],
                                  state="readonly", width=15)
        plot_combo.pack(side=tk.LEFT, padx=5)
        plot_combo.bind('<<ComboboxSelected>>', self.on_plot_type_changed)

        # 显著性水平
        ttk.Label(control_frame, text="α:").pack(side=tk.LEFT, padx=(20, 5))
        self.alpha_var = tk.StringVar(value="0.05")
        alpha_spin = ttk.Spinbox(control_frame, from_=0.001, to=0.1,
                                 textvariable=self.alpha_var, width=8)
        alpha_spin.pack(side=tk.LEFT, padx=5)

        # 保存按钮
        ttk.Button(control_frame, text="保存图表",
                   command=self.save_current_plot).pack(side=tk.RIGHT, padx=5)

        # 创建Notebook
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 效应量选项卡
        self.effect_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.effect_frame, text="效应量分析")
        self.setup_effect_tab()

        # p值分布选项卡
        self.pvalue_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pvalue_frame, text="p值分布")
        self.setup_pvalue_tab()

        # 信度分析选项卡
        self.reliability_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.reliability_frame, text="信度分析")
        self.setup_reliability_tab()

        # 功效分析选项卡
        self.power_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.power_frame, text="功效分析")
        self.setup_power_tab()

    def setup_effect_tab(self):
        """设置效应量分析选项卡"""
        # 左侧控制面板
        control_frame = ttk.LabelFrame(self.effect_frame, text="显示选项", width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)

        self.show_ci_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="显示置信区间",
                        variable=self.show_ci_var,
                        command=self.update_effect_plot).pack(anchor=tk.W, padx=5, pady=5)

        self.sort_effects_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="按效应量排序",
                        variable=self.sort_effects_var,
                        command=self.update_effect_plot).pack(anchor=tk.W, padx=5, pady=5)

        # 右侧绘图区域
        plot_frame = ttk.Frame(self.effect_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.effect_canvas = FigureCanvasTkAgg(self.effect_size_figure, plot_frame)
        self.effect_canvas.draw()
        self.effect_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 工具栏
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        self.effect_toolbar = NavigationToolbar2Tk(self.effect_canvas, toolbar_frame)
        self.effect_toolbar.update()

    def setup_pvalue_tab(self):
        """设置p值分布选项卡"""
        plot_frame = ttk.Frame(self.pvalue_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.pvalue_canvas = FigureCanvasTkAgg(self.p_value_figure, plot_frame)
        self.pvalue_canvas.draw()
        self.pvalue_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        self.pvalue_toolbar = NavigationToolbar2Tk(self.pvalue_canvas, toolbar_frame)
        self.pvalue_toolbar.update()

    def setup_reliability_tab(self):
        """设置信度分析选项卡"""
        plot_frame = ttk.Frame(self.reliability_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.reliability_canvas = FigureCanvasTkAgg(self.reliability_figure, plot_frame)
        self.reliability_canvas.draw()
        self.reliability_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        self.reliability_toolbar = NavigationToolbar2Tk(self.reliability_canvas, toolbar_frame)
        self.reliability_toolbar.update()

    def setup_power_tab(self):
        """设置功效分析选项卡"""
        # 左侧控制面板
        control_frame = ttk.LabelFrame(self.power_frame, text="参数设置", width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)

        ttk.Label(control_frame, text="效应量:").pack(anchor=tk.W, padx=5, pady=2)
        self.effect_size_entry = ttk.Entry(control_frame, width=15)
        self.effect_size_entry.insert(0, "0.5")
        self.effect_size_entry.pack(anchor=tk.W, padx=5, pady=2)

        ttk.Label(control_frame, text="样本量范围:").pack(anchor=tk.W, padx=5, pady=2)
        self.n_from_entry = ttk.Entry(control_frame, width=8)
        self.n_from_entry.insert(0, "10")
        self.n_from_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, text="to").pack(side=tk.LEFT)
        self.n_to_entry = ttk.Entry(control_frame, width=8)
        self.n_to_entry.insert(0, "100")
        self.n_to_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="更新曲线",
                   command=self.update_power_curve).pack(anchor=tk.W, padx=5, pady=10)

        # 右侧绘图区域
        plot_frame = ttk.Frame(self.power_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.power_canvas = FigureCanvasTkAgg(self.power_figure, plot_frame)
        self.power_canvas.draw()
        self.power_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        self.power_toolbar = NavigationToolbar2Tk(self.power_canvas, toolbar_frame)
        self.power_toolbar.update()

    def on_plot_type_changed(self, event=None):
        """切换选项卡"""
        plot_type = self.plot_type_var.get()
        tab_map = {
            "效应量": 0,
            "p值分布": 1,
            "信度分析": 2,
            "功效分析": 3
        }
        if plot_type in tab_map:
            self.notebook.select(tab_map[plot_type])

    def update_all_plots(self):
        """更新所有图表"""
        self.update_effect_plot()
        self.update_pvalue_plot()
        self.update_reliability_plot()
        self.update_power_curve()

    def update_effect_plot(self):
        """更新效应量森林图"""
        self.effect_size_figure.clear()
        ax = self.effect_size_figure.add_subplot(111)

        # 从stats_results中提取效应量数据
        effect_sizes = []
        effect_names = []
        cis = []

        # 示例：从advanced_stats结果中提取
        if "effect_sizes" in self.stats_results:
            for name, data in self.stats_results["effect_sizes"].items():
                effect_sizes.append(data.get("effect_size", 0))
                effect_names.append(name)
                cis.append(data.get("ci", (0, 0)))
        else:
            # 演示数据
            np.random.seed(42)
            effect_sizes = [0.8, 0.5, 0.3, 0.6, 0.2]
            effect_names = ["Cohens d", "Hedges g", "Glass Δ", "Pearson r", "Cramers V"]
            cis = [(0.5, 1.1), (0.3, 0.7), (0.1, 0.5), (0.4, 0.8), (-0.1, 0.5)]

        # 排序
        if self.sort_effects_var.get():
            sorted_idx = np.argsort(effect_sizes)
            effect_sizes = [effect_sizes[i] for i in sorted_idx]
            effect_names = [effect_names[i] for i in sorted_idx]
            cis = [cis[i] for i in sorted_idx]

        y_pos = np.arange(len(effect_sizes))

        # 绘制点估计
        ax.scatter(effect_sizes, y_pos, s=100, c='red', zorder=3)

        # 绘制置信区间
        if self.show_ci_var.get():
            for i, (ci, size) in enumerate(zip(cis, effect_sizes)):
                ax.plot([ci[0], ci[1]], [i, i], 'b-', linewidth=2, zorder=2)

        # 垂直线（无效应）
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

        # 添加效应量解释区域
        ax.axvspan(0.2, 0.5, alpha=0.1, color='yellow', label='小效应')
        ax.axvspan(0.5, 0.8, alpha=0.1, color='orange', label='中等效应')
        ax.axvspan(0.8, max(effect_sizes + [1.0]), alpha=0.1, color='red', label='大效应')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(effect_names)
        ax.set_xlabel('效应量', fontsize=12)
        ax.set_title('效应量森林图', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')

        self.effect_size_figure.tight_layout()
        self.effect_canvas.draw()

    def update_pvalue_plot(self):
        """更新p值分布图"""
        self.p_value_figure.clear()

        # 创建两个子图
        ax1 = self.p_value_figure.add_subplot(121)
        ax2 = self.p_value_figure.add_subplot(122)

        # 提取p值
        p_values = []
        if "p_values" in self.stats_results:
            p_values = self.stats_results["p_values"]
        else:
            # 演示数据
            np.random.seed(42)
            p_values = np.random.uniform(0, 1, 100)
            # 添加一些显著的结果
            p_values[:20] = np.random.uniform(0, 0.05, 20)

        # 直方图
        ax1.hist(p_values, bins=20, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0.05, color='red', linestyle='--', label='α=0.05')
        ax1.set_xlabel('p值')
        ax1.set_ylabel('频数')
        ax1.set_title('p值分布直方图')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q图
        stats.probplot(p_values, dist="uniform", plot=ax2)
        ax2.set_title('p值Q-Q图')
        ax2.grid(True, alpha=0.3)

        self.p_value_figure.tight_layout()
        self.pvalue_canvas.draw()

    def update_reliability_plot(self):
        """更新信度分析图"""
        self.reliability_figure.clear()

        # 创建Bland-Altman图（用于重测信度）
        ax = self.reliability_figure.add_subplot(111)

        if "reliability" in self.stats_results:
            data = self.stats_results["reliability"]
        else:
            # 演示数据
            np.random.seed(42)
            n = 50
            test1 = np.random.normal(100, 15, n)
            test2 = test1 + np.random.normal(0, 5, n)
            data = {"test1": test1, "test2": test2}

        if "test1" in data and "test2" in data:
            test1, test2 = data["test1"], data["test2"]
            mean = (test1 + test2) / 2
            diff = test1 - test2
            mean_diff = np.mean(diff)
            std_diff = np.std(diff, ddof=1)

            # 绘制散点
            ax.scatter(mean, diff, alpha=0.6)

            # 绘制一致性界限
            ax.axhline(y=mean_diff, color='red', linestyle='-', label=f'均值差: {mean_diff:.2f}')
            ax.axhline(y=mean_diff + 1.96 * std_diff, color='blue', linestyle='--',
                       label=f'+1.96 SD: {mean_diff + 1.96 * std_diff:.2f}')
            ax.axhline(y=mean_diff - 1.96 * std_diff, color='blue', linestyle='--',
                       label=f'-1.96 SD: {mean_diff - 1.96 * std_diff:.2f}')

            ax.set_xlabel('均值', fontsize=12)
            ax.set_ylabel('差值', fontsize=12)
            ax.set_title('Bland-Altman图 (重测信度)', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

        self.reliability_figure.tight_layout()
        self.reliability_canvas.draw()

    def update_power_curve(self):
        """更新功效曲线"""
        self.power_figure.clear()
        ax = self.power_figure.add_subplot(111)

        try:
            effect_size = float(self.effect_size_entry.get())
            n_from = int(self.n_from_entry.get())
            n_to = int(self.n_to_entry.get())

            # 计算不同样本量下的功效
            n_range = np.arange(n_from, n_to + 1, max(1, (n_to - n_from) // 20))
            powers = []

            for n in n_range:
                # 简化的功效计算（t检验）
                n_per_group = n // 2
                if n_per_group >= 2:
                    delta = effect_size * np.sqrt(n_per_group / 2)
                    # 使用非中心t分布计算功效
                    critical_t = stats.t.ppf(1 - 0.05 / 2, 2 * n_per_group - 2)
                    power = 1 - stats.nct.cdf(critical_t, 2 * n_per_group - 2, delta)
                    powers.append(power)
                else:
                    powers.append(np.nan)

            # 绘制功效曲线
            ax.plot(n_range, powers, 'b-', linewidth=2, label=f'效应量 d={effect_size}')

            # 添加参考线
            ax.axhline(y=0.8, color='red', linestyle='--', label='目标功效=0.8')

            # 找出达到0.8功效的最小样本量
            valid_idx = np.where(np.array(powers) >= 0.8)[0]
            if len(valid_idx) > 0:
                min_n = n_range[valid_idx[0]]
                ax.axvline(x=min_n, color='green', linestyle=':',
                           label=f'最小样本量: {min_n}')

            ax.set_xlabel('总样本量', fontsize=12)
            ax.set_ylabel('统计功效 (1-β)', fontsize=12)
            ax.set_title('功效曲线', fontsize=14)
            ax.set_ylim([0, 1])
            ax.set_xlim([n_from, n_to])
            ax.legend()
            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f'计算错误: {str(e)}',
                    ha='center', va='center', transform=ax.transAxes)

        self.power_figure.tight_layout()
        self.power_canvas.draw()

    def save_current_plot(self):
        """保存当前图表"""
        from tkinter import filedialog

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG图像", "*.png"), ("PDF文件", "*.pdf"), ("SVG图像", "*.svg")]
        )

        if file_path:
            current_tab = self.notebook.index(self.notebook.select())
            figures = [self.effect_size_figure, self.p_value_figure,
                       self.reliability_figure, self.power_figure]
            try:
                figures[current_tab].savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("成功", f"图表已保存到:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败:\n{str(e)}")


# 使用示例
if __name__ == "__main__":
    root = tk.Tk()
    root.title("统计分析视图")
    root.geometry("1200x800")

    # 示例数据
    sample_results = {
        "effect_sizes": {
            "t-test (Group A vs B)": {"effect_size": 0.8, "ci": (0.5, 1.1)},
            "ANOVA (F=5.67)": {"effect_size": 0.5, "ci": (0.3, 0.7)},
            "Correlation": {"effect_size": 0.3, "ci": (0.1, 0.5)},
            "ICC": {"effect_size": 0.6, "ci": (0.4, 0.8)},
            "Cohen's f²": {"effect_size": 0.2, "ci": (-0.1, 0.5)}
        },
        "p_values": np.random.uniform(0, 1, 100),
        "reliability": {
            "test1": np.random.normal(100, 15, 50),
            "test2": np.random.normal(102, 16, 50)
        }
    }

    view = StatisticalAnalysisView(root, sample_results)
    view.pack(fill=tk.BOTH, expand=True)

    root.mainloop()