# -*- coding: utf-8 -*-
"""
智融脑机 - 统计分析模块 (tkinter版)
适配 main_UI.py 的 Canvas 图片版统计分析面板
完全复刻原 analysis_panel.py (PyQt5版) 的界面布局
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

# 动态添加项目根目录到 sys.path
start_path = Path(__file__).resolve().parent
for parent in [start_path] + list(start_path.parents):
    if parent.name == 'core':
        project_root = parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        break
else:
    raise RuntimeError("未找到名为 'core' 的目录")

# 导入统计核心代码
from core.processing.Statistical_Analysis.significance_test import (
    calculate_significance,
    multiple_comparison_correction
)

# PIL 用于加载图片
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("警告: PIL 未安装，无法加载 PNG 图片，请安装 Pillow 库。")


# ========== Excel 读取函数 ==========
def read_xlsx(file_path):
    try:
        df = pd.read_excel(file_path, engine='openpyxl', sheet_name=0)
        channel_col = df.columns[0]
        ch_names = df[channel_col].astype(str).tolist()
        feature_dict = {}
        for col in df.columns[1:]:
            numeric_vals = pd.to_numeric(df[col], errors='coerce')
            vals = numeric_vals.tolist()
            feature_dict[col] = [[v] for v in vals]
        if not feature_dict:
            raise ValueError(f"文件 {file_path} 中未找到任何数值列。")
        return {
            'feature': feature_dict,
            'ch_names': ch_names
        }
    except Exception as e:
        raise ValueError(f"无法读取文件 {file_path}: {str(e)}")


def get_feature(feature_dict, channel_name=None, feature_name=None):
    if feature_name not in feature_dict["feature"]:
        raise ValueError(f"Feature '{feature_name}' not found")
    has_channels = "ch_names" in feature_dict and feature_dict["ch_names"]
    if has_channels and channel_name is None:
        raise ValueError("Data has channel information. Please select a specific channel.")
    if not has_channels and channel_name is not None:
        channel_name = None
    feature_data = feature_dict["feature"][feature_name]
    if channel_name is not None and has_channels:
        if channel_name not in feature_dict["ch_names"]:
            raise ValueError(f"Channel '{channel_name}' not found")
        channel_index = feature_dict["ch_names"].index(channel_name)
        data = feature_data[channel_index]
    else:
        data = feature_data
    if isinstance(data, (np.ndarray, list)):
        data = np.asarray(data).ravel().tolist()
    else:
        data = [data]
    numeric_data = []
    for val in data:
        try:
            numeric_data.append(float(val))
        except (ValueError, TypeError):
            numeric_data.append(np.nan)
    return numeric_data


# ========== 结果展示对话框 ==========
class ResultDialogTk(tk.Toplevel):
    """统计分析结果弹窗 - 与原版 ResultDialog 一致"""

    def __init__(self, parent, results):
        super().__init__(parent)
        self.title("Statistical Analysis Results")
        self.geometry("1000x600")
        self.configure(bg="#FFFFFF")
        self.results = results
        self.transient(parent)
        self.grab_set()
        self._init_ui()

    def _init_ui(self):
        # 标题
        title = tk.Label(self, text="Pairwise Group Comparison Results",
                         font=("Times New Roman", 18, "bold"), bg="#FFFFFF", fg="#333")
        title.pack(pady=(15, 10))

        # 表格
        columns = ("comparison", "method", "statistic", "p_value", "corrected_p", "significant")
        tree_frame = tk.Frame(self, bg="#FFFFFF")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        self.table = ttk.Treeview(tree_frame, columns=columns, show="headings")
        self.table.heading("comparison", text="Group Comparison")
        self.table.heading("method", text="Method")
        self.table.heading("statistic", text="Statistic")
        self.table.heading("p_value", text="p-value")
        self.table.heading("corrected_p", text="Corrected p-value")
        self.table.heading("significant", text="Significant")

        self.table.column("comparison", width=180, minwidth=100)
        self.table.column("method", width=140, minwidth=100)
        self.table.column("statistic", width=120, minwidth=80)
        self.table.column("p_value", width=120, minwidth=80)
        self.table.column("corrected_p", width=140, minwidth=80)
        self.table.column("significant", width=100, minwidth=60)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        if not self.results or len(self.results) == 0:
            self.table.insert("", tk.END, values=("⚠️ 无有效统计结果，请检查数据", "", "", "", "", ""))
        else:
            for res in self.results:
                comp = res.get("group_comparison", "")
                method = res.get("method", "")
                stat = f"{res.get('stat', 0):.4f}"
                p_val = f"{res.get('p_value', 1):.4f}"
                corr_p = res.get("corrected_p_value", None)
                corr_p_str = f"{corr_p:.4f}" if corr_p is not None else "N/A"
                sig = res.get("significant_after_correction", res.get("p_value", 1) < 0.05)
                sig_str = "Yes" if sig else "No"
                self.table.insert("", tk.END, values=(comp, method, stat, p_val, corr_p_str, sig_str))

        # 底部按钮
        btn_frame = tk.Frame(self, bg="#FFFFFF")
        btn_frame.pack(fill=tk.X, padx=15, pady=10)
        tk.Button(btn_frame, text="Export to Excel", command=self._export,
                  font=("Times New Roman", 10), bg="#d4e6f1", relief="flat",
                  cursor="hand2").pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Close", command=self.destroy,
                  font=("Times New Roman", 10), bg="#d4e6f1", relief="flat",
                  cursor="hand2").pack(side=tk.RIGHT)

    def _export(self):
        if not self.results or len(self.results) == 0:
            messagebox.showwarning("Warning", "无有效结果可导出！", parent=self)
            return
        save_path = filedialog.asksaveasfilename(
            parent=self, title="Export Results", defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")])
        if not save_path:
            return
        try:
            df = pd.DataFrame(self.results)
            df.to_excel(save_path, index=False)
            messagebox.showinfo("Success", f"Results exported to {save_path}", parent=self)
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}", parent=self)


# ========== 组配置对话框 ==========
class GroupDialogTk(tk.Toplevel):
    def __init__(self, parent, title="Group Configuration", group_name="", folder_path=""):
        super().__init__(parent)
        self.title(title)
        self.geometry("400x200")
        self.resizable(False, False)
        self.configure(bg="#FFFFFF")
        self.result = None
        self.transient(parent)
        self.grab_set()

        tk.Label(self, text="Group Name:", font=("Times New Roman", 11), bg="#FFFFFF").pack(
            anchor="w", padx=20, pady=(15, 2))
        self.name_entry = tk.Entry(self, font=("Times New Roman", 11))
        self.name_entry.pack(fill="x", padx=20)
        self.name_entry.insert(0, group_name)

        tk.Label(self, text="Data Folder:", font=("Times New Roman", 11), bg="#FFFFFF").pack(
            anchor="w", padx=20, pady=(10, 2))
        folder_frame = tk.Frame(self, bg="#FFFFFF")
        folder_frame.pack(fill="x", padx=20)
        self.folder_entry = tk.Entry(folder_frame, font=("Times New Roman", 11))
        self.folder_entry.pack(side=tk.LEFT, fill="x", expand=True)
        self.folder_entry.insert(0, folder_path)
        tk.Button(folder_frame, text="Browse", command=self._browse,
                  font=("Times New Roman", 9)).pack(side=tk.RIGHT, padx=(5, 0))

        tk.Button(self, text="Confirm", command=self._confirm,
                  font=("Times New Roman", 11, "bold"), bg="#d4e6f1",
                  relief="flat", cursor="hand2").pack(pady=15)

    def _browse(self):
        path = filedialog.askdirectory(title="Select Data Folder")
        if path:
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, path)

    def _confirm(self):
        name = self.name_entry.get().strip()
        folder = self.folder_entry.get().strip()
        if not name or not folder:
            messagebox.showwarning("Warning", "Please fill in both group name and folder path!", parent=self)
            return
        self.result = (name, folder)
        self.destroy()


# ========== 核心：统计分析面板 (tkinter) ==========
class StatisticalAnalysisPanel(tk.Frame):
    """统计分析面板 - tkinter版，适配 main_UI.py

    布局严格复刻原 PyQt5 版 AnalysisPanel：
    - 组管理区: QFrame at (0, 9, 258, 1006)    → 绝对坐标 (296, 9)
    - 数据分析区: QFrame at (275, 12, 855, 375) → 绝对坐标 (571, 12)
    - 可视化区: QFrame at (275, 395, 855, 614)  → 绝对坐标 (571, 395)
    - Project_Name: at (720, 324, 520, 139)      → 绝对坐标 (1016, 324)
    """

    # 原始 AnalysisPanel 在 1440 宽主窗口中的起始 x
    PANEL_X = 296

    def __init__(self, parent, show_navigation=True, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.show_navigation = show_navigation
        self.configure(bg="#FFFFFF")

        # 状态变量
        self.result = []
        self.group_select_features = {}
        self.data = None
        self.group_folder = {}
        self.group_files = {}
        self.is_valid = False
        self.enable_correction_var = tk.BooleanVar(value=True)
        self._current_fig = None

        # 存储图片引用
        self.images = {}

        # 资源目录
        self.resource_dir = project_root / "core" / "UI" / "UI_resource" / "Analysis_Panel"
        self.fe_resource_dir = project_root / "core" / "UI" / "UI_resource" / "Feature_Extraction_Panel"

        # 偏移量（嵌入模式左移289像素，隐藏导航区）
        self.offset_x = 0 if self.show_navigation else -289

        self._setup_ui()

    # ==================================================================
    # 坐标辅助：将 AnalysisPanel 内部坐标转为 canvas 绝对坐标
    # ==================================================================
    def _ax(self, panel_inner_x):
        """AnalysisPanel 内部 x → canvas x"""
        return self.PANEL_X + panel_inner_x + self.offset_x

    def _setup_ui(self):
        self.canvas = tk.Canvas(
            self, width=1440, height=1024,
            highlightthickness=0, bg="#FFFFFF"
        )
        self.canvas.pack(fill=tk.BOTH, expand=False)

        self._setup_group_area()
        self._setup_analysis_area()
        self._setup_visual_area()
        self._setup_project_name()

    # ==================================================================
    # 1. 组管理区域  — QFrame(0, 9, 258, 1006)
    # ==================================================================
    def _setup_group_area(self):
        # ---------- 背景 2.png ----------
        bg2 = self._load_image(self.resource_dir, "2.png", (258, 1006))
        if bg2:
            self.images["group_bg"] = bg2
            self.canvas.create_image(self._ax(0), 9, image=bg2, anchor="nw")

        # ---------- 装饰图 (相对于 group_widget 的坐标) ----------
        # 3.png at (29, 32) relative → absolute (296+29, 9+32) = (325, 41)
        # 4.png at (60, 20) relative → absolute (356, 29)
        # 5.png at (28, 99) relative → absolute (324, 108)
        # 6.png at (149, 101) relative → absolute (445, 110)
        decos = [
            ("3.png", 29, 32, 21, 16),
            ("4.png", 60, 20, 72, 38),
            ("5.png", 28, 99, 23, 23),
            ("6.png", 149, 101, 16, 18),
        ]
        for fname, rx, ry, w, h in decos:
            img = self._load_image(self.resource_dir, fname, (w, h))
            if img:
                self.images[f"gdeco_{fname}"] = img
                self.canvas.create_image(self._ax(rx), 9 + ry, image=img, anchor="nw")

        # ---------- 组管理内容 (VBoxLayout margins 20,30,20,30 spacing 15) ----------
        # 内容区: x=20..238(宽218), y=30..976(高946) 相对于 group_widget
        cx = 20   # 相对于 group_widget 左边
        cy_base = 9 + 30   # group_widget.y + margin_top = 9+30 = 39
        cw = 218

        # 按钮行 (HBox: "添加组" + "导入组")
        # 原版 VBoxLayout 中标题+spacing后的位置，与5.png/6.png装饰图对齐
        btn_y = 9 + 130   # 在装饰图下方留出足够空间
        btn_w = 105
        btn_h = 38

        group_btn_style = {
            "bg": "#FFFFFF", "fg": "#333",
            "font": ("微软雅黑", 9, "bold"),
            "relief": "flat", "bd": 1, "cursor": "hand2",
            "activebackground": "#E8F0FE"
        }

        self.add_group_btn = tk.Button(self, text="     添加组", command=self.add_group, **group_btn_style)
        self.canvas.create_window(self._ax(cx), btn_y, window=self.add_group_btn,
                                  anchor="nw", width=btn_w, height=btn_h)

        self.import_group_btn = tk.Button(self, text="   导入组", command=self.import_groups, **group_btn_style)
        self.canvas.create_window(self._ax(cx + btn_w + 8), btn_y, window=self.import_group_btn,
                                  anchor="nw", width=btn_w, height=btn_h)

        # 组列表 (QListWidget)
        list_y = btn_y + btn_h + 15
        list_bottom = 9 + 1006 - 30   # = 985
        validate_h = 45
        list_h = list_bottom - validate_h - 15 - list_y

        self.group_listbox = tk.Listbox(
            self, bg="#FFFFFF", fg="#333",
            font=("微软雅黑", 10),
            selectmode=tk.SINGLE, relief="flat", bd=0,
            highlightthickness=0
        )
        self.group_listbox.bind("<Double-Button-1>", self.modify_group)
        self.canvas.create_window(self._ax(cx), list_y, window=self.group_listbox,
                                  anchor="nw", width=cw, height=list_h)

        # 确认按钮
        validate_y = list_bottom - validate_h
        self.validate_btn = tk.Button(
            self, text="√     确认", command=self.validate_groups,
            bg="#FFFFFF", fg="#333",
            font=("微软雅黑", 11, "bold"),
            relief="flat", bd=1, cursor="hand2",
            activebackground="#f8f9fa"
        )
        self.canvas.create_window(self._ax(cx), validate_y, window=self.validate_btn,
                                  anchor="nw", width=cw, height=validate_h)

    # ==================================================================
    # 2. 数据分析区域  — QFrame(275, 12, 855, 375)
    # ==================================================================
    def _setup_analysis_area(self):
        ax_base = 275   # analysis_widget x (相对于 AnalysisPanel)
        ay_base = 12    # analysis_widget y

        # ---------- 背景 8.png ----------
        bg8 = self._load_image(self.resource_dir, "8.png", (855, 375))
        if bg8:
            self.images["analysis_bg"] = bg8
            self.canvas.create_image(self._ax(ax_base), ay_base, image=bg8, anchor="nw")

        # ---------- 装饰图 ----------
        # 12.png at (45, 27) relative to analysis_widget
        # 13.png at (50, 315) relative to analysis_widget
        for fname, rx, ry, w, h in [
            ("12.png", 45, 27, 23, 22),
            ("13.png", 50, 315, 18, 21),
        ]:
            img = self._load_image(self.resource_dir, fname, (w, h))
            if img:
                self.images[f"adeco_{fname}"] = img
                self.canvas.create_image(self._ax(ax_base + rx), ay_base + ry, image=img, anchor="nw")

        # ---------- 内容 (VBoxLayout margins 30,20,30,20 spacing 10) ----------
        cx = ax_base + 30     # 内容 x 起点 = 305
        cy = ay_base + 20     # 内容 y 起点 = 32
        cw = 855 - 60         # 内容宽度 = 795

        # 标题 "       数据分析" (15pt bold)
        self.canvas.create_text(
            self._ax(cx), cy,
            text="       数据分析",
            font=("微软雅黑", 15, "bold"), fill="#333", anchor="nw"
        )

        # 描述文字 (11pt, blue)
        cy += 28 + 10   # title_h + spacing
        self.canvas.create_text(
            self._ax(cx), cy,
            text="使用选定方法进行成组两两比较。当组别数量大于2时，将执行多重比较校正。",
            font=("微软雅黑", 10), fill="#2980b9", anchor="nw", width=cw
        )

        # ---------- Grid 区域 (margins 0,10,0,10 spacing 10) ----------
        cy += 30 + 10 + 10   # desc_h + spacing + grid_margin_top
        grid_y = cy
        row_h = 35
        row_spacing = 10
        label_font = ("Times New Roman", 12, "bold")
        combo_font = ("微软雅黑", 10)
        label_x = cx
        combo_x = cx + 100
        combo_w = 500

        # 加载 9.png 作为 combo 背景
        for i in range(4):
            combo_bg = self._load_image(self.resource_dir, "9.png", (combo_w, row_h))
            if combo_bg:
                self.images[f"combo_bg_{i}"] = combo_bg
                ry = grid_y + i * (row_h + row_spacing)
                self.canvas.create_image(self._ax(combo_x), ry, image=combo_bg, anchor="nw")

        # --- Row 0: Channel ---
        ry = grid_y
        self.canvas.create_text(self._ax(label_x), ry + 7, text="Channel:",
                                font=label_font, fill="#444", anchor="nw")
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(
            self, textvariable=self.channel_var, font=combo_font, state="readonly"
        )
        self.canvas.create_window(self._ax(combo_x), ry, window=self.channel_combo,
                                  anchor="nw", width=combo_w, height=row_h)

        # --- Row 1: Feature ---
        ry += row_h + row_spacing
        self.canvas.create_text(self._ax(label_x), ry + 7, text="Feature:",
                                font=label_font, fill="#444", anchor="nw")
        self.feature_var = tk.StringVar()
        self.feature_combo = ttk.Combobox(
            self, textvariable=self.feature_var, font=combo_font, state="readonly"
        )
        self.canvas.create_window(self._ax(combo_x), ry, window=self.feature_combo,
                                  anchor="nw", width=combo_w, height=row_h)

        # --- Row 2: Method ---
        ry += row_h + row_spacing
        self.canvas.create_text(self._ax(label_x), ry + 7, text="Method:",
                                font=label_font, fill="#444", anchor="nw")
        self.method_var = tk.StringVar()
        self.method_combo = ttk.Combobox(
            self, textvariable=self.method_var, font=combo_font, state="readonly",
            values=["t-test", "t-test(paired)", "anova",
                    "mann-whitney U", "wilcoxon(paired)", "kruskal-wallis"]
        )
        self.method_combo.current(0)
        self.canvas.create_window(self._ax(combo_x), ry, window=self.method_combo,
                                  anchor="nw", width=combo_w, height=row_h)

        # --- Row 3: Correction ---
        ry += row_h + row_spacing
        self.canvas.create_text(self._ax(label_x), ry + 7, text="Correction:",
                                font=label_font, fill="#444", anchor="nw")
        self.correction_var = tk.StringVar()
        self.correction_combo = ttk.Combobox(
            self, textvariable=self.correction_var, font=combo_font, state="readonly",
            values=["bonferroni", "fdr_bh", "fdr_by", "holm-sidak", "sidak"]
        )
        self.correction_combo.current(0)
        self.canvas.create_window(self._ax(combo_x), ry, window=self.correction_combo,
                                  anchor="nw", width=combo_w, height=row_h)

        # "使用" 复选框 (column 3)
        self.correction_check = tk.Checkbutton(
            self, text="使用", variable=self.enable_correction_var,
            font=("微软雅黑", 9, "bold"), bg="#FFFFFF",
            activebackground="#FFFFFF"
        )
        self.canvas.create_window(self._ax(combo_x + combo_w + 15), ry + 4,
                                  window=self.correction_check, anchor="nw")

        # --- Row 4: 底部按钮行 ---
        ry += row_h + row_spacing
        action_btn_style = {
            "bg": "#d4e6f1", "fg": "#2c3e50",
            "font": ("微软雅黑", 10, "bold"),
            "relief": "flat", "bd": 1, "cursor": "hand2",
            "activebackground": "#b8d4e3"
        }

        # 底部按钮行 — 确保绘制在 combo 背景之上
        self.run_btn = tk.Button(self, text="    开始", command=self.run_analysis, **action_btn_style)
        self.canvas.create_window(self._ax(label_x), ry + 5, window=self.run_btn,
                                  anchor="nw", width=140, height=40)

        self.status_label_var = tk.StringVar(value="状态：等待")
        self.status_label = tk.Label(
            self, textvariable=self.status_label_var,
            font=("微软雅黑", 10, "italic"), bg="#FFFFFF", fg="#333"
        )
        self.canvas.create_window(self._ax(label_x + 150), ry + 13,
                                  window=self.status_label, anchor="nw")

        self.export_btn = tk.Button(self, text="导出结果", command=self.export_results, **action_btn_style)
        self.canvas.create_window(self._ax(label_x + 310), ry + 5, window=self.export_btn,
                                  anchor="nw", width=120, height=40)

    # ==================================================================
    # 3. 数据可视化区域  — QFrame(275, 395, 855, 614)
    # ==================================================================
    def _setup_visual_area(self):
        vx_base = 275
        vy_base = 395

        # ---------- 背景 15.png ----------
        bg15 = self._load_image(self.resource_dir, "15.png", (855, 614))
        if bg15:
            self.images["visual_bg"] = bg15
            self.canvas.create_image(self._ax(vx_base), vy_base, image=bg15, anchor="nw")

        # ---------- 内容 (VBoxLayout margins 30,20,30,30 spacing 15) ----------
        cx = vx_base + 30     # = 305
        cy = vy_base + 20     # = 415
        cw = 855 - 60         # = 795

        # 标题 "   数据可视化" (18pt bold)
        self.canvas.create_text(
            self._ax(cx), cy,
            text="   数据可视化",
            font=("微软雅黑", 16, "bold"), fill="#333", anchor="nw"
        )

        # ---------- 控制行 ----------
        cy += 35 + 15   # title_h + spacing
        ctrl_y = cy

        self.canvas.create_text(
            self._ax(cx), ctrl_y + 7,
            text="类型:", font=("微软雅黑", 11, "bold"), fill="#444", anchor="nw"
        )

        self.plot_type_var = tk.StringVar()
        self.plot_type_combo = ttk.Combobox(
            self, textvariable=self.plot_type_var, font=("微软雅黑", 10),
            state="readonly",
            values=["scatter plot", "density histogram", "box plot", "violin plot"]
        )
        self.plot_type_combo.current(0)
        self.canvas.create_window(self._ax(cx + 55), ctrl_y, window=self.plot_type_combo,
                                  anchor="nw", width=180, height=35)

        vis_btn_style = {
            "bg": "#d4e6f1", "fg": "#2c3e50",
            "font": ("微软雅黑", 9, "bold"),
            "relief": "flat", "bd": 1, "cursor": "hand2",
            "activebackground": "#b8d4e3"
        }

        self.plot_btn = tk.Button(self, text="生成", command=self.generate_visualization, **vis_btn_style)
        self.canvas.create_window(self._ax(cx + 250), ctrl_y, window=self.plot_btn,
                                  anchor="nw", width=100, height=35)

        # 保存 + 设置 靠右
        self.save_plot_btn = tk.Button(self, text="保存", command=self.save_plot, **vis_btn_style)
        self.canvas.create_window(self._ax(cx + cw - 210), ctrl_y, window=self.save_plot_btn,
                                  anchor="nw", width=100, height=35)

        self.settings_btn = tk.Button(self, text="设置", **vis_btn_style)
        self.canvas.create_window(self._ax(cx + cw - 100), ctrl_y, window=self.settings_btn,
                                  anchor="nw", width=100, height=35)

        # ---------- 图表显示区域 (QScrollArea) ----------
        cy += 35 + 15
        plot_bottom = vy_base + 614 - 30   # = 979
        plot_h = plot_bottom - cy

        self.plot_frame = tk.Frame(self, bg="#FFFFFF", relief="flat", bd=0)
        self.canvas.create_window(self._ax(cx), cy, window=self.plot_frame,
                                  anchor="nw", width=cw, height=plot_h)

        self.plot_placeholder = tk.Label(
            self.plot_frame,
            text="📊 Run analysis first, then generate a plot here",
            font=("微软雅黑", 14), fg="#999", bg="#FFFFFF"
        )
        self.plot_placeholder.pack(expand=True)

    # ==================================================================
    # 4. Project_Name.png 叠加层  — at (720, 324, 520, 139)
    # ==================================================================
    def _setup_project_name(self):
        proj_img = self._load_image_keep_aspect(self.fe_resource_dir, "Project_Name.png", 520, 139)
        if proj_img:
            self.images["project_name"] = proj_img
            self.canvas.create_image(self._ax(720), 324, image=proj_img, anchor="nw")

    # ==================================================================
    # 图片加载
    # ==================================================================
    def _load_image(self, directory, filename, size=None):
        if not PIL_AVAILABLE:
            return None
        img_path = Path(directory) / filename
        if not img_path.exists():
            return None
        try:
            img = Image.open(img_path)
            if size:
                img = img.resize(size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"加载图片失败 {filename}: {e}")
            return None

    def _load_image_keep_aspect(self, directory, filename, max_w, max_h):
        """加载图片并保持宽高比缩放（不拉伸），对应 Qt.KeepAspectRatio"""
        if not PIL_AVAILABLE:
            return None
        img_path = Path(directory) / filename
        if not img_path.exists():
            return None
        try:
            img = Image.open(img_path)
            orig_w, orig_h = img.size
            scale = min(max_w / orig_w, max_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"加载图片失败 {filename}: {e}")
            return None

    # ==================================================================
    # 组管理
    # ==================================================================
    def add_group(self):
        dialog = GroupDialogTk(self)
        self.wait_window(dialog)
        if dialog.result:
            name, folder = dialog.result
            self.group_listbox.insert(tk.END, f"{name} - {folder}")
        self.is_valid = False

    def import_groups(self):
        folder = filedialog.askdirectory(title="选择包含组子文件夹的目录")
        if not folder:
            return
        folder_path = Path(folder)
        count = 0
        for sub in sorted(folder_path.iterdir()):
            if sub.is_dir():
                self.group_listbox.insert(tk.END, f"{sub.name} - {sub}")
                count += 1
        if count == 0:
            messagebox.showwarning("Warning", "所选目录下未找到子文件夹！")
        else:
            messagebox.showinfo("Success", f"已导入 {count} 个组")
        self.is_valid = False

    def modify_group(self, event=None):
        selection = self.group_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        current_text = self.group_listbox.get(idx)
        try:
            name, folder = current_text.split(" - ", 1)
        except ValueError:
            messagebox.showwarning("Error", "Invalid group format!")
            return

        dialog = GroupDialogTk(self, group_name=name, folder_path=folder)
        self.wait_window(dialog)
        if dialog.result:
            new_name, new_folder = dialog.result
            self.group_listbox.delete(idx)
            self.group_listbox.insert(idx, f"{new_name} - {new_folder}")
        self.is_valid = False

    def validate_groups(self):
        self.group_files.clear()
        self.group_select_features.clear()
        self.group_folder.clear()

        if self.group_listbox.size() < 2:
            messagebox.showwarning("Warning", "At least two groups required for analysis!")
            return

        for i in range(self.group_listbox.size()):
            try:
                group_data = self.group_listbox.get(i)
                group_name, folder_path = group_data.split(" - ", 1)
                self.group_files[group_name] = []
                if not os.path.exists(folder_path):
                    messagebox.showwarning("Error", f"Folder {folder_path} does not exist!")
                    return
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.xlsx'):
                        file_path = os.path.join(folder_path, file_name)
                        print(f"读取: {file_path} (xlsx)")
                        file_data = read_xlsx(file_path)
                        self.group_files[group_name].append(file_data)
                if not self.group_files[group_name]:
                    messagebox.showwarning("Error", f"No XLSX files found in {folder_path}!")
                    return
            except Exception as e:
                messagebox.showwarning("Error", f"Load group data failed: {str(e)}")
                return

        try:
            self.data = next(iter(self.group_files.values()))[0]
            self.feature_combo['values'] = list(self.data['feature'].keys())
            if self.feature_combo['values']:
                self.feature_combo.current(0)

            if 'ch_names' in self.data and self.data['ch_names']:
                self.channel_combo['values'] = self.data['ch_names']
                self.channel_combo.current(0)
                self.channel_combo.config(state="readonly")
            else:
                self.channel_combo['values'] = ["<无通道>"]
                self.channel_combo.current(0)
                self.channel_combo.config(state="disabled")

            self.status_label_var.set("Status: Waiting")
            self.is_valid = True
            messagebox.showinfo("Success", "✅ Data validation complete!")
        except Exception as e:
            messagebox.showwarning("Error", f"Initialize UI failed: {str(e)}")
            self.is_valid = False

    # ==================================================================
    # 统计分析
    # ==================================================================
    def run_analysis(self):
        if not self.is_valid:
            messagebox.showwarning("Action Required", "Validate groups before analysis!")
            return

        self.status_label_var.set("Status: Processing...")
        self.update_idletasks()

        try:
            self.group_select_features.clear()
            for group_name in self.group_files:
                self.group_select_features[group_name] = []
                for dataset in self.group_files[group_name]:
                    channel = self.channel_var.get() if self.channel_combo['state'] != 'disabled' else None
                    feature = self.feature_var.get()
                    self.group_select_features[group_name].extend(
                        get_feature(dataset, channel, feature)
                    )

            for group_name in list(self.group_select_features.keys()):
                cleaned = [float(v) for v in self.group_select_features[group_name]
                           if isinstance(v, (int, float))]
                data = np.array(cleaned)
                data = data[np.isfinite(data)]
                if len(data) == 0:
                    raise ValueError(f"Group '{group_name}' has no valid finite data.")
                self.group_select_features[group_name] = data.tolist()

            raw_method = self.method_var.get()
            method = raw_method.strip()
            is_paired = False
            if method == 't-test(paired)':
                method = 't-test'
                is_paired = True
            elif method == 'wilcoxon(paired)':
                method = 'wilcoxon'
                is_paired = True
            method = method.strip()

            self.result = calculate_significance(
                self.group_select_features,
                method=method,
                paired=is_paired
            )

            if self.enable_correction_var.get() and len(self.result) > 1:
                correct_method = self.correction_var.get().strip()
                self.result = multiple_comparison_correction(self.result, correct_method)

            self.status_label_var.set("状态：完成！")

            if not self.result or len(self.result) == 0:
                messagebox.showwarning("Warning", "⚠️ 未生成有效统计结果")
                return

            # 弹出结果对话框（与原版一致）
            result_dialog = ResultDialogTk(self, self.result)
            self.wait_window(result_dialog)
            messagebox.showinfo("Success", "✅ Statistical analysis completed!")

        except Exception as e:
            self.status_label_var.set("Status: Failed!")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            traceback.print_exc()

    # ==================================================================
    # 导出结果
    # ==================================================================
    def export_results(self):
        if not self.result:
            messagebox.showwarning("Warning", "No analysis results to export!")
            return
        save_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("JSON Files", "*.json")]
        )
        if not save_path:
            return
        try:
            result_df = pd.DataFrame(self.result)
            if save_path.endswith('.xlsx'):
                result_df.to_excel(save_path, index=False)
            else:
                result_df.to_json(save_path, orient='records', indent=4)
            messagebox.showinfo("Success", f"✅ Results exported to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")

    # ==================================================================
    # 可视化
    # ==================================================================
    def generate_visualization(self):
        if not self.group_select_features:
            messagebox.showwarning("Data Required", "⚠️ 请先运行分析（Run Analysis）！")
            return

        try:
            plot_type = self.plot_type_var.get()

            # 清空之前的图
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            import matplotlib.pyplot as plt
            plt.rcParams['font.family'] = 'Times New Roman'

            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)

            groups = list(self.group_select_features.keys())
            data_list = [self.group_select_features[g] for g in groups]
            colors = plt.cm.tab10.colors

            if plot_type == 'scatter plot':
                for i, (name, vals) in enumerate(self.group_select_features.items()):
                    ax.scatter([i] * len(vals), vals,
                              color=colors[i % len(colors)], label=name, alpha=0.7, s=50)
                ax.set_xticks(range(len(groups)))
                ax.set_xticklabels(groups)
                ax.set_title("Scatter Plot", fontsize=14, fontweight='bold')
                ax.legend()
            elif plot_type == 'box plot':
                bp = ax.boxplot(data_list, patch_artist=True, labels=groups)
                for patch, color in zip(bp['boxes'], colors[:len(groups)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.set_title("Box Plot", fontsize=14, fontweight='bold')
            elif plot_type == 'violin plot':
                vp = ax.violinplot(data_list, showmedians=True)
                for i, pc in enumerate(vp['bodies']):
                    pc.set_facecolor(colors[i % len(colors)])
                    pc.set_alpha(0.7)
                ax.set_xticks(range(1, len(groups) + 1))
                ax.set_xticklabels(groups)
                ax.set_title("Violin Plot", fontsize=14, fontweight='bold')
            elif plot_type == 'density histogram':
                for i, (name, vals) in enumerate(self.group_select_features.items()):
                    ax.hist(vals, bins=15, alpha=0.5, density=True,
                            color=colors[i % len(colors)], label=name, edgecolor='black')
                ax.set_title("Density Histogram", fontsize=14, fontweight='bold')
                ax.legend()

            ax.set_xlabel("Groups", fontsize=12)
            ax.set_ylabel("Values", fontsize=12)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            self._current_fig = fig
            canvas_widget = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("绘图错误", f"报错信息:\n{str(e)}")
            traceback.print_exc()

    def save_plot(self):
        if not hasattr(self, '_current_fig') or self._current_fig is None:
            messagebox.showwarning("Warning", "没有图表可保存！")
            return
        save_path = filedialog.asksaveasfilename(
            title="保存图表",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF File", "*.pdf"), ("SVG File", "*.svg")]
        )
        if save_path:
            try:
                self._current_fig.savefig(save_path, dpi=150, bbox_inches='tight')
                messagebox.showinfo("Success", f"图表已保存到 {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"保存失败: {str(e)}")
