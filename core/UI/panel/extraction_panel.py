# -*- coding: utf-8 -*-
# isort: skip_file
# flake8: noqa
"""
智融脑机 - 特征提取模块（Canvas图片版）
完全使用绝对坐标布局，所有图片元素按给定坐标放置。
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import traceback
import numpy as np
from pathlib import Path
import sys

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

# 导入业务模块
try:
    from core.io.data_io import DataLoader
    from core.processing.feature_extraction.multimodal_pipeline import MultimodalFeaturePipeline
    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    print(f"模块导入失败: {e}")
    traceback.print_exc()

# PIL 用于加载图片
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("警告: PIL 未安装，无法加载 PNG 图片，请安装 Pillow 库。")

# 特征映射表
FEATURE_MAP = {
    "EEG": {
        "time_domain": "时域特征 (均值, 方差, Hjorth参数等)",
        "freq_domain": "频域特征 (Welch PSD, 谱熵等)",
        "wavelet": "小波时频特征 (各频带能量)",
        "nonlinear": "非线性特征 (排列熵, 样本熵等)",
        "band_power": "生理频带功率 (Alpha, Beta功率)",
        "erp": "ERP事件相关电位 (需包含事件)",
        "connectivity": "功能连接性 (通道间相干性)",
        "spatial": "空间地形图特征"
    },
    "fNIRS": {
        "time_domain": "时域统计特征",
        "freq_domain": "频域功率特征",
        "wavelet": "小波分析特征",
        "nonlinear": "非线性动力学特征",
        "hbo_hbr": "双信号源特征 (HbO, HbR, HbT)",
        "channel_correlation": "通道间相关性分析"
    },
    "EMG": {
        "time_domain": "肌电时域特征 (MAV, RMS, 过零率等)",
        "freq_domain": "肌电频域特征 (中频, 均频等)",
        "wavelet": "肌电时频特征 (小波能量分布)",
        "nonlinear": "肌电非线性特征 (分形维数等)"
    },
    "ECG": {
        "morphological": "心电形态学特征 (心率, QRS波群)",
        "hrv_time": "HRV 心率变异性 - 时域 (SDNN, RMSSD)",
        "hrv_frequency": "HRV 心率变异性 - 频域 (LF, HF)",
        "wavelet": "时频特征 (STFT短时能量)",
        "nonlinear": "非线性特征 (Poincare, DFA)"
    }
}


class FeatureExtractionPanel(tk.Frame):
    """
    特征提取面板 - 固定尺寸1440×1024，Canvas绝对坐标布局
    """
    def __init__(self, parent, show_navigation=True, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.configure(width=1440, height=1024)
        self.pack_propagate(False)
        self.show_navigation = show_navigation

        # 核心数据
        self.clean_data_dict = None
        self.extracted_features = None
        self.checkbox_vars = {}          # 特征复选框变量
        self.current_filepath = None
        self.selected_modality = None     # 当前选中的模态
        self.has_unknown = False          # 数据中是否存在 UNKNOWN 模态

        # 模态图标管理
        self.modality_items = {}           # {modality: canvas_image_id}
        self.modality_states = {}           # {modality: BooleanVar}

        # 资源目录
        self.resource_dir = project_root / "core" / "UI" / "UI_resource" / "Feature_Extraction_Panel"
        self.model_sel_dir = project_root / "core" / "UI" / "UI_resource" / "Model_Selection_Buttons"
        self.selected_dir = self.model_sel_dir / "Selected"
        self.unselected_dir = self.model_sel_dir / "Unselected"

        # 存储所有图片对象
        self.images = {}

        # 构建UI
        self.setup_ui()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def setup_ui(self):
        # ... Canvas 创建 ...
        if self.show_navigation:
            self.offset_x = 0
        else:
            self.offset_x = -289  # 左移289像素，适应1151宽度

        self.canvas = tk.Canvas(
            self,
            width=1440,
            height=1024,
            highlightthickness=0,
            bg="#FFFFFF"
        )
        self.canvas.pack(fill=tk.BOTH, expand=False)

        # 1. 文件选择背景 (294,9)
        file_bg = self._load_image(self.resource_dir, "File_Selection_Background.png")
        if file_bg:
            self.images["file_bg"] = file_bg
            self.canvas.create_image(294 + self.offset_x, 9, image=file_bg, anchor="nw")

        # 2. 文件地址栏 (508,31)
        addr_bar = self._load_image(self.resource_dir, "File_Address_Bar.png", (889, 55))
        if addr_bar:
            self.images["addr_bar"] = addr_bar
            self.canvas.create_image(508 + self.offset_x, 31, image=addr_bar, anchor="nw")

        # 文件路径文本（覆盖在地址栏上）
        self.file_path_text_id = self.canvas.create_text(
            518 + self.offset_x, 31 + 55//2,
            text="",
            font=("Segoe UI", 11, "bold"),
            fill="#333333",
            anchor="w",
            width=840
        )

        # 3. 浏览按钮 Browse.png (329,120)
        browse_img = self._load_image(self.resource_dir, "Browse.png", (211, 55))
        if browse_img:
            self.images["browse"] = browse_img
            browse_id = self.canvas.create_image(308 + self.offset_x, 102, image=browse_img, anchor="nw")
            self.canvas.tag_bind(browse_id, "<Button-1>", lambda e: self.browse_file())
            self.canvas.tag_bind(browse_id, "<Enter>", lambda e: self.canvas.config(cursor="hand2"))
            self.canvas.tag_bind(browse_id, "<Leave>", lambda e: self.canvas.config(cursor=""))

        # ==================== 修改点1：日志信息栏 ====================
        # 4. 日志信息栏 Diary.png (869,400) 尺寸553x129
        diary_img = self._load_image(self.resource_dir, "Diary.png", (553, 129))
        if diary_img:
            self.images["diary"] = diary_img
            self.canvas.create_image(869 + self.offset_x, 400, image=diary_img, anchor="nw")

        # 日志文本（居中显示）
        self.status_text_id = self.canvas.create_text(
            869 + 553//2 + self.offset_x, 400 + 129//2,
            text="请上传数据文件...",
            font=("Segoe UI", 12, "bold"),
            fill="#333333",
            width=500,
            anchor="center"
        )

        # ==================== 修改点2：特征选择背景 ====================
        # 5. 特征选择背景 Feature_Selection.png (295,189) 尺寸540x340
        feat_bg = self._load_image(self.resource_dir, "Feature_Selection.png", (540, 340))
        if feat_bg:
            self.images["feat_bg"] = feat_bg
            self.canvas.create_image(295 + self.offset_x, 189, image=feat_bg, anchor="nw")

        # 6. 项目名称 Project_Name.png (905,207) - 保持原样
        proj_img = self._load_image(self.resource_dir, "Project_Name.png")
        if proj_img:
            self.images["project"] = proj_img
            self.canvas.create_image(905 + self.offset_x, 207, image=proj_img, anchor="nw")

        # 7. 结果背景 Result.png (294,553)
        res_bg = self._load_image(self.resource_dir, "Result.png")
        if res_bg:
            self.images["result_bg"] = res_bg
            self.canvas.create_image(294 + self.offset_x, 553, image=res_bg, anchor="nw")

        # ==================== 修改点3：模态选择图标 ====================
        # 从 (650,109) 开始，间隔185，高度39（等比例缩放）
        modalities = ["EEG", "ECG", "EMG", "fNIRS"]
        base_x, base_y = 650 + self.offset_x, 109
        spacing = 80
        target_height = 39

        # 加载所有图标，获取缩放后的图片和宽度
        icon_data = []  # (mod, photo, width)
        for mod in modalities:
            result = self._load_icon_with_size(mod, selected=False, height=target_height)
            if result:
                photo, width = result
                icon_data.append((mod, photo, width))
            else:
                # 如果加载失败，使用占位宽度（根据常见比例假设为80px）
                icon_data.append((mod, None, 80))

        current_x = base_x
        for mod, photo, width in icon_data:
            if photo:
                key = f"mod_{mod}"
                self.images[key] = photo
                img_id = self.canvas.create_image(current_x, base_y, image=photo, anchor="nw")
                self.modality_items[mod] = img_id
                self.modality_states[mod] = tk.BooleanVar(value=False)
                self.canvas.tag_bind(img_id, "<Button-1>", lambda e, m=mod: self.on_modality_click(m))
                self.canvas.tag_bind(img_id, "<Enter>", lambda e: self.canvas.config(cursor="hand2"))
                self.canvas.tag_bind(img_id, "<Leave>", lambda e: self.canvas.config(cursor=""))
            else:
                # 占位矩形（实际不会发生）
                rect_id = self.canvas.create_rectangle(current_x, base_y, current_x+width, base_y+target_height,
                                                       fill="gray", outline="")
                self.modality_items[mod] = rect_id
                self.modality_states[mod] = tk.BooleanVar(value=False)
                self.canvas.tag_bind(rect_id, "<Button-1>", lambda e, m=mod: self.on_modality_click(m))
                self.canvas.tag_bind(rect_id, "<Enter>", lambda e: self.canvas.config(cursor="hand2"))
                self.canvas.tag_bind(rect_id, "<Leave>", lambda e: self.canvas.config(cursor=""))
            current_x += width + spacing

        # ============ 特征复选框容器（带滚动条） ============
        # 位置 (320,256)，总宽460（左侧Canvas 440 + 滚动条20），高200
        self.check_canvas = tk.Canvas(
            self.canvas,
            bg="#ffffff",
            highlightthickness=0,
            width=440,
            height=200
        )
        self.check_scrollbar = ttk.Scrollbar(self.canvas, orient="vertical", command=self.check_canvas.yview)
        self.check_canvas.configure(yscrollcommand=self.check_scrollbar.set)

        # 在 Canvas 上创建一个内部 Frame（实际承载复选框），背景白色
        self.check_frame = tk.Frame(self.check_canvas, bg="#ffffff")
        # 设置内部 Frame 宽度为440，禁止自动调整大小
        self.check_frame.config(width=440)
        self.check_frame.pack_propagate(False)
        self.check_canvas.create_window((0 + self.offset_x, 0), window=self.check_frame, anchor="nw", tags="inner_frame")

        # 将 Canvas 和滚动条嵌入主 Canvas
        self.canvas.create_window(331 + self.offset_x, 268, window=self.check_canvas, anchor="nw", width=462, height=166)
        self.canvas.create_window(331 + self.offset_x + 440, 268, window=self.check_scrollbar, anchor="nw", width=20, height=166)

        # 绑定内部 Frame 大小变化以更新滚动区域
        def configure_inner_frame(event):
            self.check_canvas.configure(scrollregion=self.check_canvas.bbox("all"))

        self.check_frame.bind("<Configure>", configure_inner_frame)

        # 设置内部 Frame 的列权重，使两列均匀分布（各占一半）
        self.check_frame.grid_columnconfigure(0, weight=1)
        self.check_frame.grid_columnconfigure(1, weight=1)

        # 设置 ttk 样式，使 Treeview 使用 Segoe UI 字体
        style = ttk.Style()
        style.theme_use('vista')  # 使用 clam 主题以支持自定义字体
        # ============ 结果表格 ============
        # ============ 结果表格（带样式） ============
        style.configure("Treeview",
                        background="#ffffff",
                        foreground="#333333",
                        fieldbackground="#ffffff",
                        borderwidth=0,
                        relief="flat",
                        font=("Segoe UI", 10, "bold"),  # 内容字体
                        rowheight=28)  # 行高
        style.map('Treeview', background=[('selected', '#e0f0ff')])  # 选中行淡蓝色
        style.configure("Treeview.Heading",
                        background="#f5f5f5",
                        foreground="#333333",
                        font=("Segoe UI", 11, "bold"),
                        relief="flat")
        style.map("Treeview.Heading", background=[('active', '#e8e8e8')])

        # 创建表格
        tree_frame = tk.Frame(self.canvas, borderwidth=0, highlightthickness= 0)
        columns = ("Name", "Value")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)
        self.tree.heading("Name", text="特征名称")
        self.tree.heading("Value", text="特征数值")
        self.tree.column("Name", width=450, anchor="center")  # 名称左对齐
        self.tree.column("Value", width=450, anchor="center")  # 数字居中

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 嵌入Canvas
        self.canvas.create_window(380 + self.offset_x, 620, window=tree_frame, anchor="nw",
                                  width=850, height=350)
        self.canvas.create_window(380 + self.offset_x, 620, window=tree_frame, anchor="nw", width=950, height=350)

        # ============ 提取按钮 Process.png ============
        # 坐标 (600,464)，尺寸 211x55
        process_img = self._load_image(self.resource_dir, "Process.png", (211, 55))
        if process_img:
            self.images["process_btn"] = process_img
            process_id = self.canvas.create_image(600 + self.offset_x, 464, image=process_img, anchor="nw")
            self.canvas.tag_bind(process_id, "<Button-1>", lambda e: self.action_extract())
            self.canvas.tag_bind(process_id, "<Enter>", lambda e: self.canvas.config(cursor="hand2"))
            self.canvas.tag_bind(process_id, "<Leave>", lambda e: self.canvas.config(cursor=""))

        # 初始化复选框显示提示
        self.update_checkboxes(None)

    # ------------------------------------------------------------------
    # 图片加载辅助（增强版，返回图片和宽度）
    # ------------------------------------------------------------------
    def _load_image(self, directory, filename, size=None):
        if not PIL_AVAILABLE:
            return None
        img_path = Path(directory) / filename
        if not img_path.exists():
            print(f"⚠️ 图片不存在: {img_path}")
            return None
        try:
            img = Image.open(img_path)
            if size:
                img = img.resize(size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"❌ 加载图片失败 {filename}: {e}")
            return None

    def _load_icon_with_size(self, modality, selected=False, height=None):
        """加载模态图标，返回 (PhotoImage, width) 元组，若 height 指定则缩放到该高度（保持比例）"""
        if not PIL_AVAILABLE:
            return None
        subdir = self.selected_dir if selected else self.unselected_dir
        filename = f"{modality}.png"
        img_path = Path(subdir) / filename
        if not img_path.exists():
            print(f"⚠️ 图标不存在: {img_path}")
            return None
        try:
            img = Image.open(img_path)
            original_width, original_height = img.size
            if height:
                scale = height / original_height
                new_width = int(original_width * scale)
                img = img.resize((new_width, height), Image.Resampling.LANCZOS)
                width = new_width
            else:
                width = original_width
            return ImageTk.PhotoImage(img), width
        except Exception as e:
            print(f"❌ 加载图标失败 {filename}: {e}")
            return None

    def _load_icon(self, modality, selected=False):
        """简单加载图标，不缩放（保留供其他部分调用，但实际切换时使用带缩放的版本）"""
        subdir = self.selected_dir if selected else self.unselected_dir
        filename = f"{modality}.png"
        return self._load_image(subdir, filename)

    # ------------------------------------------------------------------
    # 模态点击处理（修改为使用缩放后的图标）
    # ------------------------------------------------------------------
    def on_modality_click(self, modality):
        """点击模态图标：切换选中状态，如有UNKNOWN则重新处理数据"""
        # 如果已经选中，忽略
        if self.modality_states[modality].get():
            return

        # 取消其他所有选中
        for mod, var in self.modality_states.items():
            if var.get():
                var.set(False)
                # 使用带缩放的版本加载未选中图标（高度39）
                result = self._load_icon_with_size(mod, selected=False, height=39)
                if result:
                    new_img, _ = result
                    self.images[f"mod_{mod}_unsel"] = new_img
                    self.canvas.itemconfig(self.modality_items[mod], image=new_img)

        # 设置当前为选中
        self.modality_states[modality].set(True)
        result = self._load_icon_with_size(modality, selected=True, height=39)
        if result:
            new_img, _ = result
            self.images[f"mod_{modality}_sel"] = new_img
            self.canvas.itemconfig(self.modality_items[modality], image=new_img)

        self.selected_modality = modality
        self.update_checkboxes(modality)

        # 如果存在 UNKNOWN 且已有文件路径，则用此模态重新处理数据
        if self.has_unknown and self.current_filepath:
            self.has_unknown = False
            self.load_data(self.current_filepath, manual_mod=modality)

    def update_checkboxes(self, modality):
        """根据模态刷新特征复选框"""
        # 清空内部 Frame 的所有子控件
        for widget in self.check_frame.winfo_children():
            widget.destroy()
        self.checkbox_vars.clear()

        if not modality or modality not in FEATURE_MAP:
            label = tk.Label(self.check_frame, text="请先选择有效模态", bg="#ffffff",
                             font=("Segoe UI", 12, "bold"), fg="#666666")
            label.grid(row=0, column=0, columnspan=2, sticky="ew")
            return

        features = FEATURE_MAP[modality]
        row, col = 0, 0
        # 确保两列权重
        self.check_frame.grid_columnconfigure(0, weight=1)
        self.check_frame.grid_columnconfigure(1, weight=1)

        for feat_key, feat_desc in features.items():
            var = tk.BooleanVar(value=True)
            self.checkbox_vars[feat_key] = var
            chk = tk.Checkbutton(self.check_frame, text=feat_desc, variable=var,
                                 bg="#ffffff", font=("Segoe UI", 10, "bold"),
                                 activebackground="#ffffff", selectcolor="white",
                                 wraplength=200)  # 自动换行适应列宽
            chk.grid(row=row, column=col, sticky="w", padx=5, pady=2)
            col += 1
            if col > 1:
                col = 0
                row += 1

    # ------------------------------------------------------------------
    # 文件操作
    # ------------------------------------------------------------------
    def browse_file(self):
        if not MODULES_LOADED:
            messagebox.showerror("错误", "底层模块未加载，请检查文件路径。")
            return
        filepath = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("支持的格式", "*.csv *.mat *.edf *.bdf *.npy *.npz *.snirf *.set *.pkl *.json"),
                       ("All Files", "*.*")]
        )
        if not filepath:
            return
        self.current_filepath = filepath
        self.load_data(filepath)

    def load_data(self, filepath, manual_mod=None):
        """加载数据，若 manual_mod 指定则强制替换 UNKNOWN"""
        self.update_status("正在读取文件，请稍候...", "#0056b3")
        # 清空表格
        for item in self.tree.get_children():
            self.tree.delete(item)

        def background_task():
            try:
                loader = DataLoader()
                raw_dict = loader.load(filepath)

                # 添加：标准化模态名称
                if 'signal' in raw_dict:
                    standardized_signals = {}
                    for mod_name, signal_data in raw_dict['signal'].items():
                        # 将常见变体标准化为标准名称
                        if mod_name.lower() in ['fnirs', 'nirs', 'snirf']:
                            std_name = 'fNIRS'
                        elif mod_name.lower() in ['eeg']:
                            std_name = 'EEG'
                        elif mod_name.lower() in ['ecg']:
                            std_name = 'ECG'
                        elif mod_name.lower() in ['emg']:
                            std_name = 'EMG'
                        else:
                            std_name = mod_name  # 保持原样

                        # 如果信号数据中包含signal_type，也进行标准化
                        if isinstance(signal_data, dict) and 'signal_type' in signal_data:
                            if signal_data['signal_type'].lower() in ['fnirs', 'nirs']:
                                signal_data['signal_type'] = 'fnirs'

                        standardized_signals[std_name] = signal_data

                    raw_dict['signal'] = standardized_signals

                # 处理手动指定模态
                if manual_mod:
                    # 替换 UNKNOWN 模态
                    if 'signal' in raw_dict and 'UNKNOWN' in raw_dict['signal']:
                        unknown_data = raw_dict['signal'].pop('UNKNOWN')
                        # 包装为标准格式
                        fs = raw_dict.get('fs', 250)
                        if isinstance(unknown_data, dict) and 'data' in unknown_data:
                            data = unknown_data['data']
                        else:
                            data = unknown_data
                        # 创建新条目
                        if hasattr(data, 'shape'):
                            if len(data.shape) == 1:
                                n_channels = 1
                                n_samples = data.shape[0]
                            else:
                                n_channels, n_samples = data.shape
                            signal_entry = {
                                'data': data,
                                'sampling_rate': fs,
                                'channel_names': [f'Ch{i + 1}' for i in range(n_channels)],
                                'signal_type': manual_mod.lower(),
                                'unit': 'unknown',
                                'n_channels': n_channels,
                                'n_samples': n_samples,
                                'duration': n_samples / fs
                            }
                            raw_dict['signal'][manual_mod] = signal_entry
                        raw_dict['meta']['modality'] = list(raw_dict['signal'].keys())
                        self.has_unknown = False
                    else:
                        # 没有 UNKNOWN，直接添加（但一般不会）
                        pass
                else:
                    # 正常加载，检查是否有 UNKNOWN
                    mods = list(raw_dict.get('signal', {}).keys())
                    if 'UNKNOWN' in mods:
                        self.has_unknown = True
                    else:
                        self.has_unknown = False

                self.clean_data_dict = raw_dict

                # 更新UI
                self.after(0, self._post_load_update, filepath)

            except Exception as e:
                traceback.print_exc()
                self.after(0, lambda: self.update_status(f"加载失败: {str(e)}", "red"))

        threading.Thread(target=background_task, daemon=True).start()

    def _post_load_update(self, filepath):
        """加载完成后的UI更新"""
        # 更新文件路径显示
        self.canvas.itemconfig(self.file_path_text_id, text=filepath)

        mods = list(self.clean_data_dict.get('signal', {}).keys())
        valid_mods = [m for m in mods if m in FEATURE_MAP]

        if self.has_unknown:
            self.update_status("检测到未知模态，请点击上方图标选择正确的模态类型", "orange")
            # 清空所有模态选中状态（使用带缩放的版本）
            for mod, var in self.modality_states.items():
                if var.get():
                    var.set(False)
                    result = self._load_icon_with_size(mod, selected=False, height=39)
                    if result:
                        new_img, _ = result
                        self.images[f"mod_{mod}_unsel"] = new_img
                        self.canvas.itemconfig(self.modality_items[mod], image=new_img)
            self.selected_modality = None
            self.update_checkboxes(None)
        elif valid_mods:
            # 自动选中第一个有效模态
            first_mod = valid_mods[0]
            # 触发点击逻辑（选中图标、更新复选框）
            self.on_modality_click(first_mod)
            self.update_status(f"数据就绪！模态: {', '.join(valid_mods)}", "green")
        else:
            self.update_status("未检测到支持的模态，请手动选择", "orange")
            self.selected_modality = None
            self.update_checkboxes(None)

    def update_status(self, message, color="black"):
        """更新日志信息栏"""
        self.canvas.itemconfig(self.status_text_id, text=message, fill=color)

    # ------------------------------------------------------------------
    # 特征提取
    def action_extract(self):
        if not self.clean_data_dict:
            messagebox.showwarning("警告", "请先加载数据文件")
            return
        modality = self.selected_modality
        if not modality:
            messagebox.showwarning("警告", "请先选择一种模态")
            return

        selected_cats = [key for key, var in self.checkbox_vars.items() if var.get()]
        if not selected_cats:
            messagebox.showwarning("提示", "请至少勾选一种特征集")
            return

        self.update_status(f"正在提取 {modality} 特征...", "#0056b3")

        def background_extraction():
            try:
                request = {modality: selected_cats}

                # 修复：检查模态名称是否存在，并尝试匹配
                signal_dict = self.clean_data_dict.get('signal', {})

                # 尝试找到匹配的模态键（不区分大小写）
                matched_modality = None
                for key in signal_dict.keys():
                    if key.lower() == modality.lower():
                        matched_modality = key
                        break

                if not matched_modality:
                    # 如果找不到匹配，尝试常见的变体
                    modality_variants = {
                        'fNIRS': ['fnirs', 'nirs', 'fNIRS', 'FNIRS', 'snirf'],
                        'EEG': ['eeg', 'EEG'],
                        'ECG': ['ecg', 'ECG'],
                        'EMG': ['emg', 'EMG']
                    }

                    variants = modality_variants.get(modality, [modality])
                    for variant in variants:
                        if variant in signal_dict:
                            matched_modality = variant
                            break

                if not matched_modality:
                    available_mods = list(signal_dict.keys())
                    error_msg = f"找不到模态 {modality}，可用模态: {available_mods}"
                    self.after(0, lambda msg=error_msg: self.update_status(msg, "red"))
                    return

                # 使用匹配到的模态名称
                signal_data = signal_dict[matched_modality]['data']

                # 检查数据大小
                n_samples = signal_data.shape[1]
                fs = signal_dict[matched_modality].get('sampling_rate', 250)
                duration = n_samples / fs

                self.after(0, lambda: self.update_status(
                    f"数据长度: {duration:.1f}秒 ({n_samples}样本), 正在提取特征...",
                    "#0056b3"
                ))

                # 如果是长数据，给出提示
                if duration > 300:  # 超过5分钟
                    self.after(0, lambda: messagebox.showinfo(
                        "提示",
                        f"数据较长 ({duration:.1f}秒)，特征提取可能需要几分钟时间，请耐心等待..."
                    ))

                pipeline = MultimodalFeaturePipeline(self.clean_data_dict, selected_features=request)
                final_dict = pipeline.run_pipeline()
                all_feats = final_dict.get('processed', {}).get('feature', {})
                self.extracted_features = {modality: all_feats.get(modality, {})}
                self.after(0, self._display_features, modality)

            except Exception as e:
                traceback.print_exc()
                # 修复：使用局部变量保存错误信息
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self.update_status(f"提取异常: {msg}", "red"))

        threading.Thread(target=background_extraction, daemon=True).start()

    def _display_features(self, modality):
        self.update_status(f"{modality} 特征提取完成", "green")

        # 清空表格
        for item in self.tree.get_children():
            self.tree.delete(item)

        mod_feats = self.extracted_features.get(modality, {})
        flat = {}
        for k, v in mod_feats.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    flat[f"[{k}] {sk}"] = sv
            else:
                flat[k] = v

        # 显示在表格中
        for name, val in flat.items():
            if isinstance(val, float):
                val_str = f"{val:.6f}"
            elif isinstance(val, (list, np.ndarray)):
                val_str = f"Array {np.shape(val)}"
            else:
                val_str = str(val)
            self.tree.insert('', "end", values=(name, val_str))

        # ========== 新增：自动保存特征到文件 ==========
        self._auto_save_features(modality, flat)

    def _auto_save_features(self, modality, features_dict):
        """自动保存特征数据 - 符合 data_io.py 的标准"""
        try:
            import pickle
            import json
            from datetime import datetime
            import os
            import numpy as np
            import pandas as pd
            import re
            from collections import defaultdict

            # 创建保存目录
            save_dir = Path(project_root) / "output" / "processed_data"
            save_dir.mkdir(parents=True, exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # ========== 第一步：获取原始数据字典 ==========
            original_data = self.clean_data_dict.copy() if self.clean_data_dict else {}

            # ========== 第二步：构建标准格式的特征数据 ==========

            # 从原始信号中获取通道信息
            channel_info = self._get_channel_info(modality, original_data)
            n_channels = channel_info['n_channels']
            channel_names = channel_info['channel_names']

            print(f"通道信息: {n_channels}通道, 名称: {channel_names}")
            print(f"features_dict 包含 {len(features_dict)} 个特征")

            # 构建 features 字典
            # 格式: {category: {feature_name: [ch1_val, ch2_val, ...]}}
            features = defaultdict(dict)

            # 解析 features_dict
            # 您的数据格式是: 'Ch1_mean', 'Ch1_std', ... 每个特征名包含通道前缀
            for feat_key, value in features_dict.items():
                # 提取通道号和特征名
                # 例如: 'Ch1_mean' -> 通道1, 特征名 'mean'
                match = re.match(r'[Cc]h(\d+)_(.+)', feat_key)
                if match:
                    ch_num = int(match.group(1)) - 1  # 转为0索引
                    feat_name = match.group(2)

                    # 确定类别（从特征名推断）
                    if 'wavelet' in feat_name:
                        category = 'wavelet'
                    elif any(x in feat_name for x in ['mean', 'std', 'var', 'skewness', 'kurtosis',
                                                      'max', 'min', 'rms', 'peak_to_peak', 'shape_factor',
                                                      'impulse_factor', 'hjorth', 'zero_crossing']):
                        category = 'time_domain'
                    elif any(x in feat_name for x in ['power', 'freq', 'spectral', 'peak_freq', 'total_power']):
                        category = 'freq_domain'
                    elif any(x in feat_name for x in ['entropy', 'sample_entropy', 'permutation_entropy',
                                                      'higuchi_fd', 'svd_entropy']):
                        category = 'nonlinear'
                    elif any(x in feat_name for x in ['hbo_', 'hbr_', 'hbt_', 'diff_']):
                        category = 'hbo_hbr'
                    elif 'correlation' in feat_name:
                        category = 'channel_correlation'
                    else:
                        category = 'other'

                    # 初始化特征数组
                    if feat_name not in features[category]:
                        features[category][feat_name] = [np.nan] * n_channels

                    # 填充对应通道的值
                    if 0 <= ch_num < n_channels:
                        if isinstance(value, (int, float, np.number)):
                            features[category][feat_name][ch_num] = float(value)
                        else:
                            features[category][feat_name][ch_num] = value
                else:
                    # 没有通道前缀的特征（如全局特征）
                    if 'hbo_' in feat_key or 'hbr_' in feat_key or 'hbt_' in feat_key or 'diff_' in feat_key:
                        category = 'hbo_hbr'
                        feat_name = feat_key
                    elif 'correlation' in feat_key:
                        category = 'channel_correlation'
                        feat_name = feat_key
                    else:
                        category = 'global'
                        feat_name = feat_key

                    # 全局特征扩展到所有通道
                    if feat_name not in features[category]:
                        if isinstance(value, (int, float, np.number)):
                            features[category][feat_name] = [float(value)] * n_channels
                        else:
                            features[category][feat_name] = [value] * n_channels

            # 打印构建结果
            print(f"\n构建的 features 包含 {len(features)} 个类别")
            for category, cat_features in features.items():
                print(f"  类别 {category}: {len(cat_features)} 个特征")
                # 打印前2个特征的示例值
                for i, (feat_name, values) in enumerate(cat_features.items()):
                    if i < 2:
                        print(f"    {feat_name}: 前3个值 {values[:3]}")

            # ========== 第三步：构建完整的数据字典 ==========

            # 创建符合 data_io.py 标准的数据字典
            complete_data_dict = {
                "meta": {
                    "subject_id": original_data.get("meta", {}).get("subject_id", "unknown"),
                    "session_id": original_data.get("meta", {}).get("session_id", "session1"),
                    "task": original_data.get("meta", {}).get("task", "unknown"),
                    "recording_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "file_path": self.current_filepath,
                    "format_version": "1.0",
                    "modality": [modality],
                    "device": original_data.get("meta", {}).get("device", ""),
                    "sampling_rate": channel_info.get('sampling_rate', 250),
                    "n_channels": n_channels,
                    "channel_names": channel_names,
                    "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "signal": original_data.get("signal", {}),
                "event": original_data.get("event", {}),
                "processed": {
                    "features": dict(features),  # 转换为普通字典
                    "artifacts": original_data.get("processed", {}).get("artifacts", {}),
                    "filtered_data": original_data.get("processed", {}).get("filtered_data", {}),
                    "processing_history": self._get_processing_history()
                }
            }

            # ========== 第四步：保存文件 ==========

            # 1. 保存为PKL（最完整）
            pkl_path = save_dir / f"{modality}_processed_{timestamp}.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(complete_data_dict, f)
            print(f"✅ 已保存: {pkl_path}")

            # 2. 同时保存特征摘要为Excel（方便查看）
            excel_path = save_dir / f"{modality}_features_summary_{timestamp}.xlsx"

            summary_data = []
            for category, cat_features in features.items():
                for feat_name, feat_value in cat_features.items():
                    if isinstance(feat_value, (list, np.ndarray)) and len(feat_value) > 1:
                        # 多通道数据：每个通道一行
                        for ch_idx, val in enumerate(feat_value):
                            if ch_idx < len(channel_names):
                                ch_name = channel_names[ch_idx]
                            else:
                                ch_name = f"Ch{ch_idx + 1}"
                            summary_data.append({
                                'Modality': modality,
                                'Category': category,
                                'Channel': ch_name,
                                'Feature': feat_name,
                                'Value': val,
                                'Data_Type': type(val).__name__
                            })
                    else:
                        # 单通道数据
                        val = feat_value[0] if isinstance(feat_value, (list, np.ndarray)) else feat_value
                        summary_data.append({
                            'Modality': modality,
                            'Category': category,
                            'Channel': 'Global',
                            'Feature': feat_name,
                            'Value': val,
                            'Data_Type': type(val).__name__
                        })

            if summary_data:
                df = pd.DataFrame(summary_data)
                df.to_excel(excel_path, index=False)
                print(f"✅ 特征摘要已保存: {excel_path}")
                print(f"摘要包含 {len(summary_data)} 行数据")
            else:
                print("⚠️ 警告: summary_data 为空，没有保存Excel文件")

            # 更新状态
            self.update_status(f"数据已保存: {pkl_path.name}", "green")

        except Exception as e:
            print(f"❌ 保存失败: {e}")
            traceback.print_exc()
            self.update_status(f"保存失败: {str(e)}", "orange")

        except Exception as e:
            print(f"❌ 保存失败: {e}")
            traceback.print_exc()
            self.update_status(f"保存失败: {str(e)}", "orange")

    def _get_channel_info(self, modality, data_dict):
        """从原始数据中获取通道信息"""
        result = {
            'n_channels': 1,
            'channel_names': ['Global'],
            'sampling_rate': 250
        }

        if 'signal' in data_dict:
            for key, signal_info in data_dict['signal'].items():
                if key.lower() == modality.lower() or modality.lower() in key.lower():
                    if 'data' in signal_info:
                        data = signal_info['data']
                        if hasattr(data, 'shape'):
                            if len(data.shape) > 1:
                                result['n_channels'] = data.shape[0]
                            else:
                                result['n_channels'] = 1

                    if 'channel_names' in signal_info and signal_info['channel_names']:
                        result['channel_names'] = signal_info['channel_names']
                    else:
                        result['channel_names'] = [f"Ch{i + 1}" for i in range(result['n_channels'])]

                    if 'sampling_rate' in signal_info:
                        result['sampling_rate'] = signal_info['sampling_rate']
                    break

        return result

    def _get_processing_history(self):
        """获取处理历史"""
        history = []
        if hasattr(self, 'clean_data_dict') and self.clean_data_dict:
            if 'processed' in self.clean_data_dict:
                history = self.clean_data_dict['processed'].get('processing_history', [])

        # 添加当前处理记录
        from datetime import datetime
        history.append({
            'step': 'feature_extraction',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'modality': self.selected_modality,
            'features': list(self.checkbox_vars.keys()) if self.checkbox_vars else []
        })

        return history

if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    root = tk.Tk()
    root.title("智融脑机 - 特征提取模块")
    root.geometry("1440x1024")
    root.resizable(False, False)
    root.configure(bg="#FFFFFF")

    app = FeatureExtractionPanel(root)
    app.pack(fill=tk.BOTH, expand=True)

    root.mainloop()