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
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.configure(width=1440, height=1024)
        self.pack_propagate(False)

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
            self.canvas.create_image(294, 9, image=file_bg, anchor="nw")

        # 2. 文件地址栏 (508,31)
        addr_bar = self._load_image(self.resource_dir, "File_Address_Bar.png", (889, 55))
        if addr_bar:
            self.images["addr_bar"] = addr_bar
            self.canvas.create_image(508, 31, image=addr_bar, anchor="nw")

        # 文件路径文本（覆盖在地址栏上）
        self.file_path_text_id = self.canvas.create_text(
            518, 31 + 55//2,
            text="",
            font=("微软雅黑", 11),
            fill="#333333",
            anchor="w",
            width=840
        )

        # 3. 浏览按钮 Browse.png (329,120)
        browse_img = self._load_image(self.resource_dir, "Browse.png", (211, 55))
        if browse_img:
            self.images["browse"] = browse_img
            browse_id = self.canvas.create_image(308, 102, image=browse_img, anchor="nw")
            self.canvas.tag_bind(browse_id, "<Button-1>", lambda e: self.browse_file())
            self.canvas.tag_bind(browse_id, "<Enter>", lambda e: self.canvas.config(cursor="hand2"))
            self.canvas.tag_bind(browse_id, "<Leave>", lambda e: self.canvas.config(cursor=""))

        # ==================== 修改点1：日志信息栏 ====================
        # 4. 日志信息栏 Diary.png (869,400) 尺寸553x129
        diary_img = self._load_image(self.resource_dir, "Diary.png", (553, 129))
        if diary_img:
            self.images["diary"] = diary_img
            self.canvas.create_image(869, 400, image=diary_img, anchor="nw")

        # 日志文本（居中显示）
        self.status_text_id = self.canvas.create_text(
            869 + 553//2, 400 + 129//2,
            text="请上传数据文件...",
            font=("微软雅黑", 12),
            fill="#333333",
            width=500,
            anchor="center"
        )

        # ==================== 修改点2：特征选择背景 ====================
        # 5. 特征选择背景 Feature_Selection.png (295,189) 尺寸540x340
        feat_bg = self._load_image(self.resource_dir, "Feature_Selection.png", (540, 340))
        if feat_bg:
            self.images["feat_bg"] = feat_bg
            self.canvas.create_image(295, 189, image=feat_bg, anchor="nw")

        # 6. 项目名称 Project_Name.png (905,207) - 保持原样
        proj_img = self._load_image(self.resource_dir, "Project_Name.png")
        if proj_img:
            self.images["project"] = proj_img
            self.canvas.create_image(905, 207, image=proj_img, anchor="nw")

        # 7. 结果背景 Result.png (294,553)
        res_bg = self._load_image(self.resource_dir, "Result.png")
        if res_bg:
            self.images["result_bg"] = res_bg
            self.canvas.create_image(294, 553, image=res_bg, anchor="nw")

        # ==================== 修改点3：模态选择图标 ====================
        # 从 (650,109) 开始，间隔185，高度39（等比例缩放）
        modalities = ["EEG", "ECG", "EMG", "fNIRS"]
        base_x, base_y = 650, 109
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
        self.check_canvas.create_window((0, 0), window=self.check_frame, anchor="nw", tags="inner_frame")

        # 将 Canvas 和滚动条嵌入主 Canvas
        self.canvas.create_window(320, 256, window=self.check_canvas, anchor="nw", width=440, height=200)
        self.canvas.create_window(320 + 440, 256, window=self.check_scrollbar, anchor="nw", width=20, height=200)

        # 绑定内部 Frame 大小变化以更新滚动区域
        def configure_inner_frame(event):
            self.check_canvas.configure(scrollregion=self.check_canvas.bbox("all"))

        self.check_frame.bind("<Configure>", configure_inner_frame)

        # 设置内部 Frame 的列权重，使两列均匀分布（各占一半）
        self.check_frame.grid_columnconfigure(0, weight=1)
        self.check_frame.grid_columnconfigure(1, weight=1)



        # ============ 结果表格 ============
        tree_frame = tk.Frame(self.canvas)
        columns = ("Name", "Value")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)
        self.tree.heading("Name", text="特征名称")
        self.tree.heading("Value", text="特征数值")
        self.tree.column("Name", width=400)
        self.tree.column("Value", width=300)
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        # 嵌入Canvas：从 (320,580) 到 (1070,800) 宽750高220
        self.canvas.create_window(380, 620, window=tree_frame, anchor="nw", width=850, height=350)

        # ============ 提取按钮 Process.png ============
        # 坐标 (600,464)，尺寸 211x55
        process_img = self._load_image(self.resource_dir, "Process.png", (211, 55))
        if process_img:
            self.images["process_btn"] = process_img
            process_id = self.canvas.create_image(600, 464, image=process_img, anchor="nw")
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
                             font=("微软雅黑", 12), fg="#666666")
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
                                 bg="#ffffff", font=("微软雅黑", 10),
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
                                'channel_names': [f'Ch{i+1}' for i in range(n_channels)],
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
    # ------------------------------------------------------------------
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
                pipeline = MultimodalFeaturePipeline(self.clean_data_dict, selected_features=request)
                final_dict = pipeline.run_pipeline()
                all_feats = final_dict.get('processed', {}).get('feature', {})
                self.extracted_features = {modality: all_feats.get(modality, {})}
                self.after(0, self._display_features, modality)
            except Exception as e:
                traceback.print_exc()
                self.after(0, lambda: self.update_status(f"提取异常: {str(e)}", "red"))

        threading.Thread(target=background_extraction, daemon=True).start()

    def _display_features(self, modality):
        self.update_status(f"{modality} 特征提取完成", "green")
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

        for name, val in flat.items():
            if isinstance(val, float):
                val_str = f"{val:.6f}"
            elif isinstance(val, (list, np.ndarray)):
                val_str = f"Array {np.shape(val)}"
            else:
                val_str = str(val)
            self.tree.insert('', "end", values=(name, val_str))


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