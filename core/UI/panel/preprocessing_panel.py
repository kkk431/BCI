# -*- coding: utf-8 -*-
# isort: skip_file
# flake8: noqa
"""
智融脑机 - 信号预处理模块 GUI
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import traceback
import numpy as np
from pathlib import Path
import sys
import json
from PIL import Image, ImageTk  # 引入Pillow库处理图片及透明度

# 将项目根目录动态添加到 sys.path
start_path = Path(__file__).resolve().parent
for parent in [start_path] + list(start_path.parents):
    if parent.name == 'core':
        project_root = parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            print(f"已将项目根目录 {project_root} 添加到 sys.path")
        break
else:
    project_root = Path.cwd()

# --- 导入底层业务模块 ---
try:
    from core.io.data_io import DataLoader
    from core.processing.preprocessing.multimodal_preprocessing import (
        MultiModalPreprocessor,
        MultiModalConfig,
        MultiModalConfigFactory,
        ProcessingMode,
        TimeSyncMethod,
    )
    from core.processing.preprocessing.eeg_preprocessing import (
        EEGPreprocessingConfig,
        ReferenceType,
        ICAMethod,
        ArtifactRemovalMethod,
    )
    from core.processing.preprocessing.emg_preprocessing import (
        EMGPreprocessingConfig,
        RectificationMethod,
        EnvelopeExtractionMethod,
        MuscleActivationDetectionMethod,
    )
    from core.processing.preprocessing.ecg_preprocessing import ECGConfig, ECGQualityFlag
    from core.processing.preprocessing.fnirs_preprocessing import (
        fNIRSConfig,
        MotionCorrectionMethod,
        OpticalModel,
    )
    from core.processing.preprocessing.preprocessing import (
        GeneralPreprocessor,
        PreprocessingConfig,
        FilterType,
        WaveletType,
        DetrendMethod,
    )

    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    print(f"模块导入失败: {e}")
    traceback.print_exc()

# --- 视觉配置 ---
BG_COLOR = "#ffffff"  # 为匹配UI图片内部颜色，统一为白色
FG_COLOR = "#333333"

# 更美观的字体定义（跨平台无衬线字体族）
FONT_TITLE = ("Segoe UI", 18, "bold")      # 标题字体
FONT_HEADING = ("Segoe UI", 14, "bold")    # 副标题字体
FONT_NORMAL = ("Segoe UI", 11)             # 常规字体
FONT_SMALL = ("Segoe UI", 10)              # 小号字体

ALL_MODALITIES = ["EEG", "EMG", "ECG", "fNIRS"]

# --- 参数映射配置保持不变 ---
EEG_PARAMS = {
    "use_highpass": {"type": "bool", "default": True, "label": "启用高通滤波"},
    "highpass_freq": {"type": "float", "default": 0.5, "label": "高通频率 (Hz)"},
    "use_lowpass": {"type": "bool", "default": True, "label": "启用低通滤波"},
    "lowpass_freq": {"type": "float", "default": 45.0, "label": "低通频率 (Hz)"},
    "filter_order": {"type": "int", "default": 4, "label": "滤波器阶数"},
    "filter_type": {"type": "choice", "options": [e.value for e in FilterType] if MODULES_LOADED else ["butterworth"],
                    "default": "butterworth", "label": "滤波器类型"},
    "line_freq": {"type": "float", "default": 50.0, "label": "工频频率 (Hz)"},
    "use_ica": {"type": "bool", "default": True, "label": "使用 ICA 去除伪迹"},
    "ica_method": {"type": "choice", "options": [e.value for e in ICAMethod] if MODULES_LOADED else ["infomax"],
                   "default": "infomax", "label": "ICA 方法"},
    "ica_n_components": {"type": "float", "default": 0.95, "label": "ICA 成分数 (比例/整数)"},
    "reference_type": {"type": "choice", "options": [e.value for e in ReferenceType] if MODULES_LOADED else ["average"],
                       "default": "average", "label": "重参考类型"},
    "interpolate_bad_channels": {"type": "bool", "default": True, "label": "插值坏道"},
    "bad_channel_threshold": {"type": "float", "default": 3.0, "label": "坏道检测阈值 (std倍数)"},
    "reject_by_amplitude": {"type": "bool", "default": True, "label": "振幅伪迹拒绝"},
    "rejection_threshold": {"type": "float", "default": 150e-6, "label": "拒绝阈值 (V)"},
    "target_sampling_rate": {"type": "float", "default": 250.0, "label": "降采样至 (Hz, 0=不降)"},
    "use_harmonic_notch": {"type": "bool", "default": False, "label": "谐波陷波"},
    "notch_harmonics": {"type": "int", "default": 5, "label": "谐波数量"},
}

EMG_PARAMS = {
    "emg_bandpass_low": {"type": "float", "default": 20.0, "label": "带通低截止 (Hz)"},
    "emg_bandpass_high": {"type": "float", "default": 450.0, "label": "带通高截止 (Hz)"},
    "emg_bandpass_order": {"type": "int", "default": 4, "label": "滤波器阶数"},
    "filter_type": {"type": "choice", "options": [e.value for e in FilterType] if MODULES_LOADED else ["butterworth"],
                    "default": "butterworth", "label": "滤波器类型"},
    "line_frequency": {"type": "float", "default": 50.0, "label": "工频频率 (Hz)"},
    "use_harmonic_notch": {"type": "bool", "default": True, "label": "谐波陷波"},
    "notch_harmonics": {"type": "int", "default": 5, "label": "谐波数量"},
    "rectification_method": {"type": "choice",
                             "options": [e.value for e in RectificationMethod] if MODULES_LOADED else ["full_wave"],
                             "default": "full_wave", "label": "整流方法"},
    "envelope_method": {"type": "choice",
                        "options": [e.value for e in EnvelopeExtractionMethod] if MODULES_LOADED else ["lowpass"],
                        "default": "lowpass", "label": "包络提取方法"},
    "envelope_cutoff": {"type": "float", "default": 5.0, "label": "包络截止频率 (Hz)"},
    "envelope_order": {"type": "int", "default": 4, "label": "包络滤波器阶数"},
    "remove_motion_artifacts": {"type": "bool", "default": True, "label": "去除运动伪迹"},
    "motion_artifact_threshold": {"type": "float", "default": 5.0, "label": "运动伪迹阈值 (std倍数)"},
    "detect_muscle_activation": {"type": "bool", "default": False, "label": "检测肌肉激活"},
    "activation_threshold": {"type": "float", "default": 2.0, "label": "激活阈值 (std倍数)"},
    "downsample_to": {"type": "float", "default": 1000.0, "label": "降采样至 (Hz, 0=不降)"},
    "normalize_method": {"type": "choice", "options": ["zscore", "minmax", "robust", "none"], "default": "zscore",
                         "label": "标准化方法"},
}

ECG_PARAMS = {
    "ecg_lowcut": {"type": "float", "default": 0.5, "label": "低截止 (Hz)"},
    "ecg_highcut": {"type": "float", "default": 40.0, "label": "高截止 (Hz)"},
    "ecg_notch_freq": {"type": "float", "default": 50.0, "label": "陷波频率 (Hz)"},
    "filter_order": {"type": "int", "default": 4, "label": "滤波器阶数"},
    "protect_qrs_wave": {"type": "bool", "default": True, "label": "保护QRS波形"},
    "qrs_enhancement": {"type": "bool", "default": True, "label": "增强QRS波"},
    "assess_signal_quality": {"type": "bool", "default": True, "label": "评估信号质量"},
    "detect_bad_segments": {"type": "bool", "default": True, "label": "检测坏段"},
    "multi_lead_consistency": {"type": "bool", "default": True, "label": "多导联一致性检查"},
    "target_sampling_rate": {"type": "float", "default": 250.0, "label": "降采样至 (Hz, 0=不降)"},
}

FNIRS_PARAMS = {
    "wavelengths": {"type": "str", "default": "730,850", "label": "波长 (逗号分隔 nm)"},
    "optical_model": {"type": "choice",
                      "options": [e.value for e in OpticalModel] if MODULES_LOADED else ["modified_beer_lambert"],
                      "default": "modified_beer_lambert", "label": "光学模型"},
    "motion_correction_method": {"type": "choice",
                                 "options": [e.value for e in MotionCorrectionMethod] if MODULES_LOADED else ["spline"],
                                 "default": "spline", "label": "运动校正方法"},
    "motion_correction_threshold": {"type": "float", "default": 3.0, "label": "运动检测阈值"},
    "pca_components_to_remove": {"type": "int", "default": 3, "label": "PCA 移除成分数"},
    "hemodynamic_lowcut": {"type": "float", "default": 0.01, "label": "血氧低通 (Hz)"},
    "hemodynamic_highcut": {"type": "float", "default": 0.5, "label": "血氧高通 (Hz)"},
    "use_short_channel_regression": {"type": "bool", "default": True, "label": "短通道回归"},
    "short_channel_distance_threshold": {"type": "float", "default": 1.0, "label": "短通道距离阈值 (cm)"},
    "remove_physiological_noise": {"type": "bool", "default": True, "label": "去除生理噪声"},
    "cardiac_frequency_range": {"type": "str", "default": "0.8,2.0", "label": "心搏频率范围 (Hz,逗号)"},
    "respiration_frequency_range": {"type": "str", "default": "0.1,0.5", "label": "呼吸频率范围 (Hz,逗号)"},
    "snr_threshold": {"type": "float", "default": 20.0, "label": "SNR 阈值 (dB)"},
    "target_sampling_rate": {"type": "float", "default": 10.0, "label": "降采样至 (Hz, 0=不降)"},
}

GLOBAL_PARAMS = {
    "processing_mode": {"type": "choice", "options": ["sequential", "parallel"], "default": "sequential",
                        "label": "处理模式"},
    "time_sync_method": {"type": "choice", "options": ["none", "resample", "interpolate"], "default": "none",
                         "label": "时间同步方法"},
    "reference_sampling_rate": {"type": "float", "default": 250.0, "label": "参考采样率 (Hz)"},
    "max_workers": {"type": "int", "default": 4, "label": "最大线程数"},
    "quality_check_enabled": {"type": "bool", "default": True, "label": "启用质量检查"},
    "auto_fix_issues": {"type": "bool", "default": True, "label": "自动修复问题"},
}


class ManualModalityDialog(tk.Toplevel):
    """手动指定模态对话框"""

    def __init__(self, parent, detected_mods):
        super().__init__(parent)
        self.parent = parent
        self.detected_mods = detected_mods
        self.result = {}
        self.title("手动指定模态")
        self.geometry("500x400")
        self.transient(parent)
        self.grab_set()
        self.configure(bg=BG_COLOR)

        tk.Label(self, text="以下信号无法自动识别模态，请手动指定：", font=FONT_HEADING, bg=BG_COLOR).pack(pady=10)

        self.frame = tk.Frame(self, bg=BG_COLOR)
        self.frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.widgets = {}
        for i, mod_name in enumerate(self.detected_mods):
            if mod_name.upper() in ALL_MODALITIES or mod_name in ALL_MODALITIES:
                continue
            row = tk.Frame(self.frame, bg=BG_COLOR)
            row.pack(fill="x", pady=5)
            tk.Label(row, text=f"信号 '{mod_name}':", width=20, anchor="w", bg=BG_COLOR).pack(side="left")
            combo = ttk.Combobox(row, values=ALL_MODALITIES, state="readonly")
            combo.pack(side="left", padx=5)
            combo.current(0)
            self.widgets[mod_name] = combo

        if not self.widgets:
            tk.Label(self.frame, text="没有需要手动指定的模态", bg=BG_COLOR).pack()

        btn_frame = tk.Frame(self, bg=BG_COLOR)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="确认", command=self.on_ok, bg="#4f8080", fg="white").pack(side="left", padx=5)
        tk.Button(btn_frame, text="取消", command=self.destroy).pack(side="left", padx=5)

    def on_ok(self):
        for orig, combo in self.widgets.items():
            self.result[orig] = combo.get()
        self.destroy()


class PreprocessingApp(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, bg=BG_COLOR, *args, **kwargs)
        self.parent = parent

        self.raw_data_dict = None
        self.processed_data_dict = None
        self.current_filepath = None

        self.is_processing = False
        self.is_data_loaded = False

        self.modality_vars = {mod: tk.BooleanVar(value=False) for mod in ALL_MODALITIES}
        self.eeg_widgets = {}
        self.emg_widgets = {}
        self.ecg_widgets = {}
        self.fnirs_widgets = {}
        self.global_widgets = {}

        # 用于显示处理结果的标签变量
        self.result_vars = {}

        if not MODULES_LOADED:
            messagebox.showerror("初始化失败", "底层算法模块导入失败，请检查终端输出的路径信息。")

        self.images = {}
        self._load_all_images()
        self.setup_ui()

    def _load_all_images(self):
        """预加载UI图片，并存放到字典中防止被垃圾回收"""
        img_map = {
            # 新增全局背景图片
            "global_bg": "core/UI/UI_resource/Global_Background.png",

            "nav_bg": "core/UI/UI_resource/Navigation/Background.png",
            "nav_home_unsel": "core/UI/UI_resource/Navigation/Buttons/Unselected/Home_Button.png",
            "nav_prep_unsel": "core/UI/UI_resource/Navigation/Buttons/Unselected/Preprocessing_Button.png",
            "nav_feat_unsel": "core/UI/UI_resource/Navigation/Buttons/Unselected/Feature_Extraction_Button.png",
            "nav_stat_unsel": "core/UI/UI_resource/Navigation/Buttons/Unselected/Statistical_Analysis_Button.png",
            "nav_vis_unsel": "core/UI/UI_resource/Navigation/Buttons/Unselected/Virtualization_Button.png",
            "nav_home_sel": "core/UI/UI_resource/Navigation/Buttons/Selected/Home_Button.png",
            "nav_prep_sel": "core/UI/UI_resource/Navigation/Buttons/Selected/Preprocessing_Button.png",
            "nav_feat_sel": "core/UI/UI_resource/Navigation/Buttons/Selected/Feature_Extraction_Button.png",
            "nav_stat_sel": "core/UI/UI_resource/Navigation/Buttons/Selected/Statistical_Analysis_Button.png",
            "nav_vis_sel": "core/UI/UI_resource/Navigation/Buttons/Selected/Virtualization_Button.png",

            "prep_bg": "core/UI/UI_resource/Preprocessing_Panel/Setting_Bar_Background.png",
            "global_set": "core/UI/UI_resource/Preprocessing_Panel/Global_Settings.png",
            "detail_set": "core/UI/UI_resource/Preprocessing_Panel/Detailed_Settings.png",
            "proc_info": "core/UI/UI_resource/Preprocessing_Panel/Processing_Information.png",
            "file_bar": "core/UI/UI_resource/Preprocessing_Panel/File_Address_Bar.png",
            "btn_browse": "core/UI/UI_resource/Preprocessing_Panel/Buttons/Browse.png",
            "btn_save": "core/UI/UI_resource/Preprocessing_Panel/Buttons/Save.png",
            "btn_start": "core/UI/UI_resource/Preprocessing_Panel/Buttons/Start.png",

            "eeg_unsel": "core/UI/UI_resource/Model_Selection_Buttons/Unselected/Eeg.png",
            "ecg_unsel": "core/UI/UI_resource/Model_Selection_Buttons/Unselected/Ecg.png",
            "emg_unsel": "core/UI/UI_resource/Model_Selection_Buttons/Unselected/Emg.png",
            "fnirs_unsel": "core/UI/UI_resource/Model_Selection_Buttons/Unselected/Fnirs.png",

            "eeg_sel": "core/UI/UI_resource/Model_Selection_Buttons/Selected/Eeg.png",
            "ecg_sel": "core/UI/UI_resource/Model_Selection_Buttons/Selected/Ecg.png",
            "emg_sel": "core/UI/UI_resource/Model_Selection_Buttons/Selected/Emg.png",
            "fnirs_sel": "core/UI/UI_resource/Model_Selection_Buttons/Selected/Fnirs.png",
        }

        for key, rel_path in img_map.items():
            full_path = project_root.joinpath(*rel_path.split('/'))
            try:
                img = Image.open(full_path)
                self.images[key] = ImageTk.PhotoImage(img)
            except Exception as e:
                print(f"警告: 无法加载图片 {full_path}: {e}")
                img = Image.new("RGBA", (10, 10), (255, 0, 0, 0))
                self.images[key] = ImageTk.PhotoImage(img)

    def bind_btn(self, item_id, command):
        """给 Canvas 对象绑定点击和鼠标悬停（光标变成手型）"""
        self.canvas.tag_bind(item_id, "<Button-1>", lambda e: command())
        self.canvas.tag_bind(item_id, "<Enter>", lambda e: self.canvas.config(cursor="hand2"))
        self.canvas.tag_bind(item_id, "<Leave>", lambda e: self.canvas.config(cursor=""))

    def setup_ui(self):
        # 使用整屏 Canvas 进行底层布局
        self.canvas = tk.Canvas(self, width=1440, height=1024, bg=BG_COLOR, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # ------------------- 绘制静态 UI 背景 -------------------
        # 先绘制全局背景 (left:289, top:0, width:1151, height:1024)
        self.canvas.create_image(289, 0, image=self.images["global_bg"], anchor="nw")

        # 其他背景图片
        self.canvas.create_image(9, 9, image=self.images["nav_bg"], anchor="nw")
        self.canvas.create_image(297, 9, image=self.images["prep_bg"], anchor="nw")
        self.canvas.create_image(511, 31, image=self.images["file_bar"], anchor="nw")
        self.canvas.create_image(311, 174, image=self.images["global_set"], anchor="nw")
        self.canvas.create_image(870, 167, image=self.images["detail_set"], anchor="nw")
        self.canvas.create_image(298, 772, image=self.images["proc_info"], anchor="nw")

        # ------------------- 初始化按钮和组件 -------------------
        self._create_nav_buttons()
        self._create_action_buttons()
        self._create_modality_buttons()

        # 显示文件路径信息的文本 Label
        self.lbl_status = tk.Label(self, text="请上传文件...", font=FONT_NORMAL, bg="#ffffff", fg="#666666", anchor="w")
        self.lbl_status.place(x=530, y=42, width=860, height=35)

        # ========== 1. 全局设置区域 ==========
        # 增大高度以容纳更大的行间距
        self.global_frame = tk.Frame(self, bg=BG_COLOR)
        self.global_frame.place(x=331, y=220, width=504, height=260)  # 高度从220增至260
        # 创建参数控件，行间距设为8，使文字更舒适
        self._create_param_widgets(self.global_frame, GLOBAL_PARAMS, self.global_widgets, pady=8)

        # ========== 2. 详细设置区域 ==========
        # 再下移一点，避免遮挡
        self.detail_frame = tk.Frame(self, bg=BG_COLOR)
        self.detail_frame.place(x=890, y=230, width=508, height=500)  # y从210改为230

        # 创建四个模态面板，全部放入 detail_frame 同一位置，初始隐藏
        self.modality_panels = {}
        for mod in ALL_MODALITIES:
            if mod == "EEG":
                panel = self._create_modality_tab(self.detail_frame, mod, EEG_PARAMS)
            elif mod == "EMG":
                panel = self._create_modality_tab(self.detail_frame, mod, EMG_PARAMS)
            elif mod == "ECG":
                panel = self._create_modality_tab(self.detail_frame, mod, ECG_PARAMS)
            elif mod == "fNIRS":
                panel = self._create_modality_tab(self.detail_frame, mod, FNIRS_PARAMS)
            self.modality_panels[mod] = panel
            panel.place(x=0, y=0, width=508, height=500)
            panel.place_forget()

        # ========== 3. 处理信息区域 ==========
        # 使用普通 Frame，无滚动条；位置 x=318, y=825（比原820下移5像素），宽度900，高度280
        self.proc_frame = tk.Frame(self, bg=BG_COLOR)
        self.proc_frame.place(x=318, y=825, width=900, height=280)

        # 内部内容容器
        self.proc_content = tk.Frame(self.proc_frame, bg=BG_COLOR)
        self.proc_content.pack(fill="both", expand=True)

        # 在 content 中创建用于显示处理结果的标签行
        self._create_result_display()

    def _create_nav_buttons(self):
        self.nav_items = {}
        nav_coords = {
            "home": (39, 231),
            "prep": (39, 323),
            "feat": (39, 415),
            "stat": (39, 507),
            "vis": (39, 599)
        }

        # 默认停留选中“预处理”
        self.current_nav = "prep"
        for name, (x, y) in nav_coords.items():
            img_key = f"nav_{name}_sel" if name == "prep" else f"nav_{name}_unsel"
            item_id = self.canvas.create_image(x, y, image=self.images[img_key], anchor="nw")
            self.nav_items[name] = item_id
            self.bind_btn(item_id, lambda n=name: self.on_nav_click(n))

    def on_nav_click(self, name):
        """左侧导航栏场景切换（仅留接口及高亮变化）"""
        if name == self.current_nav:
            return
        # 切换高亮图片
        self.canvas.itemconfig(self.nav_items[self.current_nav], image=self.images[f"nav_{self.current_nav}_unsel"])
        self.canvas.itemconfig(self.nav_items[name], image=self.images[f"nav_{name}_sel"])
        self.current_nav = name
        # TODO: 后续预留接入其他功能场景（特征提取/分析等）

    def _create_action_buttons(self):
        self.btn_browse = self.canvas.create_image(311, 102, image=self.images["btn_browse"], anchor="nw")
        self.bind_btn(self.btn_browse, self.action_load_data)

        self.btn_start = self.canvas.create_image(349, 525, image=self.images["btn_start"], anchor="nw")
        self.bind_btn(self.btn_start, self.action_run_preprocessing)

        self.btn_save = self.canvas.create_image(590, 525, image=self.images["btn_save"], anchor="nw")
        self.bind_btn(self.btn_save, self.action_save_results)

    def _create_modality_buttons(self):
        self.mod_items = {}
        coords = {"EEG": 653, "EMG": 838, "ECG": 1023, "fNIRS": 1208}

        for mod, x in coords.items():
            img_key = f"{mod.lower()}_unsel"
            item_id = self.canvas.create_image(x, 109, image=self.images[img_key], anchor="nw")
            self.mod_items[mod] = item_id
            self.bind_btn(item_id, lambda m=mod: self.on_mod_btn_click(m))

    def on_mod_btn_click(self, mod):
        """点击模态选择图片开关时的触发逻辑（改为单选模式）"""
        if self.is_processing:
            return

        # 获取当前点击模态的选中状态
        current = self.modality_vars[mod].get()

        if current:
            # 如果当前是选中状态，则点击后取消所有选中
            for m in ALL_MODALITIES:
                self.modality_vars[m].set(False)
        else:
            # 如果当前是未选中状态，则只选中当前模态，取消其他所有
            for m in ALL_MODALITIES:
                self.modality_vars[m].set(False)
            self.modality_vars[mod].set(True)

        # 更新所有模态按钮的图片
        for m in ALL_MODALITIES:
            img_key = f"{m.lower()}_sel" if self.modality_vars[m].get() else f"{m.lower()}_unsel"
            self.canvas.itemconfig(self.mod_items[m], image=self.images[img_key])

        # 联动右侧设置面板显示
        self.on_modality_toggle(mod)

    def _create_param_widgets(self, parent, param_dict, widget_dict, pady=6):
        """动态生成参数控件面板，可自定义行间距pady"""
        parent.columnconfigure(1, weight=1)
        row = 0
        for key, spec in param_dict.items():
            label = ttk.Label(parent, text=spec["label"] + ":", font=FONT_NORMAL, background=BG_COLOR)
            label.grid(row=row, column=0, sticky="w", padx=10, pady=pady)

            if spec["type"] == "bool":
                var = tk.BooleanVar(value=spec["default"])
                cb = tk.Checkbutton(
                    parent,
                    variable=var,
                    bg=BG_COLOR,
                    selectcolor=BG_COLOR,
                    activebackground=BG_COLOR,
                    highlightthickness=0
                )
                cb.grid(row=row, column=1, sticky="w", padx=10)
                widget_dict[key] = var
            elif spec["type"] == "choice":
                var = tk.StringVar(value=spec["default"])
                combo = ttk.Combobox(parent, textvariable=var, values=spec["options"], state="readonly", width=25)
                combo.grid(row=row, column=1, sticky="w", padx=10)
                widget_dict[key] = var
            elif spec["type"] in ("float", "int", "str"):
                var = tk.StringVar(value=str(spec["default"]))
                entry = ttk.Entry(parent, textvariable=var, width=27)
                entry.grid(row=row, column=1, sticky="w", padx=10)
                widget_dict[key] = var
            row += 1

    def _create_modality_tab(self, parent, mod_name, param_dict):
        """创建单个带有滚轮效果的模态设置页面，父容器为parent"""
        tab = tk.Frame(parent, bg=BG_COLOR)
        canvas = tk.Canvas(tab, borderwidth=0, highlightthickness=0, bg=BG_COLOR)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=BG_COLOR)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind('<Enter>', lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind('<Leave>', lambda e: canvas.unbind_all("<MouseWheel>"))

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 填充参数控件，使用稍小的行间距（4）以适应更多参数
        if mod_name == "EEG":
            self._create_param_widgets(scrollable_frame, param_dict, self.eeg_widgets, pady=4)
        elif mod_name == "EMG":
            self._create_param_widgets(scrollable_frame, param_dict, self.emg_widgets, pady=4)
        elif mod_name == "ECG":
            self._create_param_widgets(scrollable_frame, param_dict, self.ecg_widgets, pady=4)
        elif mod_name == "fNIRS":
            self._create_param_widgets(scrollable_frame, param_dict, self.fnirs_widgets, pady=4)

        return tab

    def _create_result_display(self):
        """创建处理结果展示区域，采用标签-数值对形式（无滚动条）"""
        # 预定义要显示的结果项
        result_items = [
            ("处理时间", "processing_time"),
            ("成功状态", "success"),
            ("处理模态", "modalities_processed"),
            ("总体质量", "overall_quality"),
            # 模态质量将在运行时动态添加
        ]
        # 使用grid布局，两列
        self.proc_content.columnconfigure(1, weight=1)
        row = 0
        for label_text, key in result_items:
            lbl = ttk.Label(self.proc_content, text=label_text + ":", font=FONT_NORMAL, background=BG_COLOR)
            lbl.grid(row=row, column=0, sticky="w", padx=10, pady=6)
            var = tk.StringVar(value="")
            val_lbl = ttk.Label(self.proc_content, textvariable=var, font=FONT_NORMAL, background=BG_COLOR)
            val_lbl.grid(row=row, column=1, sticky="w", padx=10, pady=6)
            self.result_vars[key] = var
            row += 1

        # 为模态质量预留动态添加的空间，用一个字典记录每个模态的质量变量
        self.mod_quality_vars = {}  # mod -> var

    def _update_result_display(self, result):
        """根据预处理结果更新显示"""
        # 基本项
        self.result_vars["processing_time"].set(f"{result.processing_time:.2f} 秒")
        self.result_vars["success"].set(str(result.success))
        # 处理模态
        if "processed" in result.processed_data and "multimodal_preprocessing" in result.processed_data["processed"]:
            mods = result.processed_data["processed"]["multimodal_preprocessing"].get("modalities_processed", [])
        else:
            mods = []
        self.result_vars["modalities_processed"].set(", ".join(mods) if mods else "无")

        # 质量报告
        quality_report = result.processed_data.get("processed", {}).get("quality_report", {})
        overall = quality_report.get("overall_quality", 0)
        self.result_vars["overall_quality"].set(f"{overall:.2f}")

        # 模态质量
        mod_quality = quality_report.get("modality_quality", {})
        # 删除之前动态添加的模态质量行（如果有）
        for widget in self.proc_content.grid_slaves():
            if int(widget.grid_info()["row"]) >= len(self.result_vars):  # 基础行之后的是动态添加的
                widget.destroy()
        self.mod_quality_vars.clear()

        # 重新添加
        row = len(self.result_vars)
        for mod, mq in mod_quality.items():
            lbl = ttk.Label(self.proc_content, text=f"{mod} 质量分数:", font=FONT_NORMAL, background=BG_COLOR)
            lbl.grid(row=row, column=0, sticky="w", padx=10, pady=6)
            var = tk.StringVar(value=f"{mq.get('quality_score', 0):.2f}")
            val_lbl = ttk.Label(self.proc_content, textvariable=var, font=FONT_NORMAL, background=BG_COLOR)
            val_lbl.grid(row=row, column=1, sticky="w", padx=10, pady=6)
            self.mod_quality_vars[mod] = var
            row += 1

    def _update_visible_panel(self):
        """根据当前选中的模态，显示第一个选中的面板，隐藏其他"""
        selected = [mod for mod, var in self.modality_vars.items() if var.get()]
        # 隐藏所有面板
        for panel in self.modality_panels.values():
            panel.place_forget()
        # 如果有选中，显示第一个
        if selected:
            first_mod = selected[0]
            self.modality_panels[first_mod].place(x=0, y=0, width=508, height=500)

    def on_modality_toggle(self, mod=None):
        """更新模态选中状态及右侧面板显示"""
        # 更新面板显示
        self._update_visible_panel()

    # ================= 核心业务功能逻辑接入点 =================

    def action_load_data(self):
        if self.is_processing: return
        if not MODULES_LOADED:
            messagebox.showerror("错误", "底层模块未加载，请检查文件路径是否正确。")
            return

        filepath = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("支持的格式", "*.csv *.mat *.edf *.bdf *.npy *.npz *.snirf *.set *.pkl *.json"),
                       ("All Files", "*.*")]
        )
        if not filepath:
            return

        self.current_filepath = filepath
        self._start_data_loading(filepath)

    def _start_data_loading(self, filepath):
        self.is_processing = True
        self.lbl_status.config(text=f"正在加载数据...", fg="#0056b3")

        def background_task():
            try:
                loader = DataLoader()
                raw_dict = loader.load(filepath)

                detected_mods = set()
                unknown_mods = []
                if 'signal' in raw_dict:
                    for mod in raw_dict['signal'].keys():
                        mod_upper = mod.upper()
                        if mod_upper in ALL_MODALITIES:
                            detected_mods.add(mod_upper)
                        elif mod_upper == 'FNIRS':
                            detected_mods.add('fNIRS')
                        else:
                            unknown_mods.append(mod)

                if unknown_mods:
                    self.after(0, lambda: self._show_manual_modality_dialog(unknown_mods, raw_dict, detected_mods))
                else:
                    self.after(0, self._update_modality_checkboxes, detected_mods, raw_dict)

            except Exception as e:
                traceback.print_exc()
                self.after(0, self._ui_update_on_error, f"加载失败: {str(e)}")

        threading.Thread(target=background_task, daemon=True).start()

    def _show_manual_modality_dialog(self, unknown_mods, raw_dict, detected_mods):
        dialog = ManualModalityDialog(self, unknown_mods)
        self.wait_window(dialog)
        if dialog.result:
            for orig, std_mod in dialog.result.items():
                if orig in raw_dict['signal']:
                    raw_dict['signal'][std_mod] = raw_dict['signal'].pop(orig)
                    detected_mods.add(std_mod)
        self._update_modality_checkboxes(detected_mods, raw_dict)

    def _update_modality_checkboxes(self, detected_mods, raw_dict):
        """自动更新复选框状态和图片UI"""
        for mod, var in self.modality_vars.items():
            is_detected = (mod in detected_mods)
            var.set(is_detected)
            # 更新对应模态的图片UI
            img_key = f"{mod.lower()}_sel" if is_detected else f"{mod.lower()}_unsel"
            self.canvas.itemconfig(self.mod_items[mod], image=self.images[img_key])

        # 更新右侧面板显示
        self.on_modality_toggle()

        self.raw_data_dict = raw_dict
        filename = Path(self.current_filepath).name
        self.lbl_status.config(text=f"文件: {filename}   |   已检测出模态: {', '.join(detected_mods)}", fg="#4f8080")
        self.is_data_loaded = True
        self.is_processing = False

    def _ui_update_on_error(self, error_msg):
        self.lbl_status.config(text=error_msg, fg="red")
        self.is_processing = False

    def action_run_preprocessing(self):
        if self.is_processing: return

        selected_mods = [mod for mod, var in self.modality_vars.items() if var.get()]
        if not selected_mods:
            messagebox.showwarning("警告", "请至少选择一个模态进行预处理")
            return

        if self.raw_data_dict is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        self.is_processing = True
        self.lbl_status.config(text="正在执行预处理，请稍候...", fg="#0056b3")

        # 清空之前的显示
        for var in self.result_vars.values():
            var.set("")
        for var in self.mod_quality_vars.values():
            var.set("")
        # 删除模态质量行（会在_update_result_display中重建）

        def background_task():
            try:
                config = self._build_multimodal_config(selected_mods)
                preprocessor = MultiModalPreprocessor(config)
                result = preprocessor.process(self.raw_data_dict)

                if not result.success:
                    raise Exception(f"预处理失败: {result.error_message}")

                self.processed_data_dict = result.processed_data
                self.after(0, self._ui_update_on_success, result)

            except Exception as e:
                traceback.print_exc()
                self.after(0, self._ui_update_on_error, f"预处理异常: {str(e)}")

        threading.Thread(target=background_task, daemon=True).start()

    def _build_multimodal_config(self, selected_mods):
        processing_mode_str = self.global_widgets["processing_mode"].get()
        processing_mode = ProcessingMode.SEQUENTIAL if processing_mode_str == "sequential" else ProcessingMode.PARALLEL

        time_sync_str = self.global_widgets["time_sync_method"].get()
        time_sync = TimeSyncMethod.NONE
        if time_sync_str == "resample":
            time_sync = TimeSyncMethod.RESAMPLE
        elif time_sync_str == "interpolate":
            time_sync = TimeSyncMethod.INTERPOLATE

        ref_sr = float(self.global_widgets["reference_sampling_rate"].get() or 250)
        max_workers = int(self.global_widgets["max_workers"].get() or 4)
        quality_check = self.global_widgets["quality_check_enabled"].get()
        auto_fix = self.global_widgets["auto_fix_issues"].get()

        eeg_config, emg_config, ecg_config, fnirs_config = None, None, None, None

        if "EEG" in selected_mods:
            eeg_config = EEGPreprocessingConfig(
                use_highpass=self.eeg_widgets["use_highpass"].get(),
                highpass_freq=float(self.eeg_widgets["highpass_freq"].get()),
                use_lowpass=self.eeg_widgets["use_lowpass"].get(),
                lowpass_freq=float(self.eeg_widgets["lowpass_freq"].get()),
                filter_order=int(self.eeg_widgets["filter_order"].get()),
                filter_type=FilterType(self.eeg_widgets["filter_type"].get()),
                line_freq=float(self.eeg_widgets["line_freq"].get()),
                use_ica=self.eeg_widgets["use_ica"].get(),
                ica_method=ICAMethod(self.eeg_widgets["ica_method"].get()),
                ica_n_components=float(self.eeg_widgets["ica_n_components"].get()),
                reference_type=ReferenceType(self.eeg_widgets["reference_type"].get()),
                interpolate_bad_channels=self.eeg_widgets["interpolate_bad_channels"].get(),
                bad_channel_threshold=float(self.eeg_widgets["bad_channel_threshold"].get()),
                reject_by_amplitude=self.eeg_widgets["reject_by_amplitude"].get(),
                rejection_threshold=float(self.eeg_widgets["rejection_threshold"].get()),
                target_sampling_rate=float(self.eeg_widgets["target_sampling_rate"].get()) if float(
                    self.eeg_widgets["target_sampling_rate"].get()) > 0 else None,
                use_harmonic_notch=self.eeg_widgets["use_harmonic_notch"].get(),
                harmonic_notch_n_harmonics=int(self.eeg_widgets["notch_harmonics"].get()),
            )

        if "EMG" in selected_mods:
            emg_config = EMGPreprocessingConfig(
                emg_bandpass_low=float(self.emg_widgets["emg_bandpass_low"].get()),
                emg_bandpass_high=float(self.emg_widgets["emg_bandpass_high"].get()),
                emg_bandpass_order=int(self.emg_widgets["emg_bandpass_order"].get()),
                filter_type=FilterType(self.emg_widgets["filter_type"].get()),
                line_frequency=float(self.emg_widgets["line_frequency"].get()),
                use_harmonic_notch=self.emg_widgets["use_harmonic_notch"].get(),
                notch_harmonics=int(self.emg_widgets["notch_harmonics"].get()),
                rectification_method=RectificationMethod(self.emg_widgets["rectification_method"].get()),
                envelope_method=EnvelopeExtractionMethod(self.emg_widgets["envelope_method"].get()),
                envelope_cutoff=float(self.emg_widgets["envelope_cutoff"].get()),
                envelope_order=int(self.emg_widgets["envelope_order"].get()),
                remove_motion_artifacts=self.emg_widgets["remove_motion_artifacts"].get(),
                motion_artifact_threshold=float(self.emg_widgets["motion_artifact_threshold"].get()),
                detect_muscle_activation=self.emg_widgets["detect_muscle_activation"].get(),
                activation_threshold=float(self.emg_widgets["activation_threshold"].get()),
                downsample_to=float(self.emg_widgets["downsample_to"].get()) if float(
                    self.emg_widgets["downsample_to"].get()) > 0 else None,
                normalize_method=self.emg_widgets["normalize_method"].get(),
            )

        if "ECG" in selected_mods:
            ecg_config = ECGConfig(
                ecg_lowcut=float(self.ecg_widgets["ecg_lowcut"].get()),
                ecg_highcut=float(self.ecg_widgets["ecg_highcut"].get()),
                ecg_notch_freq=float(self.ecg_widgets["ecg_notch_freq"].get()),
                filter_order=int(self.ecg_widgets["filter_order"].get()),
                protect_qrs_wave=self.ecg_widgets["protect_qrs_wave"].get(),
                qrs_enhancement=self.ecg_widgets["qrs_enhancement"].get(),
                assess_signal_quality=self.ecg_widgets["assess_signal_quality"].get(),
                detect_bad_segments=self.ecg_widgets["detect_bad_segments"].get(),
                multi_lead_consistency=self.ecg_widgets["multi_lead_consistency"].get(),
                target_sampling_rate=float(self.ecg_widgets["target_sampling_rate"].get()) if float(
                    self.ecg_widgets["target_sampling_rate"].get()) > 0 else None,
            )

        if "fNIRS" in selected_mods:
            wl_str = self.fnirs_widgets["wavelengths"].get()
            wavelengths = [float(x.strip()) for x in wl_str.split(",")]
            cardiac_range = [float(x.strip()) for x in self.fnirs_widgets["cardiac_frequency_range"].get().split(",")]
            resp_range = [float(x.strip()) for x in self.fnirs_widgets["respiration_frequency_range"].get().split(",")]

            fnirs_config = fNIRSConfig(
                wavelengths=wavelengths,
                optical_model=OpticalModel(self.fnirs_widgets["optical_model"].get()),
                motion_correction_method=MotionCorrectionMethod(self.fnirs_widgets["motion_correction_method"].get()),
                motion_correction_threshold=float(self.fnirs_widgets["motion_correction_threshold"].get()),
                pca_components_to_remove=int(self.fnirs_widgets["pca_components_to_remove"].get()),
                hemodynamic_lowcut=float(self.fnirs_widgets["hemodynamic_lowcut"].get()),
                hemodynamic_highcut=float(self.fnirs_widgets["hemodynamic_highcut"].get()),
                use_short_channel_regression=self.fnirs_widgets["use_short_channel_regression"].get(),
                short_channel_distance_threshold=float(self.fnirs_widgets["short_channel_distance_threshold"].get()),
                remove_physiological_noise=self.fnirs_widgets["remove_physiological_noise"].get(),
                cardiac_frequency_range=(cardiac_range[0], cardiac_range[1]),
                respiration_frequency_range=(resp_range[0], resp_range[1]),
                snr_threshold=float(self.fnirs_widgets["snr_threshold"].get()),
                target_sampling_rate=float(self.fnirs_widgets["target_sampling_rate"].get()) if float(
                    self.fnirs_widgets["target_sampling_rate"].get()) > 0 else None,
            )

        config = MultiModalConfig(
            processing_mode=processing_mode,
            time_sync_method=time_sync,
            reference_sampling_rate=ref_sr,
            max_workers=max_workers,
            eeg_config=eeg_config,
            emg_config=emg_config,
            ecg_config=ecg_config,
            fnirs_config=fnirs_config,
            enabled_modalities=selected_mods,
            process_all_modalities=False,
            quality_check_enabled=quality_check,
            auto_fix_issues=auto_fix,
        )
        return config

    def _ui_update_on_success(self, result):
        filename = Path(self.current_filepath).name
        self.lbl_status.config(text=f"[{filename}] 预处理成功！处理耗时: {result.processing_time:.2f} 秒", fg="green")
        self.is_processing = False

        # 更新处理信息区域
        self._update_result_display(result)

    def action_save_results(self):
        if self.is_processing: return
        if self.processed_data_dict is None:
            messagebox.showwarning("警告", "没有可保存的处理结果")
            return

        filepath = filedialog.asksaveasfilename(
            title="保存处理结果",
            defaultextension=".npz",
            filetypes=[("NumPy 压缩文件", "*.npz"), ("Pickle 文件", "*.pkl"), ("JSON 文件", "*.json")]
        )
        if not filepath:
            return

        try:
            if filepath.endswith(".npz"):
                signal_data = {}
                for mod, info in self.processed_data_dict.get("signal", {}).items():
                    if "data" in info:
                        signal_data[f"{mod}_data"] = info["data"]
                        signal_data[f"{mod}_fs"] = info.get("sampling_rate", 0)
                np.savez(filepath, **signal_data)
            elif filepath.endswith(".pkl"):
                import pickle
                with open(filepath, "wb") as f:
                    pickle.dump(self.processed_data_dict, f)
            else:
                def convert(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, dict):
                        return {k: convert(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [convert(x) for x in obj]
                    return obj

                serializable = convert(self.processed_data_dict)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(serializable, f, indent=2)

            messagebox.showinfo("成功", f"结果已保存到 {filepath}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))


if __name__ == "__main__":
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    root = tk.Tk()
    root.title("智融脑机 - 信号预处理模块")

    # 根据提供的背景全景进行窗口尺寸限制，拒绝拉伸破坏布局
    root.geometry("1440x1024")
    root.resizable(False, False)
    root.configure(bg=BG_COLOR)

    # 通过 ttk Style 设置统一样式，避免出现底色突兀
    style = ttk.Style()
    style.theme_use("clam")
    
    # 基础样式 - 使用新字体
    style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR, font=FONT_NORMAL)
    style.configure("TFrame", background=BG_COLOR)
    style.configure("TCheckbutton", background=BG_COLOR, font=FONT_NORMAL)
    style.configure("TEntry", font=FONT_NORMAL)

    # 自定义 Combobox 样式 - 使用新字体
    style.configure("TCombobox",
                    fieldbackground="#f0f0f0",      # 下拉框背景
                    background="#f0f0f0",            # 按钮背景
                    foreground=FG_COLOR,              # 文字颜色
                    arrowcolor="#4f8080",             # 箭头颜色（深青色）
                    borderwidth=1,
                    relief="solid",
                    font=FONT_NORMAL)                  # 设置字体
    style.map("TCombobox",
              fieldbackground=[("readonly", "#f0f0f0"), ("disabled", "#e0e0e0")],
              foreground=[("readonly", FG_COLOR)],
              arrowcolor=[("active", "#2a5a5a"), ("!active", "#4f8080")])

    # 自定义 Scrollbar 样式（用于其他区域）
    style.configure("Vertical.TScrollbar",
                    background="#f0f0f0",
                    troughcolor="#e0e0e0",
                    bordercolor="#cccccc",
                    arrowcolor="#4f8080",
                    width=16)
    style.map("Vertical.TScrollbar",
              background=[("active", "#d0d0d0")],
              arrowcolor=[("active", "#2a5a5a")])
    style.configure("Horizontal.TScrollbar",
                    background="#f0f0f0",
                    troughcolor="#e0e0e0",
                    bordercolor="#cccccc",
                    arrowcolor="#4f8080",
                    width=16)
    style.map("Horizontal.TScrollbar",
              background=[("active", "#d0d0d0")],
              arrowcolor=[("active", "#2a5a5a")])

    app = PreprocessingApp(root)
    app.pack(fill=tk.BOTH, expand=True)

    root.mainloop()
