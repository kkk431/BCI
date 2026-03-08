# -*- coding: utf-8 -*-
# isort: skip_file
# flake8: noqa
"""
智融脑机 - 信号预处理模块 GUI (增强版，自动切换标签页，修复摘要显示，修正降采样参数)
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
    raise RuntimeError("未找到名为 'core' 的目录")

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
BG_COLOR = "#f5f5f5"
FG_COLOR = "#333333"
ACCENT_COLOR = "#4f8080"
BUTTON_FG = "white"
BUTTON_BG = ACCENT_COLOR
DISABLED_BG = "#a0a0a0"

FONT_TITLE = ("微软雅黑", 18, "bold")
FONT_HEADING = ("微软雅黑", 14, "bold")
FONT_NORMAL = ("微软雅黑", 11)
FONT_SMALL = ("微软雅黑", 10)

# 所有可能支持的模态
ALL_MODALITIES = ["EEG", "EMG", "ECG", "fNIRS"]

# 定义完整的参数映射（已修正降采样参数为 target_sampling_rate，除 EMG 外）
EEG_PARAMS = {
    "use_highpass": {"type": "bool", "default": True, "label": "启用高通滤波"},
    "highpass_freq": {"type": "float", "default": 0.5, "label": "高通频率 (Hz)"},
    "use_lowpass": {"type": "bool", "default": True, "label": "启用低通滤波"},
    "lowpass_freq": {"type": "float", "default": 45.0, "label": "低通频率 (Hz)"},
    "filter_order": {"type": "int", "default": 4, "label": "滤波器阶数"},
    "filter_type": {"type": "choice", "options": [e.value for e in FilterType], "default": "butterworth", "label": "滤波器类型"},
    "line_freq": {"type": "float", "default": 50.0, "label": "工频频率 (Hz)"},
    "use_ica": {"type": "bool", "default": True, "label": "使用 ICA 去除伪迹"},
    "ica_method": {"type": "choice", "options": [e.value for e in ICAMethod], "default": "infomax", "label": "ICA 方法"},
    "ica_n_components": {"type": "float", "default": 0.95, "label": "ICA 成分数 (0~1 比例或整数)"},
    "reference_type": {"type": "choice", "options": [e.value for e in ReferenceType], "default": "average", "label": "重参考类型"},
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
    "filter_type": {"type": "choice", "options": [e.value for e in FilterType], "default": "butterworth", "label": "滤波器类型"},
    "line_frequency": {"type": "float", "default": 50.0, "label": "工频频率 (Hz)"},
    "use_harmonic_notch": {"type": "bool", "default": True, "label": "谐波陷波"},
    "notch_harmonics": {"type": "int", "default": 5, "label": "谐波数量"},
    "rectification_method": {"type": "choice", "options": [e.value for e in RectificationMethod], "default": "full_wave", "label": "整流方法"},
    "envelope_method": {"type": "choice", "options": [e.value for e in EnvelopeExtractionMethod], "default": "lowpass", "label": "包络提取方法"},
    "envelope_cutoff": {"type": "float", "default": 5.0, "label": "包络截止频率 (Hz)"},
    "envelope_order": {"type": "int", "default": 4, "label": "包络滤波器阶数"},
    "remove_motion_artifacts": {"type": "bool", "default": True, "label": "去除运动伪迹"},
    "motion_artifact_threshold": {"type": "float", "default": 5.0, "label": "运动伪迹阈值 (std倍数)"},
    "detect_muscle_activation": {"type": "bool", "default": False, "label": "检测肌肉激活"},
    "activation_threshold": {"type": "float", "default": 2.0, "label": "激活阈值 (std倍数)"},
    "downsample_to": {"type": "float", "default": 1000.0, "label": "降采样至 (Hz, 0=不降)"},
    "normalize_method": {"type": "choice", "options": ["zscore", "minmax", "robust", "none"], "default": "zscore", "label": "标准化方法"},
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
    "optical_model": {"type": "choice", "options": [e.value for e in OpticalModel], "default": "modified_beer_lambert", "label": "光学模型"},
    "motion_correction_method": {"type": "choice", "options": [e.value for e in MotionCorrectionMethod], "default": "spline", "label": "运动校正方法"},
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
    "processing_mode": {"type": "choice", "options": ["sequential", "parallel"], "default": "sequential", "label": "处理模式"},
    "time_sync_method": {"type": "choice", "options": ["none", "resample", "interpolate"], "default": "none", "label": "时间同步方法"},
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
        tk.Button(btn_frame, text="确认", command=self.on_ok, bg=BUTTON_BG, fg=BUTTON_FG).pack(side="left", padx=5)
        tk.Button(btn_frame, text="取消", command=self.destroy).pack(side="left", padx=5)

    def on_ok(self):
        for orig, combo in self.widgets.items():
            self.result[orig] = combo.get()
        self.destroy()


class PreprocessingApp(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent

        self.raw_data_dict = None
        self.processed_data_dict = None
        self.current_filepath = None

        self.modality_vars = {}
        self.eeg_widgets = {}
        self.emg_widgets = {}
        self.ecg_widgets = {}
        self.fnirs_widgets = {}
        self.global_widgets = {}

        if not MODULES_LOADED:
            messagebox.showerror("初始化失败", "底层算法模块导入失败，请检查终端输出的路径信息。")

        self.setup_ui()

    def setup_ui(self):
        self.configure(padding="10")

        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill="both", expand=True)

        # ================= 左侧面板 =================
        left_frame = ttk.Frame(main_paned, padding="10")
        main_paned.add(left_frame, weight=1)

        title_lbl = ttk.Label(left_frame, text="信号预处理", font=FONT_TITLE)
        title_lbl.pack(anchor="w", pady=(0, 20))

        # 1. 数据加载
        load_frame = ttk.LabelFrame(left_frame, text="1. 加载数据源", padding="10")
        load_frame.pack(fill="x", pady=5)

        self.btn_load = ttk.Button(load_frame, text="📁 上传文件", command=self.action_load_data)
        self.btn_load.pack(side="left", padx=5)

        self.lbl_status = ttk.Label(load_frame, text="请上传文件...", font=FONT_SMALL)
        self.lbl_status.pack(side="left", padx=10)

        # 2. 模态选择 - 使用 tk.Checkbutton 显示勾
        mod_frame = ttk.LabelFrame(left_frame, text="2. 选择要处理的模态", padding="10")
        mod_frame.pack(fill="x", pady=5)

        self.mod_check_frame = tk.Frame(mod_frame, bg=BG_COLOR)
        self.mod_check_frame.pack(anchor="w", padx=5, pady=5)

        for mod in ALL_MODALITIES:
            var = tk.BooleanVar(value=False)
            self.modality_vars[mod] = var
            cb = tk.Checkbutton(
                self.mod_check_frame,
                text=mod,
                variable=var,
                command=lambda m=mod: self.on_modality_toggle(m),  # 传入当前模态，用于自动切换
                bg=BG_COLOR,
                fg=FG_COLOR,
                selectcolor=BG_COLOR,
                activebackground=BG_COLOR,
                font=FONT_NORMAL
            )
            cb.pack(side="left", padx=10)

        # 3. 全局设置
        global_frame = ttk.LabelFrame(left_frame, text="3. 全局设置", padding="10")
        global_frame.pack(fill="x", pady=5)

        self._create_param_widgets(global_frame, GLOBAL_PARAMS, self.global_widgets)

        # 4. 运行预处理
        run_frame = ttk.Frame(left_frame)
        run_frame.pack(fill="x", pady=10)

        self.btn_run = ttk.Button(run_frame, text="⚡ 开始预处理", command=self.action_run_preprocessing, state="disabled")
        self.btn_run.pack(side="left", padx=5)

        self.btn_save = ttk.Button(run_frame, text="💾 保存结果", command=self.action_save_results, state="disabled")
        self.btn_save.pack(side="left", padx=5)

        # ================= 右侧面板 =================
        right_frame = ttk.Frame(main_paned, padding="10")
        main_paned.add(right_frame, weight=2)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill="both", expand=True)

        self.eeg_tab = self._create_modality_tab("EEG", EEG_PARAMS)
        self.emg_tab = self._create_modality_tab("EMG", EMG_PARAMS)
        self.ecg_tab = self._create_modality_tab("ECG", ECG_PARAMS)
        self.fnirs_tab = self._create_modality_tab("fNIRS", FNIRS_PARAMS)

        self.notebook.add(self.eeg_tab, text="EEG", state="disabled")
        self.notebook.add(self.emg_tab, text="EMG", state="disabled")
        self.notebook.add(self.ecg_tab, text="ECG", state="disabled")
        self.notebook.add(self.fnirs_tab, text="fNIRS", state="disabled")

        # ================= 底部结果摘要 =================
        result_frame = ttk.LabelFrame(self, text="预处理摘要", padding="10")
        result_frame.pack(fill="both", expand=True, pady=5)

        columns = ("Property", "Value")
        self.tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=8)
        self.tree.heading("Property", text="属性")
        self.tree.heading("Value", text="值")
        self.tree.column("Property", width=200)
        self.tree.column("Value", width=400)

        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _create_param_widgets(self, parent, param_dict, widget_dict):
        """动态生成参数控件，布尔类型使用 tk.Checkbutton 显示勾"""
        row = 0
        for key, spec in param_dict.items():
            label = ttk.Label(parent, text=spec["label"] + ":", font=FONT_NORMAL)
            label.grid(row=row, column=0, sticky="w", padx=5, pady=3)

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
                cb.grid(row=row, column=1, sticky="w", padx=5)
                widget_dict[key] = var
            elif spec["type"] == "choice":
                var = tk.StringVar(value=spec["default"])
                combo = ttk.Combobox(parent, textvariable=var, values=spec["options"], state="readonly", width=20)
                combo.grid(row=row, column=1, sticky="w", padx=5)
                widget_dict[key] = var
            elif spec["type"] in ("float", "int"):
                var = tk.StringVar(value=str(spec["default"]))
                entry = ttk.Entry(parent, textvariable=var, width=20)
                entry.grid(row=row, column=1, sticky="w", padx=5)
                widget_dict[key] = var
            elif spec["type"] == "str":
                var = tk.StringVar(value=spec["default"])
                entry = ttk.Entry(parent, textvariable=var, width=20)
                entry.grid(row=row, column=1, sticky="w", padx=5)
                widget_dict[key] = var
            row += 1

    def _create_modality_tab(self, mod_name, param_dict):
        """创建模态参数标签页（带滚动条）"""
        tab = ttk.Frame(self.notebook, padding="15")
        canvas = tk.Canvas(tab, borderwidth=0, highlightthickness=0, bg=BG_COLOR)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        if mod_name == "EEG":
            self._create_param_widgets(scrollable_frame, param_dict, self.eeg_widgets)
        elif mod_name == "EMG":
            self._create_param_widgets(scrollable_frame, param_dict, self.emg_widgets)
        elif mod_name == "ECG":
            self._create_param_widgets(scrollable_frame, param_dict, self.ecg_widgets)
        elif mod_name == "fNIRS":
            self._create_param_widgets(scrollable_frame, param_dict, self.fnirs_widgets)

        return tab

    def on_modality_toggle(self, mod=None):
        """
        当模态复选框状态改变时调用
        mod: 如果是由用户点击触发的，传入被点击的模态，用于自动切换到该标签页
        """
        # 先更新标签页的启用/禁用状态
        for idx, m in enumerate(ALL_MODALITIES):
            if self.modality_vars[m].get():
                self.notebook.tab(idx, state="normal")
            else:
                self.notebook.tab(idx, state="disabled")

        # 如果是由用户点击触发的，并且该模态被勾选，则自动切换到该标签页
        if mod is not None and self.modality_vars[mod].get():
            idx = ALL_MODALITIES.index(mod)
            self.notebook.select(idx)

    def action_load_data(self):
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
        self.btn_load.config(state="disabled")
        self.btn_run.config(state="disabled")
        self.lbl_status.config(text="正在加载数据...", foreground="#0056b3")

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
        """更新复选框状态，并自动切换到第一个勾选的模态标签页"""
        for mod, var in self.modality_vars.items():
            var.set(mod in detected_mods)

        # 更新标签页状态（不传入 mod，避免切换）
        self.on_modality_toggle()

        # 自动切换到第一个被勾选的模态（如果有）
        for mod in ALL_MODALITIES:
            if mod in detected_mods:
                idx = ALL_MODALITIES.index(mod)
                self.notebook.select(idx)
                break

        self.raw_data_dict = raw_dict
        self.lbl_status.config(text=f"数据加载成功，检测到模态: {', '.join(detected_mods)}", foreground="green")
        self.btn_load.config(state="normal")
        self.btn_run.config(state="normal")

    def _ui_update_on_error(self, error_msg):
        self.lbl_status.config(text=error_msg, foreground="red")
        self.btn_load.config(state="normal")
        self.btn_run.config(state="disabled")

    def action_run_preprocessing(self):
        selected_mods = [mod for mod, var in self.modality_vars.items() if var.get()]
        if not selected_mods:
            messagebox.showwarning("警告", "请至少选择一个模态进行预处理")
            return

        if self.raw_data_dict is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        self.btn_run.config(state="disabled")
        self.btn_load.config(state="disabled")
        self.btn_save.config(state="disabled")
        self.lbl_status.config(text="正在执行预处理，请稍候...", foreground="#0056b3")

        for item in self.tree.get_children():
            self.tree.delete(item)

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

        eeg_config = None
        emg_config = None
        ecg_config = None
        fnirs_config = None

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
                target_sampling_rate=float(self.eeg_widgets["target_sampling_rate"].get()) if float(self.eeg_widgets["target_sampling_rate"].get()) > 0 else None,
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
                downsample_to=float(self.emg_widgets["downsample_to"].get()) if float(self.emg_widgets["downsample_to"].get()) > 0 else None,
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
                target_sampling_rate=float(self.ecg_widgets["target_sampling_rate"].get()) if float(self.ecg_widgets["target_sampling_rate"].get()) > 0 else None,
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
                target_sampling_rate=float(self.fnirs_widgets["target_sampling_rate"].get()) if float(self.fnirs_widgets["target_sampling_rate"].get()) > 0 else None,
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
        self.lbl_status.config(text=f"预处理成功完成，耗时 {result.processing_time:.2f} 秒", foreground="green")
        self.btn_load.config(state="normal")
        self.btn_run.config(state="normal")
        self.btn_save.config(state="normal")

        # 从 processed_data 中获取实际处理的模态（修复空白问题）
        if "processed" in result.processed_data and "multimodal_preprocessing" in result.processed_data["processed"]:
            mods_processed = result.processed_data["processed"]["multimodal_preprocessing"].get("modalities_processed", [])
        else:
            mods_processed = []

        self.tree.insert("", "end", values=("处理时间", f"{result.processing_time:.2f} 秒"))
        self.tree.insert("", "end", values=("成功状态", str(result.success)))
        self.tree.insert("", "end", values=("处理模态", ", ".join(mods_processed) if mods_processed else "无"))

        quality_report = result.processed_data.get("processed", {}).get("quality_report", {})
        if quality_report:
            self.tree.insert("", "end", values=("总体质量", f"{quality_report.get('overall_quality', 0):.2f}"))
            mod_quality = quality_report.get("modality_quality", {})
            for mod, mq in mod_quality.items():
                self.tree.insert("", "end", values=(f"{mod} 质量分数", f"{mq.get('quality_score', 0):.2f}"))

    def action_save_results(self):
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
    root.geometry("1400x900")
    root.configure(bg=BG_COLOR)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR, font=FONT_NORMAL)
    style.configure("TFrame", background=BG_COLOR)
    style.configure("TButton", font=FONT_NORMAL, padding=6)
    style.configure("TNotebook", background=BG_COLOR)
    style.configure("TNotebook.Tab", font=FONT_NORMAL, padding=[10, 5])
    style.configure("TCheckbutton", background=BG_COLOR, font=FONT_NORMAL)
    style.configure("TCombobox", font=FONT_NORMAL)
    style.configure("TEntry", font=FONT_NORMAL)

    app = PreprocessingApp(root)
    app.pack(fill=tk.BOTH, expand=True)

    root.mainloop()