#!/usr/bin/env python3
"""
signal_view.py
Tkinter版本 - 通用信号视图
支持所有模态的多通道信号可视化
基于四层数据格式: meta/signal/event/processed
"""
import matplotlib
import matplotlib.pyplot as plt

# ========== 解决中文乱码问题 ==========
# 设置中文字体
try:
    # Windows系统
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
except:
    try:
        # Linux系统
        matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC']
    except:
        try:
            # macOS系统
            matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
        except:
            # 如果都没有，使用默认字体但警告
            print("警告: 未找到合适的中文字体，中文可能显示为方框")

# 解决负号显示问题
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 设置matplotlib后端 ==========
matplotlib.use('TkAgg')

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import scipy.signal
from typing import Dict, List, Optional, Tuple, Any
import os
from datetime import datetime
import copy
import importlib.util
import traceback

# ========== 检查并导入所有预处理模块 ==========
HAS_PREPROCESSING = False
HAS_MULTIMODAL = False
HAS_EEG = False
HAS_FNIRS = False
HAS_EMG = False
HAS_ECG = False

try:
    # 通用预处理
    from core.processing.preprocessing.preprocessing import GeneralPreprocessor, PreprocessingConfig, FilterType, \
        WaveletType, DetrendMethod

    HAS_PREPROCESSING = True

    # 多模态预处理
    try:
        from core.processing.preprocessing.multimodal_preprocessing import MultiModalPreprocessor, MultiModalConfig, \
            ProcessingMode, TimeSyncMethod, MultiModalConfigFactory

        HAS_MULTIMODAL = True
    except ImportError:
        pass

    # EEG专用
    try:
        from core.processing.preprocessing.eeg_preprocessing import EEGPreprocessor, EEGPreprocessingConfig, \
            ReferenceType, ICAMethod, ArtifactRemovalMethod, EEGConfigFactory

        HAS_EEG = True
    except ImportError:
        pass

    # fNIRS专用
    try:
        from core.processing.preprocessing.fnirs_preprocessing import fNIRSPreprocessor, fNIRSConfig, OpticalModel, \
            MotionCorrectionMethod

        HAS_FNIRS = True
    except ImportError:
        pass

    # EMG专用
    try:
        from core.processing.preprocessing.emg_preprocessing import EMGPreprocessor, EMGPreprocessingConfig, \
            RectificationMethod, EnvelopeExtractionMethod, MuscleActivationDetectionMethod, EMGConfigFactory

        HAS_EMG = True
    except ImportError:
        pass

    # ECG专用
    try:
        from core.processing.preprocessing.ecg_preprocessing import ECGPreprocessor, ECGConfig, ECGQualityFlag

        HAS_ECG = True
    except ImportError:
        pass

except ImportError as e:
    print(f"警告: 无法导入基础预处理模块: {e}")
    print("将使用简单实时滤波")


class SignalView(tk.Frame):
    """
    通用信号视图类 - Tkinter版本
    支持EEG/EMG/ECG/GSR/fNIRS/ET/RESP等多种模态
    集成了专业预处理模块
    """

    def __init__(self, parent, data_dict: Dict[str, Any], modality: str = None):
        """
        初始化信号视图

        Args:
            parent: 父窗口
            data_dict: 标准四层数据字典
            modality: 要显示的模态（None表示自动选择第一个）
        """
        super().__init__(parent)
        self.parent = parent
        self.original_data_dict = copy.deepcopy(data_dict)  # 备份原始数据
        self.data_dict = data_dict
        self.modality = modality
        self.available_modalities = []  # 可用模态列表

        # 解析数据（添加错误处理）
        try:
            self._parse_data()
        except Exception as e:
            messagebox.showerror("数据解析错误", f"无法解析数据: {str(e)}")
            # 创建模拟数据作为后备
            self._create_fallback_data()
            print(f"使用模拟数据作为后备: {e}")

        # 当前显示状态
        self.current_page = 0
        self.page_duration = 5.0
        self.zoom_level = 1.0
        self.markers = []
        self.selected_channels = []

        # ========== 预处理相关变量 ==========
        self.preview_preprocessing = False  # 是否预览预处理
        self.processed_data_dict = None  # 预处理后的数据
        self.current_preprocessor = None  # 当前使用的预处理器
        self.current_config = None  # 当前预处理配置
        self.preprocessing_history = []  # 预处理历史

        # 预处理UI变量
        self.preview_preprocess_var = tk.BooleanVar(value=False)
        self.preprocess_mode_var = tk.StringVar(value="auto")
        self.pp_lowcut_var = tk.StringVar(value="0.5")
        self.pp_highcut_var = tk.StringVar(value="45")
        self.pp_notch_var = tk.StringVar(value="50")
        self.pp_order_var = tk.StringVar(value="4")
        self.pp_wavelet_var = tk.BooleanVar(value=False)
        self.pp_median_var = tk.BooleanVar(value=False)
        self.pp_outlier_var = tk.BooleanVar(value=False)
        self.pp_sync_var = tk.BooleanVar(value=False)

        # 创建matplotlib图形
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = None

        # 设置UI
        self.setup_ui()

        # 初始化绘图
        self.update_plot()

    def _create_fallback_data(self):
        """创建后备数据（当数据解析失败时）"""
        print("创建后备数据...")
        self.modality = "EEG"
        self.subject_id = "fallback"
        self.session_id = "session1"
        self.task = "rest"

        # 创建模拟EEG数据
        fs = 1000
        t = np.arange(0, 10, 1 / fs)
        self.data = np.array([np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t)) for _ in range(8)])
        self.n_channels, self.n_samples = self.data.shape
        self.sampling_rate = fs
        self.channel_names = [f"Ch{i + 1}" for i in range(self.n_channels)]
        self.unit = "uV"
        self.signal_type = "eeg"
        self.duration = self.n_samples / self.sampling_rate
        self.event_times = []
        self.event_labels = []
        self.event_ids = []

    def _convert_to_numpy(self, data):
        """将各种格式的数据转换为numpy数组"""
        if data is None:
            raise ValueError("数据为空")

        # 如果是numpy数组，直接返回
        if isinstance(data, np.ndarray):
            return data

        # 如果是列表，转换为numpy数组
        if isinstance(data, list):
            try:
                return np.array(data, dtype=np.float32)
            except Exception as e:
                raise ValueError(f"列表转numpy失败: {e}")

        # 如果是其他类型，尝试转换
        try:
            return np.array(data, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"无法转换为numpy数组: {e}")

    def _ensure_2d(self, data):
        """确保数据是2D (channels × samples)"""
        if data.ndim == 1:
            # 1D数据：假设是单个通道
            return data.reshape(1, -1)
        elif data.ndim == 2:
            # 2D数据：检查是否需要转置
            if data.shape[0] > data.shape[1]:
                # 如果通道数大于样本数，可能是 (samples, channels) 格式
                print(f"检测到数据可能为 (samples, channels) 格式，形状: {data.shape}，进行转置")
                return data.T
            return data
        elif data.ndim == 3:
            # 3D数据：可能是 (channels, frequencies, samples) 或其他格式
            # 这里简单地取第一个频率或平均
            print(f"检测到3D数据，形状: {data.shape}，取第一个维度")
            return data[0, :, :] if data.shape[0] < data.shape[2] else data[:, 0, :]
        else:
            # 更高维度：展平
            print(f"检测到{data.ndim}D数据，尝试展平")
            return data.reshape(data.shape[0], -1)

    def _parse_data(self):
        """解析数据字典（增强版，支持多种数据格式）"""
        print("\n" + "=" * 60)
        print("📊 解析数据字典")
        print("=" * 60)

        # 获取元数据
        self.meta = self.data_dict.get("meta", {})
        self.subject_id = self.meta.get("subject_id", "unknown")
        self.session_id = self.meta.get("session_id", "unknown")
        self.task = self.meta.get("task", "unknown")

        # 获取信号数据
        signal_dict = self.data_dict.get("signal", {})
        if not signal_dict:
            raise ValueError("数据字典中没有'signal'字段")

        # ========== 获取所有可用模态 ==========
        self.available_modalities = list(signal_dict.keys())
        print(f"可用模态: {self.available_modalities}")

        # 确定要显示的模态
        if self.modality is None:
            # 默认选择第一个模态
            self.modality = self.available_modalities[0]
            print(f"自动选择模态: {self.modality}")

        if self.modality not in signal_dict:
            raise ValueError(f"模态 {self.modality} 不在数据中")

        signal_info = signal_dict[self.modality]
        print(f"选择模态: {self.modality}")
        print(f"信号信息字段: {list(signal_info.keys())}")

        # ========== 信号数据处理（核心修改） ==========
        raw_data = signal_info.get("data")
        if raw_data is None:
            raise ValueError("信号数据不存在")

        print(f"原始数据类型: {type(raw_data)}")
        if isinstance(raw_data, list):
            print(f"列表长度: {len(raw_data)}")
            if len(raw_data) > 0:
                print(f"第一个元素类型: {type(raw_data[0])}")
                if isinstance(raw_data[0], list):
                    print(f"第一个元素长度: {len(raw_data[0])}")

        # 转换为numpy数组
        try:
            self.data = self._convert_to_numpy(raw_data)
            print(f"转换为numpy后形状: {self.data.shape}")
        except Exception as e:
            raise ValueError(f"数据转换失败: {e}")

        # 确保数据是2D
        self.data = self._ensure_2d(self.data)
        print(f"确保2D后形状: {self.data.shape}")

        # 获取基本信息
        self.n_channels, self.n_samples = self.data.shape
        self.sampling_rate = signal_info.get("sampling_rate", 1000)
        self.channel_names = signal_info.get("channel_names",
                                             [f"Ch{i + 1}" for i in range(self.n_channels)])
        self.unit = signal_info.get("unit", "unknown")
        self.signal_type = signal_info.get("signal_type", self.modality.lower())

        # 持续时间
        self.duration = self.n_samples / self.sampling_rate

        # 获取事件数据
        self.events = self.data_dict.get("event", {})
        self.event_times = self.events.get("event_time", [])
        self.event_labels = self.events.get("event_label", [])
        self.event_ids = self.events.get("event_id", [])

        print(f"解析完成: {self.n_channels}通道, {self.n_samples}样本, {self.sampling_rate}Hz, {self.duration:.2f}秒")
        print("=" * 60 + "\n")

    def setup_ui(self):
        """设置用户界面（完整版，包含预处理控制）"""
        # 主布局
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ========== 顶部控制栏 ==========
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        # 左侧信息
        info_text = f"{self.modality} | {self.n_channels}通道 | {self.sampling_rate}Hz | {self.duration:.2f}秒"
        self.info_label = ttk.Label(control_frame, text=info_text, font=('微软雅黑', 10))
        self.info_label.pack(side=tk.LEFT, padx=5)

        # ========== 添加模态选择器 ==========
        if len(self.available_modalities) > 1:
            ttk.Label(control_frame, text="模态:").pack(side=tk.LEFT, padx=(10, 2))
            self.modality_var = tk.StringVar(value=self.modality)
            modality_combo = ttk.Combobox(control_frame, textvariable=self.modality_var,
                                          values=self.available_modalities, state="readonly", width=10)
            modality_combo.pack(side=tk.LEFT, padx=2)
            modality_combo.bind('<<ComboboxSelected>>', self.on_modality_changed)

        # 右侧翻页控制
        page_frame = ttk.Frame(control_frame)
        page_frame.pack(side=tk.RIGHT)

        ttk.Label(page_frame, text="页:").pack(side=tk.LEFT)

        self.page_var = tk.StringVar(value="1")
        self.page_spin = ttk.Spinbox(page_frame, from_=1,
                                     to=max(1, int(np.ceil(self.duration / self.page_duration))),
                                     width=5, textvariable=self.page_var,
                                     command=self.on_page_changed)
        self.page_spin.pack(side=tk.LEFT, padx=2)

        ttk.Label(page_frame, text="/").pack(side=tk.LEFT)

        self.total_pages_label = ttk.Label(page_frame,
                                           text=str(max(1, int(np.ceil(self.duration / self.page_duration)))))
        self.total_pages_label.pack(side=tk.LEFT, padx=2)

        ttk.Label(page_frame, text="每页(秒):").pack(side=tk.LEFT, padx=(10, 2))

        self.page_duration_var = tk.StringVar(value="5")
        self.page_duration_spin = ttk.Spinbox(page_frame, from_=1, to=60, width=5,
                                              textvariable=self.page_duration_var,
                                              command=self.on_page_duration_changed)
        self.page_duration_spin.pack(side=tk.LEFT, padx=2)

        ttk.Button(page_frame, text="◀", width=3,
                   command=self.prev_page).pack(side=tk.LEFT, padx=2)
        ttk.Button(page_frame, text="▶", width=3,
                   command=self.next_page).pack(side=tk.LEFT, padx=2)

        # ========== 第二行：滤波控制 ==========
        filter_frame = ttk.LabelFrame(main_frame, text="实时滤波")
        filter_frame.pack(fill=tk.X, pady=5)

        self.filter_var = tk.BooleanVar(value=False)
        tk.Checkbutton(filter_frame, text="启用实时滤波",
                       variable=self.filter_var,
                       command=self.on_filter_toggled,
                       bg='white').pack(side=tk.LEFT, padx=5)

        ttk.Label(filter_frame, text="低通(Hz):").pack(side=tk.LEFT, padx=(10, 2))
        self.lowpass_var = tk.StringVar(value="45")
        self.lowpass_entry = ttk.Entry(filter_frame, textvariable=self.lowpass_var, width=8)
        self.lowpass_entry.pack(side=tk.LEFT, padx=2)
        self.lowpass_entry.bind('<Return>', lambda e: self.update_plot())

        ttk.Label(filter_frame, text="高通(Hz):").pack(side=tk.LEFT, padx=(10, 2))
        self.highpass_var = tk.StringVar(value="0.5")
        self.highpass_entry = ttk.Entry(filter_frame, textvariable=self.highpass_var, width=8)
        self.highpass_entry.pack(side=tk.LEFT, padx=2)
        self.highpass_entry.bind('<Return>', lambda e: self.update_plot())

        ttk.Label(filter_frame, text="陷波(Hz):").pack(side=tk.LEFT, padx=(10, 2))
        self.notch_var = tk.StringVar(value="50")
        self.notch_entry = ttk.Entry(filter_frame, textvariable=self.notch_var, width=8)
        self.notch_entry.pack(side=tk.LEFT, padx=2)
        self.notch_entry.bind('<Return>', lambda e: self.update_plot())

        self.lowpass_entry.config(state='disabled')
        self.highpass_entry.config(state='disabled')
        self.notch_entry.config(state='disabled')

        # ========== 预处理控制面板 ==========
        preprocess_frame = ttk.LabelFrame(main_frame, text="专业预处理预览")
        preprocess_frame.pack(fill=tk.X, pady=5)

        # 检查预处理模块可用性
        preprocess_available = HAS_PREPROCESSING
        if not preprocess_available:
            disabled_state = tk.DISABLED
            status_text = " (预处理模块未加载)"
        else:
            disabled_state = tk.NORMAL
            status_text = ""

        # 第一行：启用预处理和模态选择
        row1 = ttk.Frame(preprocess_frame)
        row1.pack(fill=tk.X, pady=2)

        self.preview_check = tk.Checkbutton(
            row1,
            text=f"启用预处理预览{status_text}",
            variable=self.preview_preprocess_var,
            command=self.on_preprocess_toggled,
            bg='white',
            state=disabled_state
        )
        self.preview_check.pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="预处理类型:").pack(side=tk.LEFT, padx=(10, 2))

        # 根据可用模块动态生成模式选项
        mode_options = ["auto (自动选择)"]
        if HAS_EEG:
            mode_options.append("EEG专用")
        if HAS_FNIRS:
            mode_options.append("fNIRS专用")
        if HAS_EMG:
            mode_options.append("EMG专用")
        if HAS_ECG:
            mode_options.append("ECG专用")
        mode_options.append("通用预处理")

        self.preprocess_mode_combo = ttk.Combobox(
            row1,
            textvariable=self.preprocess_mode_var,
            values=mode_options,
            width=20,
            state="readonly" if preprocess_available else tk.DISABLED
        )
        self.preprocess_mode_combo.pack(side=tk.LEFT, padx=2)
        self.preprocess_mode_combo.bind('<<ComboboxSelected>>', self.on_preprocess_mode_changed)

        # 应用按钮
        self.apply_btn = ttk.Button(
            row1,
            text="应用预处理",
            command=self.apply_preprocessing,
            state=disabled_state
        )
        self.apply_btn.pack(side=tk.RIGHT, padx=5)

        # 重置按钮
        self.reset_btn = ttk.Button(
            row1,
            text="重置原始数据",
            command=self.reset_to_original,
            state=disabled_state
        )
        self.reset_btn.pack(side=tk.RIGHT, padx=5)

        # 第二行：快速配置按钮
        row2 = ttk.Frame(preprocess_frame)
        row2.pack(fill=tk.X, pady=2)

        ttk.Label(row2, text="快速配置:").pack(side=tk.LEFT, padx=5)

        config_buttons = []
        if HAS_EEG:
            config_buttons.extend([
                ("EEG-运动想象", "eeg_motor"),
                ("EEG-静息态", "eeg_rest"),
                ("EEG-P300", "eeg_p300")
            ])
        if HAS_EMG:
            config_buttons.extend([
                ("EMG-表面", "emg_surface"),
                ("EMG-高密度", "emg_hd")
            ])
        if HAS_FNIRS:
            config_buttons.append(("fNIRS-基础", "fnirs_basic"))
        if HAS_ECG:
            config_buttons.append(("ECG-基础", "ecg_basic"))
        config_buttons.append(("通用-基础", "general_basic"))

        for text, cfg in config_buttons[:6]:  # 最多显示6个
            btn = ttk.Button(
                row2,
                text=text,
                command=lambda c=cfg: self.load_preprocess_config(c),
                state=disabled_state
            )
            btn.pack(side=tk.LEFT, padx=2)

        # 第三行：高级参数折叠面板
        self.adv_btn = ttk.Button(
            preprocess_frame,
            text="▼ 高级参数设置",
            command=self.toggle_advanced,
            state=disabled_state
        )
        self.adv_btn.pack(pady=2)

        # 高级参数框架
        self.adv_frame = ttk.Frame(preprocess_frame)

        # 滤波参数
        filter_adv = ttk.LabelFrame(self.adv_frame, text="滤波参数")
        filter_adv.pack(fill=tk.X, pady=2)

        ttk.Label(filter_adv, text="低截止(Hz):").grid(row=0, column=0, padx=2, pady=2, sticky='w')
        self.pp_lowcut_entry = ttk.Entry(filter_adv, textvariable=self.pp_lowcut_var, width=8)
        self.pp_lowcut_entry.grid(row=0, column=1, padx=2)
        self.pp_lowcut_entry.bind('<Return>', lambda e: self.apply_preprocessing())

        ttk.Label(filter_adv, text="高截止(Hz):").grid(row=0, column=2, padx=2, sticky='w')
        self.pp_highcut_entry = ttk.Entry(filter_adv, textvariable=self.pp_highcut_var, width=8)
        self.pp_highcut_entry.grid(row=0, column=3, padx=2)
        self.pp_highcut_entry.bind('<Return>', lambda e: self.apply_preprocessing())

        ttk.Label(filter_adv, text="陷波(Hz):").grid(row=1, column=0, padx=2, pady=2, sticky='w')
        self.pp_notch_entry = ttk.Entry(filter_adv, textvariable=self.pp_notch_var, width=8)
        self.pp_notch_entry.grid(row=1, column=1, padx=2)
        self.pp_notch_entry.bind('<Return>', lambda e: self.apply_preprocessing())

        ttk.Label(filter_adv, text="阶数:").grid(row=1, column=2, padx=2, sticky='w')
        self.pp_order_entry = ttk.Entry(filter_adv, textvariable=self.pp_order_var, width=8)
        self.pp_order_entry.grid(row=1, column=3, padx=2)
        self.pp_order_entry.bind('<Return>', lambda e: self.apply_preprocessing())

        # 去噪参数
        denoise_adv = ttk.LabelFrame(self.adv_frame, text="去噪参数")
        denoise_adv.pack(fill=tk.X, pady=2)

        self.pp_wavelet_check = tk.Checkbutton(
            denoise_adv, text="小波去噪",
            variable=self.pp_wavelet_var,
            bg='white'
        )
        self.pp_wavelet_check.grid(row=0, column=0, padx=5, sticky='w')

        self.pp_median_check = tk.Checkbutton(
            denoise_adv, text="中值滤波",
            variable=self.pp_median_var,
            bg='white'
        )
        self.pp_median_check.grid(row=0, column=1, padx=5, sticky='w')

        self.pp_outlier_check = tk.Checkbutton(
            denoise_adv, text="去除离群值",
            variable=self.pp_outlier_var,
            bg='white'
        )
        self.pp_outlier_check.grid(row=0, column=2, padx=5, sticky='w')

        self.pp_sync_check = tk.Checkbutton(
            denoise_adv, text="时间同步",
            variable=self.pp_sync_var,
            bg='white'
        )
        self.pp_sync_check.grid(row=1, column=0, padx=5, sticky='w')

        # 状态显示
        self.preprocess_status = ttk.Label(preprocess_frame, text="", foreground="blue")
        self.preprocess_status.pack(pady=2)

        # 初始隐藏高级参数
        self.adv_visible = False

        # ========== 幅度控制 ==========
        amp_frame = ttk.LabelFrame(main_frame, text="幅度范围")
        amp_frame.pack(fill=tk.X, pady=5)

        ttk.Label(amp_frame, text="最小:").pack(side=tk.LEFT, padx=5)
        self.amp_min_var = tk.StringVar(value="-200")
        self.amp_min_entry = ttk.Entry(amp_frame, textvariable=self.amp_min_var, width=8)
        self.amp_min_entry.pack(side=tk.LEFT, padx=2)
        self.amp_min_entry.bind('<Return>', lambda e: self.update_plot())

        ttk.Label(amp_frame, text="最大:").pack(side=tk.LEFT, padx=(10, 2))
        self.amp_max_var = tk.StringVar(value="200")
        self.amp_max_entry = ttk.Entry(amp_frame, textvariable=self.amp_max_var, width=8)
        self.amp_max_entry.pack(side=tk.LEFT, padx=2)
        self.amp_max_entry.bind('<Return>', lambda e: self.update_plot())

        self.auto_amp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(amp_frame, text="自动",
                        variable=self.auto_amp_var,
                        command=self.on_auto_amp_toggled).pack(side=tk.LEFT, padx=10)

        self.amp_min_entry.config(state='disabled')
        self.amp_max_entry.config(state='disabled')

        # ========== 中间：绘图区域 + 通道列表 ==========
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 左侧：通道列表
        channel_frame = ttk.LabelFrame(middle_frame, text="通道选择", width=150)
        channel_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        channel_frame.pack_propagate(False)

        self.channel_listbox = tk.Listbox(channel_frame, selectmode=tk.MULTIPLE,
                                          exportselection=False)
        self.channel_listbox.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        for i, name in enumerate(self.channel_names):
            self.channel_listbox.insert(tk.END, f"{i + 1:02d}. {name}")
            self.channel_listbox.selection_set(i)
            self.selected_channels.append(i)

        self.channel_listbox.bind('<<ListboxSelect>>', self.on_channel_selected)

        ttk.Button(channel_frame, text="全选",
                   command=self.select_all_channels).pack(fill=tk.X, padx=2, pady=2)

        # 右侧：绘图区域
        plot_frame = ttk.Frame(middle_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # ========== 底部：事件标记区域 ==========
        event_frame = ttk.LabelFrame(main_frame, text="事件标记")
        event_frame.pack(fill=tk.X, pady=5)

        self.event_listbox = tk.Listbox(event_frame, height=3)
        self.event_listbox.pack(fill=tk.X, padx=2, pady=2)

        for t, label in zip(self.event_times, self.event_labels):
            self.event_listbox.insert(tk.END, f"{t:.3f}s: {label}")

        btn_frame = ttk.Frame(event_frame)
        btn_frame.pack(fill=tk.X, pady=2)

        ttk.Button(btn_frame, text="添加标记",
                   command=self.add_marker_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="清除标记",
                   command=self.clear_markers).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="保存标记",
                   command=self.save_markers).pack(side=tk.LEFT, padx=2)

    # ========== 模态切换方法 ==========
    def on_modality_changed(self, event=None):
        """模态切换响应"""
        new_modality = self.modality_var.get()
        if new_modality != self.modality:
            print(f"切换模态: {self.modality} -> {new_modality}")
            self.modality = new_modality

            # 重新解析当前模态的数据
            signal_dict = self.data_dict.get("signal", {})
            signal_info = signal_dict[self.modality]

            # 获取新模态的数据
            raw_data = signal_info.get("data")
            if raw_data is None:
                messagebox.showerror("错误", f"模态 {self.modality} 数据不存在")
                return

            # 转换数据
            try:
                self.data = self._convert_to_numpy(raw_data)
                self.data = self._ensure_2d(self.data)

                self.n_channels, self.n_samples = self.data.shape
                self.sampling_rate = signal_info.get("sampling_rate", 1000)
                self.channel_names = signal_info.get("channel_names",
                                                     [f"Ch{i+1}" for i in range(self.n_channels)])
                self.unit = signal_info.get("unit", "unknown")
                self.signal_type = signal_info.get("signal_type", self.modality.lower())
                self.duration = self.n_samples / self.sampling_rate

                # 更新通道列表
                self.channel_listbox.delete(0, tk.END)
                self.selected_channels = []
                for i, name in enumerate(self.channel_names):
                    self.channel_listbox.insert(tk.END, f"{i+1:02d}. {name}")
                    self.channel_listbox.selection_set(i)
                    self.selected_channels.append(i)

                # 更新翻页控件
                max_pages = max(1, int(np.ceil(self.duration / self.page_duration)))
                self.page_spin.config(to=max_pages)
                self.total_pages_label.config(text=str(max_pages))
                self.current_page = 0
                self.page_var.set("1")

                # 更新信息显示
                self.info_label.config(text=f"{self.modality} | {self.n_channels}通道 | {self.sampling_rate}Hz | {self.duration:.2f}秒")

                # 重新绘图
                self.update_plot()

                print(f"模态切换成功: {self.modality}, {self.n_channels}通道, {self.sampling_rate}Hz")

            except Exception as e:
                messagebox.showerror("错误", f"切换模态失败: {str(e)}")
                traceback.print_exc()

    # ========== 预处理相关方法 ==========

    def on_preprocess_toggled(self):
        """预处理开关响应"""
        if not HAS_PREPROCESSING:
            messagebox.showerror("错误", "预处理模块未加载，无法启用预览")
            self.preview_preprocess_var.set(False)
            return

        if self.preview_preprocess_var.get():
            # 启用预览，先应用预处理
            self.preprocess_status.config(text="正在应用预处理...")
            self.apply_preprocessing()
        else:
            # 禁用预览，恢复原始数据
            self.preprocess_status.config(text="")
            self.processed_data_dict = None
            self.update_plot()

    def on_preprocess_mode_changed(self, event=None):
        """预处理模式改变"""
        mode = self.preprocess_mode_var.get()
        self.preprocess_status.config(text=f"已选择: {mode}")

    def toggle_advanced(self):
        """切换高级参数面板"""
        if self.adv_visible:
            self.adv_frame.pack_forget()
            self.adv_btn.config(text="▼ 高级参数设置")
            self.adv_visible = False
        else:
            self.adv_frame.pack(fill=tk.X, pady=2, after=self.adv_btn)
            self.adv_btn.config(text="▲ 高级参数设置")
            self.adv_visible = True

    def load_preprocess_config(self, config_name: str):
        """加载预定义的预处理配置"""
        if not HAS_PREPROCESSING:
            messagebox.showerror("错误", "预处理模块未加载")
            return

        try:
            if config_name == "eeg_motor" and HAS_EEG:
                self.current_config = EEGConfigFactory.create_motor_imagery_config()
                self.current_preprocessor = EEGPreprocessor(self.current_config)
                self.preprocess_mode_var.set("EEG专用")
                self.preprocess_status.config(text="已加载: EEG运动想象配置")

            elif config_name == "eeg_rest" and HAS_EEG:
                self.current_config = EEGConfigFactory.create_resting_state_config()
                self.current_preprocessor = EEGPreprocessor(self.current_config)
                self.preprocess_mode_var.set("EEG专用")
                self.preprocess_status.config(text="已加载: EEG静息态配置")

            elif config_name == "eeg_p300" and HAS_EEG:
                self.current_config = EEGConfigFactory.create_p300_config()
                self.current_preprocessor = EEGPreprocessor(self.current_config)
                self.preprocess_mode_var.set("EEG专用")
                self.preprocess_status.config(text="已加载: EEG P300配置")

            elif config_name == "emg_surface" and HAS_EMG:
                self.current_config = EMGConfigFactory.create_surface_emg_config()
                self.current_preprocessor = EMGPreprocessor(self.current_config)
                self.preprocess_mode_var.set("EMG专用")
                self.preprocess_status.config(text="已加载: EMG表面配置")

            elif config_name == "emg_hd" and HAS_EMG:
                self.current_config = EMGConfigFactory.create_high_density_emg_config()
                self.current_preprocessor = EMGPreprocessor(self.current_config)
                self.preprocess_mode_var.set("EMG专用")
                self.preprocess_status.config(text="已加载: EMG高密度配置")

            elif config_name == "fnirs_basic" and HAS_FNIRS:
                self.current_config = fNIRSConfig(
                    lowcut=0.01,
                    highcut=0.5,
                    notch_freq=50.0,
                    optical_model=OpticalModel.MODIFIED_BEER_LAMBERT,
                    motion_correction_method=MotionCorrectionMethod.SPLINE
                )
                self.current_preprocessor = fNIRSPreprocessor(self.current_config)
                self.preprocess_mode_var.set("fNIRS专用")
                self.preprocess_status.config(text="已加载: fNIRS基础配置")

            elif config_name == "ecg_basic" and HAS_ECG:
                self.current_config = ECGConfig(
                    ecg_lowcut=0.5,
                    ecg_highcut=40.0,
                    ecg_notch_freq=50.0,
                    assess_signal_quality=True
                )
                self.current_preprocessor = ECGPreprocessor(self.current_config)
                self.preprocess_mode_var.set("ECG专用")
                self.preprocess_status.config(text="已加载: ECG基础配置")

            else:  # 通用基础配置
                self.current_config = PreprocessingConfig(
                    lowcut=float(self.pp_lowcut_var.get()),
                    highcut=float(self.pp_highcut_var.get()),
                    notch_freq=float(self.pp_notch_var.get()),
                    filter_order=int(self.pp_order_var.get()),
                    wavelet_level=4 if self.pp_wavelet_var.get() else 0,
                    use_median_filter=self.pp_median_var.get(),
                    remove_outliers=self.pp_outlier_var.get()
                )
                self.current_preprocessor = GeneralPreprocessor(self.current_config)
                self.preprocess_mode_var.set("通用预处理")
                self.preprocess_status.config(text="已加载: 通用基础配置")

            # 更新UI中的参数显示
            self._update_config_display()

        except Exception as e:
            messagebox.showerror("配置错误", f"加载配置失败: {str(e)}")
            traceback.print_exc()

    def _update_config_display(self):
        """更新配置显示"""
        if self.current_config:
            self.pp_lowcut_var.set(str(getattr(self.current_config, 'lowcut', '0.5')))
            self.pp_highcut_var.set(str(getattr(self.current_config, 'highcut', '45')))
            self.pp_notch_var.set(str(getattr(self.current_config, 'notch_freq', '50')))
            self.pp_order_var.set(str(getattr(self.current_config, 'filter_order', '4')))

    def apply_preprocessing(self):
        """应用预处理到数据"""
        if not HAS_PREPROCESSING:
            messagebox.showerror("错误", "预处理模块未加载")
            return

        if self.current_preprocessor is None:
            # 如果没有选择配置，使用默认配置
            self.current_config = PreprocessingConfig(
                lowcut=float(self.pp_lowcut_var.get()),
                highcut=float(self.pp_highcut_var.get()),
                notch_freq=float(self.pp_notch_var.get()),
                filter_order=int(self.pp_order_var.get()),
                wavelet_level=4 if self.pp_wavelet_var.get() else 0,
                use_median_filter=self.pp_median_var.get(),
                remove_outliers=self.pp_outlier_var.get()
            )
            self.current_preprocessor = GeneralPreprocessor(self.current_config)

        try:
            self.preprocess_status.config(text="正在处理中...")
            self.update_idletasks()

            # 根据预处理类型选择不同的处理方法
            if isinstance(self.current_preprocessor, EEGPreprocessor):
                # EEG专用处理
                self.processed_data_dict = self.current_preprocessor.process(
                    copy.deepcopy(self.original_data_dict),
                    modality=self.modality
                )
            elif isinstance(self.current_preprocessor, EMGPreprocessor):
                # EMG专用处理
                self.processed_data_dict = self.current_preprocessor.process(
                    copy.deepcopy(self.original_data_dict),
                    modality=self.modality
                )
            elif isinstance(self.current_preprocessor, ECGPreprocessor):
                # ECG专用处理
                self.processed_data_dict = self.current_preprocessor.process_ECG(
                    copy.deepcopy(self.original_data_dict),
                    modality=self.modality
                )
            elif isinstance(self.current_preprocessor, fNIRSPreprocessor):
                # fNIRS专用处理
                self.processed_data_dict = self.current_preprocessor.process_fNIRS(
                    copy.deepcopy(self.original_data_dict),
                    modality=self.modality
                )
            else:
                # 通用处理
                self.processed_data_dict = self.current_preprocessor.process(
                    copy.deepcopy(self.original_data_dict),
                    modality=self.modality
                )

            # 记录预处理历史
            self.preprocessing_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": self.preprocess_mode_var.get(),
                "config": str(self.current_config)
            })

            self.preprocess_status.config(text=f"预处理完成！")

            # 如果预览开启，立即更新显示
            if self.preview_preprocess_var.get():
                self.update_plot()

        except Exception as e:
            messagebox.showerror("预处理错误", f"预处理失败: {str(e)}")
            self.preprocess_status.config(text="预处理失败")
            traceback.print_exc()

    def reset_to_original(self):
        """重置到原始数据"""
        self.data_dict = copy.deepcopy(self.original_data_dict)
        self.processed_data_dict = None
        try:
            self._parse_data()  # 重新解析数据
        except Exception as e:
            messagebox.showerror("错误", f"重置数据失败: {str(e)}")
            self._create_fallback_data()
        self.preview_preprocess_var.set(False)
        self.preprocess_status.config(text="已重置到原始数据")
        self.update_plot()

    # ========== 数据获取方法 ==========

    def get_current_data(self):
        """获取当前要显示的数据（根据预览状态选择）"""
        if self.preview_preprocess_var.get() and self.processed_data_dict is not None:
            # 使用预处理后的数据
            try:
                return self.processed_data_dict["signal"][self.modality]["data"]
            except (KeyError, TypeError):
                return self.data
        else:
            # 使用原始数据
            return self.data

    def get_current_sampling_rate(self):
        """获取当前采样率"""
        if self.preview_preprocess_var.get() and self.processed_data_dict is not None:
            try:
                return self.processed_data_dict["signal"][self.modality]["sampling_rate"]
            except (KeyError, TypeError):
                return self.sampling_rate
        else:
            return self.sampling_rate

    def update_plot(self):
        """更新绘图（增强版，添加错误处理）"""
        try:
            self.figure.clear()

            # 获取当前要显示的数据
            current_data = self.get_current_data()
            current_sr = self.get_current_sampling_rate()

            # 获取选中的通道
            selected_channels = self.get_selected_channels()
            n_show = len(selected_channels)

            if n_show == 0:
                return

            # 计算当前页的时间范围
            t_start = self.current_page * self.page_duration
            t_end = min((self.current_page + 1) * self.page_duration, self.duration)

            start_idx = int(t_start * current_sr)
            end_idx = int(t_end * current_sr)

            # 确保索引有效
            start_idx = max(0, min(start_idx, current_data.shape[1] - 1))
            end_idx = max(start_idx + 1, min(end_idx, current_data.shape[1]))

            # 提取数据
            data_segment = current_data[:, start_idx:end_idx]
            time = np.arange(start_idx, end_idx) / current_sr

            # 应用实时滤波（如果启用）
            if self.filter_var.get():
                data_segment = self.apply_filter(data_segment, current_sr)

            # 创建子图
            gs = self.figure.add_gridspec(n_show, 1, hspace=0.1)

            # 计算幅度范围
            if self.auto_amp_var.get():
                y_min = np.min(data_segment[selected_channels])
                y_max = np.max(data_segment[selected_channels])
                margin = (y_max - y_min) * 0.1
                y_min -= margin
                y_max += margin
            else:
                try:
                    y_min = float(self.amp_min_var.get())
                    y_max = float(self.amp_max_var.get())
                except:
                    y_min = -200
                    y_max = 200

            # 绘制每个通道
            for i, ch_idx in enumerate(selected_channels):
                ax = self.figure.add_subplot(gs[i, 0])
                ax.plot(time, data_segment[ch_idx], 'b-', linewidth=0.8)

                # 设置Y轴
                ax.set_ylabel(f"{self.channel_names[ch_idx]}\n({self.unit})", fontsize=8)
                ax.set_ylim(y_min, y_max)

                if i < n_show - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('Time (s)', fontsize=9)

                ax.grid(True, alpha=0.3)

                # 绘制事件标记
                for t in self.event_times:
                    if t_start <= t <= t_end:
                        ax.axvline(x=t, color='r', linestyle='--', linewidth=1, alpha=0.7)

                # 绘制用户标记
                for t, color, label in self.markers:
                    if t_start <= t <= t_end:
                        ax.axvline(x=t, color=color, linestyle='-', linewidth=2)
                        ax.text(t, y_min + (y_max - y_min) * 0.1, label,
                                fontsize=8, color=color)

            # 添加标题，标明数据来源
            title_suffix = " (预处理预览)" if self.preview_preprocess_var.get() else ""
            self.figure.suptitle(f"{self.modality} - {self.subject_id} - {self.task}{title_suffix}\n"
                                 f"Time: {t_start:.2f} - {t_end:.2f} s",
                                 fontsize=12)

            self.canvas.draw()

        except Exception as e:
            print(f"绘图错误: {e}")
            traceback.print_exc()
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"绘图错误: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()

    def apply_filter(self, data: np.ndarray, fs: float) -> np.ndarray:
        """应用实时滤波"""
        if not self.filter_var.get():
            return data

        filtered = data.copy()

        try:
            lowcut = float(self.lowpass_var.get())
            if lowcut > 0 and lowcut < fs / 2:
                sos = scipy.signal.butter(4, lowcut, 'lowpass', fs=fs, output='sos')
                filtered = scipy.signal.sosfiltfilt(sos, filtered, axis=1)

            highcut = float(self.highpass_var.get())
            if highcut > 0:
                sos = scipy.signal.butter(4, highcut, 'highpass', fs=fs, output='sos')
                filtered = scipy.signal.sosfiltfilt(sos, filtered, axis=1)

            notch = float(self.notch_var.get())
            if notch > 0 and notch < fs / 2:
                Q = 30
                b, a = scipy.signal.iirnotch(notch, Q, fs)
                filtered = scipy.signal.filtfilt(b, a, filtered, axis=1)
        except Exception as e:
            print(f"滤波错误: {e}")

        return filtered

    # ========== 事件处理方法 ==========

    def on_filter_toggled(self):
        """滤波开关"""
        enabled = self.filter_var.get()
        self.lowpass_entry.config(state='normal' if enabled else 'disabled')
        self.highpass_entry.config(state='normal' if enabled else 'disabled')
        self.notch_entry.config(state='normal' if enabled else 'disabled')
        self.update_plot()

    def on_auto_amp_toggled(self):
        """自动幅度开关"""
        auto = self.auto_amp_var.get()
        self.amp_min_entry.config(state='disabled' if auto else 'normal')
        self.amp_max_entry.config(state='disabled' if auto else 'normal')
        self.update_plot()

    def on_page_changed(self):
        """页码改变"""
        try:
            self.current_page = int(self.page_var.get()) - 1
            self.update_plot()
        except:
            pass

    def on_page_duration_changed(self):
        """每页时长改变"""
        try:
            self.page_duration = float(self.page_duration_var.get())
            max_pages = max(1, int(np.ceil(self.duration / self.page_duration)))
            self.page_spin.config(to=max_pages)
            self.total_pages_label.config(text=str(max_pages))
            self.update_plot()
        except:
            pass

    def on_channel_selected(self, event):
        """通道选择改变"""
        self.selected_channels = self.channel_listbox.curselection()
        self.update_plot()

    def prev_page(self):
        """上一页"""
        if self.current_page > 0:
            self.current_page -= 1
            self.page_var.set(str(self.current_page + 1))
            self.update_plot()

    def next_page(self):
        """下一页"""
        max_pages = max(1, int(np.ceil(self.duration / self.page_duration)))
        if self.current_page < max_pages - 1:
            self.current_page += 1
            self.page_var.set(str(self.current_page + 1))
            self.update_plot()

    def select_all_channels(self):
        """全选通道"""
        self.channel_listbox.selection_set(0, tk.END)
        self.selected_channels = list(range(self.n_channels))
        self.update_plot()

    def get_selected_channels(self) -> List[int]:
        """获取选中的通道索引"""
        return list(self.selected_channels) if self.selected_channels else list(range(self.n_channels))

    def add_marker_dialog(self):
        """添加标记对话框"""
        current_time = (self.current_page * self.page_duration +
                        self.page_duration / 2)
        self.markers.append((current_time, 'green', f'Marker{len(self.markers) + 1}'))
        self.update_plot()
        messagebox.showinfo("添加标记", f"已添加标记 at {current_time:.2f}s")

    def clear_markers(self):
        """清除所有标记"""
        self.markers.clear()
        self.update_plot()

    def save_markers(self):
        """保存标记到事件字典"""
        if not self.markers:
            return

        if "event" not in self.data_dict:
            self.data_dict["event"] = {
                "event_id": [],
                "event_label": [],
                "event_time": [],
                "duration": []
            }

        for t, color, label in self.markers:
            self.data_dict["event"]["event_time"].append(t)
            self.data_dict["event"]["event_label"].append(label)
            self.data_dict["event"]["event_id"].append(len(self.data_dict["event"]["event_id"]) + 1)
            self.data_dict["event"]["duration"].append(0)

        messagebox.showinfo("保存成功", f"已保存{len(self.markers)}个标记到数据字典")

    def destroy(self):
        """销毁时清理"""
        plt.close(self.figure)
        super().destroy()


class fNIRSView(SignalView):
    """
    fNIRS专用视图类
    继承自SignalView，增加HbO/HbR分离显示功能
    """

    def __init__(self, parent, data_dict: Dict[str, Any], modality: str = "fNIRS"):
        # 标准化模态名称
        if 'signal' in data_dict:
            # 检查是否有全大写的 FNIRS
            if 'FNIRS' in data_dict['signal'] and modality == "fNIRS":
                print("检测到 FNIRS (全大写)，自动适配")
                modality = "FNIRS"
            # 检查是否有小写的 fnirs
            elif 'fnirs' in data_dict['signal'] and modality == "fNIRS":
                print("检测到 fnirs (全小写)，自动适配")
                modality = "fnirs"

        # 先设置fNIRS特有的属性
        self.n_hbo = 0
        self.n_hbr = 0
        self.hbo_var = tk.BooleanVar(value=True)
        self.hbr_var = tk.BooleanVar(value=True)

        # 调用父类初始化
        super().__init__(parent, data_dict, modality)

        # 添加fNIRS特有的控制
        self.setup_fnirs_controls()

    def setup_fnirs_controls(self):
        """设置fNIRS特有的控制"""
        for child in self.winfo_children():
            if isinstance(child, ttk.Frame):
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, ttk.LabelFrame) and grandchild['text'] == '实时滤波':
                        fnirs_frame = ttk.LabelFrame(child, text="fNIRS设置")
                        fnirs_frame.pack(fill=tk.X, pady=5, after=grandchild)

                        self.hbo_var = tk.BooleanVar(value=True)
                        self.hbr_var = tk.BooleanVar(value=True)

                        ttk.Checkbutton(fnirs_frame, text="显示HbO",
                                        variable=self.hbo_var,
                                        command=self.update_plot).pack(side=tk.LEFT, padx=5)
                        ttk.Checkbutton(fnirs_frame, text="显示HbR",
                                        variable=self.hbr_var,
                                        command=self.update_plot).pack(side=tk.LEFT, padx=5)

                        ttk.Label(fnirs_frame, text="通道配对:", font=('微软雅黑', 9)).pack(side=tk.LEFT, padx=(20, 2))
                        self.pair_info_var = tk.StringVar(value="自动")
                        ttk.Label(fnirs_frame, textvariable=self.pair_info_var).pack(side=tk.LEFT)
                        break
                break

    def _parse_data(self):
        """重写数据解析，处理fNIRS特有的数据格式"""
        super()._parse_data()

        if hasattr(self, 'data') and self.data is not None:
            n_channels = self.n_channels
            if n_channels % 2 == 0:
                self.n_hbo = n_channels // 2
                self.n_hbr = n_channels // 2
            else:
                self.n_hbo = (n_channels + 1) // 2
                self.n_hbr = n_channels // 2

    def get_selected_channels(self) -> List[int]:
        """重写通道选择，根据HbO/HbR设置筛选"""
        selected = super().get_selected_channels()

        if not hasattr(self, 'hbo_var'):
            return selected

        filtered = []
        for idx in selected:
            if idx < self.n_hbo and self.hbo_var.get():
                filtered.append(idx)
            elif idx >= self.n_hbo and self.hbr_var.get():
                filtered.append(idx)

        return filtered if filtered else selected

    def update_plot(self):
        """重写绘图方法，添加fNIRS特有的标记"""
        if hasattr(self, 'pair_info_var'):
            n_show = len(self.get_selected_channels())
            self.pair_info_var.set(f"显示{n_show}通道")

        super().update_plot()


# ========== 测试代码 ==========
if __name__ == "__main__":
    import sys

    root = tk.Tk()
    root.title("信号视图测试 (集成预处理模块)")
    root.geometry("1200x800")

    style = ttk.Style()
    style.configure(".", font=("Arial", 10))
    style.configure("TCheckbutton", font=("Arial", 10))

    # 创建测试数据
    fs = 1000
    t = np.arange(0, 60, 1 / fs)

    # 模拟EEG数据
    eeg_data = np.array([0.5 * np.sin(2 * np.pi * 10 * t) + 0.2 * np.random.randn(len(t))
                         for _ in range(16)]) * 1e-6

    data_dict = {
        "meta": {
            "subject_id": "test001",
            "session_id": "session1",
            "task": "rest",
            "modality": ["EEG"],
            "sampling_rate": fs
        },
        "signal": {
            "EEG": {
                "data": eeg_data,
                "sampling_rate": fs,
                "channel_names": [f"EEG_{i}" for i in range(16)],
                "unit": "uV"
            }
        }
    }

    # 检查预处理模块状态
    print("=" * 50)
    print("预处理模块加载状态:")
    print(f"基础预处理: {'✓' if HAS_PREPROCESSING else '✗'}")
    print(f"多模态: {'✓' if HAS_MULTIMODAL else '✗'}")
    print(f"EEG: {'✓' if HAS_EEG else '✗'}")
    print(f"fNIRS: {'✓' if HAS_FNIRS else '✗'}")
    print(f"EMG: {'✓' if HAS_EMG else '✗'}")
    print(f"ECG: {'✓' if HAS_ECG else '✗'}")
    print("=" * 50)

    view = SignalView(root, data_dict, modality="EEG")
    view.pack(fill=tk.BOTH, expand=True)

    root.mainloop()