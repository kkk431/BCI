# -*- coding: utf-8 -*-
"""
EMG信号专用预处理模块
基于通用预处理模块构建，专门针对肌电图信号特性优化
包含EMG特有的预处理步骤：全波整流、包络提取、运动伪迹去除等
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, resample_poly, find_peaks
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings

# 导入通用预处理模块
try:
    from preprocessing import GeneralPreprocessor, PreprocessingConfig, FilterType, WaveletType, DetrendMethod
except ImportError:
    # 如果无法导入，定义必要的类
    from enum import Enum as BaseEnum


    class FilterType(BaseEnum):
        BUTTERWORTH = "butterworth"
        CHEBYSHEV1 = "chebyshev1"
        CHEBYSHEV2 = "chebyshev2"
        BESSEL = "bessel"
        ELLIPTIC = "elliptic"
        FIR = "fir"


    class WaveletType(BaseEnum):
        DB1 = "db1"
        DB2 = "db2"
        DB4 = "db4"
        DB6 = "db6"
        DB8 = "db8"
        SYM4 = "sym4"
        SYM8 = "sym8"
        COIF1 = "coif1"
        COIF3 = "coif3"
        HAAR = "haar"


    class DetrendMethod(BaseEnum):
        LINEAR = "linear"
        CONSTANT = "constant"
        POLY = "polynomial"
        SPLINE = "spline"


    @dataclass
    class PreprocessingConfig:
        filter_type: FilterType = FilterType.BUTTERWORTH
        filter_order: int = 4
        lowcut: Optional[float] = None
        highcut: Optional[float] = None
        notch_freq: Optional[float] = None
        notch_q: float = 30.0
        wavelet_type: WaveletType = WaveletType.DB4
        wavelet_level: int = 4
        wavelet_threshold_method: str = "soft"
        target_sampling_rate: Optional[float] = None
        detrend_method: DetrendMethod = DetrendMethod.LINEAR
        normalize_method: str = "zscore"
        remove_baseline: bool = True
        remove_outliers: bool = False
        outlier_threshold: float = 3.0


    class GeneralPreprocessor:
        def __init__(self, config=None):
            self.config = config

        def process(self, data_dict, modality="EMG", channels=None):
            return data_dict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ====================== EMG专用枚举和配置 ======================

class RectificationMethod(Enum):
    """整流方法枚举"""
    NONE = "none"  # 不整流
    FULL_WAVE = "full_wave"  # 全波整流（绝对值）
    HALF_WAVE = "half_wave"  # 半波整流
    SQUARE = "square"  # 平方整流
    ROOT_MEAN_SQUARE = "rms"  # 均方根整流


class EnvelopeExtractionMethod(Enum):
    """包络提取方法枚举"""
    NONE = "none"  # 不提取包络
    MOVING_AVERAGE = "moving_average"  # 移动平均
    LOWPASS = "lowpass"  # 低通滤波
    HILBERT = "hilbert"  # 希尔伯特变换
    TEAGER_KAISER = "teager_kaiser"  # Teager-Kaiser能量算子
    ABSOLUTE = "absolute"  # 绝对值平均


class MuscleActivationDetectionMethod(Enum):
    """肌肉激活检测方法"""
    NONE = "none"  # 不检测
    THRESHOLD = "threshold"  # 阈值检测
    STATISTICAL = "statistical"  # 统计方法
    DOUBLE_THRESHOLD = "double_threshold"  # 双阈值检测
    TEAGER_KAISER = "teager_kaiser"  # Teager-Kaiser能量算子
    WAVELET = "wavelet"  # 小波变换方法


@dataclass
class EMGPreprocessingConfig(PreprocessingConfig):
    """
    EMG预处理配置类
    继承通用预处理配置，添加EMG特有参数
    """
    # ========== 基础滤波配置 ==========
    # EMG信号通常需要较宽的带通滤波范围（20-500Hz）
    emg_bandpass_low: float = 20.0  # EMG带通低截止频率（Hz）
    emg_bandpass_high: float = 500.0  # EMG带通高截止频率（Hz）
    emg_bandpass_order: int = 4  # 带通滤波器阶数

    # ========== 工频干扰去除配置 ==========
    use_harmonic_notch: bool = True  # 是否使用谐波陷波
    line_frequency: float = 50.0  # 工频频率（50或60Hz）
    notch_harmonics: int = 5  # 谐波数量
    notch_q_factor: float = 30.0  # 陷波滤波器Q值

    # ========== 整流配置 ==========
    rectification_method: RectificationMethod = RectificationMethod.FULL_WAVE  # 整流方法

    # ========== 包络提取配置 ==========
    envelope_method: EnvelopeExtractionMethod = EnvelopeExtractionMethod.LOWPASS  # 包络提取方法
    envelope_cutoff: float = 5.0  # 包络低通截止频率（Hz）
    envelope_order: int = 4  # 包络滤波器阶数

    # ========== 运动伪迹去除配置 ==========
    remove_motion_artifacts: bool = True  # 是否去除运动伪迹
    motion_artifact_threshold: float = 5.0  # 运动伪迹检测阈值（标准差倍数）

    # ========== 肌肉激活检测配置 ==========
    detect_muscle_activation: bool = True  # 是否检测肌肉激活
    activation_method: MuscleActivationDetectionMethod = MuscleActivationDetectionMethod.THRESHOLD
    activation_threshold: float = 2.0  # 激活阈值（标准差倍数）
    min_activation_duration: float = 0.05  # 最小激活持续时间（秒）

    # ========== 归一化配置 ==========
    normalize_to_mvc: bool = False  # 是否归一化到最大自主收缩
    mvc_value: Optional[float] = None  # MVC值（如果已知）
    normalize_percent: bool = True  # 是否计算百分比MVC

    # ========== 信号质量指标配置 ==========
    calculate_signal_quality: bool = True  # 是否计算信号质量指标
    signal_noise_ratio: bool = True  # 是否计算信噪比

    # ========== 高级滤波配置 ==========
    use_adaptive_filter: bool = False  # 是否使用自适应滤波去除心电干扰
    adaptive_filter_order: int = 32  # 自适应滤波器阶数
    adaptive_learning_rate: float = 0.01  # 自适应滤波器学习率

    # ========== 降采样配置 ==========
    downsample_to: Optional[float] = 1000.0  # 降采样目标频率（Hz）

    # ========== 其他配置 ==========
    remove_electrode_shifts: bool = True  # 是否去除电极偏移
    baseline_correction: bool = True  # 是否进行基线校正
    baseline_window: Tuple[float, float] = (0.0, 1.0)  # 基线时间窗口（秒）


# ====================== EMG专用预处理器 ======================

class EMGPreprocessor:
    """
    EMG信号专用预处理器
    集成通用预处理功能，添加EMG特有的处理步骤
    """

    def __init__(self, config: Optional[EMGPreprocessingConfig] = None):
        """
        初始化EMG预处理器

        Args:
            config: EMG预处理配置，None则使用默认配置
        """
        self.config = config if config is not None else EMGPreprocessingConfig()
        self.general_preprocessor = GeneralPreprocessor(self.config)
        self.history = []
        self.signal_quality_metrics = {}
        self.activation_segments = []

    def process(self, data_dict: Dict[str, Any],
                modality: str = "EMG",
                channels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        主处理函数，执行完整的EMG预处理流程

        Args:
            data_dict: 四层结构数据字典
            modality: 信号模态（应设为"EMG"）
            channels: 指定处理的通道

        Returns:
            更新后的数据字典
        """
        if modality != "EMG":
            logger.warning(f"EMG预处理器建议用于EMG信号，当前模态为{modality}")

        # 验证输入数据
        self._validate_emg_input(data_dict, modality)

        # 创建处理历史记录
        process_record = {
            "modality": modality,
            "channels": channels,
            "steps": []
        }

        # 获取原始数据
        original_data = data_dict["signal"][modality]["data"].copy()
        sampling_rate = data_dict["signal"][modality]["sampling_rate"]
        channel_names = data_dict["signal"][modality]["channel_names"]

        # 备份原始数据
        if "processed" not in data_dict:
            data_dict["processed"] = {}
        if "emg_preprocessing" not in data_dict["processed"]:
            data_dict["processed"]["emg_preprocessing"] = {}

        data_dict["processed"]["emg_preprocessing"]["original_data"] = original_data
        data_dict["processed"]["emg_preprocessing"]["original_srate"] = sampling_rate
        data_dict["processed"]["emg_preprocessing"]["channel_names"] = channel_names.copy()

        # 确保数据是2D数组
        if original_data.ndim == 1:
            original_data = original_data.reshape(1, -1)

        n_channels, n_samples = original_data.shape

        # ========== 第1阶段：基础预处理 ==========
        logger.info("开始第1阶段：基础预处理")

        # 1.1 降采样（如果需要）
        if self.config.downsample_to is not None and sampling_rate > self.config.downsample_to:
            original_data = self._downsample_data(original_data, sampling_rate, self.config.downsample_to)
            sampling_rate = self.config.downsample_to
            n_samples = original_data.shape[1]
            process_record["steps"].append({
                "step": "downsample",
                "original_fs": data_dict["signal"][modality]["sampling_rate"],
                "target_fs": sampling_rate
            })

        # 1.2 去除电极偏移
        if self.config.remove_electrode_shifts:
            original_data = self._remove_electrode_shifts(original_data)
            process_record["steps"].append({
                "step": "remove_electrode_shifts",
                "method": "median_filter"
            })

        # 1.3 基线校正
        if self.config.baseline_correction:
            original_data = self._baseline_correction(original_data, sampling_rate)
            process_record["steps"].append({
                "step": "baseline_correction",
                "window": self.config.baseline_window
            })

        # 更新数据字典中的原始数据
        data_dict["signal"][modality]["data"] = original_data
        data_dict["signal"][modality]["sampling_rate"] = sampling_rate

        # ========== 第2阶段：滤波和去噪 ==========
        logger.info("开始第2阶段：滤波和去噪")

        # 2.1 配置通用预处理器进行带通滤波
        self.config.lowcut = self.config.emg_bandpass_low
        self.config.highcut = self.config.emg_bandpass_high
        self.config.filter_order = self.config.emg_bandpass_order
        self.general_preprocessor.config = self.config

        # 2.2 应用通用预处理（带通滤波）
        data_dict = self.general_preprocessor.process(data_dict, modality, channels)
        process_record["steps"].append({
            "step": "bandpass_filter",
            "lowcut": self.config.emg_bandpass_low,
            "highcut": self.config.emg_bandpass_high,
            "order": self.config.emg_bandpass_order,
            "type": self.config.filter_type.value
        })

        # 获取滤波后的数据
        filtered_data = data_dict["signal"][modality]["data"]

        # 2.3 谐波陷波滤波（去除工频干扰）
        if self.config.use_harmonic_notch:
            filtered_data = self._apply_harmonic_notch_filter(filtered_data, sampling_rate)
            process_record["steps"].append({
                "step": "harmonic_notch_filter",
                "base_frequency": self.config.line_frequency,
                "n_harmonics": self.config.notch_harmonics,
                "Q": self.config.notch_q_factor
            })

        # 2.4 自适应滤波去除心电干扰（可选）
        if self.config.use_adaptive_filter:
            filtered_data = self._apply_adaptive_filtering(filtered_data, sampling_rate)
            process_record["steps"].append({
                "step": "adaptive_filtering",
                "order": self.config.adaptive_filter_order,
                "learning_rate": self.config.adaptive_learning_rate
            })

        # 2.5 去除运动伪迹
        if self.config.remove_motion_artifacts:
            filtered_data, motion_artifacts = self._remove_motion_artifacts(filtered_data)
            if motion_artifacts["n_artifacts"] > 0:
                process_record["steps"].append({
                    "step": "remove_motion_artifacts",
                    "n_artifacts": motion_artifacts["n_artifacts"],
                    "threshold": self.config.motion_artifact_threshold
                })

        # ========== 第3阶段：整流 ==========
        logger.info("开始第3阶段：整流")

        rectified_data = self._apply_rectification(filtered_data)
        process_record["steps"].append({
            "step": "rectification",
            "method": self.config.rectification_method.value
        })

        # ========== 第4阶段：包络提取 ==========
        logger.info("开始第4阶段：包络提取")

        if self.config.envelope_method != EnvelopeExtractionMethod.NONE:
            envelope_data = self._extract_envelope(rectified_data, sampling_rate)
            process_record["steps"].append({
                "step": "envelope_extraction",
                "method": self.config.envelope_method.value,
                "cutoff": self.config.envelope_cutoff
            })
        else:
            envelope_data = rectified_data

        # ========== 第5阶段：肌肉激活检测 ==========
        logger.info("开始第5阶段：肌肉激活检测")

        if self.config.detect_muscle_activation:
            activation_info = self._detect_muscle_activation(envelope_data, sampling_rate)
            self.activation_segments = activation_info["segments"]
            process_record["steps"].append({
                "step": "muscle_activation_detection",
                "method": self.config.activation_method.value,
                "n_activations": activation_info["n_activations"],
                "threshold": self.config.activation_threshold
            })

        # ========== 第6阶段：归一化 ==========
        logger.info("开始第6阶段：归一化")

        if self.config.normalize_to_mvc and self.config.mvc_value is not None:
            normalized_data = self._normalize_to_mvc(envelope_data)
            process_record["steps"].append({
                "step": "normalize_to_mvc",
                "mvc_value": self.config.mvc_value,
                "percent_mvc": self.config.normalize_percent
            })
        else:
            # 标准化到零均值和单位方差
            normalized_data = self._standardize_signal(envelope_data)
            process_record["steps"].append({
                "step": "standardize",
                "method": "zscore"
            })

        # ========== 第7阶段：计算信号质量指标 ==========
        logger.info("开始第7阶段：计算信号质量指标")

        if self.config.calculate_signal_quality:
            self.signal_quality_metrics = self._calculate_signal_quality_metrics(
                original_data, normalized_data, sampling_rate
            )
            process_record["steps"].append({
                "step": "calculate_signal_quality",
                "snr": self.signal_quality_metrics.get("snr_mean", 0),
                "quality_score": self.signal_quality_metrics.get("quality_score", 0)
            })

        # ========== 更新数据字典 ==========
        data_dict["signal"][modality]["data"] = normalized_data
        data_dict["signal"][modality]["sampling_rate"] = sampling_rate

        # 更新处理历史
        data_dict["processed"]["emg_preprocessing"]["history"] = process_record
        data_dict["processed"]["emg_preprocessing"]["config"] = self._config_to_dict()
        data_dict["processed"]["emg_preprocessing"]["signal_quality"] = self.signal_quality_metrics
        data_dict["processed"]["emg_preprocessing"]["activation_segments"] = self.activation_segments

        self.history.append(process_record)

        logger.info(f"EMG预处理完成，共执行{len(process_record['steps'])}个步骤")

        return data_dict

    # ====================== EMG特有方法 ======================

    def _validate_emg_input(self, data_dict: Dict, modality: str):
        """
        验证EMG输入数据

        Args:
            data_dict: 数据字典
            modality: 信号模态

        Raises:
            ValueError: 如果数据格式不符合要求
        """
        if "signal" not in data_dict:
            raise ValueError("数据字典必须包含'signal'键")

        if modality not in data_dict["signal"]:
            raise ValueError(f"模态'{modality}'不在输入数据中")

        signal_info = data_dict["signal"][modality]

        # 检查必需字段
        required_keys = ["data", "sampling_rate", "channel_names"]
        for key in required_keys:
            if key not in signal_info:
                raise ValueError(f"EMG信号必须包含'{key}'键")

        # 检查数据格式
        data = signal_info["data"]
        if not isinstance(data, np.ndarray):
            raise ValueError("EMG数据必须是numpy数组")

        # 检查采样率
        sampling_rate = signal_info["sampling_rate"]
        if sampling_rate <= 0:
            raise ValueError(f"采样率必须大于0，当前为{sampling_rate}")

        # EMG信号通常需要较高的采样率
        if sampling_rate < 100:
            logger.warning(f"EMG信号采样率较低: {sampling_rate}Hz，建议至少100Hz")

        # 检查通道数量
        n_channels = len(signal_info["channel_names"])
        if data.ndim == 1:
            logger.info("EMG数据是1维数组，将转换为2维")
        elif data.ndim == 2 and n_channels != data.shape[0]:
            raise ValueError(f"通道数量({n_channels})与数据维度({data.shape[0]})不匹配")
        elif data.ndim > 2:
            raise ValueError("EMG数据必须是1维或2维数组")

        logger.info(f"EMG数据验证通过: {n_channels}通道, {sampling_rate}Hz")

    def _downsample_data(self, data: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
        """
        降采样数据

        Args:
            data: 输入数据
            original_fs: 原始采样率
            target_fs: 目标采样率

        Returns:
            降采样后的数据
        """
        if original_fs <= target_fs:
            return data

        # 计算降采样因子
        factor = int(original_fs / target_fs)

        # 应用抗混叠滤波
        nyquist = original_fs / 2
        cutoff = target_fs / 2 * 0.8  # 80%奈奎斯特频率

        b, a = butter(4, cutoff / nyquist, btype='low')

        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape
        downsampled_data = np.zeros((n_channels, n_samples // factor))

        for i in range(n_channels):
            # 应用抗混叠滤波
            filtered = filtfilt(b, a, data[i, :])
            # 降采样
            downsampled_data[i, :] = signal.decimate(filtered, factor)

        return downsampled_data

    def _remove_electrode_shifts(self, data: np.ndarray, window_size: float = 0.1) -> np.ndarray:
        """
        去除电极偏移（基线漂移）

        Args:
            data: 输入数据
            window_size: 中值滤波窗口大小（秒）

        Returns:
            去除电极偏移后的数据
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape
        cleaned_data = np.zeros_like(data)

        # 计算窗口大小（采样点）
        if hasattr(self, 'sampling_rate'):
            fs = self.sampling_rate
        else:
            # 如果没有采样率信息，使用默认值
            fs = 1000

        window_samples = int(window_size * fs)
        if window_samples % 2 == 0:
            window_samples += 1  # 确保窗口大小为奇数

        # 应用中值滤波去除电极偏移
        for i in range(n_channels):
            # 使用中值滤波估计基线
            baseline = signal.medfilt(data[i, :], kernel_size=window_samples)
            # 减去基线
            cleaned_data[i, :] = data[i, :] - baseline

        return cleaned_data

    def _baseline_correction(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """
        基线校正

        Args:
            data: 输入数据
            sampling_rate: 采样率

        Returns:
            基线校正后的数据
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape

        # 计算基线窗口的采样点范围
        start_sample = int(self.config.baseline_window[0] * sampling_rate)
        end_sample = int(self.config.baseline_window[1] * sampling_rate)

        # 确保窗口在数据范围内
        start_sample = max(0, start_sample)
        end_sample = min(n_samples, end_sample)

        if end_sample <= start_sample:
            # 如果窗口无效，使用整个信号的均值
            return data - np.mean(data, axis=1, keepdims=True)

        corrected_data = data.copy()

        for i in range(n_channels):
            # 计算基线均值
            baseline_mean = np.mean(data[i, start_sample:end_sample])
            # 减去基线均值
            corrected_data[i, :] = data[i, :] - baseline_mean

        return corrected_data

    def _apply_harmonic_notch_filter(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """
        应用谐波陷波滤波器去除工频干扰

        Args:
            data: 输入数据
            sampling_rate: 采样率

        Returns:
            滤波后的数据
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape
        filtered_data = data.copy()

        # 计算需要去除的谐波
        max_harmonic = int(sampling_rate / 2 / self.config.line_frequency)
        if max_harmonic < 1:
            logger.warning(f"工频频率{self.config.line_frequency}Hz太高，无法应用谐波陷波")
            return data

        # 生成谐波频率列表
        freqs = [self.config.line_frequency * i for i in range(1, min(self.config.notch_harmonics, max_harmonic) + 1)]

        # 对每个谐波频率应用陷波滤波
        for freq in freqs:
            # 设计陷波滤波器
            Q = self.config.notch_q_factor
            w0 = freq / (sampling_rate / 2)  # 归一化频率

            # 二阶IIR陷波滤波器
            b, a = signal.iirnotch(w0, Q)

            for i in range(n_channels):
                filtered_data[i, :] = filtfilt(b, a, filtered_data[i, :])

        return filtered_data

    def _apply_adaptive_filtering(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """
        应用自适应滤波去除心电干扰

        Args:
            data: 输入数据
            sampling_rate: 采样率

        Returns:
            滤波后的数据
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape

        if n_channels < 2:
            logger.warning("自适应滤波需要至少2个通道，跳过")
            return data

        # 使用NLMS算法进行自适应滤波
        filtered_data = np.zeros_like(data)

        for i in range(n_channels):
            # 选择参考信号（通常选择离心脏较远的通道）
            if i == 0:
                # 第一个通道使用第二个通道作为参考
                reference = data[1, :]
            else:
                # 其他通道使用第一个通道作为参考
                reference = data[0, :]

            # 应用NLMS自适应滤波
            filtered_data[i, :] = self._nlms_filter(data[i, :], reference,
                                                    self.config.adaptive_filter_order,
                                                    self.config.adaptive_learning_rate)

        return filtered_data

    def _nlms_filter(self, primary: np.ndarray, reference: np.ndarray,
                     filter_order: int, step_size: float) -> np.ndarray:
        """
        NLMS自适应滤波算法

        Args:
            primary: 主输入信号（包含干扰）
            reference: 参考信号（干扰信号）
            filter_order: 滤波器阶数
            step_size: 步长参数

        Returns:
            滤波后的信号
        """
        n_samples = len(primary)
        filtered_signal = np.zeros(n_samples)
        w = np.zeros(filter_order)  # 滤波器权重

        for n in range(filter_order, n_samples):
            # 获取参考信号的延迟向量
            x = reference[n - filter_order + 1:n + 1][::-1]

            # 计算滤波器输出
            y = np.dot(w, x)

            # 计算误差
            e = primary[n] - y

            # 更新滤波器权重
            norm = np.dot(x, x) + 1e-10  # 防止除零
            w = w + step_size * e * x / norm

            # 保存输出
            filtered_signal[n] = e

        return filtered_signal

    def _remove_motion_artifacts(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        去除运动伪迹

        Args:
            data: 输入数据

        Returns:
            (清理后的数据, 伪迹信息字典)
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape
        cleaned_data = data.copy()
        artifact_info = {
            "n_artifacts": 0,
            "artifact_indices": [],
            "artifact_durations": []
        }

        for i in range(n_channels):
            # 计算信号的包络
            envelope = np.abs(signal.hilbert(data[i, :]))

            # 检测运动伪迹（包络的突然变化）
            threshold = np.mean(envelope) + self.config.motion_artifact_threshold * np.std(envelope)

            # 找到超过阈值的区域
            artifact_mask = envelope > threshold

            if np.any(artifact_mask):
                # 标记伪迹区域
                artifact_indices = np.where(artifact_mask)[0]

                # 对伪迹区域进行插值
                if len(artifact_indices) < n_samples:
                    time_idx = np.arange(n_samples)
                    valid_idx = np.setdiff1d(time_idx, artifact_indices)

                    if len(valid_idx) > 1:
                        cleaned_data[i, artifact_indices] = np.interp(
                            artifact_indices, valid_idx, data[i, valid_idx]
                        )

                # 记录伪迹信息
                artifact_info["n_artifacts"] += len(artifact_indices)
                artifact_info["artifact_indices"].append((i, artifact_indices.tolist()))

                # 计算伪迹持续时间
                if len(artifact_indices) > 0:
                    # 找到连续的伪迹段
                    diff_indices = np.diff(artifact_indices)
                    break_points = np.where(diff_indices > 1)[0]

                    segments = []
                    start = 0
                    for bp in break_points:
                        segments.append((artifact_indices[start], artifact_indices[bp]))
                        start = bp + 1
                    segments.append((artifact_indices[start], artifact_indices[-1]))

                    durations = [end - start + 1 for start, end in segments]
                    artifact_info["artifact_durations"].append(durations)

        return cleaned_data, artifact_info

    def _apply_rectification(self, data: np.ndarray) -> np.ndarray:
        """
        应用整流

        Args:
            data: 输入数据

        Returns:
            整流后的数据
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape

        if self.config.rectification_method == RectificationMethod.NONE:
            return data
        elif self.config.rectification_method == RectificationMethod.FULL_WAVE:
            # 全波整流（绝对值）
            return np.abs(data)
        elif self.config.rectification_method == RectificationMethod.HALF_WAVE:
            # 半波整流（保留正半波）
            return np.maximum(data, 0)
        elif self.config.rectification_method == RectificationMethod.SQUARE:
            # 平方整流
            return data ** 2
        elif self.config.rectification_method == RectificationMethod.ROOT_MEAN_SQUARE:
            # 均方根整流
            window_size = int(0.1 * (1000 if not hasattr(self, 'sampling_rate') else self.sampling_rate))
            if window_size % 2 == 0:
                window_size += 1

            rms_data = np.zeros_like(data)
            for i in range(n_channels):
                rms_data[i, :] = np.sqrt(signal.convolve(data[i, :] ** 2,
                                                         np.ones(window_size) / window_size,
                                                         mode='same'))
            return rms_data
        else:
            logger.warning(f"未知的整流方法: {self.config.rectification_method}")
            return np.abs(data)  # 默认使用全波整流

    def _extract_envelope(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """
        提取包络

        Args:
            data: 输入数据（通常为整流后的数据）
            sampling_rate: 采样率

        Returns:
            包络信号
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape

        if self.config.envelope_method == EnvelopeExtractionMethod.NONE:
            return data
        elif self.config.envelope_method == EnvelopeExtractionMethod.LOWPASS:
            # 低通滤波提取包络
            nyquist = sampling_rate / 2
            cutoff = self.config.envelope_cutoff / nyquist

            b, a = butter(self.config.envelope_order, cutoff, btype='low')

            envelope_data = np.zeros_like(data)
            for i in range(n_channels):
                envelope_data[i, :] = filtfilt(b, a, data[i, :])

            return envelope_data
        elif self.config.envelope_method == EnvelopeExtractionMethod.HILBERT:
            # 希尔伯特变换提取包络
            envelope_data = np.zeros_like(data)
            for i in range(n_channels):
                analytic_signal = signal.hilbert(data[i, :])
                envelope_data[i, :] = np.abs(analytic_signal)

            return envelope_data
        elif self.config.envelope_method == EnvelopeExtractionMethod.MOVING_AVERAGE:
            # 移动平均提取包络
            window_size = int(sampling_rate / self.config.envelope_cutoff)
            if window_size % 2 == 0:
                window_size += 1

            envelope_data = np.zeros_like(data)
            for i in range(n_channels):
                envelope_data[i, :] = signal.convolve(data[i, :],
                                                      np.ones(window_size) / window_size,
                                                      mode='same')

            return envelope_data
        elif self.config.envelope_method == EnvelopeExtractionMethod.TEAGER_KAISER:
            # Teager-Kaiser能量算子
            envelope_data = np.zeros_like(data)
            for i in range(n_channels):
                signal_sq = data[i, :] ** 2
                shifted = np.roll(data[i, :], 1)
                shifted[0] = shifted[1]
                shifted_sq = shifted ** 2

                next_shifted = np.roll(data[i, :], -1)
                next_shifted[-1] = next_shifted[-2]
                next_shifted_sq = next_shifted ** 2

                # Teager-Kaiser能量算子
                envelope_data[i, :] = np.sqrt(np.abs(signal_sq - shifted * next_shifted))

            return envelope_data
        elif self.config.envelope_method == EnvelopeExtractionMethod.ABSOLUTE:
            # 绝对值平均
            return np.abs(data)
        else:
            logger.warning(f"未知的包络提取方法: {self.config.envelope_method}")
            # 默认使用低通滤波
            return self._extract_envelope(data, sampling_rate)

    def _detect_muscle_activation(self, data: np.ndarray, sampling_rate: float) -> Dict:
        """
        检测肌肉激活

        Args:
            data: 输入数据（通常为包络信号）
            sampling_rate: 采样率

        Returns:
            激活信息字典
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape
        activation_info = {
            "n_activations": 0,
            "segments": [],
            "durations": [],
            "amplitudes": []
        }

        min_activation_samples = int(self.config.min_activation_duration * sampling_rate)

        for i in range(n_channels):
            channel_activations = []

            if self.config.activation_method == MuscleActivationDetectionMethod.THRESHOLD:
                # 阈值检测方法
                threshold = np.mean(data[i, :]) + self.config.activation_threshold * np.std(data[i, :])

                # 找到超过阈值的区域
                above_threshold = data[i, :] > threshold

                # 标记连续的区域
                diff_above = np.diff(np.concatenate(([0], above_threshold.astype(int), [0])))
                starts = np.where(diff_above == 1)[0]
                ends = np.where(diff_above == -1)[0] - 1

                for start, end in zip(starts, ends):
                    duration = (end - start + 1) / sampling_rate

                    if duration >= self.config.min_activation_duration:
                        # 计算激活幅度（峰值）
                        peak_amplitude = np.max(data[i, start:end + 1])

                        channel_activations.append({
                            "channel": i,
                            "start_sample": start,
                            "end_sample": end,
                            "start_time": start / sampling_rate,
                            "end_time": end / sampling_rate,
                            "duration": duration,
                            "peak_amplitude": peak_amplitude,
                            "mean_amplitude": np.mean(data[i, start:end + 1])
                        })

            elif self.config.activation_method == MuscleActivationDetectionMethod.STATISTICAL:
                # 统计方法（基于信号分布的异常检测）
                # 使用MAD（中位数绝对偏差）检测异常值
                median_val = np.median(data[i, :])
                mad = np.median(np.abs(data[i, :] - median_val))
                threshold = median_val + self.config.activation_threshold * mad * 1.4826

                above_threshold = data[i, :] > threshold

                # 标记连续的区域
                diff_above = np.diff(np.concatenate(([0], above_threshold.astype(int), [0])))
                starts = np.where(diff_above == 1)[0]
                ends = np.where(diff_above == -1)[0] - 1

                for start, end in zip(starts, ends):
                    duration = (end - start + 1) / sampling_rate

                    if duration >= self.config.min_activation_duration:
                        peak_amplitude = np.max(data[i, start:end + 1])

                        channel_activations.append({
                            "channel": i,
                            "start_sample": start,
                            "end_sample": end,
                            "start_time": start / sampling_rate,
                            "end_time": end / sampling_rate,
                            "duration": duration,
                            "peak_amplitude": peak_amplitude,
                            "mean_amplitude": np.mean(data[i, start:end + 1])
                        })

            elif self.config.activation_method == MuscleActivationDetectionMethod.DOUBLE_THRESHOLD:
                # 双阈值检测方法（更稳健）
                # 高阈值用于检测激活开始，低阈值用于检测激活结束
                high_threshold = np.mean(data[i, :]) + self.config.activation_threshold * np.std(data[i, :])
                low_threshold = np.mean(data[i, :]) + (self.config.activation_threshold / 2) * np.std(data[i, :])

                # 找到超过高阈值的点
                above_high = data[i, :] > high_threshold

                # 向前向后搜索直到低于低阈值
                activations = []
                n = len(data[i, :])

                idx = 0
                while idx < n:
                    if above_high[idx]:
                        # 找到激活开始
                        start = idx

                        # 向后搜索激活开始（低于低阈值）
                        j = start - 1
                        while j >= 0 and data[i, j] > low_threshold:
                            j -= 1
                        start = max(0, j + 1)

                        # 向前搜索激活结束（低于低阈值）
                        j = start
                        while j < n and (data[i, j] > low_threshold or (j - start) < min_activation_samples):
                            j += 1
                        end = min(n - 1, j - 1)

                        # 计算激活参数
                        duration = (end - start + 1) / sampling_rate

                        if duration >= self.config.min_activation_duration:
                            peak_amplitude = np.max(data[i, start:end + 1])

                            channel_activations.append({
                                "channel": i,
                                "start_sample": start,
                                "end_sample": end,
                                "start_time": start / sampling_rate,
                                "end_time": end / sampling_rate,
                                "duration": duration,
                                "peak_amplitude": peak_amplitude,
                                "mean_amplitude": np.mean(data[i, start:end + 1])
                            })

                        idx = end + 1
                    else:
                        idx += 1

            else:
                logger.warning(f"未知的激活检测方法: {self.config.activation_method}")
                # 使用阈值方法作为默认
                return self._detect_muscle_activation(data, sampling_rate)

            activation_info["segments"].extend(channel_activations)
            activation_info["n_activations"] += len(channel_activations)

            if channel_activations:
                activation_info["durations"].extend([act["duration"] for act in channel_activations])
                activation_info["amplitudes"].extend([act["peak_amplitude"] for act in channel_activations])

        return activation_info

    def _normalize_to_mvc(self, data: np.ndarray) -> np.ndarray:
        """
        归一化到最大自主收缩

        Args:
            data: 输入数据

        Returns:
            归一化后的数据
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape

        if self.config.mvc_value is not None:
            # 使用给定的MVC值
            mvc = self.config.mvc_value
        else:
            # 计算数据的最大值作为MVC估计
            mvc = np.max(data, axis=1, keepdims=True)

        # 归一化
        normalized_data = data / mvc

        if self.config.normalize_percent:
            # 转换为百分比
            normalized_data = normalized_data * 100

        return normalized_data

    def _standardize_signal(self, data: np.ndarray) -> np.ndarray:
        """
        标准化信号（零均值，单位方差）

        Args:
            data: 输入数据

        Returns:
            标准化后的数据
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape
        standardized_data = np.zeros_like(data)

        for i in range(n_channels):
            mean_val = np.mean(data[i, :])
            std_val = np.std(data[i, :])

            if std_val > 0:
                standardized_data[i, :] = (data[i, :] - mean_val) / std_val
            else:
                standardized_data[i, :] = data[i, :] - mean_val

        return standardized_data

    def _calculate_signal_quality_metrics(self, original_data: np.ndarray,
                                          processed_data: np.ndarray,
                                          sampling_rate: float) -> Dict:
        """
        计算信号质量指标

        Args:
            original_data: 原始数据
            processed_data: 处理后的数据
            sampling_rate: 采样率

        Returns:
            信号质量指标字典
        """
        if original_data.ndim == 1:
            original_data = original_data.reshape(1, -1)
        if processed_data.ndim == 1:
            processed_data = processed_data.reshape(1, -1)

        n_channels, n_samples = original_data.shape

        metrics = {
            "snr_per_channel": [],
            "snr_mean": 0,
            "snr_std": 0,
            "quality_score": 0,
            "baseline_noise": [],
            "signal_power": [],
            "noise_power": []
        }

        for i in range(n_channels):
            # 计算信噪比（SNR）
            signal_power = np.mean(processed_data[i, :] ** 2)
            noise_power = np.mean((original_data[i, :] - processed_data[i, :]) ** 2)

            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = np.inf

            metrics["snr_per_channel"].append(snr)
            metrics["signal_power"].append(signal_power)
            metrics["noise_power"].append(noise_power)

            # 计算基线噪声
            baseline_noise = np.std(processed_data[i, :100] if n_samples > 100 else processed_data[i, :])
            metrics["baseline_noise"].append(baseline_noise)

        if metrics["snr_per_channel"]:
            metrics["snr_mean"] = np.mean(metrics["snr_per_channel"])
            metrics["snr_std"] = np.std(metrics["snr_per_channel"])

        # 计算总体质量评分（0-100）
        if metrics["snr_mean"] < 0:
            quality_score = 0
        elif metrics["snr_mean"] > 50:
            quality_score = 100
        else:
            quality_score = min(100, metrics["snr_mean"] * 2)

        metrics["quality_score"] = quality_score

        return metrics

    def _config_to_dict(self) -> Dict:
        """
        将配置转换为字典

        Returns:
            配置字典
        """
        config_dict = {}
        for field in self.config.__dataclass_fields__:
            value = getattr(self.config, field)
            if isinstance(value, Enum):
                config_dict[field] = value.value
            else:
                config_dict[field] = value

        return config_dict


# ====================== 快捷配置工厂 ======================

class EMGConfigFactory:
    """
    EMG预处理配置工厂
    提供不同应用场景的推荐配置
    """

    @staticmethod
    def create_surface_emg_config() -> EMGPreprocessingConfig:
        """
        创建表面肌电信号（sEMG）的预处理配置
        用于肌肉活动监测、疲劳分析等

        Returns:
            sEMG专用配置
        """
        return EMGPreprocessingConfig(
            # 滤波配置（sEMG典型频带：20-500Hz）
            emg_bandpass_low=20.0,
            emg_bandpass_high=500.0,
            emg_bandpass_order=4,
            filter_type=FilterType.BUTTERWORTH,

            # 工频干扰去除
            use_harmonic_notch=True,
            line_frequency=50.0,
            notch_harmonics=5,
            notch_q_factor=30.0,

            # 整流和包络
            rectification_method=RectificationMethod.FULL_WAVE,
            envelope_method=EnvelopeExtractionMethod.LOWPASS,
            envelope_cutoff=5.0,
            envelope_order=4,

            # 运动伪迹去除
            remove_motion_artifacts=True,
            motion_artifact_threshold=5.0,

            # 肌肉激活检测
            detect_muscle_activation=True,
            activation_method=MuscleActivationDetectionMethod.DOUBLE_THRESHOLD,
            activation_threshold=2.0,
            min_activation_duration=0.05,

            # 归一化
            normalize_to_mvc=False,
            normalize_percent=True,

            # 信号质量
            calculate_signal_quality=True,

            # 降采样
            downsample_to=1000.0,

            # 其他配置
            remove_electrode_shifts=True,
            baseline_correction=True,
            baseline_window=(0.0, 1.0)
        )

    @staticmethod
    def create_high_density_emg_config() -> EMGPreprocessingConfig:
        """
        创建高密度肌电信号（HD-EMG）的预处理配置
        用于肌肉纤维传导速度、空间分布分析

        Returns:
            HD-EMG专用配置
        """
        return EMGPreprocessingConfig(
            # 滤波配置（HD-EMG需要更高带宽）
            emg_bandpass_low=10.0,
            emg_bandpass_high=1000.0,
            emg_bandpass_order=6,
            filter_type=FilterType.BESSEL,  # 贝塞尔滤波器保持相位信息

            # 工频干扰去除（更严格）
            use_harmonic_notch=True,
            line_frequency=50.0,
            notch_harmonics=10,
            notch_q_factor=50.0,

            # 整流和包络（通常不需要提取包络，保留原始波形）
            rectification_method=RectificationMethod.NONE,
            envelope_method=EnvelopeExtractionMethod.NONE,

            # 运动伪迹去除（更敏感）
            remove_motion_artifacts=True,
            motion_artifact_threshold=3.0,

            # 自适应滤波去除心电干扰
            use_adaptive_filter=True,
            adaptive_filter_order=64,
            adaptive_learning_rate=0.005,

            # 肌肉激活检测（更精确）
            detect_muscle_activation=True,
            activation_method=MuscleActivationDetectionMethod.STATISTICAL,
            activation_threshold=2.5,
            min_activation_duration=0.02,

            # 归一化
            normalize_to_mvc=False,

            # 信号质量
            calculate_signal_quality=True,

            # 降采样（保持较高采样率）
            downsample_to=2000.0,

            # 其他配置
            remove_electrode_shifts=True,
            baseline_correction=True,
            baseline_window=(0.0, 0.5)
        )

    @staticmethod
    def create_force_estimation_config() -> EMGPreprocessingConfig:
        """
        创建肌力估计的EMG预处理配置
        用于EMG-力关系建模

        Returns:
            肌力估计专用配置
        """
        return EMGPreprocessingConfig(
            # 滤波配置
            emg_bandpass_low=10.0,
            emg_bandpass_high=400.0,
            emg_bandpass_order=4,
            filter_type=FilterType.BUTTERWORTH,

            # 工频干扰去除
            use_harmonic_notch=True,
            line_frequency=50.0,
            notch_harmonics=5,
            notch_q_factor=30.0,

            # 整流和包络（RMS整流更适合力估计）
            rectification_method=RectificationMethod.ROOT_MEAN_SQUARE,
            envelope_method=EnvelopeExtractionMethod.LOWPASS,
            envelope_cutoff=3.0,
            envelope_order=4,

            # 运动伪迹去除
            remove_motion_artifacts=True,
            motion_artifact_threshold=4.0,

            # 肌肉激活检测
            detect_muscle_activation=False,  # 力估计通常不需要激活检测

            # 归一化到MVC
            normalize_to_mvc=True,
            normalize_percent=True,

            # 信号质量
            calculate_signal_quality=True,

            # 降采样
            downsample_to=500.0,  # 力估计不需要高频成分

            # 其他配置
            remove_electrode_shifts=True,
            baseline_correction=True,
            baseline_window=(0.0, 1.0)
        )

    @staticmethod
    def create_fatigue_analysis_config() -> EMGPreprocessingConfig:
        """
        创建肌肉疲劳分析的EMG预处理配置
        用于频域分析和中位频率计算

        Returns:
            疲劳分析专用配置
        """
        return EMGPreprocessingConfig(
            # 滤波配置（保留更宽频带用于频域分析）
            emg_bandpass_low=5.0,
            emg_bandpass_high=500.0,
            emg_bandpass_order=4,
            filter_type=FilterType.BUTTERWORTH,

            # 工频干扰去除（特别重要，工频会影响频域分析）
            use_harmonic_notch=True,
            line_frequency=50.0,
            notch_harmonics=10,
            notch_q_factor=50.0,

            # 整流和包络（通常不需要）
            rectification_method=RectificationMethod.NONE,
            envelope_method=EnvelopeExtractionMethod.NONE,

            # 运动伪迹去除（疲劳分析对伪迹敏感）
            remove_motion_artifacts=True,
            motion_artifact_threshold=3.0,

            # 肌肉激活检测
            detect_muscle_activation=True,
            activation_method=MuscleActivationDetectionMethod.THRESHOLD,
            activation_threshold=2.0,
            min_activation_duration=1.0,  # 疲劳分析需要较长的激活段

            # 归一化
            normalize_to_mvc=False,

            # 信号质量
            calculate_signal_quality=True,

            # 降采样（保持较高采样率用于频域分析）
            downsample_to=2000.0,

            # 其他配置
            remove_electrode_shifts=True,
            baseline_correction=True,
            baseline_window=(0.0, 0.5)
        )

    @staticmethod
    def create_realtime_config() -> EMGPreprocessingConfig:
        """
        创建实时处理的EMG预处理配置
        用于BCI控制、假肢控制等实时应用

        Returns:
            实时处理专用配置
        """
        return EMGPreprocessingConfig(
            # 滤波配置（平衡效果和计算效率）
            emg_bandpass_low=20.0,
            emg_bandpass_high=200.0,  # 限制高频减少计算量
            emg_bandpass_order=2,  # 低阶滤波器计算更快
            filter_type=FilterType.BUTTERWORTH,

            # 工频干扰去除
            use_harmonic_notch=True,
            line_frequency=50.0,
            notch_harmonics=3,  # 减少谐波数量
            notch_q_factor=20.0,

            # 整流和包络
            rectification_method=RectificationMethod.FULL_WAVE,
            envelope_method=EnvelopeExtractionMethod.MOVING_AVERAGE,  # 移动平均计算更快
            envelope_cutoff=10.0,

            # 运动伪迹去除（简化版本）
            remove_motion_artifacts=True,
            motion_artifact_threshold=6.0,  # 更高阈值减少误报

            # 肌肉激活检测
            detect_muscle_activation=True,
            activation_method=MuscleActivationDetectionMethod.THRESHOLD,
            activation_threshold=3.0,
            min_activation_duration=0.1,

            # 归一化
            normalize_to_mvc=True,
            normalize_percent=True,

            # 信号质量（实时应用中可能不需要）
            calculate_signal_quality=False,

            # 降采样（降低计算负担）
            downsample_to=200.0,

            # 其他配置
            remove_electrode_shifts=True,
            baseline_correction=True,
            baseline_window=(0.0, 0.5)
        )


# ====================== 使用示例和测试函数 ======================

def test_emg_preprocessing():
    """
    测试EMG预处理功能
    """
    # 创建模拟数据
    np.random.seed(42)

    # 模拟参数
    n_channels = 4  # 4个EMG通道
    n_samples = 5000  # 5秒数据，1000Hz采样率
    sampling_rate = 1000

    # 生成模拟EMG数据
    time = np.arange(n_samples) / sampling_rate

    # 基础EMG信号（模拟肌肉收缩）
    emg_data = np.zeros((n_channels, n_samples))

    # 通道1：模拟持续收缩
    contraction_times = [(1.0, 2.5), (3.5, 4.5)]
    for start, end in contraction_times:
        start_sample = int(start * sampling_rate)
        end_sample = int(end * sampling_rate)
        duration = end_sample - start_sample
        emg_data[0, start_sample:end_sample] = 50.0 * np.random.randn(duration) + 100.0

    # 通道2：模拟间歇性收缩
    for i in range(5):
        start = 0.5 + i * 0.8
        end = start + 0.3
        start_sample = int(start * sampling_rate)
        end_sample = int(end * sampling_rate)
        duration = end_sample - start_sample
        emg_data[1, start_sample:end_sample] = 30.0 * np.random.randn(duration) + 80.0

    # 通道3和4：模拟背景活动和噪声
    emg_data[2, :] = 10.0 * np.random.randn(n_samples) + 5.0
    emg_data[3, :] = 15.0 * np.random.randn(n_samples) + 8.0

    # 添加工频干扰（50Hz及其谐波）
    line_noise = 20.0 * np.sin(2 * np.pi * 50.0 * time)
    line_noise += 10.0 * np.sin(2 * np.pi * 100.0 * time)
    emg_data += line_noise.reshape(1, -1)

    # 添加运动伪迹（模拟突然移动）
    motion_times = [2.0, 4.0]
    for motion_time in motion_times:
        motion_sample = int(motion_time * sampling_rate)
        motion_duration = int(0.1 * sampling_rate)
        motion_signal = 200.0 * np.hanning(motion_duration)
        start = max(0, motion_sample - motion_duration // 2)
        end = min(n_samples, start + motion_duration)
        actual_duration = end - start
        emg_data[:, start:end] += motion_signal[:actual_duration].reshape(1, -1)

    # 添加电极偏移
    electrode_shift = 50.0 * (time > 2.5)
    emg_data += electrode_shift.reshape(1, -1)

    # 通道名称
    channel_names = ["Biceps", "Triceps", "Flexor", "Extensor"]

    # 创建模拟事件（肌肉激活开始）
    event_times = [1.0, 3.5]
    event_labels = ["contraction_start", "contraction_start"]

    # 构建数据字典
    data_dict = {
        "meta": {
            "subject_id": "S01",
            "session_id": "emg_test_session",
            "task": "muscle_activation",
            "modality": ["EMG"],
            "device": "Simulated_EMG",
            "sampling_rate": sampling_rate,
            "n_channels": n_channels,
            "channel_names": channel_names
        },
        "signal": {
            "EMG": {
                "data": emg_data,
                "sampling_rate": sampling_rate,
                "unit": "uV",
                "channel_names": channel_names,
                "time_offset": 0.0
            }
        },
        "event": {
            "event_id": [1, 1],
            "event_label": event_labels,
            "event_time": event_times,
            "event_sample": [int(t * sampling_rate) for t in event_times],
            "duration": [1.5, 1.0]
        },
        "processed": {}
    }

    print("=" * 60)
    print("EMG预处理测试")
    print("=" * 60)

    # 创建预处理器（使用sEMG配置）
    config = EMGConfigFactory.create_surface_emg_config()
    preprocessor = EMGPreprocessor(config)

    # 执行预处理
    print("\n开始预处理...")
    processed_data = preprocessor.process(data_dict)

    # 显示处理结果
    print("\n预处理完成!")
    print(f"原始数据形状: {data_dict['signal']['EMG']['data'].shape}")
    print(f"处理后数据形状: {processed_data['signal']['EMG']['data'].shape}")

    # 显示处理历史
    history = processed_data['processed']['emg_preprocessing']['history']
    print(f"\n处理步骤数量: {len(history['steps'])}")
    print("\n处理步骤详情:")
    for i, step in enumerate(history['steps']):
        step_name = step['step']
        details = {k: v for k, v in step.items() if k != 'step'}
        print(f"  {i + 1:2d}. {step_name:30s} | {details}")

    # 显示信号质量指标
    if 'signal_quality' in processed_data['processed']['emg_preprocessing']:
        quality_metrics = processed_data['processed']['emg_preprocessing']['signal_quality']
        print(f"\n信号质量指标:")
        print(f"  平均信噪比: {quality_metrics.get('snr_mean', 0):.2f} dB")
        print(f"  质量评分: {quality_metrics.get('quality_score', 0):.1f}/100")
        print(f"  各通道SNR: {[f'{snr:.1f}' for snr in quality_metrics.get('snr_per_channel', [])]}")

    # 显示肌肉激活检测结果
    if 'activation_segments' in processed_data['processed']['emg_preprocessing']:
        activations = processed_data['processed']['emg_preprocessing']['activation_segments']
        if activations:
            print(f"\n检测到的肌肉激活:")
            for i, act in enumerate(activations[:3]):  # 只显示前3个
                print(f"  激活{i + 1}: 通道{act['channel']}, "
                      f"时间[{act['start_time']:.2f}-{act['end_time']:.2f}]s, "
                      f"持续时间{act['duration']:.2f}s, "
                      f"峰值幅度{act['peak_amplitude']:.1f}")
            if len(activations) > 3:
                print(f"  ... 还有{len(activations) - 3}个激活")
        else:
            print("\n未检测到肌肉激活")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    return processed_data


# ====================== 批处理和多通道处理 ======================

class EMGBatchProcessor:
    """
    EMG批处理器
    用于批量处理多个EMG数据文件
    """

    def __init__(self, config: Optional[EMGPreprocessingConfig] = None):
        """
        初始化EMG批处理器

        Args:
            config: EMG预处理配置
        """
        self.config = config if config is not None else EMGPreprocessingConfig()
        self.preprocessor = EMGPreprocessor(self.config)
        self.results = []

    def process_batch(self, data_dicts: List[Dict[str, Any]],
                      modalities: Union[str, List[str]] = "EMG") -> List[Dict[str, Any]]:
        """
        批量处理多个数据字典

        Args:
            data_dicts: 数据字典列表
            modalities: 模态列表或单个模态

        Returns:
            处理后的数据字典列表
        """
        processed_results = []

        for i, data_dict in enumerate(data_dicts):
            logger.info(f"处理第{i + 1}/{len(data_dicts)}个数据文件...")

            try:
                if isinstance(modalities, str):
                    modalities_list = [modalities]
                else:
                    modalities_list = modalities

                for modality in modalities_list:
                    if modality in data_dict["signal"]:
                        processed_data = self.preprocessor.process(data_dict, modality)
                        processed_results.append(processed_data)
                    else:
                        logger.warning(f"数据文件{i + 1}中未找到模态{modality}")

            except Exception as e:
                logger.error(f"处理数据文件{i + 1}时出错: {str(e)}")
                # 跳过错误，继续处理下一个文件
                continue

        self.results = processed_results
        logger.info(f"批处理完成，成功处理{len(processed_results)}/{len(data_dicts)}个文件")

        return processed_results

    def save_results(self, save_path: str, format: str = "numpy"):
        """
        保存处理结果

        Args:
            save_path: 保存路径
            format: 保存格式（"numpy", "mat", "pickle"等）
        """
        import os
        import pickle

        os.makedirs(save_path, exist_ok=True)

        for i, result in enumerate(self.results):
            file_path = os.path.join(save_path, f"emg_processed_{i + 1:03d}.{format}")

            if format == "numpy":
                # 保存信号数据和元数据
                np.savez(file_path,
                         data=result["signal"]["EMG"]["data"],
                         sampling_rate=result["signal"]["EMG"]["sampling_rate"],
                         channel_names=result["signal"]["EMG"]["channel_names"],
                         meta=result["meta"],
                         processed=result.get("processed", {}))

            elif format == "pickle":
                with open(file_path, 'wb') as f:
                    pickle.dump(result, f)

            else:
                logger.warning(f"不支持的保存格式: {format}")

        logger.info(f"结果已保存到: {save_path}")


# ====================== 主程序入口 ======================

if __name__ == "__main__":
    # 运行测试
    processed_data = test_emg_preprocessing()

    # 示例：如何使用EMG预处理器
    print("\n" + "=" * 60)
    print("EMG预处理器使用示例")
    print("=" * 60)

    # 示例1：使用默认配置
    print("\n1. 使用默认配置:")
    config1 = EMGPreprocessingConfig()
    preprocessor1 = EMGPreprocessor(config1)
    print(f"  带通滤波: {config1.emg_bandpass_low}-{config1.emg_bandpass_high}Hz")
    print(f"  整流方法: {config1.rectification_method.value}")
    print(f"  包络提取: {config1.envelope_method.value}")

    # 示例2：使用表面肌电配置
    print("\n2. 使用表面肌电配置:")
    config2 = EMGConfigFactory.create_surface_emg_config()
    preprocessor2 = EMGPreprocessor(config2)
    print(f"  降采样: {config2.downsample_to}Hz")
    print(f"  激活检测: {config2.activation_method.value}")
    print(f"  运动伪迹去除: {'启用' if config2.remove_motion_artifacts else '禁用'}")

    # 示例3：使用实时处理配置
    print("\n3. 使用实时处理配置:")
    config3 = EMGConfigFactory.create_realtime_config()
    print(f"  滤波器阶数: {config3.emg_bandpass_order}阶")
    print(f"  包络截止频率: {config3.envelope_cutoff}Hz")
    print(f"  最小激活持续时间: {config3.min_activation_duration}s")

    # 示例4：批处理
    print("\n4. 批处理示例:")
    batch_processor = EMGBatchProcessor(config2)
    print("  可以处理多个EMG数据文件")
    print("  支持自动保存处理结果")

    print("\n" + "=" * 60)
    print("EMG预处理模块已准备好!")
    print("=" * 60)