# -*- coding: utf-8 -*-
"""
通用信号预处理模块
支持EEG、EOG、EMG、ECG、fNIRS等多种生理信号的通用预处理
兼容四层数据格式结构
"""

import numpy as np
from scipy import signal
from scipy.signal import iirnotch, butter, cheby1, bessel, filtfilt, resample_poly
import pywt
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ====================== 枚举和配置类 ======================

class FilterType(Enum):
    """滤波器类型枚举"""
    BUTTERWORTH = "butterworth"
    CHEBYSHEV1 = "chebyshev1"
    CHEBYSHEV2 = "chebyshev2"
    BESSEL = "bessel"
    ELLIPTIC = "elliptic"
    FIR = "fir"


class WaveletType(Enum):
    """小波类型枚举"""
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


class DetrendMethod(Enum):
    """去趋势方法枚举"""
    LINEAR = "linear"
    CONSTANT = "constant"
    POLY = "polynomial"
    SPLINE = "spline"


@dataclass
class PreprocessingConfig:
    """
    预处理配置类
    """
    # ========== 配置参数 ==========
    # 滤波器配置
    filter_type: FilterType = FilterType.BUTTERWORTH
    filter_order: int = 4

    # 带通滤波
    lowcut: Optional[float] = None
    highcut: Optional[float] = None

    # 陷波滤波
    notch_freq: Optional[float] = None
    notch_q: float = 30.0

    # 小波去噪
    wavelet_type: WaveletType = WaveletType.DB4
    wavelet_level: int = 4
    wavelet_threshold_method: str = "soft"

    # 重采样
    target_sampling_rate: Optional[float] = None

    # 去趋势
    detrend_method: DetrendMethod = DetrendMethod.LINEAR

    # 标准化/归一化
    normalize_method: str = "zscore"

    # 其他
    remove_baseline: bool = True
    remove_outliers: bool = False
    outlier_threshold: float = 3.0

    # 谐波陷波滤波
    use_harmonic_notch: bool = False  # 是否使用谐波陷波
    harmonic_notch_n_harmonics: int = 5  # 谐波数量

    # 自适应小波去噪（替代普通小波去噪）
    use_adaptive_wavelet: bool = False  # 是否使用自适应小波去噪

    # 中值滤波
    use_median_filter: bool = False  # 是否使用中值滤波
    median_window_size: int = 3  # 中值滤波窗口大小

    # Savitzky-Golay滤波
    use_savitzky_golay: bool = False  # 是否使用Savitzky-Golay滤波
    savitzky_window_size: int = 5  # Savitzky-Golay窗口大小
    savitzky_polyorder: int = 2  # 多项式阶数

    # 维纳滤波
    use_wiener_filter: bool = False  # 是否使用维纳滤波
    wiener_noise_variance: Optional[float] = None  # 噪声方差估计

    # EEMD去噪
    use_eemd: bool = False  # 是否使用EEMD去噪
    eemd_n_imfs: int = 5  # IMF数量


# ====================== 通用预处理器主类 ======================

class GeneralPreprocessor:
    """
    通用信号预处理类
    支持所有预处理方法，可通过配置灵活启用
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        初始化通用预处理器

        Args:
            config: 预处理配置，如果为None则使用默认配置
        """
        self.config = config if config is not None else PreprocessingConfig()
        self.history = []  # 记录预处理历史

    def process(self, data_dict: Dict[str, Any],
                modality: str = "EEG",
                channels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        主处理函数，对指定模态的信号进行预处理
        现在支持所有预处理方法，根据配置启用

        Args:
            data_dict: 输入的四层数据字典
            modality: 要处理的信号模态（如"EEG", "EOG", "EMG"等）
            channels: 指定要处理的通道，None表示处理所有通道

        Returns:
            更新后的四层数据字典
        """
        # 检查输入数据格式
        self._validate_input(data_dict, modality)

        # 创建处理历史记录
        process_record = {
            "modality": modality,
            "channels": channels,
            "steps": []
        }

        # 获取信号数据
        signal_data = data_dict["signal"][modality]["data"]
        sampling_rate = data_dict["signal"][modality]["sampling_rate"]

        # 处理前备份原始数据
        if "processed" not in data_dict:
            data_dict["processed"] = {}
        if "preprocessing" not in data_dict["processed"]:
            data_dict["processed"]["preprocessing"] = {}
        if modality not in data_dict["processed"]["preprocessing"]:
            data_dict["processed"]["preprocessing"][modality] = {
                "original_data": signal_data.copy(),
                "steps": []
            }

        # 选择指定通道
        if channels is not None:
            channel_indices = [
                data_dict["signal"][modality]["channel_names"].index(ch)
                for ch in channels if ch in data_dict["signal"][modality]["channel_names"]
            ]
            signal_data = signal_data[channel_indices, :]
            process_record["channels_selected"] = channels

        # 记录信号维度
        n_channels, n_samples = signal_data.shape

        # ========== 执行预处理步骤 ==========

        # 1. 去趋势
        if self.config.detrend_method:
            signal_data = self.detrend(signal_data, method=self.config.detrend_method)
            process_record["steps"].append({
                "step": "detrend",
                "method": self.config.detrend_method.value,
                "n_channels": n_channels
            })

        # 2. 去除基线
        if self.config.remove_baseline:
            signal_data = self.remove_baseline(signal_data)
            process_record["steps"].append({
                "step": "remove_baseline",
                "method": "mean_subtraction"
            })

        # 3. 带通滤波
        if self.config.lowcut is not None and self.config.highcut is not None:
            signal_data = self.bandpass_filter(
                signal_data,
                sampling_rate,
                self.config.lowcut,
                self.config.highcut,
                filter_type=self.config.filter_type,
                order=self.config.filter_order
            )
            process_record["steps"].append({
                "step": "bandpass_filter",
                "lowcut": self.config.lowcut,
                "highcut": self.config.highcut,
                "type": self.config.filter_type.value,
                "order": self.config.filter_order
            })

        # 4. 陷波滤波（普通或谐波）
        if self.config.notch_freq is not None:
            if self.config.use_harmonic_notch:
                # 使用谐波陷波滤波
                signal_data = self.harmonic_notch_filter(
                    signal_data,
                    sampling_rate,
                    self.config.notch_freq,
                    Q=self.config.notch_q,
                    n_harmonics=self.config.harmonic_notch_n_harmonics
                )
                process_record["steps"].append({
                    "step": "harmonic_notch_filter",
                    "base_frequency": self.config.notch_freq,
                    "n_harmonics": self.config.harmonic_notch_n_harmonics,
                    "Q": self.config.notch_q
                })
            else:
                # 使用普通陷波滤波
                signal_data = self.notch_filter(
                    signal_data,
                    sampling_rate,
                    self.config.notch_freq,
                    Q=self.config.notch_q
                )
                process_record["steps"].append({
                    "step": "notch_filter",
                    "frequency": self.config.notch_freq,
                    "Q": self.config.notch_q
                })

        # 5. 小波去噪（普通或自适应）
        if self.config.wavelet_level > 0:
            if self.config.use_adaptive_wavelet:
                # 使用自适应小波去噪
                signal_data = self.adaptive_wavelet_denoising(
                    signal_data,
                    wavelet=self.config.wavelet_type,
                    level=self.config.wavelet_level
                )
                process_record["steps"].append({
                    "step": "adaptive_wavelet_denoising",
                    "wavelet": self.config.wavelet_type.value,
                    "level": self.config.wavelet_level,
                    "method": "birge_massart"
                })
            else:
                # 使用普通小波去噪
                signal_data = self.wavelet_denoising(
                    signal_data,
                    wavelet=self.config.wavelet_type,
                    level=self.config.wavelet_level,
                    threshold_method=self.config.wavelet_threshold_method
                )
                process_record["steps"].append({
                    "step": "wavelet_denoising",
                    "wavelet": self.config.wavelet_type.value,
                    "level": self.config.wavelet_level,
                    "threshold_method": self.config.wavelet_threshold_method
                })

        # 6. 中值滤波（可选）
        if self.config.use_median_filter:
            signal_data = self.apply_median_filter(
                signal_data,
                window_size=self.config.median_window_size
            )
            process_record["steps"].append({
                "step": "median_filter",
                "window_size": self.config.median_window_size
            })

        # 7. Savitzky-Golay滤波（可选）
        if self.config.use_savitzky_golay:
            signal_data = self.apply_savitzky_golay(
                signal_data,
                window_size=self.config.savitzky_window_size,
                polyorder=self.config.savitzky_polyorder
            )
            process_record["steps"].append({
                "step": "savitzky_golay_filter",
                "window_size": self.config.savitzky_window_size,
                "polyorder": self.config.savitzky_polyorder
            })

        # 8. 维纳滤波（可选）
        if self.config.use_wiener_filter:
            signal_data = self.apply_wiener_filter(
                signal_data,
                noise_variance=self.config.wiener_noise_variance
            )
            process_record["steps"].append({
                "step": "wiener_filter",
                "noise_variance": self.config.wiener_noise_variance
            })

        # 9. EEMD去噪（可选）
        if self.config.use_eemd:
            signal_data = self.apply_ensemble_empirical_mode_decomposition(
                signal_data,
                n_imfs=self.config.eemd_n_imfs
            )
            process_record["steps"].append({
                "step": "eemd_denoising",
                "n_imfs": self.config.eemd_n_imfs
            })

        # 10. 去除离群值
        if self.config.remove_outliers:
            signal_data, outlier_info = self.remove_outliers(
                signal_data,
                threshold=self.config.outlier_threshold
            )
            process_record["steps"].append({
                "step": "remove_outliers",
                "threshold": self.config.outlier_threshold,
                "outliers_detected": outlier_info["n_outliers"]
            })

        # 11. 重采样
        if self.config.target_sampling_rate is not None:
            original_sr = sampling_rate
            signal_data = self.resample(
                signal_data,
                original_sr,
                self.config.target_sampling_rate
            )
            sampling_rate = self.config.target_sampling_rate
            process_record["steps"].append({
                "step": "resample",
                "original_fs": original_sr,
                "target_fs": sampling_rate
            })

        # 12. 标准化/归一化
        if self.config.normalize_method:
            signal_data = self.normalize(
                signal_data,
                method=self.config.normalize_method
            )
            process_record["steps"].append({
                "step": "normalize",
                "method": self.config.normalize_method
            })

        # 更新数据字典
        if channels is not None:
            # 如果只处理了部分通道，需要合并回去
            original_data = data_dict["signal"][modality]["data"]
            original_data[channel_indices, :] = signal_data
            data_dict["signal"][modality]["data"] = original_data
        else:
            data_dict["signal"][modality]["data"] = signal_data

        # 更新采样率（如果重采样了）
        data_dict["signal"][modality]["sampling_rate"] = sampling_rate

        # 记录处理历史
        data_dict["processed"]["preprocessing"][modality]["steps"].append(process_record)
        self.history.append(process_record)

        logger.info(f"预处理完成: {modality}, 处理步骤: {len(process_record['steps'])}")

        return data_dict

    def _validate_input(self, data_dict: Dict, modality: str):
        """验证输入数据格式"""
        if "signal" not in data_dict:
            raise ValueError("输入数据字典必须包含'signal'键")

        if modality not in data_dict["signal"]:
            raise ValueError(f"模态'{modality}'不在输入数据中")

        signal_info = data_dict["signal"][modality]
        required_keys = ["data", "sampling_rate", "channel_names"]
        for key in required_keys:
            if key not in signal_info:
                raise ValueError(f"信号信息必须包含'{key}'键")

        # 检查数据形状
        data = signal_info["data"]
        if not isinstance(data, np.ndarray):
            raise ValueError("信号数据必须是numpy数组")

        if len(data.shape) != 2:
            raise ValueError("信号数据必须是2维数组 (channels × samples)")

        if len(signal_info["channel_names"]) != data.shape[0]:
            raise ValueError("通道数量与数据维度不匹配")

    # ====================== 具体的预处理方法 ======================

    @staticmethod
    def detrend(data: np.ndarray, method: DetrendMethod = DetrendMethod.LINEAR,
                degree: int = 2) -> np.ndarray:
        """
        去除信号趋势
        """
        n_channels, n_samples = data.shape
        detrended_data = np.zeros_like(data)

        for i in range(n_channels):
            if method == DetrendMethod.LINEAR:
                detrended_data[i, :] = signal.detrend(data[i, :], type='linear')
            elif method == DetrendMethod.CONSTANT:
                detrended_data[i, :] = data[i, :] - np.mean(data[i, :])
            elif method == DetrendMethod.POLY:
                x = np.arange(n_samples)
                coeffs = np.polyfit(x, data[i, :], degree)
                trend = np.polyval(coeffs, x)
                detrended_data[i, :] = data[i, :] - trend
            elif method == DetrendMethod.SPLINE:
                x = np.arange(n_samples)
                knots = np.linspace(0, n_samples, min(10, n_samples // 10))
                from scipy.interpolate import splrep, splev
                tck = splrep(x, data[i, :], t=knots[1:-1])
                trend = splev(x, tck)
                detrended_data[i, :] = data[i, :] - trend
            else:
                detrended_data[i, :] = data[i, :]

        return detrended_data

    @staticmethod
    def remove_baseline(data: np.ndarray) -> np.ndarray:
        """
        去除基线偏移（DC成分）
        """
        return data - np.mean(data, axis=1, keepdims=True)

    @staticmethod
    def bandpass_filter(data: np.ndarray, fs: float, lowcut: float, highcut: float,
                        filter_type: FilterType = FilterType.BUTTERWORTH,
                        order: int = 4, ripple: float = 1.0,
                        attenuation: float = 40.0) -> np.ndarray:
        """
        带通滤波
        """
        if lowcut >= highcut:
            raise ValueError("低截止频率必须小于高截止频率")

        if lowcut <= 0 or highcut >= fs / 2:
            raise ValueError(f"截止频率必须在(0, {fs / 2})范围内")

        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist

        n_channels, n_samples = data.shape
        filtered_data = np.zeros_like(data)

        for i in range(n_channels):
            if filter_type == FilterType.BUTTERWORTH:
                b, a = butter(order, [low, high], btype='band')
            elif filter_type == FilterType.CHEBYSHEV1:
                b, a = cheby1(order, ripple, [low, high], btype='band')
            elif filter_type == FilterType.CHEBYSHEV2:
                from scipy.signal import cheby2
                b, a = cheby2(order, attenuation, [low, high], btype='band')
            elif filter_type == FilterType.BESSEL:
                b, a = bessel(order, [low, high], btype='band')
            elif filter_type == FilterType.ELLIPTIC:
                from scipy.signal import ellip
                b, a = ellip(order, ripple, attenuation, [low, high], btype='band')
            elif filter_type == FilterType.FIR:
                numtaps = order * 20
                from scipy.signal import firwin
                b = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
                a = 1.0
            else:
                raise ValueError(f"不支持的滤波器类型: {filter_type}")

            filtered_data[i, :] = filtfilt(b, a, data[i, :], padlen=min(3 * max(len(b), len(a)), n_samples // 2))

        return filtered_data

    @staticmethod
    def notch_filter(data: np.ndarray, fs: float, freq: float, Q: float = 30.0) -> np.ndarray:
        """
        陷波滤波器，去除特定频率干扰
        """
        n_channels, n_samples = data.shape
        filtered_data = np.zeros_like(data)

        for i in range(n_channels):
            b, a = iirnotch(freq, Q, fs)
            filtered_data[i, :] = filtfilt(b, a, data[i, :], padlen=min(200, n_samples // 2))

        return filtered_data

    @staticmethod
    def harmonic_notch_filter(data: np.ndarray, fs: float, base_freq: float,
                              Q: float = 30.0, n_harmonics: int = 5) -> np.ndarray:
        """
        谐波陷波滤波器，去除基频及其谐波
        """
        filtered_data = data.copy()
        for k in range(1, n_harmonics + 1):
            freq = base_freq * k
            if freq < fs / 2:
                filtered_data = GeneralPreprocessor.notch_filter(filtered_data, fs, freq, Q)

        return filtered_data

    @staticmethod
    def wavelet_denoising(data: np.ndarray, wavelet: WaveletType = WaveletType.DB4,
                          level: int = 4, threshold_method: str = "soft",
                          threshold_scale: float = 1.0) -> np.ndarray:
        """
        小波去噪
        """
        n_channels, n_samples = data.shape
        denoised_data = np.zeros_like(data)

        for i in range(n_channels):
            coeffs = pywt.wavedec(data[i, :], wavelet.value, level=level)

            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(n_samples)) * threshold_scale

            new_coeffs = []
            new_coeffs.append(coeffs[0])

            for j in range(1, len(coeffs)):
                if threshold_method == "soft":
                    new_coeffs.append(pywt.threshold(coeffs[j], threshold, mode='soft'))
                elif threshold_method == "hard":
                    new_coeffs.append(pywt.threshold(coeffs[j], threshold, mode='hard'))
                else:
                    new_coeffs.append(coeffs[j])

            denoised_signal = pywt.waverec(new_coeffs, wavelet.value)

            if len(denoised_signal) > n_samples:
                denoised_data[i, :] = denoised_signal[:n_samples]
            elif len(denoised_signal) < n_samples:
                denoised_data[i, :] = np.pad(
                    denoised_signal,
                    (0, n_samples - len(denoised_signal)),
                    mode='edge'
                )
            else:
                denoised_data[i, :] = denoised_signal

        return denoised_data

    @staticmethod
    def adaptive_wavelet_denoising(data: np.ndarray, wavelet: WaveletType = WaveletType.DB4,
                                   level: int = 4) -> np.ndarray:
        """
        自适应小波去噪（使用Birgé-Massart阈值策略）
        """
        n_channels, n_samples = data.shape
        denoised_data = np.zeros_like(data)

        for i in range(n_channels):
            coeffs = pywt.wavedec(data[i, :], wavelet.value, level=level)

            alpha = 3
            sorted_coeffs = np.sort(np.abs(np.concatenate(coeffs[1:])))
            threshold = sorted_coeffs[int(len(sorted_coeffs) * (1 - alpha / len(coeffs[1:])))]

            new_coeffs = [coeffs[0]]
            for coeff in coeffs[1:]:
                new_coeffs.append(pywt.threshold(coeff, threshold, mode='soft'))

            denoised_signal = pywt.waverec(new_coeffs, wavelet.value)

            if len(denoised_signal) > n_samples:
                denoised_data[i, :] = denoised_signal[:n_samples]
            else:
                denoised_data[i, :] = denoised_signal

        return denoised_data

    @staticmethod
    def remove_outliers(data: np.ndarray, threshold: float = 3.0,
                        method: str = "mad") -> Tuple[np.ndarray, Dict]:
        """
        去除离群值
        """
        n_channels, n_samples = data.shape
        cleaned_data = data.copy()
        outlier_info = {
            "n_outliers": 0,
            "outlier_indices": [],
            "outlier_values": []
        }

        for i in range(n_channels):
            if method == "std":
                mean_val = np.mean(data[i, :])
                std_val = np.std(data[i, :])
                lower_bound = mean_val - threshold * std_val
                upper_bound = mean_val + threshold * std_val

                outliers = np.where((data[i, :] < lower_bound) | (data[i, :] > upper_bound))[0]

            elif method == "mad":
                median_val = np.median(data[i, :])
                mad = np.median(np.abs(data[i, :] - median_val))
                std_estimate = mad * 1.4826
                lower_bound = median_val - threshold * std_estimate
                upper_bound = median_val + threshold * std_estimate

                outliers = np.where((data[i, :] < lower_bound) | (data[i, :] > upper_bound))[0]

            else:
                continue

            if len(outliers) > 0:
                if len(outliers) < n_samples:
                    time_idx = np.arange(n_samples)
                    valid_idx = np.setdiff1d(time_idx, outliers)
                    cleaned_data[i, outliers] = np.interp(
                        outliers, valid_idx, data[i, valid_idx]
                    )

                outlier_info["n_outliers"] += len(outliers)
                outlier_info["outlier_indices"].append((i, outliers.tolist()))
                outlier_info["outlier_values"].append(data[i, outliers].tolist())

        return cleaned_data, outlier_info

    @staticmethod
    def resample(data: np.ndarray, original_fs: float, target_fs: float,
                 method: str = "polyphase") -> np.ndarray:
        """
        重采样
        """
        if original_fs == target_fs:
            return data

        n_channels, n_samples = data.shape
        new_n_samples = int(n_samples * target_fs / original_fs)
        resampled_data = np.zeros((n_channels, new_n_samples))

        if method == "polyphase":
            up = int(target_fs)
            down = int(original_fs)

            gcd = np.gcd(up, down)
            up = up // gcd
            down = down // gcd

            for i in range(n_channels):
                resampled_data[i, :] = resample_poly(
                    data[i, :], up, down, axis=0
                )

        elif method == "fft":
            from scipy.signal import resample
            for i in range(n_channels):
                resampled_data[i, :] = resample(
                    data[i, :], new_n_samples, axis=0
                )

        else:
            raise ValueError(f"不支持的采样方法: {method}")

        return resampled_data

    @staticmethod
    def normalize(data: np.ndarray, method: str = "zscore",
                  feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        标准化/归一化
        """
        n_channels, n_samples = data.shape
        normalized_data = np.zeros_like(data)

        for i in range(n_channels):
            if method == "zscore":
                mean_val = np.mean(data[i, :])
                std_val = np.std(data[i, :])
                if std_val > 0:
                    normalized_data[i, :] = (data[i, :] - mean_val) / std_val
                else:
                    normalized_data[i, :] = np.zeros_like(data[i, :])

            elif method == "minmax":
                min_val = np.min(data[i, :])
                max_val = np.max(data[i, :])
                if max_val > min_val:
                    normalized_data[i, :] = (data[i, :] - min_val) / (max_val - min_val)
                    normalized_data[i, :] = (
                            normalized_data[i, :] * (feature_range[1] - feature_range[0])
                            + feature_range[0]
                    )
                else:
                    normalized_data[i, :] = np.zeros_like(data[i, :])

            elif method == "robust":
                median_val = np.median(data[i, :])
                q75, q25 = np.percentile(data[i, :], [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    normalized_data[i, :] = (data[i, :] - median_val) / iqr
                else:
                    normalized_data[i, :] = (data[i, :] - median_val)

            elif method == "unit_norm":
                norm = np.linalg.norm(data[i, :])
                if norm > 0:
                    normalized_data[i, :] = data[i, :] / norm
                else:
                    normalized_data[i, :] = data[i, :]

            else:
                raise ValueError(f"不支持的标准化方法: {method}")

        return normalized_data

    @staticmethod
    def apply_median_filter(data: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        中值滤波（有效去除脉冲噪声）
        """
        if window_size % 2 == 0:
            window_size += 1

        from scipy.signal import medfilt
        n_channels, n_samples = data.shape
        filtered_data = np.zeros_like(data)

        for i in range(n_channels):
            filtered_data[i, :] = medfilt(data[i, :], kernel_size=window_size)

        return filtered_data

    @staticmethod
    def apply_savitzky_golay(data: np.ndarray, window_size: int = 5,
                             polyorder: int = 2) -> np.ndarray:
        """
        Savitzky-Golay滤波（保持信号形状特征）
        """
        if window_size % 2 == 0:
            window_size += 1

        from scipy.signal import savgol_filter
        n_channels, n_samples = data.shape
        filtered_data = np.zeros_like(data)

        for i in range(n_channels):
            filtered_data[i, :] = savgol_filter(
                data[i, :], window_size, polyorder, mode='mirror'
            )

        return filtered_data

    @staticmethod
    def apply_wiener_filter(data: np.ndarray, noise_variance: Optional[float] = None) -> np.ndarray:
        """
        维纳滤波（最优线性滤波）
        """
        from scipy.signal import wiener
        n_channels, n_samples = data.shape
        filtered_data = np.zeros_like(data)

        for i in range(n_channels):
            if noise_variance is None:
                noise_estimate = np.var(data[i, :100]) if n_samples > 100 else np.var(data[i, :])
                filtered_data[i, :] = wiener(data[i, :], noise=noise_estimate)
            else:
                filtered_data[i, :] = wiener(data[i, :], noise=noise_variance)

        return filtered_data

    @staticmethod
    def apply_ensemble_empirical_mode_decomposition(data: np.ndarray,
                                                    n_imfs: int = 5) -> np.ndarray:
        """
        集成经验模态分解（EEMD）去噪
        """
        try:
            from PyEMD import EEMD
        except ImportError:
            logger.warning("PyEMD未安装，跳过EEMD去噪")
            return data

        n_channels, n_samples = data.shape
        denoised_data = np.zeros_like(data)

        for i in range(n_channels):
            eemd = EEMD()
            eemd.trials = 50
            imfs = eemd(data[i, :], max_imf=n_imfs)

            if len(imfs) > 1:
                denoised_signal = np.sum(imfs[1:], axis=0)
            else:
                denoised_signal = data[i, :]

            denoised_data[i, :] = denoised_signal

        return denoised_data
