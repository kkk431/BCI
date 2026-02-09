"""
通用特征提取模块 (common_features.py)

该模块提供了多模态脑机接口信号处理的通用特征提取方法。
包括时域、频域、时频域和非线性特征提取功能。

作者: 多模态脑机接口团队
版本: 1.0.0
日期: 2023-10-01
"""

import numpy as np
from scipy import stats, signal, fft, integrate
from scipy.spatial.distance import pdist, squareform
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import entropy as scipy_entropy
import pywt
from typing import Union, List, Tuple, Dict, Optional, Callable
import warnings


class SignalQualityWarning(Warning):
    """信号质量警告"""
    pass


class FeatureExtractor:
    """
    通用特征提取器基类

    提供各种信号处理中通用的特征提取方法，适用于EEG、fNIRS、ECG、EMG等时间序列信号。
    """

    def __init__(self,
                 sampling_rate: float = 250.0,
                 normalize: bool = True,
                 verbose: bool = False):
        """
        初始化特征提取器

        参数:
        ----------
        sampling_rate : float, 默认=250.0
            信号的采样率(Hz)
        normalize : bool, 默认=True
            是否在特征提取前对信号进行z-score标准化
        verbose : bool, 默认=False
            是否打印详细处理信息
        """
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        self.verbose = verbose

        # 初始化缓存以提高性能
        self._cache = {}

    def _validate_input(self, signal_data: np.ndarray) -> np.ndarray:
        """
        验证输入信号并返回处理后的信号

        参数:
        ----------
        signal_data : np.ndarray
            输入信号，可以是1D或2D数组

        返回:
        ----------
        np.ndarray: 验证并处理后的信号
        """
        if signal_data is None:
            raise ValueError("输入信号不能为None")

        # 确保是numpy数组
        signal_data = np.asarray(signal_data, dtype=np.float64)

        # 确保至少是1D
        if signal_data.ndim == 0:
            raise ValueError("输入信号必须是至少1维的数组")

        # 如果是1D，转换为2D (1, n_samples)
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(1, -1)

        # 检查NaN和Inf
        if np.any(np.isnan(signal_data)):
            warnings.warn("输入信号包含NaN值，将用线性插值处理", SignalQualityWarning)
            for i in range(signal_data.shape[0]):
                nan_mask = np.isnan(signal_data[i])
                if np.any(nan_mask):
                    signal_data[i, nan_mask] = np.interp(
                        np.where(nan_mask)[0],
                        np.where(~nan_mask)[0],
                        signal_data[i, ~nan_mask]
                    )

        if np.any(np.isinf(signal_data)):
            warnings.warn("输入信号包含无穷值，将用邻近值替换", SignalQualityWarning)
            for i in range(signal_data.shape[0]):
                inf_mask = np.isinf(signal_data[i])
                if np.any(inf_mask):
                    # 找到非无穷值的索引
                    valid_indices = np.where(~inf_mask)[0]
                    if len(valid_indices) > 0:
                        for idx in np.where(inf_mask)[0]:
                            # 找到最近的合法值
                            nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - idx))]
                            signal_data[i, idx] = signal_data[i, nearest_idx]
                    else:
                        # 所有值都是无穷，设置为0
                        signal_data[i, inf_mask] = 0.0

        return signal_data

    def _preprocess_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """
        预处理信号

        参数:
        ----------
        signal_data : np.ndarray
            输入信号

        返回:
        ----------
        np.ndarray: 预处理后的信号
        """
        processed_signal = self._validate_input(signal_data)

        # 去趋势（移除线性趋势）
        for i in range(processed_signal.shape[0]):
            x = np.arange(len(processed_signal[i]))
            slope, intercept = np.polyfit(x, processed_signal[i], 1)
            processed_signal[i] = processed_signal[i] - (slope * x + intercept)

        # 标准化
        if self.normalize:
            for i in range(processed_signal.shape[0]):
                std = np.std(processed_signal[i])
                if std > 1e-10:  # 避免除零
                    processed_signal[i] = (processed_signal[i] - np.mean(processed_signal[i])) / std

        return processed_signal

    def _get_cached_or_compute(self,
                               key: str,
                               compute_func: Callable,
                               *args, **kwargs) -> np.ndarray:
        """
        获取缓存结果或计算并缓存

        参数:
        ----------
        key : str
            缓存键
        compute_func : Callable
            计算函数
        *args, **kwargs :
            传递给计算函数的参数

        返回:
        ----------
        np.ndarray: 计算结果
        """
        cache_key = f"{key}_{hash(str(args))}_{hash(str(kwargs))}"

        if cache_key in self._cache:
            if self.verbose:
                print(f"从缓存获取: {key}")
            return self._cache[cache_key]
        else:
            result = compute_func(*args, **kwargs)
            self._cache[cache_key] = result
            return result

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()


class TimeDomainFeatures(FeatureExtractor):
    """时域特征提取类"""

    def compute_statistical_features(self,
                                     signal_data: np.ndarray,
                                     features: List[str] = None) -> Dict[str, np.ndarray]:
        """
        计算基本统计特征

        参数:
        ----------
        signal_data : np.ndarray
            输入信号，形状为(n_channels, n_samples)或(n_samples,)
        features : List[str], 可选
            要计算的特征列表，如果为None则计算所有特征

        返回:
        ----------
        Dict[str, np.ndarray]: 特征字典，每个特征对应一个形状为(n_channels,)的数组
        """
        processed_signal = self._preprocess_signal(signal_data)
        n_channels, n_samples = processed_signal.shape

        # 默认特征列表
        if features is None:
            features = ['mean', 'variance', 'std', 'skewness', 'kurtosis',
                        'min', 'max', 'peak_to_peak', 'rms', 'percentiles']

        feature_dict = {}

        for i in range(n_channels):
            channel_signal = processed_signal[i]

            # 确保信号长度足够
            if len(channel_signal) < 10:
                warnings.warn(f"信号长度({len(channel_signal)})可能不足以保证统计可靠性",
                              SignalQualityWarning)

            channel_features = {}

            if 'mean' in features:
                channel_features['mean'] = np.mean(channel_signal)

            if 'variance' in features or 'std' in features:
                variance = np.var(channel_signal)
                if 'variance' in features:
                    channel_features['variance'] = variance
                if 'std' in features:
                    channel_features['std'] = np.sqrt(variance)

            if 'skewness' in features:
                # 使用scipy的偏度计算，更稳定
                channel_features['skewness'] = stats.skew(channel_signal)

            if 'kurtosis' in features:
                # 使用scipy的峰度计算，Fisher定义（正态分布为0）
                channel_features['kurtosis'] = stats.kurtosis(channel_signal)

            if 'min' in features:
                channel_features['min'] = np.min(channel_signal)

            if 'max' in features:
                channel_features['max'] = np.max(channel_signal)

            if 'peak_to_peak' in features:
                channel_features['peak_to_peak'] = np.ptp(channel_signal)

            if 'rms' in features:
                channel_features['rms'] = np.sqrt(np.mean(channel_signal ** 2))

            if 'percentiles' in features:
                percentiles = [25, 50, 75]  # 25th, 50th(median), 75th
                percentile_values = np.percentile(channel_signal, percentiles)
                for p, val in zip(percentiles, percentile_values):
                    channel_features[f'percentile_{p}'] = val

            if 'iqr' in features:
                q75, q25 = np.percentile(channel_signal, [75, 25])
                channel_features['iqr'] = q75 - q25

            # 将通道特征添加到总体字典中
            for feat_name, feat_value in channel_features.items():
                if feat_name not in feature_dict:
                    feature_dict[feat_name] = np.zeros(n_channels)
                feature_dict[feat_name][i] = feat_value

        return feature_dict

    def compute_rms(self, signal_data: np.ndarray) -> np.ndarray:
        """
        计算均方根值(Root Mean Square)

        参数:
        ----------
        signal_data : np.ndarray
            输入信号

        返回:
        ----------
        np.ndarray: 每个通道的RMS值
        """
        processed_signal = self._preprocess_signal(signal_data)
        return np.sqrt(np.mean(processed_signal ** 2, axis=1))

    def compute_zcr(self,
                    signal_data: np.ndarray,
                    threshold: float = 0.0) -> np.ndarray:
        """
        计算过零点率(Zero-Crossing Rate)

        参数:
        ----------
        signal_data : np.ndarray
            输入信号
        threshold : float, 默认=0.0
            过零点检测的阈值，用于减少噪声的影响

        返回:
        ----------
        np.ndarray: 每个通道的过零点率
        """
        processed_signal = self._preprocess_signal(signal_data)
        n_channels, n_samples = processed_signal.shape

        zcr_values = np.zeros(n_channels)

        for i in range(n_channels):
            channel_signal = processed_signal[i]

            # 计算过零点
            zero_crossings = np.where(np.diff(np.sign(channel_signal - threshold)))[0]

            # 计算过零点率（每秒钟的过零点数）
            zcr_values[i] = len(zero_crossings) * self.sampling_rate / n_samples

        return zcr_values

    def compute_hjorth_parameters(self, signal_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算Hjorth参数：活动性、移动性、复杂性

        参数:
        ----------
        signal_data : np.ndarray
            输入信号

        返回:
        ----------
        Dict[str, np.ndarray]: Hjorth参数字典
        """
        processed_signal = self._preprocess_signal(signal_data)
        n_channels, n_samples = processed_signal.shape

        activity = np.zeros(n_channels)
        mobility = np.zeros(n_channels)
        complexity = np.zeros(n_channels)

        for i in range(n_channels):
            channel_signal = processed_signal[i]

            # 活动性：信号的方差
            activity[i] = np.var(channel_signal)

            if activity[i] > 1e-10:  # 避免除零
                # 一阶导数
                first_derivative = np.diff(channel_signal)
                var_first_deriv = np.var(first_derivative)

                # 移动性：一阶导数的标准差与原始信号标准差的比值
                mobility[i] = np.sqrt(var_first_deriv / activity[i])

                if var_first_deriv > 1e-10:
                    # 二阶导数
                    second_derivative = np.diff(first_derivative)
                    var_second_deriv = np.var(second_derivative)

                    # 复杂性：二阶导数的移动性与一阶导数的移动性的比值
                    mobility_second = np.sqrt(var_second_deriv / var_first_deriv)
                    complexity[i] = mobility_second / mobility[i]

        return {
            'hjorth_activity': activity,
            'hjorth_mobility': mobility,
            'hjorth_complexity': complexity
        }

    def compute_waveform_length(self, signal_data: np.ndarray) -> np.ndarray:
        """
        计算波形长度（信号绝对变化的总和）

        参数:
        ----------
        signal_data : np.ndarray
            输入信号

        返回:
        ----------
        np.ndarray: 每个通道的波形长度
        """
        processed_signal = self._preprocess_signal(signal_data)
        return np.sum(np.abs(np.diff(processed_signal, axis=1)), axis=1)

    def compute_willison_amplitude(self,
                                   signal_data: np.ndarray,
                                   threshold: float = 0.01) -> np.ndarray:
        """
        计算Willison幅值（超过阈值的差分数量）

        参数:
        ----------
        signal_data : np.ndarray
            输入信号
        threshold : float, 默认=0.01
            差分阈值

        返回:
        ----------
        np.ndarray: 每个通道的Willison幅值
        """
        processed_signal = self._preprocess_signal(signal_data)
        n_channels, n_samples = processed_signal.shape

        willison_values = np.zeros(n_channels)

        for i in range(n_channels):
            channel_signal = processed_signal[i]
            diff_signal = np.abs(np.diff(channel_signal))
            willison_values[i] = np.sum(diff_signal > threshold)

        return willison_values


class FrequencyDomainFeatures(FeatureExtractor):
    """频域特征提取类"""

    def compute_spectral_features(self,
                                  signal_data: np.ndarray,
                                  nperseg: int = 256,
                                  noverlap: int = None,
                                  window: str = 'hann') -> Dict[str, np.ndarray]:
        """
        计算频谱特征

        参数:
        ----------
        signal_data : np.ndarray
            输入信号
        nperseg : int, 默认=256
            Welch方法中每个段的长度
        noverlap : int, 可选
            段之间的重叠点数，默认为nperseg//2
        window : str, 默认='hann'
            窗函数类型

        返回:
        ----------
        Dict[str, np.ndarray]: 频谱特征字典
        """
        processed_signal = self._preprocess_signal(signal_data)
        n_channels, n_samples = processed_signal.shape

        if noverlap is None:
            noverlap = nperseg // 2

        feature_dict = {}

        for i in range(n_channels):
            channel_signal = processed_signal[i]

            # 使用Welch方法计算功率谱密度
            freqs, psd = signal.welch(
                channel_signal,
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window=window
            )

            # 总功率
            total_power = np.trapz(psd, freqs)

            # 频谱质心（加权平均频率）
            if total_power > 1e-10:
                spectral_centroid = np.trapz(freqs * psd, freqs) / total_power
            else:
                spectral_centroid = 0.0

            # 频谱带宽（二阶矩）
            if total_power > 1e-10:
                spectral_bandwidth = np.sqrt(
                    np.trapz((freqs - spectral_centroid) ** 2 * psd, freqs) / total_power
                )
            else:
                spectral_bandwidth = 0.0

            # 频谱滚降点（95%功率所在的频率）
            cumulative_power = np.cumsum(psd)
            cumulative_power_normalized = cumulative_power / cumulative_power[-1]
            spectral_rolloff = freqs[np.where(cumulative_power_normalized >= 0.95)[0][0]]

            # 频谱平坦度（维纳熵）
            geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
            arithmetic_mean = np.mean(psd)
            spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0

            # 峰值频率
            peak_freq = freqs[np.argmax(psd)]

            # 频谱不对称性
            half_idx = len(freqs) // 2
            low_freq_power = np.trapz(psd[:half_idx], freqs[:half_idx])
            high_freq_power = np.trapz(psd[half_idx:], freqs[half_idx:])
            spectral_asymmetry = (high_freq_power - low_freq_power) / (high_freq_power + low_freq_power + 1e-10)

            # 存储特征
            channel_features = {
                'spectral_centroid': spectral_centroid,
                'spectral_bandwidth': spectral_bandwidth,
                'spectral_rolloff': spectral_rolloff,
                'spectral_flatness': spectral_flatness,
                'peak_frequency': peak_freq,
                'spectral_asymmetry': spectral_asymmetry,
                'total_power': total_power
            }

            for feat_name, feat_value in channel_features.items():
                if feat_name not in feature_dict:
                    feature_dict[feat_name] = np.zeros(n_channels)
                feature_dict[feat_name][i] = feat_value

        return feature_dict

    def compute_spectral_entropy(self,
                                 signal_data: np.ndarray,
                                 normalize: bool = True) -> np.ndarray:
        """
        计算谱熵（频谱功率分布的熵）

        参数:
        ----------
        signal_data : np.ndarray
            输入信号
        normalize : bool, 默认=True
            是否归一化熵值（0-1之间）

        返回:
        ----------
        np.ndarray: 每个通道的谱熵
        """
        processed_signal = self._preprocess_signal(signal_data)
        n_channels, n_samples = processed_signal.shape

        spectral_entropies = np.zeros(n_channels)

        for i in range(n_channels):
            channel_signal = processed_signal[i]

            # 计算功率谱
            freqs, psd = signal.welch(channel_signal, fs=self.sampling_rate)

            # 归一化功率谱
            psd_normalized = psd / (np.sum(psd) + 1e-10)

            # 计算谱熵
            entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-10))

            if normalize:
                # 最大熵（均匀分布）
                max_entropy = np.log2(len(psd_normalized))
                if max_entropy > 0:
                    entropy = entropy / max_entropy

            spectral_entropies[i] = entropy

        return spectral_entropies

    def compute_spectral_edge_frequency(self,
                                        signal_data: np.ndarray,
                                        percentiles: List[float] = [50, 75, 90, 95]) -> Dict[str, np.ndarray]:
        """
        计算频谱边缘频率（特定百分比功率所在的频率）

        参数:
        ----------
        signal_data : np.ndarray
            输入信号
        percentiles : List[float], 默认=[50, 75, 90, 95]
            要计算的百分位数

        返回:
        ----------
        Dict[str, np.ndarray]: 频谱边缘频率字典
        """
        processed_signal = self._preprocess_signal(signal_data)
        n_channels, n_samples = processed_signal.shape

        feature_dict = {}

        for i in range(n_channels):
            channel_signal = processed_signal[i]

            # 计算功率谱
            freqs, psd = signal.welch(channel_signal, fs=self.sampling_rate)

            # 累积功率
            cumulative_power = np.cumsum(psd)
            total_power = cumulative_power[-1]

            if total_power > 1e-10:
                cumulative_power_normalized = cumulative_power / total_power

                for percentile in percentiles:
                    threshold = percentile / 100.0
                    edge_idx = np.where(cumulative_power_normalized >= threshold)[0]

                    if len(edge_idx) > 0:
                        edge_freq = freqs[edge_idx[0]]
                    else:
                        edge_freq = freqs[-1]

                    feat_name = f'sef_{percentile}'
                    if feat_name not in feature_dict:
                        feature_dict[feat_name] = np.zeros(n_channels)
                    feature_dict[feat_name][i] = edge_freq

        return feature_dict

    def compute_band_power_ratio(self,
                                 signal_data: np.ndarray,
                                 band_edges: List[Tuple[float, float]] = None) -> Dict[str, np.ndarray]:
        """
        计算频带功率比

        参数:
        ----------
        signal_data : np.ndarray
            输入信号
        band_edges : List[Tuple[float, float]], 可选
            频带边界列表，默认为EEG标准频带

        返回:
        ----------
        Dict[str, np.ndarray]: 频带功率比字典
        """
        processed_signal = self._preprocess_signal(signal_data)
        n_channels, n_samples = processed_signal.shape

        # 默认频带（EEG频带）
        if band_edges is None:
            band_edges = [
                (0.5, 4),  # Delta
                (4, 8),  # Theta
                (8, 13),  # Alpha
                (13, 30),  # Beta
                (30, 45)  # Gamma
            ]

        feature_dict = {}

        for i in range(n_channels):
            channel_signal = processed_signal[i]

            # 计算功率谱
            freqs, psd = signal.welch(channel_signal, fs=self.sampling_rate)

            total_power = np.trapz(psd, freqs)

            for band_idx, (low_freq, high_freq) in enumerate(band_edges):
                # 找到频带内的频率索引
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)

                if np.any(band_mask):
                    band_power = np.trapz(psd[band_mask], freqs[band_mask])

                    # 绝对功率
                    abs_power_feat = f'band_{band_idx + 1}_abs_power'
                    if abs_power_feat not in feature_dict:
                        feature_dict[abs_power_feat] = np.zeros(n_channels)
                    feature_dict[abs_power_feat][i] = band_power

                    # 相对功率（占总功率的比例）
                    if total_power > 1e-10:
                        rel_power = band_power / total_power
                        rel_power_feat = f'band_{band_idx + 1}_rel_power'
                        if rel_power_feat not in feature_dict:
                            feature_dict[rel_power_feat] = np.zeros(n_channels)
                        feature_dict[rel_power_feat][i] = rel_power

        return feature_dict


class TimeFrequencyFeatures(FeatureExtractor):
    """时频域特征提取类"""

    def compute_wavelet_features(self,
                                 signal_data: np.ndarray,
                                 wavelet: str = 'db4',
                                 max_level: int = 5,
                                 features: List[str] = None) -> Dict[str, np.ndarray]:
        """
        计算小波变换特征

        参数:
        ----------
        signal_data : np.ndarray
            输入信号
        wavelet : str, 默认='db4'
            小波基函数
        max_level : int, 默认=5
            最大分解层数
        features : List[str], 可选
            要计算的特征列表

        返回:
        ----------
        Dict[str, np.ndarray]: 小波特征字典
        """
        processed_signal = self._preprocess_signal(signal_data)
        n_channels, n_samples = processed_signal.shape

        if features is None:
            features = ['energy', 'entropy', 'std', 'mean']

        feature_dict = {}

        for i in range(n_channels):
            channel_signal = processed_signal[i]

            # 小波分解
            coeffs = pywt.wavedec(channel_signal, wavelet, level=max_level)

            # 计算近似系数和细节系数的特征
            for level, coeff in enumerate(coeffs):
                if len(coeff) == 0:
                    continue

                level_prefix = f'wavelet_level_{level}'

                # 能量
                if 'energy' in features:
                    energy = np.sum(coeff ** 2)
                    energy_key = f'{level_prefix}_energy'
                    if energy_key not in feature_dict:
                        feature_dict[energy_key] = np.zeros(n_channels)
                    feature_dict[energy_key][i] = energy

                # 熵
                if 'entropy' in features:
                    # 归一化系数
                    coeff_normalized = coeff / (np.sum(np.abs(coeff)) + 1e-10)
                    coeff_entropy = -np.sum(coeff_normalized * np.log2(coeff_normalized + 1e-10))
                    entropy_key = f'{level_prefix}_entropy'
                    if entropy_key not in feature_dict:
                        feature_dict[entropy_key] = np.zeros(n_channels)
                    feature_dict[entropy_key][i] = coeff_entropy

                # 标准差
                if 'std' in features:
                    std_value = np.std(coeff)
                    std_key = f'{level_prefix}_std'
                    if std_key not in feature_dict:
                        feature_dict[std_key] = np.zeros(n_channels)
                    feature_dict[std_key][i] = std_value

                # 均值
                if 'mean' in features:
                    mean_value = np.mean(coeff)
                    mean_key = f'{level_prefix}_mean'
                    if mean_key not in feature_dict:
                        feature_dict[mean_key] = np.zeros(n_channels)
                    feature_dict[mean_key][i] = mean_value

                # 能量比（相对于总能量）
                if 'energy_ratio' in features:
                    total_energy = np.sum(channel_signal ** 2)
                    if total_energy > 1e-10:
                        coeff_energy = np.sum(coeff ** 2)
                        energy_ratio = coeff_energy / total_energy
                        energy_ratio_key = f'{level_prefix}_energy_ratio'
                        if energy_ratio_key not in feature_dict:
                            feature_dict[energy_ratio_key] = np.zeros(n_channels)
                        feature_dict[energy_ratio_key][i] = energy_ratio

        return feature_dict

    def compute_stft_features(self,
                              signal_data: np.ndarray,
                              nperseg: int = 256,
                              noverlap: int = None,
                              window: str = 'hann',
                              features: List[str] = None) -> Dict[str, np.ndarray]:
        """
        计算短时傅里叶变换特征

        参数:
        ----------
        signal_data : np.ndarray
            输入信号
        nperseg : int, 默认=256
            每个段的长度
        noverlap : int, 可选
            段之间的重叠点数
        window : str, 默认='hann'
            窗函数类型
        features : List[str], 可选
            要计算的特征列表

        返回:
        ----------
        Dict[str, np.ndarray]: STFT特征字典
        """
        processed_signal = self._preprocess_signal(signal_data)
        n_channels, n_samples = processed_signal.shape

        if noverlap is None:
            noverlap = nperseg // 2

        if features is None:
            features = ['mean_spectrum', 'std_spectrum', 'spectral_flux']

        feature_dict = {}

        for i in range(n_channels):
            channel_signal = processed_signal[i]

            # 计算STFT
            f, t, Zxx = signal.stft(
                channel_signal,
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window=window
            )

            # 幅度谱
            magnitude = np.abs(Zxx)

            # 时频谱的统计特征
            if 'mean_spectrum' in features:
                mean_spec = np.mean(magnitude, axis=1)
                mean_key = 'stft_mean_spectrum'
                if mean_key not in feature_dict:
                    # 存储整个频谱，而不仅仅是标量
                    feature_dict[mean_key] = np.zeros((n_channels, len(f)))
                feature_dict[mean_key][i, :] = mean_spec

            if 'std_spectrum' in features:
                std_spec = np.std(magnitude, axis=1)
                std_key = 'stft_std_spectrum'
                if std_key not in feature_dict:
                    feature_dict[std_key] = np.zeros((n_channels, len(f)))
                feature_dict[std_key][i, :] = std_spec

            if 'spectral_flux' in features:
                # 谱通量：相邻时间帧之间的频谱变化
                spectral_flux = np.mean(np.diff(magnitude, axis=1) ** 2, axis=0)
                flux_key = 'stft_spectral_flux'
                if flux_key not in feature_dict:
                    feature_dict[flux_key] = np.zeros((n_channels, len(t) - 1))
                feature_dict[flux_key][i, :] = spectral_flux

            # 时频谱的能量
            if 'total_energy' in features:
                total_energy = np.sum(magnitude ** 2)
                energy_key = 'stft_total_energy'
                if energy_key not in feature_dict:
                    feature_dict[energy_key] = np.zeros(n_channels)
                feature_dict[energy_key][i] = total_energy

        return feature_dict

    def compute_hilbert_huang_features(self,
                                       signal_data: np.ndarray,
                                       n_imfs: int = 5) -> Dict[str, np.ndarray]:
        """
        计算希尔伯特-黄变换特征（基于经验模态分解）

        注意：这里使用简单的Hilbert变换作为替代，完整的EMD实现较复杂

        参数:
        ----------
        signal_data : np.ndarray
            输入信号
        n_imfs : int, 默认=5
            期望的IMF数量

        返回:
        ----------
        Dict[str, np.ndarray]: HHT特征字典
        """
        processed_signal = self._preprocess_signal(signal_data)
        n_channels, n_samples = processed_signal.shape

        feature_dict = {}

        for i in range(n_channels):
            channel_signal = processed_signal[i]

            # 使用Hilbert变换计算瞬时频率和幅度
            analytic_signal = hilbert(channel_signal)
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * self.sampling_rate

            # 确保长度匹配
            if len(instantaneous_frequency) < len(amplitude_envelope):
                instantaneous_frequency = np.append(instantaneous_frequency, instantaneous_frequency[-1])

            # 计算统计特征
            feature_dict.setdefault('hilbert_mean_amplitude', np.zeros(n_channels))
            feature_dict['hilbert_mean_amplitude'][i] = np.mean(amplitude_envelope)

            feature_dict.setdefault('hilbert_std_amplitude', np.zeros(n_channels))
            feature_dict['hilbert_std_amplitude'][i] = np.std(amplitude_envelope)

            feature_dict.setdefault('hilbert_mean_frequency', np.zeros(n_channels))
            feature_dict['hilbert_mean_frequency'][i] = np.mean(instantaneous_frequency)

            feature_dict.setdefault('hilbert_std_frequency', np.zeros(n_channels))
            feature_dict['hilbert_std_frequency'][i] = np.std(instantaneous_frequency)

            # 幅度调制深度
            if np.mean(amplitude_envelope) > 1e-10:
                modulation_depth = np.std(amplitude_envelope) / np.mean(amplitude_envelope)
                feature_dict.setdefault('hilbert_modulation_depth', np.zeros(n_channels))
                feature_dict['hilbert_modulation_depth'][i] = modulation_depth

        return feature_dict


class NonlinearFeatures(FeatureExtractor):
    """非线性特征提取类"""

    def compute_entropy_features(self,
                                 signal_data: np.ndarray,
                                 entropy_types: List[str] = None,
                                 m: int = 2,
                                 r: float = 0.2,
                                 delay: int = 1) -> Dict[str, np.ndarray]:
        """
        计算各种熵特征

        参数:
        ----------
        signal_data : np.ndarray
            输入信号
        entropy_types : List[str], 可选
            要计算的熵类型列表
        m : int, 默认=2
            嵌入维数
        r : float, 默认=0.2
            相似度阈值（通常为标准差的倍数）
        delay : int, 默认=1
            时间延迟

        返回:
        ----------
        Dict[str, np.ndarray]: 熵特征字典
        """
        processed_signal = self._preprocess_signal(signal_data)
        n_channels, n_samples = processed_signal.shape

        if entropy_types is None:
            entropy_types = ['sample_entropy', 'approximate_entropy', 'permutation_entropy']

        feature_dict = {}

        for i in range(n_channels):
            channel_signal = processed_signal[i]

            # 确保信号长度足够
            min_length = 10 * m * delay
            if len(channel_signal) < min_length:
                warnings.warn(f"信号长度({len(channel_signal)})可能不足以保证熵计算的可靠性",
                              SignalQualityWarning)
                # 使用默认值
                for entropy_type in entropy_types:
                    feat_name = f'{entropy_type}'
                    if feat_name not in feature_dict:
                        feature_dict[feat_name] = np.zeros(n_channels)
                    feature_dict[feat_name][i] = np.nan
                continue

            # 样本熵
            if 'sample_entropy' in entropy_types:
                sample_entropy_value = self._compute_sample_entropy(channel_signal, m, r)
                feature_dict.setdefault('sample_entropy', np.zeros(n_channels))
                feature_dict['sample_entropy'][i] = sample_entropy_value

            # 近似熵
            if 'approximate_entropy' in entropy_types:
                approx_entropy_value = self._compute_approximate_entropy(channel_signal, m, r)
                feature_dict.setdefault('approximate_entropy', np.zeros(n_channels))
                feature_dict['approximate_entropy'][i] = approx_entropy_value

            # 排列熵
            if 'permutation_entropy' in entropy_types:
                perm_entropy_value = self._compute_permutation_entropy(channel_signal, m, delay)
                feature_dict.setdefault('permutation_entropy', np.zeros(n_channels))
                feature_dict['permutation_entropy'][i] = perm_entropy_value

            # 模糊熵
            if 'fuzzy_entropy' in entropy_types:
                fuzzy_entropy_value = self._compute_fuzzy_entropy(channel_signal, m, r)
                feature_dict.setdefault('fuzzy_entropy', np.zeros(n_channels))
                feature_dict['fuzzy_entropy'][i] = fuzzy_entropy_value

            # 多尺度熵（简化版本）
            if 'multiscale_entropy' in entropy_types:
                mse_values = self._compute_multiscale_entropy(channel_signal, m, r, max_scale=5)
                for scale, mse_val in enumerate(mse_values, 1):
                    feat_name = f'multiscale_entropy_scale_{scale}'
                    if feat_name not in feature_dict:
                        feature_dict[feat_name] = np.zeros(n_channels)
                    feature_dict[feat_name][i] = mse_val

        return feature_dict

    def _compute_sample_entropy(self, signal_data: np.ndarray, m: int, r: float) -> float:
        """计算样本熵"""
        N = len(signal_data)

        # 分割序列
        def _get_vectors(m):
            return np.array([signal_data[i:i + m] for i in range(N - m + 1)])

        # 计算距离
        def _phi(m):
            vectors = _get_vectors(m)
            C = np.zeros(len(vectors))

            for i in range(len(vectors)):
                # 排除自比较
                distances = np.max(np.abs(vectors[i + 1:] - vectors[i]), axis=1)
                C[i] = np.sum(distances <= r * np.std(signal_data)) / (N - m)

            return np.sum(C) / (N - m + 1)

        if N <= m:
            return 0

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)

        if phi_m1 == 0 or phi_m == 0:
            return 0

        return -np.log(phi_m1 / phi_m)

    def _compute_approximate_entropy(self, signal_data: np.ndarray, m: int, r: float) -> float:
        """计算近似熵"""
        N = len(signal_data)

        def _phi(m):
            vectors = np.array([signal_data[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(len(vectors))

            for i in range(len(vectors)):
                distances = np.max(np.abs(vectors - vectors[i]), axis=1)
                C[i] = np.sum(distances <= r * np.std(signal_data)) / (N - m + 1)

            return np.mean(np.log(C + 1e-10))

        if N <= m:
            return 0

        return _phi(m) - _phi(m + 1)

    def _compute_permutation_entropy(self, signal_data: np.ndarray, m: int, delay: int) -> float:
        """计算排列熵"""
        N = len(signal_data)

        # 生成排列模式
        permutations = []
        for i in range(N - (m - 1) * delay):
            segment = signal_data[i:i + m * delay:delay]
            permutations.append(tuple(np.argsort(segment)))

        # 计算每种排列的概率
        unique_perms, counts = np.unique(permutations, return_counts=True, axis=0)
        probs = counts / len(permutations)

        # 计算排列熵
        perm_entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # 归一化（除以最大可能熵）
        max_entropy = np.log2(np.math.factorial(m))
        if max_entropy > 0:
            perm_entropy = perm_entropy / max_entropy

        return perm_entropy

    def _compute_fuzzy_entropy(self, signal_data: np.ndarray, m: int, r: float) -> float:
        """计算模糊熵（简化实现）"""
        N = len(signal_data)
        std = np.std(signal_data)

        def _get_vectors(m):
            vectors = np.array([signal_data[i:i + m] for i in range(N - m + 1)])
            # 去除基线
            vectors = vectors - np.mean(vectors, axis=1, keepdims=True)
            return vectors

        vectors_m = _get_vectors(m)
        vectors_m1 = _get_vectors(m + 1)

        # 模糊隶属度函数
        def _fuzzy_membership(d, r):
            return np.exp(-(d ** 2) / r)

        # 计算相似度
        phi_m = 0
        for i in range(len(vectors_m)):
            distances = np.max(np.abs(vectors_m - vectors_m[i]), axis=1)
            similarity = _fuzzy_membership(distances, r * std)
            # 排除自相似
            similarity = np.delete(similarity, i)
            phi_m += np.mean(similarity)
        phi_m /= len(vectors_m)

        phi_m1 = 0
        for i in range(len(vectors_m1)):
            distances = np.max(np.abs(vectors_m1 - vectors_m1[i]), axis=1)
            similarity = _fuzzy_membership(distances, r * std)
            similarity = np.delete(similarity, i)
            phi_m1 += np.mean(similarity)
        phi_m1 /= len(vectors_m1)

        if phi_m1 == 0 or phi_m == 0:
            return 0

        return np.log(phi_m) - np.log(phi_m1)

    def _compute_multiscale_entropy(self,
                                    signal_data: np.ndarray,
                                    m: int,
                                    r: float,
                                    max_scale: int = 5) -> np.ndarray:
        """计算多尺度熵（简化版本）"""
        mse_values = []

        for scale in range(1, max_scale + 1):
            # 粗粒化
            coarse_grained = []
            for i in range(0, len(signal_data) - scale + 1, scale):
                coarse_grained.append(np.mean(signal_data[i:i + scale]))

            if len(coarse_grained) > 10 * m:  # 确保足够长度
                se = self._compute_sample_entropy(np.array(coarse_grained), m, r)
                mse_values.append(se)
            else:
                mse_values.append(np.nan)

        return np.array(mse_values)

    def compute_fractal_dimension(self,
                                  signal_data: np.ndarray,
                                  method: str = 'higuchi',
                                  kmax: int = 10) -> np.ndarray:
        """
        计算分形维数

        参数:
        ----------
        signal_data : np.ndarray
            输入信号
        method : str, 默认='higuchi'
            计算方法：'higuchi'或'box'
        kmax : int, 默认=10
            Higuchi方法中的最大k值

        返回:
        ----------
        np.ndarray: 分形维数值
        """
        processed_signal = self._preprocess_signal(signal_data)
        n_channels, n_samples = processed_signal.shape

        fd_values = np.zeros(n_channels)

        for i in range(n_channels):
            channel_signal = processed_signal[i]

            if method.lower() == 'higuchi':
                fd_values[i] = self._compute_higuchi_fd(channel_signal, kmax)
            elif method.lower() == 'box':
                fd_values[i] = self._compute_box_counting_fd(channel_signal)
            else:
                raise ValueError(f"未知的分形维数计算方法: {method}")

        return fd_values

    def _compute_higuchi_fd(self, signal_data: np.ndarray, kmax: int) -> float:
        """计算Higuchi分形维数"""
        N = len(signal_data)
        L = []
        x = []

        for k in range(1, kmax + 1):
            Lk = 0
            for m in range(k):
                # 创建子序列
                indices = np.arange(m, N, k)
                if len(indices) > 1:
                    Lmk = np.sum(np.abs(np.diff(signal_data[indices])))
                    Lmk = Lmk * (N - 1) / (len(indices) - 1) / k
                    Lk += Lmk

            L.append(np.log(Lk / k))
            x.append(np.log(1.0 / k))

        # 线性拟合
        if len(x) > 1:
            slope, _ = np.polyfit(x, L, 1)
            return -slope
        else:
            return 1.0

    def _compute_box_counting_fd(self, signal_data: np.ndarray) -> float:
        """计算盒计数分形维数"""
        N = len(signal_data)

        # 归一化信号到[0,1]区间
        signal_min, signal_max = np.min(signal_data), np.max(signal_data)
        if signal_max - signal_min > 1e-10:
            signal_norm = (signal_data - signal_min) / (signal_max - signal_min)
        else:
            signal_norm = signal_data

        # 不同尺度下的盒子计数
        scales = np.logspace(0, np.log10(N / 2), 20, dtype=int)
        scales = scales[scales > 1]

        counts = []
        for scale in scales:
            # 将信号分成scale个区间
            x_bins = np.linspace(0, N, scale + 1)
            y_bins = np.linspace(0, 1, scale + 1)

            # 计数覆盖信号的盒子数
            box_count = 0
            for i in range(scale):
                x_start, x_end = x_bins[i], x_bins[i + 1]
                indices = np.where((np.arange(N) >= x_start) & (np.arange(N) < x_end))[0]

                if len(indices) > 0:
                    y_min = np.min(signal_norm[indices])
                    y_max = np.max(signal_norm[indices])

                    # 找出覆盖y范围的盒子索引
                    y_idx_min = int(y_min * scale)
                    y_idx_max = int(y_max * scale)
                    box_count += (y_idx_max - y_idx_min + 1)

            counts.append(box_count)

        # 线性拟合log(1/scale) vs log(counts)
        if len(scales) > 1:
            x = np.log(1.0 / scales)
            y = np.log(counts)
            slope, _ = np.polyfit(x, y, 1)
            return slope
        else:
            return 1.0

    def compute_detrended_fluctuation_analysis(self,
                                               signal_data: np.ndarray,
                                               scale_ranges: List[Tuple[int, int]] = None) -> Dict[str, np.ndarray]:
        """
        计算去趋势波动分析(DFA)

        参数:
        ----------
        signal_data : np.ndarray
            输入信号
        scale_ranges : List[Tuple[int, int]], 可选
            尺度范围列表

        返回:
        ----------
        Dict[str, np.ndarray]: DFA特征字典
        """
        processed_signal = self._preprocess_signal(signal_data)
        n_channels, n_samples = processed_signal.shape

        if scale_ranges is None:
            scale_ranges = [(10, 50), (50, 200), (200, 1000)]  # 短程、中程、长程

        feature_dict = {}

        for i in range(n_channels):
            channel_signal = processed_signal[i]
            N = len(channel_signal)

            # 累积和
            y = np.cumsum(channel_signal - np.mean(channel_signal))

            # 不同尺度下的波动
            scales = np.unique(np.logspace(np.log10(4), np.log10(N / 4), 20, dtype=int))
            scales = scales[scales < N / 4]  # 确保有足够的段

            fluctuations = []

            for scale in scales:
                # 分段
                n_segments = int(N / scale)
                if n_segments < 2:
                    continue

                # 每段进行去趋势
                F = np.zeros(n_segments)
                for j in range(n_segments):
                    segment = y[j * scale:(j + 1) * scale]
                    x = np.arange(len(segment))

                    # 线性拟合去趋势
                    coeff = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeff, x)

                    F[j] = np.sqrt(np.mean((segment - trend) ** 2))

                fluctuations.append(np.mean(F))

            if len(scales) > 1 and len(fluctuations) > 1:
                # 对每个尺度范围进行拟合
                scales_log = np.log10(scales)
                fluct_log = np.log10(fluctuations)

                for range_idx, (min_scale, max_scale) in enumerate(scale_ranges):
                    mask = (scales >= min_scale) & (scales <= max_scale)

                    if np.sum(mask) >= 3:  # 至少3个点才能拟合
                        x_range = scales_log[mask]
                        y_range = fluct_log[mask]

                        slope, intercept = np.polyfit(x_range, y_range, 1)

                        feat_name = f'dfa_alpha_range_{range_idx + 1}'
                        feature_dict.setdefault(feat_name, np.zeros(n_channels))
                        feature_dict[feat_name][i] = slope

                        # 拟合质量（R^2）
                        y_pred = slope * x_range + intercept
                        ss_res = np.sum((y_range - y_pred) ** 2)
                        ss_tot = np.sum((y_range - np.mean(y_range)) ** 2)
                        r_squared = 1 - (ss_res / (ss_tot + 1e-10))

                        rsq_name = f'dfa_r2_range_{range_idx + 1}'
                        feature_dict.setdefault(rsq_name, np.zeros(n_channels))
                        feature_dict[rsq_name][i] = r_squared

        return feature_dict

    def compute_recurrence_quantification_analysis(self,
                                                   signal_data: np.ndarray,
                                                   m: int = 1,
                                                   delay: int = 1,
                                                   threshold: float = 0.1) -> Dict[str, np.ndarray]:
        """
        计算递归定量分析(RQA)特征

        参数:
        ----------
        signal_data : np.ndarray
            输入信号
        m : int, 默认=1
            嵌入维数
        delay : int, 默认=1
            时间延迟
        threshold : float, 默认=0.1
            递归阈值

        返回:
        ----------
        Dict[str, np.ndarray]: RQA特征字典
        """
        processed_signal = self._preprocess_signal(signal_data)
        n_channels, n_samples = processed_signal.shape

        feature_dict = {}

        for i in range(n_channels):
            channel_signal = processed_signal[i]
            N = len(channel_signal)

            # 相空间重构（简化版本，使用时间延迟嵌入）
            if m > 1:
                embedded = np.array([channel_signal[j:j + (m - 1) * delay + 1:delay]
                                     for j in range(N - (m - 1) * delay)])
            else:
                embedded = channel_signal.reshape(-1, 1)

            # 计算距离矩阵
            if embedded.ndim == 2:
                distances = squareform(pdist(embedded, metric='euclidean'))
            else:
                distances = np.abs(embedded[:, None] - embedded)

            # 递归矩阵
            recurrence_matrix = distances <= (threshold * np.std(channel_signal))

            # 对角线设为False（排除自递归）
            np.fill_diagonal(recurrence_matrix, False)

            # RQA特征
            N_points = recurrence_matrix.size - len(recurrence_matrix)  # 排除对角线

            if N_points > 0:
                # 递归率
                recurrence_rate = np.sum(recurrence_matrix) / N_points
                feature_dict.setdefault('rqa_recurrence_rate', np.zeros(n_channels))
                feature_dict['rqa_recurrence_rate'][i] = recurrence_rate

                # 确定性
                diagonal_lines = []
                for d in range(-len(recurrence_matrix) + 1, len(recurrence_matrix)):
                    diagonal = np.diag(recurrence_matrix, d)
                    if len(diagonal) > 1:
                        # 寻找对角线上的线段
                        in_line = False
                        line_length = 0
                        for val in diagonal:
                            if val and not in_line:
                                in_line = True
                                line_length = 1
                            elif val and in_line:
                                line_length += 1
                            elif not val and in_line:
                                if line_length >= 2:  # 最小对角线长度
                                    diagonal_lines.append(line_length)
                                in_line = False
                                line_length = 0

                        if in_line and line_length >= 2:
                            diagonal_lines.append(line_length)

                if diagonal_lines:
                    diagonal_lines = np.array(diagonal_lines)
                    determinism = np.sum(diagonal_lines) / np.sum(recurrence_matrix)
                    feature_dict.setdefault('rqa_determinism', np.zeros(n_channels))
                    feature_dict['rqa_determinism'][i] = determinism

                    # 平均对角线长度
                    mean_diag_length = np.mean(diagonal_lines)
                    feature_dict.setdefault('rqa_mean_diag_length', np.zeros(n_channels))
                    feature_dict['rqa_mean_diag_length'][i] = mean_diag_length

                    # 最大对角线长度
                    max_diag_length = np.max(diagonal_lines)
                    feature_dict.setdefault('rqa_max_diag_length', np.zeros(n_channels))
                    feature_dict['rqa_max_diag_length'][i] = max_diag_length

                # 层流性（垂直线段）
                vertical_lines = []
                for col in range(len(recurrence_matrix)):
                    column = recurrence_matrix[:, col]
                    in_line = False
                    line_length = 0
                    for val in column:
                        if val and not in_line:
                            in_line = True
                            line_length = 1
                        elif val and in_line:
                            line_length += 1
                        elif not val and in_line:
                            if line_length >= 2:
                                vertical_lines.append(line_length)
                            in_line = False
                            line_length = 0

                    if in_line and line_length >= 2:
                        vertical_lines.append(line_length)

                if vertical_lines:
                    vertical_lines = np.array(vertical_lines)
                    laminarity = np.sum(vertical_lines) / np.sum(recurrence_matrix)
                    feature_dict.setdefault('rqa_laminarity', np.zeros(n_channels))
                    feature_dict['rqa_laminarity'][i] = laminarity

        return feature_dict


class CommonFeatureExtractor(TimeDomainFeatures,
                             FrequencyDomainFeatures,
                             TimeFrequencyFeatures,
                             NonlinearFeatures):
    """
    通用特征提取器综合类

    集成了所有通用特征提取方法，提供统一的接口
    """

    def __init__(self, **kwargs):
        """初始化综合特征提取器"""
        super().__init__(**kwargs)

    def extract_all_features(self,
                             signal_data: np.ndarray,
                             feature_groups: List[str] = None,
                             config: Dict = None) -> Dict[str, np.ndarray]:
        """
        提取所有通用特征

        参数:
        ----------
        signal_data : np.ndarray
            输入信号
        feature_groups : List[str], 可选
            要提取的特征组列表，可选值：
            ['time', 'frequency', 'time_frequency', 'nonlinear']
        config : Dict, 可选
            各特征组的配置参数

        返回:
        ----------
        Dict[str, np.ndarray]: 所有特征的字典
        """
        if feature_groups is None:
            feature_groups = ['time', 'frequency', 'time_frequency', 'nonlinear']

        if config is None:
            config = {}

        all_features = {}

        # 时域特征
        if 'time' in feature_groups:
            time_config = config.get('time', {})

            # 统计特征
            stat_features = self.compute_statistical_features(
                signal_data,
                features=time_config.get('stat_features')
            )
            all_features.update(stat_features)

            # Hjorth参数
            hjorth_features = self.compute_hjorth_parameters(signal_data)
            all_features.update(hjorth_features)

            # 其他时域特征
            all_features['rms'] = self.compute_rms(signal_data)
            all_features['zcr'] = self.compute_zcr(
                signal_data,
                threshold=time_config.get('zcr_threshold', 0.0)
            )
            all_features['waveform_length'] = self.compute_waveform_length(signal_data)
            all_features['willison_amplitude'] = self.compute_willison_amplitude(
                signal_data,
                threshold=time_config.get('willison_threshold', 0.01)
            )

        # 频域特征
        if 'frequency' in feature_groups:
            freq_config = config.get('frequency', {})

            # 频谱特征
            spectral_features = self.compute_spectral_features(
                signal_data,
                nperseg=freq_config.get('nperseg', 256),
                noverlap=freq_config.get('noverlap'),
                window=freq_config.get('window', 'hann')
            )
            all_features.update(spectral_features)

            # 谱熵
            all_features['spectral_entropy'] = self.compute_spectral_entropy(
                signal_data,
                normalize=freq_config.get('normalize_entropy', True)
            )

            # 频谱边缘频率
            sef_features = self.compute_spectral_edge_frequency(
                signal_data,
                percentiles=freq_config.get('sef_percentiles', [50, 75, 90, 95])
            )
            all_features.update(sef_features)

            # 频带功率比
            band_features = self.compute_band_power_ratio(
                signal_data,
                band_edges=freq_config.get('band_edges')
            )
            all_features.update(band_features)

        # 时频域特征
        if 'time_frequency' in feature_groups:
            tf_config = config.get('time_frequency', {})

            # 小波特征
            wavelet_features = self.compute_wavelet_features(
                signal_data,
                wavelet=tf_config.get('wavelet', 'db4'),
                max_level=tf_config.get('max_level', 5),
                features=tf_config.get('wavelet_features')
            )
            all_features.update(wavelet_features)

            # STFT特征
            stft_features = self.compute_stft_features(
                signal_data,
                nperseg=tf_config.get('stft_nperseg', 256),
                noverlap=tf_config.get('stft_noverlap'),
                window=tf_config.get('stft_window', 'hann'),
                features=tf_config.get('stft_features')
            )
            all_features.update(stft_features)

            # Hilbert-Huang特征
            hht_features = self.compute_hilbert_huang_features(
                signal_data,
                n_imfs=tf_config.get('n_imfs', 5)
            )
            all_features.update(hht_features)

        # 非线性特征
        if 'nonlinear' in feature_groups:
            nonlinear_config = config.get('nonlinear', {})

            # 熵特征
            entropy_features = self.compute_entropy_features(
                signal_data,
                entropy_types=nonlinear_config.get('entropy_types'),
                m=nonlinear_config.get('m', 2),
                r=nonlinear_config.get('r', 0.2),
                delay=nonlinear_config.get('delay', 1)
            )
            all_features.update(entropy_features)

            # 分形维数
            all_features['fractal_dimension'] = self.compute_fractal_dimension(
                signal_data,
                method=nonlinear_config.get('fd_method', 'higuchi'),
                kmax=nonlinear_config.get('kmax', 10)
            )

            # DFA特征
            dfa_features = self.compute_detrended_fluctuation_analysis(
                signal_data,
                scale_ranges=nonlinear_config.get('dfa_scale_ranges')
            )
            all_features.update(dfa_features)

            # RQA特征（可选，计算量较大）
            if nonlinear_config.get('compute_rqa', False):
                rqa_features = self.compute_recurrence_quantification_analysis(
                    signal_data,
                    m=nonlinear_config.get('rqa_m', 1),
                    delay=nonlinear_config.get('rqa_delay', 1),
                    threshold=nonlinear_config.get('rqa_threshold', 0.1)
                )
                all_features.update(rqa_features)

        return all_features

    def extract_features_by_config(self,
                                   signal_data: np.ndarray,
                                   config: Dict) -> Dict[str, np.ndarray]:
        """
        根据配置文件提取特征

        参数:
        ----------
        signal_data : np.ndarray
            输入信号
        config : Dict
            特征提取配置字典

        返回:
        ----------
        Dict[str, np.ndarray]: 提取的特征字典
        """
        return self.extract_all_features(signal_data, config=config)

    def get_feature_names(self, feature_dict: Dict[str, np.ndarray]) -> List[str]:
        """
        获取特征名称列表

        参数:
        ----------
        feature_dict : Dict[str, np.ndarray]
            特征字典

        返回:
        ----------
        List[str]: 特征名称列表
        """
        return list(feature_dict.keys())

    def flatten_features(self, feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        将特征字典展平为特征向量

        参数:
        ----------
        feature_dict : Dict[str, np.ndarray]
            特征字典

        返回:
        ----------
        np.ndarray: 展平的特征向量
        """
        feature_list = []

        for feat_name, feat_values in feature_dict.items():
            # 处理标量特征
            if feat_values.ndim == 1:
                feature_list.extend(feat_values.tolist())
            # 处理向量特征（如频谱）
            elif feat_values.ndim == 2:
                # 展平每个通道的向量特征
                for i in range(feat_values.shape[0]):
                    feature_list.extend(feat_values[i].tolist())

        return np.array(feature_list)


# 实用函数
def create_feature_extractor(sampling_rate: float = 250.0, **kwargs) -> CommonFeatureExtractor:
    """
    创建特征提取器实例的工厂函数

    参数:
    ----------
    sampling_rate : float, 默认=250.0
        采样率
    **kwargs :
        传递给CommonFeatureExtractor的额外参数

    返回:
    ----------
    CommonFeatureExtractor: 特征提取器实例
    """
    return CommonFeatureExtractor(sampling_rate=sampling_rate, **kwargs)


def batch_extract_features(signal_batch: np.ndarray,
                           feature_extractor: CommonFeatureExtractor,
                           config: Dict = None) -> List[Dict[str, np.ndarray]]:
    """
    批量提取特征

    参数:
    ----------
    signal_batch : np.ndarray
        信号批次，形状为(n_samples, n_channels, n_timesteps)
    feature_extractor : CommonFeatureExtractor
        特征提取器实例
    config : Dict, 可选
        特征提取配置

    返回:
    ----------
    List[Dict[str, np.ndarray]]: 每个样本的特征字典列表
    """
    if signal_batch.ndim == 2:
        # (n_timesteps, n_channels) -> 单个样本
        signal_batch = signal_batch[np.newaxis, ...]
    elif signal_batch.ndim == 3:
        # (n_samples, n_channels, n_timesteps)
        pass
    else:
        raise ValueError(f"输入信号维度不正确: {signal_batch.ndim}，应为2或3维")

    features_list = []

    for i in range(signal_batch.shape[0]):
        # 转置为(n_channels, n_timesteps)
        signal_sample = signal_batch[i].T

        # 提取特征
        features = feature_extractor.extract_all_features(signal_sample, config=config)
        features_list.append(features)

    return features_list


# 测试函数
def test_feature_extraction():
    """测试特征提取功能"""
    print("测试通用特征提取模块...")

    # 生成测试信号
    fs = 250.0  # 采样率
    t = np.arange(0, 5, 1 / fs)  # 5秒信号
    n_channels = 2

    # 创建测试信号：正弦波 + 噪声
    signals = []
    for i in range(n_channels):
        freq = 10 + i * 5  # 不同频率
        signal = np.sin(2 * np.pi * freq * t)
        signal += 0.1 * np.random.randn(len(t))  # 添加噪声
        signals.append(signal)

    signal_data = np.array(signals)  # 形状: (n_channels, n_samples)

    # 创建特征提取器
    extractor = create_feature_extractor(sampling_rate=fs, verbose=True)

    # 测试时域特征
    print("\n1. 测试时域特征...")
    time_features = extractor.compute_statistical_features(signal_data)
    print(f"提取的时域特征: {list(time_features.keys())[:5]}...")

    # 测试频域特征
    print("\n2. 测试频域特征...")
    freq_features = extractor.compute_spectral_features(signal_data)
    print(f"提取的频域特征: {list(freq_features.keys())[:5]}...")

    # 测试熵特征
    print("\n3. 测试非线性特征...")
    entropy_features = extractor.compute_entropy_features(signal_data)
    print(f"提取的熵特征: {list(entropy_features.keys())}")

    # 测试综合特征提取
    print("\n4. 测试综合特征提取...")
    all_features = extractor.extract_all_features(
        signal_data,
        feature_groups=['time', 'frequency'],
        config={
            'time': {'stat_features': ['mean', 'std', 'skewness']},
            'frequency': {'nperseg': 128}
        }
    )
    print(f"总共提取了 {len(all_features)} 个特征")

    # 测试特征展平
    print("\n5. 测试特征展平...")
    flat_features = extractor.flatten_features(all_features)
    print(f"展平后的特征向量维度: {flat_features.shape}")

    # 清除缓存
    extractor.clear_cache()

    print("\n测试完成!")
    return all_features


if __name__ == "__main__":
    # 运行测试
    test_results = test_feature_extraction()

    # 显示示例特征值
    print("\n示例特征值:")
    for feat_name, feat_value in list(test_results.items())[:10]:
        print(f"{feat_name}: {feat_value[:2] if feat_value.ndim == 1 else feat_value.shape}")
