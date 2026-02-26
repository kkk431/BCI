# -*- coding: utf-8 -*-
"""
ECG信号特征提取模块 (基于CommonFeatureExtractor)
提取时域、频域、时频和非线性特征
适用于已预处理的ECG信号

输入: 符合BCI标准的四层数据字典
输出: 包含所有特征的字典，可保存到processed层

依赖: numpy, scipy, pywt (PyWavelets), neurokit2
安装: pip install numpy scipy PyWavelets neurokit2
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# 导入通用特征提取器
try:
    from core.processing.feature_extraction.common_features import CommonFeatureExtractor
except ImportError:
    raise ImportError("需要common_features.py文件，请确保它在同一目录下")


# ========================= 依赖检查 =========================

def check_dependencies():
    """检查必需的依赖包"""
    missing = []

    try:
        import scipy
    except ImportError:
        missing.append('scipy')

    try:
        import pywt
    except ImportError:
        missing.append('PyWavelets')

    try:
        import neurokit2
    except ImportError:
        missing.append('neurokit2')

    if missing:
        raise ImportError(
            f"缺少必需的依赖包: {', '.join(missing)}\n"
            f"请运行: pip install {' '.join(missing)}"
        )


check_dependencies()

from scipy import signal
from scipy.interpolate import interp1d
import pywt
import neurokit2 as nk


# ========================= ECG特征提取器类 =========================

class ECGFeatureExtractor(CommonFeatureExtractor):
    """
    ECG特征提取器，继承自CommonFeatureExtractor。

    在父类通用特征的基础上，添加ECG特有的特征：
    - 心率相关特征（HR, HRV）
    - R波形态特征
    - HRV时域和频域特征
    - 非线性心率变异性特征

    使用方法:
        extractor = ECGFeatureExtractor(sampling_rate=500)
        features = extractor.extract_ecg_features(data_dict)
    """

    def __init__(self, sampling_rate: float):
        """
        初始化ECG特征提取器

        Parameters
        ----------
        sampling_rate : float
            ECG信号采样率 (Hz)
        """
        # 调用父类构造函数
        super().__init__(fs=sampling_rate)

        self.sampling_rate = sampling_rate
        self.r_peaks = None
        self.rr_intervals = None

    # ========================= 主要接口 =========================

    def extract_ecg_features(self, data_dict: Dict[str, Any],
                             ecg_channel: Optional[int] = 0) -> Dict[str, Any]:
        """
        从ECG信号提取完整特征集

        Parameters
        ----------
        data_dict : dict
            符合BCI标准的四层数据字典
        ecg_channel : int, optional
            使用的ECG通道索引，默认为0

        Returns
        -------
        features : dict
            包含所有特征的字典
        """

        # 1. 提取ECG信号
        if 'ECG' not in data_dict['signal']:
            raise ValueError("数据字典中未找到ECG信号模态")

        ecg_signal = data_dict['signal']['ECG']['data']

        # 如果是多通道，选择指定通道
        if ecg_signal.ndim == 2:
            ecg = ecg_signal[ecg_channel, :]
        else:
            ecg = ecg_signal

        print(f"ECG信号长度: {len(ecg)} 采样点")
        print(f"采样率: {self.sampling_rate} Hz")
        print(f"信号时长: {len(ecg) / self.sampling_rate:.2f} 秒")

        # 2. R波检测
        print("\n[1/5] 检测R波峰值...")
        self.r_peaks = self.detect_r_peaks(ecg)
        print(f"检测到 {len(self.r_peaks)} 个R波峰")

        # 计算RR间期
        if len(self.r_peaks) >= 2:
            self.rr_intervals = np.diff(self.r_peaks) / self.sampling_rate * 1000  # ms
        else:
            self.rr_intervals = np.array([])

        # 3. 提取各类特征
        features = {}

        # 使用父类方法提取通用特征
        print("[2/5] 提取通用信号特征...")
        features['common'] = super().extract_all_features(ecg)

        # ECG特有特征
        print("[3/5] 提取ECG形态学特征...")
        features['morphological'] = self.extract_morphological_features(ecg)

        print("[4/5] 提取HRV特征...")
        features['hrv_time'] = self.extract_hrv_time_features()
        features['hrv_frequency'] = self.extract_hrv_frequency_features()

        print("[5/5] 提取HRV非线性特征...")
        features['hrv_nonlinear'] = self.extract_hrv_nonlinear_features()

        print("特征提取完成!")

        return features

    # ========================= R波检测 =========================

    def detect_r_peaks(self, ecg: np.ndarray) -> np.ndarray:
        """
        检测ECG信号中的R波峰值

        Parameters
        ----------
        ecg : np.ndarray
            ECG信号

        Returns
        -------
        r_peaks : np.ndarray
            R波峰值的索引数组
        """
        try:
            _, info = nk.ecg_peaks(ecg, sampling_rate=self.sampling_rate)
            r_peaks = info['ECG_R_Peaks']
        except Exception as e:
            print(f"警告: neurokit2检测失败，使用备用方法: {str(e)}")
            r_peaks = self._simple_r_peak_detection(ecg)

        return np.array(r_peaks)

    def _simple_r_peak_detection(self, ecg: np.ndarray) -> np.ndarray:
        """简单的R波峰值检测（备用方法）"""
        # 带通滤波 (5-15 Hz)
        sos = signal.butter(4, [5, 15], btype='bandpass',
                            fs=self.sampling_rate, output='sos')
        ecg_filtered = signal.sosfilt(sos, ecg)

        # 找峰值
        distance = int(0.6 * self.sampling_rate)  # 最小间隔600ms
        peaks, _ = signal.find_peaks(ecg_filtered, distance=distance,
                                     height=np.std(ecg_filtered))

        return peaks

    # ========================= ECG形态学特征 =========================

    def extract_morphological_features(self, ecg: np.ndarray) -> Dict[str, float]:
        """
        提取ECG时域形态学特征

        包括:
        - 心率和RR间期统计
        - P波、QRS波、T波振幅（如果可检测）
        - PR、QT、QRS间期

        Parameters
        ----------
        ecg : np.ndarray
            ECG信号

        Returns
        -------
        features : dict
            形态学特征字典
        """
        features = {}

        if len(self.rr_intervals) == 0:
            print("警告: RR间期为空，跳过形态学特征")
            return features

        # 基本心率特征
        features['mean_rr_interval'] = float(np.mean(self.rr_intervals))
        features['std_rr_interval'] = float(np.std(self.rr_intervals))
        features['min_rr_interval'] = float(np.min(self.rr_intervals))
        features['max_rr_interval'] = float(np.max(self.rr_intervals))

        # 心率
        features['mean_hr'] = float(60000 / np.mean(self.rr_intervals))
        features['min_hr'] = float(60000 / np.max(self.rr_intervals))
        features['max_hr'] = float(60000 / np.min(self.rr_intervals))

        # 尝试波形分析
        try:
            _, waves = nk.ecg_delineate(ecg, self.r_peaks,
                                        sampling_rate=self.sampling_rate,
                                        method='peak')  # 指定方法避免复杂计算

            # P波特征 - 更健壮的处理
            if 'ECG_P_Peaks' in waves and waves['ECG_P_Peaks'] is not None:
                p_peaks = waves['ECG_P_Peaks']
                # 确保是数组并过滤NaN
                p_peaks = np.array(p_peaks, dtype=float)
                valid_p_peaks = p_peaks[~np.isnan(p_peaks)]

                if len(valid_p_peaks) > 0:
                    # 转换为整数索引前确保是标量值
                    valid_p_peaks = valid_p_peaks.astype(int)
                    # 确保索引在有效范围内
                    valid_p_peaks = valid_p_peaks[(valid_p_peaks >= 0) & (valid_p_peaks < len(ecg))]

                    if len(valid_p_peaks) > 0:
                        p_amplitudes = ecg[valid_p_peaks]
                        features['p_wave_amplitude_mean'] = float(np.mean(p_amplitudes))
                        features['p_wave_amplitude_std'] = float(np.std(p_amplitudes))

            # T波特征 - 更健壮的处理
            if 'ECG_T_Peaks' in waves and waves['ECG_T_Peaks'] is not None:
                t_peaks = waves['ECG_T_Peaks']
                # 确保是数组并过滤NaN
                t_peaks = np.array(t_peaks, dtype=float)
                valid_t_peaks = t_peaks[~np.isnan(t_peaks)]

                if len(valid_t_peaks) > 0:
                    # 转换为整数索引前确保是标量值
                    valid_t_peaks = valid_t_peaks.astype(int)
                    # 确保索引在有效范围内
                    valid_t_peaks = valid_t_peaks[(valid_t_peaks >= 0) & (valid_t_peaks < len(ecg))]

                    if len(valid_t_peaks) > 0:
                        t_amplitudes = ecg[valid_t_peaks]
                        features['t_wave_amplitude_mean'] = float(np.mean(t_amplitudes))
                        features['t_wave_amplitude_std'] = float(np.std(t_amplitudes))

            # QRS特征 - R峰已经在self.r_peaks中，确保有效
            valid_r_peaks = self.r_peaks[(self.r_peaks >= 0) & (self.r_peaks < len(ecg))]
            if len(valid_r_peaks) > 0:
                qrs_amplitudes = ecg[valid_r_peaks]
                features['qrs_amplitude_mean'] = float(np.mean(qrs_amplitudes))
                features['qrs_amplitude_std'] = float(np.std(qrs_amplitudes))

        except Exception as e:
            print(f"警告: 波形分析失败: {str(e)}，使用备用方法")

            # 备用方法：仅使用R峰特征
            valid_r_peaks = self.r_peaks[(self.r_peaks >= 0) & (self.r_peaks < len(ecg))]
            if len(valid_r_peaks) > 0:
                qrs_amplitudes = ecg[valid_r_peaks]
                features['qrs_amplitude_mean'] = float(np.mean(qrs_amplitudes))
                features['qrs_amplitude_std'] = float(np.std(qrs_amplitudes))
            else:
                # 如果连R峰都无效，设置默认值
                features['qrs_amplitude_mean'] = 0.0
                features['qrs_amplitude_std'] = 0.0

        return features

    # ========================= HRV时域特征 =========================

    def extract_hrv_time_features(self) -> Dict[str, float]:
        """
        提取HRV时域统计特征

        包括:
        - SDNN: RR间期标准差
        - RMSSD: 相邻RR差平方均值根
        - pNN50: 相邻RR差>50ms的比例

        Returns
        -------
        features : dict
            HRV时域特征字典
        """
        features = {}

        if len(self.rr_intervals) < 2:
            print("警告: RR间期数量不足，跳过HRV时域特征")
            return features

        # SDNN
        features['SDNN'] = float(np.std(self.rr_intervals, ddof=1))

        # 相邻RR间期差值
        diff_rr = np.diff(self.rr_intervals)

        # RMSSD
        features['RMSSD'] = float(np.sqrt(np.mean(diff_rr ** 2)))

        # NN50和pNN50
        nn50 = np.sum(np.abs(diff_rr) > 50)
        features['NN50'] = int(nn50)
        features['pNN50'] = float(nn50 / len(diff_rr) * 100) if len(diff_rr) > 0 else 0.0

        # 额外指标
        features['mean_nn'] = float(np.mean(self.rr_intervals))
        features['median_nn'] = float(np.median(self.rr_intervals))
        features['range_nn'] = float(np.max(self.rr_intervals) - np.min(self.rr_intervals))
        features['cv_nn'] = float(np.std(self.rr_intervals) / np.mean(self.rr_intervals))
        features['SDSD'] = float(np.std(diff_rr, ddof=1)) if len(diff_rr) > 1 else 0.0

        return features

    # ========================= HRV频域特征 =========================

    # ========================= HRV频域特征 =========================

    def extract_hrv_frequency_features(self, resample_rate: float = 4.0) -> Dict[str, float]:
        """
        提取HRV频域特征

        包括:
        - VLF功率 (0.003-0.04 Hz)
        - LF功率 (0.04-0.15 Hz)
        - HF功率 (0.15-0.4 Hz)
        - LF/HF比

        Parameters
        ----------
        resample_rate : float
            重采样率，默认4Hz

        Returns
        -------
        features : dict
            HRV频域特征字典
        """
        features = {}

        if len(self.rr_intervals) < 20:  # 降低要求到20个RR间期
            print(f"警告: RR间期数量不足({len(self.rr_intervals)})，需要至少20个")
            return features

        # 数据质量检查
        valid_rr = self.rr_intervals[(self.rr_intervals > 300) & (self.rr_intervals < 2000)]
        if len(valid_rr) < len(self.rr_intervals) * 0.7:  # 降低要求到70%
            print(f"警告: 有效RR间期比例过低({len(valid_rr)}/{len(self.rr_intervals)})")
            return features

        rr_intervals = valid_rr

        try:
            # 创建时间轴并插值
            time_rr = np.cumsum(rr_intervals) / 1000
            time_rr = np.insert(time_rr, 0, 0)

            total_duration = time_rr[-1]

            # 根据信号时长自适应调整参数
            if total_duration < 30:
                print(f"警告: 信号时长过短({total_duration:.1f}秒)，使用简化HRV分析")
                # 对于超短信号，只计算简单的频域特征
                return self._extract_simple_frequency_features(rr_intervals)
            elif total_duration < 60:
                print(f"信号时长: {total_duration:.1f}秒，使用短时HRV分析方法")
                # 短信号：调整频段和参数
                min_duration = total_duration
                resample_rate = max(2.0, resample_rate)  # 降低重采样率
                # 调整频段范围
                vlf_band = (0.003, min(0.04, 0.5))
                lf_band = (0.04, min(0.15, 0.5))
                hf_band = (0.15, min(0.4, 0.5))
            else:
                min_duration = total_duration
                vlf_band = (0.003, 0.04)
                lf_band = (0.04, 0.15)
                hf_band = (0.15, 0.4)

            time_interp = np.arange(0, min_duration, 1 / resample_rate)

            if len(time_interp) < 20:  # 降低要求到20个点
                print(f"警告: 插值点数过少({len(time_interp)})")
                return self._extract_simple_frequency_features(rr_intervals)

            # 插值RR间期
            rr_interp = interp1d(time_rr, np.append(rr_intervals[0], rr_intervals),
                                 kind='cubic', fill_value='extrapolate')(time_interp)

            # 去趋势
            rr_detrend = signal.detrend(rr_interp)

            if np.std(rr_detrend) < 0.5:  # 降低阈值
                print(f"警告: 去趋势后变异性过低(std={np.std(rr_detrend):.2f})")
                return self._extract_simple_frequency_features(rr_intervals)

            # 根据信号长度自适应调整nperseg
            nperseg = min(128, len(rr_detrend) // 3)  # 使用更小的窗口
            nperseg = max(16, nperseg)  # 确保最小为16

            if nperseg < 16:
                print(f"警告: nperseg={nperseg} < 16")
                return self._extract_simple_frequency_features(rr_intervals)

            # 计算PSD
            freqs, psd = signal.welch(rr_detrend, fs=resample_rate,
                                      nperseg=nperseg,
                                      noverlap=nperseg // 2,
                                      scaling='density')

            if np.sum(psd) == 0 or np.isnan(np.sum(psd)):
                print("警告: 功率谱密度计算异常")
                return self._extract_simple_frequency_features(rr_intervals)

            # 计算功率
            vlf_mask = (freqs >= vlf_band[0]) & (freqs < vlf_band[1])
            lf_mask = (freqs >= lf_band[0]) & (freqs < lf_band[1])
            hf_mask = (freqs >= hf_band[0]) & (freqs < hf_band[1])

            # 如果某个频段没有足够的点，使用相邻点
            if np.sum(vlf_mask) < 2:
                # 扩展VLF频段
                vlf_mask = (freqs >= vlf_band[0]) & (freqs < lf_band[1])
            if np.sum(lf_mask) < 2:
                # 扩展LF频段
                lf_mask = (freqs >= lf_band[0]) & (freqs < hf_band[1])
            if np.sum(hf_mask) < 2:
                # 扩展HF频段
                hf_mask = (freqs >= hf_band[0]) & (freqs <= max(0.5, hf_band[1]))

            vlf_power = np.trapz(psd[vlf_mask], freqs[vlf_mask]) if np.sum(vlf_mask) > 0 else 0.0
            lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.sum(lf_mask) > 0 else 0.0
            hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.sum(hf_mask) > 0 else 0.0

            total_power = vlf_power + lf_power + hf_power

            # 保存功率值
            features['VLF_power'] = float(vlf_power)
            features['LF_power'] = float(lf_power)
            features['HF_power'] = float(hf_power)
            features['total_power'] = float(total_power)

            # LF/HF比
            features['LF_HF_ratio'] = float(lf_power / hf_power) if hf_power > 1e-10 else 0.0

            # 归一化功率
            if total_power > vlf_power and (total_power - vlf_power) > 1e-10:
                features['LF_norm'] = float(lf_power / (total_power - vlf_power) * 100)
                features['HF_norm'] = float(hf_power / (total_power - vlf_power) * 100)
            else:
                features['LF_norm'] = 0.0
                features['HF_norm'] = 0.0

            # 峰值频率
            if np.sum(lf_mask) > 0:
                lf_psd_subset = psd[lf_mask]
                lf_freqs = freqs[lf_mask]
                if len(lf_psd_subset) > 0:
                    lf_peak_idx = np.argmax(lf_psd_subset)
                    features['LF_peak_freq'] = float(lf_freqs[lf_peak_idx])

            if np.sum(hf_mask) > 0:
                hf_psd_subset = psd[hf_mask]
                hf_freqs = freqs[hf_mask]
                if len(hf_psd_subset) > 0:
                    hf_peak_idx = np.argmax(hf_psd_subset)
                    features['HF_peak_freq'] = float(hf_freqs[hf_peak_idx])

            # 添加信号质量指示
            features['signal_duration'] = float(total_duration)
            features['n_rr_intervals'] = len(rr_intervals)

        except Exception as e:
            print(f"警告: 频域分析失败: {str(e)}，使用简化方法")
            return self._extract_simple_frequency_features(rr_intervals)

        return features

    def _extract_simple_frequency_features(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        简化版的频域特征提取（用于短信号）

        Parameters
        ----------
        rr_intervals : np.ndarray
            RR间期序列

        Returns
        -------
        features : dict
            简化的频域特征
        """
        features = {}

        try:
            # 计算基本统计量作为替代
            features['rr_mean'] = float(np.mean(rr_intervals))
            features['rr_std'] = float(np.std(rr_intervals))
            features['rr_cv'] = float(np.std(rr_intervals) / np.mean(rr_intervals))

            # 使用自相关作为频域的简单替代
            if len(rr_intervals) > 10:
                # 计算相邻RR间期的相关性
                rr_corr = np.corrcoef(rr_intervals[:-1], rr_intervals[1:])[0, 1]
                features['rr_autocorrelation'] = float(rr_corr) if not np.isnan(rr_corr) else 0.0

                # 计算高频波动的简单估计（使用差分）
                diff_rr = np.diff(rr_intervals)
                features['high_frequency_estimate'] = float(np.std(diff_rr))
                features['low_frequency_estimate'] = float(np.std(rr_intervals))

                # 估计LF/HF比（基于变异性比例）
                if features['high_frequency_estimate'] > 0:
                    features['LF_HF_ratio_estimate'] = float(
                        features['low_frequency_estimate'] / features['high_frequency_estimate']
                    )
                else:
                    features['LF_HF_ratio_estimate'] = 0.0

            # 标记为简化特征
            features['is_simplified'] = 1.0

        except Exception as e:
            print(f"警告: 简化频域分析也失败: {str(e)}")

        return features

    # ========================= HRV非线性特征 =========================

    def extract_hrv_nonlinear_features(self) -> Dict[str, float]:
        """
        提取HRV非线性特征

        包括:
        - DFA (Detrended Fluctuation Analysis)
        - Poincaré图参数 (SD1, SD2)

        Returns
        -------
        features : dict
            HRV非线性特征字典
        """
        features = {}

        if len(self.rr_intervals) < 50:
            print("警告: RR间期数量不足，跳过非线性特征")
            return features

        try:
            # DFA
            dfa_alpha1, dfa_alpha2 = self._detrended_fluctuation_analysis(self.rr_intervals)
            features['DFA_alpha1'] = float(dfa_alpha1)
            features['DFA_alpha2'] = float(dfa_alpha2)

            # Poincaré图参数
            sd1, sd2, sd_ratio = self._poincare_features(self.rr_intervals)
            features['poincare_SD1'] = float(sd1)
            features['poincare_SD2'] = float(sd2)
            features['poincare_SD_ratio'] = float(sd_ratio)
            features['poincare_ellipse_area'] = float(np.pi * sd1 * sd2)

        except Exception as e:
            print(f"警告: 非线性分析失败: {str(e)}")

        return features

    def _detrended_fluctuation_analysis(self, data: np.ndarray) -> Tuple[float, float]:
        """DFA分析"""
        N = len(data)
        y = np.cumsum(data - np.mean(data))

        scales = np.unique(np.logspace(0.5, 2, 20).astype(int))
        scales = scales[scales < N // 4]

        F = []
        for n in scales:
            n_segments = N // n
            F_n = []
            for i in range(n_segments):
                segment = y[i * n:(i + 1) * n]
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                fit = np.polyval(coeffs, x)
                F_n.append(np.sqrt(np.mean((segment - fit) ** 2)))
            F.append(np.mean(F_n))

        F = np.array(F)
        scales = scales[:len(F)]

        log_scales = np.log(scales)
        log_F = np.log(F)

        short_idx = scales <= 16
        long_idx = scales > 16

        alpha1 = np.polyfit(log_scales[short_idx], log_F[short_idx], 1)[0] if np.sum(short_idx) > 2 else 1.0
        alpha2 = np.polyfit(log_scales[long_idx], log_F[long_idx], 1)[0] if np.sum(long_idx) > 2 else 1.0

        return alpha1, alpha2

    def _poincare_features(self, rr_intervals: np.ndarray) -> Tuple[float, float, float]:
        """Poincaré图参数"""
        rr1 = rr_intervals[:-1]
        rr2 = rr_intervals[1:]

        diff_rr = rr2 - rr1
        sum_rr = rr2 + rr1

        SD1 = np.std(diff_rr, ddof=1) / np.sqrt(2)
        SD2 = np.std(sum_rr, ddof=1) / np.sqrt(2)
        SD_ratio = SD1 / SD2 if SD2 > 0 else 0.0

        return SD1, SD2, SD_ratio


# ========================= 辅助函数 =========================

def summarize_features(features: Dict[str, Any]) -> None:
    """打印特征摘要"""
    print("\n" + "=" * 60)
    print("ECG特征提取摘要")
    print("=" * 60)

    # 通用特征
    if 'common' in features and features['common']:
        print("\n[通用信号特征]")
        common = features['common']
        print(f"均值: {common.get('mean', 0):.3f}")
        print(f"RMS: {common.get('rms', 0):.3f}")
        print(f"Hjorth复杂度: {common.get('hjorth_complexity', 0):.3f}")
        print(f"频谱质心: {common.get('spectral_centroid', 0):.2f} Hz")

    # 形态学特征
    if 'morphological' in features and features['morphological']:
        print("\n[形态学特征]")
        morph = features['morphological']
        if 'mean_hr' in morph:
            print(f"平均心率: {morph['mean_hr']:.1f} bpm")
        if 'qrs_amplitude_mean' in morph:
            print(f"QRS平均振幅: {morph['qrs_amplitude_mean']:.3f}")

    # HRV时域
    if 'hrv_time' in features and features['hrv_time']:
        print("\n[HRV时域特征]")
        hrv_t = features['hrv_time']
        if 'SDNN' in hrv_t:
            print(f"SDNN: {hrv_t['SDNN']:.2f} ms")
        if 'RMSSD' in hrv_t:
            print(f"RMSSD: {hrv_t['RMSSD']:.2f} ms")
        if 'pNN50' in hrv_t:
            print(f"pNN50: {hrv_t['pNN50']:.2f}%")

    # HRV频域
    if 'hrv_frequency' in features and features['hrv_frequency']:
        print("\n[HRV频域特征]")
        hrv_f = features['hrv_frequency']
        if 'LF_power' in hrv_f:
            print(f"LF功率: {hrv_f['LF_power']:.2f}")
        if 'HF_power' in hrv_f:
            print(f"HF功率: {hrv_f['HF_power']:.2f}")
        if 'LF_HF_ratio' in hrv_f:
            print(f"LF/HF比: {hrv_f['LF_HF_ratio']:.2f}")

    # HRV非线性
    if 'hrv_nonlinear' in features and features['hrv_nonlinear']:
        print("\n[HRV非线性特征]")
        hrv_nl = features['hrv_nonlinear']
        if 'DFA_alpha1' in hrv_nl:
            print(f"DFA α1: {hrv_nl['DFA_alpha1']:.3f}")
        if 'poincare_SD1' in hrv_nl:
            print(f"Poincaré SD1: {hrv_nl['poincare_SD1']:.2f}")

    # 统计总数
    total_features = sum(
        len(v) if isinstance(v, dict) else 1
        for v in features.values()
    )
    print(f"\n总计提取特征数: {total_features}")
    print("=" * 60 + "\n")


def add_features_to_data_dict(data_dict: Dict[str, Any],
                              features: Dict[str, Any]) -> Dict[str, Any]:
    """将特征添加到数据字典"""
    if 'processed' not in data_dict:
        data_dict['processed'] = {}

    data_dict['processed']['ecg_features'] = features

    return data_dict


# ========================= 命令行接口 =========================

if __name__ == "__main__":
    print("运行演示示例...\n")

    # 创建示例ECG数据
    sampling_rate = 500
    duration = 120
    t = np.arange(0, duration, 1 / sampling_rate)

    # 生成真实ECG信号
    np.random.seed(42)
    heart_rate = 75
    ecg_signal = np.zeros_like(t)

    hr_variation = np.random.randn(int(duration * heart_rate / 60)) * 3

    beat_times = []
    current_time = 0
    beat_count = 0

    while current_time < duration:
        if beat_count < len(hr_variation):
            instantaneous_hr = heart_rate + hr_variation[beat_count]
        else:
            instantaneous_hr = heart_rate

        beat_interval = 60.0 / instantaneous_hr
        current_time += beat_interval

        if current_time < duration:
            beat_times.append(current_time)
            beat_count += 1

        # 生成PQRST波形
    for beat_time in beat_times:
        beat_idx = int(beat_time * sampling_rate)

        # P波
        p_start = beat_idx - int(0.16 * sampling_rate)
        p_width = int(0.08 * sampling_rate)
        if 0 < p_start < len(ecg_signal) - p_width:
            p_wave = 0.15 * np.sin(np.linspace(0, np.pi, p_width))
            ecg_signal[p_start:p_start + p_width] += p_wave

        # QRS波群
        qrs_start = beat_idx - int(0.02 * sampling_rate)
        qrs_width = int(0.08 * sampling_rate)
        if 0 < qrs_start < len(ecg_signal) - qrs_width:
            q_width = int(0.02 * sampling_rate)
            ecg_signal[qrs_start:qrs_start + q_width] -= np.linspace(0, 0.1, q_width)
            r_start = qrs_start + q_width
            r_width = int(0.03 * sampling_rate)
            if r_start + r_width < len(ecg_signal):
                r_wave = 1.2 * np.sin(np.linspace(0, np.pi, r_width))
                ecg_signal[r_start:r_start + r_width] += r_wave
            s_start = r_start + r_width
            s_width = int(0.03 * sampling_rate)
            if s_start + s_width < len(ecg_signal):
                ecg_signal[s_start:s_start + s_width] -= np.linspace(0.2, 0, s_width)

        # T波
        t_start = beat_idx + int(0.15 * sampling_rate)
        t_width = int(0.16 * sampling_rate)
        if 0 < t_start < len(ecg_signal) - t_width:
            t_wave = 0.25 * np.sin(np.linspace(0, np.pi, t_width))
            ecg_signal[t_start:t_start + t_width] += t_wave

    # 添加噪声
    baseline_drift = 0.05 * np.sin(2 * np.pi * 0.2 * t)
    noise = np.random.normal(0, 0.02, len(t))
    ecg_signal += baseline_drift + noise

    print(f"生成ECG信号: {duration}秒, {len(beat_times)}个心跳")

    # 创建数据字典
    demo_data = {
        'meta': {
            'subject_id': 'DEMO',
            'session_id': 'demo_session',
            'task': 'resting',
            'modality': ['ECG'],
            'device': 'Simulated',
            'sampling_rate': sampling_rate,
        },
        'signal': {
            'ECG': {
                'data': ecg_signal.reshape(1, -1),
                'sampling_rate': sampling_rate,
                'unit': 'mV',
                'channel_names': ['Lead II'],
                'reference': 'none',
                'time_offset': 0.0,
            }
        },
        'event': {},
        'processed': {}
    }

    # 实例化提取器
    extractor = ECGFeatureExtractor(sampling_rate=sampling_rate)

    # 提取特征
    features = extractor.extract_ecg_features(demo_data)

    # 显示摘要
    summarize_features(features)

    # 保存示例
    demo_data = add_features_to_data_dict(demo_data, features)

    print("演示完成!")