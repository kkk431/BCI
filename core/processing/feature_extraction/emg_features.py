# -*- coding: utf-8 -*-
"""
EMG信号特征提取模块 (基于CommonFeatureExtractor)
提取时域、频域、时频和非线性特征
适用于已预处理的EMG信号

输入: 符合BCI标准的四层数据字典
输出: 包含所有特征的字典，可保存到processed层

依赖: numpy, scipy, pywt (PyWavelets), antropy, nolds
安装: pip install numpy scipy PyWavelets antropy nolds
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# 导入通用特征提取器
try:
    from common_features import CommonFeatureExtractor
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
        import antropy
    except ImportError:
        missing.append('antropy')

    try:
        import nolds
    except ImportError:
        missing.append('nolds')

    if missing:
        raise ImportError(
            f"缺少必需的依赖包: {', '.join(missing)}\n"
            f"请运行: pip install {' '.join(missing)}"
        )


check_dependencies()

from scipy import signal
from scipy.stats import kurtosis, skew
import pywt
import antropy as ant
import nolds


# ========================= EMG特征提取器类 =========================

class EMGFeatureExtractor(CommonFeatureExtractor):
    """
    EMG特征提取器，继承自CommonFeatureExtractor。

    在父类通用特征的基础上，添加EMG特有的特征：
    - MAV, RMS, IEMG等时域特征
    - 平均频率、中位频率等频域特征
    - 肌疲劳相关特征

    使用方法:
        extractor = EMGFeatureExtractor(sampling_rate=2000)
        features = extractor.extract_emg_features(data_dict)
    """

    def __init__(self, sampling_rate: float):
        """
        初始化EMG特征提取器

        Parameters
        ----------
        sampling_rate : float
            EMG信号采样率 (Hz)
        """
        # 调用父类构造函数
        super().__init__(fs=sampling_rate)

        self.sampling_rate = sampling_rate

    # ========================= 主要接口 =========================

    def extract_emg_features(self, data_dict: Dict[str, Any],
                             emg_channel: Optional[int] = 0,
                             window_size: Optional[float] = None) -> Dict[str, Any]:
        """
        从EMG信号提取完整特征集

        Parameters
        ----------
        data_dict : dict
            符合BCI标准的四层数据字典
        emg_channel : int, optional
            使用的EMG通道索引，默认为0
        window_size : float, optional
            分析窗口大小（秒），如果为None则使用整段信号

        Returns
        -------
        features : dict
            包含所有特征的字典
        """

        # 1. 提取EMG信号
        if 'EMG' not in data_dict['signal']:
            raise ValueError("数据字典中未找到EMG信号模态")

        emg_signal = data_dict['signal']['EMG']['data']

        # 如果是多通道，选择指定通道
        if emg_signal.ndim == 2:
            emg = emg_signal[emg_channel, :]
        else:
            emg = emg_signal

        print(f"EMG信号长度: {len(emg)} 采样点")
        print(f"采样率: {self.sampling_rate} Hz")
        print(f"信号时长: {len(emg) / self.sampling_rate:.2f} 秒")

        # 如果指定了窗口大小，则分段处理
        if window_size is not None:
            window_samples = int(window_size * self.sampling_rate)
            print(f"使用窗口分析，窗口大小: {window_size} 秒")
            if len(emg) > window_samples:
                emg = emg[:window_samples]

        # 2. 提取各类特征
        features = {}

        # 使用父类方法提取通用特征
        print("\n[1/4] 提取通用信号特征...")
        features['common'] = super().extract_all_features(emg)

        # EMG特有特征
        print("[2/4] 提取EMG时域特征...")
        features['time_domain'] = self.extract_emg_time_features(emg)

        print("[3/4] 提取EMG频域特征...")
        features['frequency_domain'] = self.extract_emg_frequency_features(emg)

        print("[4/4] 提取EMG非线性特征...")
        features['nonlinear'] = self.extract_emg_nonlinear_features(emg)

        print("特征提取完成!")

        return features

    # ========================= EMG时域特征 =========================

    def extract_emg_time_features(self, emg: np.ndarray) -> Dict[str, float]:
        """
        提取EMG时域特征

        包括:
        - MAV: 平均绝对值
        - RMS: 均方根值
        - IEMG: 积分肌电值
        - VAR: 方差
        - WL: 波形长度
        - ZC: 过零次数
        - SSC: 斜率符号变化次数

        Parameters
        ----------
        emg : np.ndarray
            EMG信号

        Returns
        -------
        features : dict
            时域特征字典
        """
        features = {}

        # 1. MAV (Mean Absolute Value)
        features['MAV'] = float(np.mean(np.abs(emg)))

        # 2. RMS (Root Mean Square)
        features['RMS'] = float(np.sqrt(np.mean(emg ** 2)))

        # 3. IEMG (Integrated EMG)
        features['IEMG'] = float(np.sum(np.abs(emg)))

        # 4. VAR (Variance)
        features['VAR'] = float(np.var(emg))

        # 5. WL (Waveform Length)
        features['WL'] = float(np.sum(np.abs(np.diff(emg))))

        # 6. ZC (Zero Crossing)
        threshold = 0.01 * np.max(np.abs(emg))
        zc = np.sum(((emg[:-1] * emg[1:]) < 0) & (np.abs(emg[:-1] - emg[1:]) >= threshold))
        features['ZC'] = int(zc)

        # 7. SSC (Slope Sign Changes)
        diff_signal = np.diff(emg)
        ssc = np.sum(((diff_signal[:-1] * diff_signal[1:]) < 0) &
                     (np.abs(diff_signal[:-1] - diff_signal[1:]) >= threshold))
        features['SSC'] = int(ssc)

        # 额外的时域特征
        features['peak_amplitude'] = float(np.max(np.abs(emg)))
        features['DASDV'] = float(np.sqrt(np.mean(np.diff(emg) ** 2)))
        features['LOG'] = float(np.exp(np.mean(np.log(np.abs(emg) + 1e-10))))

        # Modified MAV
        N = len(emg)
        w = np.ones(N)
        w[:int(0.25 * N)] = 0.5
        w[int(0.75 * N):] = 0.5
        features['MAV1'] = float(np.sum(w * np.abs(emg)) / N)

        w2 = np.ones(N)
        for i in range(N):
            if i < 0.25 * N:
                w2[i] = 4 * i / N
            elif i > 0.75 * N:
                w2[i] = 4 * (N - i) / N
        features['MAV2'] = float(np.sum(w2 * np.abs(emg)) / N)

        # WAMP (Willison Amplitude)
        wamp_threshold = 0.01 * np.max(np.abs(emg))
        features['WAMP'] = int(np.sum(np.abs(np.diff(emg)) > wamp_threshold))

        # MYOP (Myopulse Percentage Rate)
        myop_threshold = 0.1 * np.max(np.abs(emg))
        features['MYOP'] = float(np.sum(np.abs(emg) > myop_threshold) / len(emg) * 100)

        # 统计特征
        features['skewness'] = float(skew(emg))
        features['kurtosis'] = float(kurtosis(emg))
        features['average_power'] = float(np.mean(emg ** 2))

        return features

    # ========================= EMG频域特征 =========================

    def extract_emg_frequency_features(self, emg: np.ndarray) -> Dict[str, float]:
        """
        提取EMG频域特征

        包括:
        - Mean Frequency: 平均频率
        - Median Frequency: 中位频率
        - Peak Frequency: 峰值频率

        Parameters
        ----------
        emg : np.ndarray
            EMG信号

        Returns
        -------
        features : dict
            频域特征字典
        """
        features = {}

        # 计算功率谱密度
        freqs, psd = signal.welch(emg, fs=self.sampling_rate, nperseg=min(256, len(emg)))

        # 归一化PSD
        psd_norm = psd / np.sum(psd)

        # 1. Mean Frequency (MNF)
        features['mean_frequency'] = float(np.sum(freqs * psd_norm))

        # 2. Median Frequency (MDF)
        cumsum_psd = np.cumsum(psd_norm)
        median_idx = np.where(cumsum_psd >= 0.5)[0]
        if len(median_idx) > 0:
            features['median_frequency'] = float(freqs[median_idx[0]])
        else:
            features['median_frequency'] = 0.0

        # 3. Peak Frequency (PKF)
        peak_idx = np.argmax(psd)
        features['peak_frequency'] = float(freqs[peak_idx])

        # 额外的频域特征
        features['total_power'] = float(np.sum(psd))
        features['frequency_variance'] = float(
            np.sum(((freqs - features['mean_frequency']) ** 2) * psd_norm)
        )
        features['frequency_std'] = float(np.sqrt(features['frequency_variance']))

        # 频率偏度和峰度
        if features['frequency_std'] > 0:
            freq_skew = np.sum(
                ((freqs - features['mean_frequency']) ** 3) * psd_norm
            ) / (features['frequency_std'] ** 3)
            features['frequency_skewness'] = float(freq_skew)

            freq_kurt = np.sum(
                ((freqs - features['mean_frequency']) ** 4) * psd_norm
            ) / (features['frequency_std'] ** 4)
            features['frequency_kurtosis'] = float(freq_kurt)
        else:
            features['frequency_skewness'] = 0.0
            features['frequency_kurtosis'] = 0.0

        # 功率谱熵
        psd_prob = psd / (np.sum(psd) + 1e-10)
        features['spectral_entropy'] = float(-np.sum(psd_prob * np.log2(psd_prob + 1e-10)))

        # 谱质心和谱扩展
        features['spectral_centroid'] = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-10))
        features['spectral_spread'] = float(
            np.sqrt(np.sum(((freqs - features['spectral_centroid']) ** 2) * psd) /
                    (np.sum(psd) + 1e-10))
        )

        # 谱滚降
        cumsum_psd_abs = np.cumsum(psd)
        rolloff_idx = np.where(cumsum_psd_abs >= 0.95 * cumsum_psd_abs[-1])[0]
        if len(rolloff_idx) > 0:
            features['spectral_rolloff'] = float(freqs[rolloff_idx[0]])
        else:
            features['spectral_rolloff'] = float(freqs[-1])

        # 频率比特征（用于肌疲劳分析）
        low_freq_mask = (freqs >= 20) & (freqs <= 45)
        high_freq_mask = (freqs >= 95) & (freqs <= 150)

        low_freq_power = np.sum(psd[low_freq_mask])
        high_freq_power = np.sum(psd[high_freq_mask])

        features['low_frequency_power'] = float(low_freq_power)
        features['high_frequency_power'] = float(high_freq_power)

        if high_freq_power > 0:
            features['frequency_ratio'] = float(low_freq_power / high_freq_power)
        else:
            features['frequency_ratio'] = 0.0

        return features

    # ========================= EMG非线性特征 =========================

    def extract_emg_nonlinear_features(self, emg: np.ndarray) -> Dict[str, float]:
        """
        提取EMG非线性特征

        包括:
        - Entropy: 信号复杂度与随机性
        - Fractal Dimension: 几何复杂结构
        - Hjorth参数

        Parameters
        ----------
        emg : np.ndarray
            EMG信号

        Returns
        -------
        features : dict
            非线性特征字典
        """
        features = {}

        # 1. 熵特征
        try:
            features['sample_entropy'] = float(ant.sample_entropy(emg, order=2))
        except Exception as e:
            print(f"警告: 样本熵计算失败: {str(e)}")
            features['sample_entropy'] = 0.0

        try:
            features['approximate_entropy'] = float(ant.app_entropy(emg, order=2))
        except Exception as e:
            print(f"警告: 近似熵计算失败: {str(e)}")
            features['approximate_entropy'] = 0.0

        try:
            features['permutation_entropy'] = float(
                ant.perm_entropy(emg, order=3, normalize=True)
            )
        except Exception as e:
            print(f"警告: 排列熵计算失败: {str(e)}")
            features['permutation_entropy'] = 0.0

        # Hjorth参数 - 修复版本
        try:
            # antropy的hjorth_params返回两个值: activity, mobility
            hjorth_activity, hjorth_mobility = ant.hjorth_params(emg)
            features['hjorth_activity'] = float(hjorth_activity)
            features['hjorth_mobility'] = float(hjorth_mobility)

            # 手动计算复杂度: mobility of first derivative / mobility
            if len(emg) > 2:
                # 计算一阶导数的mobility
                first_deriv = np.diff(emg)
                _, mobility_deriv = ant.hjorth_params(first_deriv)
                features['hjorth_complexity'] = float(mobility_deriv / hjorth_mobility) if hjorth_mobility > 0 else 0.0
            else:
                features['hjorth_complexity'] = 0.0

        except Exception as e:
            print(f"警告: Hjorth参数计算失败: {str(e)}")
            features['hjorth_activity'] = 0.0
            features['hjorth_mobility'] = 0.0
            features['hjorth_complexity'] = 0.0

        # 光谱熵
        try:
            features['spectral_entropy'] = float(
                ant.spectral_entropy(emg, sf=self.sampling_rate,
                                     method='welch', normalize=True)
            )
        except Exception as e:
            print(f"警告: 光谱熵计算失败: {str(e)}")
            features['spectral_entropy'] = 0.0

        # SVD熵
        try:
            features['svd_entropy'] = float(
                ant.svd_entropy(emg, order=3, delay=1, normalize=True)
            )
        except Exception as e:
            print(f"警告: SVD熵计算失败: {str(e)}")
            features['svd_entropy'] = 0.0

        # 2. 分形维数
        try:
            features['higuchi_fd'] = float(ant.higuchi_fd(emg, kmax=10))
        except Exception as e:
            print(f"警告: Higuchi分形维数计算失败: {str(e)}")
            features['higuchi_fd'] = 0.0

        try:
            features['katz_fd'] = float(ant.katz_fd(emg))
        except Exception as e:
            print(f"警告: Katz分形维数计算失败: {str(e)}")
            features['katz_fd'] = 0.0

        try:
            features['petrosian_fd'] = float(ant.petrosian_fd(emg))
        except Exception as e:
            print(f"警告: Petrosian分形维数计算失败: {str(e)}")
            features['petrosian_fd'] = 0.0

        # DFA
        try:
            features['dfa'] = float(nolds.dfa(emg))
        except Exception as e:
            print(f"警告: DFA计算失败: {str(e)}")
            features['dfa'] = 0.0

        # 3. Lyapunov指数（仅对短信号）
        if len(emg) < 5000:
            try:
                lyap = nolds.lyap_r(emg, emb_dim=10, lag=None, min_tsep=None)
                features['lyapunov_exponent'] = float(lyap)
            except Exception as e:
                print(f"警告: Lyapunov指数计算失败: {str(e)}")
                features['lyapunov_exponent'] = 0.0

        # Hurst指数
        try:
            features['hurst_exponent'] = float(nolds.hurst_rs(emg))
        except Exception as e:
            print(f"警告: Hurst指数计算失败: {str(e)}")
            features['hurst_exponent'] = 0.0

        # 相关维数（仅对短信号）
        if len(emg) < 2000:
            try:
                corr_dim = nolds.corr_dim(emg, emb_dim=10)
                features['correlation_dimension'] = float(corr_dim)
            except Exception as e:
                print(f"警告: 相关维数计算失败: {str(e)}")
                features['correlation_dimension'] = 0.0

        return features

    # ========================= 滑动窗口特征提取 =========================

    def extract_features_windowed(self, emg: np.ndarray,
                                  window_size: float = 0.25,
                                  overlap: float = 0.5) -> List[Dict[str, Any]]:
        """
        使用滑动窗口提取特征

        Parameters
        ----------
        emg : np.ndarray
            EMG信号
        window_size : float
            窗口大小（秒）
        overlap : float
            重叠比例 (0-1)

        Returns
        -------
        features_list : list
            每个窗口的特征字典列表
        """
        window_samples = int(window_size * self.sampling_rate)
        step_samples = int(window_samples * (1 - overlap))

        features_list = []

        for start in range(0, len(emg) - window_samples + 1, step_samples):
            end = start + window_samples
            window_emg = emg[start:end]

            # 提取特征（不包含通用特征以提高速度）
            features = {
                'time_domain': self.extract_emg_time_features(window_emg),
                'frequency_domain': self.extract_emg_frequency_features(window_emg),
            }

            features['window_start'] = start / self.sampling_rate
            features['window_end'] = end / self.sampling_rate

            features_list.append(features)

        return features_list


# ========================= 辅助函数 =========================

def summarize_features(features: Dict[str, Any]) -> None:
    """打印特征摘要"""
    print("\n" + "=" * 60)
    print("EMG特征提取摘要")
    print("=" * 60)

    # 通用特征
    if 'common' in features and features['common']:
        print("\n[通用信号特征]")
        common = features['common']
        print(f"均值: {common.get('mean', 0):.3f}")
        print(f"RMS: {common.get('rms', 0):.3f}")
        print(f"Hjorth复杂度: {common.get('hjorth_complexity', 0):.3f}")
        print(f"频谱质心: {common.get('spectral_centroid', 0):.2f} Hz")

    # 时域特征
    if 'time_domain' in features and features['time_domain']:
        print("\n[EMG时域特征]")
        td = features['time_domain']
        key_features = ['MAV', 'RMS', 'IEMG', 'VAR', 'WL', 'ZC', 'SSC']
        for feat in key_features:
            if feat in td:
                print(f"  {feat:20s}: {td[feat]}")

    # 频域特征
    if 'frequency_domain' in features and features['frequency_domain']:
        print("\n[EMG频域特征]")
        fd = features['frequency_domain']
        key_features = ['mean_frequency', 'median_frequency', 'peak_frequency']
        for feat in key_features:
            if feat in fd:
                print(f"  {feat:20s}: {fd[feat]:.2f}")

    # 非线性特征
    if 'nonlinear' in features and features['nonlinear']:
        print("\n[EMG非线性特征]")
        nl = features['nonlinear']
        key_features = ['sample_entropy', 'approximate_entropy', 'higuchi_fd']
        for feat in key_features:
            if feat in nl:
                print(f"  {feat:20s}: {nl[feat]:.3f}")

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

    data_dict['processed']['emg_features'] = features

    return data_dict


# ========================= 主入口 =========================

if __name__ == "__main__":
    import sys
    print("运行演示示例...\n")

    # 创建示例EMG数据
    sampling_rate = 2000
    duration = 10
    t = np.arange(0, duration, 1 / sampling_rate)

    # 生成模拟EMG信号
    np.random.seed(42)
    emg_signal = np.zeros_like(t)

    # 添加几个收缩周期
    contraction_times = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    for ct in contraction_times:
        start_idx = int(ct * sampling_rate)
        duration_samples = int(0.3 * sampling_rate)

        if start_idx + duration_samples < len(t):
            # 多个频率成分
            burst = np.zeros(duration_samples)
            for j in range(3):
                freq = 80 + (j - 1) * 20 + np.random.rand() * 40
                amplitude = 0.5 * (0.5 + 0.3 * np.random.rand())
                burst += amplitude * np.sin(2 * np.pi * freq *
                                            t[start_idx:start_idx + duration_samples])

            # 高斯包络
            envelope = np.exp(-((np.arange(duration_samples) - duration_samples / 2) ** 2) /
                              (2 * (duration_samples / 4) ** 2))

            burst = burst * envelope
            emg_signal[start_idx:start_idx + duration_samples] += burst

    # 添加噪声
    baseline_noise = np.random.normal(0, 0.02, len(t))
    powerline = 0.01 * np.sin(2 * np.pi * 50 * t)
    emg_signal += baseline_noise + powerline

    # 归一化
    emg_signal = emg_signal / (np.max(np.abs(emg_signal)) + 1e-10)

    print(f"生成EMG信号: {duration}秒, {len(contraction_times)}次收缩")

    # 创建数据字典
    demo_data = {
        'meta': {
            'subject_id': 'DEMO',
            'session_id': 'demo_session',
            'task': 'isometric_contraction',
            'modality': ['EMG'],
            'device': 'Simulated',
            'sampling_rate': sampling_rate,
        },
        'signal': {
            'EMG': {
                'data': emg_signal.reshape(1, -1),
                'sampling_rate': sampling_rate,
                'unit': 'mV',
                'channel_names': ['Biceps'],
                'reference': 'bipolar',
                'time_offset': 0.0,
            }
        },
        'event': {},
        'processed': {}
    }

    # 实例化提取器
    extractor = EMGFeatureExtractor(sampling_rate=sampling_rate)

    # 提取特征
    features = extractor.extract_emg_features(demo_data)

    # 显示摘要
    summarize_features(features)

    # 保存示例
    demo_data = add_features_to_data_dict(demo_data, features)

    print("演示完成!")