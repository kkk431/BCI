# -*- coding: utf-8 -*-
"""
EMG信号特征提取模块
提取时域、频域、时频和非线性特征
适用于已预处理的EMG信号

输入: 符合BCI标准的四层数据字典
输出: 包含所有特征的字典，可保存到processed层

依赖: numpy, scipy, pywt (PyWavelets), antropy
安装: pip install numpy scipy PyWavelets antropy nolds
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

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


# ========================= 主要特征提取函数 =========================

def extract_emg_features(data_dict: Dict[str, Any], 
                         emg_channel: Optional[int] = 0,
                         sampling_rate: Optional[float] = None,
                         window_size: Optional[float] = None) -> Dict[str, Any]:
    """
    从EMG信号提取完整特征集
    
    Parameters
    ----------
    data_dict : dict
        符合BCI标准的四层数据字典
    emg_channel : int, optional
        使用的EMG通道索引，默认为0（第一个通道）
    sampling_rate : float, optional
        采样率，如果为None则从data_dict中获取
    window_size : float, optional
        分析窗口大小（秒），如果为None则使用整段信号
    
    Returns
    -------
    features : dict
        包含所有特征的字典，结构如下:
        {
            'time_domain': {...},      # 时域特征
            'frequency_domain': {...}, # 频域特征
            'time_frequency': {...},   # 时频特征
            'nonlinear': {...}         # 非线性特征
        }
    
    Examples
    --------
    # >>> features = extract_emg_features(data_dict)
    # >>> print(features['time_domain']['MAV'])
    # >>> print(features['frequency_domain']['mean_frequency'])
    """
    
    # 提取EMG信号
    if 'EMG' not in data_dict['signal']:
        raise ValueError("数据字典中未找到EMG信号模态")
    
    emg_signal = data_dict['signal']['EMG']['data']
    
    # 如果是多通道，选择指定通道
    if emg_signal.ndim == 2:
        emg = emg_signal[emg_channel, :]
    else:
        emg = emg_signal
    
    # 获取采样率
    if sampling_rate is None:
        sampling_rate = data_dict['signal']['EMG'].get('sampling_rate', 2000)
    
    print(f"EMG信号长度: {len(emg)} 采样点")
    print(f"采样率: {sampling_rate} Hz")
    print(f"信号时长: {len(emg)/sampling_rate:.2f} 秒")
    
    # 如果指定了窗口大小，则分段处理
    if window_size is not None:
        window_samples = int(window_size * sampling_rate)
        print(f"使用窗口分析，窗口大小: {window_size} 秒 ({window_samples} 采样点)")
        # 这里简化处理，只使用第一个窗口
        if len(emg) > window_samples:
            emg = emg[:window_samples]
    
    # 提取各类特征
    features = {}
    
    # 1. 时域特征
    print("\n[1/4] 提取时域特征...")
    features['time_domain'] = extract_time_domain_features(emg)
    
    # 2. 频域特征
    print("[2/4] 提取频域特征...")
    features['frequency_domain'] = extract_frequency_domain_features(
        emg, sampling_rate
    )
    
    # 3. 时频特征
    print("[3/4] 提取时频特征...")
    features['time_frequency'] = extract_time_frequency_features(
        emg, sampling_rate
    )
    
    # 4. 非线性特征
    print("[4/4] 提取非线性特征...")
    features['nonlinear'] = extract_nonlinear_features(emg, sampling_rate)
    
    print("✓ 特征提取完成!")
    
    return features


# ========================= 时域特征 =========================

def extract_time_domain_features(emg: np.ndarray) -> Dict[str, float]:
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
    
    # 1. MAV (Mean Absolute Value) - 平均绝对值
    # 表示肌肉激活强度
    features['MAV'] = float(np.mean(np.abs(emg)))
    
    # 2. RMS (Root Mean Square) - 均方根
    # 衡量信号能量与收缩力度
    features['RMS'] = float(np.sqrt(np.mean(emg ** 2)))
    
    # 3. IEMG (Integrated EMG) - 积分肌电
    # 反映整体肌肉活动量
    features['IEMG'] = float(np.sum(np.abs(emg)))
    
    # 4. VAR (Variance) - 方差
    # 描述幅度离散程度
    features['VAR'] = float(np.var(emg))
    
    # 5. WL (Waveform Length) - 波形长度
    # 刻画信号复杂度与变化速度
    features['WL'] = float(np.sum(np.abs(np.diff(emg))))
    
    # 6. ZC (Zero Crossing) - 过零次数
    # 用于估计信号频率成分变化
    # 使用小阈值避免噪声影响
    threshold = 0.01 * np.max(np.abs(emg))
    zc = np.sum(((emg[:-1] * emg[1:]) < 0) & (np.abs(emg[:-1] - emg[1:]) >= threshold))
    features['ZC'] = int(zc)
    
    # 7. SSC (Slope Sign Changes) - 斜率符号变化次数
    # 反映波形结构复杂度
    diff_signal = np.diff(emg)
    ssc = np.sum(((diff_signal[:-1] * diff_signal[1:]) < 0) & 
                 (np.abs(diff_signal[:-1] - diff_signal[1:]) >= threshold))
    features['SSC'] = int(ssc)
    
    # 额外的时域特征
    
    # 8. 峰值幅度
    features['peak_amplitude'] = float(np.max(np.abs(emg)))
    
    # 9. 平均幅度变化 (DASDV - Difference Absolute Standard Deviation Value)
    features['DASDV'] = float(np.sqrt(np.mean(np.diff(emg) ** 2)))
    
    # 10. 对数检波器 (LOG)
    features['LOG'] = float(np.exp(np.mean(np.log(np.abs(emg) + 1e-10))))
    
    # 11. Modified Mean Absolute Value (MAV1)
    # 对边缘部分加权
    N = len(emg)
    w = np.ones(N)
    w[:int(0.25*N)] = 0.5
    w[int(0.75*N):] = 0.5
    features['MAV1'] = float(np.sum(w * np.abs(emg)) / N)
    
    # 12. Modified Mean Absolute Value (MAV2)
    # 另一种加权方式
    w2 = np.ones(N)
    for i in range(N):
        if i < 0.25 * N:
            w2[i] = 4 * i / N
        elif i > 0.75 * N:
            w2[i] = 4 * (N - i) / N
    features['MAV2'] = float(np.sum(w2 * np.abs(emg)) / N)
    
    # 13. Willison Amplitude (WAMP)
    # 阈值判定的相邻样本差异计数
    wamp_threshold = 0.01 * np.max(np.abs(emg))
    features['WAMP'] = int(np.sum(np.abs(np.diff(emg)) > wamp_threshold))
    
    # 14. Myopulse Percentage Rate (MYOP)
    # 超过阈值的样本比例
    myop_threshold = 0.1 * np.max(np.abs(emg))
    features['MYOP'] = float(np.sum(np.abs(emg) > myop_threshold) / len(emg) * 100)
    
    # 15. 偏度和峰度
    features['skewness'] = float(skew(emg))
    features['kurtosis'] = float(kurtosis(emg))
    
    # 16. 平均功率
    features['average_power'] = float(np.mean(emg ** 2))
    
    return features


# ========================= 频域特征 =========================

def extract_frequency_domain_features(emg: np.ndarray, 
                                      sampling_rate: float) -> Dict[str, float]:
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
    sampling_rate : float
        采样率
    
    Returns
    -------
    features : dict
        频域特征字典
    """
    features = {}
    
    # 计算功率谱密度
    freqs, psd = signal.welch(emg, fs=sampling_rate, nperseg=min(256, len(emg)))
    
    # 归一化PSD
    psd_norm = psd / np.sum(psd)
    
    # 1. Mean Frequency (MNF) - 平均频率
    # 功率谱平均频率，反映肌纤维激活分布
    features['mean_frequency'] = float(np.sum(freqs * psd_norm))
    
    # 2. Median Frequency (MDF) - 中位频率
    # 将功率分为两半的频率，常用于肌疲劳分析
    cumsum_psd = np.cumsum(psd_norm)
    median_idx = np.where(cumsum_psd >= 0.5)[0]
    if len(median_idx) > 0:
        features['median_frequency'] = float(freqs[median_idx[0]])
    else:
        features['median_frequency'] = 0.0
    
    # 3. Peak Frequency (PKF) - 峰值频率
    # 功率最大频率点，用于识别主频成分
    peak_idx = np.argmax(psd)
    features['peak_frequency'] = float(freqs[peak_idx])
    
    # 额外的频域特征
    
    # 4. 总功率
    features['total_power'] = float(np.sum(psd))
    
    # 5. 频率方差
    features['frequency_variance'] = float(np.sum(((freqs - features['mean_frequency']) ** 2) * psd_norm))
    
    # 6. 频率标准差
    features['frequency_std'] = float(np.sqrt(features['frequency_variance']))
    
    # 7. 频率偏度
    if features['frequency_std'] > 0:
        freq_skew = np.sum(((freqs - features['mean_frequency']) ** 3) * psd_norm) / (features['frequency_std'] ** 3)
        features['frequency_skewness'] = float(freq_skew)
    else:
        features['frequency_skewness'] = 0.0
    
    # 8. 频率峰度
    if features['frequency_std'] > 0:
        freq_kurt = np.sum(((freqs - features['mean_frequency']) ** 4) * psd_norm) / (features['frequency_std'] ** 4)
        features['frequency_kurtosis'] = float(freq_kurt)
    else:
        features['frequency_kurtosis'] = 0.0
    
    # 9. 功率谱熵
    # 归一化后的熵，表示频率分布的均匀程度
    psd_prob = psd / (np.sum(psd) + 1e-10)
    features['spectral_entropy'] = float(-np.sum(psd_prob * np.log2(psd_prob + 1e-10)))
    
    # 10. 谱质心
    features['spectral_centroid'] = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-10))
    
    # 11. 谱扩展
    features['spectral_spread'] = float(
        np.sqrt(np.sum(((freqs - features['spectral_centroid']) ** 2) * psd) / (np.sum(psd) + 1e-10))
    )
    
    # 12. 谱滚降（95%能量所在频率）
    cumsum_psd_abs = np.cumsum(psd)
    rolloff_idx = np.where(cumsum_psd_abs >= 0.95 * cumsum_psd_abs[-1])[0]
    if len(rolloff_idx) > 0:
        features['spectral_rolloff'] = float(freqs[rolloff_idx[0]])
    else:
        features['spectral_rolloff'] = float(freqs[-1])
    
    # 13. 频率比特征
    # 低频功率 (20-45 Hz) vs 高频功率 (95-150 Hz)
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


# ========================= 时频特征 =========================

def extract_time_frequency_features(emg: np.ndarray,
                                    sampling_rate: float) -> Dict[str, Any]:
    """
    提取EMG时频特征
    
    包括:
    - STFT Energy: 短时傅里叶变换能量
    - Wavelet Energy: 小波能量
    
    Parameters
    ----------
    emg : np.ndarray
        EMG信号
    sampling_rate : float
        采样率
    
    Returns
    -------
    features : dict
        时频特征字典
    """
    features = {}
    
    try:
        # 1. STFT特征
        # 短时频谱能量分布，用于动态频率分析
        nperseg = min(256, len(emg) // 4)
        f, t, Zxx = signal.stft(emg, fs=sampling_rate, nperseg=nperseg)
        
        # STFT能量
        stft_energy = np.abs(Zxx) ** 2
        
        features['stft_mean_energy'] = float(np.mean(stft_energy))
        features['stft_max_energy'] = float(np.max(stft_energy))
        features['stft_std_energy'] = float(np.std(stft_energy))
        features['stft_total_energy'] = float(np.sum(stft_energy))
        
        # 时间维度的能量变化
        time_energy = np.sum(stft_energy, axis=0)
        features['stft_energy_std_time'] = float(np.std(time_energy))
        
        # 频率维度的能量变化
        freq_energy = np.sum(stft_energy, axis=1)
        features['stft_energy_std_freq'] = float(np.std(freq_energy))
        
        # 主频率成分随时间的变化
        dominant_freqs = f[np.argmax(stft_energy, axis=0)]
        features['stft_dominant_freq_mean'] = float(np.mean(dominant_freqs))
        features['stft_dominant_freq_std'] = float(np.std(dominant_freqs))
        
        # 能量集中度
        stft_energy_flat = stft_energy.flatten()
        stft_energy_norm = stft_energy_flat / (np.sum(stft_energy_flat) + 1e-10)
        stft_entropy = -np.sum(stft_energy_norm * np.log2(stft_energy_norm + 1e-10))
        features['stft_energy_entropy'] = float(stft_entropy)
        
    except Exception as e:
        print(f"      警告: STFT分析失败: {str(e)}")
    
    try:
        # 2. 小波变换特征
        # 小波多尺度能量，捕获瞬态肌肉活动
        wavelet = 'db4'
        level = min(6, pywt.dwt_max_level(len(emg), wavelet))
        
        # 离散小波分解
        coeffs = pywt.wavedec(emg, wavelet, level=level)
        
        # 计算各层能量
        wavelet_energies = []
        for i, coef in enumerate(coeffs):
            energy = np.sum(coef ** 2)
            wavelet_energies.append(energy)
            features[f'wavelet_energy_level_{i}'] = float(energy)
        
        total_wavelet_energy = sum(wavelet_energies)
        features['wavelet_total_energy'] = float(total_wavelet_energy)
        
        # 归一化能量分布
        for i, energy in enumerate(wavelet_energies):
            if total_wavelet_energy > 0:
                features[f'wavelet_energy_ratio_level_{i}'] = float(
                    energy / total_wavelet_energy
                )
        
        # 能量熵
        energy_probs = np.array(wavelet_energies) / (total_wavelet_energy + 1e-10)
        wavelet_entropy = -np.sum(energy_probs * np.log2(energy_probs + 1e-10))
        features['wavelet_energy_entropy'] = float(wavelet_entropy)
        
        # 3. 连续小波变换 (CWT) - 可选
        if len(emg) < 10000:  # 只对较短信号计算CWT，避免内存问题
            scales = np.arange(1, min(128, len(emg)//10))
            coefficients, frequencies = pywt.cwt(emg, scales, wavelet,
                                                sampling_period=1/sampling_rate)
            
            cwt_energy = np.abs(coefficients) ** 2
            
            features['cwt_mean_energy'] = float(np.mean(cwt_energy))
            features['cwt_max_energy'] = float(np.max(cwt_energy))
            features['cwt_total_energy'] = float(np.sum(cwt_energy))
            
            # 主要尺度（最大能量的尺度）
            scale_energy = np.sum(cwt_energy, axis=1)
            dominant_scale_idx = np.argmax(scale_energy)
            features['cwt_dominant_scale'] = float(scales[dominant_scale_idx])
            if len(frequencies) > dominant_scale_idx:
                features['cwt_dominant_frequency'] = float(frequencies[dominant_scale_idx])
        
    except Exception as e:
        print(f"      警告: 小波分析失败: {str(e)}")
    
    return features


# ========================= 非线性特征 =========================

def extract_nonlinear_features(emg: np.ndarray, 
                               sampling_rate: float) -> Dict[str, float]:
    """
    提取EMG非线性特征
    
    包括:
    - Entropy: 信号复杂度与随机性
    - Fractal Dimension: 几何复杂结构
    - Lyapunov Exponent: 混沌程度
    
    Parameters
    ----------
    emg : np.ndarray
        EMG信号
    sampling_rate : float
        采样率
    
    Returns
    -------
    features : dict
        非线性特征字典
    """
    features = {}
    
    # 1. 熵特征 - 测量信号复杂度与随机性
    
    try:
        # 样本熵 (Sample Entropy)
        # m=2, r=0.2*std是标准参数
        features['sample_entropy'] = float(ant.sample_entropy(emg, order=2))
    except Exception as e:
        print(f"      警告: 样本熵计算失败: {str(e)}")
        features['sample_entropy'] = 0.0
    
    try:
        # 近似熵 (Approximate Entropy)
        features['approximate_entropy'] = float(ant.app_entropy(emg, order=2))
    except Exception as e:
        print(f"      警告: 近似熵计算失败: {str(e)}")
        features['approximate_entropy'] = 0.0
    
    try:
        # 排列熵 (Permutation Entropy)
        # 对EMG信号特别有用
        features['permutation_entropy'] = float(ant.perm_entropy(emg, order=3, normalize=True))
    except Exception as e:
        print(f"      警告: 排列熵计算失败: {str(e)}")
        features['permutation_entropy'] = 0.0
    
    try:
        # Hjorth参数 - 描述信号的活动性、移动性和复杂性
        hjorth_activity, hjorth_mobility, hjorth_complexity = ant.hjorth_params(emg)
        features['hjorth_activity'] = float(hjorth_activity)
        features['hjorth_mobility'] = float(hjorth_mobility)
        features['hjorth_complexity'] = float(hjorth_complexity)
    except Exception as e:
        print(f"      警告: Hjorth参数计算失败: {str(e)}")
    
    try:
        # 光谱熵 (Spectral Entropy)
        features['spectral_entropy'] = float(ant.spectral_entropy(emg, sf=sampling_rate, 
                                                                  method='welch', normalize=True))
    except Exception as e:
        print(f"      警告: 光谱熵计算失败: {str(e)}")
    
    try:
        # SVD熵 (Singular Value Decomposition Entropy)
        features['svd_entropy'] = float(ant.svd_entropy(emg, order=3, delay=1, normalize=True))
    except Exception as e:
        print(f"      警告: SVD熵计算失败: {str(e)}")
    
    # 2. 分形维数 - 描述信号几何复杂结构
    
    try:
        # Higuchi分形维数
        # 常用于EMG信号分析
        features['higuchi_fd'] = float(ant.higuchi_fd(emg, kmax=10))
    except Exception as e:
        print(f"      警告: Higuchi分形维数计算失败: {str(e)}")
    
    try:
        # Katz分形维数
        features['katz_fd'] = float(ant.katz_fd(emg))
    except Exception as e:
        print(f"      警告: Katz分形维数计算失败: {str(e)}")
    
    try:
        # Petrosian分形维数
        features['petrosian_fd'] = float(ant.petrosian_fd(emg))
    except Exception as e:
        print(f"      警告: Petrosian分形维数计算失败: {str(e)}")
    
    try:
        # Detrended Fluctuation Analysis (DFA)
        # 用nolds库计算
        features['dfa'] = float(nolds.dfa(emg))
    except Exception as e:
        print(f"      警告: DFA计算失败: {str(e)}")
    
    # 3. Lyapunov指数 - 刻画动态系统混沌程度
    
    try:
        # 最大Lyapunov指数
        # 正值表示混沌，负值表示稳定
        # 对于EMG信号，这个计算可能比较耗时
        if len(emg) < 5000:  # 只对较短信号计算
            lyap = nolds.lyap_r(emg, emb_dim=10, lag=None, min_tsep=None)
            features['lyapunov_exponent'] = float(lyap)
    except Exception as e:
        print(f"      警告: Lyapunov指数计算失败: {str(e)}")
    
    try:
        # Hurst指数
        # H > 0.5: 长程正相关（趋势性）
        # H = 0.5: 随机游走
        # H < 0.5: 长程负相关（均值回归）
        features['hurst_exponent'] = float(nolds.hurst_rs(emg))
    except Exception as e:
        print(f"      警告: Hurst指数计算失败: {str(e)}")
    
    # 4. 相关维数
    try:
        # 相关维数（计算量大，仅对短信号）
        if len(emg) < 2000:
            corr_dim = nolds.corr_dim(emg, emb_dim=10)
            features['correlation_dimension'] = float(corr_dim)
    except Exception as e:
        print(f"      警告: 相关维数计算失败: {str(e)}")
    
    # 5. 零交叉率的熵
    try:
        # 计算零交叉的模式熵
        zero_crossings = (emg[:-1] * emg[1:]) < 0
        zc_intervals = np.diff(np.where(zero_crossings)[0])
        if len(zc_intervals) > 0:
            zc_hist, _ = np.histogram(zc_intervals, bins=10, density=True)
            zc_hist = zc_hist[zc_hist > 0]
            features['zc_entropy'] = float(-np.sum(zc_hist * np.log2(zc_hist)))
    except Exception as e:
        print(f"      警告: 零交叉熵计算失败: {str(e)}")
    
    return features


# ========================= 特征汇总和保存 =========================

def summarize_features(features: Dict[str, Any]) -> None:
    """
    打印特征摘要
    
    Parameters
    ----------
    features : dict
        特征字典
    """
    print("\n" + "="*60)
    print("特征提取摘要")
    print("="*60)
    
    # 时域特征
    if 'time_domain' in features and features['time_domain']:
        print("\n[时域特征]")
        td = features['time_domain']
        key_features = ['MAV', 'RMS', 'IEMG', 'VAR', 'WL', 'ZC', 'SSC']
        for feat in key_features:
            if feat in td:
                print(f"  {feat:20s}: {td[feat]:.4f}")
    
    # 频域特征
    if 'frequency_domain' in features and features['frequency_domain']:
        print("\n[频域特征]")
        fd = features['frequency_domain']
        key_features = ['mean_frequency', 'median_frequency', 'peak_frequency', 
                       'total_power', 'spectral_entropy']
        for feat in key_features:
            if feat in fd:
                print(f"  {feat:20s}: {fd[feat]:.4f}")
    
    # 时频特征
    if 'time_frequency' in features and features['time_frequency']:
        print("\n[时频特征]")
        tf = features['time_frequency']
        key_features = ['stft_mean_energy', 'stft_total_energy', 
                       'wavelet_total_energy', 'wavelet_energy_entropy']
        for feat in key_features:
            if feat in tf:
                print(f"  {feat:20s}: {tf[feat]:.4f}")
    
    # 非线性特征
    if 'nonlinear' in features and features['nonlinear']:
        print("\n[非线性特征]")
        nl = features['nonlinear']
        key_features = ['sample_entropy', 'approximate_entropy', 'higuchi_fd', 
                       'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity']
        for feat in key_features:
            if feat in nl:
                print(f"  {feat:20s}: {nl[feat]:.4f}")
    
    # 统计特征数量
    total_features = sum(
        len(v) if isinstance(v, dict) else 1 
        for v in features.values()
    )
    print(f"\n总计提取特征数: {total_features}")
    print("="*60 + "\n")


def add_features_to_data_dict(data_dict: Dict[str, Any],
                              features: Dict[str, Any]) -> Dict[str, Any]:
    """
    将提取的特征添加到数据字典的processed层
    
    Parameters
    ----------
    data_dict : dict
        BCI标准数据字典
    features : dict
        提取的特征字典
    
    Returns
    -------
    data_dict : dict
        更新后的数据字典
    """
    if 'processed' not in data_dict:
        data_dict['processed'] = {}
    
    data_dict['processed']['emg_features'] = features
    
    return data_dict


# ========================= 批处理功能 =========================

def extract_features_windowed(emg: np.ndarray,
                              sampling_rate: float,
                              window_size: float = 0.25,
                              overlap: float = 0.5) -> List[Dict[str, Any]]:
    """
    使用滑动窗口提取特征
    
    Parameters
    ----------
    emg : np.ndarray
        EMG信号
    sampling_rate : float
        采样率
    window_size : float
        窗口大小（秒）
    overlap : float
        重叠比例 (0-1)
    
    Returns
    -------
    features_list : list
        每个窗口的特征字典列表
    """
    window_samples = int(window_size * sampling_rate)
    step_samples = int(window_samples * (1 - overlap))
    
    features_list = []
    
    for start in range(0, len(emg) - window_samples + 1, step_samples):
        end = start + window_samples
        window_emg = emg[start:end]
        
        # 提取特征
        features = {
            'time_domain': extract_time_domain_features(window_emg),
            'frequency_domain': extract_frequency_domain_features(window_emg, sampling_rate),
        }
        
        features['window_start'] = start / sampling_rate
        features['window_end'] = end / sampling_rate
        
        features_list.append(features)
    
    return features_list


# ========================= 命令行接口 =========================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print("运行演示示例...\n")
        
        # 创建示例EMG数据
        sampling_rate = 2000  # Hz
        duration = 5  # 秒
        t = np.arange(0, duration, 1/sampling_rate)
        
        # 生成模拟EMG信号
        # 模拟肌肉收缩: 突发性高频活动
        emg_signal = np.zeros_like(t)
        
        # 添加几个收缩周期
        contraction_times = [0.5, 1.5, 2.5, 3.5]
        for ct in contraction_times:
            start_idx = int(ct * sampling_rate)
            duration_samples = int(0.3 * sampling_rate)  # 300ms收缩
            
            if start_idx + duration_samples < len(t):
                # 高频成分 (50-150 Hz)
                freq = 80 + np.random.rand() * 40
                burst = np.sin(2 * np.pi * freq * t[start_idx:start_idx+duration_samples])
                
                # 幅度调制（高斯包络）
                envelope = np.exp(-((np.arange(duration_samples) - duration_samples/2) ** 2) / 
                                 (2 * (duration_samples/6) ** 2))
                
                emg_signal[start_idx:start_idx+duration_samples] = burst * envelope * 0.5
        
        # 添加基线噪声
        emg_signal += np.random.normal(0, 0.05, len(t))
        
        # 创建数据字典
        demo_data = {
            'meta': {
                'subject_id': 'DEMO',
                'session_id': 'demo_session',
                'task': 'grip_force',
                'modality': ['EMG'],
                'device': 'Simulated',
                'sampling_rate': sampling_rate,
            },
            'signal': {
                'EMG': {
                    'data': emg_signal.reshape(1, -1),
                    'sampling_rate': sampling_rate,
                    'unit': 'mV',
                    'channel_names': ['Flexor'],
                    'reference': 'bipolar',
                    'time_offset': 0.0,
                }
            },
            'event': {},
            'processed': {}
        }
        
        # 提取特征
        features = extract_emg_features(demo_data)
        
        # 显示摘要
        summarize_features(features)
        
        # 保存示例
        demo_data = add_features_to_data_dict(demo_data, features)
        
        print("演示完成!")
        print("\n使用方法:")
        print("  from emg_features import extract_emg_features")
        print("  features = extract_emg_features(your_data_dict)")
        
    else:
        print("使用方法:")
        print("  python emg_features.py --demo  # 运行演示")
        print("\n在代码中使用:")
        print("  from emg_features import extract_emg_features")
        print("  features = extract_emg_features(data_dict)")
