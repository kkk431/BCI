# -*- coding: utf-8 -*-
"""
ECG信号特征提取模块
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

from scipy import signal, stats
from scipy.interpolate import interp1d
import pywt
import neurokit2 as nk


# ========================= 主要特征提取函数 =========================

def extract_ecg_features(data_dict: Dict[str, Any], 
                         ecg_channel: Optional[int] = 0,
                         sampling_rate: Optional[float] = None) -> Dict[str, Any]:
    """
    从ECG信号提取完整特征集
    
    Parameters
    ----------
    data_dict : dict
        符合BCI标准的四层数据字典
    ecg_channel : int, optional
        使用的ECG通道索引，默认为0（第一个通道）
    sampling_rate : float, optional
        采样率，如果为None则从data_dict中获取
    
    Returns
    -------
    features : dict
        包含所有特征的字典，结构如下:
        {
            'morphological': {...},  # 时域形态学特征
            'hrv_time': {...},       # HRV时域特征
            'hrv_frequency': {...},  # HRV频域特征
            'time_frequency': {...}, # 时频特征
            'nonlinear': {...}       # 非线性特征
        }
    
    Examples
    --------
    # >>> features = extract_ecg_features(data_dict)
    # >>> print(features['morphological']['mean_hr'])
    # >>> print(features['hrv_time']['SDNN'])
    """
    
    # 提取ECG信号
    if 'ECG' not in data_dict['signal']:
        raise ValueError("数据字典中未找到ECG信号模态")
    
    ecg_signal = data_dict['signal']['ECG']['data']
    
    # 如果是多通道，选择指定通道
    if ecg_signal.ndim == 2:
        ecg = ecg_signal[ecg_channel, :]
    else:
        ecg = ecg_signal
    
    # 获取采样率
    if sampling_rate is None:
        sampling_rate = data_dict['signal']['ECG'].get('sampling_rate', 1000)
    
    print(f"ECG信号长度: {len(ecg)} 采样点")
    print(f"采样率: {sampling_rate} Hz")
    print(f"信号时长: {len(ecg)/sampling_rate:.2f} 秒")
    
    # 检测R波峰值
    print("\n[1/5] 检测R波峰值...")
    r_peaks = detect_r_peaks(ecg, sampling_rate)
    print(f"检测到 {len(r_peaks)} 个R波峰")
    
    # 计算RR间期
    rr_intervals = np.diff(r_peaks) / sampling_rate * 1000  # 转换为毫秒
    
    # 提取各类特征
    features = {}
    
    # 1. 时域形态学特征
    print("[2/5] 提取时域形态学特征...")
    features['morphological'] = extract_morphological_features(
        ecg, r_peaks, sampling_rate
    )
    
    # 2. HRV时域特征
    print("[3/5] 提取HRV时域特征...")
    features['hrv_time'] = extract_hrv_time_features(rr_intervals)
    
    # 3. HRV频域特征
    print("[4/5] 提取HRV频域特征...")
    features['hrv_frequency'] = extract_hrv_frequency_features(
        rr_intervals, sampling_rate
    )
    
    # 4. 时频特征
    print("[5/5] 提取时频和非线性特征...")
    features['time_frequency'] = extract_time_frequency_features(
        ecg, sampling_rate
    )
    
    # 5. 非线性特征
    features['nonlinear'] = extract_nonlinear_features(rr_intervals)
    
    print("特征提取完成!")
    
    return features


# ========================= R波检测 =========================

def detect_r_peaks(ecg: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    检测ECG信号中的R波峰值
    
    Parameters
    ----------
    ecg : np.ndarray
        ECG信号
    sampling_rate : float
        采样率
    
    Returns
    -------
    r_peaks : np.ndarray
        R波峰值的索引数组
    """
    # 使用neurokit2进行R波检测
    try:
        _, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
        r_peaks = info['ECG_R_Peaks']
    except Exception as e:
        print(f"警告: neurokit2检测失败，使用备用方法: {str(e)}")
        # 备用方法: 简单的峰值检测
        r_peaks = _simple_r_peak_detection(ecg, sampling_rate)
    
    return np.array(r_peaks)


def _simple_r_peak_detection(ecg: np.ndarray, sampling_rate: float) -> np.ndarray:
    """简单的R波峰值检测（备用方法）"""
    # 带通滤波 (5-15 Hz)
    sos = signal.butter(4, [5, 15], btype='bandpass', 
                        fs=sampling_rate, output='sos')
    ecg_filtered = signal.sosfilt(sos, ecg)
    
    # 找峰值
    distance = int(0.6 * sampling_rate)  # 最小间隔600ms
    peaks, _ = signal.find_peaks(ecg_filtered, distance=distance, 
                                  height=np.std(ecg_filtered))
    
    return peaks


# ========================= 时域形态学特征 =========================

def extract_morphological_features(ecg: np.ndarray, 
                                   r_peaks: np.ndarray,
                                   sampling_rate: float) -> Dict[str, float]:
    """
    提取时域形态学特征
    
    包括:
    - P波、QRS波、T波振幅
    - PR间期、QT间期、RR间期
    - QRS持续时间
    - 平均心率、心率变异性
    
    Parameters
    ----------
    ecg : np.ndarray
        ECG信号
    r_peaks : np.ndarray
        R波峰值索引
    sampling_rate : float
        采样率
    
    Returns
    -------
    features : dict
        形态学特征字典
    """
    features = {}
    
    # 计算RR间期
    rr_intervals = np.diff(r_peaks) / sampling_rate * 1000  # ms
    
    # 基本心率特征
    features['mean_rr_interval'] = float(np.mean(rr_intervals))  # ms
    features['std_rr_interval'] = float(np.std(rr_intervals))    # ms
    features['min_rr_interval'] = float(np.min(rr_intervals))
    features['max_rr_interval'] = float(np.max(rr_intervals))
    
    # 心率
    features['mean_hr'] = float(60000 / np.mean(rr_intervals))  # bpm
    features['min_hr'] = float(60000 / np.max(rr_intervals))
    features['max_hr'] = float(60000 / np.min(rr_intervals))
    
    # 尝试使用neurokit2进行详细波形分析
    try:
        # 检测所有波形
        _, waves = nk.ecg_delineate(ecg, r_peaks, sampling_rate=sampling_rate)
        
        # P波特征
        if 'ECG_P_Peaks' in waves and waves['ECG_P_Peaks'] is not None:
            p_peaks = waves['ECG_P_Peaks'][~np.isnan(waves['ECG_P_Peaks'])]
            if len(p_peaks) > 0:
                p_peaks = p_peaks.astype(int)
                p_amplitudes = ecg[p_peaks]
                features['p_wave_amplitude_mean'] = float(np.mean(p_amplitudes))
                features['p_wave_amplitude_std'] = float(np.std(p_amplitudes))
        
        # T波特征
        if 'ECG_T_Peaks' in waves and waves['ECG_T_Peaks'] is not None:
            t_peaks = waves['ECG_T_Peaks'][~np.isnan(waves['ECG_T_Peaks'])]
            if len(t_peaks) > 0:
                t_peaks = t_peaks.astype(int)
                t_amplitudes = ecg[t_peaks]
                features['t_wave_amplitude_mean'] = float(np.mean(t_amplitudes))
                features['t_wave_amplitude_std'] = float(np.std(t_amplitudes))
        
        # QRS特征
        qrs_amplitudes = ecg[r_peaks]
        features['qrs_amplitude_mean'] = float(np.mean(qrs_amplitudes))
        features['qrs_amplitude_std'] = float(np.std(qrs_amplitudes))
        
        # PR间期
        if 'ECG_P_Peaks' in waves and 'ECG_R_Peaks' in waves:
            p_peaks = waves['ECG_P_Peaks'][~np.isnan(waves['ECG_P_Peaks'])].astype(int)
            if len(p_peaks) > 0:
                pr_intervals = []
                for i, r in enumerate(r_peaks[:len(p_peaks)]):
                    if i < len(p_peaks):
                        pr = (r - p_peaks[i]) / sampling_rate * 1000
                        if 0 < pr < 300:  # 合理范围
                            pr_intervals.append(pr)
                
                if pr_intervals:
                    features['pr_interval_mean'] = float(np.mean(pr_intervals))
                    features['pr_interval_std'] = float(np.std(pr_intervals))
        
        # QT间期
        if 'ECG_T_Offsets' in waves:
            t_offsets = waves['ECG_T_Offsets'][~np.isnan(waves['ECG_T_Offsets'])].astype(int)
            if len(t_offsets) > 0:
                qt_intervals = []
                for i, r in enumerate(r_peaks[:len(t_offsets)]):
                    if i < len(t_offsets):
                        qt = (t_offsets[i] - r) / sampling_rate * 1000
                        if 0 < qt < 600:  # 合理范围
                            qt_intervals.append(qt)
                
                if qt_intervals:
                    features['qt_interval_mean'] = float(np.mean(qt_intervals))
                    features['qt_interval_std'] = float(np.std(qt_intervals))
        
        # QRS持续时间
        if 'ECG_R_Onsets' in waves and 'ECG_R_Offsets' in waves:
            r_onsets = waves['ECG_R_Onsets'][~np.isnan(waves['ECG_R_Onsets'])].astype(int)
            r_offsets = waves['ECG_R_Offsets'][~np.isnan(waves['ECG_R_Offsets'])].astype(int)
            
            if len(r_onsets) > 0 and len(r_offsets) > 0:
                qrs_durations = []
                for onset, offset in zip(r_onsets, r_offsets):
                    if offset > onset:
                        duration = (offset - onset) / sampling_rate * 1000
                        if 0 < duration < 200:  # 合理范围
                            qrs_durations.append(duration)
                
                if qrs_durations:
                    features['qrs_duration_mean'] = float(np.mean(qrs_durations))
                    features['qrs_duration_std'] = float(np.std(qrs_durations))
    
    except Exception as e:
        print(f"警告: 详细波形分析失败: {str(e)}")
        # 如果详细分析失败，至少保存基本的QRS振幅
        qrs_amplitudes = ecg[r_peaks]
        features['qrs_amplitude_mean'] = float(np.mean(qrs_amplitudes))
        features['qrs_amplitude_std'] = float(np.std(qrs_amplitudes))
    
    return features


# ========================= HRV时域特征 =========================

def extract_hrv_time_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    提取HRV时域统计特征
    
    包括:
    - SDNN: RR间隔标准差
    - RMSSD: 相邻RR差平方均值根
    - pNN50: 相邻RR差值超过50ms的比例
    - NN50: 相邻RR差值超过50ms的计数
    
    Parameters
    ----------
    rr_intervals : np.ndarray
        RR间期数组 (单位: ms)
    
    Returns
    -------
    features : dict
        HRV时域特征字典
    """
    features = {}
    
    if len(rr_intervals) < 2:
        print("警告: RR间期数量不足，跳过HRV时域特征")
        return features
    
    # SDNN - RR间隔标准差
    features['SDNN'] = float(np.std(rr_intervals, ddof=1))
    
    # 相邻RR间期差值
    diff_rr = np.diff(rr_intervals)
    
    # RMSSD - 相邻RR差平方均值根
    features['RMSSD'] = float(np.sqrt(np.mean(diff_rr ** 2)))
    
    # NN50 - 相邻RR差值超过50ms的计数
    nn50 = np.sum(np.abs(diff_rr) > 50)
    features['NN50'] = int(nn50)
    
    # pNN50 - NN50的百分比
    features['pNN50'] = float(nn50 / len(diff_rr) * 100) if len(diff_rr) > 0 else 0.0
    
    # 额外的时域指标
    features['mean_nn'] = float(np.mean(rr_intervals))
    features['median_nn'] = float(np.median(rr_intervals))
    features['range_nn'] = float(np.max(rr_intervals) - np.min(rr_intervals))
    features['cv_nn'] = float(np.std(rr_intervals) / np.mean(rr_intervals))  # 变异系数
    
    # SDSD - 相邻RR差值的标准差
    features['SDSD'] = float(np.std(diff_rr, ddof=1)) if len(diff_rr) > 1 else 0.0
    
    return features


# ========================= HRV频域特征 =========================

def extract_hrv_frequency_features(rr_intervals: np.ndarray,
                                   sampling_rate: float = 4.0) -> Dict[str, float]:
    """
    提取HRV频域特征
    
    包括:
    - VLF功率 (0.003-0.04 Hz): 极低频功率
    - LF功率 (0.04-0.15 Hz): 低频功率
    - HF功率 (0.15-0.4 Hz): 高频功率
    - LF/HF比: 交感/副交感神经平衡指标
    - 总功率
    - 归一化LF、HF功率
    
    Parameters
    ----------
    rr_intervals : np.ndarray
        RR间期数组 (单位: ms)
    sampling_rate : float
        重采样率，默认4Hz
    
    Returns
    -------
    features : dict
        HRV频域特征字典
    """
    features = {}
    
    if len(rr_intervals) < 10:
        print("警告: RR间期数量不足，跳过HRV频域特征")
        return features
    
    try:
        # 创建时间轴
        time_rr = np.cumsum(rr_intervals) / 1000  # 转换为秒
        time_rr = np.insert(time_rr, 0, 0)  # 添加起始点
        
        # 插值到均匀采样
        time_interp = np.arange(0, time_rr[-1], 1/sampling_rate)
        rr_interp = interp1d(time_rr, np.append(rr_intervals[0], rr_intervals), 
                            kind='cubic')(time_interp)
        
        # 去趋势
        rr_detrend = signal.detrend(rr_interp)
        
        # 计算功率谱密度 (Welch方法)
        nperseg = min(256, len(rr_detrend))
        freqs, psd = signal.welch(rr_detrend, fs=sampling_rate, 
                                 nperseg=nperseg, scaling='density')
        
        # 定义频段
        vlf_band = (0.003, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        
        # 计算各频段功率
        vlf_power = np.trapz(psd[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])], 
                            freqs[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
        
        lf_power = np.trapz(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])], 
                           freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
        
        hf_power = np.trapz(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])], 
                           freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
        
        total_power = vlf_power + lf_power + hf_power
        
        # 保存功率值
        features['VLF_power'] = float(vlf_power)
        features['LF_power'] = float(lf_power)
        features['HF_power'] = float(hf_power)
        features['total_power'] = float(total_power)
        
        # LF/HF比
        features['LF_HF_ratio'] = float(lf_power / hf_power) if hf_power > 0 else 0.0
        
        # 归一化功率
        if total_power > 0:
            features['LF_norm'] = float(lf_power / (total_power - vlf_power) * 100)
            features['HF_norm'] = float(hf_power / (total_power - vlf_power) * 100)
        else:
            features['LF_norm'] = 0.0
            features['HF_norm'] = 0.0
        
        # 峰值频率
        lf_peak_idx = np.argmax(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
        hf_peak_idx = np.argmax(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
        
        lf_freqs = freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])]
        hf_freqs = freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])]
        
        if len(lf_freqs) > 0:
            features['LF_peak_freq'] = float(lf_freqs[lf_peak_idx])
        if len(hf_freqs) > 0:
            features['HF_peak_freq'] = float(hf_freqs[hf_peak_idx])
    
    except Exception as e:
        print(f"警告: 频域分析失败: {str(e)}")
    
    return features


# ========================= 时频特征 =========================

def extract_time_frequency_features(ecg: np.ndarray,
                                    sampling_rate: float) -> Dict[str, Any]:
    """
    提取时频特征
    
    包括:
    - STFT能量分布统计
    - 小波能量分布
    - CWT scalogram特征
    
    Parameters
    ----------
    ecg : np.ndarray
        ECG信号
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
        nperseg = int(sampling_rate * 0.5)  # 0.5秒窗口
        f, t, Zxx = signal.stft(ecg, fs=sampling_rate, nperseg=nperseg)
        
        # STFT能量
        stft_energy = np.abs(Zxx) ** 2
        
        features['stft_mean_energy'] = float(np.mean(stft_energy))
        features['stft_max_energy'] = float(np.max(stft_energy))
        features['stft_energy_std'] = float(np.std(stft_energy))
        
        # 主频率成分
        freq_power = np.mean(stft_energy, axis=1)
        dominant_freq_idx = np.argmax(freq_power)
        features['stft_dominant_freq'] = float(f[dominant_freq_idx])
        
        # 2. 小波变换特征
        # 选择小波基
        wavelet = 'db4'
        level = min(5, pywt.dwt_max_level(len(ecg), wavelet))
        
        # 离散小波分解
        coeffs = pywt.wavedec(ecg, wavelet, level=level)
        
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
        
        # 3. 连续小波变换 (CWT)
        # 使用较少的尺度以提高速度
        scales = np.arange(1, min(128, len(ecg)//10))
        coefficients, frequencies = pywt.cwt(ecg, scales, wavelet, 
                                            sampling_period=1/sampling_rate)
        
        # CWT scalogram能量特征
        cwt_energy = np.abs(coefficients) ** 2
        
        features['cwt_mean_energy'] = float(np.mean(cwt_energy))
        features['cwt_max_energy'] = float(np.max(cwt_energy))
        features['cwt_energy_std'] = float(np.std(cwt_energy))
        
        # 能量集中度 (信息熵)
        cwt_energy_flat = cwt_energy.flatten()
        cwt_energy_norm = cwt_energy_flat / (np.sum(cwt_energy_flat) + 1e-10)
        cwt_entropy = -np.sum(cwt_energy_norm * np.log2(cwt_energy_norm + 1e-10))
        features['cwt_energy_entropy'] = float(cwt_entropy)
        
    except Exception as e:
        print(f"警告: 时频分析失败: {str(e)}")
    
    return features


# ========================= 非线性特征 =========================

def extract_nonlinear_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    提取非线性特征
    
    包括:
    - 样本熵 (Sample Entropy)
    - 近似熵 (Approximate Entropy)
    - DFA (Detrended Fluctuation Analysis)
    - Poincaré图参数 (SD1, SD2)
    
    Parameters
    ----------
    rr_intervals : np.ndarray
        RR间期数组 (单位: ms)
    
    Returns
    -------
    features : dict
        非线性特征字典
    """
    features = {}
    
    if len(rr_intervals) < 50:
        print("警告: RR间期数量不足，跳过非线性特征")
        return features
    
    try:
        # 1. 样本熵 (Sample Entropy)
        features['sample_entropy'] = float(sample_entropy(rr_intervals, m=2, r=0.2))
        
        # 2. 近似熵 (Approximate Entropy)
        features['approximate_entropy'] = float(approximate_entropy(rr_intervals, m=2, r=0.2))
        
        # 3. DFA (Detrended Fluctuation Analysis)
        dfa_alpha1, dfa_alpha2 = detrended_fluctuation_analysis(rr_intervals)
        features['DFA_alpha1'] = float(dfa_alpha1)  # 短期相关性
        features['DFA_alpha2'] = float(dfa_alpha2)  # 长期相关性
        
        # 4. Poincaré图参数
        sd1, sd2, sd_ratio = poincare_features(rr_intervals)
        features['poincare_SD1'] = float(sd1)  # 短期变异
        features['poincare_SD2'] = float(sd2)  # 长期变异
        features['poincare_SD_ratio'] = float(sd_ratio)  # SD1/SD2比
        
        # 额外的Poincaré特征
        features['poincare_ellipse_area'] = float(np.pi * sd1 * sd2)
        
    except Exception as e:
        print(f"警告: 非线性分析失败: {str(e)}")
    
    return features


def sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    计算样本熵
    
    Parameters
    ----------
    data : np.ndarray
        时间序列
    m : int
        嵌入维度
    r : float
        容差（相对于标准差的比例）
    
    Returns
    -------
    sampen : float
        样本熵值
    """
    N = len(data)
    
    # 标准化容差
    r = r * np.std(data, ddof=1)
    
    def _maxdist(xi, xj):
        return np.max(np.abs(xi - xj))
    
    def _phi(m):
        patterns = np.array([data[i:i+m] for i in range(N-m)])
        C = np.zeros(N-m)
        
        for i in range(N-m):
            # 计算与其他模式的距离
            distances = [_maxdist(patterns[i], patterns[j]) 
                        for j in range(N-m) if i != j]
            # 计算在容差内的比例
            C[i] = np.sum(np.array(distances) <= r)
        
        return np.sum(C) / (N-m) / (N-m-1)
    
    phi_m = _phi(m)
    phi_m1 = _phi(m+1)
    
    if phi_m == 0 or phi_m1 == 0:
        return 0.0
    
    sampen = -np.log(phi_m1 / phi_m)
    return sampen


def approximate_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    计算近似熵
    
    Parameters
    ----------
    data : np.ndarray
        时间序列
    m : int
        嵌入维度
    r : float
        容差（相对于标准差的比例）
    
    Returns
    -------
    apen : float
        近似熵值
    """
    N = len(data)
    r = r * np.std(data, ddof=1)
    
    def _phi(m):
        patterns = np.array([data[i:i+m] for i in range(N-m+1)])
        C = np.zeros(N-m+1)
        
        for i in range(N-m+1):
            distances = np.max(np.abs(patterns - patterns[i]), axis=1)
            C[i] = np.sum(distances <= r) / (N-m+1)
        
        phi = np.sum(np.log(C)) / (N-m+1)
        return phi
    
    return _phi(m) - _phi(m+1)


def detrended_fluctuation_analysis(data: np.ndarray) -> Tuple[float, float]:
    """
    去趋势波动分析 (DFA)
    
    Parameters
    ----------
    data : np.ndarray
        RR间期序列
    
    Returns
    -------
    alpha1 : float
        短期DFA指数 (4-16个心跳)
    alpha2 : float
        长期DFA指数 (16-64个心跳)
    """
    N = len(data)
    
    # 去均值并累加
    y = np.cumsum(data - np.mean(data))
    
    # 定义窗口大小
    scales = np.unique(np.logspace(0.5, 2, 20).astype(int))
    scales = scales[scales < N//4]
    
    F = []
    
    for n in scales:
        # 分段
        n_segments = N // n
        
        # 每段进行去趋势
        F_n = []
        for i in range(n_segments):
            segment = y[i*n:(i+1)*n]
            # 线性拟合
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            fit = np.polyval(coeffs, x)
            # 计算波动
            F_n.append(np.sqrt(np.mean((segment - fit)**2)))
        
        F.append(np.mean(F_n))
    
    F = np.array(F)
    scales = scales[:len(F)]
    
    # 在对数坐标下拟合
    log_scales = np.log(scales)
    log_F = np.log(F)
    
    # 短期和长期分别拟合
    short_idx = scales <= 16
    long_idx = scales > 16
    
    if np.sum(short_idx) > 2:
        alpha1 = np.polyfit(log_scales[short_idx], log_F[short_idx], 1)[0]
    else:
        alpha1 = 1.0
    
    if np.sum(long_idx) > 2:
        alpha2 = np.polyfit(log_scales[long_idx], log_F[long_idx], 1)[0]
    else:
        alpha2 = 1.0
    
    return alpha1, alpha2


def poincare_features(rr_intervals: np.ndarray) -> Tuple[float, float, float]:
    """
    计算Poincaré图参数
    
    Parameters
    ----------
    rr_intervals : np.ndarray
        RR间期序列
    
    Returns
    -------
    SD1 : float
        短期变异（椭圆短轴标准差）
    SD2 : float
        长期变异（椭圆长轴标准差）
    SD_ratio : float
        SD1/SD2比值
    """
    # 创建Poincaré图点对
    rr1 = rr_intervals[:-1]
    rr2 = rr_intervals[1:]
    
    # 计算SD1和SD2
    diff_rr = rr2 - rr1
    sum_rr = rr2 + rr1
    
    SD1 = np.std(diff_rr, ddof=1) / np.sqrt(2)
    SD2 = np.std(sum_rr, ddof=1) / np.sqrt(2)
    
    SD_ratio = SD1 / SD2 if SD2 > 0 else 0.0
    
    return SD1, SD2, SD_ratio


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
    
    # 形态学特征
    if 'morphological' in features and features['morphological']:
        print("\n[时域形态学特征]")
        morph = features['morphological']
        if 'mean_hr' in morph:
            print(f"  平均心率: {morph['mean_hr']:.1f} bpm")
        if 'qrs_amplitude_mean' in morph:
            print(f"  QRS平均振幅: {morph['qrs_amplitude_mean']:.3f}")
        if 'mean_rr_interval' in morph:
            print(f"  平均RR间期: {morph['mean_rr_interval']:.1f} ms")
    
    # HRV时域特征
    if 'hrv_time' in features and features['hrv_time']:
        print("\n[HRV时域特征]")
        hrv_t = features['hrv_time']
        if 'SDNN' in hrv_t:
            print(f"  SDNN: {hrv_t['SDNN']:.2f} ms")
        if 'RMSSD' in hrv_t:
            print(f"  RMSSD: {hrv_t['RMSSD']:.2f} ms")
        if 'pNN50' in hrv_t:
            print(f"  pNN50: {hrv_t['pNN50']:.2f}%")
    
    # HRV频域特征
    if 'hrv_frequency' in features and features['hrv_frequency']:
        print("\n[HRV频域特征]")
        hrv_f = features['hrv_frequency']
        if 'LF_power' in hrv_f:
            print(f"  LF功率: {hrv_f['LF_power']:.2f}")
        if 'HF_power' in hrv_f:
            print(f"  HF功率: {hrv_f['HF_power']:.2f}")
        if 'LF_HF_ratio' in hrv_f:
            print(f"  LF/HF比: {hrv_f['LF_HF_ratio']:.2f}")
    
    # 非线性特征
    if 'nonlinear' in features and features['nonlinear']:
        print("\n[非线性特征]")
        nonlin = features['nonlinear']
        if 'sample_entropy' in nonlin:
            print(f"  样本熵: {nonlin['sample_entropy']:.3f}")
        if 'DFA_alpha1' in nonlin:
            print(f"  DFA α1: {nonlin['DFA_alpha1']:.3f}")
        if 'poincare_SD1' in nonlin:
            print(f"  Poincaré SD1: {nonlin['poincare_SD1']:.2f}")
    
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
    
    data_dict['processed']['ecg_features'] = features
    
    return data_dict


# ========================= 命令行接口 =========================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print("运行演示示例...\n")
        
        # 创建示例ECG数据
        sampling_rate = 500  # Hz
        duration = 60  # 秒
        t = np.arange(0, duration, 1/sampling_rate)
        
        # 生成模拟ECG信号（简单的合成信号）
        heart_rate = 75  # bpm
        ecg_signal = np.zeros_like(t)
        
        # 添加QRS波群
        beat_interval = 60 / heart_rate
        for i in range(int(duration / beat_interval)):
            peak_time = i * beat_interval
            peak_idx = int(peak_time * sampling_rate)
            
            if peak_idx < len(t):
                # R波
                width = int(0.02 * sampling_rate)
                if peak_idx + width < len(t):
                    ecg_signal[peak_idx:peak_idx+width] = 1.0
        
        # 添加噪声
        ecg_signal += np.random.normal(0, 0.05, len(t))
        
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
        
        # 提取特征
        features = extract_ecg_features(demo_data)
        
        # 显示摘要
        summarize_features(features)
        
        # 保存示例
        demo_data = add_features_to_data_dict(demo_data, features)
        
        print("演示完成!")
        print("\n使用方法:")
        print("  from ecg_features import extract_ecg_features")
        print("  features = extract_ecg_features(your_data_dict)")
        
    else:
        print("使用方法:")
        print("  python ecg_features.py --demo  # 运行演示")
        print("\n在代码中使用:")
        print("  from ecg_features import extract_ecg_features")
        print("  features = extract_ecg_features(data_dict)")
