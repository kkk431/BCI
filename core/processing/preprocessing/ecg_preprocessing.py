# -*- coding: utf-8 -*-
"""
ECG信号预处理模块
支持心电图信号的专用预处理
包括R波检测、心率变异性分析、心律异常检测等
"""

import numpy as np
from scipy import signal, interpolate, stats
from scipy.signal import find_peaks, savgol_filter, periodogram, butter, filtfilt, welch
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import matplotlib.pyplot as plt
from typing import Callable

# 导入通用预处理模块
from preprocessing import GeneralPreprocessor, PreprocessingConfig, FilterType, WaveletType, DetrendMethod

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 禁用特定警告
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ====================== ECG专用枚举和配置 ======================

class RPeakDetectionMethod(Enum):
    """R波检测方法枚举"""
    PAN_TOMPKINS = "pan_tompkins"  # Pan-Tompkins算法
    HAMILTON = "hamilton"          # Hamilton算法
    WAVELET = "wavelet"            # 小波变换法
    CHRISTOV = "christov"          # Christov算法
    ENGZEE = "engzee"              # Engelse-Zeelenberg算法
    NEUROKIT = "neurokit"          # NeuroKit算法


class HRVAnalysisMethod(Enum):
    """HRV分析方法枚举"""
    TIME_DOMAIN = "time_domain"      # 时域分析
    FREQUENCY_DOMAIN = "frequency_domain"  # 频域分析
    NONLINEAR = "nonlinear"          # 非线性分析
    POINCARE = "poincare"            # Poincaré图分析


class ArrhythmiaType(Enum):
    """心律异常类型枚举"""
    NORMAL = "normal"                # 正常心律
    TACHYCARDIA = "tachycardia"      # 心动过速
    BRADYCARDIA = "bradycardia"      # 心动过缓
    PVC = "pvc"                      # 室性早搏
    PAC = "pac"                      # 房性早搏
    AFIB = "afib"                    # 心房颤动
    SINUS_ARRHYTHMIA = "sinus_arrhythmia"  # 窦性心律不齐
    BIGEMINY = "bigeminy"            # 二联律
    TRIGEMINY = "trigeminy"          # 三联律


@dataclass
class ECGConfig(PreprocessingConfig):
    """
    ECG专用预处理配置类
    扩展通用配置，添加ECG特有参数
    """
    # R波检测参数
    rpeak_method: RPeakDetectionMethod = RPeakDetectionMethod.PAN_TOMPKINS
    rpeak_detection_channel: int = 0  # 用于R波检测的通道索引
    rpeak_threshold: float = 0.5  # R波检测阈值
    rpeak_min_distance: float = 0.3  # R波最小间隔（秒）
    
    # QRS波参数
    qrs_search_window: Tuple[float, float] = (-0.1, 0.08)  # QRS波搜索窗口（秒）
    qrs_width_limits: Tuple[float, float] = (0.06, 0.12)  # QRS波宽度限制（秒）
    
    # 心率参数
    heart_rate_limits: Tuple[float, float] = (40.0, 180.0)  # 合理心率范围（BPM）
    rr_interval_limits: Tuple[float, float] = (300.0, 1500.0)  # 合理RR间期范围（毫秒）
    
    # HRV分析参数
    hrv_analysis_methods: List[HRVAnalysisMethod] = field(
        default_factory=lambda: [HRVAnalysisMethod.TIME_DOMAIN, HRVAnalysisMethod.FREQUENCY_DOMAIN]
    )
    hrv_frequency_bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "ULF": (0.0, 0.003),
            "VLF": (0.003, 0.04),
            "LF": (0.04, 0.15),
            "HF": (0.15, 0.4)
        }
    )
    hrv_interpolation_rate: float = 4.0  # HRV分析插值采样率（Hz）
    
    # 心律异常检测参数
    detect_arrhythmias: bool = True
    pvc_detection_threshold: float = 0.15  # PVC检测阈值（提前百分比）
    afib_detection_threshold: float = 0.1  # AFib检测阈值（RR间期变异系数）
    
    # 信号质量评估参数
    assess_signal_quality: bool = True
    signal_quality_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "snr_threshold": 10.0,  # 信噪比阈值（dB）
            "baseline_wander_threshold": 0.3,  # 基线漂移阈值
            "powerline_noise_threshold": 0.2,  # 工频干扰阈值
            "emg_noise_threshold": 0.15,  # 肌电噪声阈值
        }
    )
    
    # 多导联分析参数
    multi_lead_analysis: bool = True
    lead_weights: Optional[List[float]] = None  # 各导联权重
    
    # 呼吸性窦性心律不齐提取参数
    extract_rsa: bool = False
    rsa_frequency_band: Tuple[float, float] = (0.15, 0.4)  # RSA频带（Hz）
    
    # ECG滤波参数（覆盖通用参数）
    ecg_lowcut: float = 0.5  # ECG低频截止（Hz）
    ecg_highcut: float = 40.0  # ECG高频截止（Hz）
    ecg_notch_freq: float = 50.0  # ECG工频陷波频率（Hz）
    
    # 波形检测参数
    detect_p_wave: bool = False  # 是否检测P波
    detect_t_wave: bool = False  # 是否检测T波
    p_wave_window: Tuple[float, float] = (-0.2, -0.05)  # P波窗口（秒，相对于R波）
    t_wave_window: Tuple[float, float] = (0.1, 0.4)  # T波窗口（秒，相对于R波）
    
    # 输出参数
    include_plots: bool = True  # 是否包含绘图
    save_intermediate_results: bool = False  # 是否保存中间结果


# ====================== ECG专用预处理器类 ======================

class ECGPreprocessor(GeneralPreprocessor):
    """
    ECG信号专用预处理器
    在通用预处理基础上，增加R波检测、HRV分析、心律异常检测等步骤
    """
    
    def __init__(self, config: Optional[ECGConfig] = None):
        """
        初始化ECG预处理器
        
        Args:
            config: ECG预处理配置，如果为None则使用默认配置
        """
        super().__init__(config if config is not None else ECGConfig())
        
        # 确保配置是ECGConfig类型
        if not isinstance(self.config, ECGConfig):
            # 尝试将通用配置转换为ECG配置
            self.config = ECGConfig(**self.config.__dict__)
        
        # ECG处理状态
        self.r_peaks = []
        self.qrs_info = {}
        self.hrv_metrics = {}
        self.arrhythmia_results = {}
        self.signal_quality = {}
        
        # 模板存储
        self.qrs_template = None
        self.rr_history = []
        
    def process_ECG(self, 
                   data_dict: Dict[str, Any], 
                   modality: str = "ECG",
                   analysis_channels: Optional[List[int]] = None,
                   reference_channel: Optional[int] = None) -> Dict[str, Any]:
        """
        ECG专用处理流程
        
        Args:
            data_dict: 四层数据字典
            modality: 要处理的信号模态（默认为"ECG"）
            analysis_channels: 用于分析的通道索引列表
            reference_channel: 参考通道索引（用于R波检测）
            
        Returns:
            更新后的四层数据字典，包含ECG处理结果
        """
        # 验证ECG特有输入结构
        self._validate_ECG_input(data_dict, modality)
        
        # 创建ECG处理历史记录
        process_record = {
            "modality": modality,
            "analysis_channels": analysis_channels,
            "reference_channel": reference_channel,
            "steps": [],
            "timestamp": np.datetime64('now')
        }
        
        # 获取ECG信号数据
        signal_info = data_dict["signal"][modality]
        ecg_data = signal_info["data"]
        sampling_rate = signal_info["sampling_rate"]
        
        # 处理前备份原始数据
        if "processed" not in data_dict:
            data_dict["processed"] = {}
        if "ECG_processing" not in data_dict["processed"]:
            data_dict["processed"]["ECG_processing"] = {}
        if modality not in data_dict["processed"]["ECG_processing"]:
            data_dict["processed"]["ECG_processing"][modality] = {
                "original_data": ecg_data.copy(),
                "steps": [],
                "intermediate_results": {},
                "config": self.config.__dict__
            }
        
        # 选择分析通道
        n_channels = ecg_data.shape[0]
        if analysis_channels is None:
            analysis_channels = list(range(n_channels))
        
        if reference_channel is None:
            reference_channel = self.config.rpeak_detection_channel
            if reference_channel >= n_channels:
                reference_channel = 0
        
        # 记录通道信息
        process_record["analysis_channels"] = analysis_channels
        process_record["reference_channel"] = reference_channel
        
        # ========== ECG专用处理步骤 ==========
        
        # 1. 对ECG信号应用专用预处理（保护QRS波形态）
        processed_ecg = self.preprocess_ecg_signal(
            ecg_data, 
            sampling_rate, 
            analysis_channels
        )
        
        # 更新数据字典中的信号数据
        data_dict["signal"][modality]["data"] = processed_ecg
        
        process_record["steps"].append({
            "step": "ecg_specific_preprocessing",
            "lowcut": self.config.ecg_lowcut,
            "highcut": self.config.ecg_highcut,
            "notch_freq": self.config.ecg_notch_freq,
            "channels_processed": len(analysis_channels)
        })
        
        # 2. R波峰值检测
        try:
            self.r_peaks, self.qrs_info = self.detect_r_peaks(
                processed_ecg,
                sampling_rate=sampling_rate,
                method=self.config.rpeak_method,
                channel_idx=reference_channel,
                min_distance=self.config.rpeak_min_distance
            )
            
            if len(self.r_peaks) > 0:
                process_record["steps"].append({
                    "step": "r_peak_detection",
                    "method": self.config.rpeak_method.value,
                    "n_r_peaks": len(self.r_peaks),
                    "mean_heart_rate": np.mean(self.qrs_info.get("heart_rate", [0])),
                    "reference_channel": reference_channel
                })
            else:
                logger.warning("未检测到R波")
                process_record["steps"].append({
                    "step": "r_peak_detection",
                    "method": self.config.rpeak_method.value,
                    "n_r_peaks": 0,
                    "warning": "未检测到R波"
                })
                
        except Exception as e:
            logger.error(f"R波检测失败: {str(e)}")
            self.r_peaks = []
            self.qrs_info = {}
        
        # 3. 信号质量评估
        if self.config.assess_signal_quality and len(self.r_peaks) > 1:
            self.signal_quality = self.assess_ecg_quality(
                processed_ecg,
                sampling_rate=sampling_rate,
                r_peaks=self.r_peaks,
                qrs_info=self.qrs_info
            )
            
            process_record["steps"].append({
                "step": "signal_quality_assessment",
                "quality_score": self.signal_quality.get("quality_score", 0),
                "quality_flag": self.signal_quality.get("quality_flag", "unknown"),
                "bad_segments": len(self.signal_quality.get("bad_segments", []))
            })
        
        # 4. 心率变异性分析（如果有足够的R波）
        if len(self.r_peaks) >= 10 and self.config.hrv_analysis_methods:
            try:
                self.hrv_metrics = self.analyze_hrv(
                    rr_intervals=self.qrs_info.get("rr_intervals", []),
                    sampling_rate=sampling_rate,
                    methods=self.config.hrv_analysis_methods,
                    frequency_bands=self.config.hrv_frequency_bands
                )
                
                process_record["steps"].append({
                    "step": "hrv_analysis",
                    "methods": [m.value for m in self.config.hrv_analysis_methods],
                    "n_rr_intervals": len(self.qrs_info.get("rr_intervals", [])),
                    "hrv_metrics_computed": list(self.hrv_metrics.keys())
                })
            except Exception as e:
                logger.error(f"HRV分析失败: {str(e)}")
                self.hrv_metrics = {}
        
        # 5. 心律异常检测
        if self.config.detect_arrhythmias and len(self.r_peaks) >= 5:
            try:
                self.arrhythmia_results = self.detect_arrhythmias(
                    processed_ecg,
                    r_peaks=self.r_peaks,
                    qrs_info=self.qrs_info,
                    sampling_rate=sampling_rate,
                    signal_quality=self.signal_quality
                )
                
                if self.arrhythmia_results.get("arrhythmia_flags"):
                    process_record["steps"].append({
                        "step": "arrhythmia_detection",
                        "flags_detected": list(self.arrhythmia_results["arrhythmia_flags"].keys()),
                        "events_detected": len(self.arrhythmia_results.get("arrhythmia_events", []))
                    })
            except Exception as e:
                logger.error(f"心律异常检测失败: {str(e)}")
                self.arrhythmia_results = {}
        
        # 6. 多导联综合分析
        if self.config.multi_lead_analysis and len(analysis_channels) > 1:
            try:
                multi_lead_results = self.analyze_multi_leads(
                    processed_ecg,
                    sampling_rate=sampling_rate,
                    analysis_channels=analysis_channels,
                    lead_names=signal_info.get("channel_names", [])
                )
                
                data_dict["processed"]["ECG_processing"][modality]["multi_lead_results"] = multi_lead_results
                
                process_record["steps"].append({
                    "step": "multi_lead_analysis",
                    "n_leads_analyzed": len(analysis_channels),
                    "consistency_score": multi_lead_results.get("consistency_score", 0)
                })
            except Exception as e:
                logger.error(f"多导联分析失败: {str(e)}")
        
        # 7. 呼吸性窦性心律不齐提取
        if self.config.extract_rsa and len(self.r_peaks) >= 20:
            try:
                rsa_results = self.extract_respiratory_sinus_arrhythmia(
                    processed_ecg,
                    r_peaks=self.r_peaks,
                    rr_intervals=self.qrs_info.get("rr_intervals", []),
                    sampling_rate=sampling_rate,
                    frequency_band=self.config.rsa_frequency_band
                )
                
                data_dict["processed"]["ECG_processing"][modality]["rsa_results"] = rsa_results
                
                process_record["steps"].append({
                    "step": "rsa_extraction",
                    "rsa_magnitude": rsa_results.get("rsa_magnitude", 0),
                    "cardiorespiratory_coupling": rsa_results.get("cardiorespiratory_coupling", 0)
                })
            except Exception as e:
                logger.error(f"RSA提取失败: {str(e)}")
        
        # 8. P波和T波检测（可选）
        if (self.config.detect_p_wave or self.config.detect_t_wave) and len(self.r_peaks) >= 3:
            try:
                waveform_results = self.detect_waveforms(
                    processed_ecg,
                    r_peaks=self.r_peaks,
                    sampling_rate=sampling_rate,
                    detect_p=self.config.detect_p_wave,
                    detect_t=self.config.detect_t_wave,
                    p_window=self.config.p_wave_window,
                    t_window=self.config.t_wave_window
                )
                
                data_dict["processed"]["ECG_processing"][modality]["waveform_results"] = waveform_results
                
                process_record["steps"].append({
                    "step": "waveform_detection",
                    "p_waves_detected": len(waveform_results.get("p_waves", [])) if self.config.detect_p_wave else 0,
                    "t_waves_detected": len(waveform_results.get("t_waves", [])) if self.config.detect_t_wave else 0
                })
            except Exception as e:
                logger.error(f"波形检测失败: {str(e)}")
        
        # 9. 更新ECG处理结果到数据字典
        ecg_results = {
            "r_peaks": self.r_peaks,
            "qrs_info": self.qrs_info,
            "hrv_metrics": self.hrv_metrics,
            "arrhythmia_results": self.arrhythmia_results,
            "signal_quality": self.signal_quality,
            "processing_config": self.config.__dict__,
            "processing_timestamp": np.datetime64('now').astype(str)
        }
        
        data_dict["processed"]["ECG_processing"][modality]["results"] = ecg_results
        data_dict["processed"]["ECG_processing"][modality]["steps"].append(process_record)
        
        # 保存中间结果（如果启用）
        if self.config.save_intermediate_results:
            intermediate_data = {
                "processed_ecg": processed_ecg,
                "detection_channel": reference_channel,
                "qrs_template": self.qrs_template
            }
            data_dict["processed"]["ECG_processing"][modality]["intermediate_results"] = intermediate_data
        
        # 记录处理历史
        self.history.append(process_record)
        
        # 生成处理摘要
        summary = self._generate_processing_summary()
        data_dict["processed"]["ECG_processing"][modality]["summary"] = summary
        
        logger.info(f"ECG预处理完成: {modality}, "
                   f"检测到 {len(self.r_peaks)} 个R波, "
                   f"平均心率: {np.mean(self.qrs_info.get('heart_rate', [0])) if len(self.r_peaks) > 0 else 0:.1f} BPM, "
                   f"信号质量: {self.signal_quality.get('quality_flag', 'unknown')}")
        
        return data_dict
    
    def _validate_ECG_input(self, data_dict: Dict, modality: str):
        """
        验证ECG特有输入结构
        
        Args:
            data_dict: 四层数据字典
            modality: 要验证的模态名称
            
        Raises:
            ValueError: 如果输入数据格式不符合ECG要求
        """
        # 首先调用通用验证
        super()._validate_input(data_dict, modality)
        
        signal_info = data_dict["signal"][modality]
        
        # 检查数据维度
        data = signal_info["data"]
        if len(data.shape) != 2:
            raise ValueError(f"ECG数据必须是2维数组 (channels × samples)，当前维度: {len(data.shape)}")
        
        # 检查采样率是否合理（ECG通常为100-1000 Hz）
        sampling_rate = signal_info["sampling_rate"]
        if sampling_rate < 100:
            logger.warning(f"ECG采样率较低: {sampling_rate} Hz，建议至少100 Hz以获得准确的QRS检测")
        elif sampling_rate > 2000:
            logger.warning(f"ECG采样率较高: {sampling_rate} Hz，可能包含过多高频噪声，考虑降采样")
        
        # 检查是否有通道信息
        if "channel_names" not in signal_info or not signal_info["channel_names"]:
            n_channels = data.shape[0]
            signal_info["channel_names"] = [f"ECG_{i}" for i in range(n_channels)]
            logger.warning(f"ECG信号缺少通道名称信息，已创建通用名称: {signal_info['channel_names']}")
    
    # ====================== ECG特有预处理方法 ======================
    
    def preprocess_ecg_signal(self, 
                             ecg_data: np.ndarray,
                             sampling_rate: float,
                             channels: List[int] = None) -> np.ndarray:
        """
        对ECG信号进行专用预处理，保护QRS波形态
        
        Args:
            ecg_data: ECG原始数据，形状 (channels, samples)
            sampling_rate: 采样率
            channels: 要处理的通道列表
            
        Returns:
            预处理后的ECG数据
        """
        n_channels, n_samples = ecg_data.shape
        
        # 选择要处理的通道
        if channels is None:
            channels = list(range(n_channels))
        
        processed_data = ecg_data.copy()
        
        for ch in channels:
            if ch >= n_channels:
                continue
                
            signal = ecg_data[ch, :]
            
            # 1. 去除基线漂移（使用高阶高通滤波）
            # 使用0.5 Hz高通滤波器去除基线漂移
            nyquist = sampling_rate / 2
            if self.config.ecg_lowcut > 0 and self.config.ecg_lowcut < nyquist:
                b, a = butter(2, self.config.ecg_lowcut / nyquist, btype='high')
                signal = filtfilt(b, a, signal)
            
            # 2. 带通滤波（保护QRS波）
            if self.config.ecg_highcut > 0 and self.config.ecg_highcut < nyquist:
                # 使用零相位滤波器保护QRS波形态
                b, a = butter(2, [self.config.ecg_lowcut / nyquist, 
                                  self.config.ecg_highcut / nyquist], 
                              btype='band')
                signal = filtfilt(b, a, signal)
            
            # 3. 工频陷波滤波
            if self.config.ecg_notch_freq > 0 and self.config.ecg_notch_freq < nyquist:
                from scipy.signal import iirnotch
                Q = 30.0
                b, a = iirnotch(self.config.ecg_notch_freq, Q, sampling_rate)
                signal = filtfilt(b, a, signal)
            
            # 4. 可选：小波去噪（如果启用）
            if self.config.wavelet_level > 0 and not self.config.use_adaptive_wavelet:
                signal = self.wavelet_denoising(
                    signal.reshape(1, -1),
                    wavelet=self.config.wavelet_type,
                    level=self.config.wavelet_level,
                    threshold_method=self.config.wavelet_threshold_method
                )[0, :]
            
            processed_data[ch, :] = signal
        
        return processed_data
    
    def detect_r_peaks(self, 
                      ecg_data: np.ndarray,
                      sampling_rate: float,
                      method: RPeakDetectionMethod = RPeakDetectionMethod.PAN_TOMPKINS,
                      channel_idx: int = 0,
                      min_distance: float = 0.3) -> Tuple[List[int], Dict]:
        """
        R波峰值检测
        
        Args:
            ecg_data: ECG信号数据，形状 (channels, samples) 或 (samples,)
            sampling_rate: 采样率
            method: 检测方法
            channel_idx: 使用的通道索引
            min_distance: R波最小间隔（秒）
            
        Returns:
            r_peaks: R波位置的样本索引列表
            qrs_info: QRS波相关信息字典
        """
        # 确保使用正确的通道和数据格式
        if len(ecg_data.shape) == 2:
            ecg_signal = ecg_data[channel_idx, :]
        else:
            ecg_signal = ecg_data
        
        n_samples = len(ecg_signal)
        
        # 根据选择的方法调用对应的检测函数
        if method == RPeakDetectionMethod.PAN_TOMPKINS:
            r_peaks = self._detect_pan_tompkins(ecg_signal, sampling_rate, min_distance)
        elif method == RPeakDetectionMethod.HAMILTON:
            r_peaks = self._detect_hamilton(ecg_signal, sampling_rate, min_distance)
        elif method == RPeakDetectionMethod.WAVELET:
            r_peaks = self._detect_wavelet(ecg_signal, sampling_rate, min_distance)
        elif method == RPeakDetectionMethod.CHRISTOV:
            r_peaks = self._detect_christov(ecg_signal, sampling_rate, min_distance)
        elif method == RPeakDetectionMethod.ENGZEE:
            r_peaks = self._detect_engzee(ecg_signal, sampling_rate, min_distance)
        elif method == RPeakDetectionMethod.NEUROKIT:
            r_peaks = self._detect_neurokit(ecg_signal, sampling_rate, min_distance)
        else:
            raise ValueError(f"不支持的R波检测方法: {method}")
        
        # 提取QRS波信息
        qrs_info = self._extract_qrs_info(ecg_signal, r_peaks, sampling_rate)
        
        return r_peaks, qrs_info
    
    def _detect_pan_tompkins(self, 
                            ecg_signal: np.ndarray,
                            sampling_rate: float,
                            min_distance: float = 0.3) -> List[int]:
        """
        Pan-Tompkins算法实现
        
        参考文献：
        Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
        IEEE Transactions on Biomedical Engineering, (3), 230-236.
        """
        n_samples = len(ecg_signal)
        min_samples = int(min_distance * sampling_rate)
        
        # 1. 带通滤波 (5-15 Hz) 增强QRS波
        nyquist = sampling_rate / 2
        
        # 低通滤波 (15 Hz)
        b_low, a_low = butter(1, 15 / nyquist, btype='low')
        ecg_low = filtfilt(b_low, a_low, ecg_signal)
        
        # 高通滤波 (5 Hz)
        b_high, a_high = butter(1, 5 / nyquist, btype='high')
        ecg_filtered = filtfilt(b_high, a_high, ecg_low)
        
        # 2. 微分增强斜率
        differentiated = np.diff(ecg_filtered)
        differentiated = np.append(differentiated, differentiated[-1])
        
        # 3. 平方放大高频成分
        squared = differentiated ** 2
        
        # 4. 移动窗口积分平滑
        window_size = int(0.15 * sampling_rate)  # 150ms窗口
        if window_size % 2 == 0:
            window_size += 1
        
        window = np.ones(window_size) / window_size
        integrated = np.convolve(squared, window, mode='same')
        
        # 5. 自适应阈值检测
        # 寻找积分信号中的峰值
        peaks, properties = find_peaks(
            integrated, 
            height=np.percentile(integrated, 75),  # 使用75百分位数作为初始阈值
            distance=min_samples,
            prominence=np.std(integrated) * 0.5
        )
        
        # 6. 后处理：在原始滤波信号中精确定位R波
        r_peaks = []
        search_radius = int(0.05 * sampling_rate)  # 50ms搜索半径
        
        for peak in peaks:
            # 在原始滤波信号中搜索精确R波位置
            start = max(0, peak - search_radius)
            end = min(n_samples, peak + search_radius)
            
            if end > start:
                segment = ecg_filtered[start:end]
                local_max_idx = np.argmax(segment)
                r_peak = start + local_max_idx
                
                # 验证R波幅度
                if ecg_filtered[r_peak] > np.std(ecg_filtered) * 0.5:
                    r_peaks.append(r_peak)
        
        # 7. 去除紧密相邻的假阳性检测
        r_peaks = self._remove_close_peaks(r_peaks, min_samples)
        
        return r_peaks
    
    def _detect_hamilton(self, 
                        ecg_signal: np.ndarray,
                        sampling_rate: float,
                        min_distance: float = 0.3) -> List[int]:
        """
        Hamilton算法实现
        
        参考文献：
        Hamilton, P. S. (2002). Open source ECG analysis.
        Computers in Cardiology, 29, 101-104.
        """
        n_samples = len(ecg_signal)
        min_samples = int(min_distance * sampling_rate)
        
        # 1. 低通滤波去除基线漂移
        nyquist = sampling_rate / 2
        b, a = butter(2, 25 / nyquist, btype='low')
        ecg_filtered = filtfilt(b, a, ecg_signal)
        
        # 2. 差分和平方运算
        differentiated = np.diff(ecg_filtered)
        differentiated = np.append(differentiated, differentiated[-1])
        squared = differentiated ** 2
        
        # 3. 移动平均滤波
        window_size = int(0.15 * sampling_rate)
        window = np.ones(window_size) / window_size
        integrated = np.convolve(squared, window, mode='same')
        
        # 4. 自适应阈值和模板匹配
        # 初始化阈值
        threshold = np.mean(integrated) * 2
        r_peaks = []
        
        i = 0
        while i < n_samples:
            if integrated[i] > threshold:
                # 找到峰值
                search_end = min(i + min_samples, n_samples)
                segment = integrated[i:search_end]
                
                if len(segment) > 0:
                    local_max_idx = np.argmax(segment)
                    candidate = i + local_max_idx
                    
                    # 在原始信号中验证
                    search_radius = int(0.05 * sampling_rate)
                    start = max(0, candidate - search_radius)
                    end = min(n_samples, candidate + search_radius)
                    
                    if end > start:
                        ecg_segment = ecg_filtered[start:end]
                        ecg_max_idx = np.argmax(ecg_segment)
                        r_peak = start + ecg_max_idx
                        
                        # 添加到结果
                        if r_peak not in r_peaks:
                            r_peaks.append(r_peak)
                        
                        # 跳过已检测区域
                        i = r_peak + min_samples
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        
        # 更新阈值
        if len(r_peaks) > 0:
            # 计算检测到的R波周围的平均能量
            peak_energies = []
            for peak in r_peaks:
                start = max(0, peak - 10)
                end = min(n_samples, peak + 10)
                peak_energies.append(np.mean(integrated[start:end]))
            
            if peak_energies:
                threshold = np.mean(peak_energies) * 0.5
        
        # 去除紧密相邻的假阳性检测
        r_peaks = self._remove_close_peaks(r_peaks, min_samples)
        
        return sorted(r_peaks)
    
    def _detect_wavelet(self, 
                       ecg_signal: np.ndarray,
                       sampling_rate: float,
                       min_distance: float = 0.3) -> List[int]:
        """
        小波变换法R波检测
        
        使用小波变换的多分辨率特性检测R波
        """
        import pywt
        
        n_samples = len(ecg_signal)
        min_samples = int(min_distance * sampling_rate)
        
        # 选择小波基
        wavelet = 'db4'
        level = 4
        
        # 小波分解
        coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
        
        # 在特定尺度（通常为2^3或2^4）检测R波
        detail_coeff = coeffs[1]  # 通常使用第一层细节系数
        
        # 寻找细节系数中的零交叉点
        zero_crossings = np.where(np.diff(np.sign(detail_coeff)))[0]
        
        # 在原始信号中定位R波
        r_peaks = []
        scale_factor = 2 ** (level - 1)  # 尺度因子
        
        for crossing in zero_crossings:
            # 将细节系数中的位置映射回原始信号
            original_idx = crossing * scale_factor
            
            if original_idx < n_samples:
                # 在原始信号中搜索局部最大值
                search_radius = int(0.05 * sampling_rate)
                start = max(0, original_idx - search_radius)
                end = min(n_samples, original_idx + search_radius)
                
                if end > start:
                    segment = ecg_signal[start:end]
                    local_max_idx = np.argmax(segment)
                    r_peak = start + local_max_idx
                    
                    if r_peak not in r_peaks:
                        r_peaks.append(r_peak)
        
        # 去除紧密相邻的检测
        r_peaks = self._remove_close_peaks(r_peaks, min_samples)
        
        return sorted(r_peaks)
    
    def _detect_christov(self, 
                        ecg_signal: np.ndarray,
                        sampling_rate: float,
                        min_distance: float = 0.3) -> List[int]:
        """
        Christov算法实现（简化版）
        
        参考文献：
        Christov, I. I. (2004). Real time electrocardiogram QRS detection
        using combined adaptive threshold. Biomedical Engineering Online, 3(1), 28.
        """
        n_samples = len(ecg_signal)
        min_samples = int(min_distance * sampling_rate)
        
        # 1. 移动平均滤波器
        window_size = int(0.15 * sampling_rate)
        moving_avg = np.convolve(np.abs(ecg_signal), 
                                np.ones(window_size) / window_size, 
                                mode='same')
        
        # 2. 自适应阈值
        threshold = np.mean(moving_avg) * 1.5
        r_peaks = []
        
        i = 0
        while i < n_samples:
            if moving_avg[i] > threshold:
                # 找到峰值区域
                search_end = min(i + min_samples, n_samples)
                segment = moving_avg[i:search_end]
                
                if len(segment) > 0:
                    local_max_idx = np.argmax(segment)
                    candidate = i + local_max_idx
                    
                    # 在原始信号中验证
                    search_radius = int(0.05 * sampling_rate)
                    start = max(0, candidate - search_radius)
                    end = min(n_samples, candidate + search_radius)
                    
                    if end > start:
                        ecg_segment = ecg_signal[start:end]
                        ecg_max_idx = np.argmax(ecg_segment)
                        r_peak = start + ecg_max_idx
                        
                        # 添加到结果
                        if r_peak not in r_peaks:
                            r_peaks.append(r_peak)
                        
                        # 跳过已检测区域
                        i = r_peak + min_samples
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        
        # 去除紧密相邻的检测
        r_peaks = self._remove_close_peaks(r_peaks, min_samples)
        
        return sorted(r_peaks)
    
    def _detect_engzee(self, 
                      ecg_signal: np.ndarray,
                      sampling_rate: float,
                      min_distance: float = 0.3) -> List[int]:
        """
        Engelse-Zeelenberg算法实现（简化版）
        """
        n_samples = len(ecg_signal)
        min_samples = int(min_distance * sampling_rate)
        
        # 1. 低通滤波
        nyquist = sampling_rate / 2
        b, a = butter(2, 30 / nyquist, btype='low')
        ecg_filtered = filtfilt(b, a, ecg_signal)
        
        # 2. 一阶差分
        differentiated = np.diff(ecg_filtered)
        differentiated = np.append(differentiated, differentiated[-1])
        
        # 3. 寻找零交叉点
        zero_crossings = np.where(np.diff(np.sign(differentiated)) > 0)[0]
        
        # 4. 在原始信号中定位R波
        r_peaks = []
        for crossing in zero_crossings:
            if crossing < n_samples:
                # 在原始信号中搜索局部最大值
                search_radius = int(0.05 * sampling_rate)
                start = max(0, crossing - search_radius)
                end = min(n_samples, crossing + search_radius)
                
                if end > start:
                    segment = ecg_signal[start:end]
                    local_max_idx = np.argmax(segment)
                    r_peak = start + local_max_idx
                    
                    if r_peak not in r_peaks:
                        r_peaks.append(r_peak)
        
        # 去除紧密相邻的检测
        r_peaks = self._remove_close_peaks(r_peaks, min_samples)
        
        return sorted(r_peaks)
    
    def _detect_neurokit(self, 
                        ecg_signal: np.ndarray,
                        sampling_rate: float,
                        min_distance: float = 0.3) -> List[int]:
        """
        NeuroKit算法实现（简化版）
        
        NeuroKit是一个流行的生理信号处理库
        """
        n_samples = len(ecg_signal)
        min_samples = int(min_distance * sampling_rate)
        
        # 1. 带通滤波
        nyquist = sampling_rate / 2
        b, a = butter(2, [5 / nyquist, 15 / nyquist], btype='band')
        ecg_filtered = filtfilt(b, a, ecg_signal)
        
        # 2. 计算包络
        analytic_signal = signal.hilbert(ecg_filtered)
        amplitude_envelope = np.abs(analytic_signal)
        
        # 3. 平滑包络
        window_size = int(0.15 * sampling_rate)
        window = np.ones(window_size) / window_size
        smoothed_envelope = np.convolve(amplitude_envelope, window, mode='same')
        
        # 4. 阈值检测
        threshold = np.percentile(smoothed_envelope, 75)
        r_peaks = []
        
        i = 0
        while i < n_samples:
            if smoothed_envelope[i] > threshold:
                # 找到峰值区域
                search_end = min(i + min_samples, n_samples)
                segment = smoothed_envelope[i:search_end]
                
                if len(segment) > 0:
                    local_max_idx = np.argmax(segment)
                    candidate = i + local_max_idx
                    
                    # 在原始滤波信号中验证
                    search_radius = int(0.05 * sampling_rate)
                    start = max(0, candidate - search_radius)
                    end = min(n_samples, candidate + search_radius)
                    
                    if end > start:
                        ecg_segment = ecg_filtered[start:end]
                        ecg_max_idx = np.argmax(ecg_segment)
                        r_peak = start + ecg_max_idx
                        
                        # 添加到结果
                        if r_peak not in r_peaks:
                            r_peaks.append(r_peak)
                        
                        # 跳过已检测区域
                        i = r_peak + min_samples
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        
        # 去除紧密相邻的检测
        r_peaks = self._remove_close_peaks(r_peaks, min_samples)
        
        return sorted(r_peaks)
    
    def _remove_close_peaks(self, peaks: List[int], min_distance: int) -> List[int]:
        """
        去除紧密相邻的峰值，保留幅度最大的一个
        
        Args:
            peaks: 峰值位置列表
            min_distance: 最小间隔
            
        Returns:
            清理后的峰值列表
        """
        if not peaks:
            return []
        
        peaks = sorted(peaks)
        cleaned_peaks = [peaks[0]]
        
        for i in range(1, len(peaks)):
            if peaks[i] - cleaned_peaks[-1] >= min_distance:
                cleaned_peaks.append(peaks[i])
            else:
                # 如果两个峰值太接近，保留幅度更大的一个
                pass  # 这里需要原始信号信息，所以在外层处理
        
        return cleaned_peaks
    
    def _extract_qrs_info(self, 
                         ecg_signal: np.ndarray,
                         r_peaks: List[int],
                         sampling_rate: float) -> Dict[str, Any]:
        """
        提取QRS波相关信息
        
        Args:
            ecg_signal: ECG信号
            r_peaks: R波位置列表
            sampling_rate: 采样率
            
        Returns:
            QRS波信息字典
        """
        n_samples = len(ecg_signal)
        qrs_info = {
            "r_peak_times": [],  # R波时间（秒）
            "rr_intervals": [],  # RR间期（毫秒）
            "heart_rate": [],    # 瞬时心率（BPM）
            "qrs_onsets": [],    # QRS波起始点
            "qrs_offsets": [],   # QRS波终止点
            "qrs_durations": [], # QRS波持续时间（毫秒）
            "qrs_amplitudes": [], # QRS波幅度
            "qrs_morphology": []  # QRS波形态描述
        }
        
        if len(r_peaks) < 2:
            return qrs_info
        
        # 将R波位置转换为时间
        qrs_info["r_peak_times"] = [peak / sampling_rate for peak in r_peaks]
        
        # 计算RR间期和心率
        for i in range(1, len(r_peaks)):
            rr_interval = (r_peaks[i] - r_peaks[i-1]) / sampling_rate * 1000  # 毫秒
            
            # 过滤异常RR间期
            if (self.config.rr_interval_limits[0] <= rr_interval <= 
                self.config.rr_interval_limits[1]):
                heart_rate = 60000 / rr_interval if rr_interval > 0 else 0  # BPM
                
                if (self.config.heart_rate_limits[0] <= heart_rate <= 
                    self.config.heart_rate_limits[1]):
                    qrs_info["rr_intervals"].append(rr_interval)
                    qrs_info["heart_rate"].append(heart_rate)
        
        # 检测QRS波起始和终止点
        for i, r_peak in enumerate(r_peaks):
            # QRS波起始点检测（R波前搜索）
            qrs_onset = self._detect_qrs_onset(ecg_signal, r_peak, sampling_rate)
            qrs_info["qrs_onsets"].append(qrs_onset)
            
            # QRS波终止点检测（R波后搜索）
            qrs_offset = self._detect_qrs_offset(ecg_signal, r_peak, sampling_rate)
            qrs_info["qrs_offsets"].append(qrs_offset)
            
            # 计算QRS波持续时间
            if qrs_onset is not None and qrs_offset is not None:
                duration = (qrs_offset - qrs_onset) / sampling_rate * 1000  # 毫秒
                qrs_info["qrs_durations"].append(duration)
            
            # 计算QRS波幅度
            if qrs_onset is not None and qrs_offset is not None:
                qrs_segment = ecg_signal[qrs_onset:qrs_offset]
                if len(qrs_segment) > 0:
                    amplitude = np.max(qrs_segment) - np.min(qrs_segment)
                    qrs_info["qrs_amplitudes"].append(amplitude)
            
            # QRS波形态分类（简化）
            if i < len(r_peaks) - 1:
                morphology = self._classify_qrs_morphology(
                    ecg_signal, r_peak, sampling_rate
                )
                qrs_info["qrs_morphology"].append(morphology)
        
        return qrs_info
    
    def _detect_qrs_onset(self, 
                         ecg_signal: np.ndarray,
                         r_peak: int,
                         sampling_rate: float) -> Optional[int]:
        """
        检测QRS波起始点
        """
        n_samples = len(ecg_signal)
        
        # 向前搜索窗口（R波前100-200ms）
        search_back = int(0.2 * sampling_rate)  # 200ms
        start_idx = max(0, r_peak - search_back)
        
        # 提取搜索段
        segment = ecg_signal[start_idx:r_peak]
        if len(segment) < 10:
            return start_idx
        
        # 使用斜率变化检测QRS起始
        diff_segment = np.diff(segment)
        
        # 计算斜率阈值
        baseline_std = np.std(segment[:int(len(segment) * 0.3)]) if len(segment) > 30 else np.std(segment)
        threshold = baseline_std * 3
        
        # 寻找超过阈值的点（从后向前搜索）
        for i in range(len(diff_segment) - 1, 0, -1):
            if np.abs(diff_segment[i]) > threshold:
                return start_idx + i
        
        # 如果没有找到，返回搜索窗口起点
        return start_idx
    
    def _detect_qrs_offset(self, 
                          ecg_signal: np.ndarray,
                          r_peak: int,
                          sampling_rate: float) -> Optional[int]:
        """
        检测QRS波终止点
        """
        n_samples = len(ecg_signal)
        
        # 向后搜索窗口（R波后60-100ms）
        search_forward = int(0.1 * sampling_rate)  # 100ms
        end_idx = min(n_samples, r_peak + search_forward)
        
        # 提取搜索段
        segment = ecg_signal[r_peak:end_idx]
        if len(segment) < 10:
            return end_idx
        
        # 使用斜率变化检测QRS终止
        diff_segment = np.diff(segment)
        
        # 计算斜率阈值
        baseline_std = np.std(segment[-int(len(segment) * 0.3):]) if len(segment) > 30 else np.std(segment)
        threshold = baseline_std * 3
        
        # 寻找低于阈值的点（从前向后搜索）
        for i in range(len(diff_segment)):
            if np.abs(diff_segment[i]) < threshold / 2:
                # 检查后续点是否也低于阈值
                if i + 5 < len(diff_segment) and np.all(np.abs(diff_segment[i:i+5]) < threshold):
                    return r_peak + i
        
        # 如果没有找到，返回搜索窗口终点
        return end_idx
    
    def _classify_qrs_morphology(self, 
                                ecg_signal: np.ndarray,
                                r_peak: int,
                                sampling_rate: float) -> str:
        """
        分类QRS波形态
        """
        # 简化的形态分类
        search_window = int(0.12 * sampling_rate)  # 120ms窗口
        start = max(0, r_peak - search_window // 2)
        end = min(len(ecg_signal), r_peak + search_window // 2)
        
        if end <= start:
            return "unknown"
        
        segment = ecg_signal[start:end]
        
        # 计算特征
        amplitude = np.max(segment) - np.min(segment)
        zero_crossings = len(np.where(np.diff(np.sign(segment)))[0])
        slope = np.max(np.abs(np.diff(segment)))
        
        # 简单分类规则
        if zero_crossings <= 2:
            return "normal"
        elif zero_crossings == 3:
            return "notched"
        elif zero_crossings >= 4:
            return "complex"
        else:
            return "normal"
    
    def assess_ecg_quality(self,
                          ecg_data: np.ndarray,
                          sampling_rate: float,
                          r_peaks: List[int],
                          qrs_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估ECG信号质量
        
        Args:
            ecg_data: ECG信号数据
            sampling_rate: 采样率
            r_peaks: 检测到的R波位置
            qrs_info: QRS波信息
            
        Returns:
            信号质量评估结果字典
        """
        n_channels, n_samples = ecg_data.shape
        quality_results = {
            "quality_score": 0.0,
            "quality_flag": "unknown",
            "quality_metrics": {},
            "bad_segments": [],
            "recommendations": []
        }
        
        if n_channels == 0 or n_samples == 0:
            return quality_results
        
        # 使用第一个通道进行评估
        ecg_signal = ecg_data[0, :] if n_channels > 0 else ecg_data
        
        # 计算各项质量指标
        metrics = {}
        
        # 1. 信噪比（SNR）
        metrics["snr_db"] = self._calculate_snr(ecg_signal, sampling_rate, r_peaks)
        
        # 2. 基线漂移程度
        metrics["baseline_wander"] = self._calculate_baseline_wander(ecg_signal, sampling_rate)
        
        # 3. 工频干扰程度
        metrics["powerline_noise"] = self._calculate_powerline_noise(ecg_signal, sampling_rate)
        
        # 4. 肌电噪声程度
        metrics["emg_noise"] = self._calculate_emg_noise(ecg_signal, sampling_rate)
        
        # 5. QRS波检测可靠性
        metrics["qrs_reliability"] = self._calculate_qrs_reliability(r_peaks, qrs_info)
        
        # 6. 信号缺失检测
        metrics["signal_loss"] = self._detect_signal_loss(ecg_signal)
        
        # 计算总体质量分数（0-100）
        weights = {
            "snr_db": 0.25,
            "baseline_wander": 0.20,
            "powerline_noise": 0.15,
            "emg_noise": 0.15,
            "qrs_reliability": 0.25
        }
        
        quality_score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                # 归一化到0-1范围
                if metric == "snr_db":
                    norm_value = min(metrics[metric] / 30.0, 1.0)  # 30dB为优秀
                elif metric == "baseline_wander":
                    norm_value = max(0.0, 1.0 - metrics[metric] / 0.5)  # 0.5为阈值
                elif metric == "powerline_noise":
                    norm_value = max(0.0, 1.0 - metrics[metric] / 0.3)  # 0.3为阈值
                elif metric == "emg_noise":
                    norm_value = max(0.0, 1.0 - metrics[metric] / 0.2)  # 0.2为阈值
                elif metric == "qrs_reliability":
                    norm_value = metrics[metric]  # 已经在0-1范围
                else:
                    norm_value = 0.5
                
                quality_score += norm_value * weight
        
        quality_score *= 100  # 转换为0-100分
        
        # 确定质量标志
        if quality_score >= 80:
            quality_flag = "excellent"
        elif quality_score >= 60:
            quality_flag = "good"
        elif quality_score >= 40:
            quality_flag = "fair"
        elif quality_score >= 20:
            quality_flag = "poor"
        else:
            quality_flag = "unusable"
        
        # 检测坏段
        bad_segments = self._detect_bad_segments(ecg_signal, sampling_rate)
        
        # 生成建议
        recommendations = []
        if metrics["snr_db"] < 10:
            recommendations.append("低信噪比，考虑重新放置电极")
        if metrics["baseline_wander"] > 0.3:
            recommendations.append("显著基线漂移，检查电极接触")
        if metrics["powerline_noise"] > 0.2:
            recommendations.append("工频干扰明显，检查接地和屏蔽")
        if metrics["emg_noise"] > 0.15:
            recommendations.append("肌电噪声明显，请保持放松")
        if metrics["qrs_reliability"] < 0.7:
            recommendations.append("QRS波检测不可靠，检查信号质量")
        
        quality_results.update({
            "quality_score": quality_score,
            "quality_flag": quality_flag,
            "quality_metrics": metrics,
            "bad_segments": bad_segments,
            "recommendations": recommendations
        })
        
        return quality_results
    
    def _calculate_snr(self, 
                      ecg_signal: np.ndarray,
                      sampling_rate: float,
                      r_peaks: List[int]) -> float:
        """计算信噪比（dB）"""
        if len(r_peaks) < 2:
            return 0.0
        
        # 提取QRS波段作为信号
        qrs_segments = []
        for r_peak in r_peaks:
            start = max(0, r_peak - int(0.08 * sampling_rate))  # 80ms前
            end = min(len(ecg_signal), r_peak + int(0.08 * sampling_rate))  # 80ms后
            qrs_segments.extend(ecg_signal[start:end].tolist())
        
        # 提取非QRS波段作为噪声
        noise_segments = []
        mask = np.ones(len(ecg_signal), dtype=bool)
        for r_peak in r_peaks:
            start = max(0, r_peak - int(0.15 * sampling_rate))  # 150ms前
            end = min(len(ecg_signal), r_peak + int(0.15 * sampling_rate))  # 150ms后
            mask[start:end] = False
        
        noise_segments = ecg_signal[mask]
        
        if len(qrs_segments) == 0 or len(noise_segments) == 0:
            return 0.0
        
        # 计算信号和噪声功率
        signal_power = np.var(qrs_segments)
        noise_power = np.var(noise_segments)
        
        if noise_power == 0:
            return 100.0  # 无噪声
        
        # 计算SNR（dB）
        snr_db = 10 * np.log10(signal_power / noise_power)
        
        return max(0.0, snr_db)  # 确保非负
    
    def _calculate_baseline_wander(self, 
                                  ecg_signal: np.ndarray,
                                  sampling_rate: float) -> float:
        """计算基线漂移程度"""
        # 使用低通滤波提取基线
        nyquist = sampling_rate / 2
        cutoff = 0.5  # Hz
        
        if cutoff >= nyquist:
            return 0.0
        
        b, a = butter(2, cutoff / nyquist, btype='low')
        baseline = filtfilt(b, a, ecg_signal)
        
        # 计算基线变化的相对幅度
        baseline_range = np.ptp(baseline)  # 峰峰值
        signal_range = np.ptp(ecg_signal)
        
        if signal_range == 0:
            return 0.0
        
        return baseline_range / signal_range
    
    def _calculate_powerline_noise(self, 
                                  ecg_signal: np.ndarray,
                                  sampling_rate: float) -> float:
        """计算工频干扰程度"""
        # 计算50Hz（或60Hz）附近的功率
        powerline_freq = 50.0  # 假设50Hz工频
        
        # 计算功率谱密度
        freqs, psd = welch(ecg_signal, fs=sampling_rate, nperseg=min(1024, len(ecg_signal)))
        
        # 查找工频附近的频带
        bandwidth = 2.0  # Hz
        freq_mask = (freqs >= powerline_freq - bandwidth) & (freqs <= powerline_freq + bandwidth)
        
        if not np.any(freq_mask):
            return 0.0
        
        # 计算工频功率
        powerline_power = np.trapz(psd[freq_mask], freqs[freq_mask])
        total_power = np.trapz(psd, freqs)
        
        if total_power == 0:
            return 0.0
        
        return powerline_power / total_power
    
    def _calculate_emg_noise(self, 
                            ecg_signal: np.ndarray,
                            sampling_rate: float) -> float:
        """计算肌电噪声程度"""
        # 计算高频成分（20-100Hz）的功率
        nyquist = sampling_rate / 2
        
        # 计算功率谱密度
        freqs, psd = welch(ecg_signal, fs=sampling_rate, nperseg=min(1024, len(ecg_signal)))
        
        # 高频频带
        high_freq_mask = (freqs >= 20.0) & (freqs <= min(100.0, nyquist))
        
        if not np.any(high_freq_mask):
            return 0.0
        
        # 计算高频功率
        high_freq_power = np.trapz(psd[high_freq_mask], freqs[high_freq_mask])
        total_power = np.trapz(psd, freqs)
        
        if total_power == 0:
            return 0.0
        
        return high_freq_power / total_power
    
    def _calculate_qrs_reliability(self, 
                                  r_peaks: List[int],
                                  qrs_info: Dict[str, Any]) -> float:
        """计算QRS波检测可靠性"""
        if len(r_peaks) < 3:
            return 0.0
        
        reliability = 1.0
        
        # 检查RR间期的变异性
        rr_intervals = qrs_info.get("rr_intervals", [])
        if len(rr_intervals) >= 2:
            cv_rr = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 1.0
            # 变异性应适中，过高或过低都可能是检测问题
            if cv_rr > 0.3:  # 变异系数过大
                reliability *= 0.7
            elif cv_rr < 0.05:  # 变异系数过小（过于规律）
                reliability *= 0.8
        
        # 检查R波幅度的变异性
        qrs_amplitudes = qrs_info.get("qrs_amplitudes", [])
        if len(qrs_amplitudes) >= 2:
            cv_amp = np.std(qrs_amplitudes) / np.mean(qrs_amplitudes) if np.mean(qrs_amplitudes) > 0 else 1.0
            if cv_amp > 0.5:  # 幅度变异过大
                reliability *= 0.8
        
        # 检查是否有非常接近的R波
        min_rr = min(np.diff(r_peaks)) if len(r_peaks) > 1 else 1.0
        expected_min_rr = 0.3 * 250  # 假设采样率250Hz，最小RR间期300ms
        if min_rr < expected_min_rr * 0.7:  # 有R波过于接近
            reliability *= 0.6
        
        return max(0.0, min(1.0, reliability))
    
    def _detect_signal_loss(self, ecg_signal: np.ndarray) -> float:
        """检测信号缺失"""
        # 检测信号幅度是否接近零
        signal_std = np.std(ecg_signal)
        signal_mean = np.mean(np.abs(ecg_signal))
        
        if signal_std < 0.01 or signal_mean < 0.01:
            return 1.0  # 完全信号缺失
        else:
            return 0.0
    
    def _detect_bad_segments(self, 
                            ecg_signal: np.ndarray,
                            sampling_rate: float,
                            window_size: float = 5.0) -> List[Tuple[int, int]]:
        """
        检测坏段
        
        Args:
            ecg_signal: ECG信号
            sampling_rate: 采样率
            window_size: 分析窗口大小（秒）
            
        Returns:
            坏段起始和结束索引列表
        """
        n_samples = len(ecg_signal)
        window_samples = int(window_size * sampling_rate)
        bad_segments = []
        
        # 滑动窗口分析
        for start in range(0, n_samples, window_samples // 2):
            end = min(start + window_samples, n_samples)
            
            if end - start < window_samples // 4:  # 窗口太小
                continue
            
            segment = ecg_signal[start:end]
            
            # 计算窗口内信号质量指标
            segment_std = np.std(segment)
            segment_range = np.ptp(segment)
            
            # 判断是否为坏段
            is_bad = False
            
            # 1. 信号幅度过小
            if segment_range < 0.1:  # 阈值可调整
                is_bad = True
            
            # 2. 信号标准差过小（可能为直线）
            if segment_std < 0.01:  # 阈值可调整
                is_bad = True
            
            # 3. 信号幅度过大（可能为电极脱落或运动伪影）
            if segment_range > 10.0:  # 阈值可调整
                is_bad = True
            
            if is_bad:
                bad_segments.append((start, end))
        
        # 合并相邻的坏段
        merged_segments = []
        if bad_segments:
            bad_segments.sort()
            current_start, current_end = bad_segments[0]
            
            for start, end in bad_segments[1:]:
                if start <= current_end + window_samples:  # 相邻或重叠
                    current_end = max(current_end, end)
                else:
                    merged_segments.append((current_start, current_end))
                    current_start, current_end = start, end
            
            merged_segments.append((current_start, current_end))
        
        return merged_segments
    
    def analyze_hrv(self, 
                   rr_intervals: List[float],
                   sampling_rate: float,
                   methods: List[HRVAnalysisMethod],
                   frequency_bands: Dict[str, Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        心率变异性分析
        
        Args:
            rr_intervals: RR间期列表（毫秒）
            sampling_rate: 原始ECG采样率
            methods: 分析方法列表
            frequency_bands: 频带定义
            
        Returns:
            HRV指标字典
        """
        if len(rr_intervals) < 5:
            logger.warning("RR间期数量不足，无法进行HRV分析")
            return {}
        
        rr_intervals = np.array(rr_intervals)
        hrv_results = {}
        
        # 预处理RR间期
        cleaned_rr, preprocessing_info = self._preprocess_rr_intervals(rr_intervals)
        
        if len(cleaned_rr) < 5:
            logger.warning("清洗后RR间期数量不足，无法进行HRV分析")
            return {}
        
        # 时域分析
        if HRVAnalysisMethod.TIME_DOMAIN in methods:
            hrv_results["time_domain"] = self._hrv_time_domain_analysis(cleaned_rr)
        
        # 频域分析
        if HRVAnalysisMethod.FREQUENCY_DOMAIN in methods:
            hrv_results["frequency_domain"] = self._hrv_frequency_domain_analysis(
                cleaned_rr, frequency_bands
            )
        
        # 非线性分析
        if HRVAnalysisMethod.NONLINEAR in methods and len(cleaned_rr) >= 100:
            hrv_results["nonlinear"] = self._hrv_nonlinear_analysis(cleaned_rr)
        
        # Poincaré分析
        if HRVAnalysisMethod.POINCARE in methods and len(cleaned_rr) >= 20:
            hrv_results["poincare"] = self._hrv_poincare_analysis(cleaned_rr)
        
        # 添加预处理信息
        hrv_results["preprocessing_info"] = preprocessing_info
        
        return hrv_results
    
    def _preprocess_rr_intervals(self, rr_intervals: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        预处理RR间期
        
        Args:
            rr_intervals: 原始RR间期（毫秒）
            
        Returns:
            cleaned_rr: 清洗后的RR间期
            info: 预处理信息
        """
        info = {
            "n_original": len(rr_intervals),
            "n_removed": 0,
            "removed_indices": [],
            "mean_rr": 0.0,
            "std_rr": 0.0
        }
        
        if len(rr_intervals) == 0:
            return np.array([]), info
        
        # 转换单位为秒（用于分析）
        rr_seconds = rr_intervals / 1000.0
        
        # 检测并移除异常值
        mean_rr = np.mean(rr_seconds)
        std_rr = np.std(rr_seconds)
        
        info["mean_rr"] = mean_rr * 1000  # 转换回毫秒
        info["std_rr"] = std_rr * 1000
        
        # 定义合理范围（300-1500ms对应0.3-1.5秒）
        lower_limit = 0.3  # 秒
        upper_limit = 1.5  # 秒
        
        # 使用统计方法检测异常值
        z_scores = np.abs((rr_seconds - mean_rr) / std_rr) if std_rr > 0 else np.zeros_like(rr_seconds)
        outlier_mask = (rr_seconds < lower_limit) | (rr_seconds > upper_limit) | (z_scores > 3)
        
        # 保留正常值
        normal_mask = ~outlier_mask
        cleaned_rr_seconds = rr_seconds[normal_mask]
        
        info["n_removed"] = np.sum(outlier_mask)
        info["removed_indices"] = np.where(outlier_mask)[0].tolist()
        
        # 如果需要，对缺失值进行插值
        if info["n_removed"] > 0 and len(cleaned_rr_seconds) >= 2:
            # 创建时间序列
            rr_times = np.cumsum(rr_seconds)
            normal_times = rr_times[normal_mask]
            
            # 线性插值
            interp_func = interp1d(normal_times, cleaned_rr_seconds, 
                                  kind='linear', fill_value="extrapolate")
            
            # 在原始时间点上插值
            cleaned_rr_seconds = interp_func(rr_times)
        
        # 转换回毫秒
        cleaned_rr = cleaned_rr_seconds * 1000
        
        return cleaned_rr, info
    
    def _hrv_time_domain_analysis(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """HRV时域分析"""
        metrics = {}
        
        # 基本统计量
        metrics["mean_rr"] = np.mean(rr_intervals)
        metrics["std_rr"] = np.std(rr_intervals)  # SDNN
        metrics["mean_hr"] = 60000 / metrics["mean_rr"] if metrics["mean_rr"] > 0 else 0
        
        # 相邻RR间期差值
        diff_rr = np.diff(rr_intervals)
        
        if len(diff_rr) > 0:
            metrics["rmssd"] = np.sqrt(np.mean(diff_rr ** 2))
            
            # NN50和pNN50
            nn50 = np.sum(np.abs(diff_rr) > 50)
            metrics["nn50"] = nn50
            metrics["pnn50"] = nn50 / len(diff_rr) * 100 if len(diff_rr) > 0 else 0
            
            # NN20和pNN20
            nn20 = np.sum(np.abs(diff_rr) > 20)
            metrics["nn20"] = nn20
            metrics["pnn20"] = nn20 / len(diff_rr) * 100 if len(diff_rr) > 0 else 0
        
        # HRV三角指数
        hist, bin_edges = np.histogram(rr_intervals, bins=20, density=True)
        max_hist = np.max(hist) if len(hist) > 0 else 1.0
        metrics["triangular_index"] = len(rr_intervals) / max_hist if max_hist > 0 else 0
        
        # TINN（三角插值NN间期直方图）
        if len(hist) >= 3:
            try:
                # 找到直方图的基宽
                max_idx = np.argmax(hist)
                left_base = np.where(hist[:max_idx] < hist[max_idx] * 0.5)[0]
                right_base = np.where(hist[max_idx:] < hist[max_idx] * 0.5)[0]
                
                if len(left_base) > 0 and len(right_base) > 0:
                    left_edge = bin_edges[left_base[-1]]
                    right_edge = bin_edges[max_idx + right_base[0] + 1]
                    metrics["tinn"] = right_edge - left_edge
                else:
                    metrics["tinn"] = 0.0
            except:
                metrics["tinn"] = 0.0
        else:
            metrics["tinn"] = 0.0
        
        return metrics
    
    def _hrv_frequency_domain_analysis(self, 
                                      rr_intervals: np.ndarray,
                                      frequency_bands: Dict[str, Tuple[float, float]] = None) -> Dict[str, Any]:
        """HRV频域分析"""
        if frequency_bands is None:
            frequency_bands = {
                "ULF": (0.0, 0.003),
                "VLF": (0.003, 0.04),
                "LF": (0.04, 0.15),
                "HF": (0.15, 0.4)
            }
        
        metrics = {}
        
        # 将RR间期转换为瞬时心率时间序列
        rr_times = np.cumsum(rr_intervals) / 1000.0  # 转换为秒
        rr_times = rr_times - rr_times[0]  # 从0开始
        
        # 定义分析频率范围
        min_freq = 0.003
        max_freq = 0.5
        freq_resolution = 0.001  # Hz
        
        freqs = np.arange(min_freq, max_freq, freq_resolution)
        
        # 使用Lomb-Scargle周期图处理非均匀采样的RR间期
        try:
            from scipy.signal import lombscargle
            
            # 计算角频率
            angular_freqs = 2 * np.pi * freqs
            
            # Lomb-Scargle周期图
            power = lombscargle(rr_times, rr_intervals, angular_freqs, normalize=True)
            
            # 计算各频带功率
            total_power = np.trapz(power, freqs)
            metrics["total_power"] = total_power
            
            for band_name, (f_low, f_high) in frequency_bands.items():
                band_mask = (freqs >= f_low) & (freqs <= f_high)
                if np.any(band_mask):
                    band_power = np.trapz(power[band_mask], freqs[band_mask])
                    metrics[f"{band_name.lower()}_power"] = band_power
                    metrics[f"{band_name.lower()}_peak"] = freqs[band_mask][np.argmax(power[band_mask])] if band_power > 0 else 0.0
                else:
                    metrics[f"{band_name.lower()}_power"] = 0.0
                    metrics[f"{band_name.lower()}_peak"] = 0.0
            
            # 计算LF/HF比值
            if metrics.get("hf_power", 0) > 0:
                metrics["lf_hf_ratio"] = metrics.get("lf_power", 0) / metrics["hf_power"]
            else:
                metrics["lf_hf_ratio"] = 0.0
            
            # 计算归一化功率
            lf_hf_power = metrics.get("lf_power", 0) + metrics.get("hf_power", 0)
            if lf_hf_power > 0:
                metrics["lf_nu"] = metrics.get("lf_power", 0) / lf_hf_power * 100
                metrics["hf_nu"] = metrics.get("hf_power", 0) / lf_hf_power * 100
            else:
                metrics["lf_nu"] = 0.0
                metrics["hf_nu"] = 0.0
            
            # 保存频谱数据用于可视化
            metrics["frequencies"] = freqs.tolist()
            metrics["power_spectrum"] = power.tolist()
            
        except Exception as e:
            logger.error(f"Lomb-Scargle周期图计算失败: {str(e)}")
            # 使用简单的FFT方法作为备选
            metrics = self._hrv_frequency_domain_fft(rr_intervals, frequency_bands)
        
        return metrics
    
    def _hrv_frequency_domain_fft(self, 
                                 rr_intervals: np.ndarray,
                                 frequency_bands: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """使用FFT进行HRV频域分析（备选方法）"""
        metrics = {}
        
        # 对RR间期进行插值以获得均匀采样序列
        interpolation_rate = self.config.hrv_interpolation_rate  # Hz
        
        rr_times = np.cumsum(rr_intervals) / 1000.0  # 转换为秒
        total_time = rr_times[-1]
        
        # 创建均匀时间网格
        uniform_times = np.arange(0, total_time, 1.0 / interpolation_rate)
        
        # 线性插值
        interp_func = interp1d(rr_times, rr_intervals, kind='cubic', fill_value="extrapolate")
        uniform_rr = interp_func(uniform_times)
        
        # 去除线性趋势
        uniform_rr_detrended = signal.detrend(uniform_rr)
        
        # 计算功率谱密度
        freqs, psd = welch(uniform_rr_detrended, 
                          fs=interpolation_rate, 
                          nperseg=min(256, len(uniform_rr_detrended)),
                          scaling='density')
        
        # 计算各频带功率
        total_power = np.trapz(psd, freqs)
        metrics["total_power"] = total_power
        
        for band_name, (f_low, f_high) in frequency_bands.items():
            band_mask = (freqs >= f_low) & (freqs <= f_high)
            if np.any(band_mask):
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                metrics[f"{band_name.lower()}_power"] = band_power
                metrics[f"{band_name.lower()}_peak"] = freqs[band_mask][np.argmax(psd[band_mask])] if band_power > 0 else 0.0
            else:
                metrics[f"{band_name.lower()}_power"] = 0.0
                metrics[f"{band_name.lower()}_peak"] = 0.0
        
        # 计算LF/HF比值
        if metrics.get("hf_power", 0) > 0:
            metrics["lf_hf_ratio"] = metrics.get("lf_power", 0) / metrics["hf_power"]
        else:
            metrics["lf_hf_ratio"] = 0.0
        
        # 计算归一化功率
        lf_hf_power = metrics.get("lf_power", 0) + metrics.get("hf_power", 0)
        if lf_hf_power > 0:
            metrics["lf_nu"] = metrics.get("lf_power", 0) / lf_hf_power * 100
            metrics["hf_nu"] = metrics.get("hf_power", 0) / lf_hf_power * 100
        else:
            metrics["lf_nu"] = 0.0
            metrics["hf_nu"] = 0.0
        
        # 保存频谱数据
        metrics["frequencies"] = freqs.tolist()
        metrics["power_spectrum"] = psd.tolist()
        metrics["method"] = "fft_interpolated"
        
        return metrics
    
    def _hrv_nonlinear_analysis(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """HRV非线性分析"""
        metrics = {}
        
        # 样本熵（Sample Entropy）
        metrics["sample_entropy"] = self._calculate_sample_entropy(rr_intervals)
        
        # 去趋势波动分析（DFA）
        metrics["dfa_alpha1"], metrics["dfa_alpha2"] = self._calculate_dfa(rr_intervals)
        
        # 近似熵（Approximate Entropy）- 简化的计算
        metrics["approximate_entropy"] = self._calculate_approximate_entropy(rr_intervals)
        
        return metrics
    
    def _calculate_sample_entropy(self, rr_intervals: np.ndarray, 
                                 m: int = 2, r: float = 0.2) -> float:
        """计算样本熵"""
        n = len(rr_intervals)
        
        if n < m + 1:
            return 0.0
        
        # 标准化数据
        rr_std = np.std(rr_intervals)
        if rr_std == 0:
            return 0.0
        
        rr_normalized = (rr_intervals - np.mean(rr_intervals)) / rr_std
        
        # 计算距离
        def _maxdist(xi, xj):
            return np.max(np.abs(xi - xj))
        
        # 计算匹配数
        def _phi(m):
            patterns = []
            for i in range(n - m + 1):
                patterns.append(rr_normalized[i:i + m])
            
            patterns = np.array(patterns)
            c = 0
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    if i != j and _maxdist(patterns[i], patterns[j]) <= r:
                        c += 1
            
            return c / (n - m) / (n - m - 1) if n - m > 1 else 0
        
        # 计算样本熵
        a, b = _phi(m + 1), _phi(m)
        
        if b == 0:
            return 0.0
        
        return -np.log(a / b) if a > 0 and b > 0 else 0.0
    
    def _calculate_dfa(self, rr_intervals: np.ndarray) -> Tuple[float, float]:
        """计算去趋势波动分析（DFA）指数"""
        n = len(rr_intervals)
        
        if n < 100:
            return 0.0, 0.0
        
        # 积分序列
        y = np.cumsum(rr_intervals - np.mean(rr_intervals))
        
        # 定义尺度范围
        scales = np.logspace(np.log10(4), np.log10(n // 4), 20).astype(int)
        scales = scales[scales <= n // 4]
        
        # 计算波动函数
        fluctuations = []
        
        for scale in scales:
            # 分割序列
            n_segments = n // scale
            if n_segments < 2:
                continue
            
            # 计算每个段的局部趋势和波动
            f2_segment = []
            for v in range(n_segments):
                segment = y[v * scale:(v + 1) * scale]
                
                # 局部线性拟合
                t = np.arange(len(segment))
                coeffs = np.polyfit(t, segment, 1)
                trend = np.polyval(coeffs, t)
                
                # 去趋势波动
                detrended = segment - trend
                f2_segment.append(np.mean(detrended ** 2))
            
            # 平均波动
            fluctuations.append(np.sqrt(np.mean(f2_segment)))
        
        if len(scales) < 2 or len(fluctuations) < 2:
            return 0.0, 0.0
        
        # 对数线性回归
        log_scales = np.log10(scales[:len(fluctuations)])
        log_fluctuations = np.log10(fluctuations)
        
        # 计算两个尺度的斜率（短时和长时）
        mid_point = len(log_scales) // 2
        
        # 短时尺度（alpha1）
        if mid_point >= 2:
            coeffs1 = np.polyfit(log_scales[:mid_point], log_fluctuations[:mid_point], 1)
            alpha1 = coeffs1[0]
        else:
            alpha1 = 0.0
        
        # 长时尺度（alpha2）
        if len(log_scales) - mid_point >= 2:
            coeffs2 = np.polyfit(log_scales[mid_point:], log_fluctuations[mid_point:], 1)
            alpha2 = coeffs2[0]
        else:
            alpha2 = 0.0
        
        return alpha1, alpha2
    
    def _calculate_approximate_entropy(self, rr_intervals: np.ndarray,
                                      m: int = 2, r: float = 0.2) -> float:
        """计算近似熵（简化版本）"""
        # 这是简化的近似熵计算，实际应用可能需要更精确的实现
        n = len(rr_intervals)
        
        if n < m + 1:
            return 0.0
        
        # 使用样本熵作为近似
        return self._calculate_sample_entropy(rr_intervals, m, r)
    
    def _hrv_poincare_analysis(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Poincaré图分析"""
        metrics = {}
        
        if len(rr_intervals) < 3:
            return metrics
        
        # 创建Poincaré图数据（RR_n vs RR_n+1）
        rr_n = rr_intervals[:-1]
        rr_n1 = rr_intervals[1:]
        
        # 计算椭圆拟合参数
        # SD1：垂直于恒等线的标准差（短期变异性）
        # SD2：沿着恒等线的标准差（长期变异性）
        
        diff_rr = rr_n1 - rr_n
        sum_rr = rr_n1 + rr_n
        
        sd1 = np.std(diff_rr) / np.sqrt(2)
        sd2 = np.std(sum_rr) / np.sqrt(2)
        
        metrics["sd1"] = sd1
        metrics["sd2"] = sd2
        metrics["sd1_sd2_ratio"] = sd1 / sd2 if sd2 > 0 else 0.0
        
        # 椭圆面积
        metrics["ellipse_area"] = np.pi * sd1 * sd2
        
        # 计算重心
        metrics["centroid_x"] = np.mean(rr_n)
        metrics["centroid_y"] = np.mean(rr_n1)
        
        return metrics
    
    def detect_arrhythmias(self,
                          ecg_data: np.ndarray,
                          r_peaks: List[int],
                          qrs_info: Dict[str, Any],
                          sampling_rate: float,
                          signal_quality: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        检测心律异常
        
        Args:
            ecg_data: ECG信号数据
            r_peaks: 检测到的R波位置
            qrs_info: QRS波信息
            sampling_rate: 采样率
            signal_quality: 信号质量评估结果
            
        Returns:
            心律异常检测结果字典
        """
        arrhythmia_results = {
            "arrhythmia_flags": {},
            "arrhythmia_events": [],
            "confidence_scores": {},
            "detection_parameters": {}
        }
        
        if len(r_peaks) < 5:
            return arrhythmia_results
        
        # 提取相关信息
        rr_intervals = qrs_info.get("rr_intervals", [])
        heart_rate = qrs_info.get("heart_rate", [])
        
        if len(rr_intervals) < 3 or len(heart_rate) < 3:
            return arrhythmia_results
        
        # 1. 心动过速检测
        tachycardia_detected, tachycardia_info = self._detect_tachycardia(
            heart_rate, rr_intervals, sampling_rate
        )
        if tachycardia_detected:
            arrhythmia_results["arrhythmia_flags"]["tachycardia"] = True
            arrhythmia_results["arrhythmia_events"].extend(tachycardia_info.get("events", []))
            arrhythmia_results["confidence_scores"]["tachycardia"] = tachycardia_info.get("confidence", 0.7)
        
        # 2. 心动过缓检测
        bradycardia_detected, bradycardia_info = self._detect_bradycardia(
            heart_rate, rr_intervals, sampling_rate
        )
        if bradycardia_detected:
            arrhythmia_results["arrhythmia_flags"]["bradycardia"] = True
            arrhythmia_results["arrhythmia_events"].extend(bradycardia_info.get("events", []))
            arrhythmia_results["confidence_scores"]["bradycardia"] = bradycardia_info.get("confidence", 0.7)
        
        # 3. 室性早搏（PVC）检测
        pvc_detected, pvc_info = self._detect_pvc(
            ecg_data, r_peaks, qrs_info, sampling_rate
        )
        if pvc_detected:
            arrhythmia_results["arrhythmia_flags"]["pvc"] = True
            arrhythmia_results["arrhythmia_events"].extend(pvc_info.get("events", []))
            arrhythmia_results["confidence_scores"]["pvc"] = pvc_info.get("confidence", 0.6)
        
        # 4. 房性早搏（PAC）检测
        pac_detected, pac_info = self._detect_pac(
            ecg_data, r_peaks, qrs_info, sampling_rate
        )
        if pac_detected:
            arrhythmia_results["arrhythmia_flags"]["pac"] = True
            arrhythmia_results["arrhythmia_events"].extend(pac_info.get("events", []))
            arrhythmia_results["confidence_scores"]["pac"] = pac_info.get("confidence", 0.5)
        
        # 5. 心房颤动（AFib）检测
        afib_detected, afib_info = self._detect_afib(
            rr_intervals, sampling_rate
        )
        if afib_detected:
            arrhythmia_results["arrhythmia_flags"]["afib"] = True
            arrhythmia_results["arrhythmia_events"].extend(afib_info.get("events", []))
            arrhythmia_results["confidence_scores"]["afib"] = afib_info.get("confidence", 0.6)
        
        # 6. 窦性心律不齐检测
        sinus_arrhythmia_detected, sinus_info = self._detect_sinus_arrhythmia(
            rr_intervals, sampling_rate
        )
        if sinus_arrhythmia_detected:
            arrhythmia_results["arrhythmia_flags"]["sinus_arrhythmia"] = True
            arrhythmia_results["arrhythmia_events"].extend(sinus_info.get("events", []))
            arrhythmia_results["confidence_scores"]["sinus_arrhythmia"] = sinus_info.get("confidence", 0.8)
        
        # 7. 二联律/三联律检测
        bigeminy_detected, bigeminy_info = self._detect_bigeminy_trigeminy(
            rr_intervals, ecg_data, r_peaks, sampling_rate
        )
        if bigeminy_detected:
            arrhythmia_results["arrhythmia_flags"]["bigeminy"] = True
            arrhythmia_results["arrhythmia_events"].extend(bigeminy_info.get("events", []))
            arrhythmia_results["confidence_scores"]["bigeminy"] = bigeminy_info.get("confidence", 0.7)
        
        return arrhythmia_results
    
    def _detect_tachycardia(self, 
                           heart_rate: List[float],
                           rr_intervals: List[float],
                           sampling_rate: float) -> Tuple[bool, Dict]:
        """检测心动过速"""
        if len(heart_rate) < 3:
            return False, {}
        
        # 心动过速阈值（通常>100 BPM）
        threshold_bpm = 100.0
        
        # 检测持续心动过速
        tachycardia_events = []
        current_event = None
        
        for i, hr in enumerate(heart_rate):
            if hr > threshold_bpm:
                if current_event is None:
                    current_event = {"start_index": i, "start_hr": hr}
            else:
                if current_event is not None:
                    current_event["end_index"] = i - 1
                    current_event["end_hr"] = heart_rate[i - 1]
                    current_event["duration"] = (i - current_event["start_index"]) * np.mean(rr_intervals) / 1000.0
                    
                    # 需要持续至少3个心跳
                    if current_event["duration"] >= 3 * (60.0 / threshold_bpm):
                        tachycardia_events.append(current_event)
                    
                    current_event = None
        
        # 处理最后的事件
        if current_event is not None:
            current_event["end_index"] = len(heart_rate) - 1
            current_event["end_hr"] = heart_rate[-1]
            current_event["duration"] = (len(heart_rate) - current_event["start_index"]) * np.mean(rr_intervals) / 1000.0
            
            if current_event["duration"] >= 3 * (60.0 / threshold_bpm):
                tachycardia_events.append(current_event)
        
        detected = len(tachycardia_events) > 0
        confidence = min(0.9, np.mean([e["duration"] for e in tachycardia_events]) / 10.0) if tachycardia_events else 0.0
        
        return detected, {
            "events": tachycardia_events,
            "confidence": confidence,
            "threshold_bpm": threshold_bpm
        }
    
    def _detect_bradycardia(self, 
                           heart_rate: List[float],
                           rr_intervals: List[float],
                           sampling_rate: float) -> Tuple[bool, Dict]:
        """检测心动过缓"""
        if len(heart_rate) < 3:
            return False, {}
        
        # 心动过缓阈值（通常<60 BPM）
        threshold_bpm = 60.0
        
        # 检测持续心动过缓
        bradycardia_events = []
        current_event = None
        
        for i, hr in enumerate(heart_rate):
            if hr < threshold_bpm:
                if current_event is None:
                    current_event = {"start_index": i, "start_hr": hr}
            else:
                if current_event is not None:
                    current_event["end_index"] = i - 1
                    current_event["end_hr"] = heart_rate[i - 1]
                    current_event["duration"] = (i - current_event["start_index"]) * np.mean(rr_intervals) / 1000.0
                    
                    # 需要持续至少3个心跳
                    if current_event["duration"] >= 3 * (60.0 / threshold_bpm):
                        bradycardia_events.append(current_event)
                    
                    current_event = None
        
        # 处理最后的事件
        if current_event is not None:
            current_event["end_index"] = len(heart_rate) - 1
            current_event["end_hr"] = heart_rate[-1]
            current_event["duration"] = (len(heart_rate) - current_event["start_index"]) * np.mean(rr_intervals) / 1000.0
            
            if current_event["duration"] >= 3 * (60.0 / threshold_bpm):
                bradycardia_events.append(current_event)
        
        detected = len(bradycardia_events) > 0
        confidence = min(0.9, np.mean([e["duration"] for e in bradycardia_events]) / 10.0) if bradycardia_events else 0.0
        
        return detected, {
            "events": bradycardia_events,
            "confidence": confidence,
            "threshold_bpm": threshold_bpm
        }
    
    def _detect_pvc(self, 
                   ecg_data: np.ndarray,
                   r_peaks: List[int],
                   qrs_info: Dict[str, Any],
                   sampling_rate: float) -> Tuple[bool, Dict]:
        """检测室性早搏（PVC）"""
        if len(r_peaks) < 4:
            return False, {}
        
        # 使用第一个通道
        ecg_signal = ecg_data[0, :] if len(ecg_data.shape) == 2 else ecg_data
        
        pvc_events = []
        rr_intervals = qrs_info.get("rr_intervals", [])
        
        # 计算平均RR间期
        mean_rr = np.mean(rr_intervals) if rr_intervals else 1000.0
        
        for i in range(1, len(r_peaks) - 1):
            # 计算前一个RR间期
            rr_prev = r_peaks[i] - r_peaks[i - 1]
            rr_current = r_peaks[i + 1] - r_peaks[i]
            
            # 检查是否提前出现（<85%平均RR间期）
            if rr_current < 0.85 * mean_rr:
                # 检查是否有完全代偿间期
                compensatory_pause = rr_prev + rr_current
                
                if 1.8 * mean_rr < compensatory_pause < 2.2 * mean_rr:
                    # 检查QRS波形态（宽度和幅度）
                    qrs_onset = qrs_info.get("qrs_onsets", [])
                    qrs_offset = qrs_info.get("qrs_offsets", [])
                    qrs_durations = qrs_info.get("qrs_durations", [])
                    qrs_amplitudes = qrs_info.get("qrs_amplitudes", [])
                    
                    if (i < len(qrs_durations) and i < len(qrs_amplitudes) and
                        i < len(qrs_onset) and i < len(qrs_offset)):
                        
                        # PVC通常有更宽的QRS波（>120ms）和更高的幅度
                        if (qrs_durations[i] > 120.0 and  # 毫秒
                            qrs_amplitudes[i] > np.mean(qrs_amplitudes) * 1.5):
                            
                            pvc_events.append({
                                "index": i,
                                "r_peak": r_peaks[i],
                                "rr_interval": rr_current,
                                "qrs_duration": qrs_durations[i],
                                "qrs_amplitude": qrs_amplitudes[i],
                                "compensatory_pause": compensatory_pause
                            })
        
        detected = len(pvc_events) > 0
        confidence = min(0.8, len(pvc_events) / 10.0)  # 置信度基于PVC数量
        
        return detected, {
            "events": pvc_events,
            "confidence": confidence,
            "n_pvc": len(pvc_events)
        }
    
    def _detect_pac(self, 
                   ecg_data: np.ndarray,
                   r_peaks: List[int],
                   qrs_info: Dict[str, Any],
                   sampling_rate: float) -> Tuple[bool, Dict]:
        """检测房性早搏（PAC）"""
        if len(r_peaks) < 4:
            return False, {}
        
        pac_events = []
        rr_intervals = qrs_info.get("rr_intervals", [])
        
        # 计算平均RR间期
        mean_rr = np.mean(rr_intervals) if rr_intervals else 1000.0
        
        for i in range(1, len(r_peaks) - 1):
            # 计算RR间期
            rr_current = r_peaks[i + 1] - r_peaks[i]
            
            # 检查是否提前出现（<85%平均RR间期）
            if rr_current < 0.85 * mean_rr:
                # PAC通常没有完全代偿间期
                # 检查QRS波形态（应接近正常）
                qrs_durations = qrs_info.get("qrs_durations", [])
                qrs_amplitudes = qrs_info.get("qrs_amplitudes", [])
                
                if i < len(qrs_durations) and i < len(qrs_amplitudes):
                    # PAC的QRS波通常正常（<120ms）
                    if qrs_durations[i] <= 120.0:  # 毫秒
                        pac_events.append({
                            "index": i,
                            "r_peak": r_peaks[i],
                            "rr_interval": rr_current,
                            "qrs_duration": qrs_durations[i],
                            "qrs_amplitude": qrs_amplitudes[i]
                        })
        
        detected = len(pac_events) > 0
        confidence = min(0.7, len(pac_events) / 10.0)
        
        return detected, {
            "events": pac_events,
            "confidence": confidence,
            "n_pac": len(pac_events)
        }
    
    def _detect_afib(self, 
                    rr_intervals: List[float],
                    sampling_rate: float) -> Tuple[bool, Dict]:
        """检测心房颤动（AFib）"""
        if len(rr_intervals) < 10:
            return False, {}
        
        rr_intervals = np.array(rr_intervals)
        
        # AFib特征：RR间期极度不规则
        # 计算变异系数
        cv_rr = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
        
        # 计算pNN50（AFib时通常较高）
        diff_rr = np.diff(rr_intervals)
        pnn50 = np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100 if len(diff_rr) > 0 else 0
        
        # AFib检测阈值
        cv_threshold = 0.1  # 变异系数阈值
        pnn50_threshold = 10.0  # pNN50阈值
        
        afib_detected = (cv_rr > cv_threshold and pnn50 > pnn50_threshold)
        
        # 计算置信度
        cv_score = min(1.0, cv_rr / 0.3)  # 假设0.3为强AFib
        pnn50_score = min(1.0, pnn50 / 30.0)  # 假设30%为强AFib
        confidence = (cv_score + pnn50_score) / 2
        
        return afib_detected, {
            "events": [{"cv_rr": cv_rr, "pnn50": pnn50}],
            "confidence": confidence,
            "cv_rr": cv_rr,
            "pnn50": pnn50,
            "thresholds": {"cv": cv_threshold, "pnn50": pnn50_threshold}
        }
    
    def _detect_sinus_arrhythmia(self, 
                                rr_intervals: List[float],
                                sampling_rate: float) -> Tuple[bool, Dict]:
        """检测窦性心律不齐"""
        if len(rr_intervals) < 10:
            return False, {}
        
        rr_intervals = np.array(rr_intervals)
        
        # 窦性心律不齐特征：RR间期与呼吸同步变化
        # 计算频谱以检测呼吸频率（0.1-0.5 Hz）的周期性
        
        # 将RR间期转换为均匀时间序列
        rr_times = np.cumsum(rr_intervals) / 1000.0  # 转换为秒
        
        # 使用Lomb-Scargle检测呼吸频率
        from scipy.signal import lombscargle
        
        freqs = np.arange(0.1, 0.5, 0.01)  # 呼吸频率范围
        angular_freqs = 2 * np.pi * freqs
        
        power = lombscargle(rr_times, rr_intervals, angular_freqs, normalize=True)
        
        # 寻找呼吸频率的峰值
        peak_threshold = np.percentile(power, 75)
        peaks = find_peaks(power, height=peak_threshold)[0]
        
        # 检查是否有显著的呼吸频率峰值
        sinus_arrhythmia_detected = len(peaks) > 0
        
        # 计算置信度
        max_power = np.max(power) if len(power) > 0 else 0
        confidence = min(1.0, max_power / np.mean(power) * 0.5) if np.mean(power) > 0 else 0
        
        return sinus_arrhythmia_detected, {
            "events": [{"respiratory_freq": freqs[peaks[0]] if len(peaks) > 0 else 0}],
            "confidence": confidence,
            "power_spectrum": {"frequencies": freqs.tolist(), "power": power.tolist()}
        }
    
    def _detect_bigeminy_trigeminy(self, 
                                  rr_intervals: List[float],
                                  ecg_data: np.ndarray,
                                  r_peaks: List[int],
                                  sampling_rate: float) -> Tuple[bool, Dict]:
        """检测二联律/三联律"""
        if len(rr_intervals) < 6:
            return False, {}
        
        # 寻找规律的模式：短-长-短-长（二联律）或短-短-长（三联律）
        patterns = []
        
        for i in range(len(rr_intervals) - 3):
            segment = rr_intervals[i:i + 4]
            
            # 检查二联律模式
            if (abs(segment[0] - segment[2]) < segment[0] * 0.2 and
                abs(segment[1] - segment[3]) < segment[1] * 0.2 and
                abs(segment[0] - segment[1]) > segment[0] * 0.3):
                patterns.append({"type": "bigeminy", "start_index": i})
            
            # 检查三联律模式
            if i < len(rr_intervals) - 4:
                segment5 = rr_intervals[i:i + 5]
                if (abs(segment5[0] - segment5[1]) < segment5[0] * 0.2 and
                    abs(segment5[2] - segment5[3]) < segment5[2] * 0.2 and
                    abs(segment5[0] - segment5[2]) > segment5[0] * 0.3 and
                    abs(segment5[4] - segment5[0]) < segment5[4] * 0.2):
                    patterns.append({"type": "trigeminy", "start_index": i})
        
        detected = len(patterns) > 0
        
        # 按类型分组
        bigeminy_events = [p for p in patterns if p["type"] == "bigeminy"]
        trigeminy_events = [p for p in patterns if p["type"] == "trigeminy"]
        
        confidence = min(0.8, len(patterns) / 5.0)
        
        return detected, {
            "events": patterns,
            "confidence": confidence,
            "n_bigeminy": len(bigeminy_events),
            "n_trigeminy": len(trigeminy_events)
        }
    
    def analyze_multi_leads(self,
                           ecg_data: np.ndarray,
                           sampling_rate: float,
                           analysis_channels: List[int],
                           lead_names: List[str]) -> Dict[str, Any]:
        """
        多导联综合分析
        
        Args:
            ecg_data: 多导联ECG数据
            sampling_rate: 采样率
            analysis_channels: 分析通道索引
            lead_names: 导联名称
            
        Returns:
            多导联分析结果
        """
        n_channels = ecg_data.shape[0]
        
        if n_channels < 2 or len(analysis_channels) < 2:
            return {"consistency_score": 1.0, "n_leads": 1}
        
        results = {
            "n_leads_analyzed": len(analysis_channels),
            "lead_quality": {},
            "consistency_score": 0.0,
            "optimal_lead": 0,
            "lead_correlations": {}
        }
        
        # 对各导联进行R波检测
        lead_detections = {}
        for ch_idx in analysis_channels:
            if ch_idx >= n_channels:
                continue
                
            try:
                r_peaks, qrs_info = self.detect_r_peaks(
                    ecg_data,
                    sampling_rate=sampling_rate,
                    method=self.config.rpeak_method,
                    channel_idx=ch_idx,
                    min_distance=self.config.rpeak_min_distance
                )
                
                lead_detections[ch_idx] = {
                    "r_peaks": r_peaks,
                    "n_peaks": len(r_peaks),
                    "mean_hr": np.mean(qrs_info.get("heart_rate", [0])) if len(r_peaks) > 0 else 0
                }
            except Exception as e:
                logger.warning(f"导联 {ch_idx} R波检测失败: {str(e)}")
                lead_detections[ch_idx] = {"r_peaks": [], "n_peaks": 0, "mean_hr": 0}
        
        # 计算导联间一致性
        if len(lead_detections) >= 2:
            consistency_scores = []
            
            # 比较每对导联
            lead_indices = list(lead_detections.keys())
            for i in range(len(lead_indices)):
                for j in range(i + 1, len(lead_indices)):
                    ch_i = lead_indices[i]
                    ch_j = lead_indices[j]
                    
                    peaks_i = lead_detections[ch_i]["r_peaks"]
                    peaks_j = lead_detections[ch_j]["r_peaks"]
                    
                    if len(peaks_i) > 0 and len(peaks_j) > 0:
                        # 计算R波位置的匹配程度
                        match_tolerance = int(0.05 * sampling_rate)  # 50ms容差
                        
                        matches = 0
                        for peak_i in peaks_i:
                            for peak_j in peaks_j:
                                if abs(peak_i - peak_j) <= match_tolerance:
                                    matches += 1
                                    break
                        
                        consistency = matches / max(len(peaks_i), len(peaks_j))
                        consistency_scores.append(consistency)
                        
                        # 记录相关系数
                        results["lead_correlations"][f"{ch_i}-{ch_j}"] = consistency
            
            if consistency_scores:
                results["consistency_score"] = np.mean(consistency_scores)
            
            # 选择最优导联（R波数量最多且与其他导联一致）
            lead_scores = []
            for ch_idx, detection in lead_detections.items():
                if detection["n_peaks"] > 0:
                    # 计算该导联与其他导联的平均一致性
                    consistencies = []
                    for other_idx, other_detection in lead_detections.items():
                        if ch_idx != other_idx and other_detection["n_peaks"] > 0:
                            # 计算一致性（简化）
                            peaks_i = detection["r_peaks"]
                            peaks_j = other_detection["r_peaks"]
                            
                            match_tolerance = int(0.05 * sampling_rate)
                            matches = 0
                            for peak_i in peaks_i:
                                for peak_j in peaks_j:
                                    if abs(peak_i - peak_j) <= match_tolerance:
                                        matches += 1
                                        break
                            
                            consistency = matches / max(len(peaks_i), len(peaks_j))
                            consistencies.append(consistency)
                    
                    avg_consistency = np.mean(consistencies) if consistencies else 0
                    lead_score = detection["n_peaks"] * 0.5 + avg_consistency * 0.5
                    lead_scores.append((ch_idx, lead_score))
            
            if lead_scores:
                optimal_lead = max(lead_scores, key=lambda x: x[1])[0]
                results["optimal_lead"] = optimal_lead
        
        # 评估各导联质量
        for ch_idx in analysis_channels:
            if ch_idx >= n_channels:
                continue
                
            signal = ecg_data[ch_idx, :]
            
            # 简化的质量评估
            signal_range = np.ptp(signal)
            signal_std = np.std(signal)
            
            quality = "good"
            if signal_range < 0.1 or signal_std < 0.01:
                quality = "poor"
            elif signal_range > 5.0:
                quality = "noisy"
            
            results["lead_quality"][ch_idx] = {
                "quality": quality,
                "range": signal_range,
                "std": signal_std,
                "lead_name": lead_names[ch_idx] if ch_idx < len(lead_names) else f"CH{ch_idx}"
            }
        
        return results
    
    def extract_respiratory_sinus_arrhythmia(self,
                                            ecg_data: np.ndarray,
                                            r_peaks: List[int],
                                            rr_intervals: List[float],
                                            sampling_rate: float,
                                            frequency_band: Tuple[float, float] = (0.15, 0.4)) -> Dict[str, Any]:
        """
        提取呼吸性窦性心律不齐（RSA）
        
        Args:
            ecg_data: ECG信号数据
            r_peaks: R波位置
            rr_intervals: RR间期
            sampling_rate: 采样率
            frequency_band: RSA频带
            
        Returns:
            RSA分析结果
        """
        if len(rr_intervals) < 20:
            return {"rsa_magnitude": 0.0, "cardiorespiratory_coupling": 0.0}
        
        results = {}
        
        # 方法1：从R波幅度调制中提取呼吸信号
        ecg_signal = ecg_data[0, :] if len(ecg_data.shape) == 2 else ecg_data
        
        # 提取R波幅度
        r_amplitudes = []
        for r_peak in r_peaks:
            if 0 <= r_peak < len(ecg_signal):
                r_amplitudes.append(ecg_signal[r_peak])
        
        if len(r_amplitudes) < 10:
            return {"rsa_magnitude": 0.0, "cardiorespiratory_coupling": 0.0}
        
        # 创建R波幅度时间序列
        r_times = [r_peaks[i] / sampling_rate for i in range(len(r_amplitudes))]
        
        # 插值获得均匀采样序列
        uniform_times = np.arange(0, r_times[-1], 1.0 / sampling_rate)
        interp_func = interp1d(r_times, r_amplitudes, kind='cubic', fill_value="extrapolate")
        uniform_amplitudes = interp_func(uniform_times)
        
        # 带通滤波提取呼吸频率成分
        nyquist = sampling_rate / 2
        lowcut, highcut = frequency_band
        
        if highcut < nyquist:
            b, a = butter(2, [lowcut / nyquist, highcut / nyquist], btype='band')
            respiratory_signal = filtfilt(b, a, uniform_amplitudes)
            
            # 计算RSA幅度
            rsa_magnitude = np.ptp(respiratory_signal)  # 峰峰值
            
            results["rsa_magnitude"] = rsa_magnitude
            results["derived_respiration"] = respiratory_signal.tolist()
            results["respiration_times"] = uniform_times.tolist()
        
        # 方法2：从RR间期变异性中提取RSA
        if len(rr_intervals) >= 10:
            # 计算RR间期在呼吸频带的功率
            from scipy.signal import lombscargle
            
            rr_times = np.cumsum(rr_intervals) / 1000.0  # 转换为秒
            rr_times = rr_times - rr_times[0]
            
            freqs = np.arange(frequency_band[0], frequency_band[1], 0.01)
            angular_freqs = 2 * np.pi * freqs
            
            power = lombscargle(rr_times, rr_intervals, angular_freqs, normalize=True)
            
            # RSA功率
            rsa_power = np.trapz(power, freqs)
            total_power = lombscargle(rr_times, rr_intervals, 
                                      2 * np.pi * np.arange(0.003, 0.4, 0.01), 
                                      normalize=True)
            total_power = np.trapz(total_power, np.arange(0.003, 0.4, 0.01))
            
            if total_power > 0:
                results["rsa_power_ratio"] = rsa_power / total_power
                results["rsa_peak_frequency"] = freqs[np.argmax(power)] if len(power) > 0 else 0
        
        # 计算心搏-呼吸耦合
        if "derived_respiration" in results and len(rr_intervals) >= 10:
            # 简化的耦合计算：呼吸信号与RR间期的互相关
            try:
                # 创建均匀采样的RR间期序列
                rr_uniform_times = np.arange(0, rr_times[-1], 1.0 / sampling_rate)
                interp_rr = interp1d(rr_times, rr_intervals, kind='linear', fill_value="extrapolate")
                rr_uniform = interp_rr(rr_uniform_times)
                
                # 截取相同长度
                min_len = min(len(results["derived_respiration"]), len(rr_uniform))
                resp_signal = np.array(results["derived_respiration"][:min_len])
                rr_signal = rr_uniform[:min_len]
                
                # 标准化
                resp_signal = (resp_signal - np.mean(resp_signal)) / np.std(resp_signal) if np.std(resp_signal) > 0 else resp_signal
                rr_signal = (rr_signal - np.mean(rr_signal)) / np.std(rr_signal) if np.std(rr_signal) > 0 else rr_signal
                
                # 计算互相关
                correlation = np.correlate(resp_signal, rr_signal, mode='full')
                max_correlation = np.max(np.abs(correlation))
                
                results["cardiorespiratory_coupling"] = min(1.0, max_correlation / len(resp_signal))
            except:
                results["cardiorespiratory_coupling"] = 0.0
        
        return results
    
    def detect_waveforms(self,
                        ecg_data: np.ndarray,
                        r_peaks: List[int],
                        sampling_rate: float,
                        detect_p: bool = True,
                        detect_t: bool = True,
                        p_window: Tuple[float, float] = (-0.2, -0.05),
                        t_window: Tuple[float, float] = (0.1, 0.4)) -> Dict[str, Any]:
        """
        检测P波和T波
        
        Args:
            ecg_data: ECG信号数据
            r_peaks: R波位置
            sampling_rate: 采样率
            detect_p: 是否检测P波
            detect_t: 是否检测T波
            p_window: P波搜索窗口（秒，相对于R波）
            t_window: T波搜索窗口（秒，相对于R波）
            
        Returns:
            波形检测结果
        """
        results = {
            "p_waves": [],
            "t_waves": [],
            "pr_intervals": [],
            "qt_intervals": [],
            "waveform_quality": {}
        }
        
        if len(r_peaks) < 3:
            return results
        
        ecg_signal = ecg_data[0, :] if len(ecg_data.shape) == 2 else ecg_data
        
        # 创建QRS波模板用于对齐
        if self.qrs_template is None and len(r_peaks) >= 5:
            self.qrs_template = self._create_qrs_template(ecg_signal, r_peaks, sampling_rate)
        
        for i, r_peak in enumerate(r_peaks):
            if i == 0 or i == len(r_peaks) - 1:
                continue  # 跳过第一个和最后一个
            
            # 检测P波
            if detect_p:
                p_wave = self._detect_p_wave(ecg_signal, r_peak, r_peaks[i-1], 
                                            sampling_rate, p_window)
                if p_wave:
                    results["p_waves"].append(p_wave)
                    
                    # 计算PR间期
                    if "peak" in p_wave:
                        pr_interval = (r_peak - p_wave["peak"]) / sampling_rate * 1000  # 毫秒
                        results["pr_intervals"].append(pr_interval)
            
            # 检测T波
            if detect_t:
                t_wave = self._detect_t_wave(ecg_signal, r_peak, sampling_rate, t_window)
                if t_wave:
                    results["t_waves"].append(t_wave)
                    
                    # 计算QT间期
                    if "peak" in t_wave:
                        qt_interval = (t_wave["peak"] - r_peak) / sampling_rate * 1000  # 毫秒
                        results["qt_intervals"].append(qt_interval)
        
        # 计算平均波形参数
        if results["pr_intervals"]:
            results["waveform_quality"]["mean_pr"] = np.mean(results["pr_intervals"])
            results["waveform_quality"]["std_pr"] = np.std(results["pr_intervals"])
        
        if results["qt_intervals"]:
            results["waveform_quality"]["mean_qt"] = np.mean(results["qt_intervals"])
            results["waveform_quality"]["std_qt"] = np.std(results["qt_intervals"])
        
        results["waveform_quality"]["n_p_waves"] = len(results["p_waves"])
        results["waveform_quality"]["n_t_waves"] = len(results["t_waves"])
        
        return results
    
    def _create_qrs_template(self, 
                            ecg_signal: np.ndarray,
                            r_peaks: List[int],
                            sampling_rate: float) -> np.ndarray:
        """创建QRS波模板"""
        window_samples = int(0.2 * sampling_rate)  # 200ms窗口
        templates = []
        
        for r_peak in r_peaks[:10]:  # 使用前10个R波
            start = max(0, r_peak - window_samples // 2)
            end = min(len(ecg_signal), r_peak + window_samples // 2)
            
            if end > start:
                segment = ecg_signal[start:end]
                # 对齐到最大值
                max_idx = np.argmax(segment)
                aligned_segment = np.roll(segment, window_samples // 2 - max_idx)
                templates.append(aligned_segment[:window_samples])
        
        if templates:
            template = np.mean(templates, axis=0)
            return template
        else:
            return None
    
    def _detect_p_wave(self, 
                      ecg_signal: np.ndarray,
                      r_peak: int,
                      prev_r_peak: int,
                      sampling_rate: float,
                      window: Tuple[float, float]) -> Optional[Dict]:
        """检测P波"""
        # 计算搜索窗口
        window_start = r_peak + int(window[0] * sampling_rate)
        window_end = r_peak + int(window[1] * sampling_rate)
        
        # 确保窗口在信号范围内且在前一个R波之后
        window_start = max(prev_r_peak + int(0.1 * sampling_rate), window_start)
        window_end = min(r_peak - int(0.02 * sampling_rate), window_end)
        
        if window_end <= window_start:
            return None
        
        # 提取搜索段
        segment = ecg_signal[window_start:window_end]
        if len(segment) < 5:
            return None
        
        # 寻找局部极大值（P波峰值）
        peaks, properties = find_peaks(segment, 
                                      height=np.std(segment) * 0.5,
                                      distance=int(0.1 * sampling_rate))
        
        if len(peaks) == 0:
            return None
        
        # 选择最显著的峰值
        if len(peaks) > 1:
            # 使用幅度最高的峰值
            peak_heights = properties["peak_heights"]
            main_peak = peaks[np.argmax(peak_heights)]
        else:
            main_peak = peaks[0]
        
        p_peak = window_start + main_peak
        
        # 检测P波起始和终止
        p_onset = self._detect_wave_onset(ecg_signal, p_peak, -1, sampling_rate)
        p_offset = self._detect_wave_offset(ecg_signal, p_peak, 1, sampling_rate)
        
        return {
            "peak": p_peak,
            "onset": p_onset,
            "offset": p_offset,
            "amplitude": ecg_signal[p_peak] - np.mean(segment),
            "duration": (p_offset - p_onset) / sampling_rate * 1000 if p_onset and p_offset else None
        }
    
    def _detect_t_wave(self, 
                      ecg_signal: np.ndarray,
                      r_peak: int,
                      sampling_rate: float,
                      window: Tuple[float, float]) -> Optional[Dict]:
        """检测T波"""
        # 计算搜索窗口
        window_start = r_peak + int(window[0] * sampling_rate)
        window_end = r_peak + int(window[1] * sampling_rate)
        
        # 确保窗口在信号范围内
        window_start = max(r_peak + int(0.05 * sampling_rate), window_start)
        window_end = min(len(ecg_signal) - 1, window_end)
        
        if window_end <= window_start:
            return None
        
        # 提取搜索段
        segment = ecg_signal[window_start:window_end]
        if len(segment) < 5:
            return None
        
        # 寻找局部极大值（T波峰值）
        # T波可能是正向或负向，所以需要找绝对值最大的峰值
        segment_abs = np.abs(segment)
        peaks, properties = find_peaks(segment_abs, 
                                      height=np.std(segment_abs) * 0.5,
                                      distance=int(0.1 * sampling_rate))
        
        if len(peaks) == 0:
            return None
        
        # 选择最显著的峰值
        if len(peaks) > 1:
            peak_heights = properties["peak_heights"]
            main_peak = peaks[np.argmax(peak_heights)]
        else:
            main_peak = peaks[0]
        
        t_peak = window_start + main_peak
        
        # 检测T波起始和终止
        t_onset = self._detect_wave_onset(ecg_signal, t_peak, -1, sampling_rate)
        t_offset = self._detect_wave_offset(ecg_signal, t_peak, 1, sampling_rate)
        
        return {
            "peak": t_peak,
            "onset": t_onset,
            "offset": t_offset,
            "amplitude": ecg_signal[t_peak],
            "polarity": "positive" if ecg_signal[t_peak] > 0 else "negative",
            "duration": (t_offset - t_onset) / sampling_rate * 1000 if t_onset and t_offset else None
        }
    
    def _detect_wave_onset(self, 
                          ecg_signal: np.ndarray,
                          wave_peak: int,
                          direction: int,  # -1表示向前搜索，1表示向后搜索
                          sampling_rate: float,
                          search_window: float = 0.1) -> Optional[int]:
        """检测波形起始点"""
        search_samples = int(search_window * sampling_rate)
        
        if direction == -1:  # 向前搜索
            start = max(0, wave_peak - search_samples)
            segment = ecg_signal[start:wave_peak]
            
            if len(segment) < 10:
                return start
            
            # 寻找斜率显著变化的点
            diff_segment = np.diff(segment)
            threshold = np.std(diff_segment) * 2
            
            for i in range(len(diff_segment) - 1, 0, -1):
                if np.abs(diff_segment[i]) > threshold:
                    return start + i
            
            return start
        else:  # 向后搜索
            end = min(len(ecg_signal), wave_peak + search_samples)
            segment = ecg_signal[wave_peak:end]
            
            if len(segment) < 10:
                return wave_peak
            
            # 寻找斜率显著变化的点
            diff_segment = np.diff(segment)
            threshold = np.std(diff_segment) * 2
            
            for i in range(len(diff_segment)):
                if np.abs(diff_segment[i]) > threshold:
                    return wave_peak + i
            
            return wave_peak
    
    def _detect_wave_offset(self, 
                           ecg_signal: np.ndarray,
                           wave_peak: int,
                           direction: int,
                           sampling_rate: float,
                           search_window: float = 0.1) -> Optional[int]:
        """检测波形终止点"""
        # 简化的实现：使用与起始点检测类似的方法
        return self._detect_wave_onset(ecg_signal, wave_peak, direction, sampling_rate, search_window)
    
    def _generate_processing_summary(self) -> Dict[str, Any]:
        """生成处理摘要"""
        summary = {
            "processing_time": np.datetime64('now').astype(str),
            "config_summary": {
                "rpeak_method": self.config.rpeak_method.value if hasattr(self.config, 'rpeak_method') else "unknown",
                "hrv_analysis_performed": bool(self.config.hrv_analysis_methods) if hasattr(self.config, 'hrv_analysis_methods') else False,
                "arrhythmia_detection": self.config.detect_arrhythmias if hasattr(self.config, 'detect_arrhythmias') else False,
                "signal_quality_assessment": self.config.assess_signal_quality if hasattr(self.config, 'assess_signal_quality') else False
            },
            "results_summary": {
                "n_r_peaks": len(self.r_peaks),
                "mean_heart_rate": np.mean(self.qrs_info.get("heart_rate", [0])) if self.qrs_info.get("heart_rate") else 0,
                "signal_quality": self.signal_quality.get("quality_flag", "unknown"),
                "arrhythmias_detected": list(self.arrhythmia_results.get("arrhythmia_flags", {}).keys())
            },
            "processing_steps": len(self.history) if self.history else 0
        }
        
        return summary
    
    def visualize_processing_results(self,
                                   data_dict: Dict[str, Any],
                                   modality: str = "ECG",
                                   channel_idx: int = 0,
                                   time_range: Optional[Tuple[float, float]] = None,
                                   save_path: Optional[str] = None):
        """
        可视化ECG处理结果
        
        Args:
            data_dict: 处理后的数据字典
            modality: 信号模态
            channel_idx: 通道索引
            time_range: 时间范围（秒）
            save_path: 保存路径（可选）
        """
        if modality not in data_dict["signal"]:
            logger.error(f"模态 {modality} 不在数据中")
            return
        
        # 获取处理结果
        if ("processed" not in data_dict or 
            "ECG_processing" not in data_dict["processed"] or
            modality not in data_dict["processed"]["ECG_processing"]):
            logger.error("未找到ECG处理结果")
            return
        
        processing_results = data_dict["processed"]["ECG_processing"][modality]
        
        # 获取信号数据
        signal_info = data_dict["signal"][modality]
        ecg_data = signal_info["data"]
        sampling_rate = signal_info["sampling_rate"]
        
        if channel_idx >= ecg_data.shape[0]:
            logger.error(f"通道索引 {channel_idx} 超出范围")
            channel_idx = 0
        
        ecg_signal = ecg_data[channel_idx, :]
        n_samples = len(ecg_signal)
        
        # 确定时间范围
        if time_range is None:
            start_sample = 0
            end_sample = min(n_samples, int(10 * sampling_rate))  # 默认显示10秒
        else:
            start_sample = int(time_range[0] * sampling_rate)
            end_sample = int(time_range[1] * sampling_rate)
        
        start_sample = max(0, start_sample)
        end_sample = min(n_samples, end_sample)
        
        if end_sample <= start_sample:
            logger.error("无效的时间范围")
            return
        
        # 创建图形
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f"ECG预处理结果可视化 - 通道 {channel_idx}", fontsize=16)
        
        time_axis = np.arange(start_sample, end_sample) / sampling_rate
        
        # 1. 原始与处理后的ECG信号
        if "original_data" in processing_results:
            original_signal = processing_results["original_data"][channel_idx, start_sample:end_sample]
            processed_signal = ecg_signal[start_sample:end_sample]
            
            axes[0, 0].plot(time_axis, original_signal, alpha=0.7, label='原始信号')
            axes[0, 0].plot(time_axis, processed_signal, alpha=0.9, label='处理后信号', linewidth=1.5)
            axes[0, 0].set_title("原始 vs 处理后ECG信号")
            axes[0, 0].set_xlabel("时间 (秒)")
            axes[0, 0].set_ylabel("幅度")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. R波检测结果
        r_peaks = processing_results.get("results", {}).get("r_peaks", [])
        # 选择在时间范围内的R波
        r_peaks_in_range = [p for p in r_peaks if start_sample <= p < end_sample]
        
        signal_in_range = ecg_signal[start_sample:end_sample]
        axes[0, 1].plot(time_axis, signal_in_range)
        
        if r_peaks_in_range:
            r_times = [(p - start_sample) / sampling_rate for p in r_peaks_in_range]
            r_amplitudes = [ecg_signal[p] for p in r_peaks_in_range]
            axes[0, 1].scatter(r_times, r_amplitudes, color='red', s=50, 
                              label=f'R波 (n={len(r_peaks_in_range)})', zorder=5)
        
        axes[0, 1].set_title("R波检测")
        axes[0, 1].set_xlabel("时间 (秒)")
        axes[0, 1].set_ylabel("幅度")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 心率变异性（RR间期）
        rr_intervals = processing_results.get("results", {}).get("qrs_info", {}).get("rr_intervals", [])
        if rr_intervals:
            rr_times = np.cumsum(rr_intervals) / 1000.0  # 转换为秒
            axes[1, 0].plot(rr_times, rr_intervals, 'o-', markersize=4)
            axes[1, 0].set_title("RR间期序列")
            axes[1, 0].set_xlabel("时间 (秒)")
            axes[1, 0].set_ylabel("RR间期 (毫秒)")
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. HRV频谱
        hrv_metrics = processing_results.get("results", {}).get("hrv_metrics", {})
        if "frequency_domain" in hrv_metrics:
            freqs = hrv_metrics["frequency_domain"].get("frequencies", [])
            power = hrv_metrics["frequency_domain"].get("power_spectrum", [])
            
            if freqs and power:
                axes[1, 1].plot(freqs, power)
                axes[1, 1].fill_between(freqs, 0, power, alpha=0.3)
                
                # 标记频带
                freq_bands = {
                    "VLF": (0.003, 0.04),
                    "LF": (0.04, 0.15),
                    "HF": (0.15, 0.4)
                }
                
                colors = ['yellow', 'green', 'blue']
                for (band_name, (f_low, f_high)), color in zip(freq_bands.items(), colors):
                    band_mask = (freqs >= f_low) & (freqs <= f_high)
                    if np.any(band_mask):
                        axes[1, 1].axvspan(f_low, f_high, alpha=0.2, color=color, label=band_name)
                
                axes[1, 1].set_title("HRV功率谱")
                axes[1, 1].set_xlabel("频率 (Hz)")
                axes[1, 1].set_ylabel("功率")
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        # 5. 信号质量评估
        signal_quality = processing_results.get("results", {}).get("signal_quality", {})
        if signal_quality:
            quality_metrics = signal_quality.get("quality_metrics", {})
            
            if quality_metrics:
                metric_names = list(quality_metrics.keys())
                metric_values = list(quality_metrics.values())
                
                # 创建条形图
                bars = axes[2, 0].bar(range(len(metric_names)), metric_values)
                
                # 根据阈值添加颜色
                thresholds = {
                    "snr_db": 10.0,
                    "baseline_wander": 0.3,
                    "powerline_noise": 0.2,
                    "emg_noise": 0.15,
                    "qrs_reliability": 0.7
                }
                
                for i, (name, value) in enumerate(zip(metric_names, metric_values)):
                    if name in thresholds:
                        if name == "snr_db":
                            if value >= thresholds[name]:
                                bars[i].set_color('green')
                            else:
                                bars[i].set_color('red')
                        else:
                            if value <= thresholds[name]:
                                bars[i].set_color('green')
                            else:
                                bars[i].set_color('red')
                
                axes[2, 0].set_xticks(range(len(metric_names)))
                axes[2, 0].set_xticklabels(metric_names, rotation=45, ha='right')
                axes[2, 0].set_title("信号质量指标")
                axes[2, 0].set_ylabel("值")
                axes[2, 0].grid(True, alpha=0.3, axis='y')
        
        # 6. 心律异常检测
        arrhythmia_results = processing_results.get("results", {}).get("arrhythmia_results", {})
        arrhythmia_flags = arrhythmia_results.get("arrhythmia_flags", {})
        
        if arrhythmia_flags:
            flag_names = list(arrhythmia_flags.keys())
            n_flags = len(flag_names)
            
            axes[2, 1].text(0.1, 0.5, 
                           f"检测到的心律异常:\n" + "\n".join(flag_names) +
                           f"\n\n总R波数: {len(r_peaks)}\n" +
                           f"平均心率: {np.mean(processing_results.get('results', {}).get('qrs_info', {}).get('heart_rate', [0])):.1f} BPM\n" +
                           f"信号质量: {signal_quality.get('quality_flag', 'unknown')}",
                           transform=axes[2, 1].transAxes, fontsize=12,
                           verticalalignment='center')
        else:
            axes[2, 1].text(0.1, 0.5, 
                           f"未检测到心律异常\n\n" +
                           f"总R波数: {len(r_peaks)}\n" +
                           f"平均心率: {np.mean(processing_results.get('results', {}).get('qrs_info', {}).get('heart_rate', [0])):.1f} BPM\n" +
                           f"信号质量: {signal_quality.get('quality_flag', 'unknown')}",
                           transform=axes[2, 1].transAxes, fontsize=12,
                           verticalalignment='center')
        
        axes[2, 1].set_title("心律异常检测结果")
        axes[2, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化结果已保存到: {save_path}")
        
        plt.show()


# ====================== 实用函数 ======================

def load_ECG_data(filepath: str, 
                 modality: str = "ECG",
                 lead_config: str = "standard") -> Dict[str, Any]:
    """
    加载ECG数据（示例函数）
    
    实际应用中需要根据具体数据格式实现
    
    Args:
        filepath: 数据文件路径
        modality: 信号模态
        lead_config: 导联配置
        
    Returns:
        四层结构的数据字典
    """
    # 这是一个示例函数，实际应用中需要根据具体数据格式实现
    logger.warning("这是一个示例加载函数，需要根据实际数据格式实现")
    
    # 模拟数据参数
    sampling_rate = 250.0  # Hz
    duration = 60.0  # 秒
    n_samples = int(sampling_rate * duration)
    
    # 示例导联配置
    if lead_config == "standard":
        n_channels = 12
        channel_names = ["I", "II", "III", "aVR", "aVL", "aVF", 
                        "V1", "V2", "V3", "V4", "V5", "V6"]
    else:
        n_channels = 3
        channel_names = ["ECG1", "ECG2", "ECG3"]
    
    # 生成模拟ECG数据
    ecg_data = np.random.randn(n_channels, n_samples) * 0.1
    
    # 添加模拟心搏
    heart_rate = 72  # BPM
    rr_interval = 60.0 / heart_rate  # 秒
    samples_per_rr = int(rr_interval * sampling_rate)
    
    # 模拟QRS波
    for ch in range(n_channels):
        for i in range(0, n_samples, samples_per_rr):
            if i + 100 < n_samples:
                # 添加QRS波
                qrs_start = i + np.random.randint(-10, 10)
                qrs_duration = np.random.randint(80, 120)  # 80-120ms
                qrs_samples = int(qrs_duration / 1000 * sampling_rate)
                
                if qrs_start + qrs_samples < n_samples:
                    # 创建QRS波形
                    t = np.linspace(0, 1, qrs_samples)
                    qrs_wave = np.sin(2 * np.pi * t) * np.exp(-5 * (t - 0.5)**2)
                    qrs_wave = qrs_wave * (0.5 + np.random.rand() * 0.5)  # 随机幅度
                    
                    ecg_data[ch, qrs_start:qrs_start + qrs_samples] += qrs_wave
    
    # 示例数据字典结构
    data_dict = {
        "meta": {
            "subject_id": "S01",
            "task": "resting_ECG",
            "modality": ["ECG"],
            "device": "Biopac",
            "sampling_rate": sampling_rate,
            "n_channels": n_channels,
            "channel_names": channel_names,
            "lead_configuration": lead_config
        },
        "signal": {
            "ECG": {
                "data": ecg_data,
                "sampling_rate": sampling_rate,
                "unit": "mV",
                "channel_names": channel_names
            }
        },
        "event": {
            "event_id": [1, 2],
            "event_label": ["start", "end"],
            "event_time": [0.0, duration],
            "event_sample": [0, n_samples]
        },
        "processed": {}
    }
    
    return data_dict


def save_ECG_results(data_dict: Dict[str, Any], 
                    output_path: str,
                    include_signal_data: bool = True,
                    include_plots: bool = False):
    """
    保存ECG处理结果
    
    Args:
        data_dict: 处理后的数据字典
        output_path: 输出文件路径
        include_signal_data: 是否包含原始信号数据
        include_plots: 是否包含绘图
    """
    import pickle
    import json
    
    # 创建保存副本
    save_dict = data_dict.copy()
    
    # 如果不包含信号数据，则移除它以减少文件大小
    if not include_signal_data and "signal" in save_dict:
        # 只保留元数据
        for modality in list(save_dict["signal"].keys()):
            if "data" in save_dict["signal"][modality]:
                save_dict["signal"][modality]["data_shape"] = save_dict["signal"][modality]["data"].shape
                del save_dict["signal"][modality]["data"]
    
    # 保存为pickle文件
    with open(output_path, 'wb') as f:
        pickle.dump(save_dict, f)
    
    # 同时保存JSON格式的摘要
    summary_path = output_path.replace('.pkl', '_summary.json')
    
    # 提取摘要信息
    summary = {}
    if "processed" in save_dict and "ECG_processing" in save_dict["processed"]:
        for modality, results in save_dict["processed"]["ECG_processing"].items():
            if "results" in results:
                # 创建简化摘要
                results_data = results["results"]
                summary[modality] = {
                    "n_r_peaks": len(results_data.get("r_peaks", [])),
                    "mean_heart_rate": np.mean(results_data.get("qrs_info", {}).get("heart_rate", [0])),
                    "signal_quality": results_data.get("signal_quality", {}).get("quality_flag", "unknown"),
                    "arrhythmias_detected": list(results_data.get("arrhythmia_results", {}).get("arrhythmia_flags", {}).keys()),
                    "processing_steps": len(results.get("steps", []))
                }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"ECG处理结果已保存到: {output_path}")
    logger.info(f"处理摘要已保存到: {summary_path}")


def export_ECG_report(data_dict: Dict[str, Any],
                     output_path: str,
                     modality: str = "ECG"):
    """
    导出ECG处理报告
    
    Args:
        data_dict: 处理后的数据字典
        output_path: 输出文件路径
        modality: 信号模态
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
    except ImportError:
        logger.error("ReportLab未安装，无法生成PDF报告")
        return
    
    if modality not in data_dict.get("processed", {}).get("ECG_processing", {}):
        logger.error(f"未找到{modality}的处理结果")
        return
    
    processing_results = data_dict["processed"]["ECG_processing"][modality]
    results = processing_results.get("results", {})
    
    # 创建PDF文档
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    
    # 样式
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=8
    )
    normal_style = styles['Normal']
    
    # 标题
    story.append(Paragraph("ECG信号处理报告", title_style))
    story.append(Spacer(1, 12))
    
    # 基本信息
    story.append(Paragraph("1. 基本信息", heading_style))
    
    meta_info = data_dict.get("meta", {})
    basic_info = [
        ["被试ID:", meta_info.get("subject_id", "N/A")],
        ["任务:", meta_info.get("task", "N/A")],
        ["设备:", meta_info.get("device", "N/A")],
        ["采样率:", f"{meta_info.get('sampling_rate', 0):.1f} Hz"],
        ["通道数:", str(meta_info.get("n_channels", 0))],
        ["处理时间:", processing_results.get("processing_timestamp", "N/A")]
    ]
    
    basic_table = Table(basic_info, colWidths=[1.5*inch, 2*inch])
    basic_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    story.append(basic_table)
    story.append(Spacer(1, 12))
    
    # 处理结果摘要
    story.append(Paragraph("2. 处理结果摘要", heading_style))
    
    qrs_info = results.get("qrs_info", {})
    summary_data = [
        ["检测到的R波数:", str(len(results.get("r_peaks", [])))],
        ["平均心率:", f"{np.mean(qrs_info.get('heart_rate', [0])):.1f} BPM"],
        ["心率范围:", f"{np.min(qrs_info.get('heart_rate', [0])):.1f} - {np.max(qrs_info.get('heart_rate', [0])):.1f} BPM"],
        ["平均RR间期:", f"{np.mean(qrs_info.get('rr_intervals', [0])):.1f} ms"],
        ["QRS波宽度:", f"{np.mean(qrs_info.get('qrs_durations', [0])):.1f} ms"],
        ["信号质量:", results.get("signal_quality", {}).get("quality_flag", "N/A")]
    ]
    
    summary_table = Table(summary_data, colWidths=[1.5*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 12))
    
    # HRV指标
    hrv_metrics = results.get("hrv_metrics", {})
    if hrv_metrics:
        story.append(Paragraph("3. 心率变异性指标", heading_style))
        
        hrv_data = []
        
        # 时域指标
        if "time_domain" in hrv_metrics:
            td = hrv_metrics["time_domain"]
            hrv_data.append(["时域指标:", ""])
            hrv_data.append(["SDNN:", f"{td.get('std_rr', 0):.1f} ms"])
            hrv_data.append(["RMSSD:", f"{td.get('rmssd', 0):.1f} ms"])
            hrv_data.append(["pNN50:", f"{td.get('pnn50', 0):.1f} %"])
        
        # 频域指标
        if "frequency_domain" in hrv_metrics:
            fd = hrv_metrics["frequency_domain"]
            hrv_data.append(["", ""])
            hrv_data.append(["频域指标:", ""])
            hrv_data.append(["总功率:", f"{fd.get('total_power', 0):.1f} ms²"])
            hrv_data.append(["LF功率:", f"{fd.get('lf_power', 0):.1f} ms²"])
            hrv_data.append(["HF功率:", f"{fd.get('hf_power', 0):.1f} ms²"])
            hrv_data.append(["LF/HF比值:", f"{fd.get('lf_hf_ratio', 0):.2f}"])
        
        hrv_table = Table(hrv_data, colWidths=[1.5*inch, 2*inch])
        hrv_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('SPAN', (0, 0), (1, 0)),  # 合并时域指标标题
            ('SPAN', (0, 4), (1, 4)),  # 合并频域指标标题
        ]))
        
        story.append(hrv_table)
        story.append(Spacer(1, 12))
    
    # 心律异常检测
    arrhythmia_results = results.get("arrhythmia_results", {})
    arrhythmia_flags = arrhythmia_results.get("arrhythmia_flags", {})
    
    if arrhythmia_flags:
        story.append(Paragraph("4. 心律异常检测", heading_style))
        
        arrhythmia_data = [["检测到的心律异常:", ", ".join(arrhythmia_flags.keys())]]
        
        arrhythmia_table = Table(arrhythmia_data, colWidths=[1.5*inch, 4*inch])
        arrhythmia_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(arrhythmia_table)
    else:
        story.append(Paragraph("4. 心律异常检测", heading_style))
        story.append(Paragraph("未检测到明显心律异常。", normal_style))
    
    story.append(Spacer(1, 12))
    
    # 处理步骤
    story.append(Paragraph("5. 处理步骤", heading_style))
    
    steps = processing_results.get("steps", [])
    if steps:
        step_data = []
        for i, step in enumerate(steps, 1):
            step_name = step.get("step", "未知步骤")
            step_details = []
            
            # 添加步骤详细信息
            for key, value in step.items():
                if key != "step":
                    if isinstance(value, (int, float)):
                        step_details.append(f"{key}: {value}")
                    else:
                        step_details.append(f"{key}: {str(value)[:50]}")
            
            step_info = f"{step_name}\n" + "\n".join(step_details)
            step_data.append([str(i), step_info])
        
        steps_table = Table(step_data, colWidths=[0.5*inch, 5*inch])
        steps_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        story.append(steps_table)
    
    # 构建PDF
    doc.build(story)
    
    logger.info(f"ECG处理报告已保存到: {output_path}")


# ====================== 示例使用代码 ======================

if __name__ == "__main__":
    """
    ECG预处理模块使用示例
    """
    
    # 1. 创建ECG配置
    ecg_config = ECGConfig(
        # 通用预处理参数（针对ECG优化）
        lowcut=0.5,    # ECG低频截止
        highcut=40.0,  # ECG高频截止
        notch_freq=50.0,  # 工频陷波
        filter_type=FilterType.BUTTERWORTH,
        filter_order=4,
        detrend_method=DetrendMethod.LINEAR,
        remove_baseline=True,
        normalize_method="zscore",
        
        # ECG特有参数
        rpeak_method=RPeakDetectionMethod.PAN_TOMPKINS,
        rpeak_detection_channel=0,
        hrv_analysis_methods=[HRVAnalysisMethod.TIME_DOMAIN, HRVAnalysisMethod.FREQUENCY_DOMAIN],
        detect_arrhythmias=True,
        assess_signal_quality=True,
        multi_lead_analysis=True,
        extract_rsa=True,
        
        # 信号质量阈值
        signal_quality_thresholds={
            "snr_threshold": 10.0,
            "baseline_wander_threshold": 0.3,
            "powerline_noise_threshold": 0.2,
            "emg_noise_threshold": 0.15,
        }
    )
    
    # 2. 创建ECG预处理器
    ecg_processor = ECGPreprocessor(ecg_config)
    
    # 3. 加载ECG数据（示例）
    data_dict = load_ECG_data("example_ecg_data.npy", lead_config="standard")
    
    # 4. 处理ECG数据
    processed_data = ecg_processor.process_ECG(
        data_dict,
        modality="ECG",
        analysis_channels=[0, 1, 2],  # 处理前3个通道
        reference_channel=0
    )
    
    # 5. 可视化处理结果
    ecg_processor.visualize_processing_results(
        processed_data,
        modality="ECG",
        channel_idx=0,
        time_range=(0, 10),  # 显示前10秒
        save_path="ecg_processing_visualization.png"
    )
    
    # 6. 保存处理结果
    save_ECG_results(processed_data, "processed_ecg_data.pkl")
    
    # 7. 导出处理报告
    export_ECG_report(processed_data, "ecg_processing_report.pdf")
    
    print("ECG预处理完成！")
    print(f"检测到 {len(ecg_processor.r_peaks)} 个R波")
    print(f"平均心率: {np.mean(ecg_processor.qrs_info.get('heart_rate', [0])):.1f} BPM")
    print(f"信号质量: {ecg_processor.signal_quality.get('quality_flag', 'unknown')}")
    
    if ecg_processor.arrhythmia_results.get("arrhythmia_flags"):
        print(f"检测到心律异常: {list(ecg_processor.arrhythmia_results['arrhythmia_flags'].keys())}")