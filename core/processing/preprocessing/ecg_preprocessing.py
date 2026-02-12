# -*- coding: utf-8 -*-
"""
ECG信号预处理模块
专为心电图信号设计，包含专用滤波和信号质量评估
"""

import numpy as np
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt, iirnotch, welch
from scipy.interpolate import interp1d
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import matplotlib.pyplot as plt

# 导入通用预处理模块
from preprocessing import GeneralPreprocessor, PreprocessingConfig, FilterType, WaveletType, DetrendMethod

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 禁用特定警告
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ====================== ECG专用枚举和配置 ======================

class ECGQualityFlag(Enum):
    """ECG信号质量标志枚举"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class ECGConfig(PreprocessingConfig):
    """
    ECG专用预处理配置类
    扩展通用配置，添加ECG特有参数
    """
    # ECG滤波参数（覆盖通用参数）
    ecg_lowcut: float = 0.5      # ECG低频截止（Hz） - 去除基线漂移
    ecg_highcut: float = 40.0    # ECG高频截止（Hz） - 去除肌电噪声，保护QRS波
    ecg_notch_freq: float = 50.0  # ECG工频陷波频率（Hz）
    
    # ECG特定参数
    protect_qrs_wave: bool = True  # 是否特别保护QRS波形态
    qrs_enhancement: bool = True   # 是否增强QRS波（便于后续检测）
    qrs_enhancement_band: Tuple[float, float] = (5.0, 15.0)  # QRS增强频带
    
    # 信号质量评估参数
    assess_signal_quality: bool = True
    signal_quality_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "snr_threshold": 10.0,      # 信噪比阈值（dB）
            "baseline_wander_threshold": 0.3,  # 基线漂移阈值
            "powerline_noise_threshold": 0.2,  # 工频干扰阈值
            "emg_noise_threshold": 0.15,  # 肌电噪声阈值
        }
    )
    
    # 坏段检测参数
    detect_bad_segments: bool = True
    bad_segment_min_duration: float = 1.0  # 坏段最小持续时间（秒）
    
    # 多导联分析参数
    multi_lead_consistency: bool = True  # 检查多导联一致性
    
    # 输出参数
    include_quality_report: bool = True
    save_intermediate_results: bool = False


# ====================== ECG专用预处理器类 ======================

class ECGPreprocessor(GeneralPreprocessor):
    """
    ECG信号专用预处理器
    专注于信号预处理和信号质量评估
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
        self.signal_quality = {}
        self.bad_segments = []
        self.quality_report = {}
        
    def process_ECG(self, 
                   data_dict: Dict[str, Any], 
                   modality: str = "ECG",
                   channels: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        ECG专用预处理流程（仅信号预处理）
        
        Args:
            data_dict: 四层数据字典
            modality: 要处理的信号模态（默认为"ECG"）
            channels: 用于分析的通道索引列表，None表示所有通道
            
        Returns:
            更新后的四层数据字典，包含ECG预处理结果和信号质量评估
        """
        # 验证ECG输入结构
        self._validate_ECG_input(data_dict, modality)
        
        # 创建ECG处理历史记录
        process_record = {
            "modality": modality,
            "channels": channels,
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
        if "ECG_preprocessing" not in data_dict["processed"]:
            data_dict["processed"]["ECG_preprocessing"] = {}
        if modality not in data_dict["processed"]["ECG_preprocessing"]:
            data_dict["processed"]["ECG_preprocessing"][modality] = {
                "original_data": ecg_data.copy(),
                "original_sampling_rate": sampling_rate,
                "steps": [],
                "config": self.config.__dict__
            }
        
        # 选择处理通道
        n_channels = ecg_data.shape[0]
        if channels is None:
            channels = list(range(n_channels))
        
        process_record["channels_processed"] = channels
        
        # ========== ECG专用预处理步骤 ==========
        
        # 1. 应用ECG专用预处理（保护QRS波形态）
        processed_ecg = self.preprocess_ecg_signal(
            ecg_data, 
            sampling_rate, 
            channels
        )
        
        # 更新数据字典中的信号数据
        data_dict["signal"][modality]["data"] = processed_ecg
        
        process_record["steps"].append({
            "step": "ecg_specific_preprocessing",
            "lowcut": self.config.ecg_lowcut,
            "highcut": self.config.ecg_highcut,
            "notch_freq": self.config.ecg_notch_freq,
            "protect_qrs": self.config.protect_qrs_wave,
            "channels_processed": len(channels)
        })
        
        # 2. 信号质量评估
        if self.config.assess_signal_quality:
            self.signal_quality = self.assess_ecg_quality(
                processed_ecg,
                sampling_rate=sampling_rate,
                channels=channels
            )
            
            process_record["steps"].append({
                "step": "signal_quality_assessment",
                "quality_score": self.signal_quality.get("quality_score", 0),
                "quality_flag": self.signal_quality.get("quality_flag", "unknown"),
                "bad_segments": len(self.signal_quality.get("bad_segments", []))
            })
        
        # 3. 多导联一致性检查（仅多导联数据）
        if self.config.multi_lead_consistency and len(channels) > 1:
            try:
                consistency_results = self.check_multi_lead_consistency(
                    processed_ecg,
                    sampling_rate=sampling_rate,
                    channels=channels
                )
                
                data_dict["processed"]["ECG_preprocessing"][modality]["consistency_results"] = consistency_results
                
                process_record["steps"].append({
                    "step": "multi_lead_consistency_check",
                    "n_leads": len(channels),
                    "consistency_score": consistency_results.get("overall_consistency", 0),
                    "inconsistent_leads": consistency_results.get("inconsistent_leads", [])
                })
            except Exception as e:
                logger.error(f"多导联一致性检查失败: {str(e)}")
        
        # 4. 更新ECG预处理结果到数据字典
        ecg_results = {
            "signal_quality": self.signal_quality,
            "processing_config": self.config.__dict__,
            "processing_timestamp": np.datetime64('now').astype(str)
        }
        
        data_dict["processed"]["ECG_preprocessing"][modality]["results"] = ecg_results
        data_dict["processed"]["ECG_preprocessing"][modality]["steps"].append(process_record)
        
        # 保存中间结果（如果启用）
        if self.config.save_intermediate_results:
            intermediate_data = {
                "processed_ecg": processed_ecg,
                "channels_processed": channels
            }
            data_dict["processed"]["ECG_preprocessing"][modality]["intermediate_results"] = intermediate_data
        
        # 记录处理历史
        self.history.append(process_record)
        
        # 生成处理摘要
        summary = self._generate_processing_summary()
        data_dict["processed"]["ECG_preprocessing"][modality]["summary"] = summary
        
        logger.info(f"ECG预处理完成: {modality}, "
                   f"处理通道: {len(channels)}个, "
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
            logger.warning(f"ECG采样率较低: {sampling_rate} Hz，建议至少100 Hz以获得准确分析")
        elif sampling_rate > 2000:
            logger.warning(f"ECG采样率较高: {sampling_rate} Hz，可能包含过多高频噪声")
        
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
        对ECG信号进行专用预处理，特别保护QRS波形态
        
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
                # 使用零相位滤波器避免相位失真
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
                Q = 30.0
                b, a = iirnotch(self.config.ecg_notch_freq, Q, sampling_rate)
                signal = filtfilt(b, a, signal)
            
            # 4. 可选：QRS波增强（便于后续R波检测）
            if self.config.qrs_enhancement:
                signal = self.enhance_qrs_wave(signal, sampling_rate)
            
            # 5. 可选：小波去噪（如果启用）
            if self.config.wavelet_level > 0 and not self.config.use_adaptive_wavelet:
                signal = self.wavelet_denoising(
                    signal.reshape(1, -1),
                    wavelet=self.config.wavelet_type,
                    level=self.config.wavelet_level,
                    threshold_method=self.config.wavelet_threshold_method
                )[0, :]
            
            # 6. 去除离群值（如果启用）
            if self.config.remove_outliers:
                signal, _ = self.remove_outliers(
                    signal.reshape(1, -1),
                    threshold=self.config.outlier_threshold
                )
                signal = signal[0, :]
            
            processed_data[ch, :] = signal
        
        return processed_data
    
    def enhance_qrs_wave(self,
                        ecg_signal: np.ndarray,
                        sampling_rate: float) -> np.ndarray:
        """
        增强QRS波，便于后续R波检测
        
        Args:
            ecg_signal: 单个通道ECG信号
            sampling_rate: 采样率
            
        Returns:
            增强后的ECG信号
        """
        # 创建QRS增强滤波器（5-15 Hz带通）
        nyquist = sampling_rate / 2
        low_freq, high_freq = self.config.qrs_enhancement_band
        
        if low_freq < nyquist and high_freq < nyquist:
            b, a = butter(2, [low_freq / nyquist, high_freq / nyquist], btype='band')
            enhanced_signal = filtfilt(b, a, ecg_signal)
            
            # 可选：平方运算进一步放大QRS波
            if self.config.protect_qrs_wave:
                # 使用绝对值而不是平方，减少高频噪声放大
                enhanced_signal = np.abs(enhanced_signal)
            else:
                enhanced_signal = enhanced_signal ** 2
            
            return enhanced_signal
        else:
            return ecg_signal
    
    def assess_ecg_quality(self,
                          ecg_data: np.ndarray,
                          sampling_rate: float,
                          channels: List[int] = None) -> Dict[str, Any]:
        """
        评估ECG信号质量
        
        Args:
            ecg_data: ECG信号数据
            sampling_rate: 采样率
            channels: 要评估的通道列表
            
        Returns:
            信号质量评估结果字典
        """
        n_channels, n_samples = ecg_data.shape
        
        if channels is None:
            channels = list(range(n_channels))
        
        quality_results = {
            "quality_score": 0.0,
            "quality_flag": "unknown",
            "quality_metrics": {},
            "bad_segments": [],
            "recommendations": [],
            "per_channel_quality": {}
        }
        
        if n_channels == 0 or n_samples == 0:
            return quality_results
        
        all_channel_metrics = []
        
        for ch_idx in channels:
            if ch_idx >= n_channels:
                continue
                
            ecg_signal = ecg_data[ch_idx, :]
            
            # 计算各项质量指标
            metrics = {}
            
            # 1. 信噪比（SNR）
            metrics["snr_db"] = self._calculate_snr(ecg_signal, sampling_rate)
            
            # 2. 基线漂移程度
            metrics["baseline_wander"] = self._calculate_baseline_wander(ecg_signal, sampling_rate)
            
            # 3. 工频干扰程度
            metrics["powerline_noise"] = self._calculate_powerline_noise(ecg_signal, sampling_rate)
            
            # 4. 肌电噪声程度
            metrics["emg_noise"] = self._calculate_emg_noise(ecg_signal, sampling_rate)
            
            # 5. 信号缺失检测
            metrics["signal_loss"] = self._detect_signal_loss(ecg_signal)
            
            # 6. 信号动态范围
            metrics["dynamic_range"] = self._calculate_dynamic_range(ecg_signal)
            
            # 计算该通道质量分数
            channel_score = self._calculate_channel_quality_score(metrics)
            channel_flag = self._determine_quality_flag(channel_score)
            
            quality_results["per_channel_quality"][ch_idx] = {
                "quality_score": channel_score,
                "quality_flag": channel_flag,
                "metrics": metrics
            }
            
            all_channel_metrics.append(metrics)
        
        # 计算整体质量分数（所有通道平均）
        if all_channel_metrics:
            # 使用最差通道的质量分数（保守估计）
            channel_scores = [q["quality_score"] for q in quality_results["per_channel_quality"].values()]
            overall_score = np.min(channel_scores) if channel_scores else 0.0
            
            quality_results["quality_score"] = overall_score
            quality_results["quality_flag"] = self._determine_quality_flag(overall_score)
        
        # 检测坏段
        if self.config.detect_bad_segments:
            bad_segments = self._detect_bad_segments(ecg_data, sampling_rate, channels)
            quality_results["bad_segments"] = bad_segments
        
        # 生成建议
        recommendations = self._generate_quality_recommendations(quality_results)
        quality_results["recommendations"] = recommendations
        
        return quality_results
    
    def _calculate_snr(self, 
                      ecg_signal: np.ndarray,
                      sampling_rate: float) -> float:
        """计算信噪比（dB）"""
        # 简化版本：使用整个信号的统计特性估计SNR
        signal_power = np.var(ecg_signal)
        
        # 估计噪声功率：使用高频成分
        nyquist = sampling_rate / 2
        if nyquist > 100:
            # 计算高频噪声（100 Hz以上）
            b, a = butter(2, 100 / nyquist, btype='high')
            high_freq = filtfilt(b, a, ecg_signal)
            noise_power = np.var(high_freq)
        else:
            # 使用总体方差作为噪声估计
            noise_power = signal_power * 0.1  # 假设10%为噪声
        
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
        cutoff = 1.0  # Hz
        
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
    
    def _detect_signal_loss(self, ecg_signal: np.ndarray) -> float:
        """检测信号缺失"""
        # 检测信号幅度是否接近零
        signal_std = np.std(ecg_signal)
        signal_mean = np.mean(np.abs(ecg_signal))
        
        if signal_std < 0.01 or signal_mean < 0.01:
            return 1.0  # 完全信号缺失
        else:
            return 0.0
    
    def _calculate_dynamic_range(self, ecg_signal: np.ndarray) -> float:
        """计算信号动态范围（dB）"""
        signal_range = np.ptp(ecg_signal)  # 峰峰值
        if signal_range > 0:
            # 转换为dB，参考1 mV
            dynamic_range_db = 20 * np.log10(signal_range / 0.001)  # 0.001 = 1 mV
            return max(0, dynamic_range_db)
        else:
            return 0.0
    
    def _calculate_channel_quality_score(self, metrics: Dict[str, float]) -> float:
        """计算单个通道的质量分数（0-100）"""
        weights = {
            "snr_db": 0.25,
            "baseline_wander": 0.20,
            "powerline_noise": 0.15,
            "emg_noise": 0.15,
            "signal_loss": 0.25  # 信号缺失权重最高
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
                elif metric == "signal_loss":
                    norm_value = 1.0 - metrics[metric]  # 直接使用
                else:
                    norm_value = 0.5
                
                quality_score += norm_value * weight
        
        return quality_score * 100  # 转换为0-100分
    
    def _determine_quality_flag(self, quality_score: float) -> str:
        """根据质量分数确定质量标志"""
        if quality_score >= 80:
            return "excellent"
        elif quality_score >= 60:
            return "good"
        elif quality_score >= 40:
            return "fair"
        elif quality_score >= 20:
            return "poor"
        else:
            return "unusable"
    
    def _detect_bad_segments(self, 
                            ecg_data: np.ndarray,
                            sampling_rate: float,
                            channels: List[int],
                            window_size: float = 2.0) -> List[Dict]:
        """
        检测坏段
        
        Args:
            ecg_data: ECG信号数据
            sampling_rate: 采样率
            channels: 要检查的通道列表
            window_size: 分析窗口大小（秒）
            
        Returns:
            坏段信息列表
        """
        n_channels, n_samples = ecg_data.shape
        window_samples = int(window_size * sampling_rate)
        bad_segments = []
        
        # 滑动窗口分析
        for start_sample in range(0, n_samples, window_samples // 2):
            end_sample = min(start_sample + window_samples, n_samples)
            
            if end_sample - start_sample < window_samples // 4:
                continue
            
            segment_bad_channels = 0
            
            for ch_idx in channels:
                if ch_idx >= n_channels:
                    continue
                
                segment = ecg_data[ch_idx, start_sample:end_sample]
                
                # 计算窗口内信号质量指标
                segment_std = np.std(segment)
                segment_range = np.ptp(segment)
                
                # 判断是否为坏段
                is_bad = False
                
                # 1. 信号幅度过小
                if segment_range < 0.05:  # 阈值可调整
                    is_bad = True
                
                # 2. 信号标准差过小（可能为直线）
                if segment_std < 0.01:
                    is_bad = True
                
                # 3. 信号幅度过大（可能为电极脱落或运动伪影）
                if segment_range > 5.0:  # 阈值可调整
                    is_bad = True
                
                if is_bad:
                    segment_bad_channels += 1
            
            # 如果超过一半的通道在这个窗口内是坏的，则标记为坏段
            if segment_bad_channels > len(channels) // 2:
                bad_segments.append({
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "start_time": start_sample / sampling_rate,
                    "end_time": end_sample / sampling_rate,
                    "duration": (end_sample - start_sample) / sampling_rate,
                    "bad_channels": segment_bad_channels
                })
        
        return bad_segments
    
    def _generate_quality_recommendations(self, quality_results: Dict) -> List[str]:
        """生成质量改进建议"""
        recommendations = []
        
        # 检查整体质量
        overall_flag = quality_results.get("quality_flag", "unknown")
        overall_score = quality_results.get("quality_score", 0)
        
        if overall_flag in ["poor", "unusable"]:
            recommendations.append("信号质量较差，建议重新采集数据")
        
        # 检查各通道质量
        per_channel = quality_results.get("per_channel_quality", {})
        
        for ch_idx, channel_info in per_channel.items():
            ch_flag = channel_info.get("quality_flag", "unknown")
            metrics = channel_info.get("metrics", {})
            
            if ch_flag in ["poor", "unusable"]:
                recommendations.append(f"通道 {ch_idx} 信号质量差")
            
            # 具体问题建议
            if metrics.get("snr_db", 0) < 10:
                recommendations.append(f"通道 {ch_idx}: 低信噪比，检查电极接触")
            if metrics.get("baseline_wander", 0) > 0.3:
                recommendations.append(f"通道 {ch_idx}: 显著基线漂移，确保电极稳定")
            if metrics.get("powerline_noise", 0) > 0.2:
                recommendations.append(f"通道 {ch_idx}: 工频干扰明显，检查设备接地")
            if metrics.get("emg_noise", 0) > 0.15:
                recommendations.append(f"通道 {ch_idx}: 肌电噪声明显，请保持放松")
        
        # 去重
        recommendations = list(set(recommendations))
        
        return recommendations
    
    def check_multi_lead_consistency(self,
                                    ecg_data: np.ndarray,
                                    sampling_rate: float,
                                    channels: List[int]) -> Dict[str, Any]:
        """
        检查多导联一致性
        
        Args:
            ecg_data: 多导联ECG数据
            sampling_rate: 采样率
            channels: 要检查的通道列表
            
        Returns:
            一致性检查结果
        """
        n_channels = ecg_data.shape[0]
        
        if len(channels) < 2:
            return {
                "overall_consistency": 1.0,
                "n_leads": len(channels),
                "consistent": True
            }
        
        results = {
            "n_leads_checked": len(channels),
            "pairwise_correlations": {},
            "inconsistent_leads": [],
            "overall_consistency": 0.0
        }
        
        # 计算每对导联的相关系数
        consistency_scores = []
        
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                ch_i = channels[i]
                ch_j = channels[j]
                
                if ch_i >= n_channels or ch_j >= n_channels:
                    continue
                
                # 计算相关系数
                signal_i = ecg_data[ch_i, :]
                signal_j = ecg_data[ch_j, :]
                
                correlation = np.corrcoef(signal_i, signal_j)[0, 1]
                
                # 心电图导联间通常有正相关或负相关关系
                abs_correlation = abs(correlation)
                
                results["pairwise_correlations"][f"{ch_i}-{ch_j}"] = {
                    "correlation": correlation,
                    "abs_correlation": abs_correlation,
                    "consistent": abs_correlation > 0.3  # 阈值可调整
                }
                
                consistency_scores.append(abs_correlation)
        
        if consistency_scores:
            results["overall_consistency"] = np.mean(consistency_scores)
        
        # 识别不一致的导联
        for ch_idx in channels:
            if ch_idx >= n_channels:
                continue
            
            # 计算该导联与其他导联的平均相关性
            lead_correlations = []
            for key, corr_info in results["pairwise_correlations"].items():
                if str(ch_idx) in key:
                    lead_correlations.append(corr_info["abs_correlation"])
            
            if lead_correlations:
                avg_correlation = np.mean(lead_correlations)
                if avg_correlation < 0.2:  # 阈值可调整
                    results["inconsistent_leads"].append(ch_idx)
        
        return results
    
    def _generate_processing_summary(self) -> Dict[str, Any]:
        """生成处理摘要"""
        summary = {
            "processing_time": np.datetime64('now').astype(str),
            "config_summary": {
                "lowcut": self.config.ecg_lowcut,
                "highcut": self.config.ecg_highcut,
                "notch_freq": self.config.ecg_notch_freq,
                "assess_signal_quality": self.config.assess_signal_quality,
                "protect_qrs_wave": self.config.protect_qrs_wave
            },
            "results_summary": {
                "signal_quality": self.signal_quality.get("quality_flag", "unknown"),
                "quality_score": self.signal_quality.get("quality_score", 0),
                "bad_segments": len(self.signal_quality.get("bad_segments", []))
            },
            "processing_steps": len(self.history) if self.history else 0
        }
        
        return summary
    
    def visualize_preprocessing_results(self,
                                      data_dict: Dict[str, Any],
                                      modality: str = "ECG",
                                      channel_idx: int = 0,
                                      time_range: Optional[Tuple[float, float]] = None,
                                      save_path: Optional[str] = None):
        """
        可视化ECG预处理结果
        
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
            "ECG_preprocessing" not in data_dict["processed"] or
            modality not in data_dict["processed"]["ECG_preprocessing"]):
            logger.error("未找到ECG预处理结果")
            return
        
        processing_results = data_dict["processed"]["ECG_preprocessing"][modality]
        
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
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"ECG预处理结果可视化 - 通道 {channel_idx}", fontsize=16)
        
        time_axis = np.arange(start_sample, end_sample) / sampling_rate
        
        # 1. 原始与处理后的ECG信号
        if "original_data" in processing_results:
            original_signal = processing_results["original_data"][channel_idx, start_sample:end_sample]
            processed_signal = ecg_signal[start_sample:end_sample]
            
            axes[0, 0].plot(time_axis, original_signal, alpha=0.7, label='原始信号', linewidth=1)
            axes[0, 0].plot(time_axis, processed_signal, alpha=0.9, label='处理后信号', linewidth=1.5)
            axes[0, 0].set_title("原始 vs 处理后ECG信号")
            axes[0, 0].set_xlabel("时间 (秒)")
            axes[0, 0].set_ylabel("幅度 (mV)")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 频率响应（频谱）
        from scipy.fft import fft, fftfreq
        
        # 计算原始和处理后信号的频谱
        if "original_data" in processing_results:
            original_fft = np.abs(fft(original_signal))
            processed_fft = np.abs(fft(processed_signal))
            freq_axis = fftfreq(len(original_signal), 1/sampling_rate)
            
            # 只显示正频率
            pos_freq_mask = freq_axis >= 0
            freq_axis = freq_axis[pos_freq_mask]
            original_fft = original_fft[pos_freq_mask]
            processed_fft = processed_fft[pos_freq_mask]
            
            axes[0, 1].plot(freq_axis, original_fft, alpha=0.7, label='原始频谱')
            axes[0, 1].plot(freq_axis, processed_fft, alpha=0.9, label='处理后频谱')
            axes[0, 1].set_title("频率响应")
            axes[0, 1].set_xlabel("频率 (Hz)")
            axes[0, 1].set_ylabel("幅度")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xlim([0, 100])  # 限制显示到100Hz
        
        # 3. 信号质量指标
        signal_quality = processing_results.get("results", {}).get("signal_quality", {})
        quality_metrics = signal_quality.get("quality_metrics", {})
        
        if quality_metrics:
            metric_names = list(quality_metrics.keys())
            metric_values = list(quality_metrics.values())
            
            # 创建条形图
            bars = axes[1, 0].bar(range(len(metric_names)), metric_values)
            
            # 根据阈值添加颜色
            thresholds = {
                "snr_db": 10.0,
                "baseline_wander": 0.3,
                "powerline_noise": 0.2,
                "emg_noise": 0.15
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
            
            axes[1, 0].set_xticks(range(len(metric_names)))
            axes[1, 0].set_xticklabels(metric_names, rotation=45, ha='right')
            axes[1, 0].set_title("信号质量指标")
            axes[1, 0].set_ylabel("值")
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. 处理摘要
        summary = processing_results.get("summary", {})
        config_summary = summary.get("config_summary", {})
        results_summary = summary.get("results_summary", {})
        
        summary_text = (
            f"预处理配置:\n"
            f"• 低截止: {config_summary.get('lowcut', 'N/A')} Hz\n"
            f"• 高截止: {config_summary.get('highcut', 'N/A')} Hz\n"
            f"• 陷波频率: {config_summary.get('notch_freq', 'N/A')} Hz\n"
            f"• 保护QRS波: {config_summary.get('protect_qrs_wave', 'N/A')}\n\n"
            f"处理结果:\n"
            f"• 信号质量: {results_summary.get('signal_quality', 'N/A')}\n"
            f"• 质量分数: {results_summary.get('quality_score', 'N/A'):.1f}\n"
            f"• 坏段数量: {results_summary.get('bad_segments', 'N/A')}\n"
            f"• 处理步骤: {summary.get('processing_steps', 'N/A')}"
        )
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].set_title("处理摘要")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化结果已保存到: {save_path}")
        
        plt.show()
