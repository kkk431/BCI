# -*- coding: utf-8 -*-
"""
fNIRS信号预处理模块
支持功能性近红外光谱信号的专用预处理
包括光学密度转换、运动伪影校正、血氧浓度计算等
"""

import numpy as np
from scipy import signal, interpolate, stats
from scipy.signal import savgol_filter, wiener, medfilt
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

# 导入通用预处理模块
from preprocessing import GeneralPreprocessor, PreprocessingConfig, FilterType, WaveletType, DetrendMethod

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ====================== fNIRS专用枚举和配置 ======================

class MotionCorrectionMethod(Enum):
    """运动伪影校正方法枚举"""
    SPLINE = "spline"  # 样条插值法
    PCA = "pca"        # 主成分分析法
    WAVELET = "wavelet"  # 小波去噪法
    HMP = "hmp"        # 基于心跳的运动伪影校正


class OpticalModel(Enum):
    """光学模型枚举"""
    MODIFIED_BEER_LAMBERT = "modified_beer_lambert"  # 修正的比尔-朗伯定律
    SPATIALLY_RESOLVED = "spatially_resolved"        # 空间分辨光谱法


@dataclass
class fNIRSConfig(PreprocessingConfig):
    """
    fNIRS专用预处理配置类
    扩展通用配置，添加fNIRS特有参数
    """
    # 光学转换参数
    optical_model: OpticalModel = OpticalModel.MODIFIED_BEER_LAMBERT
    use_intensity_data: bool = True  # 是否使用原始强度数据
    
    # 波长相关参数
    wavelengths: List[float] = field(default_factory=lambda: [730.0, 850.0])  # 常用波长
    
    # 微分路径因子 (Differential Pathlength Factor, DPF)
    dpf_values: Dict[float, float] = field(default_factory=lambda: {730.0: 6.0, 850.0: 5.0})
    
    # 摩尔吸光系数 (单位: 1/(mmol*cm))
    extinction_coefficients: Dict[float, Dict[str, float]] = field(default_factory=lambda: {
        730.0: {"HbO": 0.873, "HbR": 1.916},
        850.0: {"HbO": 1.169, "HbR": 0.863}
    })
    
    # 运动伪影校正
    motion_correction_method: MotionCorrectionMethod = MotionCorrectionMethod.SPLINE
    motion_correction_threshold: float = 3.0  # 运动检测阈值
    pca_components_to_remove: int = 3  # PCA方法中要移除的成分数
    
    # 通道质量评估
    snr_threshold: float = 20.0  # 信噪比阈值(dB)
    intensity_cv_threshold: float = 0.3  # 强度变异系数阈值
    use_channel_quality_assessment: bool = True
    
    # 短通道回归
    use_short_channel_regression: bool = True
    short_channel_distance_threshold: float = 1.0  # 短通道距离阈值(cm)
    
    # 血流动力学响应滤波
    hemodynamic_lowcut: float = 0.01  # 血流动力学响应的低频截止
    hemodynamic_highcut: float = 0.5   # 血流动力学响应的高频截止
    
    # 生理噪声去除
    remove_physiological_noise: bool = True
    cardiac_frequency_range: Tuple[float, float] = (0.8, 2.0)  # 心搏频率范围(Hz)
    respiration_frequency_range: Tuple[float, float] = (0.1, 0.5)  # 呼吸频率范围(Hz)
    
    # 基线校正
    baseline_correction_window: Tuple[float, float] = (-5.0, 0.0)  # 基线窗口(秒)
    use_percentage_baseline: bool = False  # 是否使用百分比基线校正
    
    # 通道几何信息
    source_positions: Optional[np.ndarray] = None  # 光源位置
    detector_positions: Optional[np.ndarray] = None  # 探测器位置
    channel_distances: Optional[np.ndarray] = None  # 源-探测器距离


# ====================== fNIRS专用预处理器类 ======================

class fNIRSPreprocessor(GeneralPreprocessor):
    """
    fNIRS信号专用预处理器
    在通用预处理基础上，增加光学转换、运动校正、血氧计算等步骤
    """
    
    def __init__(self, config: Optional[fNIRSConfig] = None):
        """
        初始化fNIRS预处理器
        
        Args:
            config: fNIRS预处理配置，如果为None则使用默认配置
        """
        super().__init__(config if config is not None else fNIRSConfig())
        
        # 确保配置是fNIRSConfig类型
        if not isinstance(self.config, fNIRSConfig):
            self.config = fNIRSConfig(**self.config.__dict__)
        
        # fNIRS处理状态
        self.channel_quality_metrics = {}
        self.motion_artifact_info = {}
        
    def process_fNIRS(self, 
                     data_dict: Dict[str, Any], 
                     modality: str = "fnirs",
                     return_hb_types: List[str] = ["HbO", "HbR"]) -> Dict[str, Any]:
        """
        fNIRS专用处理流程
        
        Args:
            data_dict: 四层数据字典
            modality: 要处理的信号模态（默认为"fnirs"）
            return_hb_types: 返回的血氧类型列表
            
        Returns:
            更新后的四层数据字典，包含处理后的HbO和HbR信号
        """
        # 验证fNIRS特有输入结构
        self._validate_fNIRS_input(data_dict, modality)
        
        # 创建fNIRS处理历史记录
        process_record = {
            "modality": modality,
            "steps": [],
            "hb_types": return_hb_types
        }
        
        # 获取fNIRS信号数据
        signal_info = data_dict["signal"][modality]
        
        # 检查数据维度
        data = signal_info["data"]
        sampling_rate = signal_info["sampling_rate"]
        
        if len(data.shape) == 3:
            # 三维数据: (channels, wavelengths, samples)
            n_channels, n_wavelengths, n_samples = data.shape
            is_3d = True
        elif len(data.shape) == 2:
            # 二维数据: (channels, samples) - 假设已经是光学密度
            n_channels, n_samples = data.shape
            n_wavelengths = 1
            is_3d = False
        else:
            raise ValueError(f"fNIRS数据必须是2维或3维数组，当前维度: {len(data.shape)}")
        
        # 处理前备份原始数据
        if "processed" not in data_dict:
            data_dict["processed"] = {}
        if "fNIRS_processing" not in data_dict["processed"]:
            data_dict["processed"]["fNIRS_processing"] = {}
        if modality not in data_dict["processed"]["fNIRS_processing"]:
            data_dict["processed"]["fNIRS_processing"][modality] = {
                "original_data": data.copy(),
                "steps": [],
                "intermediate_data": {}
            }
        
        # ========== fNIRS专用处理步骤 ==========
        
        # 1. 通道质量评估与筛选
        if self.config.use_channel_quality_assessment and is_3d:
            good_channels, channel_metrics = self.channel_quality_assessment(
                data, sampling_rate, signal_info.get("distances", None)
            )
            
            # 只保留质量好的通道
            if len(good_channels) < n_channels:
                data = data[good_channels, :, :] if is_3d else data[good_channels, :]
                n_channels = len(good_channels)
                
                # 更新通道名称
                if "channel_names" in signal_info:
                    signal_info["channel_names"] = [signal_info["channel_names"][i] for i in good_channels]
                
                if "distances" in signal_info:
                    signal_info["distances"] = signal_info["distances"][good_channels]
            
            self.channel_quality_metrics = channel_metrics
            process_record["steps"].append({
                "step": "channel_quality_assessment",
                "n_good_channels": len(good_channels),
                "n_total_channels": n_channels,
                "metrics": channel_metrics
            })
        
        # 2. 转换为光学密度
        if is_3d and self.config.use_intensity_data:
            optical_density = self.convert_to_optical_density(
                data, 
                baseline_period=None
            )
            
            # 保存中间数据
            data_dict["processed"]["fNIRS_processing"][modality]["intermediate_data"]["optical_density"] = optical_density
            
            process_record["steps"].append({
                "step": "convert_to_optical_density",
                "n_wavelengths": n_wavelengths,
                "method": "log_ratio"
            })
        else:
            # 如果已经是光学密度或不需要转换
            optical_density = data
        
        # 3. 运动伪影校正
        if self.config.motion_correction_method:
            corrected_od, motion_info = self.motion_artifact_correction(
                optical_density,
                sampling_rate=sampling_rate,
                method=self.config.motion_correction_method,
                threshold=self.config.motion_correction_threshold
            )
            
            self.motion_artifact_info = motion_info
            process_record["steps"].append({
                "step": "motion_artifact_correction",
                "method": self.config.motion_correction_method.value,
                "n_motion_events": motion_info.get("n_motion_events", 0),
                "threshold": self.config.motion_correction_threshold
            })
        else:
            corrected_od = optical_density
        
        # 4. 血氧浓度计算
        hb_data = self.convert_to_concentration(
            corrected_od,
            wavelengths=self.config.wavelengths,
            distances=signal_info.get("distances", None),
            dpf=self.config.dpf_values,
            extinction_coeffs=self.config.extinction_coefficients,
            optical_model=self.config.optical_model
        )
        
        process_record["steps"].append({
            "step": "convert_to_concentration",
            "optical_model": self.config.optical_model.value,
            "wavelengths": self.config.wavelengths,
            "hb_types": list(hb_data.keys())
        })
        
        # 5. 短通道回归（如果启用且距离信息可用）
        if (self.config.use_short_channel_regression and 
            "distances" in signal_info and
            "HbO" in hb_data and "HbR" in hb_data):
            
            short_channel_indices = np.where(signal_info["distances"] <= self.config.short_channel_distance_threshold)[0]
            long_channel_indices = np.where(signal_info["distances"] > self.config.short_channel_distance_threshold)[0]
            
            if len(short_channel_indices) > 0 and len(long_channel_indices) > 0:
                hb_data = self.short_channel_regression(
                    hb_data,
                    short_channel_indices=short_channel_indices,
                    long_channel_indices=long_channel_indices
                )
                
                process_record["steps"].append({
                    "step": "short_channel_regression",
                    "n_short_channels": len(short_channel_indices),
                    "n_long_channels": len(long_channel_indices),
                    "distance_threshold": self.config.short_channel_distance_threshold
                })
        
        # 6. 生理噪声去除（可选）
        if self.config.remove_physiological_noise:
            hb_data = self.remove_physiological_noise(
                hb_data,
                sampling_rate=sampling_rate,
                cardiac_range=self.config.cardiac_frequency_range,
                respiration_range=self.config.respiration_frequency_range
            )
            
            process_record["steps"].append({
                "step": "remove_physiological_noise",
                "cardiac_range": self.config.cardiac_frequency_range,
                "respiration_range": self.config.respiration_frequency_range
            })
        
        # 7. 基线校正
        if self.config.baseline_correction_window and "event" in data_dict:
            hb_data = self.baseline_correction(
                hb_data,
                sampling_rate=sampling_rate,
                event_times=data_dict["event"].get("event_time", []),
                baseline_window=self.config.baseline_correction_window,
                use_percentage=self.config.use_percentage_baseline
            )
            
            process_record["steps"].append({
                "step": "baseline_correction",
                "baseline_window": self.config.baseline_correction_window,
                "use_percentage": self.config.use_percentage_baseline
            })
        
        # ========== 对血氧信号应用通用预处理 ==========
        
        # 更新数据字典，添加血氧信号作为新模态
        for hb_type in return_hb_types:
            if hb_type in hb_data:
                # 将血氧信号作为新模态添加到signal层
                data_dict["signal"][hb_type] = {
                    "data": hb_data[hb_type],
                    "sampling_rate": sampling_rate,
                    "unit": "uM",  # 微摩尔浓度
                    "channel_names": signal_info.get("channel_names", [f"CH{i}" for i in range(n_channels)]),
                    "source_modality": modality
                }
                
                # 对血氧信号应用通用预处理
                if (self.config.hemodynamic_lowcut is not None and 
                    self.config.hemodynamic_highcut is not None):
                    
                    # 创建血氧专用的配置
                    hb_config = PreprocessingConfig(
                        lowcut=self.config.hemodynamic_lowcut,
                        highcut=self.config.hemodynamic_highcut,
                        filter_type=self.config.filter_type,
                        filter_order=self.config.filter_order,
                        detrend_method=self.config.detrend_method,
                        remove_baseline=False,  # 基线已在前面处理
                        normalize_method=self.config.normalize_method,
                        remove_outliers=self.config.remove_outliers,
                        outlier_threshold=self.config.outlier_threshold
                    )
                    
                    # 创建临时预处理器处理血氧信号
                    hb_processor = GeneralPreprocessor(hb_config)
                    data_dict = hb_processor.process(data_dict, modality=hb_type)
        
        # 记录fNIRS处理历史
        data_dict["processed"]["fNIRS_processing"][modality]["steps"].append(process_record)
        data_dict["processed"]["fNIRS_processing"][modality]["hb_data"] = hb_data
        data_dict["processed"]["fNIRS_processing"][modality]["channel_quality"] = self.channel_quality_metrics
        data_dict["processed"]["fNIRS_processing"][modality]["motion_info"] = self.motion_artifact_info
        
        self.history.append(process_record)
        
        logger.info(f"fNIRS预处理完成: {modality}, "
                   f"血氧类型: {return_hb_types}, "
                   f"处理步骤: {len(process_record['steps'])}")
        
        return data_dict
    
    def _validate_fNIRS_input(self, data_dict: Dict, modality: str):
        """
        验证fNIRS特有输入结构
        
        Args:
            data_dict: 四层数据字典
            modality: 要验证的模态名称
            
        Raises:
            ValueError: 如果输入数据格式不符合fNIRS要求
        """
        # 首先调用通用验证
        super()._validate_input(data_dict, modality)
        
        signal_info = data_dict["signal"][modality]
        
        # 检查fNIRS特有字段
        if "wavelengths" not in signal_info and self.config.use_intensity_data:
            logger.warning(f"fNIRS信号缺少'wavelengths'字段，将使用配置中的波长: {self.config.wavelengths}")
            data_dict["signal"][modality]["wavelengths"] = self.config.wavelengths
        
        # 检查数据维度
        data = signal_info["data"]
        if len(data.shape) not in [2, 3]:
            raise ValueError(f"fNIRS数据必须是2维或3维数组，当前维度: {len(data.shape)}")
        
        if len(data.shape) == 3:
            # 三维数据: 检查波长数量
            n_channels, n_wavelengths, _ = data.shape
            if "wavelengths" in signal_info and len(signal_info["wavelengths"]) != n_wavelengths:
                raise ValueError(f"波长数量({len(signal_info['wavelengths'])})与数据第二维度({n_wavelengths})不匹配")
    
    # ====================== fNIRS特有预处理方法 ======================
    
    def convert_to_optical_density(self, 
                                  intensity_data: np.ndarray,
                                  baseline_period: Optional[Union[int, Tuple[int, int]]] = None) -> np.ndarray:
        """
        将原始光强度信号转换为光学密度(OD)
        
        Args:
            intensity_data: 原始光强度数据，形状为 (channels, wavelengths, samples) 或 (channels, samples)
            baseline_period: 基线时段，可以是:
                            - None: 使用整个时间段平均作为基线
                            - int: 前n个样本作为基线
                            - Tuple[int, int]: 起始和结束样本索引
        
        Returns:
            光学密度数据，形状与输入相同
        """
        # 确保数据是浮点型
        intensity_data = intensity_data.astype(np.float64)
        
        # 处理基线
        if baseline_period is None:
            # 使用整个时间段平均作为基线
            I0 = np.mean(intensity_data, axis=-1, keepdims=True)
        elif isinstance(baseline_period, int):
            # 前n个样本作为基线
            I0 = np.mean(intensity_data[..., :baseline_period], axis=-1, keepdims=True)
        elif isinstance(baseline_period, tuple) and len(baseline_period) == 2:
            # 指定范围的样本作为基线
            start, end = baseline_period
            I0 = np.mean(intensity_data[..., start:end], axis=-1, keepdims=True)
        else:
            raise ValueError("baseline_period 必须是 None, int 或 (int, int) 元组")
        
        # 避免除以零
        I0 = np.where(I0 <= 0, np.finfo(float).eps, I0)
        
        # 计算光学密度: OD = -log10(I/I0)
        # 添加小常数避免log(0)
        eps = np.finfo(float).eps
        optical_density = -np.log10((intensity_data + eps) / (I0 + eps))
        
        return optical_density
    
    def motion_artifact_correction(self,
                                   optical_density: np.ndarray,
                                   sampling_rate: float,
                                   method: MotionCorrectionMethod = MotionCorrectionMethod.SPLINE,
                                   threshold: float = 3.0,
                                   **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        运动伪影校正
        
        Args:
            optical_density: 光学密度数据
            sampling_rate: 采样率
            method: 运动伪影校正方法
            threshold: 运动检测阈值
            **kwargs: 其他方法特定参数
        
        Returns:
            corrected_data: 校正后的光学密度数据
            motion_info: 运动伪影信息字典
        """
        n_channels, n_wavelengths, n_samples = optical_density.shape
        corrected_data = optical_density.copy()
        motion_info = {
            "n_motion_events": 0,
            "motion_indices": [],
            "method": method.value
        }
        
        if method == MotionCorrectionMethod.SPLINE:
            # 样条插值法
            corrected_data, motion_info = self._motion_correction_spline(
                optical_density, sampling_rate, threshold
            )
            
        elif method == MotionCorrectionMethod.PCA:
            # 主成分分析法
            corrected_data, motion_info = self._motion_correction_pca(
                optical_density, 
                n_components=self.config.pca_components_to_remove
            )
            
        elif method == MotionCorrectionMethod.WAVELET:
            # 小波去噪法
            corrected_data, motion_info = self._motion_correction_wavelet(
                optical_density
            )
            
        elif method == MotionCorrectionMethod.HMP:
            # 基于心跳的运动伪影校正
            corrected_data, motion_info = self._motion_correction_hmp(
                optical_density, sampling_rate
            )
        
        return corrected_data, motion_info
    
    def _motion_correction_spline(self,
                                  optical_density: np.ndarray,
                                  sampling_rate: float,
                                  threshold: float = 3.0) -> Tuple[np.ndarray, Dict]:
        """
        样条插值法运动伪影校正
        
        算法步骤:
        1. 检测运动尖峰（超出阈值的快速变化）
        2. 使用三次样条插值替换运动段
        """
        n_channels, n_wavelengths, n_samples = optical_density.shape
        corrected_data = optical_density.copy()
        motion_info = {
            "n_motion_events": 0,
            "motion_indices": []
        }
        
        # 对每个通道和波长进行处理
        for ch in range(n_channels):
            for wl in range(n_wavelengths):
                signal = optical_density[ch, wl, :]
                
                # 计算信号的一阶差分（速度）
                diff_signal = np.diff(signal)
                
                # 计算移动标准差作为运动检测指标
                window_size = int(0.5 * sampling_rate)  # 500ms窗口
                if window_size % 2 == 0:
                    window_size += 1
                
                # 使用中值绝对偏差(MAD)作为运动检测的阈值
                mad = np.median(np.abs(diff_signal - np.median(diff_signal)))
                std_estimate = mad * 1.4826
                
                # 检测运动尖峰
                motion_mask = np.abs(diff_signal) > threshold * std_estimate
                
                # 扩展运动段（前后各加一些样本）
                extend_samples = int(0.2 * sampling_rate)  # 200ms
                motion_mask_extended = np.zeros_like(motion_mask, dtype=bool)
                
                for i in range(len(motion_mask)):
                    if motion_mask[i]:
                        start = max(0, i - extend_samples)
                        end = min(len(motion_mask), i + extend_samples + 1)
                        motion_mask_extended[start:end] = True
                
                # 将运动段转换为索引
                motion_indices = np.where(motion_mask_extended)[0]
                
                if len(motion_indices) > 0:
                    motion_info["n_motion_events"] += 1
                    motion_info["motion_indices"].append((ch, wl, motion_indices.tolist()))
                    
                    # 使用样条插值替换运动段
                    time_points = np.arange(n_samples)
                    valid_indices = np.setdiff1d(time_points, motion_indices)
                    
                    if len(valid_indices) > 3:  # 需要至少3个点进行样条插值
                        # 三次样条插值
                        spline = interpolate.interp1d(
                            valid_indices, 
                            signal[valid_indices], 
                            kind='cubic',
                            bounds_error=False,
                            fill_value="extrapolate"
                        )
                        
                        # 插值替换运动段
                        corrected_signal = signal.copy()
                        corrected_signal[motion_indices] = spline(motion_indices)
                        corrected_data[ch, wl, :] = corrected_signal
        
        return corrected_data, motion_info
    
    def _motion_correction_pca(self,
                               optical_density: np.ndarray,
                               n_components: int = 3) -> Tuple[np.ndarray, Dict]:
        """
        主成分分析法运动伪影校正
        
        算法步骤:
        1. 对光学密度数据进行PCA分解
        2. 去除前n个主要成分（假设主要包含运动伪影）
        3. 重构信号
        """
        from sklearn.decomposition import PCA
        
        n_channels, n_wavelengths, n_samples = optical_density.shape
        motion_info = {
            "n_motion_events": 0,
            "method": "PCA",
            "components_removed": n_components
        }
        
        # 重塑数据为2D: (n_channels * n_wavelengths, n_samples)
        data_2d = optical_density.reshape(-1, n_samples).T  # (n_samples, n_features)
        
        # 应用PCA
        pca = PCA(n_components=min(n_components, data_2d.shape[1]))
        pca.fit(data_2d)
        
        # 去除前n个主要成分
        components = pca.components_
        transformed = pca.transform(data_2d)
        
        # 将前n个成分置零
        transformed[:, :n_components] = 0
        
        # 反变换
        reconstructed = pca.inverse_transform(transformed)
        
        # 重塑回原始形状
        corrected_data = reconstructed.T.reshape(n_channels, n_wavelengths, n_samples)
        
        # 计算方差解释比例
        motion_info["variance_explained"] = np.sum(pca.explained_variance_ratio_[:n_components])
        
        return corrected_data, motion_info
    
    def _motion_correction_wavelet(self,
                                   optical_density: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        小波去噪法运动伪影校正
        
        算法步骤:
        1. 对每个通道进行小波分解
        2. 对小波系数应用阈值处理
        3. 重构信号
        """
        import pywt
        
        n_channels, n_wavelengths, n_samples = optical_density.shape
        corrected_data = optical_density.copy()
        motion_info = {
            "n_motion_events": 0,
            "method": "wavelet"
        }
        
        # 小波参数
        wavelet = 'db4'
        level = 4
        
        for ch in range(n_channels):
            for wl in range(n_wavelengths):
                signal = optical_density[ch, wl, :]
                
                # 小波分解
                coeffs = pywt.wavedec(signal, wavelet, level=level)
                
                # 计算噪声标准差估计
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                
                # 计算通用阈值
                universal_threshold = sigma * np.sqrt(2 * np.log(n_samples))
                
                # 应用软阈值
                new_coeffs = [coeffs[0]]  # 近似系数保留
                for coeff in coeffs[1:]:
                    new_coeffs.append(pywt.threshold(coeff, universal_threshold, mode='soft'))
                
                # 小波重构
                denoised_signal = pywt.waverec(new_coeffs, wavelet)
                
                # 确保长度匹配
                if len(denoised_signal) > n_samples:
                    corrected_data[ch, wl, :] = denoised_signal[:n_samples]
                else:
                    corrected_data[ch, wl, :] = denoised_signal
        
        return corrected_data, motion_info
    
    def _motion_correction_hmp(self,
                               optical_density: np.ndarray,
                               sampling_rate: float) -> Tuple[np.ndarray, Dict]:
        """
        基于心跳的运动伪影校正(HMP: Heartbeat-based Motion artifact correction using Phase synchronization)
        
        算法步骤:
        1. 提取心跳信号（假设有辅助ECG信号或从fNIRS中提取心跳）
        2. 利用心跳相位同步检测运动伪影
        3. 校正运动伪影
        """
        # 这是一个简化的实现，实际应用中需要ECG信号或从fNIRS中提取心跳
        n_channels, n_wavelengths, n_samples = optical_density.shape
        motion_info = {
            "n_motion_events": 0,
            "method": "HMP",
            "note": "需要ECG信号进行完整实现"
        }
        
        # 这里返回原始数据，实际应用中需要完整实现
        return optical_density, motion_info
    
    def convert_to_concentration(self,
                                 optical_density: np.ndarray,
                                 wavelengths: List[float],
                                 distances: Optional[np.ndarray] = None,
                                 dpf: Optional[Dict[float, float]] = None,
                                 extinction_coeffs: Optional[Dict[float, Dict[str, float]]] = None,
                                 optical_model: OpticalModel = OpticalModel.MODIFIED_BEER_LAMBERT) -> Dict[str, np.ndarray]:
        """
        根据修正的比尔-朗伯定律，将光学密度转换为血氧浓度
        
        Args:
            optical_density: 光学密度数据，形状为 (channels, wavelengths, samples)
            wavelengths: 波长列表
            distances: 源-探测器距离数组，形状为 (channels,)
            dpf: 微分路径因子字典，键为波长，值为DPF
            extinction_coeffs: 摩尔吸光系数字典
            optical_model: 光学模型
            
        Returns:
            血氧浓度数据字典，包含"HbO"、"HbR"等键
        """
        if len(optical_density.shape) != 3:
            raise ValueError(f"光学密度数据必须是3维数组，当前维度: {len(optical_density.shape)}")
        
        n_channels, n_wavelengths, n_samples = optical_density.shape
        
        if len(wavelengths) != n_wavelengths:
            raise ValueError(f"波长数量({len(wavelengths)})与数据第二维度({n_wavelengths})不匹配")
        
        # 使用默认参数如果未提供
        if dpf is None:
            dpf = self.config.dpf_values
        
        if extinction_coeffs is None:
            extinction_coeffs = self.config.extinction_coefficients
        
        # 检查是否所有波长都有对应的参数
        for wl in wavelengths:
            if wl not in dpf:
                raise ValueError(f"波长 {wl} 没有对应的DPF值")
            if wl not in extinction_coeffs:
                raise ValueError(f"波长 {wl} 没有对应的摩尔吸光系数")
        
        if optical_model == OpticalModel.MODIFIED_BEER_LAMBERT:
            return self._convert_mbl(optical_density, wavelengths, distances, dpf, extinction_coeffs)
        elif optical_model == OpticalModel.SPATIALLY_RESOLVED:
            return self._convert_srs(optical_density, wavelengths, distances, dpf, extinction_coeffs)
        else:
            raise ValueError(f"不支持的光学模型: {optical_model}")
    
    def _convert_mbl(self,
                     optical_density: np.ndarray,
                     wavelengths: List[float],
                     distances: Optional[np.ndarray],
                     dpf: Dict[float, float],
                     extinction_coeffs: Dict[float, Dict[str, float]]) -> Dict[str, np.ndarray]:
        """
        使用修正的比尔-朗伯定律(MBL)计算血氧浓度
        
        公式:
        [ΔHbO] = (ε_HbR(λ2)·ΔOD(λ1) - ε_HbR(λ1)·ΔOD(λ2)) / (d·DPF·(ε_HbO(λ1)·ε_HbR(λ2) - ε_HbO(λ2)·ε_HbR(λ1)))
        [ΔHbR] = (ε_HbO(λ1)·ΔOD(λ2) - ε_HbO(λ2)·ΔOD(λ1)) / (d·DPF·(ε_HbO(λ1)·ε_HbR(λ2) - ε_HbO(λ2)·ε_HbR(λ1)))
        
        对于两个波长的情况
        """
        n_channels, n_wavelengths, n_samples = optical_density.shape
        
        if n_wavelengths != 2:
            raise ValueError(f"修正的比尔-朗伯定律需要2个波长，当前提供: {n_wavelengths}")
        
        # 提取波长
        wl1, wl2 = wavelengths
        
        # 提取摩尔吸光系数
        e1_HbO = extinction_coeffs[wl1]["HbO"]
        e1_HbR = extinction_coeffs[wl1]["HbR"]
        e2_HbO = extinction_coeffs[wl2]["HbO"]
        e2_HbR = extinction_coeffs[wl2]["HbR"]
        
        # 提取光学密度
        OD1 = optical_density[:, 0, :]  # 波长1的光学密度
        OD2 = optical_density[:, 1, :]  # 波长2的光学密度
        
        # 计算分母
        denominator = e1_HbO * e2_HbR - e2_HbO * e1_HbR
        
        if np.abs(denominator) < 1e-10:
            raise ValueError("分母接近零，无法计算血氧浓度")
        
        # 初始化血氧浓度数组
        HbO = np.zeros((n_channels, n_samples))
        HbR = np.zeros((n_channels, n_samples))
        
        # 对每个通道计算
        for ch in range(n_channels):
            # 获取距离（如果提供）
            if distances is not None and len(distances) > ch:
                d = distances[ch]
                dpf1 = dpf.get(wl1, 1.0)
                dpf2 = dpf.get(wl2, 1.0)
                
                # 使用两个波长的DPF平均值
                dpf_avg = (dpf1 + dpf2) / 2.0
                d_total = d * dpf_avg
            else:
                # 如果没有距离信息，假设为单位长度
                d_total = 1.0
            
            # 计算血氧浓度变化
            HbO[ch, :] = (e2_HbR * OD1[ch, :] - e1_HbR * OD2[ch, :]) / (d_total * denominator)
            HbR[ch, :] = (e1_HbO * OD2[ch, :] - e2_HbO * OD1[ch, :]) / (d_total * denominator)
        
        # 可选：计算总血红蛋白
        HbT = HbO + HbR
        
        return {
            "HbO": HbO,
            "HbR": HbR,
            "HbT": HbT
        }
    
    def _convert_srs(self,
                     optical_density: np.ndarray,
                     wavelengths: List[float],
                     distances: Optional[np.ndarray],
                     dpf: Dict[float, float],
                     extinction_coeffs: Dict[float, Dict[str, float]]) -> Dict[str, np.ndarray]:
        """
        使用空间分辨光谱法(SRS)计算血氧浓度
        
        这种方法使用多个距离的测量来分离浅层和深层组织信号
        这是一个简化的实现
        """
        # 简化的SRS实现 - 实际应用需要更复杂的模型
        logger.warning("空间分辨光谱法(SRS)是简化实现，实际应用需要完整模型")
        
        # 暂时使用MBL方法
        return self._convert_mbl(optical_density, wavelengths, distances, dpf, extinction_coeffs)
    
    def channel_quality_assessment(self,
                                   intensity_data: np.ndarray,
                                   sampling_rate: float,
                                   distances: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        通道质量评估与筛选
        
        Args:
            intensity_data: 原始强度数据，形状为 (channels, wavelengths, samples)
            sampling_rate: 采样率
            distances: 源-探测器距离
            
        Returns:
            good_channels: 通过质量检测的通道索引
            quality_metrics: 各通道质量指标字典
        """
        n_channels, n_wavelengths, n_samples = intensity_data.shape
        
        quality_metrics = {
            "snr": np.zeros((n_channels, n_wavelengths)),
            "intensity_cv": np.zeros((n_channels, n_wavelengths)),
            "signal_power": np.zeros((n_channels, n_wavelengths)),
            "noise_power": np.zeros((n_channels, n_wavelengths)),
            "is_good": np.zeros(n_channels, dtype=bool)
        }
        
        # 对每个通道和波长计算质量指标
        for ch in range(n_channels):
            channel_good = True
            
            for wl in range(n_wavelengths):
                signal = intensity_data[ch, wl, :]
                
                # 1. 计算信噪比(SNR)
                # 假设低频部分是信号，高频部分是噪声
                from scipy.signal import periodogram
                freqs, psd = periodogram(signal, fs=sampling_rate)
                
                # 信号功率：低频部分 (0-0.5 Hz)
                signal_mask = freqs <= 0.5
                signal_power = np.mean(psd[signal_mask]) if np.any(signal_mask) else 0
                
                # 噪声功率：高频部分 (5 Hz以上)
                noise_mask = freqs >= 5.0
                noise_power = np.mean(psd[noise_mask]) if np.any(noise_mask) else 0
                
                # 计算SNR (dB)
                if noise_power > 0:
                    snr_db = 10 * np.log10(signal_power / noise_power)
                else:
                    snr_db = np.inf
                
                quality_metrics["snr"][ch, wl] = snr_db
                quality_metrics["signal_power"][ch, wl] = signal_power
                quality_metrics["noise_power"][ch, wl] = noise_power
                
                # 2. 计算强度变异系数(CV)
                mean_intensity = np.mean(signal)
                std_intensity = np.std(signal)
                cv = std_intensity / mean_intensity if mean_intensity > 0 else np.inf
                quality_metrics["intensity_cv"][ch, wl] = cv
                
                # 3. 检查是否满足阈值
                if snr_db < self.config.snr_threshold:
                    channel_good = False
                    logger.debug(f"通道 {ch} 波长 {wl} SNR过低: {snr_db:.2f} dB < {self.config.snr_threshold} dB")
                
                if cv > self.config.intensity_cv_threshold:
                    channel_good = False
                    logger.debug(f"通道 {ch} 波长 {wl} 强度变异过高: {cv:.3f} > {self.config.intensity_cv_threshold}")
            
            # 4. 检查距离相关性（如果有距离信息）
            if distances is not None and len(distances) > ch:
                distance = distances[ch]
                
                # 计算平均强度
                mean_intensity = np.mean(intensity_data[ch, :, :])
                
                # 简单检查：强度是否与距离合理相关（距离越大，强度越小）
                # 这是一个启发式检查，实际关系更复杂
                if distance > 3.0 and mean_intensity > np.percentile(intensity_data, 90):
                    channel_good = False
                    logger.debug(f"通道 {ch} 距离({distance} cm)与强度不匹配")
            
            quality_metrics["is_good"][ch] = channel_good
        
        # 获取通过质量检测的通道
        good_channels = np.where(quality_metrics["is_good"])[0]
        
        logger.info(f"通道质量评估完成: {len(good_channels)}/{n_channels} 个通道通过检测")
        
        return good_channels, quality_metrics
    
    def short_channel_regression(self,
                                 hb_data: Dict[str, np.ndarray],
                                 short_channel_indices: np.ndarray,
                                 long_channel_indices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        短通道回归：使用短通道信号回归掉浅层（头皮）生理噪声
        
        Args:
            hb_data: 血氧浓度数据字典
            short_channel_indices: 短通道索引
            long_channel_indices: 长通道索引
            
        Returns:
            回归后的血氧数据字典
        """
        if len(short_channel_indices) == 0 or len(long_channel_indices) == 0:
            logger.warning("短通道或长通道数量为零，跳过短通道回归")
            return hb_data
        
        regressed_data = {}
        
        for hb_type, data in hb_data.items():
            n_channels, n_samples = data.shape
            
            # 创建回归后的数据副本
            regressed = data.copy()
            
            # 对每个长通道进行回归
            for long_idx in long_channel_indices:
                if long_idx >= n_channels:
                    continue
                
                # 提取长通道信号
                long_signal = data[long_idx, :]
                
                # 构建短通道信号矩阵（浅层噪声估计）
                short_signals = []
                for short_idx in short_channel_indices:
                    if short_idx < n_channels:
                        short_signals.append(data[short_idx, :])
                
                if len(short_signals) == 0:
                    continue
                
                short_matrix = np.vstack(short_signals).T  # (n_samples, n_short_channels)
                
                # 添加截距项
                X = np.column_stack([np.ones(n_samples), short_matrix])
                
                # 最小二乘回归
                try:
                    beta = np.linalg.lstsq(X, long_signal, rcond=None)[0]
                    
                    # 预测浅层噪声
                    shallow_noise = X @ beta
                    
                    # 从长通道信号中减去浅层噪声
                    regressed[long_idx, :] = long_signal - shallow_noise
                    
                except np.linalg.LinAlgError:
                    logger.warning(f"长通道 {long_idx} 回归失败，跳过")
                    continue
            
            regressed_data[hb_type] = regressed
        
        return regressed_data
    
    def remove_physiological_noise(self,
                                   hb_data: Dict[str, np.ndarray],
                                   sampling_rate: float,
                                   cardiac_range: Tuple[float, float] = (0.8, 2.0),
                                   respiration_range: Tuple[float, float] = (0.1, 0.5)) -> Dict[str, np.ndarray]:
        """
        去除生理噪声（心跳和呼吸）
        
        Args:
            hb_data: 血氧浓度数据
            sampling_rate: 采样率
            cardiac_range: 心搏频率范围
            respiration_range: 呼吸频率范围
            
        Returns:
            去除生理噪声后的血氧数据
        """
        denoised_data = {}
        
        for hb_type, data in hb_data.items():
            n_channels, n_samples = data.shape
            
            # 设计带阻滤波器去除心搏和呼吸频率
            from scipy.signal import butter, filtfilt
            
            # 心搏频率带阻滤波器
            if cardiac_range[1] < sampling_rate / 2:
                b_cardiac, a_cardiac = butter(
                    4, 
                    [cardiac_range[0] / (sampling_rate / 2), cardiac_range[1] / (sampling_rate / 2)],
                    btype='bandstop'
                )
            else:
                b_cardiac, a_cardiac = None, None
            
            # 呼吸频率带阻滤波器
            if respiration_range[1] < sampling_rate / 2:
                b_resp, a_resp = butter(
                    4,
                    [respiration_range[0] / (sampling_rate / 2), respiration_range[1] / (sampling_rate / 2)],
                    btype='bandstop'
                )
            else:
                b_resp, a_resp = None, None
            
            # 对每个通道应用滤波器
            denoised = np.zeros_like(data)
            
            for ch in range(n_channels):
                signal = data[ch, :]
                
                # 应用心搏频率带阻滤波器
                if b_cardiac is not None and a_cardiac is not None:
                    signal = filtfilt(b_cardiac, a_cardiac, signal)
                
                # 应用呼吸频率带阻滤波器
                if b_resp is not None and a_resp is not None:
                    signal = filtfilt(b_resp, a_resp, signal)
                
                denoised[ch, :] = signal
            
            denoised_data[hb_type] = denoised
        
        return denoised_data
    
    def baseline_correction(self,
                            hb_data: Dict[str, np.ndarray],
                            sampling_rate: float,
                            event_times: List[float],
                            baseline_window: Tuple[float, float] = (-5.0, 0.0),
                            use_percentage: bool = False) -> Dict[str, np.ndarray]:
        """
        基线校正
        
        Args:
            hb_data: 血氧浓度数据
            sampling_rate: 采样率
            event_times: 事件时间列表
            baseline_window: 基线窗口(秒)
            use_percentage: 是否使用百分比基线校正
            
        Returns:
            基线校正后的血氧数据
        """
        if not event_times:
            logger.warning("没有事件时间信息，跳过基线校正")
            return hb_data
        
        corrected_data = {}
        
        for hb_type, data in hb_data.items():
            n_channels, n_samples = data.shape
            
            # 将时间转换为样本索引
            time_points = np.arange(n_samples) / sampling_rate
            
            # 对每个事件进行基线校正
            corrected = data.copy()
            
            for event_time in event_times:
                # 找到事件时间的样本索引
                event_sample = int(event_time * sampling_rate)
                
                # 计算基线窗口
                baseline_start = event_sample + int(baseline_window[0] * sampling_rate)
                baseline_end = event_sample + int(baseline_window[1] * sampling_rate)
                
                # 确保基线窗口在数据范围内
                baseline_start = max(0, baseline_start)
                baseline_end = min(n_samples, baseline_end)
                
                if baseline_start >= baseline_end:
                    continue
                
                # 对每个通道计算基线值
                for ch in range(n_channels):
                    baseline_data = data[ch, baseline_start:baseline_end]
                    
                    if len(baseline_data) > 0:
                        if use_percentage:
                            # 百分比基线校正: 信号减去基线均值后除以基线均值
                            baseline_mean = np.mean(baseline_data)
                            if baseline_mean != 0:
                                corrected[ch, :] = 100 * (corrected[ch, :] - baseline_mean) / baseline_mean
                        else:
                            # 减法基线校正: 信号减去基线均值
                            baseline_mean = np.mean(baseline_data)
                            corrected[ch, :] = corrected[ch, :] - baseline_mean
            
            corrected_data[hb_type] = corrected
        
        return corrected_data
    
    def visualize_preprocessing(self,
                                data_dict: Dict[str, Any],
                                modality: str = "fnirs",
                                channel_idx: int = 0,
                                wavelength_idx: int = 0):
        """
        可视化fNIRS预处理过程
        
        Args:
            data_dict: 处理后的数据字典
            modality: 信号模态
            channel_idx: 通道索引
            wavelength_idx: 波长索引
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib未安装，无法可视化")
            return
        
        if modality not in data_dict["signal"]:
            logger.error(f"模态 {modality} 不在数据中")
            return
        
        # 获取处理历史
        if ("processed" in data_dict and 
            "fNIRS_processing" in data_dict["processed"] and
            modality in data_dict["processed"]["fNIRS_processing"]):
            
            processing_info = data_dict["processed"]["fNIRS_processing"][modality]
            intermediate_data = processing_info.get("intermediate_data", {})
            
            # 创建图形
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle(f"fNIRS预处理可视化 - 通道 {channel_idx}, 波长 {wavelength_idx}", fontsize=16)
            
            # 1. 原始强度信号
            if "original_data" in processing_info:
                original_data = processing_info["original_data"]
                if len(original_data.shape) == 3 and channel_idx < original_data.shape[0] and wavelength_idx < original_data.shape[1]:
                    time_axis = np.arange(original_data.shape[2]) / data_dict["signal"][modality]["sampling_rate"]
                    axes[0, 0].plot(time_axis, original_data[channel_idx, wavelength_idx, :])
                    axes[0, 0].set_title("原始强度信号")
                    axes[0, 0].set_xlabel("时间 (秒)")
                    axes[0, 0].set_ylabel("强度")
                    axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 光学密度
            if "optical_density" in intermediate_data:
                od_data = intermediate_data["optical_density"]
                if len(od_data.shape) == 3 and channel_idx < od_data.shape[0] and wavelength_idx < od_data.shape[1]:
                    time_axis = np.arange(od_data.shape[2]) / data_dict["signal"][modality]["sampling_rate"]
                    axes[0, 1].plot(time_axis, od_data[channel_idx, wavelength_idx, :])
                    axes[0, 1].set_title("光学密度 (OD)")
                    axes[0, 1].set_xlabel("时间 (秒)")
                    axes[0, 1].set_ylabel("OD")
                    axes[0, 1].grid(True, alpha=0.3)
            
            # 3. HbO信号
            if "HbO" in data_dict["signal"] and channel_idx < data_dict["signal"]["HbO"]["data"].shape[0]:
                hbo_data = data_dict["signal"]["HbO"]["data"]
                time_axis = np.arange(hbo_data.shape[1]) / data_dict["signal"]["HbO"]["sampling_rate"]
                axes[1, 0].plot(time_axis, hbo_data[channel_idx, :])
                axes[1, 0].set_title("含氧血红蛋白 (HbO)")
                axes[1, 0].set_xlabel("时间 (秒)")
                axes[1, 0].set_ylabel("浓度变化 (μM)")
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. HbR信号
            if "HbR" in data_dict["signal"] and channel_idx < data_dict["signal"]["HbR"]["data"].shape[0]:
                hbr_data = data_dict["signal"]["HbR"]["data"]
                time_axis = np.arange(hbr_data.shape[1]) / data_dict["signal"]["HbR"]["sampling_rate"]
                axes[1, 1].plot(time_axis, hbr_data[channel_idx, :])
                axes[1, 1].set_title("脱氧血红蛋白 (HbR)")
                axes[1, 1].set_xlabel("时间 (秒)")
                axes[1, 1].set_ylabel("浓度变化 (μM)")
                axes[1, 1].grid(True, alpha=0.3)
            
            # 5. 通道质量指标
            if "channel_quality" in processing_info:
                quality_metrics = processing_info["channel_quality"]
                if "snr" in quality_metrics:
                    axes[2, 0].bar(range(len(quality_metrics["snr"])), quality_metrics["snr"][:, wavelength_idx])
                    axes[2, 0].axhline(y=self.config.snr_threshold, color='r', linestyle='--', label=f'阈值: {self.config.snr_threshold} dB')
                    axes[2, 0].set_title("通道信噪比 (SNR)")
                    axes[2, 0].set_xlabel("通道索引")
                    axes[2, 0].set_ylabel("SNR (dB)")
                    axes[2, 0].legend()
                    axes[2, 0].grid(True, alpha=0.3)
            
            # 6. 运动伪影信息
            if "motion_info" in processing_info:
                motion_info = processing_info["motion_info"]
                axes[2, 1].text(0.1, 0.5, f"运动事件数量: {motion_info.get('n_motion_events', 0)}\n"
                                        f"校正方法: {motion_info.get('method', 'N/A')}\n"
                                        f"运动检测阈值: {self.config.motion_correction_threshold}",
                                transform=axes[2, 1].transAxes, fontsize=12,
                                verticalalignment='center')
                axes[2, 1].set_title("运动伪影校正信息")
                axes[2, 1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        else:
            logger.warning("没有找到预处理历史信息")


# ====================== 实用函数 ======================

def load_fNIRS_data(filepath: str, 
                   modality: str = "fnirs",
                   wavelengths: List[float] = None) -> Dict[str, Any]:
    """
    加载fNIRS数据（示例函数）
    
    实际应用中需要根据具体数据格式实现
    
    Args:
        filepath: 数据文件路径
        modality: 信号模态
        wavelengths: 波长列表
        
    Returns:
        四层结构的数据字典
    """
    # 这是一个示例函数，实际应用中需要根据具体数据格式实现
    logger.warning("这是一个示例加载函数，需要根据实际数据格式实现")
    
    # 示例数据字典结构
    data_dict = {
        "meta": {
            "subject_id": "S01",
            "task": "resting_state",
            "modality": ["fnirs"],
            "device": "NIRx",
            "sampling_rate": 10.0,  # fNIRS典型采样率
            "n_channels": 20,
            "channel_names": [f"CH{i}" for i in range(20)],
            "wavelengths": wavelengths or [730.0, 850.0]
        },
        "signal": {
            "fnirs": {
                "data": np.random.randn(20, 2, 1000),  # 模拟数据
                "sampling_rate": 10.0,
                "unit": "V",
                "channel_names": [f"CH{i}" for i in range(20)],
                "wavelengths": wavelengths or [730.0, 850.0],
                "distances": np.random.uniform(1.0, 3.0, 20)  # 模拟距离
            }
        },
        "event": {
            "event_id": [1, 2, 1],
            "event_label": ["stimulus", "rest", "stimulus"],
            "event_time": [10.0, 30.0, 50.0],
            "event_sample": [100, 300, 500]
        },
        "processed": {}
    }
    
    return data_dict


def save_fNIRS_results(data_dict: Dict[str, Any], 
                      output_path: str,
                      include_intermediate: bool = False):
    """
    保存fNIRS处理结果
    
    Args:
        data_dict: 处理后的数据字典
        output_path: 输出文件路径
        include_intermediate: 是否包含中间数据
    """
    import pickle
    
    # 如果不包含中间数据，则移除它们以减少文件大小
    save_dict = data_dict.copy()
    
    if not include_intermediate and "processed" in save_dict:
        for modality in list(save_dict["processed"].get("fNIRS_processing", {}).keys()):
            if "intermediate_data" in save_dict["processed"]["fNIRS_processing"][modality]:
                del save_dict["processed"]["fNIRS_processing"][modality]["intermediate_data"]
    
    # 保存为pickle文件
    with open(output_path, 'wb') as f:
        pickle.dump(save_dict, f)
    
    logger.info(f"fNIRS处理结果已保存到: {output_path}")


# ====================== 示例使用代码 ======================

if __name__ == "__main__":
    """
    fNIRS预处理模块使用示例
    """
    
    # 1. 创建fNIRS配置
    fnirs_config = fNIRSConfig(
        # 通用预处理参数
        lowcut=0.01,  # 血流动力学响应的低频截止
        highcut=0.5,   # 血流动力学响应的高频截止
        filter_type=FilterType.BUTTERWORTH,
        filter_order=4,
        detrend_method=DetrendMethod.LINEAR,
        remove_baseline=True,
        normalize_method="zscore",
        
        # fNIRS特有参数
        motion_correction_method=MotionCorrectionMethod.SPLINE,
        motion_correction_threshold=3.0,
        use_channel_quality_assessment=True,
        snr_threshold=15.0,
        use_short_channel_regression=True,
        short_channel_distance_threshold=1.0,
        remove_physiological_noise=True,
        baseline_correction_window=(-5.0, 0.0)
    )
    
    # 2. 创建fNIRS预处理器
    fnirs_processor = fNIRSPreprocessor(fnirs_config)
    
    # 3. 加载fNIRS数据（示例）
    data_dict = load_fNIRS_data("example_fnirs_data.npy")
    
    # 4. 处理fNIRS数据
    processed_data = fnirs_processor.process_fNIRS(
        data_dict,
        modality="fnirs",
        return_hb_types=["HbO", "HbR", "HbT"]
    )
    
    # 5. 可视化预处理结果
    fnirs_processor.visualize_preprocessing(
        processed_data,
        modality="fnirs",
        channel_idx=0,
        wavelength_idx=0
    )
    
    # 6. 保存处理结果
    save_fNIRS_results(processed_data, "processed_fnirs_data.pkl")
    
    print("fNIRS预处理完成！")