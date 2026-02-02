# -*- coding: utf-8 -*-
"""
EEG信号专用预处理模块
基于通用预处理模块构建，专门针对脑电图信号特性优化
包含EEG特有的预处理步骤：重参考、ICA伪迹去除、眼电校正等
"""

import numpy as np
import mne
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs, EOGRegression
from mne_icalabel import label_components
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, field
from enum import Enum

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

        def process(self, data_dict, modality="EEG", channels=None):
            return data_dict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ====================== EEG专用枚举和配置 ======================

class ReferenceType(Enum):
    """重参考类型枚举"""
    AVERAGE = "average"  # 平均参考
    LINKED_MASTOIDS = "linked_mastoids"  # 双侧乳突参考
    CZ = "cz"  # Cz参考
    REST = "rest"  # REST参考（参考电极标准化技术）
    NONE = "none"  # 不进行重参考（保持原始参考）


class ICAMethod(Enum):
    """ICA方法枚举"""
    INFOMAX = "infomax"  # Infomax算法（默认）
    FASTICA = "fastica"  # FastICA算法
    PICARD = "picard"  # Picard算法
    EXTENDED_INFOMAX = "extended_infomax"  # 扩展Infomax


class ArtifactRemovalMethod(Enum):
    """伪迹去除方法枚举"""
    ICA_AUTO = "ica_auto"  # ICA自动标记去除
    ICA_MANUAL = "ica_manual"  # ICA手动标记去除
    REGRESSION = "regression"  # 回归校正（EOG/ECG）
    SSP = "ssp"  # 信号空间投影
    ASR = "asr"  # 自动分段回归（Artifact Subspace Reconstruction）
    ADAPTIVE = "adaptive"  # 自适应滤波


@dataclass
class EEGPreprocessingConfig(PreprocessingConfig):
    """
    EEG预处理配置类
    继承通用预处理配置，添加EEG特有参数
    """
    # ========== 重参考配置 ==========
    reference_type: ReferenceType = ReferenceType.AVERAGE
    reference_channels: Optional[List[str]] = None  # 自定义参考通道

    # ========== ICA配置 ==========
    use_ica: bool = True  # 是否使用ICA
    ica_method: ICAMethod = ICAMethod.INFOMAX  # ICA算法
    ica_n_components: Optional[Union[int, float]] = None  # ICA成分数，None为自动
    ica_max_iter: int = 500  # ICA最大迭代次数
    ica_random_state: int = 42  # 随机种子

    # ========== 伪迹去除配置 ==========
    artifact_removal: ArtifactRemovalMethod = ArtifactRemovalMethod.ICA_AUTO
    eog_channels: Optional[List[str]] = None  # EOG通道名称
    ecg_channels: Optional[List[str]] = None  # ECG通道名称

    # ========== 坏道处理配置 ==========
    interpolate_bad_channels: bool = True  # 是否插值坏道
    bad_channel_threshold: float = 3.0  # 坏道检测阈值（标准差倍数）
    max_bad_channels: float = 0.1  # 最大坏道比例（0-1）

    # ========== 分段处理配置 ==========
    epoch_data: bool = False  # 是否分段
    epoch_tmin: float = -0.2  # 分段开始时间（相对于事件）
    epoch_tmax: float = 1.0  # 分段结束时间
    baseline_correction: Tuple[float, float] = (-0.2, 0.0)  # 基线校正时间窗口

    # ========== 高级滤波配置 ==========
    # 针对EEG信号的专用滤波配置
    use_highpass: bool = True  # 使用高通滤波（去除低频漂移）
    highpass_freq: float = 0.5  # 高通频率（Hz）
    use_lowpass: bool = True  # 使用低通滤波（去除高频噪声）
    lowpass_freq: float = 45.0  # 低通频率（Hz）

    # ========== 伪迹拒绝配置 ==========
    reject_by_amplitude: bool = True  # 根据振幅拒绝
    rejection_threshold: float = 100e-6  # 拒绝阈值（uV）
    flat_threshold: float = 1e-6  # 平坦信号阈值

    # ========== 其他EEG特有配置 ==========
    montage: str = "standard_1020"  # 电极位置模板
    line_freq: float = 50.0  # 工频频率（50或60Hz）
    downsample_to: Optional[float] = None  # 降采样目标频率


# ====================== EEG专用预处理器 ======================

class EEGPreprocessor:
    """
    EEG信号专用预处理器
    集成通用预处理功能，添加EEG特有的处理步骤
    """

    def __init__(self, config: Optional[EEGPreprocessingConfig] = None):
        """
        初始化EEG预处理器

        Args:
            config: EEG预处理配置，None则使用默认配置
        """
        self.config = config if config is not None else EEGPreprocessingConfig()
        self.general_preprocessor = GeneralPreprocessor(self.config)
        self.history = []
        self.ica_components = None  # 存储ICA组件信息
        self.bad_channels = []  # 存储检测到的坏道

    def process(self, data_dict: Dict[str, Any],
                modality: str = "EEG",
                channels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        主处理函数，执行完整的EEG预处理流程

        Args:
            data_dict: 四层结构数据字典
            modality: 信号模态（应设为"EEG"）
            channels: 指定处理的通道

        Returns:
            更新后的数据字典
        """
        if modality != "EEG":
            logger.warning(f"EEG预处理器建议用于EEG信号，当前模态为{modality}")

        # 验证输入数据
        self._validate_eeg_input(data_dict, modality)

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
        if "eeg_preprocessing" not in data_dict["processed"]:
            data_dict["processed"]["eeg_preprocessing"] = {}

        data_dict["processed"]["eeg_preprocessing"]["original_data"] = original_data
        data_dict["processed"]["eeg_preprocessing"]["original_srate"] = sampling_rate
        data_dict["processed"]["eeg_preprocessing"]["channel_names"] = channel_names.copy()

        # ========== 第1阶段：基础预处理 ==========
        logger.info("开始第1阶段：基础预处理")

        # 1.1 创建MNE Raw对象（便于使用MNE的高级功能）
        raw = self._create_mne_raw(data_dict, modality)

        # 1.2 设置电极位置
        raw = self._set_montage(raw)
        process_record["steps"].append({
            "step": "set_montage",
            "montage": self.config.montage
        })

        # 1.3 检测和标记坏道
        raw, bad_channels = self._detect_bad_channels(raw)
        self.bad_channels = bad_channels
        if bad_channels:
            process_record["steps"].append({
                "step": "detect_bad_channels",
                "bad_channels": bad_channels,
                "threshold": self.config.bad_channel_threshold
            })

        # ========== 第2阶段：滤波和去噪 ==========
        logger.info("开始第2阶段：滤波和去噪")

        # 2.1 使用通用预处理进行基础滤波和去噪
        data_dict = self.general_preprocessor.process(data_dict, modality, channels)

        # 2.2 EEG专用滤波（如果配置了）
        if self.config.use_highpass or self.config.use_lowpass:
            raw = self._apply_eeg_specific_filtering(raw)
            process_record["steps"].append({
                "step": "eeg_specific_filtering",
                "highpass": f"{self.config.highpass_freq}Hz" if self.config.use_highpass else "None",
                "lowpass": f"{self.config.lowpass_freq}Hz" if self.config.use_lowpass else "None"
            })

        # 2.3 谐波陷波滤波（去除工频谐波）
        raw = self._apply_harmonic_notch(raw)
        process_record["steps"].append({
            "step": "harmonic_notch_filter",
            "base_freq": self.config.line_freq,
            "n_harmonics": 5
        })

        # ========== 第3阶段：重参考 ==========
        logger.info("开始第3阶段：重参考")

        raw = self._apply_rereferencing(raw)
        process_record["steps"].append({
            "step": "rereferencing",
            "type": self.config.reference_type.value,
            "channels": self.config.reference_channels
        })

        # ========== 第4阶段：伪迹去除 ==========
        logger.info("开始第4阶段：伪迹去除")

        # 4.1 插值坏道
        if self.config.interpolate_bad_channels and bad_channels:
            raw = self._interpolate_bad_channels(raw)
            process_record["steps"].append({
                "step": "interpolate_bad_channels",
                "bad_channels": bad_channels
            })

        # 4.2 ICA伪迹去除
        if self.config.use_ica:
            raw, ica_info = self._apply_ica_artifact_removal(raw)
            self.ica_components = ica_info
            process_record["steps"].append({
                "step": "ica_artifact_removal",
                "method": self.config.ica_method.value,
                "components_removed": ica_info.get("n_components_removed", 0)
            })

        # 4.3 EOG/ECG回归校正（如果提供了眼电/心电通道）
        if self.config.eog_channels or self.config.ecg_channels:
            raw = self._apply_regression_correction(raw, data_dict)
            process_record["steps"].append({
                "step": "regression_correction",
                "eog_channels": self.config.eog_channels,
                "ecg_channels": self.config.ecg_channels
            })

        # ========== 第5阶段：降采样和最终处理 ==========
        logger.info("开始第5阶段：降采样和最终处理")

        # 5.1 降采样（如果需要）
        if self.config.downsample_to is not None and self.config.downsample_to < sampling_rate:
            raw = self._downsample(raw)
            process_record["steps"].append({
                "step": "downsample",
                "original_fs": sampling_rate,
                "target_fs": self.config.downsample_to
            })

        # 5.2 提取处理后的数据
        processed_data = raw.get_data()

        # 5.3 最终标准化
        processed_data = self._final_normalization(processed_data)
        process_record["steps"].append({
            "step": "final_normalization",
            "method": self.config.normalize_method
        })

        # ========== 第6阶段：分段处理（如果需要） ==========
        if self.config.epoch_data and "event" in data_dict:
            logger.info("开始第6阶段：分段处理")
            epochs_info = self._create_epochs(raw, data_dict["event"])
            data_dict["processed"]["eeg_preprocessing"]["epochs"] = epochs_info
            process_record["steps"].append({
                "step": "epoching",
                "tmin": self.config.epoch_tmin,
                "tmax": self.config.epoch_tmax,
                "n_epochs": len(epochs_info["data"]) if "data" in epochs_info else 0
            })

        # ========== 更新数据字典 ==========
        data_dict["signal"][modality]["data"] = processed_data
        data_dict["signal"][modality]["sampling_rate"] = raw.info["sfreq"]

        # 更新处理历史
        data_dict["processed"]["eeg_preprocessing"]["history"] = process_record
        data_dict["processed"]["eeg_preprocessing"]["config"] = self._config_to_dict()
        data_dict["processed"]["eeg_preprocessing"]["bad_channels"] = self.bad_channels
        data_dict["processed"]["eeg_preprocessing"]["ica_info"] = self.ica_components

        self.history.append(process_record)

        logger.info(f"EEG预处理完成，共执行{len(process_record['steps'])}个步骤")

        return data_dict

    # ====================== EEG特有方法 ======================

    def _validate_eeg_input(self, data_dict: Dict, modality: str):
        """
        验证EEG输入数据

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
                raise ValueError(f"EEG信号必须包含'{key}'键")

        # 检查数据格式
        data = signal_info["data"]
        if not isinstance(data, np.ndarray):
            raise ValueError("EEG数据必须是numpy数组")

        if len(data.shape) != 2:
            raise ValueError("EEG数据必须是2维数组 (channels × samples)")

        # 检查通道数量
        n_channels = len(signal_info["channel_names"])
        if n_channels != data.shape[0]:
            raise ValueError(f"通道数量({n_channels})与数据维度({data.shape[0]})不匹配")

        # 检查采样率
        sampling_rate = signal_info["sampling_rate"]
        if sampling_rate <= 0:
            raise ValueError(f"采样率必须大于0，当前为{sampling_rate}")

        logger.info(f"EEG数据验证通过: {n_channels}通道, {data.shape[1]}采样点, {sampling_rate}Hz")

    def _create_mne_raw(self, data_dict: Dict, modality: str) -> mne.io.RawArray:
        """
        从数据字典创建MNE Raw对象

        Args:
            data_dict: 数据字典
            modality: 信号模态

        Returns:
            MNE Raw对象
        """
        signal_info = data_dict["signal"][modality]
        data = signal_info["data"]
        sfreq = signal_info["sampling_rate"]
        ch_names = signal_info["channel_names"]

        # 创建info对象
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types='eeg'
        )

        # 创建Raw对象
        raw = mne.io.RawArray(data, info)

        # 设置单位（假设为uV）
        if "unit" in signal_info:
            unit = signal_info["unit"]
            if unit.lower() in ["uv", "μv"]:
                # MNE默认单位为V，需要转换为V
                raw._data = raw._data * 1e-6

        logger.info(f"创建MNE Raw对象: {len(ch_names)}通道, {sfreq}Hz, {data.shape[1] / sfreq:.2f}秒")

        return raw

    def _set_montage(self, raw: mne.io.RawArray) -> mne.io.RawArray:
        """
        设置电极位置模板

        Args:
            raw: MNE Raw对象

        Returns:
            更新后的Raw对象
        """
        try:
            raw.set_montage(self.config.montage)
            logger.info(f"已设置电极位置模板: {self.config.montage}")
        except Exception as e:
            logger.warning(f"无法设置电极位置模板 {self.config.montage}: {str(e)}")
            # 尝试使用标准1020模板
            try:
                raw.set_montage("standard_1020")
                logger.info("已使用标准1020模板")
            except:
                logger.warning("无法设置任何电极位置模板，空间信息可能不准确")

        return raw

    def _detect_bad_channels(self, raw: mne.io.RawArray) -> Tuple[mne.io.RawArray, List[str]]:
        """
        检测和标记坏道

        Args:
            raw: MNE Raw对象

        Returns:
            (更新后的Raw对象, 坏道列表)
        """
        bad_channels = []

        # 方法1: 基于振幅异常检测
        if self.config.reject_by_amplitude:
            # 计算每个通道的振幅范围
            data = raw.get_data()
            channel_ranges = np.ptp(data, axis=1)  # 峰峰值

            # 检测异常通道
            median_range = np.median(channel_ranges)
            mad_range = np.median(np.abs(channel_ranges - median_range))

            # 使用MAD阈值
            threshold = median_range + self.config.bad_channel_threshold * mad_range * 1.4826

            for i, ch_name in enumerate(raw.ch_names):
                if channel_ranges[i] > threshold:
                    bad_channels.append(ch_name)
                    logger.debug(f"检测到高振幅坏道: {ch_name}, 范围: {channel_ranges[i]:.2e}, 阈值: {threshold:.2e}")

        # 方法2: 检测平坦通道
        if self.config.flat_threshold > 0:
            data = raw.get_data()
            for i, ch_name in enumerate(raw.ch_names):
                if np.std(data[i, :]) < self.config.flat_threshold:
                    if ch_name not in bad_channels:
                        bad_channels.append(ch_name)
                        logger.debug(f"检测到平坦通道: {ch_name}, 标准差: {np.std(data[i, :]):.2e}")

        # 检查坏道数量是否超过限制
        max_bad = int(len(raw.ch_names) * self.config.max_bad_channels)
        if len(bad_channels) > max_bad:
            logger.warning(f"检测到{len(bad_channels)}个坏道，超过最大限制{max_bad}，将标记最异常的{max_bad}个")
            # 重新计算异常程度，只保留最异常的
            data = raw.get_data()
            anomalies = []
            for i, ch_name in enumerate(raw.ch_names):
                if ch_name in bad_channels:
                    # 使用峰峰值作为异常指标
                    anomaly_score = np.ptp(data[i, :]) / np.median(np.ptp(data, axis=1))
                    anomalies.append((ch_name, anomaly_score))

            # 按异常程度排序，保留最异常的
            anomalies.sort(key=lambda x: x[1], reverse=True)
            bad_channels = [ch for ch, _ in anomalies[:max_bad]]

        # 标记坏道
        if bad_channels:
            raw.info['bads'] = bad_channels
            logger.info(f"标记了{len(bad_channels)}个坏道: {bad_channels}")

        return raw, bad_channels

    def _apply_eeg_specific_filtering(self, raw: mne.io.RawArray) -> mne.io.RawArray:
        """
        应用EEG专用滤波

        Args:
            raw: MNE Raw对象

        Returns:
            滤波后的Raw对象
        """
        # 高通滤波（去除低频漂移）
        if self.config.use_highpass and self.config.highpass_freq > 0:
            raw = raw.copy().filter(
                l_freq=self.config.highpass_freq,
                h_freq=None,
                method='fir',
                fir_design='firwin',
                phase='zero-double',
                verbose=False
            )

        # 低通滤波（去除高频噪声）
        if self.config.use_lowpass and self.config.lowpass_freq > 0:
            raw = raw.copy().filter(
                l_freq=None,
                h_freq=self.config.lowpass_freq,
                method='fir',
                fir_design='firwin',
                phase='zero-double',
                verbose=False
            )

        return raw

    def _apply_harmonic_notch(self, raw: mne.io.RawArray) -> mne.io.RawArray:
        """
        应用谐波陷波滤波去除工频干扰

        Args:
            raw: MNE Raw对象

        Returns:
            滤波后的Raw对象
        """
        sfreq = raw.info['sfreq']
        line_freq = self.config.line_freq

        # 计算需要去除的谐波
        max_harmonic = int(sfreq / 2 / line_freq)
        if max_harmonic < 1:
            logger.warning(f"工频频率{line_freq}Hz太高，无法应用谐波陷波")
            return raw

        # 生成谐波频率列表
        freqs = np.arange(line_freq, sfreq / 2, line_freq)

        # 应用陷波滤波
        if len(freqs) > 0:
            raw = raw.copy().notch_filter(
                freqs=freqs,
                picks='eeg',
                method='fir',
                fir_design='firwin',
                phase='zero-double',
                verbose=False
            )
            logger.info(f"应用了谐波陷波滤波，去除{len(freqs)}个谐波")

        return raw

    def _apply_rereferencing(self, raw: mne.io.RawArray) -> mne.io.RawArray:
        """
        应用重参考

        Args:
            raw: MNE Raw对象

        Returns:
            重参考后的Raw对象
        """
        if self.config.reference_type == ReferenceType.NONE:
            logger.info("不进行重参考")
            return raw

        # 移除已有的参考
        raw = raw.copy()

        if self.config.reference_type == ReferenceType.AVERAGE:
            # 平均参考
            raw.set_eeg_reference(ref_channels='average', projection=False)
            logger.info("应用平均参考")

        elif self.config.reference_type == ReferenceType.LINKED_MASTOIDS:
            # 双侧乳突参考
            ref_channels = ['M1', 'M2']
            available_refs = [ch for ch in ref_channels if ch in raw.ch_names]
            if len(available_refs) >= 1:
                raw.set_eeg_reference(ref_channels=available_refs, projection=False)
                logger.info(f"应用双侧乳突参考: {available_refs}")
            else:
                logger.warning("无法找到乳突通道，使用平均参考")
                raw.set_eeg_reference(ref_channels='average', projection=False)

        elif self.config.reference_type == ReferenceType.CZ:
            # Cz参考
            if 'Cz' in raw.ch_names:
                raw.set_eeg_reference(ref_channels=['Cz'], projection=False)
                logger.info("应用Cz参考")
            else:
                logger.warning("无法找到Cz通道，使用平均参考")
                raw.set_eeg_reference(ref_channels='average', projection=False)

        elif self.config.reference_type == ReferenceType.REST:
            # REST参考（需要额外的处理）
            logger.warning("REST参考需要专用算法，暂时使用平均参考")
            raw.set_eeg_reference(ref_channels='average', projection=False)

        elif self.config.reference_channels:
            # 自定义参考通道
            available_refs = [ch for ch in self.config.reference_channels if ch in raw.ch_names]
            if available_refs:
                raw.set_eeg_reference(ref_channels=available_refs, projection=False)
                logger.info(f"应用自定义参考: {available_refs}")
            else:
                logger.warning("自定义参考通道不存在，使用平均参考")
                raw.set_eeg_reference(ref_channels='average', projection=False)

        return raw

    def _interpolate_bad_channels(self, raw: mne.io.RawArray) -> mne.io.RawArray:
        """
        插值坏道

        Args:
            raw: MNE Raw对象

        Returns:
            插值后的Raw对象
        """
        if not raw.info['bads']:
            return raw

        # 检查是否有足够的电极位置信息
        if raw.info['dig'] is None:
            logger.warning("没有电极位置信息，无法进行空间插值")
            return raw

        try:
            # 使用球面样条插值
            raw = raw.copy().interpolate_bads(reset_bads=True)
            logger.info(f"已插值{len(self.bad_channels)}个坏道")
        except Exception as e:
            logger.warning(f"坏道插值失败: {str(e)}")
            # 移除坏道标记
            raw.info['bads'] = []

        return raw

    def _apply_ica_artifact_removal(self, raw: mne.io.RawArray) -> Tuple[mne.io.RawArray, Dict]:
        """
        应用ICA伪迹去除

        Args:
            raw: MNE Raw对象

        Returns:
            (处理后的Raw对象, ICA信息字典)
        """
        ica_info = {
            "method": self.config.ica_method.value,
            "n_components": None,
            "components_removed": [],
            "n_components_removed": 0
        }

        # 确定ICA成分数量
        if self.config.ica_n_components is None:
            # 使用MNE的默认策略：使用解释95%方差的成分数量
            n_components = None
        elif isinstance(self.config.ica_n_components, float) and 0 < self.config.ica_n_components < 1:
            # 解释指定方差比例
            n_components = self.config.ica_n_components
        else:
            # 固定成分数量
            n_components = int(self.config.ica_n_components)

        ica_info["n_components"] = n_components

        # 创建ICA对象
        ica = ICA(
            n_components=n_components,
            method=self.config.ica_method.value,
            max_iter=self.config.ica_max_iter,
            random_state=self.config.ica_random_state,
            fit_params=dict(extended=True) if self.config.ica_method == ICAMethod.EXTENDED_INFOMAX else None
        )

        # 拟合ICA
        try:
            raw_for_ica = raw.copy()

            # 高通滤波（1Hz）以改善ICA性能
            raw_for_ica.filter(l_freq=1.0, h_freq=None, method='fir', verbose=False)

            # 拟合ICA
            ica.fit(raw_for_ica, picks='eeg', verbose=False)

            # 自动标记成分
            if self.config.artifact_removal == ArtifactRemovalMethod.ICA_AUTO:
                # 使用ICLabel自动标记
                try:
                    ic_labels = label_components(raw_for_ica, ica, method='iclabel')

                    # 确定要排除的成分（非大脑成分）
                    exclude_idx = []
                    labels = ic_labels['labels']
                    for i, label in enumerate(labels):
                        if label not in ['brain', 'other']:
                            exclude_idx.append(i)

                    ica_info["components_removed"] = exclude_idx
                    ica_info["n_components_removed"] = len(exclude_idx)

                    if exclude_idx:
                        logger.info(f"ICA自动标记排除{len(exclude_idx)}个成分: {exclude_idx}")
                        # 应用ICA
                        ica.apply(raw, exclude=exclude_idx)
                    else:
                        logger.info("ICA未发现需要排除的成分")

                except Exception as e:
                    logger.warning(f"ICA自动标记失败: {str(e)}，使用手动标记策略")
                    # 回退到基于峰度的自动检测
                    exclude_idx = self._detect_artifact_components_by_kurtosis(ica)
                    ica_info["components_removed"] = exclude_idx
                    ica_info["n_components_removed"] = len(exclude_idx)

                    if exclude_idx:
                        ica.apply(raw, exclude=exclude_idx)

            elif self.config.artifact_removal == ArtifactRemovalMethod.ICA_MANUAL:
                # 手动标记（需要用户交互，这里只返回ICA对象信息）
                logger.info("ICA手动标记模式，需要用户交互")
                ica_info["ica_object"] = ica  # 保存ICA对象供后续使用

            else:
                logger.warning(f"不支持的伪迹去除方法: {self.config.artifact_removal}")

        except Exception as e:
            logger.error(f"ICA处理失败: {str(e)}")
            # 如果ICA失败，返回原始数据

        return raw, ica_info

    def _detect_artifact_components_by_kurtosis(self, ica: ICA) -> List[int]:
        """
        基于峰度检测伪迹成分

        Args:
            ica: 拟合好的ICA对象

        Returns:
            要排除的成分索引列表
        """
        exclude_idx = []

        # 计算每个成分的峰度
        sources = ica.get_sources(ica._raw).get_data()
        kurtosis_values = []

        for i in range(sources.shape[0]):
            source = sources[i, :]
            kurt = np.mean((source - np.mean(source)) ** 4) / (np.std(source) ** 4) - 3
            kurtosis_values.append(abs(kurt))

        # 检测异常峰度（使用MAD方法）
        median_kurt = np.median(kurtosis_values)
        mad_kurt = np.median(np.abs(kurtosis_values - median_kurt))

        threshold = median_kurt + 3 * mad_kurt * 1.4826

        for i, kurt in enumerate(kurtosis_values):
            if kurt > threshold:
                exclude_idx.append(i)

        if exclude_idx:
            logger.info(f"基于峰度检测到{len(exclude_idx)}个伪迹成分: {exclude_idx}")

        return exclude_idx

    def _apply_regression_correction(self, raw: mne.io.RawArray, data_dict: Dict) -> mne.io.RawArray:
        """
        应用回归校正去除EOG/ECG伪迹

        Args:
            raw: MNE Raw对象
            data_dict: 数据字典（可能包含EOG/ECG数据）

        Returns:
            校正后的Raw对象
        """
        # 检查是否有EOG/ECG数据
        has_eog = False
        has_ecg = False

        # 查找EOG/ECG数据
        for modality in data_dict["signal"]:
            if modality.upper() == "EOG" and self.config.eog_channels:
                has_eog = True
                eog_data = data_dict["signal"][modality]["data"]
                eog_sfreq = data_dict["signal"][modality]["sampling_rate"]

            if modality.upper() == "ECG" and self.config.ecg_channels:
                has_ecg = True
                ecg_data = data_dict["signal"][modality]["data"]
                ecg_sfreq = data_dict["signal"][modality]["sampling_rate"]

        # 暂时简化处理，实际应用中需要对齐时间并创建虚拟通道
        logger.info("回归校正功能需要时间对齐的EOG/ECG数据，当前版本暂未实现完整功能")

        return raw

    def _downsample(self, raw: mne.io.RawArray) -> mne.io.RawArray:
        """
        降采样

        Args:
            raw: MNE Raw对象

        Returns:
            降采样后的Raw对象
        """
        if self.config.downsample_to >= raw.info['sfreq']:
            logger.warning(f"降采样频率({self.config.downsample_to}Hz)不低于当前频率({raw.info['sfreq']}Hz)，跳过降采样")
            return raw

        # 应用抗混叠滤波
        raw = raw.copy().filter(
            l_freq=None,
            h_freq=self.config.downsample_to * 0.8,  # 保留80%奈奎斯特频率
            method='fir',
            fir_design='firwin',
            verbose=False
        )

        # 降采样
        raw = raw.resample(self.config.downsample_to, npad='auto')

        logger.info(f"降采样到{self.config.downsample_to}Hz")

        return raw

    def _final_normalization(self, data: np.ndarray) -> np.ndarray:
        """
        最终标准化处理

        Args:
            data: 输入数据

        Returns:
            标准化后的数据
        """
        n_channels, n_samples = data.shape
        normalized_data = np.zeros_like(data)

        for i in range(n_channels):
            channel_data = data[i, :]

            if self.config.normalize_method == "zscore":
                # Z-score标准化
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                if std_val > 0:
                    normalized_data[i, :] = (channel_data - mean_val) / std_val
                else:
                    normalized_data[i, :] = np.zeros_like(channel_data)

            elif self.config.normalize_method == "minmax":
                # Min-Max归一化
                min_val = np.min(channel_data)
                max_val = np.max(channel_data)
                if max_val > min_val:
                    normalized_data[i, :] = (channel_data - min_val) / (max_val - min_val)
                else:
                    normalized_data[i, :] = np.zeros_like(channel_data)

            elif self.config.normalize_method == "robust":
                # 鲁棒标准化
                median_val = np.median(channel_data)
                q75, q25 = np.percentile(channel_data, [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    normalized_data[i, :] = (channel_data - median_val) / iqr
                else:
                    normalized_data[i, :] = (channel_data - median_val)

            elif self.config.normalize_method == "unit_norm":
                # 单位范数归一化
                norm = np.linalg.norm(channel_data)
                if norm > 0:
                    normalized_data[i, :] = channel_data / norm
                else:
                    normalized_data[i, :] = channel_data

        return normalized_data

    def _create_epochs(self, raw: mne.io.RawArray, event_info: Dict) -> Dict:
        """
        创建分段数据

        Args:
            raw: MNE Raw对象
            event_info: 事件信息字典

        Returns:
            分段信息字典
        """
        epochs_info = {
            "tmin": self.config.epoch_tmin,
            "tmax": self.config.epoch_tmax,
            "baseline": self.config.baseline_correction
        }

        # 检查是否有足够的事件信息
        if "event_time" not in event_info or not event_info["event_time"]:
            logger.warning("没有事件信息，无法进行分段")
            return epochs_info

        # 创建事件数组
        event_times = event_info["event_time"]
        event_samples = [int(t * raw.info['sfreq']) for t in event_times]

        if "event_id" in event_info:
            event_ids = event_info["event_id"]
        else:
            event_ids = [1] * len(event_times)

        # 创建MNE事件数组
        events = np.column_stack([event_samples, [0] * len(event_samples), event_ids])

        # 创建分段
        try:
            epochs = mne.Epochs(
                raw,
                events=events,
                event_id=None,  # 使用所有事件
                tmin=self.config.epoch_tmin,
                tmax=self.config.epoch_tmax,
                baseline=self.config.baseline_correction,
                preload=True,
                verbose=False
            )

            # 应用振幅拒绝
            if self.config.reject_by_amplitude:
                epochs.drop_bad(reject=dict(eeg=self.config.rejection_threshold * 1e-6))

            # 提取分段数据
            epochs_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

            epochs_info["data"] = epochs_data
            epochs_info["events"] = events
            epochs_info["event_ids"] = event_ids
            epochs_info["n_epochs"] = len(epochs_data)
            epochs_info["epoch_times"] = epochs.times

            logger.info(f"创建了{len(epochs_data)}个分段，每个分段{epochs_data.shape[2]}个时间点")

        except Exception as e:
            logger.error(f"分段创建失败: {str(e)}")

        return epochs_info

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

class EEGConfigFactory:
    """
    EEG预处理配置工厂
    提供不同实验范式的推荐配置
    """

    @staticmethod
    def create_motor_imagery_config() -> EEGPreprocessingConfig:
        """
        创建运动想象实验的EEG预处理配置

        Returns:
            运动想象专用配置
        """
        return EEGPreprocessingConfig(
            # 滤波配置（运动想象关注8-30Hz频段）
            use_highpass=True,
            highpass_freq=1.0,
            use_lowpass=True,
            lowpass_freq=40.0,

            # 重参考配置
            reference_type=ReferenceType.AVERAGE,

            # ICA配置
            use_ica=True,
            ica_method=ICAMethod.INFOMAX,

            # 分段配置
            epoch_data=True,
            epoch_tmin=-1.0,
            epoch_tmax=4.0,
            baseline_correction=(-1.0, 0.0),

            # 其他配置
            montage="standard_1020",
            line_freq=50.0,
            downsample_to=250.0
        )

    @staticmethod
    def create_p300_config() -> EEGPreprocessingConfig:
        """
        创建P300实验的EEG预处理配置

        Returns:
            P300专用配置
        """
        return EEGPreprocessingConfig(
            # 滤波配置（P300需要低频信息）
            use_highpass=True,
            highpass_freq=0.1,
            use_lowpass=True,
            lowpass_freq=20.0,
            filter_type=FilterType.BESSEL,  # 贝塞尔滤波器保持波形形状

            # 重参考配置
            reference_type=ReferenceType.AVERAGE,

            # ICA配置
            use_ica=True,
            ica_method=ICAMethod.INFOMAX,

            # 分段配置
            epoch_data=True,
            epoch_tmin=-0.2,
            epoch_tmax=1.0,
            baseline_correction=(-0.2, 0.0),

            # 其他配置
            montage="standard_1020",
            line_freq=50.0,
            downsample_to=250.0,

            # 伪迹拒绝
            reject_by_amplitude=True,
            rejection_threshold=100e-6
        )

    @staticmethod
    def create_ssvep_config() -> EEGPreprocessingConfig:
        """
        创建SSVEP实验的EEG预处理配置

        Returns:
            SSVEP专用配置
        """
        return EEGPreprocessingConfig(
            # 滤波配置（SSVEP关注特定频率）
            use_highpass=True,
            highpass_freq=5.0,
            use_lowpass=True,
            lowpass_freq=60.0,

            # 重参考配置
            reference_type=ReferenceType.AVERAGE,

            # ICA配置
            use_ica=True,
            ica_method=ICAMethod.EXTENDED_INFOMAX,

            # 谐波陷波加强（SSVEP对工频干扰敏感）
            line_freq=50.0,

            # 分段配置
            epoch_data=True,
            epoch_tmin=0.0,
            epoch_tmax=5.0,
            baseline_correction=None,

            # 其他配置
            montage="standard_1020",
            downsample_to=500.0  # SSVEP需要较高采样率
        )

    @staticmethod
    def create_resting_state_config() -> EEGPreprocessingConfig:
        """
        创建静息态实验的EEG预处理配置

        Returns:
            静息态专用配置
        """
        return EEGPreprocessingConfig(
            # 滤波配置（静息态需要低频信息）
            use_highpass=True,
            highpass_freq=0.5,
            use_lowpass=True,
            lowpass_freq=45.0,

            # 重参考配置
            reference_type=ReferenceType.AVERAGE,

            # ICA配置（静息态需要强伪迹去除）
            use_ica=True,
            ica_method=ICAMethod.INFOMAX,
            artifact_removal=ArtifactRemovalMethod.ICA_AUTO,

            # 不进行分段（连续数据）
            epoch_data=False,

            # 其他配置
            montage="standard_1020",
            line_freq=50.0,

            # 坏道处理更严格
            bad_channel_threshold=2.5,
            max_bad_channels=0.05  # 最多5%坏道
        )


# ====================== 使用示例和测试函数 ======================

def test_eeg_preprocessing():
    """
    测试EEG预处理功能
    """
    # 创建模拟数据
    np.random.seed(42)

    # 模拟参数
    n_channels = 32
    n_samples = 10000  # 10秒数据，1000Hz采样率
    sampling_rate = 1000

    # 生成模拟EEG数据（包含噪声和伪迹）
    time = np.arange(n_samples) / sampling_rate

    # 基础EEG信号（模拟alpha波）
    eeg_data = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        # 模拟alpha波（8-12Hz）
        alpha_freq = 10 + np.random.randn() * 1.0
        alpha_amp = np.random.rand() * 20 + 10  # 10-30uV

        # 模拟beta波（13-30Hz）
        beta_freq = 20 + np.random.randn() * 5.0
        beta_amp = np.random.rand() * 10 + 5  # 5-15uV

        # 组合信号
        eeg_data[i, :] = (
                alpha_amp * np.sin(2 * np.pi * alpha_freq * time) +
                beta_amp * np.sin(2 * np.pi * beta_freq * time)
        )

    # 添加噪声
    noise_level = 5.0
    eeg_data += np.random.randn(n_channels, n_samples) * noise_level

    # 添加工频干扰（50Hz）
    line_noise = 20.0 * np.sin(2 * np.pi * 50.0 * time)
    eeg_data += line_noise.reshape(1, -1)

    # 添加眼电伪迹（模拟眨眼）
    blink_times = [1.5, 4.2, 7.8]
    for blink_time in blink_times:
        blink_sample = int(blink_time * sampling_rate)
        blink_duration = int(0.3 * sampling_rate)  # 300ms
        blink_signal = 100.0 * np.hanning(blink_duration)  # 100uV眨眼
        start = max(0, blink_sample - blink_duration // 2)
        end = min(n_samples, start + blink_duration)
        actual_duration = end - start

        # 主要影响前部电极
        for i in range(min(8, n_channels)):  # 前8个通道
            eeg_data[i, start:end] += blink_signal[:actual_duration]

    # 创建通道名称（标准10-20系统）
    channel_names = [
                        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                        'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',
                        'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6',
                        'TP9', 'TP10', 'POz', 'PO4'
                    ][:n_channels]

    # 创建模拟事件
    event_times = [1.0, 3.0, 5.0, 7.0, 9.0]
    event_ids = [1, 2, 1, 2, 1]  # 两类事件

    # 构建数据字典
    data_dict = {
        "meta": {
            "subject_id": "S01",
            "session_id": "test_session",
            "task": "motor_imagery",
            "modality": ["EEG"],
            "device": "Simulated",
            "sampling_rate": sampling_rate,
            "n_channels": n_channels,
            "channel_names": channel_names
        },
        "signal": {
            "EEG": {
                "data": eeg_data,
                "sampling_rate": sampling_rate,
                "unit": "uV",
                "channel_names": channel_names,
                "reference": "unknown",
                "time_offset": 0.0
            }
        },
        "event": {
            "event_id": event_ids,
            "event_label": ["left", "right", "left", "right", "left"],
            "event_time": event_times,
            "event_sample": [int(t * sampling_rate) for t in event_times],
            "duration": [2.0, 2.0, 2.0, 2.0, 2.0]
        },
        "processed": {}
    }

    print("=" * 60)
    print("EEG预处理测试")
    print("=" * 60)

    # 创建预处理器（使用运动想象配置）
    config = EEGConfigFactory.create_motor_imagery_config()
    preprocessor = EEGPreprocessor(config)

    # 执行预处理
    print("\n开始预处理...")
    processed_data = preprocessor.process(data_dict)

    # 显示处理结果
    print("\n预处理完成!")
    print(f"原始数据形状: {data_dict['signal']['EEG']['data'].shape}")
    print(f"处理后数据形状: {processed_data['signal']['EEG']['data'].shape}")

    # 显示处理历史
    history = processed_data['processed']['eeg_preprocessing']['history']
    print(f"\n处理步骤数量: {len(history['steps'])}")
    print("\n处理步骤详情:")
    for i, step in enumerate(history['steps']):
        print(f"  {i + 1:2d}. {step['step']:30s} | {str(step)[:50]}...")

    # 显示坏道信息
    if 'bad_channels' in processed_data['processed']['eeg_preprocessing']:
        bad_channels = processed_data['processed']['eeg_preprocessing']['bad_channels']
        if bad_channels:
            print(f"\n检测到的坏道: {bad_channels}")
        else:
            print("\n未检测到坏道")

    # 显示ICA信息
    if 'ica_info' in processed_data['processed']['eeg_preprocessing']:
        ica_info = processed_data['processed']['eeg_preprocessing']['ica_info']
        if 'n_components_removed' in ica_info:
            print(f"\nICA去除的成分数量: {ica_info['n_components_removed']}")

    # 显示分段信息
    if 'epochs' in processed_data['processed']['eeg_preprocessing']:
        epochs_info = processed_data['processed']['eeg_preprocessing']['epochs']
        if 'n_epochs' in epochs_info:
            print(f"\n创建的分段数量: {epochs_info['n_epochs']}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    return processed_data


# ====================== 主程序入口 ======================

if __name__ == "__main__":
    # 运行测试
    processed_data = test_eeg_preprocessing()

    # 示例：如何使用EEG预处理器
    print("\n" + "=" * 60)
    print("EEG预处理器使用示例")
    print("=" * 60)

    # 示例1：使用默认配置
    print("\n1. 使用默认配置:")
    config1 = EEGPreprocessingConfig()
    preprocessor1 = EEGPreprocessor(config1)
    print(f"  采样率: {config1.downsample_to or '保持原始'}")
    print(f"  参考方式: {config1.reference_type.value}")
    print(f"  ICA: {'启用' if config1.use_ica else '禁用'}")

    # 示例2：使用运动想象配置
    print("\n2. 使用运动想象配置:")
    config2 = EEGConfigFactory.create_motor_imagery_config()
    preprocessor2 = EEGPreprocessor(config2)
    print(f"  滤波范围: {config2.highpass_freq}-{config2.lowpass_freq}Hz")
    print(f"  分段时间: {config2.epoch_tmin}-{config2.epoch_tmax}s")
    print(f"  降采样: {config2.downsample_to}Hz")

    # 示例3：使用P300配置
    print("\n3. 使用P300配置:")
    config3 = EEGConfigFactory.create_p300_config()
    print(f"  滤波器类型: {config3.filter_type.value}")
    print(f"  基线校正: {config3.baseline_correction}")
    print(f"  振幅拒绝阈值: {config3.rejection_threshold}uV")

    print("\n" + "=" * 60)
