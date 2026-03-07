# -*- coding: utf-8 -*-

import numpy as np
import mne
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from mne_icalabel import label_components
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, field
from enum import Enum

# 导入通用预处理模块
try:
    from core.processing.preprocessing.preprocessing import GeneralPreprocessor, PreprocessingConfig, FilterType, WaveletType, DetrendMethod
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
    baseline_correction: Optional[Tuple[float, float]] = (-0.2, 0.0)  # 基线校正时间窗口

    # ========== 高级滤波配置 ==========
    # 针对EEG信号的专用滤波配置
    use_highpass: bool = True  # 使用高通滤波（去除低频漂移）
    highpass_freq: float = 0.5  # 高通频率（Hz）
    use_lowpass: bool = True  # 使用低通滤波（去除高频噪声）
    lowpass_freq: float = 45.0  # 低通频率（Hz）

    # ========== 伪迹拒绝配置 ==========
    reject_by_amplitude: bool = True  # 根据振幅拒绝
    rejection_threshold: float = 150e-6  # 拒绝阈值（uV），从100e-6增加到150e-6
    flat_threshold: float = 1e-6  # 平坦信号阈值

    # ========== 其他EEG特有配置 ==========
    montage: str = "standard_1020"  # 电极位置模板
    line_freq: float = 50.0  # 工频频率（50或60Hz）
    downsample_to: Optional[float] = None  # 降采样目标频率


# ====================== EEG专用预处理器 ======================

class EEGPreprocessor:
    """
    EEG信号专用预处理器（修复版）
    修复了分段创建为0的问题
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
        if self.config.downsample_to is not None and self.config.downsample_to < raw.info['sfreq']:
            raw = self._downsample(raw)
            process_record["steps"].append({
                "step": "downsample",
                "original_fs": raw.info['sfreq'],
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
                "n_epochs": epochs_info.get("n_epochs", 0)
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
        验证EEG输入数据（增强版，支持单通道和缺失通道信息）
        """
        if "signal" not in data_dict:
            raise ValueError("数据字典必须包含'signal'键")

        if modality not in data_dict["signal"]:
            # 如果指定的模态不存在，尝试创建一个默认的
            logger.warning(f"模态'{modality}'不在输入数据中，将创建默认结构")
            data_dict["signal"][modality] = {}

        signal_info = data_dict["signal"][modality]

        # 检查数据字段 - 这是必须的
        if "data" not in signal_info:
            raise ValueError(f"EEG信号必须包含'data'键")

        data = signal_info["data"]
        if not isinstance(data, np.ndarray):
            raise ValueError("EEG数据必须是numpy数组")

        if len(data.shape) != 2:
            # 如果是一维数据，reshape为二维 (1, n_samples)
            if len(data.shape) == 1:
                logger.info(f"检测到一维数据，将reshape为(1, {len(data)})")
                data = data.reshape(1, -1)
                signal_info["data"] = data
            else:
                raise ValueError("EEG数据必须是2维数组 (channels × samples)")

        # 检查采样率 - 这是必须的
        if "sampling_rate" not in signal_info:
            logger.warning("采样率未指定，使用默认值256Hz")
            signal_info["sampling_rate"] = 256.0

        sampling_rate = signal_info["sampling_rate"]
        if sampling_rate <= 0:
            logger.warning(f"采样率必须大于0，当前为{sampling_rate}，使用默认值256Hz")
            signal_info["sampling_rate"] = 256.0

        # 检查通道名称 - 如果没有则创建默认名称
        if "channel_names" not in signal_info:
            n_channels = data.shape[0]
            logger.warning(f"通道名称未指定，创建默认通道名称: EEG_0 到 EEG_{n_channels - 1}")
            signal_info["channel_names"] = [f"EEG_{i}" for i in range(n_channels)]
        else:
            n_channels = len(signal_info["channel_names"])
            if n_channels != data.shape[0]:
                logger.warning(f"通道数量({n_channels})与数据维度({data.shape[0]})不匹配，将更新通道名称")
                # 重新创建通道名称
                signal_info["channel_names"] = [f"EEG_{i}" for i in range(data.shape[0])]

        logger.info(f"EEG数据验证通过: {data.shape[0]}通道, {data.shape[1]}采样点, {sampling_rate}Hz")

    def _create_mne_raw(self, data_dict: Dict, modality: str) -> mne.io.RawArray:
        """
        从数据字典创建MNE Raw对象（增强版，支持单通道）
        """
        signal_info = data_dict["signal"][modality]
        data = signal_info["data"]
        sfreq = signal_info["sampling_rate"]
        ch_names = signal_info["channel_names"]

        # 确保数据是二维的
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
            logger.info(f"将一维数据reshape为: {data.shape}")

        # 确保通道名称数量匹配
        n_channels = data.shape[0]
        if len(ch_names) != n_channels:
            logger.warning(f"通道名称数量({len(ch_names)})与数据通道数({n_channels})不匹配，重新生成")
            ch_names = [f"EEG_{i}" for i in range(n_channels)]
            signal_info["channel_names"] = ch_names

        # 创建info对象
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=['eeg'] * n_channels  # 所有通道都设为eeg类型
        )

        # 创建Raw对象
        raw = mne.io.RawArray(data, info)

        # 设置单位（假设为uV）
        if "unit" in signal_info:
            unit = signal_info["unit"]
            if unit.lower() in ["uv", "μv"]:
                # MNE默认单位为V，需要转换为V
                raw._data = raw._data * 1e-6
                logger.info(f"数据单位从{unit}转换为伏特(V)")
        else:
            # 如果没有单位信息，根据数据范围推测
            data_range = np.ptp(data)
            if data_range > 100:  # 如果数据范围很大，可能是微伏
                logger.info(f"数据范围较大({data_range:.2e})，假设单位为微伏(uV)，转换为伏特(V)")
                raw._data = raw._data * 1e-6
            elif data_range < 1e-3:  # 如果数据范围很小，可能是伏特
                logger.info(f"数据范围较小({data_range:.2e})，假设单位为伏特(V)")
                # 不需要转换
            else:
                logger.info(f"数据范围为{data_range:.2e}，假设单位为伏特(V)")

        logger.info(f"创建MNE Raw对象: {n_channels}通道, {sfreq}Hz, {data.shape[1] / sfreq:.2f}秒")

        return raw

    def _set_montage(self, raw: mne.io.RawArray) -> mne.io.RawArray:
        """
        设置电极位置模板（改进版）
        """
        try:
            # 首先尝试标准模板
            raw.set_montage(self.config.montage, on_missing='warn')
            logger.info(f"已设置电极位置模板: {self.config.montage}")
            return raw
        except Exception as e:
            logger.warning(f"无法设置电极位置模板 {self.config.montage}: {str(e)}")

            # 检查是否是模拟数据（通道名类似EEG_0, EEG_1）
            ch_names = raw.ch_names
            if all(name.startswith('EEG_') for name in ch_names):
                logger.info("检测到模拟数据，创建虚拟电极布局")
                return self._create_simulated_montage(raw)
            else:
                logger.warning("跳过电极位置设置")
                return raw

    def _create_simulated_montage(self, raw: mne.io.RawArray) -> mne.io.RawArray:
        """
        为模拟数据创建虚拟电极布局
        """
        try:
            from mne.channels import make_dig_montage

            ch_names = raw.ch_names
            n_channels = len(ch_names)

            # 创建一个简单的圆形布局
            montage_positions = {}
            radius = 0.1  # 半径

            for i, ch_name in enumerate(ch_names):
                angle = 2 * np.pi * i / n_channels
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = 0.0
                montage_positions[ch_name] = [x, y, z]

            # 创建数字化蒙太奇
            montage = make_dig_montage(
                ch_pos=montage_positions,
                coord_frame='head'
            )
            raw.set_montage(montage)
            logger.info(f"已为{len(ch_names)}个通道创建虚拟电极布局")

        except Exception as e:
            logger.warning(f"创建虚拟电极布局失败: {str(e)}")

        return raw

    def _detect_bad_channels(self, raw: mne.io.RawArray) -> Tuple[mne.io.RawArray, List[str]]:
        """
        检测和标记坏道（适配单通道情况）
        """
        bad_channels = []
        n_channels = len(raw.ch_names)

        # 如果是单通道，直接返回（无法进行坏道检测）
        if n_channels == 1:
            logger.info("单通道数据，跳过坏道检测")
            raw.info['bads'] = []
            return raw, []

        # 方法1: 基于振幅异常检测
        if self.config.reject_by_amplitude:
            # 计算每个通道的振幅范围
            data = raw.get_data()
            channel_ranges = np.ptp(data, axis=1)  # 峰峰值

            # 检测异常通道
            median_range = np.median(channel_ranges)
            mad_range = np.median(np.abs(channel_ranges - median_range))

            # 避免MAD为0
            if mad_range == 0:
                mad_range = np.std(channel_ranges) or 1.0

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
        if bad_channels:
            max_bad = int(n_channels * self.config.max_bad_channels)

            # 如果是单通道且被标记为坏道，特殊处理
            if n_channels == 1 and bad_channels:
                logger.warning("单通道数据被标记为坏道，但无法舍弃唯一通道，将忽略坏道标记")
                bad_channels = []
                raw.info['bads'] = []
            elif max_bad <= 0:
                # 如果最大坏道数为0，则不允许标记任何坏道
                logger.info(
                    f"最大坏道比例设置为{self.config.max_bad_channels}，不允许标记坏道，将忽略检测到的{len(bad_channels)}个坏道")
                bad_channels = []
                raw.info['bads'] = []
            elif len(bad_channels) > max_bad:
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
        else:
            logger.info("未检测到坏道")

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
        插值坏道（适配单通道情况）
        """
        if not raw.info['bads']:
            return raw

        # 单通道无法插值
        if len(raw.ch_names) == 1:
            logger.warning("单通道数据无法进行坏道插值，将忽略坏道标记")
            raw.info['bads'] = []
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
        应用ICA伪迹去除（适配单通道情况）
        """
        ica_info = {
            "method": self.config.ica_method.value,
            "n_components": None,
            "components_removed": [],
            "n_components_removed": 0
        }

        # 单通道数据无法进行ICA
        if len(raw.ch_names) == 1:
            logger.warning("单通道数据无法进行ICA，跳过ICA处理")
            ica_info["error"] = "单通道数据无法进行ICA"
            return raw, ica_info

        try:
            # 检查数据大小，如果太大则提醒用户
            n_samples = raw.n_times
            n_channels = len(raw.ch_names)
            logger.info(f"开始ICA处理: {n_channels}通道, {n_samples}采样点, 预计需要一定时间...")

            # 数据太大时给出警告
            if n_samples > 1000000:  # 大于1M采样点
                logger.warning(f"数据量较大({n_samples / 1e6:.1f}M采样点)，ICA处理可能需要几分钟时间")

            # 确定ICA成分数量
            if self.config.ica_n_components is None:
                n_components = None
                logger.info(f"ICA将自动确定成分数量 (最大{n_channels})")
            elif isinstance(self.config.ica_n_components, float) and 0 < self.config.ica_n_components < 1:
                n_components = self.config.ica_n_components
                logger.info(f"ICA将使用 {n_components:.0%} 的成分")
            else:
                n_components = int(self.config.ica_n_components)
                # 确保成分数不超过通道数
                if n_components > n_channels:
                    logger.warning(f"指定的成分数({n_components})超过通道数({n_channels})，将使用{n_channels}")
                    n_components = n_channels
                logger.info(f"ICA将使用 {n_components} 个成分")

            ica_info["n_components"] = n_components

            # 创建ICA对象
            logger.info(f"创建ICA对象 (方法: {self.config.ica_method.value}, 最大迭代: {self.config.ica_max_iter})")

            ica = ICA(
                n_components=n_components,
                method=self.config.ica_method.value,
                max_iter=self.config.ica_max_iter,
                random_state=self.config.ica_random_state,
                fit_params=dict(extended=True) if self.config.ica_method == ICAMethod.EXTENDED_INFOMAX else None
            )

            # 拟合ICA前进行1Hz高通滤波
            logger.info("为ICA准备数据 (应用1Hz高通滤波)...")
            raw_for_ica = raw.copy()
            raw_for_ica.filter(l_freq=1.0, h_freq=None, method='fir', verbose=False)

            # 拟合ICA（添加进度信息）
            logger.info("开始拟合ICA，这可能需要几分钟...")

            # 使用一个简单的进度指示器
            import time
            start_time = time.time()

            # 拟合ICA
            ica.fit(raw_for_ica, verbose=False)

            elapsed_time = time.time() - start_time
            logger.info(f"ICA拟合完成，耗时: {elapsed_time:.1f}秒")

            # 自动标记成分
            if self.config.artifact_removal == ArtifactRemovalMethod.ICA_AUTO:
                logger.info("开始自动标记伪迹成分...")

                # 尝试使用ICLabel自动标记
                try:
                    from mne_icalabel import label_components

                    start_time = time.time()
                    ic_labels = label_components(raw_for_ica, ica, method='iclabel')
                    elapsed_time = time.time() - start_time

                    logger.info(f"ICLabel标记完成，耗时: {elapsed_time:.1f}秒")

                    exclude_idx = []
                    for i, label in enumerate(ic_labels['labels']):
                        if label not in ['brain', 'other']:
                            exclude_idx.append(i)

                        # 每10个成分输出一次进度
                        if (i + 1) % 10 == 0:
                            logger.debug(f"已处理 {i + 1}/{len(ic_labels['labels'])} 个成分")

                    ica_info["components_removed"] = exclude_idx
                    ica_info["n_components_removed"] = len(exclude_idx)
                    ica_info["component_labels"] = ic_labels['labels']
                    ica_info["component_probas"] = ic_labels['y_pred_proba']

                    if exclude_idx:
                        logger.info(f"ICA自动标记排除 {len(exclude_idx)}/{len(ic_labels['labels'])} 个成分")
                        logger.info(f"排除的成分类型: {[ic_labels['labels'][i] for i in exclude_idx[:10]]}")
                        if len(exclude_idx) > 10:
                            logger.info(f"...等 {len(exclude_idx)} 个成分")

                        ica.apply(raw, exclude=exclude_idx)
                        logger.info("已应用ICA排除")
                    else:
                        logger.info("ICA未发现需要排除的成分")

                except Exception as e:
                    logger.warning(f"ICA自动标记失败: {str(e)}，使用基于峰度的检测")
                    # 使用基于峰度的检测
                    sources = ica.get_sources(raw_for_ica).get_data()
                    kurtosis_values = []

                    logger.info("计算成分峰度...")
                    for i in range(sources.shape[0]):
                        source = sources[i, :]
                        kurt = np.mean((source - np.mean(source)) ** 4) / (np.std(source) ** 4) - 3
                        kurtosis_values.append(abs(kurt))

                        # 每10个成分输出一次进度
                        if (i + 1) % 10 == 0:
                            logger.debug(f"已计算 {i + 1}/{sources.shape[0]} 个成分的峰度")

                    # 检测异常峰度
                    median_kurt = np.median(kurtosis_values)
                    mad_kurt = np.median(np.abs(kurtosis_values - median_kurt))
                    threshold = median_kurt + 3 * mad_kurt * 1.4826

                    exclude_idx = [i for i, kurt in enumerate(kurtosis_values) if kurt > threshold]

                    ica_info["components_removed"] = exclude_idx
                    ica_info["n_components_removed"] = len(exclude_idx)

                    if exclude_idx:
                        ica.apply(raw, exclude=exclude_idx)
                        logger.info(f"基于峰度排除 {len(exclude_idx)}/{len(kurtosis_values)} 个成分")
                    else:
                        logger.info("基于峰度未发现需要排除的成分")

            elif self.config.artifact_removal == ArtifactRemovalMethod.ICA_MANUAL:
                logger.info("ICA手动标记模式，需要用户交互")
                ica_info["ica_object"] = ica

        except Exception as e:
            logger.error(f"ICA处理失败: {str(e)}")
            # 如果ICA失败，记录错误但继续处理
            ica_info["error"] = str(e)
            import traceback
            logger.debug(traceback.format_exc())

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
        创建分段数据（修复版）
        修复了分段创建为0的问题

        Args:
            raw: MNE Raw对象
            event_info: 事件信息字典

        Returns:
            分段信息字典
        """
        epochs_info = {
            "tmin": self.config.epoch_tmin,
            "tmax": self.config.epoch_tmax,
            "baseline": self.config.baseline_correction,
            "n_epochs": 0
        }

        # 检查是否有足够的事件信息
        if "event_time" not in event_info or not event_info["event_time"]:
            logger.warning("没有事件信息，无法进行分段")
            return epochs_info

        # 获取事件信息
        event_times = event_info["event_time"]
        sfreq = raw.info['sfreq']
        n_samples = raw.n_times
        data_duration = n_samples / sfreq

        logger.info(f"数据长度: {data_duration:.2f}秒 ({n_samples}个采样点)")
        logger.info(f"事件数量: {len(event_times)}个")
        logger.info(f"分段时间窗口: [{self.config.epoch_tmin}, {self.config.epoch_tmax}]秒")

        # 筛选有效事件（确保分段在数据范围内）
        valid_events = []
        valid_event_times = []

        for i, event_time in enumerate(event_times):
            epoch_start = event_time + self.config.epoch_tmin
            epoch_end = event_time + self.config.epoch_tmax

            # 检查分段是否在数据范围内（留出0.1秒的缓冲区）
            if epoch_start >= 0.1 and epoch_end <= data_duration - 0.1:
                valid_events.append(i)
                valid_event_times.append(event_time)
            else:
                logger.warning(f"事件{i + 1}在{event_time:.2f}秒的分段越界: "
                               f"[{epoch_start:.2f}, {epoch_end:.2f}]秒，数据范围: [0, {data_duration:.2f}]秒")

        if not valid_events:
            logger.error("没有有效的事件可以创建分段")
            return epochs_info

        # 如果有事件被排除，调整事件数组
        if len(valid_events) < len(event_times):
            logger.warning(f"只有{len(valid_events)}/{len(event_times)}个事件可以创建分段")

            # 更新事件信息
            if "event_id" in event_info:
                event_ids = [event_info["event_id"][i] for i in valid_events]
            else:
                event_ids = [1] * len(valid_events)
        else:
            if "event_id" in event_info:
                event_ids = event_info["event_id"]
            else:
                event_ids = [1] * len(event_times)

        # 创建事件样本数组
        event_samples = [int(t * sfreq) for t in valid_event_times]
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

            logger.info(f"创建了{len(epochs)}个分段")

            # 应用振幅拒绝
            if self.config.reject_by_amplitude:
                n_before = len(epochs)
                epochs.drop_bad(reject=dict(eeg=self.config.rejection_threshold * 1e-6))
                n_after = len(epochs)

                if n_before > n_after:
                    logger.info(f"振幅拒绝: 从{n_before}个分段中移除了{n_before - n_after}个分段")
                else:
                    logger.info(f"振幅拒绝: 没有分段被移除")

            # 检查分段数量
            if len(epochs) > 0:
                # 提取分段数据
                epochs_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

                epochs_info["data"] = epochs_data
                epochs_info["events"] = events
                epochs_info["event_ids"] = event_ids
                epochs_info["n_epochs"] = len(epochs_data)
                epochs_info["epoch_times"] = epochs.times
                epochs_info["valid_event_indices"] = valid_events

                logger.info(f"成功创建{len(epochs_data)}个分段，每个分段{epochs_data.shape[2]}个时间点")
            else:
                logger.warning("没有可用的分段（全部被拒绝）")
                epochs_info["n_epochs"] = 0

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

            # 伪迹拒绝配置
            reject_by_amplitude=True,
            rejection_threshold=150e-6,  # 增加拒绝阈值

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

            # 伪迹拒绝配置
            reject_by_amplitude=True,
            rejection_threshold=200e-6,  # P300信号较小，需要更宽松的阈值

            # 其他配置
            montage="standard_1020",
            line_freq=50.0,
            downsample_to=250.0
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

            # 伪迹拒绝配置
            reject_by_amplitude=True,
            rejection_threshold=200e-6,  # SSVEP信号较强，可以使用较高阈值

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

            # 伪迹拒绝配置（连续数据不需要分段拒绝）
            reject_by_amplitude=False,

            # 其他配置
            montage="standard_1020",
            line_freq=50.0,

            # 坏道处理更严格
            bad_channel_threshold=2.5,
            max_bad_channels=0.05  # 最多5%坏道
        )
