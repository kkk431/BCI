# -*- coding: utf-8 -*-
"""
多模态信号预处理主模块
统一调度和管理EEG、fNIRS、ECG、EMG等多种生理信号的预处理
支持四层数据格式结构，自动识别和并行处理多模态信号
"""
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
import warnings
import copy

# 导入各模态预处理模块
try:
    from core.processing.preprocessing.eeg_preprocessing import EEGPreprocessor, EEGPreprocessingConfig, EEGConfigFactory
    from core.processing.preprocessing.fnirs_preprocessing import fNIRSPreprocessor, fNIRSConfig
    from core.processing.preprocessing.ecg_preprocessing import ECGPreprocessor, ECGConfig
    from core.processing.preprocessing.emg_preprocessing import EMGPreprocessor, EMGPreprocessingConfig, EMGConfigFactory
    from core.processing.preprocessing.preprocessing import GeneralPreprocessor, PreprocessingConfig

    HAS_MODULES = True
except ImportError as e:
    HAS_MODULES = False
    print(f"某些预处理模块导入失败: {str(e)}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 禁用特定警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# ====================== 枚举和配置类 ======================

class ProcessingMode(Enum):
    """处理模式枚举"""
    SEQUENTIAL = "sequential"  # 顺序处理
    PARALLEL = "parallel"  # 并行处理
    SELECTIVE = "selective"  # 选择处理


class TimeSyncMethod(Enum):
    """时间同步方法枚举"""
    RESAMPLE = "resample"  # 重采样对齐
    INTERPOLATE = "interpolate"  # 插值对齐
    EVENT_ALIGN = "event_align"  # 事件对齐
    NONE = "none"  # 不同步


@dataclass
class MultiModalConfig:
    """
    多模态预处理配置类
    """
    # ========== 通用处理配置 ==========
    processing_mode: ProcessingMode = ProcessingMode.SEQUENTIAL
    time_sync_method: TimeSyncMethod = TimeSyncMethod.NONE
    reference_sampling_rate: Optional[float] = None  # 参考采样率（用于同步）
    max_workers: int = 4  # 并行处理最大线程数

    # ========== 各模态配置 ==========
    eeg_config: Optional[Any] = None  # EEG预处理配置
    fnirs_config: Optional[Any] = None  # fNIRS预处理配置
    ecg_config: Optional[Any] = None  # ECG预处理配置
    emg_config: Optional[Any] = None  # EMG预处理配置
    general_config: Optional[Any] = None  # 通用预处理配置

    # ========== 模态选择配置 ==========
    enabled_modalities: List[str] = field(default_factory=lambda: ["EEG", "EMG", "ECG", "fNIRS"])
    process_all_modalities: bool = True  # 是否处理所有模态

    # ========== 输出配置 ==========
    save_intermediate_results: bool = False
    save_processing_log: bool = True
    visualize_results: bool = False
    output_format: str = "dict"  # dict, numpy, pickle

    # ========== 质量控制配置 ==========
    quality_check_enabled: bool = True
    min_signal_quality: float = 0.5  # 最低信号质量阈值（0-1）
    auto_fix_issues: bool = True  # 自动修复问题


@dataclass
class ProcessingResult:
    """
    处理结果数据结构
    """
    success: bool = False
    processed_data: Optional[Dict[str, Any]] = None
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0


# ====================== 多模态预处理器主类 ======================

class MultiModalPreprocessor:
    """
    多模态信号预处理器
    统一调度和管理多种生理信号的预处理流程
    """

    def __init__(self, config: Optional[MultiModalConfig] = None):
        """
        初始化多模态预处理器

        Args:
            config: 多模态预处理配置，None则使用默认配置
        """
        self.config = config if config is not None else MultiModalConfig()
        self.modality_processors = {}
        self.modality_configs = {}
        self.processing_history = []
        self._initialize_processors()

    def _initialize_processors(self):
        """初始化各模态预处理器"""
        if not HAS_MODULES:
            logger.error("未能导入必要的预处理模块")
            return

        # 初始化各模态预处理器
        try:
            # EEG预处理器
            if self.config.eeg_config is not None:
                self.modality_processors["EEG"] = EEGPreprocessor(self.config.eeg_config)
                self.modality_configs["EEG"] = self.config.eeg_config
            else:
                # 使用默认配置
                self.modality_processors["EEG"] = EEGPreprocessor()
                self.modality_configs["EEG"] = EEGPreprocessingConfig()

            # fNIRS预处理器
            if self.config.fnirs_config is not None:
                self.modality_processors["fNIRS"] = fNIRSPreprocessor(self.config.fnirs_config)
                self.modality_configs["fNIRS"] = self.config.fnirs_config
            else:
                self.modality_processors["fNIRS"] = fNIRSPreprocessor()
                self.modality_configs["fNIRS"] = fNIRSConfig()

            # ECG预处理器
            if self.config.ecg_config is not None:
                self.modality_processors["ECG"] = ECGPreprocessor(self.config.ecg_config)
                self.modality_configs["ECG"] = self.config.ecg_config
            else:
                self.modality_processors["ECG"] = ECGPreprocessor()
                self.modality_configs["ECG"] = ECGConfig()

            # EMG预处理器
            if self.config.emg_config is not None:
                self.modality_processors["EMG"] = EMGPreprocessor(self.config.emg_config)
                self.modality_configs["EMG"] = self.config.emg_config
            else:
                self.modality_processors["EMG"] = EMGPreprocessor()
                self.modality_configs["EMG"] = EMGPreprocessingConfig()

            # 通用预处理器
            if self.config.general_config is not None:
                self.modality_processors["GENERAL"] = GeneralPreprocessor(self.config.general_config)
                self.modality_configs["GENERAL"] = self.config.general_config
            else:
                self.modality_processors["GENERAL"] = GeneralPreprocessor()
                self.modality_configs["GENERAL"] = PreprocessingConfig()

            logger.info("成功初始化所有模态预处理器")

        except Exception as e:
            logger.error(f"初始化预处理器失败: {str(e)}")

    def process(self, data_dict: Dict[str, Any]) -> ProcessingResult:
        """
        主处理函数：执行多模态信号预处理

        Args:
            data_dict: 四层结构的数据字典

        Returns:
            处理结果对象
        """
        import time
        start_time = time.time()
        result = ProcessingResult()

        try:
            # 1. 验证输入数据格式
            self._validate_data_dict(data_dict)

            # 2. 备份原始数据
            original_data = copy.deepcopy(data_dict)

            # 3. 检测数据中的模态
            detected_modalities = self._detect_modalities(data_dict)
            logger.info(f"检测到模态: {detected_modalities}")

            # 4. 选择要处理的模态
            modalities_to_process = self._select_modalities_to_process(detected_modalities)
            if not modalities_to_process:
                raise ValueError("没有可处理的信号模态")

            logger.info(f"将处理以下模态: {modalities_to_process}")

            # 5. 时间同步（如果需要）
            if self.config.time_sync_method != TimeSyncMethod.NONE:
                data_dict = self._synchronize_modalities(data_dict, modalities_to_process)

            # 6. 执行预处理
            processed_data = self._execute_preprocessing(data_dict, modalities_to_process)

            # 7. 质量检查
            if self.config.quality_check_enabled:
                quality_report = self._check_quality(processed_data, modalities_to_process)
                processed_data["processed"]["quality_report"] = quality_report

                # 自动修复问题
                if self.config.auto_fix_issues and quality_report.get("has_issues", False):
                    processed_data = self._auto_fix_issues(processed_data, quality_report)

            # 8. 记录处理历史
            processing_stats = self._collect_processing_stats(processed_data)
            self._update_processing_history(processing_stats)

            # 9. 生成处理结果
            result.success = True
            result.processed_data = processed_data
            result.processing_stats = processing_stats
            result.processing_time = time.time() - start_time

            logger.info(f"多模态预处理完成，耗时: {result.processing_time:.2f}秒")

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"处理失败: {str(e)}")

        return result

    def _validate_data_dict(self, data_dict: Dict[str, Any]):
        """
        验证四层数据字典格式
        """
        required_layers = ["meta", "signal"]
        for layer in required_layers:
            if layer not in data_dict:
                raise ValueError(f"数据字典必须包含'{layer}'层")

        # 验证meta层
        meta = data_dict["meta"]
        required_meta_fields = ["subject_id", "task", "modality", "sampling_rate"]
        for field in required_meta_fields:
            if field not in meta:
                raise ValueError(f"meta层必须包含'{field}'字段")

        # 验证signal层
        signal = data_dict["signal"]
        if not isinstance(signal, dict) or len(signal) == 0:
            raise ValueError("signal层必须是非空字典")

        for modality, signal_info in signal.items():
            required_signal_fields = ["data", "sampling_rate", "channel_names"]
            for field in required_signal_fields:
                if field not in signal_info:
                    raise ValueError(f"{modality}信号必须包含'{field}'字段")

            # 验证数据形状
            data = signal_info["data"]
            if not isinstance(data, np.ndarray):
                raise ValueError(f"{modality}数据必须是numpy数组")

            channel_names = signal_info["channel_names"]
            if len(data.shape) != 2:
                raise ValueError(f"{modality}数据必须是2维数组 (channels × samples)")

            if data.shape[0] != len(channel_names):
                raise ValueError(f"{modality}通道数量与数据维度不匹配")

        logger.info("数据格式验证通过")

    def _detect_modalities(self, data_dict: Dict[str, Any]) -> List[str]:
        """
        检测数据中存在的信号模态
        """
        detected_modalities = []

        # 从signal层检测
        signal_modalities = list(data_dict["signal"].keys())

        # 标准化模态名称
        modality_mapping = {
            "EEG": ["EEG", "eeg", "Electroencephalography"],
            "fNIRS": ["fNIRS", "fnirs", "NIRS", "nir"],
            "ECG": ["ECG", "ecg", "Electrocardiography"],
            "EMG": ["EMG", "emg", "Electromyography"],
            "EOG": ["EOG", "eog", "Electrooculography"],
            "GSR": ["GSR", "gsr", "EDA", "eda"],
            "RESP": ["RESP", "resp", "Respiration"],
        }

        for detected in signal_modalities:
            detected_upper = detected.upper()
            for standard_name, variants in modality_mapping.items():
                if detected_upper in variants or detected_upper == standard_name:
                    if standard_name not in detected_modalities:
                        detected_modalities.append(standard_name)
                    break

        # 如果没有匹配到标准名称，直接添加原始名称
        for detected in signal_modalities:
            if detected not in detected_modalities:
                detected_modalities.append(detected)

        return detected_modalities

    def _select_modalities_to_process(self, detected_modalities: List[str]) -> List[str]:
        """
        选择要处理的模态
        """
        if self.config.process_all_modalities:
            return detected_modalities

        # 只处理启用列表中的模态
        modalities_to_process = []
        for modality in detected_modalities:
            if modality in self.config.enabled_modalities:
                modalities_to_process.append(modality)
            elif modality.upper() in [m.upper() for m in self.config.enabled_modalities]:
                # 大小写不敏感的匹配
                modalities_to_process.append(modality)

        return modalities_to_process

    def _synchronize_modalities(self, data_dict: Dict[str, Any],
                                modalities: List[str]) -> Dict[str, Any]:
        """
        同步多模态信号的时间
        """
        if len(modalities) <= 1:
            logger.info("只有一个模态，无需同步")
            return data_dict

        logger.info(f"开始时间同步，方法: {self.config.time_sync_method.value}")

        if self.config.time_sync_method == TimeSyncMethod.RESAMPLE:
            return self._synchronize_by_resampling(data_dict, modalities)
        elif self.config.time_sync_method == TimeSyncMethod.INTERPOLATE:
            return self._synchronize_by_interpolation(data_dict, modalities)
        elif self.config.time_sync_method == TimeSyncMethod.EVENT_ALIGN:
            return self._synchronize_by_event_alignment(data_dict, modalities)
        else:
            return data_dict

    def _synchronize_by_resampling(self, data_dict: Dict[str, Any],
                                   modalities: List[str]) -> Dict[str, Any]:
        """
        通过重采样进行时间同步
        """
        if self.config.reference_sampling_rate is None:
            # 使用最高采样率作为参考
            max_fs = 0
            for modality in modalities:
                fs = data_dict["signal"][modality]["sampling_rate"]
                if fs > max_fs:
                    max_fs = fs
            target_fs = max_fs
        else:
            target_fs = self.config.reference_sampling_rate

        # 重要：记录重采样后的奈奎斯特频率
        nyquist_freq = target_fs / 2
        logger.info(f"重采样后奈奎斯特频率: {nyquist_freq} Hz")

        for modality in modalities:
            signal_info = data_dict["signal"][modality]
            original_fs = signal_info["sampling_rate"]

            if original_fs != target_fs:
                # 对于EMG信号，需要特别处理滤波参数
                if modality == "EMG":
                    # 在重采样前检查并调整滤波参数
                    self._adjust_emg_filter_for_resampling(signal_info, original_fs, target_fs)
                
                data = signal_info["data"]

                # 计算新的样本数
                n_channels, n_samples = data.shape
                new_n_samples = int(n_samples * target_fs / original_fs)

                # 重采样
                from scipy.signal import resample_poly
                resampled_data = np.zeros((n_channels, new_n_samples))

                for ch in range(n_channels):
                    resampled_data[ch, :] = resample_poly(
                        data[ch, :],
                        int(target_fs),
                        int(original_fs)
                    )

                # 更新数据字典
                signal_info["data"] = resampled_data
                signal_info["sampling_rate"] = target_fs
                signal_info["original_sampling_rate"] = original_fs  # 保存原始采样率

                logger.info(f"{modality}: {original_fs} Hz -> {target_fs} Hz")

        return data_dict

    def _synchronize_by_interpolation(self, data_dict: Dict[str, Any],
                                      modalities: List[str]) -> Dict[str, Any]:
        """
        通过插值进行时间同步
        """
        # 找出最长的信号作为时间基准
        max_duration = 0
        reference_modality = None

        for modality in modalities:
            signal_info = data_dict["signal"][modality]
            n_samples = signal_info["data"].shape[1]
            fs = signal_info["sampling_rate"]
            duration = n_samples / fs

            if duration > max_duration:
                max_duration = duration
                reference_modality = modality

        if reference_modality is None:
            return data_dict

        reference_fs = data_dict["signal"][reference_modality]["sampling_rate"]
        reference_n_samples = data_dict["signal"][reference_modality]["data"].shape[1]

        logger.info(f"使用{reference_modality}作为时间参考，采样率: {reference_fs} Hz")

        # 创建统一的时间轴
        time_axis = np.arange(reference_n_samples) / reference_fs

        for modality in modalities:
            if modality == reference_modality:
                continue

            signal_info = data_dict["signal"][modality]
            original_data = signal_info["data"]
            original_fs = signal_info["sampling_rate"]
            original_n_samples = original_data.shape[1]

            # 原始时间轴
            original_time = np.arange(original_n_samples) / original_fs

            # 插值到参考时间轴
            from scipy.interpolate import interp1d
            interpolated_data = np.zeros((original_data.shape[0], reference_n_samples))

            for ch in range(original_data.shape[0]):
                interp_func = interp1d(original_time, original_data[ch, :],
                                       kind='cubic', fill_value="extrapolate")
                interpolated_data[ch, :] = interp_func(time_axis)

            # 更新数据字典
            signal_info["data"] = interpolated_data
            signal_info["sampling_rate"] = reference_fs
            signal_info["original_sampling_rate"] = original_fs
            signal_info["interpolated"] = True

            logger.info(f"{modality}: 插值到{reference_fs} Hz")

        return data_dict

    def _synchronize_by_event_alignment(self, data_dict: Dict[str, Any],
                                        modalities: List[str]) -> Dict[str, Any]:
        """
        通过事件对齐进行时间同步
        """
        logger.warning("事件对齐同步是简化实现，实际应用需要完整事件对齐逻辑")
        if "event" not in data_dict:
            logger.warning("没有事件信息，跳过事件对齐")
            return data_dict
        return data_dict

    def _execute_preprocessing(self, data_dict: Dict[str, Any],
                               modalities: List[str]) -> Dict[str, Any]:
        """
        执行各模态的预处理
        """
        processed_data = copy.deepcopy(data_dict)

        # 确保processed层存在
        if "processed" not in processed_data:
            processed_data["processed"] = {}

        if "multimodal_preprocessing" not in processed_data["processed"]:
            processed_data["processed"]["multimodal_preprocessing"] = {
                "modalities_processed": [],
                "processing_timeline": [],
                "config": self.config.__dict__
            }

        # 根据处理模式选择执行方式
        if self.config.processing_mode == ProcessingMode.PARALLEL:
            return self._execute_parallel_preprocessing(processed_data, modalities)
        else:
            return self._execute_sequential_preprocessing(processed_data, modalities)

    def _execute_sequential_preprocessing(self, data_dict: Dict[str, Any],
                                          modalities: List[str]) -> Dict[str, Any]:
        """
        顺序执行预处理
        """
        import time

        for modality in modalities:
            logger.info(f"开始处理 {modality} 信号")

            start_time = time.time()

            try:
                # 检查是否有专门的预处理器
                if modality in self.modality_processors:
                    processor = self.modality_processors[modality]

                    # 调用对应的处理函数
                    if modality == "EEG":
                        data_dict = processor.process(data_dict, modality="EEG")
                    elif modality == "fNIRS":
                        data_dict = processor.process_fNIRS(data_dict, modality="fnirs")
                    elif modality == "ECG":
                        data_dict = processor.process_ECG(data_dict, modality="ECG")
                    elif modality == "EMG":
                        data_dict = processor.process(data_dict, modality="EMG")
                    else:
                        # 使用通用预处理器
                        data_dict = self.modality_processors["GENERAL"].process(
                            data_dict, modality=modality
                        )

                    # 记录处理信息
                    processing_time = time.time() - start_time
                    data_dict["processed"]["multimodal_preprocessing"]["modalities_processed"].append(modality)
                    data_dict["processed"]["multimodal_preprocessing"]["processing_timeline"].append({
                        "modality": modality,
                        "processor": processor.__class__.__name__,
                        "processing_time": processing_time,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })

                    logger.info(f"{modality} 处理完成，耗时: {processing_time:.2f}秒")

                else:
                    logger.warning(f"没有找到 {modality} 的预处理器，跳过")

            except Exception as e:
                logger.error(f"{modality} 处理失败: {str(e)}")
                if self.config.auto_fix_issues:
                    logger.info(f"尝试使用通用处理器处理 {modality}")
                    try:
                        data_dict = self.modality_processors["GENERAL"].process(
                            data_dict, modality=modality
                        )
                        logger.info(f"通用处理器成功处理 {modality}")
                    except Exception as e2:
                        logger.error(f"通用处理器也失败: {str(e2)}")

        return data_dict

    def _execute_parallel_preprocessing(self, data_dict: Dict[str, Any],
                                        modalities: List[str]) -> Dict[str, Any]:
        """
        并行执行预处理
        """
        import time
        from concurrent.futures import ThreadPoolExecutor

        logger.info(f"开始并行处理 {len(modalities)} 个模态")

        # 创建数据副本用于并行处理
        data_copies = {modality: copy.deepcopy(data_dict) for modality in modalities}
        results = {}

        def process_single_modality(modality, data_copy):
            """处理单个模态的辅助函数"""
            start_time = time.time()

            try:
                if modality in self.modality_processors:
                    processor = self.modality_processors[modality]

                    if modality == "EEG":
                        result = processor.process(data_copy, modality="EEG")
                    elif modality == "fNIRS":
                        result = processor.process_fNIRS(data_copy, modality="fnirs")
                    elif modality == "ECG":
                        result = processor.process_ECG(data_copy, modality="ECG")
                    elif modality == "EMG":
                        result = processor.process(data_copy, modality="EMG")
                    else:
                        result = self.modality_processors["GENERAL"].process(
                            data_copy, modality=modality
                        )

                    processing_time = time.time() - start_time

                    return {
                        "modality": modality,
                        "data": result,
                        "success": True,
                        "processing_time": processing_time,
                        "processor": processor.__class__.__name__
                    }
                else:
                    return {
                        "modality": modality,
                        "data": data_copy,
                        "success": False,
                        "error": f"没有找到 {modality} 的预处理器",
                        "processing_time": time.time() - start_time
                    }

            except Exception as e:
                return {
                    "modality": modality,
                    "data": data_copy,
                    "success": False,
                    "error": str(e),
                    "processing_time": time.time() - start_time
                }

        # 并行执行
        with ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(modalities))) as executor:
            future_to_modality = {
                executor.submit(process_single_modality, modality, data_copies[modality]): modality
                for modality in modalities
            }

            for future in concurrent.futures.as_completed(future_to_modality):
                modality = future_to_modality[future]
                try:
                    result = future.result()
                    results[modality] = result

                    if result["success"]:
                        logger.info(f"{modality} 处理完成，耗时: {result['processing_time']:.2f}秒")
                    else:
                        logger.warning(f"{modality} 处理失败: {result.get('error', '未知错误')}")

                except Exception as e:
                    logger.error(f"{modality} 处理异常: {str(e)}")
                    results[modality] = {
                        "modality": modality,
                        "success": False,
                        "error": str(e)
                    }

        # 合并结果
        merged_data = copy.deepcopy(data_dict)

        if "processed" not in merged_data:
            merged_data["processed"] = {}

        if "multimodal_preprocessing" not in merged_data["processed"]:
            merged_data["processed"]["multimodal_preprocessing"] = {
                "modalities_processed": [],
                "processing_timeline": [],
                "parallel_results": {}
            }

        # 更新signal层和处理历史
        for modality, result in results.items():
            if result["success"]:
                # 更新signal层
                if modality in result["data"]["signal"]:
                    merged_data["signal"][modality] = result["data"]["signal"][modality]

                # 更新processed层
                if "processed" in result["data"]:
                    for key, value in result["data"]["processed"].items():
                        if key not in merged_data["processed"]:
                            merged_data["processed"][key] = {}
                        merged_data["processed"][key][modality] = value

                # 记录处理信息
                merged_data["processed"]["multimodal_preprocessing"]["modalities_processed"].append(modality)
                merged_data["processed"]["multimodal_preprocessing"]["processing_timeline"].append({
                    "modality": modality,
                    "processor": result.get("processor", "Unknown"),
                    "processing_time": result["processing_time"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "parallel": True
                })

        return merged_data

    def _check_quality(self, data_dict: Dict[str, Any],
                       modalities: List[str]) -> Dict[str, Any]:
        """
        检查处理后的信号质量
        """
        quality_report = {
            "overall_quality": 1.0,
            "modality_quality": {},
            "issues": [],
            "has_issues": False,
            "timestamp": np.datetime64('now').astype(str)
        }

        for modality in modalities:
            if modality not in data_dict["signal"]:
                quality_report["issues"].append(f"{modality}: 信号缺失")
                quality_report["has_issues"] = True
                continue

            signal_info = data_dict["signal"][modality]
            data = signal_info["data"]

            # 基本质量检查
            modality_report = {
                "n_channels": data.shape[0],
                "n_samples": data.shape[1],
                "sampling_rate": signal_info.get("sampling_rate", 0),
                "quality_score": 1.0,
                "checks_passed": 0,
                "checks_total": 0,
                "issues": []
            }

            # 检查1: 数据是否全为0
            if np.all(data == 0):
                modality_report["issues"].append("数据全为0")
                modality_report["quality_score"] *= 0.1

            # 检查2: 数据是否包含NaN或Inf
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                modality_report["issues"].append("数据包含NaN或Inf")
                modality_report["quality_score"] *= 0.5

            # 检查3: 数据范围是否合理
            data_range = np.ptp(data, axis=1)
            median_range = np.median(data_range)

            # 基于模态的经验范围阈值
            range_thresholds = {
                "EEG": (1e-6, 0.1),  # V
                "EMG": (0.001, 0.5),  # V
                "ECG": (0.5e-3, 5e-3),  # V
                "fNIRS": (0.01, 10.0),  # 光学密度或浓度
            }

            if modality in range_thresholds:
                min_th, max_th = range_thresholds[modality]
                if median_range < min_th or median_range > max_th:
                    modality_report["issues"].append(f"数据范围异常: {median_range:.2e}")
                    modality_report["quality_score"] *= 0.7

            # 检查4: 信噪比估计
            if data.shape[1] > 100:
                # 简单信噪比估计：低频功率 vs 高频功率
                from scipy.signal import welch
                _, psd = welch(data[0, :], fs=signal_info.get("sampling_rate", 1000))
                low_freq_power = np.mean(psd[:len(psd) // 4])
                high_freq_power = np.mean(psd[3 * len(psd) // 4:])

                if high_freq_power > 0:
                    snr_estimate = low_freq_power / high_freq_power
                    if snr_estimate < 1.0:
                        modality_report["issues"].append(f"信噪比较低: {snr_estimate:.2f}")
                        modality_report["quality_score"] *= 0.8

            # 更新质量报告
            if modality_report["issues"]:
                quality_report["has_issues"] = True

            modality_report["checks_total"] = 4
            modality_report["checks_passed"] = 4 - len(modality_report["issues"])

            quality_report["modality_quality"][modality] = modality_report

        # 计算总体质量
        if quality_report["modality_quality"]:
            quality_scores = [r["quality_score"] for r in quality_report["modality_quality"].values()]
            quality_report["overall_quality"] = np.mean(quality_scores)

        logger.info(f"质量检查完成，总体质量: {quality_report['overall_quality']:.2f}")

        return quality_report

    def _auto_fix_issues(self, data_dict: Dict[str, Any],
                         quality_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        自动修复质量问题
        """
        logger.info("开始自动修复质量问题")

        fixed_data = copy.deepcopy(data_dict)

        for modality, modality_report in quality_report.get("modality_quality", {}).items():
            if not modality_report.get("issues"):
                continue

            if modality not in fixed_data["signal"]:
                continue

            signal_info = fixed_data["signal"][modality]
            data = signal_info["data"]

            for issue in modality_report["issues"]:
                if "数据包含NaN或Inf" in issue:
                    # 修复NaN和Inf
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                    signal_info["data"] = data
                    logger.info(f"{modality}: 修复了NaN和Inf值")

                elif "数据全为0" in issue:
                    # 如果数据全为0，尝试从原始数据恢复
                    if "original_data" in signal_info:
                        signal_info["data"] = signal_info["original_data"].copy()
                        logger.info(f"{modality}: 从原始数据恢复")

                elif "数据范围异常" in issue:
                    # 重新标准化数据
                    from scipy import stats
                    for ch in range(data.shape[0]):
                        if np.std(data[ch, :]) > 0:
                            data[ch, :] = stats.zscore(data[ch, :])
                        else:
                            data[ch, :] = 0
                    logger.info(f"{modality}: 重新标准化数据")

        return fixed_data

    def _collect_processing_stats(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        收集处理统计信息
        """
        stats = {
            "modalities_processed": [],
            "processing_times": {},
            "data_shapes": {},
            "quality_scores": {},
            "timestamp": np.datetime64('now').astype(str)
        }

        if "processed" in data_dict and "multimodal_preprocessing" in data_dict["processed"]:
            mp_info = data_dict["processed"]["multimodal_preprocessing"]

            if "modalities_processed" in mp_info:
                stats["modalities_processed"] = mp_info["modalities_processed"]

            if "processing_timeline" in mp_info:
                for timeline in mp_info["processing_timeline"]:
                    modality = timeline.get("modality", "unknown")
                    processing_time = timeline.get("processing_time", 0)
                    stats["processing_times"][modality] = processing_time

        # 收集各模态的数据形状
        for modality, signal_info in data_dict["signal"].items():
            if "data" in signal_info:
                data = signal_info["data"]
                stats["data_shapes"][modality] = {
                    "n_channels": data.shape[0],
                    "n_samples": data.shape[1],
                    "sampling_rate": signal_info.get("sampling_rate", 0)
                }

        # 收集质量分数
        if "processed" in data_dict and "quality_report" in data_dict["processed"]:
            quality_report = data_dict["processed"]["quality_report"]
            stats["overall_quality"] = quality_report.get("overall_quality", 0)

            if "modality_quality" in quality_report:
                for modality, modality_report in quality_report["modality_quality"].items():
                    stats["quality_scores"][modality] = modality_report.get("quality_score", 0)

        return stats

    def _update_processing_history(self, stats: Dict[str, Any]):
        """
        更新处理历史
        """
        self.processing_history.append(stats)

        # 限制历史记录数量
        if len(self.processing_history) > 100:
            self.processing_history = self.processing_history[-100:]

    def get_processing_summary(self) -> Dict[str, Any]:
        """
        获取处理摘要
        """
        summary = {
            "total_processes": len(self.processing_history),
            "recent_processes": self.processing_history[-5:] if self.processing_history else [],
            "config": self.config.__dict__,
            "available_processors": list(self.modality_processors.keys()),
            "processor_status": "OK" if HAS_MODULES else "Missing modules"
        }

        return summary

    def save_config(self, filepath: str):
        """
        保存配置到文件
        """
        import json

        config_dict = self.config.__dict__

        # 处理枚举类型
        for key, value in config_dict.items():
            if isinstance(value, Enum):
                config_dict[key] = value.value

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"配置已保存到: {filepath}")

    def load_config(self, filepath: str):
        """
        从文件加载配置
        """
        import json

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            # 重新创建配置对象
            self.config = MultiModalConfig(**config_dict)

            # 重新初始化处理器
            self._initialize_processors()

            logger.info(f"配置已从 {filepath} 加载")

        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}")

    def _adjust_emg_filter_for_resampling(self, signal_info: Dict, original_fs: float, target_fs: float):
        """
        调整EMG滤波参数以适应重采样
        """
        # 计算奈奎斯特频率
        nyquist_original = original_fs / 2
        nyquist_target = target_fs / 2

        # 获取EMG配置
        if hasattr(self.config, 'emg_config') and self.config.emg_config:
            emg_config = self.config.emg_config

            # 检查高截止频率是否超过目标奈奎斯特频率
            if hasattr(emg_config, 'emg_bandpass_high'):
                if emg_config.emg_bandpass_high > nyquist_target:
                    old_value = emg_config.emg_bandpass_high
                    # 调整为安全值（80%奈奎斯特频率）
                    safe_value = nyquist_target * 0.8
                    emg_config.emg_bandpass_high = safe_value
                    logger.warning(
                        f"EMG高截止频率从{old_value}Hz调整到{safe_value}Hz "
                        f"(奈奎斯特频率: {nyquist_target}Hz)"
                    )

            # 检查低截止频率
            if hasattr(emg_config, 'emg_bandpass_low'):
                if emg_config.emg_bandpass_low >= emg_config.emg_bandpass_high:
                    # 如果低截止频率大于等于高截止频率，调整低截止频率
                    emg_config.emg_bandpass_low = emg_config.emg_bandpass_high * 0.1
                    logger.warning(
                        f"EMG低截止频率调整到{emg_config.emg_bandpass_low}Hz"
                    )


# ====================== 配置工厂 ======================

class MultiModalConfigFactory:
    """
    多模态配置工厂
    提供不同实验范式的推荐配置
    """

    @staticmethod
    def create_motor_imagery_config() -> MultiModalConfig:
        """
        创建运动想象实验的多模态配置
        （通常包含EEG、EMG、ECG）
        """
        from eeg_preprocessing import EEGConfigFactory
        from emg_preprocessing import EMGConfigFactory

        # 创建EEG配置
        eeg_config = EEGConfigFactory.create_motor_imagery_config()

        # 创建EMG配置 - 特别注意高截止频率
        emg_config = EMGConfigFactory.create_surface_emg_config()
        # 修改EMG高截止频率，确保在重采样到250Hz后仍有效
        emg_config.emg_bandpass_high = 100.0  # 250Hz采样率的奈奎斯特频率是125Hz，这里设为100Hz更安全

        return MultiModalConfig(
            processing_mode=ProcessingMode.PARALLEL,
            time_sync_method=TimeSyncMethod.RESAMPLE,
            reference_sampling_rate=250.0,  # 注意：这个值会影响EMG滤波！
            max_workers=3,

            eeg_config=eeg_config,
            emg_config=emg_config,
            ecg_config=None,  # 使用默认配置

            enabled_modalities=["EEG", "EMG", "ECG"],
            process_all_modalities=False,

            quality_check_enabled=True,
            min_signal_quality=0.6,
            auto_fix_issues=True
        )

    @staticmethod
    def create_affective_computing_config() -> MultiModalConfig:
        """
        创建情感计算实验的多模态配置
        （通常包含EEG、ECG、GSR、RESP）
        """
        return MultiModalConfig(
            processing_mode=ProcessingMode.SEQUENTIAL,
            time_sync_method=TimeSyncMethod.INTERPOLATE,
            reference_sampling_rate=100.0,
            max_workers=4,

            enabled_modalities=["EEG", "ECG", "GSR", "RESP"],
            process_all_modalities=True,

            quality_check_enabled=True,
            min_signal_quality=0.7,
            auto_fix_issues=True,

            save_intermediate_results=True,
            visualize_results=True
        )

    @staticmethod
    def create_cognitive_load_config() -> MultiModalConfig:
        """
        创建认知负荷实验的多模态配置
        （通常包含EEG、fNIRS、Eye Tracking）
        """
        return MultiModalConfig(
            processing_mode=ProcessingMode.PARALLEL,
            time_sync_method=TimeSyncMethod.EVENT_ALIGN,
            reference_sampling_rate=50.0,
            max_workers=3,

            enabled_modalities=["EEG", "fNIRS"],
            process_all_modalities=True,

            quality_check_enabled=True,
            min_signal_quality=0.65,
            auto_fix_issues=True
        )

    @staticmethod
    def create_resting_state_config() -> MultiModalConfig:
        """
        创建静息态实验的多模态配置
        """
        return MultiModalConfig(
            processing_mode=ProcessingMode.SEQUENTIAL,
            time_sync_method=TimeSyncMethod.NONE,
            max_workers=2,

            enabled_modalities=["EEG", "ECG", "RESP"],
            process_all_modalities=True,

            quality_check_enabled=True,
            min_signal_quality=0.5,
            auto_fix_issues=True
        )

    @staticmethod
    def create_realtime_config() -> MultiModalConfig:
        """
        创建实时处理的配置
        （优化处理速度）
        """
        return MultiModalConfig(
            processing_mode=ProcessingMode.SEQUENTIAL,
            time_sync_method=TimeSyncMethod.NONE,
            max_workers=1,

            enabled_modalities=["EEG", "EMG"],
            process_all_modalities=False,

            quality_check_enabled=False,
            auto_fix_issues=False,

            save_intermediate_results=False,
            save_processing_log=False
        )


# ====================== 使用示例 ======================

def example_usage():
    """
    多模态预处理器使用示例
    """
    print("=" * 70)
    print("多模态信号预处理器 - 使用示例")
    print("=" * 70)

    # 1. 创建模拟的多模态数据
    print("\n1. 创建模拟的多模态数据...")

    # 模拟参数
    sampling_rate = 1000
    n_samples = 5000  # 5秒数据
    n_eeg_channels = 32
    n_emg_channels = 4
    n_ecg_channels = 2

    # 创建模拟EEG数据
    eeg_data = np.random.randn(n_eeg_channels, n_samples) * 1e-6
    # 添加模拟alpha波
    time = np.arange(n_samples) / sampling_rate
    for ch in range(min(8, n_eeg_channels)):
        eeg_data[ch, :] += 20e-6 * np.sin(2 * np.pi * 10 * time)  # 10Hz alpha波

    # 创建模拟EMG数据
    emg_data = np.random.randn(n_emg_channels, n_samples) * 1e-3
    # 添加模拟肌肉收缩
    contraction_start = int(2.0 * sampling_rate)
    contraction_end = int(3.5 * sampling_rate)
    emg_data[:, contraction_start:contraction_end] += 5e-3 * np.random.randn(
        n_emg_channels, contraction_end - contraction_start
    )

    # 创建模拟ECG数据
    ecg_data = np.random.randn(n_ecg_channels, n_samples) * 1e-4
    # 添加模拟心搏
    heart_rate = 72  # BPM
    rr_interval = 60 / heart_rate  # 秒
    for i in range(0, n_samples, int(rr_interval * sampling_rate)):
        if i + 100 < n_samples:
            # 模拟QRS波
            qrs_wave = np.hanning(100) * 2e-3
            ecg_data[0, i:i + 100] += qrs_wave  # 修复这里的笔误：从 emg_data 改为 ecg_data

    # 构建四层数据字典
    data_dict = {
        "meta": {
            "subject_id": "S01",
            "session_id": "2024_01_15_01",
            "task": "motor_imagery",
            "modality": ["EEG", "EMG", "ECG"],
            "device": "Simulated",
            "sampling_rate": sampling_rate,
            "n_channels": n_eeg_channels,
            "channel_names": [f"EEG_{i}" for i in range(n_eeg_channels)]
        },
        "signal": {
            "EEG": {
                "data": eeg_data,
                "sampling_rate": sampling_rate,
                "unit": "V",
                "channel_names": [f"EEG_{i}" for i in range(n_eeg_channels)],
                "reference": "average"
            },
            "EMG": {
                "data": emg_data,
                "sampling_rate": sampling_rate,
                "unit": "V",
                "channel_names": ["Biceps", "Triceps", "Flexor", "Extensor"],
                "time_offset": 0.0
            },
            "ECG": {
                "data": ecg_data,
                "sampling_rate": sampling_rate,
                "unit": "V",
                "channel_names": ["ECG1", "ECG2"],
                "time_offset": 0.0
            }
        },
        "event": {
            "event_id": [1, 2],
            "event_label": ["left", "right"],
            "event_time": [1.5, 3.0], 
            "event_sample": [1500, 3000],
            "duration": [2.0, 2.0]
        },
        "processed": {}
    }

    print(f"创建了包含 {len(data_dict['signal'])} 个模态的数据:")
    for modality, info in data_dict["signal"].items():
        print(f"  - {modality}: {info['data'].shape[0]}通道, {info['data'].shape[1]}样本点")

    # 2. 创建多模态预处理器
    print("\n2. 创建多模态预处理器...")

    # 使用运动想象配置
    config = MultiModalConfigFactory.create_motor_imagery_config()
    preprocessor = MultiModalPreprocessor(config)

    print(f"处理模式: {config.processing_mode.value}")
    print(f"时间同步: {config.time_sync_method.value}")
    print(f"启用模态: {config.enabled_modalities}")

    # 3. 执行预处理
    print("\n3. 执行预处理...")

    result = preprocessor.process(data_dict)

    # 4. 显示结果
    print("\n4. 处理结果:")

    if result.success:
        print(f"✓ 处理成功，耗时: {result.processing_time:.2f}秒")
        processed_data = result.processed_data
        print(f"处理后的模态: {processed_data['processed']['multimodal_preprocessing']['modalities_processed']}")
    else:
        print(f"✗ 处理失败: {result.error_message}")

    print("\n" + "=" * 70)
    print("示例完成!")
    print("=" * 70)

    return result if result.success else None


if __name__ == "__main__":
    example_usage()
