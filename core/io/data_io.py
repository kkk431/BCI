"""
universal_biosignal_reader.py
万能生物信号数据读取器
基于 BrainFusion 项目结构，适配标准化的四层 data_dict 格式。
支持：EEG, EMG, ECG, fNIRS, EOG, EDA (皮肤电), Respiration (呼吸) 等。
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from pathlib import Path
import warnings
import logging
import re

# --- 导入项目中已有的模块（根据您的项目结构调整） ---
try:
    # 尝试导入项目中已存在的专用读取器
    from BrainFusion.io.File_IO import read_file as bf_read_file, read_neuracle_bdf, read_minilab_snirf, read_minilab_bdf
    from BrainFusion.io.snirf_io import create_snirf_file
    from readMinilabDataset import read_minilab_snirf as read_ml_snirf, read_minilab_bdf as read_ml_bdf
except ImportError:
    warnings.warn("部分BrainFusion模块未找到，某些高级功能可能受限。")
    # 可以在此处定义这些函数的简化版本或设为None

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====================== 1. 标准数据字典构建器 ======================

class DataDictBuilder:
    """
    根据标准四层结构构建和验证 data_dict。
    """

    @staticmethod
    def create_empty_data_dict() -> Dict[str, Any]:
        """创建一个空的、结构正确的四层 data_dict 模板。"""
        return {
            "meta": {},
            "signal": {},
            "event": {},
            "processed": {}
        }

    @staticmethod
    def build_meta(subject_id: str = "unknown",
                   session_id: str = None,
                   task: str = "unknown",
                   modality: List[str] = None,
                   device: str = "unknown",
                   sampling_rate: float = None,
                   n_channels: int = None,
                   channel_names: List[str] = None,
                   **extra_meta) -> Dict[str, Any]:
        """
        构建标准化的 meta 层。
        参数与飞书文档完全对应。
        """
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if modality is None:
            modality = []

        meta = {
            "subject_id": subject_id,
            "session_id": session_id,
            "task": task,
            "modality": modality,
            "device": device,
            "sampling_rate": sampling_rate,
            "n_channels": n_channels,
            "channel_names": channel_names if channel_names else [],
            "creation_time": datetime.now().isoformat(),
            **extra_meta  # 用于容纳其他自定义元数据
        }
        return {k: v for k, v in meta.items() if v is not None}  # 过滤掉 None 值

    @staticmethod
    def add_signal(data_dict: Dict,
                   modality: str,
                   data: np.ndarray,
                   sampling_rate: float,
                   channel_names: List[str],
                   unit: str = "unknown",
                   time_offset: float = 0.0,
                   reference: str = None,
                   **extra_signal_info):
        """
        向 data_dict 的 signal 层添加一种模态的信号。
        严格按照飞书文档中 signal 子字典的结构。
        """
        if "signal" not in data_dict:
            data_dict["signal"] = {}

        signal_info = {
            "data": np.asarray(data),
            "sampling_rate": float(sampling_rate),
            "unit": unit,
            "channel_names": list(channel_names),
            "time_offset": float(time_offset),
        }
        if reference is not None:
            signal_info["reference"] = reference
        signal_info.update(extra_signal_info)

        data_dict["signal"][modality.upper()] = signal_info  # 模态名统一为大写

        # 自动更新 meta 层中的 modality 列表和全局信息（如果可能）
        if "meta" in data_dict:
            if modality.upper() not in data_dict["meta"].get("modality", []):
                data_dict["meta"]["modality"] = data_dict["meta"].get("modality", []) + [modality.upper()]
            # 如果meta里没有总体采样率/通道数，且当前是第一个信号，可以设置一个参考值（注意：多模态时可能不同）
            if data_dict["meta"].get("sampling_rate") is None:
                data_dict["meta"]["sampling_rate"] = sampling_rate
            if data_dict["meta"].get("n_channels") is None:
                data_dict["meta"]["n_channels"] = data.shape[0] if data.ndim > 1 else 1

    @staticmethod
    def add_event(data_dict: Dict,
                  event_id: List[int],
                  event_label: List[str],
                  event_time: List[float],
                  event_sample: List[int] = None,
                  duration: List[float] = None):
        """
        向 data_dict 的 event 层添加事件信息。
        结构参考现有代码和通用实践。
        """
        if "event" not in data_dict:
            data_dict["event"] = {}

        n_events = len(event_time)
        # 如果未提供 sample，尝试根据第一个信号的采样率计算（如果存在）
        if event_sample is None and data_dict.get("signal"):
            # 获取第一个可用信号的采样率
            first_signal = next(iter(data_dict["signal"].values()))
            fs = first_signal.get("sampling_rate", 1)
            event_sample = [int(t * fs) for t in event_time]
        elif event_sample is None:
            event_sample = [0] * n_events

        if duration is None:
            duration = [0.0] * n_events

        data_dict["event"].update({
            "event_id": list(event_id),
            "event_label": list(event_label),
            "event_time": list(event_time),
            "event_sample": list(event_sample),
            "duration": list(duration)
        })

    @staticmethod
    def validate_data_dict(data_dict: Dict) -> Tuple[bool, List[str]]:
        """
        验证 data_dict 是否符合基本四层结构。
        返回 (是否有效, 错误/警告信息列表)。
        """
        errors = []
        required_top_keys = ["meta", "signal", "event", "processed"]
        for key in required_top_keys:
            if key not in data_dict:
                errors.append(f"缺失顶层键: '{key}'")
            elif not isinstance(data_dict[key], dict):
                errors.append(f"顶层键 '{key}' 的值必须是字典类型。")

        # 检查 signal 层内每个模态的数据结构
        if "signal" in data_dict:
            for mod_name, mod_info in data_dict["signal"].items():
                required_signal_keys = ["data", "sampling_rate", "channel_names"]
                for skey in required_signal_keys:
                    if skey not in mod_info:
                        errors.append(f"信号模态 '{mod_name}' 中缺失关键字段: '{skey}'")
                if "data" in mod_info:
                    data = mod_info["data"]
                    ch_names = mod_info.get("channel_names", [])
                    if data.ndim != 2:
                        errors.append(f"信号模态 '{mod_name}' 的 data 必须是2维数组 (channels x time)。当前维度: {data.ndim}")
                    elif data.shape[0] != len(ch_names):
                        errors.append(f"信号模态 '{mod_name}' 的通道数 ({data.shape[0]}) 与 channel_names 长度 ({len(ch_names)}) 不匹配。")

        is_valid = len(errors) == 0
        return is_valid, errors

# ====================== 2. 专用格式读取器适配层 ======================

class BioSignalReader:
    """
    万能生物信号读取器主类。
    作为调度中心，调用各种专用读取器，并将结果统一到标准 data_dict。
    """

    def __init__(self, default_subject_id="S01"):
        self.builder = DataDictBuilder()
        self.default_subject_id = default_subject_id

    def read(self, file_path: Union[str, List],
             modality_hint: str = None,
             subject_id: str = None,
             task: str = "unknown",
             **kwargs) -> Dict[str, Any]:
        """
        主读取函数。自动检测文件格式并调用相应的读取器。

        参数:
            file_path: 文件路径或路径列表（如BDF的数据+事件文件）。
            modality_hint: 提示信号模态 (如 'EEG', 'EMG')，辅助自动识别。
            subject_id: 被试ID，用于填充meta。
            task: 实验任务名称。
            **kwargs: 传递给底层读取器的其他参数。

        返回:
            符合四层结构的 data_dict。
        """
        logger.info(f"开始读取文件: {file_path}")

        # 1. 创建空数据结构
        data_dict = self.builder.create_empty_data_dict()

        # 2. 根据文件扩展名和提示选择读取策略
        raw_data, meta_info, events, actual_modality = self._dispatch_reader(file_path, modality_hint, **kwargs)

        # 3. 构建 meta 层
        data_dict["meta"] = self.builder.build_meta(
            subject_id=subject_id or self.default_subject_id,
            session_id=kwargs.get('session_id'),
            task=task,
            modality=[actual_modality] if actual_modality else [],
            device=meta_info.get('device', 'unknown'),
            sampling_rate=meta_info.get('sampling_rate'),
            n_channels=meta_info.get('n_channels'),
            channel_names=meta_info.get('channel_names', []),
            **meta_info  # 其他从文件读取的元信息
        )

        # 4. 构建 signal 层
        if raw_data is not None and 'data' in raw_data:
            # 处理多模态情况：raw_data可能是一个字典，key为模态名
            if isinstance(raw_data, dict) and 'data' not in raw_data:
                # raw_data 已经是 {‘EEG’: {…}, ‘EMG’: {…}} 的形式
                for mod_name, mod_data in raw_data.items():
                    self.builder.add_signal(data_dict, mod_name, **mod_data)
            else:
                # raw_data 是单一模态
                self.builder.add_signal(data_dict,
                                        actual_modality or 'UNKNOWN',
                                        data=raw_data['data'],
                                        sampling_rate=raw_data.get('sampling_rate', meta_info.get('sampling_rate')),
                                        channel_names=raw_data.get('channel_names', meta_info.get('channel_names', [])),
                                        unit=raw_data.get('unit', 'uV'),
                                        time_offset=raw_data.get('time_offset', 0.0))

        # 5. 构建 event 层
        if events:
            # 假设 events 是列表格式 [[onset, duration, label], ...]
            # 将其转换为 data_dict 的 event 层结构
            event_times = [e[0] for e in events]
            event_labels = [str(e[2]) for e in events]
            event_durations = [e[1] for e in events]
            self.builder.add_event(data_dict,
                                   event_id=list(range(1, len(events)+1)),
                                   event_label=event_labels,
                                   event_time=event_times,
                                   duration=event_durations)

        # 6. 验证并返回
        is_valid, validation_errors = self.builder.validate_data_dict(data_dict)
        if not is_valid:
            logger.warning(f"数据字典验证发现一些问题: {validation_errors}")
        else:
            logger.info("数据字典构建与验证成功。")

        return data_dict

    def _dispatch_reader(self, file_path, modality_hint, **kwargs):
        """
        根据文件类型分派到具体的读取函数。
        这里是读取器扩展的核心。
        """
        file_ext = Path(file_path[0] if isinstance(file_path, list) else file_path).suffix.lower()

        raw_data, meta_info, events, actual_modality = None, {}, [], modality_hint

        # 映射：文件扩展名 -> (读取函数, 默认模态)
        reader_map = {
            '.bdf': (self._read_bdf_wrapper, 'EEG'),  # 可包含EEG, EMG, EOG等
            '.edf': (self._read_edf_wrapper, 'EEG'),
            '.snirf': (self._read_snirf_wrapper, 'FNIRS'),
            '.nirs': (self._read_nirs_wrapper, 'FNIRS'),
            '.mat': (self._read_mat_wrapper, None),  # MATLAB文件，需内部判断
            '.csv': (self._read_csv_wrapper, None),
            '.txt': (self._read_csv_wrapper, None),
            '.xlsx': (self._read_excel_wrapper, None),
            '.vhdr': (self._read_brainvision_wrapper, 'EEG'),  # BrainVision
            '.set': (self._read_eeglab_wrapper, 'EEG'),  # EEGLAB
            '.acq': (self._read_acqknowledge_wrapper, None),  # BIOPAC
            '.eeg': (self._read_curry_wrapper, 'EEG'),  # Curry
        }

        if file_ext in reader_map:
            reader_func, default_modality = reader_map[file_ext]
            if actual_modality is None:
                actual_modality = default_modality
            try:
                raw_data, meta_info, events = reader_func(file_path, **kwargs)
            except Exception as e:
                logger.error(f"使用 {reader_func.__name__} 读取文件失败: {e}")
                raise
        else:
            # 尝试通用读取或项目内置读取器
            logger.warning(f"未直接支持扩展名 {file_ext}，尝试通用方法。")
            raw_data, meta_info, events = self._read_generic(file_path, **kwargs)

        return raw_data, meta_info, events, actual_modality

    # ---------- 以下为具体格式的读取适配器 ----------
    # 每个函数目标：返回 (raw_data_dict, meta_info_dict, events_list)

    def _read_bdf_wrapper(self, file_path, **kwargs):
        """包装读取 BDF 格式 (Neuracle, Biosemi等)"""
        # 优先使用项目中已有的强大读取器
        if isinstance(file_path, list) and len(file_path) == 2:
            # 数据+事件文件对
            data_from_reader = read_neuracle_bdf(file_path, is_data_transform=True)
        else:
            # 单文件，使用 read_bdf 或 read_minilab_bdf
            single_file = file_path[0] if isinstance(file_path, list) else file_path
            # 根据文件内容或路径判断使用哪个
            if 'minilab' in str(single_file).lower():
                data_from_reader = read_minilab_bdf(single_file, **kwargs)
            else:
                # 调用 File_IO.py 中的 read_bdf
                from BrainFusion.io.File_IO import read_bdf
                data_from_reader = read_bdf(single_file, **kwargs)

        # 将返回的数据字典转换为我们需要的格式
        raw_data = {
            'data': np.array(data_from_reader.get('data')),
            'sampling_rate': data_from_reader.get('srate'),
            'channel_names': data_from_reader.get('ch_names', []),
            'unit': 'uV'
        }
        meta = {
            'device': 'Neuracle' if 'neuracle' in str(file_path).lower() else 'Biosemi/Unknown',
            'n_channels': data_from_reader.get('nchan'),
            'original_format': 'BDF'
        }
        events = data_from_reader.get('events', [])
        return raw_data, meta, events

    def _read_snirf_wrapper(self, file_path, **kwargs):
        """包装读取 SNIRF 格式 (fNIRS)"""
        single_file = file_path[0] if isinstance(file_path, list) else file_path
        data_from_reader = read_minilab_snirf(single_file, **kwargs)

        raw_data = {
            'data': data_from_reader.get('data'),
            'sampling_rate': data_from_reader.get('srate'),
            'channel_names': data_from_reader.get('ch_names', []),
            'unit': 'V'  # fNIRS 常用光学密度或浓度变化
        }
        meta = {
            'device': 'MiniLab or SNIRF-compatible',
            'n_channels': data_from_reader.get('nchan'),
            'wavelengths': data_from_reader.get('wavelengths', []),
            'original_format': 'SNIRF'
        }
        events = data_from_reader.get('events', [])
        return raw_data, meta, events

    def _read_edf_wrapper(self, file_path, **kwargs):
        """包装读取 EDF/EDF+ 格式"""
        import pyedflib
        # ... EDF读取实现 (与BDF类似) ...
        pass

    def _read_mat_wrapper(self, file_path, **kwargs):
        """读取 MATLAB .mat 文件，自动识别常见生理信号数据集结构"""
        from scipy.io import loadmat
        data = loadmat(file_path)
        # 尝试根据常见键名猜测结构，例如来自BCI竞赛、OpenBMI等数据集
        pass

    def _read_csv_wrapper(self, file_path, **kwargs):
        """读取 CSV/TXT 文件，需通过参数明确指定列含义"""
        df = pd.read_csv(file_path, **kwargs.get('csv_args', {}))
        # 假设列名为通道名，或通过参数指定哪些列是数据、哪列是时间戳
        pass

    def _read_acqknowledge_wrapper(self, file_path, **kwargs):
        """读取 BIOPAC AcqKnowledge 文件 (用于EDA/ECG/EMG/呼吸)"""
        # 可能需要使用 bioread 库: `pip install bioread`
        try:
            import bioread
            data = bioread.read(file_path)
            # 解析多个通道，识别其类型 (EDA, ECG, RESP等)
        except ImportError:
            raise ImportError("读取 .acq 文件需要 'bioread' 库。请运行 `pip install bioread`。")
        pass

    def _read_generic(self, file_path, **kwargs):
        """通用后备读取方法，使用项目中已存在的 read_file 函数"""
        data_from_reader = bf_read_file(file_path)
        # 尝试从返回的字典中提取标准化信息
        pass

# ====================== 3. 工厂函数与快捷方式 ======================

def read_biosignal(file_path, **kwargs):
    """
    万能读取的快捷函数。
    示例:
        data = read_biosignal('sub-01_eeg.bdf', modality_hint='EEG', subject_id='S01', task='rest')
        data = read_biosignal('sub-01_emg.edf', modality_hint='EMG')
        data = read_biosignal('sub-01_eda.acq', modality_hint='EDA')
    """
    reader = BioSignalReader()
    return reader.read(file_path, **kwargs)

def load_saved_datadict(json_file_path: str) -> Dict[str, Any]:
    """加载之前保存为 JSON 的标准化 data_dict。"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    # JSON中数组需转回numpy array
    if 'signal' in data:
        for mod in data['signal'].values():
            mod['data'] = np.array(mod['data'])
    return data

def save_datadict(data_dict: Dict[str, Any], json_file_path: str):
    """将 data_dict 保存为 JSON 文件（数据会转换为列表）。"""
    # 创建可JSON序列化的副本
    save_dict = json.loads(json.dumps(data_dict, default=_json_serializer))
    with open(json_file_path, 'w') as f:
        json.dump(save_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"data_dict 已保存至: {json_file_path}")

def _json_serializer(obj):
    """JSON序列化辅助函数，用于处理numpy数组等类型。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

# ====================== 4. 使用示例与测试 ======================

if __name__ == "__main__":
    """
    直接运行此文件进行功能测试。
    """

    print("=== 万能生物信号读取器测试 ===\n")

    # 示例1：模拟创建一个包含EMG和EEG的多模态数据
    print("1. 创建模拟多模态数据字典...")
    builder = DataDictBuilder()
    test_dict = builder.create_empty_data_dict()

    # 添加meta
    test_dict["meta"] = builder.build_meta(
        subject_id="TEST01",
        task="multi_modal_test",
        modality=["EEG", "EMG"],
        device="Simulator",
        sampling_rate=1000,
        n_channels=6,
        channel_names=["Fz", "Cz", "Pz", "Bicep", "Tricep", "Flexor"]
    )

    # 添加EEG信号
    import numpy as np
    eeg_data = np.random.randn(3, 1000) * 50  # 3个通道，1000个时间点
    builder.add_signal(test_dict, "EEG", eeg_data,
                       sampling_rate=1000,
                       channel_names=["Fz", "Cz", "Pz"],
                       unit="uV",
                       reference="Cz")

    # 添加EMG信号
    emg_data = np.random.randn(3, 2000) * 100  # 3个通道，2000个时间点（更高采样率）
    builder.add_signal(test_dict, "EMG", emg_data,
                       sampling_rate=2000,
                       channel_names=["Bicep", "Tricep", "Flexor"],
                       unit="uV",
                       time_offset=0.001)

    # 添加事件
    builder.add_event(test_dict,
                      event_id=[1, 2],
                      event_label=["left_move", "right_move"],
                      event_time=[1.5, 3.8],
                      duration=[0.5, 0.5])

    # 验证
    is_valid, errors = builder.validate_data_dict(test_dict)
    if is_valid:
        print("   模拟数据字典验证成功！")
        print(f"   包含信号模态: {list(test_dict['signal'].keys())}")
    else:
        print("   验证失败:", errors)

    # 示例2：演示使用快捷函数读取（此处为模拟，实际需替换为真实文件路径）
    print("\n2. 演示文件读取流程（需真实文件）...")
    # 假设有以下文件，请根据实际情况取消注释测试
    # try:
    #     # 读取一个BDF文件 (EEG/EMG)
    #     eeg_dict = read_biosignal('path/to/your/data.bdf', modality_hint='EEG', subject_id='S01', task='finger_tapping')
    #     print(f"   BDF文件读取成功，采样率: {eeg_dict['meta']['sampling_rate']}")
    #
    #     # 读取一个SNIRF文件 (fNIRS)
    #     fnirs_dict = read_biosignal('path/to/your/data.snirf', subject_id='S02', task='n_back')
    #     print(f"   SNIRF文件读取成功，通道数: {fnirs_dict['meta']['n_channels']}")
    #
    #     # 读取一个BIOPAC文件 (EDA/ECG/Resp)
    #     eda_dict = read_biosignal('path/to/your/data.acq', modality_hint='EDA')
    #     print(f"   ACQ文件读取成功，信号模态: {eda_dict['meta']['modality']}")
    #
    # except FileNotFoundError as e:
    #     print(f"   测试文件未找到，请将示例路径替换为您的真实文件路径。")

    print("\n=== 测试完成 ===")