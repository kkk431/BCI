"""
universal_biosignal_reader.py
万能生物信号数据读取器
依赖通用库：h5py, pyedflib, numpy, pandas, scipy
不依赖 BrainFusion 专有模块
完全独立运行

整合所有功能，支持多种生物信号格式读取
基于标准化的四层 data_dict 格式。
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.io
import h5py
import pyedflib
import warnings
import logging
import struct
import csv
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from pathlib import Path

# 尝试导入其他通用库（可选）
try:
    import snirf
    SNIRF_AVAILABLE = True
except ImportError:
    SNIRF_AVAILABLE = False
    warnings.warn("snirf 库未安装，SNIRF 读取功能可能受限")

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    warnings.warn("mne 库未安装，BrainVision/EEGLAB 读取功能可能受限")

try:
    import bioread
    BIOREAD_AVAILABLE = True
except ImportError:
    BIOREAD_AVAILABLE = False
    warnings.warn("bioread 库未安装，BIOPAC AcqKnowledge 读取功能可能受限")

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
        return {k: v for k, v in meta.items() if v is not None}

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

        data_dict["signal"][modality.upper()] = signal_info

        # 自动更新 meta 层
        if "meta" in data_dict:
            if modality.upper() not in data_dict["meta"].get("modality", []):
                data_dict["meta"]["modality"] = data_dict["meta"].get("modality", []) + [modality.upper()]
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
        """
        if "event" not in data_dict:
            data_dict["event"] = {}

        n_events = len(event_time)
        if event_sample is None and data_dict.get("signal"):
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

# ====================== 2. 核心读取功能（不依赖 BrainFusion） ======================

def remove_dtype(value):
    """
    Convert NumPy arrays to Python lists for serialization
    从 readTUBerlinBCI.py 整合
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

def read_minilab_bdf(path, signal_type='eeg', montage=''):
    """
    Read MiniLab BDF file.
    使用 pyedflib（通用库）
    """
    try:
        bdf_file = pyedflib.EdfReader(path)
        n_channels = bdf_file.signals_in_file
        data = []

        for i in range(n_channels):
            signal = bdf_file.readSignal(i, digital=False) * 0.000001
            data.append(signal)

        # Parse events
        events = np.array(bdf_file.readAnnotations()).T.tolist()
        processed_events = []
        for event in events:
            time, duration, label = event
            label = str(int(float(label)))
            time = float(time)
            duration = float(duration)
            processed_events.append([time, duration, label])

        result = {
            'data': data,
            'sampling_rate': bdf_file.getSampleFrequencies()[0],
            'events': processed_events,
            'num_channels': n_channels,
            'channel_names': [label.replace('.', '') for label in bdf_file.getSignalLabels()],
            'signal_type': signal_type,
            'montage': montage,
            'file_info': {
                'PhysicalMaximum': bdf_file.getPhysicalMaximum(),
                'PhysicalMinimum': bdf_file.getPhysicalMinimum(),
                'DigitalMaximum': bdf_file.getDigitalMaximum(),
                'DigitalMinimum': bdf_file.getDigitalMinimum()
            }
        }

        bdf_file.close()
        return result
    except Exception as e:
        raise Exception(f"读取 BDF 文件失败: {e}")

def read_minilab_snirf(path=None, signal_type='fnirs'):
    """
    Read MiniLab SNIRF file.
    使用 snirf 库（通用库）
    """
    if not SNIRF_AVAILABLE:
        raise ImportError("需要安装 snirf 库：pip install snirf")

    try:
        snirf_file = snirf.loadSnirf(path)
        nirs_data = snirf_file.nirs[0]

        # Time-series data
        data_dict = {
            'data': nirs_data.data[0].dataTimeSeries,
            'time': nirs_data.data[0].time,
            'signal_type': signal_type
        }

        # Event markers
        events = []
        if len(nirs_data.stim) > 0:
            for stim in nirs_data.stim:
                if stim.data is not None and len(stim.data) > 0:
                    for event in stim.data:
                        time, duration, label = event
                        time = float(time)
                        duration = float(duration)
                        label = str(int(float(label)))
                        events.append([time, duration, label])

        if events:
            events.sort(key=lambda x: x[0])
            data_dict['events'] = events
        else:
            data_dict['events'] = None

        # Channel information
        data_dict['num_channels'] = data_dict['data'].shape[1]
        data_dict['channel_names'] = nirs_data.probe.landmarkLabels

        # Optode positions
        data_dict['locations'] = {
            'source_positions': nirs_data.probe.sourcePos3D,
            'detector_positions': nirs_data.probe.detectorPos3D,
            'landmark_positions': nirs_data.probe.landmarkPos3D
        }

        # Source-detector pairs
        sd_pairs = []
        for ch_name in data_dict['channel_names'][:data_dict['num_channels'] // 2]:
            parts = ch_name.split(" ")[0]
            source, detector = parts.split("_")
            source_idx = int(source[1:])
            detector_idx = int(detector[1:])
            sd_pairs.append((source_idx, detector_idx))

        data_dict['source_detector_pairs'] = sd_pairs
        data_dict['wavelengths'] = nirs_data.probe.wavelengths

        return data_dict
    except Exception as e:
        raise Exception(f"读取 SNIRF 文件失败: {e}")

def read_tu_berlin_bci_eeg_data(folder_path):
    """
    Load TU Berlin BCI EEG dataset subject folder
    使用 scipy.io（通用库）
    """
    try:
        cnt_file = os.path.join(folder_path, 'cnt.mat')
        mrk_file = os.path.join(folder_path, 'mrk.mat')
        mnt_file = os.path.join(folder_path, 'mnt.mat')

        cnt_data = scipy.io.loadmat(cnt_file, struct_as_record=False, squeeze_me=True)
        mrk_data = scipy.io.loadmat(mrk_file, struct_as_record=False, squeeze_me=True)
        mnt_data = scipy.io.loadmat(mnt_file, struct_as_record=False, squeeze_me=True)

        cnt = cnt_data['cnt']
        mrk = mrk_data['mrk']
        mnt = mnt_data['mnt']

        combined_data = {}

        for i, (cnt_struct, mrk_struct) in enumerate(zip(cnt, mrk)):
            struct_key = f"struct_{i + 1}"
            eeg_dict = {
                'data': remove_dtype(cnt_struct.x.T / 1000000),
                'num_channels': len(remove_dtype(cnt_struct.clab)),
                'channel_names': remove_dtype(cnt_struct.clab),
                'sampling_rate': remove_dtype(cnt_struct.fs),
                'title': remove_dtype(cnt_struct.title),
                'time': remove_dtype(cnt_struct.T),
                'units': remove_dtype(cnt_struct.yUnit),
                'signal_type': 'eeg',
                'montage': 'standard_1005',
            }

            # Process events
            event_times = remove_dtype(mrk_struct.time)
            event_ids = remove_dtype(mrk_struct.event.desc)
            events = [[time, 0, event_id] for time, event_id in zip(event_times, event_ids)]
            eeg_dict['events'] = events

            # Add electrode positions
            eeg_dict['locations'] = {
                'x': remove_dtype(mnt.x),
                'y': remove_dtype(mnt.y),
                'positions_3d': remove_dtype(mnt.positions_3d),
                'channel_names': remove_dtype(mnt.clab),
            }

            combined_data[struct_key] = eeg_dict

        return combined_data
    except Exception as e:
        raise Exception(f"读取 TU Berlin EEG 数据失败: {e}")

def read_tu_berlin_bci_nirs_data(folder_path):
    """
    Load TU Berlin BCI NIRS dataset subject folder
    使用 scipy.io（通用库）
    """
    try:
        cnt_file = os.path.join(folder_path, 'cnt.mat')
        mrk_file = os.path.join(folder_path, 'mrk.mat')
        mnt_file = os.path.join(folder_path, 'mnt.mat')

        cnt_data = scipy.io.loadmat(cnt_file, struct_as_record=False, squeeze_me=True)
        mrk_data = scipy.io.loadmat(mrk_file, struct_as_record=False, squeeze_me=True)
        mnt_data = scipy.io.loadmat(mnt_file, struct_as_record=False, squeeze_me=True)

        cnt = cnt_data['cnt']
        mrk = mrk_data['mrk']
        mnt = mnt_data['mnt']

        nirs_data = {}

        for i, (cnt_struct, mrk_struct) in enumerate(zip(cnt, mrk)):
            struct_key = f"struct_{i + 1}"
            nirs_dict = {
                'channel_names': remove_dtype(cnt_struct.clab),
                'sampling_rate': remove_dtype(cnt_struct.fs),
                'title': remove_dtype(cnt_struct.title),
                'data': remove_dtype(cnt_struct.x.T),
                'wavelengths': remove_dtype(cnt_struct.wavelengths),
                'signal_type': 'fnirs',
                'num_channels': len(remove_dtype(cnt_struct.clab)),
                'montage': None,
            }

            # Process events
            event_times = remove_dtype(mrk_struct.time)
            event_ids = remove_dtype(mrk_struct.event.desc)
            events = [[time, 0, event_id] for time, event_id in zip(event_times, event_ids)]
            nirs_dict['events'] = events

            # Process positions
            nirs_dict.update({
                'source_positions_2d': [
                    [x, y] for x, y in
                    zip(remove_dtype(mnt.source.x), remove_dtype(mnt.source.y))
                ],
                'detector_positions_2d': [
                    [x, y] for x, y in
                    zip(remove_dtype(mnt.detector.x), remove_dtype(mnt.detector.y))
                ],
                'source_positions_3d': remove_dtype(mnt.source.positions_3d.T),
                'detector_positions_3d': remove_dtype(mnt.detector.positions_3d.T),
                'source_labels': remove_dtype(mnt.source.clab),
                'detector_labels': remove_dtype(mnt.detector.clab),
                'landmark_positions_2d': [
                    [x, y] for x, y in
                    zip(remove_dtype(mnt.x), remove_dtype(mnt.y))
                ],
                'landmark_positions_3d': remove_dtype(mnt.positions_3d.T),
                'landmark_labels': remove_dtype(mnt.clab),
                'source_detector_pairs': remove_dtype(mnt.sd),
            })

            nirs_dict = interleave_channels(nirs_dict)
            nirs_data[struct_key] = nirs_dict

        return nirs_data
    except Exception as e:
        raise Exception(f"读取 TU Berlin fNIRS 数据失败: {e}")

def interleave_channels(data):
    """
    Interleave fNIRS channel data by wavelength
    从 readTUBerlinBCI.py 整合
    """
    num_channels = len(data['channel_names'])
    if num_channels % 2 != 0:
        raise ValueError("Number of channels must be even")

    half_size = num_channels // 2
    interleaved_names = []
    interleaved_data = []

    for i in range(half_size):
        # Append first wavelength channel
        interleaved_names.append(data['channel_names'][i])
        interleaved_data.append(data['data'][i])

        # Append second wavelength channel
        interleaved_names.append(data['channel_names'][i + half_size])
        interleaved_data.append(data['data'][i + half_size])

    data['channel_names'] = interleaved_names
    data['data'] = interleaved_data

    return data

def filter_optode_positions(data):
    """
    Filter invalid optode positions marked with '-'
    从 readTUBerlinBCI.py 整合
    """
    # Identify invalid sources
    source_mask = ['-' in label for label in data['source_labels']]
    detector_mask = ['-' in label for label in data['detector_labels']]

    # Adjust source-detector pairs
    sd_pairs = data['source_detector_pairs']
    offset = 0
    for i, is_invalid in enumerate(source_mask):
        if is_invalid:
            for pair in sd_pairs:
                if pair[0] > i - offset:
                    pair[0] -= 1
            offset += 1

    # Filter positions
    data['source_positions_3d'] = np.array([
        pos for pos, is_invalid in zip(data['source_positions_3d'], source_mask)
        if not is_invalid
    ]) * 10

    data['detector_positions_3d'] = np.array([
        pos for pos, is_invalid in zip(data['detector_positions_3d'], detector_mask)
        if not is_invalid
    ]) * 10

    # Apply coordinate adjustments
    data['source_positions_3d'][:, 2] += 5
    data['detector_positions_3d'][:, 2] += 5
    data['source_positions_3d'][:, 1] += 1
    data['detector_positions_3d'][:, 1] += 1

    return data

def load_events_from_excel(input_path):
    """
    Load events from Excel.
    使用 pandas（通用库）
    """
    try:
        data = pd.read_excel(input_path)
        required = ['Time', 'Duration', 'Label']

        if not all(col in data.columns for col in required):
            raise ValueError("Excel file missing required columns")

        return data[required].values.tolist()
    except Exception as e:
        raise Exception(f"从 Excel 加载事件失败: {e}")

def save_events_to_excel(events, output_path):
    """
    Save events to Excel.
    使用 pandas（通用库）
    """
    try:
        df = pd.DataFrame(events, columns=['Time', 'Duration', 'Label'])
        df.to_excel(output_path, index=False)
        logger.info(f"Events saved: {output_path}")
    except Exception as e:
        raise Exception(f"保存事件到 Excel 失败: {e}")

# ====================== 3. SNIRF IO 功能（使用 h5py 通用库） ======================

def create_snirf_file_integrated(filename, format_version='1.0', metadata=None, data_time_series=None,
                                 time_points=None, measurement_lists=None,
                                 source_pos_3d=None, detector_pos_3d=None, wavelengths=None,
                                 wavelengths_emission=None, source_pos_2d=None, detector_pos_2d=None,
                                 frequencies=None, time_delays=None, time_delay_widths=None,
                                 moment_orders=None, correlation_time_delays=None,
                                 correlation_time_delay_widths=None, source_labels=None,
                                 detector_labels=None, landmark_pos_2d=None, landmark_pos_3d=None,
                                 landmark_labels=None, coordinate_system='',
                                 coordinate_system_description='', use_local_index=0,
                                 stim_lists=None):
    """
    Create a SNIRF-compliant HDF5 file
    使用 h5py（通用库）
    """
    try:
        default_metadata = {
            'SubjectID': 'Unknown',
            'MeasurementDate': datetime.now().strftime('%Y-%m-%d'),
            'MeasurementTime': datetime.now().strftime('%H:%M:%S'),
            'LengthUnit': 'cm',
            'TimeUnit': 's',
            'FrequencyUnit': 'Hz'
        }

        # Create variable-length string datatype
        varlen_str_dtype = h5py.string_dtype(encoding='ascii', length=None)

        with h5py.File(filename, 'w') as f:
            # Set format version
            f.create_dataset('formatVersion', dtype=varlen_str_dtype, data=format_version)
            nirs_group = f.create_group('/nirs')

            # Create metadata tags
            metadata_group = nirs_group.create_group('metaDataTags')
            if metadata:
                for key, value in metadata.items():
                    metadata_group.create_dataset(key, dtype=varlen_str_dtype, data=value)
            else:
                for key, value in default_metadata.items():
                    metadata_group.create_dataset(key, dtype=varlen_str_dtype, data=value)

            # Create primary data container
            data_group = nirs_group.create_group('data1')
            if data_time_series is not None:
                data_group.create_dataset('dataTimeSeries', dtype='f8', data=data_time_series)
            if time_points is not None:
                data_group.create_dataset('time', dtype='f8', data=time_points)

            # Create measurement lists
            if measurement_lists:
                i = 1
                for measurement in measurement_lists:
                    measurement_list_group = data_group.create_group(f'measurementList{i}')

                    # Required fields
                    measurement_list_group.create_dataset('sourceIndex', dtype='i4',
                                                          data=measurement.get('sourceIndex', 1))
                    measurement_list_group.create_dataset('detectorIndex', dtype='i4',
                                                          data=measurement.get('detectorIndex', 1))
                    measurement_list_group.create_dataset('wavelengthIndex', dtype='i4',
                                                          data=measurement.get('wavelengthIndex', 1))
                    measurement_list_group.create_dataset('dataType', dtype='i4',
                                                          data=measurement.get('dataType', 1))
                    measurement_list_group.create_dataset('dataTypeIndex', dtype='i4',
                                                          data=measurement.get('dataTypeIndex', 1))

                    i += 1

            # Create probe section
            probe_group = nirs_group.create_group('probe')

            # Positional data
            if source_pos_3d is not None:
                probe_group.create_dataset('sourcePos3D', dtype='f8', data=source_pos_3d)
            if detector_pos_3d is not None:
                probe_group.create_dataset('detectorPos3D', dtype='f8', data=detector_pos_3d)

            # Wavelength data
            if wavelengths is not None:
                probe_group.create_dataset('wavelengths', dtype='f8', data=wavelengths)

            # Label information
            if landmark_labels is not None:
                probe_group.create_dataset('landmarkLabels', dtype=varlen_str_dtype,
                                           data=landmark_labels)

            # Coordinate system
            probe_group.create_dataset('coordinateSystem', dtype=varlen_str_dtype,
                                       data=coordinate_system)

            # Create stimulation section
            if stim_lists:
                for i, stim_dict in enumerate(stim_lists, start=1):
                    stim_group = nirs_group.create_group(f'stim{i}')
                    if 'name' in stim_dict:
                        stim_group.create_dataset('name', dtype=varlen_str_dtype, data=stim_dict['name'])
                    if 'data' in stim_dict and stim_dict['data']:
                        stim_group.create_dataset('data', dtype='f8', data=np.array(stim_dict['data']))

        logger.info(f"SNIRF file created successfully: {filename}")
    except Exception as e:
        raise Exception(f"创建 SNIRF 文件失败: {e}")

# ====================== 4. BDF 文件创建功能（替代 BrainFusion） ======================

def create_standard_bdf_file(file_name, num_channels, signals, channel_names, sampling_frequency,
                             physical_mins=None, physical_maxs=None, digital_mins=None,
                             digital_maxs=None, annotations=None):
    """
    创建标准 BDF 文件
    替代 BrainFusion 中的 create_standard_bdf_file 函数
    使用 pyedflib（通用库）
    """
    try:
        # 设置默认值
        if physical_mins is None:
            physical_mins = [-32768.0] * num_channels
        if physical_maxs is None:
            physical_maxs = [32767.0] * num_channels
        if digital_mins is None:
            digital_mins = [-8388608] * num_channels
        if digital_maxs is None:
            digital_maxs = [8388607] * num_channels

        # 确保信号是二维数组
        if isinstance(signals, list):
            signals = np.array(signals)

        # 获取信号长度
        n_samples = signals.shape[1] if signals.ndim > 1 else len(signals)

        # 创建 EDF 文件头
        file_duration = n_samples / sampling_frequency

        # 使用 pyedflib 创建文件
        writer = pyedflib.EdfWriter(file_name, num_channels, file_type=pyedflib.FILETYPE_BDFPLUS)

        # 设置通道信息
        channel_info = []
        for i in range(num_channels):
            info_dict = {
                'label': channel_names[i] if i < len(channel_names) else f'CH{i+1:03d}',
                'dimension': 'uV',
                'sample_rate': sampling_frequency,
                'physical_min': physical_mins[i],
                'physical_max': physical_maxs[i],
                'digital_min': digital_mins[i],
                'digital_max': digital_maxs[i],
                'transducer': '',
                'prefilter': ''
            }
            channel_info.append(info_dict)

        writer.setSignalHeaders(channel_info)

        # 写入信号数据
        for i in range(num_channels):
            if signals.ndim == 1:
                writer.writePhysicalSamples(signals[i])
            else:
                writer.writePhysicalSamples(signals[i, :])

        # 写入注释（事件）
        if annotations:
            for annotation in annotations:
                onset, duration, label = annotation
                writer.writeAnnotation(onset, duration, str(label))

        writer.close()
        logger.info(f"BDF file created successfully: {file_name}")

    except Exception as e:
        raise Exception(f"创建 BDF 文件失败: {e}")

# ====================== 5. 万能生物信号读取器主类 ======================

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
            **meta_info
        )

        # 4. 构建 signal 层
        if raw_data is not None and 'data' in raw_data:
            if isinstance(raw_data, dict) and 'data' not in raw_data:
                for mod_name, mod_data in raw_data.items():
                    self.builder.add_signal(data_dict, mod_name, **mod_data)
            else:
                self.builder.add_signal(data_dict,
                                        actual_modality or 'UNKNOWN',
                                        data=raw_data['data'],
                                        sampling_rate=raw_data.get('sampling_rate', meta_info.get('sampling_rate')),
                                        channel_names=raw_data.get('channel_names', meta_info.get('channel_names', [])),
                                        unit=raw_data.get('unit', 'uV'),
                                        time_offset=raw_data.get('time_offset', 0.0))

        # 5. 构建 event 层
        if events:
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
        """
        file_ext = Path(file_path[0] if isinstance(file_path, list) else file_path).suffix.lower()

        raw_data, meta_info, events, actual_modality = None, {}, [], modality_hint

        # 1. 检测是否为 TU Berlin 数据集
        if self._is_tu_berlin_dataset(file_path):
            logger.info("检测到 TU Berlin BCI 数据集格式")
            return self._read_tu_berlin_wrapper(file_path, **kwargs)

        # 映射：文件扩展名 -> (读取函数, 默认模态)
        reader_map = {
            '.bdf': (self._read_bdf_wrapper, 'EEG'),
            '.edf': (self._read_edf_wrapper, 'EEG'),
            '.snirf': (self._read_snirf_wrapper, 'FNIRS'),
            '.nirs': (self._read_nirs_wrapper, 'FNIRS'),
            '.mat': (self._read_mat_wrapper, None),
            '.csv': (self._read_csv_wrapper, None),
            '.txt': (self._read_csv_wrapper, None),
            '.xlsx': (self._read_excel_wrapper, None),
            '.vhdr': (self._read_brainvision_wrapper, 'EEG'),
            '.set': (self._read_eeglab_wrapper, 'EEG'),
            '.acq': (self._read_acqknowledge_wrapper, None),
            '.eeg': (self._read_curry_wrapper, 'EEG'),
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
            logger.warning(f"未直接支持扩展名 {file_ext}，尝试通用方法。")
            raw_data, meta_info, events = self._read_generic(file_path, **kwargs)

        return raw_data, meta_info, events, actual_modality

    def _is_tu_berlin_dataset(self, file_path):
        """检测是否为 TU Berlin BCI 数据集"""
        path_str = str(file_path).lower()
        patterns = ['tu_berlin', 'tuberlin', 'berlin_bci']

        if any(pattern in path_str for pattern in patterns):
            return True

        if os.path.isdir(file_path):
            mat_files = ['cnt.mat', 'mrk.mat', 'mnt.mat']
            for mat_file in mat_files:
                if os.path.exists(os.path.join(file_path, mat_file)):
                    return True

        return False

    def _read_tu_berlin_wrapper(self, folder_path, **kwargs):
        """统一读取 TU Berlin 数据集"""
        logger.info(f"读取 TU Berlin 数据集: {folder_path}")

        if os.path.exists(os.path.join(folder_path, 'cnt.mat')):
            try:
                mat_data = scipy.io.loadmat(
                    os.path.join(folder_path, 'cnt.mat'),
                    struct_as_record=False,
                    squeeze_me=True
                )

                cnt = mat_data['cnt']
                if isinstance(cnt, np.ndarray) and len(cnt) > 0:
                    first_cnt = cnt[0] if isinstance(cnt, np.ndarray) else cnt

                    if hasattr(first_cnt, 'wavelengths'):
                        logger.info("检测到 fNIRS 数据")
                        return self._read_tu_berlin_nirs(folder_path, **kwargs)
                    else:
                        logger.info("检测到 EEG 数据")
                        return self._read_tu_berlin_eeg(folder_path, **kwargs)
            except Exception as e:
                logger.warning(f"无法自动检测 TU Berlin 数据类型，默认按 EEG 处理: {e}")

        return self._read_tu_berlin_eeg(folder_path, **kwargs)

    def _read_tu_berlin_eeg(self, folder_path, **kwargs):
        """读取 TU Berlin EEG 数据"""
        raw_data = read_tu_berlin_bci_eeg_data(folder_path)

        first_key = list(raw_data.keys())[0]
        eeg_data = raw_data[first_key]

        result_data = {
            'data': np.array(eeg_data['data']).T,
            'sampling_rate': float(eeg_data['sampling_rate']),
            'channel_names': eeg_data['channel_names'],
            'unit': 'uV',
            'time': eeg_data.get('time', [])
        }

        meta = {
            'device': 'TU Berlin BCI EEG',
            'dataset': 'TU Berlin BCI',
            'original_format': 'MATLAB',
            'n_channels': eeg_data['num_channels'],
            'locations': eeg_data.get('locations', {}),
            'title': eeg_data.get('title', ''),
            'montage': eeg_data.get('montage', '')
        }

        events = eeg_data.get('events', [])

        return result_data, meta, events

    def _read_tu_berlin_nirs(self, folder_path, **kwargs):
        """读取 TU Berlin fNIRS 数据"""
        raw_data = read_tu_berlin_bci_nirs_data(folder_path)

        first_key = list(raw_data.keys())[0]
        nirs_data = raw_data[first_key]

        if 'interleaved' not in kwargs or kwargs.get('interleaved', False):
            nirs_data = interleave_channels(nirs_data)

        nirs_data = filter_optode_positions(nirs_data)

        result_data = {
            'data': np.array(nirs_data['data']).T,
            'sampling_rate': float(nirs_data['sampling_rate']),
            'channel_names': nirs_data['channel_names'],
            'unit': 'optical_density',
            'wavelengths': nirs_data.get('wavelengths', []),
            'source_detector_pairs': nirs_data.get('source_detector_pairs', [])
        }

        meta = {
            'device': 'TU Berlin BCI fNIRS',
            'dataset': 'TU Berlin BCI',
            'original_format': 'MATLAB',
            'n_channels': nirs_data['num_channels'],
            'source_positions': nirs_data.get('source_positions_3d', []),
            'detector_positions': nirs_data.get('detector_positions_3d', []),
            'landmark_labels': nirs_data.get('landmark_labels', []),
            'title': nirs_data.get('title', '')
        }

        events = nirs_data.get('events', [])

        return result_data, meta, events

    # ---------- 以下为具体格式的读取适配器 ----------

    def _read_bdf_wrapper(self, file_path, **kwargs):
        """包装读取 BDF 格式"""
        single_file = file_path[0] if isinstance(file_path, list) else file_path

        if 'minilab' in str(single_file).lower():
            logger.info("使用 MiniLab BDF 读取器")
            data_from_reader = read_minilab_bdf(single_file, **kwargs)
        else:
            logger.info("使用通用 BDF 读取器")
            data_from_reader = read_minilab_bdf(single_file, **kwargs)

        if 'data' in data_from_reader and isinstance(data_from_reader['data'], list):
            raw_data = {
                'data': np.array(data_from_reader['data']),
                'sampling_rate': data_from_reader.get('sampling_rate', 256),
                'channel_names': data_from_reader.get('channel_names', []),
                'unit': 'uV'
            }
        else:
            raw_data = {
                'data': np.array(data_from_reader.get('data')),
                'sampling_rate': data_from_reader.get('sampling_rate'),
                'channel_names': data_from_reader.get('channel_names', []),
                'unit': 'uV'
            }

        meta = {
            'device': 'MiniLab' if 'minilab' in str(file_path).lower() else 'Biosemi/Neuracle',
            'n_channels': data_from_reader.get('num_channels', 0),
            'original_format': 'BDF',
            'signal_type': data_from_reader.get('signal_type', 'eeg')
        }
        events = data_from_reader.get('events', [])
        return raw_data, meta, events

    def _read_snirf_wrapper(self, file_path, **kwargs):
        """包装读取 SNIRF 格式"""
        single_file = file_path[0] if isinstance(file_path, list) else file_path

        if 'minilab' in str(single_file).lower():
            logger.info("使用 MiniLab SNIRF 读取器")
            data_from_reader = read_minilab_snirf(single_file, **kwargs)
        else:
            if SNIRF_AVAILABLE:
                logger.info("使用通用 SNIRF 读取器")
                data_from_reader = read_minilab_snirf(single_file, **kwargs)
            else:
                raise ImportError("需要安装 snirf 库：pip install snirf")

        raw_data = {
            'data': data_from_reader.get('data'),
            'sampling_rate': 1.0 / np.mean(np.diff(data_from_reader.get('time', [0, 1]))) if 'time' in data_from_reader else 10.0,
            'channel_names': data_from_reader.get('channel_names', []),
            'unit': 'optical_density',
            'wavelengths': data_from_reader.get('wavelengths', []),
            'source_detector_pairs': data_from_reader.get('source_detector_pairs', [])
        }

        meta = {
            'device': 'MiniLab or SNIRF-compatible',
            'n_channels': data_from_reader.get('num_channels', 0),
            'wavelengths': data_from_reader.get('wavelengths', []),
            'original_format': 'SNIRF',
            'locations': data_from_reader.get('locations', {})
        }

        events = data_from_reader.get('events', [])
        return raw_data, meta, events

    def _read_edf_wrapper(self, file_path, **kwargs):
        """包装读取 EDF/EDF+ 格式"""
        try:
            single_file = file_path[0] if isinstance(file_path, list) else file_path
            edf_file = pyedflib.EdfReader(single_file)
            n_channels = edf_file.signals_in_file
            data = []

            for i in range(n_channels):
                signal = edf_file.readSignal(i, digital=False)
                data.append(signal)

            events = []
            annotations = edf_file.readAnnotations()
            for annotation in annotations:
                time, duration, label = annotation
                events.append([float(time), float(duration), str(label)])

            raw_data = {
                'data': np.array(data),
                'sampling_rate': edf_file.getSampleFrequency(0),
                'channel_names': edf_file.getSignalLabels(),
                'unit': 'uV'
            }

            meta = {
                'device': 'EDF-compatible',
                'n_channels': n_channels,
                'original_format': 'EDF'
            }

            edf_file.close()
            return raw_data, meta, events

        except Exception as e:
            raise Exception(f"读取 EDF 文件失败: {e}")

    def _read_mat_wrapper(self, file_path, **kwargs):
        """读取 MATLAB .mat 文件"""
        single_file = file_path[0] if isinstance(file_path, list) else file_path
        data = scipy.io.loadmat(single_file)

        raw_data, meta, events = None, {}, []

        if 'data' in data and 'sampling_rate' in data:
            raw_data = {
                'data': data['data'],
                'sampling_rate': float(data['sampling_rate'][0][0]),
                'channel_names': [f'Ch{i+1}' for i in range(data['data'].shape[0])],
                'unit': 'unknown'
            }
            meta = {
                'device': 'MATLAB-generated',
                'n_channels': data['data'].shape[0],
                'original_format': 'MATLAB'
            }

        return raw_data, meta, events

    def _read_csv_wrapper(self, file_path, **kwargs):
        """读取 CSV/TXT 文件"""
        single_file = file_path[0] if isinstance(file_path, list) else file_path
        df = pd.read_csv(single_file, **kwargs.get('csv_args', {}))

        data_columns = kwargs.get('data_columns', df.columns[:-1])
        time_column = kwargs.get('time_column', df.columns[-1])

        raw_data = {
            'data': df[data_columns].values.T,
            'sampling_rate': kwargs.get('sampling_rate', 1.0),
            'channel_names': list(data_columns),
            'unit': kwargs.get('unit', 'unknown')
        }

        meta = {
            'device': 'CSV/TXT file',
            'n_channels': len(data_columns),
            'original_format': 'CSV'
        }

        return raw_data, meta, []

    def _read_excel_wrapper(self, file_path, **kwargs):
        """读取 Excel 文件"""
        single_file = file_path[0] if isinstance(file_path, list) else file_path
        df = pd.read_excel(single_file, **kwargs.get('excel_args', {}))

        return self._read_csv_wrapper([df], **kwargs)

    def _read_acqknowledge_wrapper(self, file_path, **kwargs):
        """读取 BIOPAC AcqKnowledge 文件"""
        if not BIOREAD_AVAILABLE:
            raise ImportError("读取 .acq 文件需要 'bioread' 库。请运行 `pip install bioread`。")

        try:
            single_file = file_path[0] if isinstance(file_path, list) else file_path
            data = bioread.read(single_file)

            channels_data = []
            channel_names = []

            for channel in data.channels:
                channels_data.append(channel.data)
                channel_names.append(channel.name)

            raw_data = {
                'data': np.array(channels_data),
                'sampling_rate': data.samples_per_second,
                'channel_names': channel_names,
                'unit': 'V'
            }

            meta = {
                'device': 'BIOPAC AcqKnowledge',
                'n_channels': len(channels_data),
                'original_format': 'ACQ'
            }

            return raw_data, meta, []

        except Exception as e:
            raise Exception(f"读取 BIOPAC 文件失败: {e}")

    def _read_brainvision_wrapper(self, file_path, **kwargs):
        """读取 BrainVision 格式"""
        if not MNE_AVAILABLE:
            raise ImportError("读取 BrainVision 文件需要 'mne' 库。")

        try:
            raw = mne.io.read_raw_brainvision(file_path, preload=True)

            raw_data = {
                'data': raw.get_data(),
                'sampling_rate': raw.info['sfreq'],
                'channel_names': raw.ch_names,
                'unit': 'V'
            }

            meta = {
                'device': 'BrainVision',
                'n_channels': len(raw.ch_names),
                'original_format': 'BrainVision'
            }

            events, event_dict = mne.events_from_annotations(raw)
            events_list = []
            for event in events:
                events_list.append([event[0]/raw.info['sfreq'], 0, event[2]])

            return raw_data, meta, events_list

        except Exception as e:
            raise Exception(f"读取 BrainVision 文件失败: {e}")

    def _read_eeglab_wrapper(self, file_path, **kwargs):
        """读取 EEGLAB .set 文件"""
        if not MNE_AVAILABLE:
            raise ImportError("读取 EEGLAB 文件需要 'mne' 库。")

        try:
            raw = mne.io.read_raw_eeglab(file_path, preload=True)

            raw_data = {
                'data': raw.get_data(),
                'sampling_rate': raw.info['sfreq'],
                'channel_names': raw.ch_names,
                'unit': 'V'
            }

            meta = {
                'device': 'EEGLAB',
                'n_channels': len(raw.ch_names),
                'original_format': 'EEGLAB'
            }

            return raw_data, meta, []

        except Exception as e:
            raise Exception(f"读取 EEGLAB 文件失败: {e}")

    def _read_generic(self, file_path, **kwargs):
        """通用后备读取方法"""
        logger.warning(f"使用通用方法读取文件: {file_path}")

        # 尝试基于文件扩展名猜测格式
        file_ext = Path(file_path[0] if isinstance(file_path, list) else file_path).suffix.lower()

        if file_ext == '.txt' or file_ext == '.csv':
            return self._read_csv_wrapper(file_path, **kwargs)
        elif file_ext == '.xlsx':
            return self._read_excel_wrapper(file_path, **kwargs)
        elif file_ext == '.mat':
            return self._read_mat_wrapper(file_path, **kwargs)

        return None, {}, []

# ====================== 6. 格式转换功能 ======================

def convert_to_standard_datadict(source_path, output_path=None, target_format='standard', **kwargs):
    """
    将任意格式数据转换为标准 data_dict 格式
    """
    reader = BioSignalReader()

    # 1. 读取数据
    data_dict = reader.read(source_path, **kwargs)

    # 2. 如果指定了输出路径，保存为相应格式
    if output_path:
        if target_format == 'standard' or target_format == 'json':
            save_datadict(data_dict, output_path)
        elif target_format == 'bdf':
            _convert_to_bdf(data_dict, output_path)
        elif target_format == 'snirf':
            _convert_to_snirf(data_dict, output_path)

    return data_dict

def _convert_to_bdf(data_dict, output_path):
    """将标准 data_dict 转换为 BDF 格式"""
    if 'EEG' not in data_dict.get('signal', {}):
        raise ValueError("数据字典中没有 EEG 信号，无法转换为 BDF")

    eeg_signal = data_dict['signal']['EEG']

    annotations = []
    if 'event' in data_dict:
        events = data_dict['event']
        if 'event_time' in events:
            for i in range(len(events['event_time'])):
                duration = events.get('duration', [0.0]*len(events['event_time']))[i] if 'duration' in events else 0.0
                label = events['event_label'][i] if 'event_label' in events else str(i+1)
                annotations.append([
                    events['event_time'][i],
                    duration,
                    str(label)
                ])

    create_standard_bdf_file(
        file_name=output_path,
        num_channels=len(eeg_signal['channel_names']),
        signals=eeg_signal['data'] * 1000000,
        channel_names=eeg_signal['channel_names'],
        sampling_frequency=eeg_signal['sampling_rate'],
        annotations=annotations
    )
    logger.info(f"已保存 BDF 文件: {output_path}")

def _convert_to_snirf(data_dict, output_path):
    """将标准 data_dict 转换为 SNIRF 格式"""
    if 'FNIRS' not in data_dict.get('signal', {}):
        raise ValueError("数据字典中没有 fNIRS 信号，无法转换为 SNIRF")

    fnirs_signal = data_dict['signal']['FNIRS']

    time_points = np.arange(fnirs_signal['data'].shape[1]) / fnirs_signal['sampling_rate']

    measurement_lists = []
    if 'source_detector_pairs' in fnirs_signal:
        for i, pair in enumerate(fnirs_signal['source_detector_pairs']):
            for j, wavelength in enumerate(fnirs_signal.get('wavelengths', [760, 850])):
                measurement_lists.append({
                    'sourceIndex': pair[0],
                    'detectorIndex': pair[1],
                    'wavelengthIndex': j + 1,
                    'dataType': 1,
                    'dataTypeIndex': 1,
                })

    stim_lists = []
    if 'event' in data_dict:
        events = data_dict['event']
        unique_labels = set(events.get('event_label', []))

        for label in unique_labels:
            stim_data = []
            for i in range(len(events.get('event_time', []))):
                if events['event_label'][i] == label:
                    stim_data.append([
                        events['event_time'][i],
                        events.get('duration', [0.0]*len(events['event_time']))[i],
                        float(label) if label.isdigit() else 1.0
                    ])

            if stim_data:
                stim_lists.append({
                    'name': str(label),
                    'data': stim_data
                })

    create_snirf_file_integrated(
        filename=output_path,
        data_time_series=fnirs_signal['data'].T,
        time_points=time_points,
        measurement_lists=measurement_lists,
        wavelengths=np.array(fnirs_signal.get('wavelengths', [760, 850])),
        stim_lists=stim_lists,
        metadata={
            'SubjectID': data_dict['meta'].get('subject_id', 'Unknown'),
            'Task': data_dict['meta'].get('task', 'Unknown')
        }
    )

    logger.info(f"已保存 SNIRF 文件: {output_path}")

# ====================== 7. 工厂函数与快捷方式 ======================

def read_biosignal(file_path, **kwargs):
    """
    万能读取的快捷函数。
    """
    reader = BioSignalReader()
    return reader.read(file_path, **kwargs)

def load_saved_datadict(json_file_path: str) -> Dict[str, Any]:
    """加载之前保存为 JSON 的标准化 data_dict。"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    if 'signal' in data:
        for mod in data['signal'].values():
            if 'data' in mod and isinstance(mod['data'], list):
                mod['data'] = np.array(mod['data'])
    return data

def save_datadict(data_dict: Dict[str, Any], json_file_path: str):
    """将 data_dict 保存为 JSON 文件。"""
    save_dict = json.loads(json.dumps(data_dict, default=_json_serializer))
    with open(json_file_path, 'w') as f:
        json.dump(save_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"data_dict 已保存至: {json_file_path}")

def _json_serializer(obj):
    """JSON序列化辅助函数。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

# ====================== 8. 使用示例与测试 ======================

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
    eeg_data = np.random.randn(3, 1000) * 50
    builder.add_signal(test_dict, "EEG", eeg_data,
                       sampling_rate=1000,
                       channel_names=["Fz", "Cz", "Pz"],
                       unit="uV",
                       reference="Cz")

    # 添加EMG信号
    emg_data = np.random.randn(3, 2000) * 100
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

    # 保存测试数据
    save_datadict(test_dict, "test_data_dict.json")
    print("   测试数据已保存为 test_data_dict.json")

    # 示例2：演示使用快捷函数读取
    print("\n2. 演示文件读取流程...")
    print("   请根据实际情况测试以下功能:")
    print("   - read_biosignal('path/to/data.bdf', modality_hint='EEG')")
    print("   - read_biosignal('path/to/data.snirf', modality_hint='FNIRS')")
    print("   - read_biosignal('path/to/tu_berlin_folder/', subject_id='S01')")
    print("   - convert_to_standard_datadict('input.bdf', 'output.json')")

    print("\n=== 测试完成 ===")