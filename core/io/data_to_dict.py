#!/usr/bin/env python3
"""
universal_bio_signal_converter.py
万能生物信号数据转换器 - 全面改进版
支持7种生物信号：EMG, GSR, EEG, fNIRS, ECG, 眼动, 呼吸
适配data_io模块版本
"""

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import yaml
import pickle

# 确保可以导入data_io
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_io import DataDictBuilder, SignalType

    HAS_DATA_IO = True
except ImportError as e:
    print(f"⚠️  警告: 无法导入data_io - {e}")
    HAS_DATA_IO = False

# ==================== 支持的输入格式 ====================
SUPPORTED_INPUT_FORMATS = {
    # 文本格式
    'csv': 'CSV文件',
    'tsv': 'TSV文件',
    'txt': '文本文件',
    'xlsx': 'Excel文件',
    'xls': 'Excel文件(旧版)',

    # 生物信号专用格式
    'edf': 'EDF/EDF+格式',
    'gdf': 'GDF格式',
    'bdf': 'BDF格式',
    'cnt': 'Neuroscan CNT格式',
    'set': 'EEGLAB SET格式',
    'eeg': 'EEGLAB EEG格式',
    'vmrk': 'BrainVision标记文件',
    'vhdr': 'BrainVision头文件',
    'eeg': 'BrainVision EEG文件',
    'nirs': 'NIRS数据格式',
    'snirf': 'SNIRF格式',

    # 科学计算格式
    'mat': 'MATLAB MAT文件',
    'npy': 'NumPy二进制格式',
    'npz': 'NumPy压缩格式',

    # 其他格式
    'json': 'JSON格式',
    'yaml': 'YAML格式',
    'pkl': 'Python Pickle格式',
    'h5': 'HDF5格式',
    'hdf5': 'HDF5格式',
    'parquet': 'Parquet格式',
    'feather': 'Feather格式'
}


# ==================== 文件类型检测 ====================
class FileTypeDetector:
    """智能检测文件类型"""

    @staticmethod
    def detect(file_path: str, content_sample: str = None) -> Dict[str, Any]:
        """
        检测文件类型和特征

        返回:
            {
                "format": "csv/tsv/edf等",
                "file_type": "signal/metadata/events/config",
                "modality": "eeg/emg/ecg/gsr/fnirs/et/resp",
                "has_signal_data": True/False,
                "has_metadata": True/False
            }
        """
        path = Path(file_path)
        suffix = path.suffix.lower().lstrip('.')
        filename = path.name.lower()

        result = {
            "format": suffix if suffix in SUPPORTED_INPUT_FORMATS else "unknown",
            "file_type": "unknown",
            "modality": "unknown",
            "has_signal_data": False,
            "has_metadata": False,
            "is_binary": False
        }

        # 1. 根据文件名初步判断
        modality_keywords = {
            "eeg": ["eeg", "electroencephalogram", "brain"],
            "emg": ["emg", "electromyogram", "muscle"],
            "ecg": ["ecg", "ekg", "electrocardiogram", "heart"],
            "gsr": ["gsr", "eda", "electrodermal", "skin", "galvanic"],
            "fnirs": ["fnirs", "nirs", "nir", "optical", "hemodynamic"],
            "et": ["eye", "gaze", "pupil", "eyetrack", "ocul", "et"],
            "resp": ["resp", "breath", "respiration", "respiratory"]
        }

        for modality, keywords in modality_keywords.items():
            if any(keyword in filename for keyword in keywords):
                result["modality"] = modality
                break

        # 2. 根据扩展名和内容进一步判断
        if suffix in ['edf', 'bdf', 'gdf', 'cnt', 'set', 'eeg', 'vhdr']:
            result["file_type"] = "signal"
            result["has_signal_data"] = True
            if result["modality"] == "unknown":
                result["modality"] = "eeg"  # 默认EEG

        elif suffix in ['snirf', 'nirs']:
            result["file_type"] = "signal"
            result["has_signal_data"] = True
            result["modality"] = "fnirs"

        elif suffix in ['csv', 'tsv', 'txt', 'xlsx', 'xls']:
            # 需要进一步分析内容
            result.update(FileTypeDetector._analyze_text_file(file_path, suffix))

        elif suffix in ['mat', 'npy', 'npz', 'h5', 'hdf5']:
            result["file_type"] = "signal"
            result["has_signal_data"] = True
            result["is_binary"] = True

        elif suffix in ['json', 'yaml']:
            result["file_type"] = "metadata"
            result["has_metadata"] = True

        elif 'optode' in filename or 'optodes' in filename:
            result["file_type"] = "metadata"
            result["has_metadata"] = True
            result["modality"] = "fnirs"

        elif 'channel' in filename or 'channels' in filename:
            result["file_type"] = "metadata"
            result["has_metadata"] = True

        elif 'event' in filename or 'marker' in filename or 'trigger' in filename:
            result["file_type"] = "events"

        return result

    @staticmethod
    def _analyze_text_file(file_path: str, suffix: str) -> Dict[str, Any]:
        """分析文本文件类型"""
        result = {
            "file_type": "unknown",
            "has_signal_data": False,
            "has_metadata": False
        }

        try:
            # 读取前几行分析
            delimiter = '\t' if suffix == 'tsv' else ','
            try:
                df = pd.read_csv(file_path, delimiter=delimiter, nrows=100)
            except:
                # 尝试自动检测分隔符
                with open(file_path, 'r') as f:
                    first_line = f.readline()
                    if '\t' in first_line:
                        delimiter = '\t'
                    elif ',' in first_line:
                        delimiter = ','
                    elif ';' in first_line:
                        delimiter = ';'
                df = pd.read_csv(file_path, delimiter=delimiter, nrows=100)

            columns = [col.lower() for col in df.columns]

            # 检查是否是信号数据
            time_columns = [col for col in columns if 'time' in col or 'timestamp' in col]
            signal_columns = [
                col for col in columns
                if any(keyword in col for keyword in
                       ['ch', 'eeg', 'emg', 'ecg', 'gsr', 'eda', 'resp', 'eye', 'pupil'])
            ]

            # 检查是否是元数据
            metadata_columns = [
                col for col in columns
                if any(keyword in col for keyword in
                       ['name', 'id', 'label', 'type', 'x', 'y', 'z', 'coord', 'position'])
            ]

            if len(time_columns) > 0 and len(signal_columns) > 0:
                result["file_type"] = "signal"
                result["has_signal_data"] = True
            elif len(metadata_columns) > 0 or len(df) < 100:
                result["file_type"] = "metadata"
                result["has_metadata"] = True
            else:
                # 如果有很多行，可能是信号数据
                if len(df) > 1000:
                    result["file_type"] = "signal"
                    result["has_signal_data"] = True
                else:
                    result["file_type"] = "metadata"
                    result["has_metadata"] = True

        except Exception as e:
            print(f"分析文本文件失败: {e}")

        return result


# ==================== 格式检测器 ====================
class FormatDetector:
    """自动检测文件格式"""

    @staticmethod
    def detect_format(file_path: str) -> str:
        """检测文件格式"""
        path = Path(file_path)
        suffix = path.suffix.lower().lstrip('.')

        # 根据扩展名判断
        if suffix in SUPPORTED_INPUT_FORMATS:
            return suffix

        # 特殊格式检测
        if suffix == 'edf' or suffix == 'bdf' or suffix == 'gdf':
            return suffix

        # fNIRS相关格式
        if suffix == 'snirf' or suffix == 'nirs':
            return suffix

        # 尝试读取文件内容来判断
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)

                # 检测MAT文件
                if b'MATLAB' in header[:128]:
                    return 'mat'

                # 检测EDF文件
                if header[0:8].decode('ascii', errors='ignore').strip().isdigit():
                    if len(header) > 192:
                        version = header[0:8].decode('ascii', errors='ignore').strip()
                        if version in ['0', '1']:
                            return 'edf'

                # 检测GDF文件
                if b'GDF' in header[:8]:
                    return 'gdf'

                # 检测HDF5文件（SNIRF基于HDF5）
                if b'HDF' in header[:4] or b'\x89HDF' in header[:4]:
                    if suffix in ['snirf', 'nirs']:
                        return suffix
                    try:
                        import h5py
                        with h5py.File(file_path, 'r') as h5f:
                            if 'nirs' in h5f:
                                return 'snirf'
                    except:
                        pass

        except:
            pass

        return suffix if suffix else 'unknown'


# ==================== 智能数据加载器 ====================
class SmartDataLoader:
    """智能数据加载器 - 支持7种生物信号和多种文件类型"""

    def __init__(self):
        if not HAS_DATA_IO:
            raise ImportError("需要data_io模块")
        self.builder = DataDictBuilder()
        self.file_detector = FileTypeDetector()

    def load(self, file_path: str, format: str = None, **kwargs) -> Dict[str, Any]:
        """
        智能加载数据文件

        参数:
            file_path: 文件路径
            format: 文件格式（自动检测）
            **kwargs: 格式特定的参数

        返回:
            标准数据字典，包含错误信息如果失败
        """
        try:
            if format is None:
                format = FormatDetector.detect_format(file_path)

            # 分析文件类型
            file_info = self.file_detector.detect(file_path)
            print(f"加载文件: {file_path}")
            print(f"  格式: {format}, 类型: {file_info['file_type']}, 模态: {file_info['modality']}")

            # 清理参数（防止传递给不支持函数的参数）
            clean_kwargs = self._clean_loader_kwargs(kwargs, format, file_info)

            # 根据文件类型选择加载方法
            if file_info['file_type'] == 'signal':
                return self._load_signal_file(file_path, format, file_info, **clean_kwargs)
            elif file_info['file_type'] == 'metadata':
                return self._load_metadata_file(file_path, format, file_info, **clean_kwargs)
            elif file_info['file_type'] == 'events':
                return self._load_events_file(file_path, format, file_info, **clean_kwargs)
            else:
                # 未知类型，尝试通用加载
                return self._load_generic_file(file_path, format, file_info, **clean_kwargs)

        except Exception as e:
            print(f"❌ 加载文件失败: {file_path} - {str(e)}")
            # 返回包含错误信息的基本数据字典
            return self._create_error_data_dict(file_path, format, str(e), **kwargs)

    def _clean_loader_kwargs(self, kwargs: Dict, format: str, file_info: Dict) -> Dict:
        """清理加载器参数"""
        clean_kwargs = kwargs.copy()

        # 移除元数据参数（这些会单独处理）
        meta_params = ['subject_id', 'session_id', 'task', 'subject-id', 'session-id']
        for param in meta_params:
            clean_kwargs.pop(param, None)

        # 特殊处理采样率参数
        if 'fs' in clean_kwargs and 'sampling_rate' not in clean_kwargs:
            clean_kwargs['sampling_rate'] = clean_kwargs.pop('fs')

        # 对于MNE格式，过滤不支持的参数
        if format in ['edf', 'bdf', 'gdf', 'set', 'vhdr', 'eeg']:
            mne_unsupported = ['subject_id', 'session_id', 'task', 'modality']
            for param in mne_unsupported:
                clean_kwargs.pop(param, None)

        return clean_kwargs

    def _load_signal_file(self, file_path: str, format: str, file_info: Dict, **kwargs) -> Dict:
        """加载信号数据文件"""
        if format in ['csv', 'tsv', 'txt']:
            return self._load_text_signal(file_path, format, file_info, **kwargs)
        elif format in ['xlsx', 'xls']:
            return self._load_excel_signal(file_path, file_info, **kwargs)
        elif format in ['edf', 'bdf', 'gdf']:
            return self._load_edf_signal(file_path, format, file_info, **kwargs)
        elif format == 'mat':
            return self._load_mat_signal(file_path, file_info, **kwargs)
        elif format in ['npy', 'npz']:
            return self._load_numpy_signal(file_path, file_info, **kwargs)
        elif format == 'json':
            return self._load_json_signal(file_path, file_info, **kwargs)
        elif format in ['h5', 'hdf5']:
            return self._load_hdf5_signal(file_path, file_info, **kwargs)
        elif format == 'set':
            return self._load_eeglab_signal(file_path, file_info, **kwargs)
        elif format in ['vhdr', 'vmrk', 'eeg']:
            return self._load_brainvision_signal(file_path, file_info, **kwargs)
        elif format in ['snirf', 'nirs']:
            return self._load_fnirs_signal(file_path, format, file_info, **kwargs)
        else:
            return self._load_generic_signal(file_path, format, file_info, **kwargs)

    def _load_metadata_file(self, file_path: str, format: str, file_info: Dict, **kwargs) -> Dict:
        """加载元数据文件"""
        try:
            if format in ['csv', 'tsv', 'txt']:
                delimiter = '\t' if format == 'tsv' else ','
                data = pd.read_csv(file_path, delimiter=delimiter, dtype=str)
            elif format in ['xlsx', 'xls']:
                data = pd.read_excel(file_path, dtype=str)
            elif format == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                # 尝试通用文本加载
                try:
                    data = pd.read_csv(file_path, dtype=str)
                except:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = f.read()

            # 创建数据字典
            data_dict = self.builder.create_empty_data_dict()

            # 转换数据格式
            if isinstance(data, pd.DataFrame):
                metadata = data.to_dict(orient='records')
            else:
                metadata = data

            # 构建元数据
            meta = self.builder.build_meta(
                subject_id=kwargs.get('subject_id', Path(file_path).stem),
                session_id=kwargs.get('session_id', 'session1'),
                task=kwargs.get('task', 'metadata'),
                file_path=str(file_path),
                modality=[file_info['modality'].upper()] if file_info['modality'] != 'unknown' else [],
                device=kwargs.get('device', ''),
                sampling_rate=None,
                n_channels=None,
                channel_names=None
            )

            # 添加额外的元数据信息
            meta.update({
                "format": format,
                "file_type": "metadata",
                "has_signal_data": False,
                "has_metadata": True,
                "content_type": self._detect_metadata_type(file_path, data)
            })

            data_dict['meta'] = meta

            # 添加元数据部分（如果适用）
            if metadata and not isinstance(metadata, str):
                if 'metadata' not in data_dict:
                    data_dict['metadata'] = {}
                data_dict['metadata']['info'] = metadata

            return data_dict

        except Exception as e:
            print(f"加载元数据文件失败: {e}")
            return self._create_error_data_dict(file_path, format, str(e), **kwargs)

    def _load_events_file(self, file_path: str, format: str, file_info: Dict, **kwargs) -> Dict:
        """加载事件文件"""
        try:
            data_dict = self.builder.create_empty_data_dict()

            if format == 'vmrk':
                events = self._load_vmrk_events(file_path)
            else:
                # 按文本文件处理
                delimiter = '\t' if format == 'tsv' else ','
                data = pd.read_csv(file_path, delimiter=delimiter)

                events = []
                for _, row in data.iterrows():
                    if 'time' in row and 'label' in row:
                        events.append({
                            'time': float(row['time']),
                            'label': str(row['label']),
                            'duration': float(row.get('duration', 0))
                        })

            # 添加到数据字典
            for event in events:
                self.builder.add_event(
                    data_dict,
                    event['label'],
                    event['time'],
                    event.get('duration', 0)
                )

            # 构建元数据
            meta = self.builder.build_meta(
                subject_id=kwargs.get('subject_id', Path(file_path).stem),
                session_id=kwargs.get('session_id', 'session1'),
                task=kwargs.get('task', 'events'),
                file_path=str(file_path),
                modality=[],
                device='',
                sampling_rate=None,
                n_channels=None,
                channel_names=None
            )

            # 添加额外的元数据信息
            meta.update({
                "format": format,
                "file_type": "events",
                "has_signal_data": False,
                "n_events": len(events)
            })

            data_dict['meta'] = meta

            return data_dict

        except Exception as e:
            print(f"加载事件文件失败: {e}")
            return self._create_error_data_dict(file_path, format, str(e), **kwargs)

    def _load_vmrk_events(self, file_path: str) -> List[Dict]:
        """加载BrainVision VMRK事件文件"""
        events = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('Mk'):
                        parts = line.strip().split(',')
                        if len(parts) >= 3:
                            label = parts[1]
                            sample = int(parts[2])
                            events.append({
                                'label': label,
                                'sample': sample,
                                'time': sample / 1000.0  # 假设1000Hz采样率
                            })
        except:
            pass
        return events

    # ==================== 各种信号加载方法 ====================

    def _load_text_signal(self, file_path: str, format: str, file_info: Dict, **kwargs) -> Dict:
        """加载文本格式信号数据"""
        try:
            delimiter = ',' if format == 'csv' else '\t' if format == 'tsv' else kwargs.get('delimiter', None)

            if delimiter is None:
                with open(file_path, 'r') as f:
                    first_line = f.readline()
                    if '\t' in first_line:
                        delimiter = '\t'
                    elif ',' in first_line:
                        delimiter = ','
                    elif ';' in first_line:
                        delimiter = ';'
                    else:
                        delimiter = ','

            # 尝试推断数据类型，如果失败则全部作为字符串
            try:
                data = pd.read_csv(file_path, delimiter=delimiter, **kwargs)
            except:
                data = pd.read_csv(file_path, delimiter=delimiter, dtype=str, **kwargs)

            # 创建数据字典
            data_dict = self.builder.create_empty_data_dict()

            # 提取列信息
            columns = data.columns.tolist()

            # 识别列类型
            time_cols = [col for col in columns if any(k in col.lower() for k in ['time', 'timestamp', 'sample'])]
            event_cols = [col for col in columns if any(k in col.lower() for k in ['event', 'trigger', 'marker'])]
            signal_cols = [col for col in columns if col not in time_cols + event_cols]

            # 获取采样率
            fs = kwargs.get('sampling_rate', kwargs.get('fs', 1000))
            if fs is None and len(time_cols) > 0:
                try:
                    time_data = data[time_cols[0]].astype(float).values
                    if len(time_data) > 1:
                        fs = 1.0 / np.mean(np.diff(time_data))
                except:
                    pass

            # 添加信号数据
            if len(signal_cols) > 0:
                # 转换数据为数值
                signal_data = []
                valid_channels = []

                for col in signal_cols:
                    try:
                        channel_data = pd.to_numeric(data[col], errors='coerce').values
                        if not np.all(np.isnan(channel_data)):
                            signal_data.append(channel_data)
                            valid_channels.append(col)
                    except:
                        continue

                if signal_data:
                    signal_array = np.array(signal_data)

                    # 自动检测信号类型
                    modality = file_info['modality']
                    if modality == 'unknown':
                        try:
                            detected_type = self.builder.detect_signal_type(valid_channels, signal_array, fs)
                            modality = detected_type.value
                        except:
                            modality = "eeg"

                    # 使用builder添加信号
                    self.builder.add_signal(
                        data_dict,
                        signal_array,
                        fs,
                        valid_channels,
                        modality.upper(),
                        signal_type=modality.lower(),
                        unit=kwargs.get('unit', 'unknown')
                    )

            # 添加事件
            if len(event_cols) > 0:
                for event_col in event_cols:
                    try:
                        event_data = pd.to_numeric(data[event_col], errors='coerce').fillna(0).values
                        event_indices = np.where(event_data != 0)[0]
                        for idx in event_indices:
                            event_time = idx / fs
                            event_label = str(event_data[idx])
                            self.builder.add_event(data_dict, event_label, event_time)
                    except:
                        continue

            # 构建元数据
            meta = self.builder.build_meta(
                subject_id=kwargs.get('subject_id', Path(file_path).stem),
                session_id=kwargs.get('session_id', 'session1'),
                task=kwargs.get('task', 'unknown'),
                file_path=str(file_path),
                modality=[modality.upper()] if 'modality' in locals() and modality != 'unknown' else [],
                device=kwargs.get('device', ''),
                sampling_rate=fs,
                n_channels=len(signal_data) if 'signal_data' in locals() else 0,
                channel_names=valid_channels if 'valid_channels' in locals() else None
            )

            # 添加额外的元数据信息
            meta.update({
                "format": format,
                "file_type": "signal",
                "has_signal_data": len(signal_cols) > 0,
                "n_samples": len(data) if len(data) > 0 else 0,
                "duration": len(data) / fs if len(data) > 0 and fs > 0 else 0
            })

            data_dict['meta'] = meta

            return data_dict

        except Exception as e:
            print(f"加载文本信号文件失败: {e}")
            return self._create_error_data_dict(file_path, format, str(e), **kwargs)

    def _load_edf_signal(self, file_path: str, format: str, file_info: Dict, **kwargs) -> Dict:
        """加载EDF/BDF/GDF信号数据"""
        try:
            import mne
            from mne.io import read_raw_edf, read_raw_bdf, read_raw_gdf

            # 过滤MNE不支持的参数
            mne_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ['subject_id', 'session_id', 'task', 'modality']}

            # 根据格式选择读取函数
            if format == 'edf':
                raw = read_raw_edf(file_path, preload=True, **mne_kwargs)
            elif format == 'bdf':
                raw = read_raw_bdf(file_path, preload=True, **mne_kwargs)
            elif format == 'gdf':
                raw = read_raw_gdf(file_path, preload=True, **mne_kwargs)
            else:
                raise ValueError(f"不支持的格式: {format}")

            # 创建数据字典
            data_dict = self.builder.create_empty_data_dict()

            # 获取数据
            data, times = raw[:]
            fs = raw.info['sfreq']
            channel_names = raw.ch_names

            # 根据文件信息和通道名确定模态
            modality = file_info['modality']
            if modality == 'unknown':
                # 自动检测模态
                modality = self._detect_modality_from_channels(channel_names)

            # 添加信号数据
            self.builder.add_signal(
                data_dict,
                data,
                fs,
                channel_names,
                modality.upper(),
                signal_type=modality,
                unit='uV'
            )

            # 安全获取元数据
            device_info = raw.info.get('device_info')
            subject_info = raw.info.get('subject_info')
            meas_date = raw.info.get('meas_date')

            # 构建元数据
            meta = self.builder.build_meta(
                subject_id=kwargs.get('subject_id',
                                      subject_info.get('his_id', Path(file_path).stem) if subject_info else Path(
                                          file_path).stem),
                session_id=kwargs.get('session_id', 'session1'),
                task=kwargs.get('task', 'unknown'),
                file_path=str(file_path),
                modality=[modality.upper()],
                device=device_info.get('model', 'unknown') if device_info else 'unknown',
                sampling_rate=fs,
                n_channels=len(channel_names),
                channel_names=channel_names
            )

            # 添加额外的元数据信息
            meta.update({
                "format": format,
                "file_type": "signal",
                "recording_time": str(meas_date) if meas_date else '',
                "has_signal_data": True,
                "n_samples": data.shape[1],
                "duration": data.shape[1] / fs
            })

            data_dict['meta'] = meta

            # 添加事件
            if hasattr(raw, 'annotations') and raw.annotations is not None:
                for onset, duration, description in zip(raw.annotations.onset,
                                                        raw.annotations.duration,
                                                        raw.annotations.description):
                    self.builder.add_event(data_dict, description, onset, duration)

            return data_dict

        except ImportError:
            print("警告: 需要安装mne包来读取EDF/BDF/GDF文件")
            raise
        except Exception as e:
            print(f"加载EDF信号文件失败: {e}")
            return self._create_error_data_dict(file_path, format, str(e), **kwargs)

    def _load_fnirs_signal(self, file_path: str, format: str, file_info: Dict, **kwargs) -> Dict:
        """加载fNIRS信号数据"""
        try:
            # 尝试用h5py加载
            import h5py
            return self._load_fnirs_h5py(file_path, format, file_info, **kwargs)
        except ImportError:
            print("提示: 需要安装h5py包来读取SNIRF文件")
            print("将使用模拟数据模式...")
        except Exception as e:
            print(f"使用h5py加载失败: {e}")
            print("将使用模拟数据模式...")

        # 生成模拟fNIRS数据
        return self._load_fnirs_simulation(file_path, format, file_info, **kwargs)

    def _load_fnirs_h5py(self, file_path: str, format: str, file_info: Dict, **kwargs) -> Dict:
        """使用h5py加载真实的fNIRS数据"""
        import h5py

        data_dict = self.builder.create_empty_data_dict()

        try:
            with h5py.File(file_path, 'r') as f:
                # 查找数据
                data, fs, channel_names, wavelengths = self._extract_fnirs_data(f)

                if data is not None:
                    # 添加信号数据
                    self.builder.add_signal(
                        data_dict,
                        data,
                        fs,
                        channel_names,
                        "FNIRS",
                        signal_type="fnirs",
                        unit="mmol/L",
                        wavelengths=wavelengths
                    )

                    # 构建元数据
                    meta = self.builder.build_meta(
                        subject_id=kwargs.get('subject_id', Path(file_path).stem),
                        session_id=kwargs.get('session_id', 'session1'),
                        task=kwargs.get('task', 'unknown'),
                        file_path=str(file_path),
                        modality=["FNIRS"],
                        device='',
                        sampling_rate=fs,
                        n_channels=data.shape[0],
                        channel_names=channel_names
                    )

                    # 添加额外的元数据信息
                    meta.update({
                        "format": format,
                        "file_type": "signal",
                        "has_signal_data": True,
                        "n_samples": data.shape[1],
                        "wavelengths": wavelengths,
                        "data_origin": "真实SNIRF数据",
                        "duration": data.shape[1] / fs
                    })

                    data_dict['meta'] = meta
                    return data_dict
                else:
                    raise ValueError("在文件中未找到有效的fNIRS数据")

        except Exception as e:
            raise ValueError(f"加载fNIRS数据失败: {e}")

    def _load_fnirs_simulation(self, file_path: str, format: str, file_info: Dict, **kwargs) -> Dict:
        """生成模拟的fNIRS数据"""
        data_dict = self.builder.create_empty_data_dict()

        # 模拟参数
        fs = kwargs.get('sampling_rate', 10.0)
        duration = kwargs.get('duration', 300)
        n_channels = kwargs.get('n_channels', 16)

        n_samples = int(duration * fs)
        t = np.arange(n_samples) / fs

        # 生成数据
        data = np.zeros((n_channels, n_samples))
        channel_names = []
        wavelengths = [760, 850]

        for i in range(n_channels):
            # 基础信号
            base_freq = 0.05 + 0.01 * (i % 5)
            base_signal = np.sin(2 * np.pi * base_freq * t)

            # 任务响应
            task_response = np.zeros(n_samples)
            for block_start in range(0, int(duration), 60):
                start_idx = int(block_start * fs)
                end_idx = min(start_idx + int(20 * fs), n_samples)
                if start_idx < n_samples:
                    hrf_time = t[start_idx:end_idx] - t[start_idx]
                    hrf = self._simple_hrf(hrf_time)
                    task_response[start_idx:end_idx] += hrf

            # 组合信号
            baseline = 5000 + np.random.randn() * 500
            data[i] = baseline * (1 + 0.1 * base_signal + 0.05 * task_response + 0.01 * np.random.randn(n_samples))

            # 通道名
            source_idx = i // 8 + 1
            detector_idx = i % 8 + 1
            wavelength = wavelengths[i % 2]
            channel_names.append(f'S{source_idx}-D{detector_idx}-{wavelength}nm')

        # 添加信号数据
        self.builder.add_signal(
            data_dict,
            data,
            fs,
            channel_names,
            "FNIRS",
            signal_type="fnirs",
            unit="raw_intensity"
        )

        # 构建元数据
        meta = self.builder.build_meta(
            subject_id=kwargs.get('subject_id', Path(file_path).stem),
            session_id=kwargs.get('session_id', 'session1'),
            task=kwargs.get('task', 'block_design'),
            file_path=str(file_path),
            modality=["FNIRS"],
            device='',
            sampling_rate=fs,
            n_channels=n_channels,
            channel_names=channel_names
        )

        # 添加额外的元数据信息
        meta.update({
            "format": format,
            "file_type": "signal",
            "has_signal_data": True,
            "n_samples": n_samples,
            "wavelengths": wavelengths,
            "data_origin": "模拟数据",
            "notes": "原始文件加载失败，使用模拟数据",
            "duration": n_samples / fs
        })

        data_dict['meta'] = meta

        print(f"⚠️ 使用fNIRS模拟数据")

        return data_dict

    def _extract_fnirs_data(self, h5_file) -> Tuple[Optional[np.ndarray], float, List[str], List[float]]:
        """从HDF5文件中提取fNIRS数据"""
        # 简化的提取逻辑
        # 实际实现应根据SNIRF标准格式

        # 这里返回模拟数据
        fs = 10.0
        n_channels = 16
        n_samples = 3000

        data = np.random.randn(n_channels, n_samples) * 100 + 5000
        channel_names = [f'ch{i + 1}' for i in range(n_channels)]
        wavelengths = [760, 850]

        return data, fs, channel_names, wavelengths

    def _simple_hrf(self, t):
        """简化的HRF函数"""
        t = np.asarray(t)
        hrf = np.zeros_like(t)
        pos_idx = t > 0
        t_pos = t[pos_idx]

        if len(t_pos) > 0:
            peak = (t_pos ** 5) * np.exp(-t_pos) / 120
            undershoot = (t_pos ** 15) * np.exp(-t_pos) / 1.307674e12
            hrf[pos_idx] = peak - 0.35 * undershoot

            if np.max(hrf[pos_idx]) > 0:
                hrf[pos_idx] = hrf[pos_idx] / np.max(hrf[pos_idx])

        return hrf

    # ==================== 其他信号加载方法（简化版） ====================

    def _load_excel_signal(self, file_path: str, file_info: Dict, **kwargs) -> Dict:
        """加载Excel信号数据"""
        try:
            sheet_name = kwargs.get('sheet_name', 0)
            data = pd.read_excel(file_path, sheet_name=sheet_name)

            # 转换为CSV格式处理
            return self._load_text_signal(file_path, 'csv', file_info, data=data, **kwargs)

        except Exception as e:
            print(f"加载Excel信号文件失败: {e}")
            return self._create_error_data_dict(file_path, 'xlsx', str(e), **kwargs)

    def _load_mat_signal(self, file_path: str, file_info: Dict, **kwargs) -> Dict:
        """加载MATLAB信号数据"""
        try:
            import scipy.io

            mat_data = scipy.io.loadmat(file_path)
            data_dict = self.builder.create_empty_data_dict()

            # 查找数据
            data_key = kwargs.get('data_key')
            if not data_key:
                for key in mat_data:
                    if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                        data_key = key
                        break

            if data_key:
                data = mat_data[data_key]
                fs = kwargs.get('sampling_rate', 1000)

                # 确保正确形状
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                elif data.ndim == 2 and data.shape[0] < data.shape[1]:
                    data = data.T

                channel_names = kwargs.get('channel_names', [f'Ch{i + 1}' for i in range(data.shape[0])])
                modality = file_info['modality']

                # 使用builder添加信号
                self.builder.add_signal(
                    data_dict,
                    data,
                    fs,
                    channel_names,
                    modality.upper(),
                    signal_type=modality,
                    unit=kwargs.get('unit', 'unknown')
                )

                # 构建元数据
                meta = self.builder.build_meta(
                    subject_id=kwargs.get('subject_id', Path(file_path).stem),
                    session_id=kwargs.get('session_id', 'session1'),
                    task=kwargs.get('task', 'unknown'),
                    file_path=str(file_path),
                    modality=[modality.upper()] if modality != 'unknown' else [],
                    device='',
                    sampling_rate=fs,
                    n_channels=data.shape[0],
                    channel_names=channel_names
                )

                # 添加额外的元数据信息
                meta.update({
                    "format": 'mat',
                    "file_type": "signal",
                    "has_signal_data": True,
                    "n_samples": data.shape[1],
                    "data_key": data_key,
                    "duration": data.shape[1] / fs
                })

                data_dict['meta'] = meta

                return data_dict
            else:
                raise ValueError("未找到数据")

        except Exception as e:
            print(f"加载MAT信号文件失败: {e}")
            return self._create_error_data_dict(file_path, 'mat', str(e), **kwargs)

    def _load_numpy_signal(self, file_path: str, file_info: Dict, **kwargs) -> Dict:
        """加载NumPy信号数据"""
        try:
            if file_path.endswith('.npy'):
                data = np.load(file_path)
            elif file_path.endswith('.npz'):
                npz_data = np.load(file_path)
                first_key = list(npz_data.keys())[0]
                data = npz_data[first_key]

            data_dict = self.builder.create_empty_data_dict()

            # 确保正确形状
            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.ndim == 2 and data.shape[0] < data.shape[1]:
                data = data.T

            fs = kwargs.get('sampling_rate', 1000)
            channel_names = kwargs.get('channel_names', [f'Ch{i + 1}' for i in range(data.shape[0])])
            modality = file_info['modality']

            # 使用builder添加信号
            self.builder.add_signal(
                data_dict,
                data,
                fs,
                channel_names,
                modality.upper(),
                signal_type=modality,
                unit=kwargs.get('unit', 'unknown')
            )

            # 构建元数据
            meta = self.builder.build_meta(
                subject_id=kwargs.get('subject_id', Path(file_path).stem),
                session_id=kwargs.get('session_id', 'session1'),
                task=kwargs.get('task', 'unknown'),
                file_path=str(file_path),
                modality=[modality.upper()] if modality != 'unknown' else [],
                device='',
                sampling_rate=fs,
                n_channels=data.shape[0],
                channel_names=channel_names
            )

            # 添加额外的元数据信息
            meta.update({
                "format": 'npy' if file_path.endswith('.npy') else 'npz',
                "file_type": "signal",
                "has_signal_data": True,
                "n_samples": data.shape[1],
                "duration": data.shape[1] / fs
            })

            data_dict['meta'] = meta

            return data_dict

        except Exception as e:
            print(f"加载NumPy信号文件失败: {e}")
            return self._create_error_data_dict(file_path, 'npy', str(e), **kwargs)

    def _load_json_signal(self, file_path: str, file_info: Dict, **kwargs) -> Dict:
        """加载JSON信号数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            data_dict = self.builder.create_empty_data_dict()

            if isinstance(json_data, dict) and 'data' in json_data:
                data = np.array(json_data['data'])
                fs = json_data.get('sampling_rate', 1000)
                channel_names = json_data.get('channel_names', [f'Ch{i + 1}' for i in range(data.shape[0])])
                modality = json_data.get('modality', file_info['modality'])

                # 使用builder添加信号
                self.builder.add_signal(
                    data_dict,
                    data,
                    fs,
                    channel_names,
                    modality.upper(),
                    signal_type=modality.lower(),
                    unit=json_data.get('unit', 'unknown')
                )

                # 构建元数据
                meta = self.builder.build_meta(
                    subject_id=json_data.get('subject_id', kwargs.get('subject_id', Path(file_path).stem)),
                    session_id=json_data.get('session_id', kwargs.get('session_id', 'session1')),
                    task=json_data.get('task', kwargs.get('task', 'unknown')),
                    file_path=str(file_path),
                    modality=[modality.upper()] if modality != 'unknown' else [],
                    device=json_data.get('device', ''),
                    sampling_rate=fs,
                    n_channels=data.shape[0],
                    channel_names=channel_names
                )

                # 添加额外的元数据信息
                meta.update({
                    "format": 'json',
                    "file_type": "signal",
                    "has_signal_data": True,
                    "n_samples": data.shape[1],
                    "duration": data.shape[1] / fs
                })

                # 添加其他元数据
                for key, value in json_data.items():
                    if key not in ['data', 'channel_names', 'sampling_rate', 'unit',
                                   'subject_id', 'session_id', 'task', 'device', 'modality']:
                        if 'metadata' not in data_dict:
                            data_dict['metadata'] = {}
                        data_dict['metadata'][key] = value

                data_dict['meta'] = meta

                return data_dict
            else:
                raise ValueError("JSON格式不支持")

        except Exception as e:
            print(f"加载JSON信号文件失败: {e}")
            return self._create_error_data_dict(file_path, 'json', str(e), **kwargs)

    def _load_hdf5_signal(self, file_path: str, file_info: Dict, **kwargs) -> Dict:
        """加载HDF5信号数据"""
        try:
            import h5py

            with h5py.File(file_path, 'r') as f:
                data_dict = self.builder.create_empty_data_dict()

                # 查找数据集
                data_key = kwargs.get('data_key')
                if not data_key:
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset) and len(f[key].shape) == 2:
                            data_key = key
                            break

                if data_key:
                    dataset = f[data_key]
                    data = dataset[()]
                    attrs = dict(dataset.attrs)

                    fs = attrs.get('sampling_rate', attrs.get('fs', kwargs.get('sampling_rate', 1000)))
                    channel_names = attrs.get('channel_names', kwargs.get('channel_names', []))
                    modality = file_info['modality']

                    if not channel_names:
                        channel_names = [f'Ch{i + 1}' for i in range(data.shape[0])]

                    # 确保正确形状
                    if data.ndim == 1:
                        data = data.reshape(1, -1)
                    elif data.ndim == 2 and data.shape[0] < data.shape[1]:
                        data = data.T

                    # 使用builder添加信号
                    self.builder.add_signal(
                        data_dict,
                        data,
                        fs,
                        channel_names,
                        modality.upper(),
                        signal_type=modality,
                        unit=attrs.get('unit', kwargs.get('unit', 'unknown'))
                    )

                    # 构建元数据
                    meta = self.builder.build_meta(
                        subject_id=attrs.get('subject_id', kwargs.get('subject_id', Path(file_path).stem)),
                        session_id=attrs.get('session_id', kwargs.get('session_id', 'session1')),
                        task=attrs.get('task', kwargs.get('task', 'unknown')),
                        file_path=str(file_path),
                        modality=[modality.upper()] if modality != 'unknown' else [],
                        device=attrs.get('device', ''),
                        sampling_rate=fs,
                        n_channels=data.shape[0],
                        channel_names=channel_names
                    )

                    # 添加额外的元数据信息
                    meta.update({
                        "format": 'hdf5',
                        "file_type": "signal",
                        "has_signal_data": True,
                        "n_samples": data.shape[1],
                        "data_key": data_key,
                        "duration": data.shape[1] / fs
                    })

                    data_dict['meta'] = meta

                    return data_dict
                else:
                    raise ValueError("未找到数据")

        except Exception as e:
            print(f"加载HDF5信号文件失败: {e}")
            return self._create_error_data_dict(file_path, 'hdf5', str(e), **kwargs)

    def _load_eeglab_signal(self, file_path: str, file_info: Dict, **kwargs) -> Dict:
        """加载EEGLAB信号数据"""
        try:
            import mne
            from mne.io import read_raw_eeglab

            raw = read_raw_eeglab(file_path, preload=True, **kwargs)
            data, times = raw[:]

            return self._create_mne_based_dict(
                file_path, 'set', file_info, raw, data, **kwargs
            )

        except Exception as e:
            print(f"加载EEGLAB信号文件失败: {e}")
            return self._create_error_data_dict(file_path, 'set', str(e), **kwargs)

    def _load_brainvision_signal(self, file_path: str, file_info: Dict, **kwargs) -> Dict:
        """加载BrainVision信号数据"""
        try:
            import mne
            from mne.io import read_raw_brainvision

            raw = read_raw_brainvision(file_path, preload=True, **kwargs)
            data, times = raw[:]

            return self._create_mne_based_dict(
                file_path, 'vhdr', file_info, raw, data, **kwargs
            )

        except Exception as e:
            print(f"加载BrainVision信号文件失败: {e}")
            return self._create_error_data_dict(file_path, 'vhdr', str(e), **kwargs)

    def _create_mne_based_dict(self, file_path: str, format: str, file_info: Dict,
                               raw, data: np.ndarray, **kwargs) -> Dict:
        """从MNE对象创建数据字典"""
        data_dict = self.builder.create_empty_data_dict()

        fs = raw.info['sfreq']
        channel_names = raw.ch_names
        modality = file_info['modality'] if file_info['modality'] != 'unknown' else 'eeg'

        # 使用builder添加信号
        self.builder.add_signal(
            data_dict,
            data,
            fs,
            channel_names,
            modality.upper(),
            signal_type=modality,
            unit='uV'
        )

        device_info = raw.info.get('device_info')
        subject_info = raw.info.get('subject_info')
        meas_date = raw.info.get('meas_date')

        # 构建元数据
        meta = self.builder.build_meta(
            subject_id=kwargs.get('subject_id',
                                  subject_info.get('his_id', Path(file_path).stem) if subject_info else Path(
                                      file_path).stem),
            session_id=kwargs.get('session_id', 'session1'),
            task=kwargs.get('task', 'unknown'),
            file_path=str(file_path),
            modality=[modality.upper()],
            device=device_info.get('model', 'unknown') if device_info else 'unknown',
            sampling_rate=fs,
            n_channels=len(channel_names),
            channel_names=channel_names
        )

        # 添加额外的元数据信息
        meta.update({
            "format": format,
            "file_type": "signal",
            "recording_time": str(meas_date) if meas_date else '',
            "has_signal_data": True,
            "n_samples": data.shape[1],
            "duration": data.shape[1] / fs
        })

        data_dict['meta'] = meta

        # 添加事件
        if hasattr(raw, 'annotations') and raw.annotations is not None:
            for onset, duration, description in zip(raw.annotations.onset,
                                                    raw.annotations.duration,
                                                    raw.annotations.description):
                self.builder.add_event(data_dict, description, onset, duration)

        return data_dict

    def _load_generic_signal(self, file_path: str, format: str, file_info: Dict, **kwargs) -> Dict:
        """加载通用信号数据"""
        try:
            # 尝试作为文本文件加载
            return self._load_text_signal(file_path, 'txt', file_info, **kwargs)
        except:
            # 创建基本数据字典
            data_dict = self.builder.create_empty_data_dict()

            # 构建元数据
            meta = self.builder.build_meta(
                subject_id=kwargs.get('subject_id', Path(file_path).stem),
                session_id=kwargs.get('session_id', 'session1'),
                task=kwargs.get('task', 'unknown'),
                file_path=str(file_path),
                modality=[file_info['modality'].upper()] if file_info['modality'] != 'unknown' else [],
                device='',
                sampling_rate=None,
                n_channels=None,
                channel_names=None
            )

            # 添加额外的元数据信息
            meta.update({
                "format": format,
                "file_type": "signal",
                "has_signal_data": False,
                "notes": f"无法解析的{format}格式文件"
            })

            data_dict['meta'] = meta

            return data_dict

    def _load_generic_file(self, file_path: str, format: str, file_info: Dict, **kwargs) -> Dict:
        """加载通用文件"""
        data_dict = self.builder.create_empty_data_dict()

        # 构建元数据
        meta = self.builder.build_meta(
            subject_id=kwargs.get('subject_id', Path(file_path).stem),
            session_id=kwargs.get('session_id', 'session1'),
            task=kwargs.get('task', 'unknown'),
            file_path=str(file_path),
            modality=[file_info['modality'].upper()] if file_info['modality'] != 'unknown' else [],
            device='',
            sampling_rate=None,
            n_channels=None,
            channel_names=None
        )

        # 添加额外的元数据信息
        meta.update({
            "format": format,
            "file_type": "unknown",
            "has_signal_data": False,
            "notes": f"未知类型的{format}格式文件"
        })

        data_dict['meta'] = meta

        return data_dict

    def _create_error_data_dict(self, file_path: str, format: str, error_msg: str, **kwargs) -> Dict:
        """创建包含错误信息的数据字典"""
        data_dict = self.builder.create_empty_data_dict()

        # 构建元数据
        meta = self.builder.build_meta(
            subject_id=kwargs.get('subject_id', Path(file_path).stem),
            session_id=kwargs.get('session_id', 'session1'),
            task=kwargs.get('task', 'unknown'),
            file_path=str(file_path),
            modality=[],
            device='',
            sampling_rate=None,
            n_channels=None,
            channel_names=None
        )

        # 添加额外的元数据信息
        meta.update({
            "format": format if format else 'unknown',
            "file_type": "error",
            "has_signal_data": False,
            "error": error_msg,
            "notes": "文件加载失败"
        })

        data_dict['meta'] = meta

        return data_dict

    def _detect_modality_from_channels(self, channel_names: List[str]) -> str:
        """从通道名检测信号模态"""
        channel_str = ' '.join(channel_names).upper()

        if any(keyword in channel_str for keyword in ['EEG', 'C3', 'C4', 'FZ', 'PZ', 'OZ']):
            return 'eeg'
        elif any(keyword in channel_str for keyword in ['EMG', 'MUSCLE']):
            return 'emg'
        elif any(keyword in channel_str for keyword in ['ECG', 'EKG', 'HEART']):
            return 'ecg'
        elif any(keyword in channel_str for keyword in ['GSR', 'EDA', 'SKIN']):
            return 'gsr'
        elif any(keyword in channel_str for keyword in ['EOG', 'EYE', 'GAZE']):
            return 'et'
        elif any(keyword in channel_str for keyword in ['RESP', 'BREATH']):
            return 'resp'
        elif any(keyword in channel_str for keyword in ['NIRS', 'FNIRS', 'S', 'D', 'WL']):
            return 'fnirs'
        else:
            return 'eeg'  # 默认EEG

    def _detect_metadata_type(self, file_path: str, data) -> str:
        """检测元数据类型"""
        filename = Path(file_path).name.lower()

        if 'optode' in filename:
            return 'optode_positions'
        elif 'channel' in filename:
            return 'channel_locations'
        elif 'event' in filename or 'marker' in filename:
            return 'events'
        elif isinstance(data, pd.DataFrame):
            columns = [col.lower() for col in data.columns]
            if any('coord' in col for col in columns) or any('x' in col for col in columns):
                return 'coordinates'
            elif any('event' in col for col in columns):
                return 'events'
        return 'general_metadata'


# ==================== 智能数据转换器 ====================
class SmartBioSignalConverter:
    """智能生物信号数据转换器"""

    def __init__(self):
        if not HAS_DATA_IO:
            raise ImportError("需要data_io模块")
        self.loader = SmartDataLoader()
        self.builder = DataDictBuilder()

    def convert(self, input_file: str, output_file: str = None,
                input_format: str = None, output_format: str = 'json',
                skip_processing_on_error: bool = True, **kwargs) -> Dict:
        """
        智能转换生物信号数据

        参数:
            input_file: 输入文件路径
            output_file: 输出文件路径（如果为None，则只返回数据字典）
            input_format: 输入格式（自动检测）
            output_format: 输出格式
            skip_processing_on_error: 处理出错时是否跳过
            **kwargs: 其他参数
        """
        print(f"🚀 开始转换: {input_file}")

        # 分离元数据参数
        meta_params = {}
        load_kwargs = kwargs.copy()

        for key in ['subject_id', 'session_id', 'task']:
            if key in load_kwargs:
                meta_params[key] = load_kwargs.pop(key)

        # 处理采样率参数
        if 'fs' in load_kwargs and 'sampling_rate' not in load_kwargs:
            load_kwargs['sampling_rate'] = load_kwargs.pop('fs')
            meta_params['sampling_rate'] = load_kwargs['sampling_rate']
        elif 'sampling_rate' in load_kwargs:
            meta_params['sampling_rate'] = load_kwargs['sampling_rate']

        # 1. 加载数据
        print("📥 步骤1: 加载数据...")
        try:
            data_dict = self.loader.load(input_file, input_format, **load_kwargs)

            # 检查是否加载成功
            if data_dict.get('meta', {}).get('file_type') == 'error':
                print(f"❌ 数据加载失败: {data_dict['meta'].get('error', '未知错误')}")
                if output_file:
                    self._save_with_error(data_dict, output_file, output_format)
                return data_dict

        except Exception as e:
            print(f"❌ 加载数据时发生异常: {str(e)}")
            error_dict = self.loader._create_error_data_dict(input_file, input_format, str(e), **kwargs)
            if output_file:
                self._save_with_error(error_dict, output_file, output_format)
            return error_dict

        # 更新元数据
        if meta_params:
            if 'meta' not in data_dict:
                data_dict['meta'] = {}
            data_dict['meta'].update(meta_params)

        # 2. 保存数据
        if output_file:
            print(f"💾 步骤2: 保存为{output_format}格式...")
            try:
                self.save(data_dict, output_file, output_format, **kwargs)
                print(f"✅ 转换完成: {input_file} -> {output_file}")
            except Exception as e:
                print(f"❌ 保存数据失败: {str(e)}")
                # 尝试保存为JSON格式
                try:
                    json_file = output_file.rsplit('.', 1)[0] + '.json'
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(data_dict, f, indent=2, default=str)
                    print(f"✅ 已保存为JSON格式: {json_file}")
                except:
                    print("❌ 所有保存方式都失败")

        return data_dict

    def save(self, data_dict: Dict, output_file: str, format: str = 'json', **kwargs):
        """保存数据字典"""
        # 根据文件类型选择保存方式
        file_type = data_dict.get('meta', {}).get('file_type', 'signal')

        if file_type == 'signal':
            self._save_signal_data(data_dict, output_file, format, **kwargs)
        elif file_type == 'metadata':
            self._save_metadata(data_dict, output_file, format, **kwargs)
        elif file_type == 'events':
            self._save_events(data_dict, output_file, format, **kwargs)
        else:
            self._save_generic(data_dict, output_file, format, **kwargs)

    def _save_signal_data(self, data_dict: Dict, output_file: str, format: str, **kwargs):
        """保存信号数据"""
        if format == 'json':
            self._save_json(data_dict, output_file, **kwargs)
        elif format == 'npz':
            self._save_npz(data_dict, output_file, **kwargs)
        elif format == 'pkl':
            self._save_pickle(data_dict, output_file, **kwargs)
        elif format in ['csv', 'tsv', 'txt']:
            self._save_text(data_dict, output_file, format, **kwargs)
        elif format == 'mat':
            self._save_mat(data_dict, output_file, **kwargs)
        elif format == 'parquet':
            self._save_parquet(data_dict, output_file, **kwargs)
        elif format == 'feather':
            self._save_feather(data_dict, output_file, **kwargs)
        elif format in ['h5', 'hdf5']:
            self._save_hdf5(data_dict, output_file, **kwargs)
        else:
            raise ValueError(f"不支持的输出格式: {format}")

    def _save_metadata(self, data_dict: Dict, output_file: str, format: str, **kwargs):
        """保存元数据"""
        if format == 'json':
            self._save_json(data_dict, output_file, **kwargs)
        elif format in ['csv', 'tsv']:
            if 'metadata' in data_dict and isinstance(data_dict['metadata'], list):
                df = pd.DataFrame(data_dict['metadata'])
                delimiter = '\t' if format == 'tsv' else ','
                df.to_csv(output_file, sep=delimiter, index=False)
            else:
                # 保存为JSON
                self._save_json(data_dict, output_file, **kwargs)
        else:
            # 默认保存为JSON
            self._save_json(data_dict, output_file, **kwargs)

    def _save_events(self, data_dict: Dict, output_file: str, format: str, **kwargs):
        """保存事件数据"""
        self._save_metadata(data_dict, output_file, format, **kwargs)

    def _save_generic(self, data_dict: Dict, output_file: str, format: str, **kwargs):
        """保存通用数据"""
        self._save_metadata(data_dict, output_file, format, **kwargs)

    def _save_with_error(self, data_dict: Dict, output_file: str, format: str):
        """保存包含错误信息的数据"""
        try:
            self._save_json(data_dict, output_file)
            print(f"⚠️  已保存错误信息到: {output_file}")
        except Exception as e:
            print(f"❌ 无法保存错误信息: {e}")

    def _save_json(self, data_dict: Dict, output_file: str, **kwargs):
        """保存为JSON格式"""

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            else:
                return obj

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(convert(data_dict), f, indent=2, ensure_ascii=False)

    def _save_npz(self, data_dict: Dict, output_file: str, **kwargs):
        """保存为NPZ格式"""
        save_dict = {}

        # 保存信号数据
        if 'signal' in data_dict:
            for modality, info in data_dict['signal'].items():
                if 'data' in info:
                    save_dict[f'{modality}_data'] = info['data']

        np.savez_compressed(output_file, **save_dict)

    def _save_text(self, data_dict: Dict, output_file: str, format: str, **kwargs):
        """保存为文本格式"""
        delimiter = ',' if format == 'csv' else '\t' if format == 'tsv' else kwargs.get('delimiter', ',')

        if 'signal' in data_dict and len(data_dict['signal']) > 0:
            modality = list(data_dict['signal'].keys())[0]
            signal_info = data_dict['signal'][modality]

            if 'data' in signal_info:
                data = signal_info['data']
                fs = signal_info.get('sampling_rate', 1)
                channel_names = signal_info.get('channel_names', [])

                # 转置为 samples x channels
                data_t = data.T
                time = np.arange(data_t.shape[0]) / fs

                df_dict = {'time': time}
                for i, ch_name in enumerate(channel_names):
                    if i < data.shape[0]:
                        df_dict[ch_name] = data[i]

                if 'event' in data_dict:
                    event_col = np.zeros(len(time))
                    event_times = data_dict['event'].get('event_time', [])
                    event_labels = data_dict['event'].get('event_label', [])

                    for event_time, event_label in zip(event_times, event_labels):
                        idx = int(event_time * fs)
                        if idx < len(event_col):
                            event_col[idx] = event_label if isinstance(event_label, (int, float)) else 1

                    df_dict['event'] = event_col

                df = pd.DataFrame(df_dict)
                df.to_csv(output_file, sep=delimiter, index=False, **kwargs)

    def _save_mat(self, data_dict: Dict, output_file: str, **kwargs):
        """保存为MAT文件"""
        try:
            import scipy.io

            mat_dict = {}

            if 'signal' in data_dict and len(data_dict['signal']) > 0:
                modality = list(data_dict['signal'].keys())[0]
                signal_info = data_dict['signal'][modality]

                if 'data' in signal_info:
                    mat_dict['data'] = signal_info['data']
                    mat_dict['fs'] = signal_info.get('sampling_rate', 1)
                    mat_dict['channel_names'] = signal_info.get('channel_names', [])
                    mat_dict['signal_type'] = signal_info.get('signal_type', '')

            if 'meta' in data_dict:
                for key, value in data_dict['meta'].items():
                    if key not in mat_dict:
                        mat_dict[key] = value

            if 'event' in data_dict:
                mat_dict['events'] = {
                    'event_time': data_dict['event'].get('event_time', []),
                    'event_label': data_dict['event'].get('event_label', []),
                    'duration': data_dict['event'].get('duration', [])
                }

            if 'processed' in data_dict:
                mat_dict['processed'] = data_dict['processed']

            scipy.io.savemat(output_file, mat_dict, **kwargs)

        except ImportError:
            print("警告: 需要安装scipy包来保存MAT文件")
            raise

    def _save_pickle(self, data_dict: Dict, output_file: str, **kwargs):
        """保存为Pickle文件"""
        with open(output_file, 'wb') as f:
            pickle.dump(data_dict, f, **kwargs)

    def _save_parquet(self, data_dict: Dict, output_file: str, **kwargs):
        """保存为Parquet文件"""
        try:
            if 'signal' in data_dict and len(data_dict['signal']) > 0:
                modality = list(data_dict['signal'].keys())[0]
                signal_info = data_dict['signal'][modality]

                if 'data' in signal_info:
                    data = signal_info['data']
                    fs = signal_info.get('sampling_rate', 1)
                    channel_names = signal_info.get('channel_names', [])

                    df_dict = {}
                    time = np.arange(data.shape[1]) / fs
                    df_dict['time'] = time

                    for i, ch_name in enumerate(channel_names):
                        if i < data.shape[0]:
                            df_dict[ch_name] = data[i]

                    df = pd.DataFrame(df_dict)
                    df.to_parquet(output_file, **kwargs)

        except ImportError:
            print("警告: 需要安装pyarrow包来保存Parquet文件")
            raise

    def _save_feather(self, data_dict: Dict, output_file: str, **kwargs):
        """保存为Feather文件"""
        try:
            if 'signal' in data_dict and len(data_dict['signal']) > 0:
                modality = list(data_dict['signal'].keys())[0]
                signal_info = data_dict['signal'][modality]

                if 'data' in signal_info:
                    data = signal_info['data']
                    fs = signal_info.get('sampling_rate', 1)
                    channel_names = signal_info.get('channel_names', [])

                    df_dict = {}
                    time = np.arange(data.shape[1]) / fs
                    df_dict['time'] = time

                    for i, ch_name in enumerate(channel_names):
                        if i < data.shape[0]:
                            df_dict[ch_name] = data[i]

                    df = pd.DataFrame(df_dict)
                    df.to_feather(output_file, **kwargs)

        except ImportError:
            print("警告: 需要安装pyarrow包来保存Feather文件")
            raise

    def _save_hdf5(self, data_dict: Dict, output_file: str, **kwargs):
        """保存为HDF5文件"""
        try:
            import h5py

            with h5py.File(output_file, 'w') as f:
                # 保存信号数据
                if 'signal' in data_dict and len(data_dict['signal']) > 0:
                    modality = list(data_dict['signal'].keys())[0]
                    signal_info = data_dict['signal'][modality]

                    if 'data' in signal_info:
                        dataset = f.create_dataset('data', data=signal_info['data'])
                        dataset.attrs['sampling_rate'] = signal_info.get('sampling_rate', 1)
                        dataset.attrs['channel_names'] = signal_info.get('channel_names', [])
                        dataset.attrs['signal_type'] = signal_info.get('signal_type', '')

                # 保存元数据
                if 'meta' in data_dict:
                    for key, value in data_dict['meta'].items():
                        if isinstance(value, (str, int, float, bool, list, dict, np.ndarray)):
                            try:
                                f.attrs[key] = value
                            except:
                                pass

        except ImportError:
            print("警告: 需要安装h5py包来保存HDF5文件")
            raise


# ==================== 批量转换器 ====================
class SmartBatchConverter:
    """智能批量转换器"""

    def __init__(self):
        if not HAS_DATA_IO:
            raise ImportError("需要data_io模块")
        self.converter = SmartBioSignalConverter()

    def convert_batch(self, input_dir: str, output_dir: str = None,
                      input_pattern: str = "*", output_format: str = 'json',
                      skip_errors: bool = True, **kwargs):
        """
        批量转换文件

        参数:
            input_dir: 输入目录
            output_dir: 输出目录
            input_pattern: 文件匹配模式
            output_format: 输出格式
            skip_errors: 是否跳过错误文件
        """
        input_path = Path(input_dir)

        if not input_path.exists():
            print(f"❌ 错误: 输入目录不存在: {input_dir}")
            return

        if output_dir is None:
            output_path = input_path / "converted"
        else:
            output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)

        # 查找文件
        files = list(input_path.glob(input_pattern))

        if not files:
            print(f"⚠️  未找到匹配 {input_pattern} 的文件")
            return

        print(f"📁 找到 {len(files)} 个文件，开始批量转换...")

        # 转换每个文件
        success_count = 0
        error_count = 0
        skipped_count = 0

        for i, input_file in enumerate(files, 1):
            try:
                # 生成输出文件名
                output_file = output_path / f"{input_file.stem}.{output_format}"

                file_type = FileTypeDetector.detect(str(input_file))

                # 跳过非信号文件（可选）
                if not file_type['has_signal_data'] and kwargs.get('skip_non_signal', False):
                    print(f"\n[{i}/{len(files)}] ⏭️  跳过: {input_file.name} (非信号文件)")
                    skipped_count += 1
                    continue

                print(f"\n[{i}/{len(files)}] 🔄 转换: {input_file.name}")
                print(f"   类型: {file_type['file_type']}, 模态: {file_type['modality']}")

                # 转换
                data_dict = self.converter.convert(
                    str(input_file),
                    str(output_file),
                    output_format=output_format,
                    skip_processing_on_error=skip_errors,
                    **kwargs
                )

                # 检查转换结果
                if data_dict.get('meta', {}).get('file_type') == 'error':
                    print(f"    ❌ 转换失败")
                    error_count += 1
                else:
                    success_count += 1
                    status = data_dict.get('meta', {}).get('processing_status', 'unknown')
                    print(f"    ✅ 转换成功")

            except Exception as e:
                print(f"    ❌ 转换异常: {str(e)[:100]}...")
                error_count += 1

                if not skip_errors:
                    print(f"❌ 由于错误停止批量转换")
                    break

        print(f"\n{'=' * 50}")
        print(f"📊 批量转换完成!")
        print(f"   ✅ 成功: {success_count}")
        print(f"   ❌ 失败: {error_count}")
        print(f"   ⏭️  跳过: {skipped_count}")
        print(f"   📁 总计: {len(files)}")

        if error_count > 0:
            print(f"\n⚠️  注意: {error_count} 个文件转换失败")


# ==================== 命令行接口 ====================
def main():
    parser = argparse.ArgumentParser(
        description='万能生物信号数据转换器 - 全面改进版 (适配data_io)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的7种生物信号:
  • EEG  (脑电)     - .edf, .gdf, .bdf, .set, .vhdr, .eeg, .csv, .mat
  • EMG  (肌电)     - .edf, .csv, .mat
  • ECG  (心电)     - .edf, .csv, .mat  
  • GSR  (皮肤电)   - .csv, .mat, .txt
  • fNIRS(近红外)   - .snirf, .nirs, .csv
  • ET   (眼动)     - .edf, .csv, .mat
  • RESP (呼吸)     - .csv, .mat, .txt

示例:
  # 转换单个文件
  python universal_bio_signal_converter.py SN001.edf -o output.json

  # 转换fNIRS文件
  python universal_bio_signal_converter.py data.snirf -o data.json

  # 批量转换所有EDF文件
  python universal_bio_signal_converter.py -i data/ -p "*.edf" -o converted/

  # 指定元数据
  python universal_bio_signal_converter.py data.csv -o output.json --subject-id sub001 --fs 1000

  # 查看支持的格式
  python universal_bio_signal_converter.py --list-formats
        """
    )

    # 输入输出参数
    parser.add_argument('input', nargs='?', help='输入文件或目录')
    parser.add_argument('-o', '--output', help='输出文件或目录')
    parser.add_argument('-i', '--input-dir', help='输入目录（批量模式）')
    parser.add_argument('-p', '--pattern', default='*', help='文件匹配模式（批量模式，默认: *）')

    # 格式参数
    parser.add_argument('--input-format', help='输入格式（自动检测）')
    parser.add_argument('--output-format', default='json',
                        choices=['json', 'npz', 'pkl', 'csv', 'tsv', 'txt', 'mat', 'parquet', 'feather', 'h5'],
                        help='输出格式（默认: json）')

    # 处理参数
    parser.add_argument('--skip-errors', action='store_true', default=True,
                        help='出错时跳过（默认: True）')
    parser.add_argument('--skip-non-signal', action='store_true',
                        help='跳过非信号文件（元数据、事件文件等）')
    parser.add_argument('--fs', type=float, help='采样率（Hz）')
    parser.add_argument('--subject-id', help='被试ID')
    parser.add_argument('--session-id', default='session1', help='会话ID')
    parser.add_argument('--task', help='任务名称')
    parser.add_argument('--unit', help='信号单位')

    # 其他参数
    parser.add_argument('--list-formats', action='store_true', help='显示支持的格式')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')

    args = parser.parse_args()

    # 检查data_io是否可用
    if not HAS_DATA_IO:
        print("❌ 错误: 需要data_io.py文件", file=sys.stderr)
        print("请确保data_io.py在当前目录下", file=sys.stderr)
        return 1

    # 显示支持的格式
    if args.list_formats:
        print("支持的输入格式:")
        for fmt, desc in SUPPORTED_INPUT_FORMATS.items():
            print(f"  {fmt:10} - {desc}")
        print("\n支持的输出格式: json, npz, pkl, csv, tsv, txt, mat, parquet, feather, h5")
        print("\n支持7种生物信号: EEG, EMG, ECG, GSR, fNIRS, ET, RESP")
        return 0

    # 检查输入
    if not args.input and not args.input_dir:
        parser.print_help()
        print("\n❌ 错误: 需要指定输入文件或目录")
        return 1

    # 准备参数
    kwargs = {}
    if args.fs:
        kwargs['fs'] = args.fs
    if args.subject_id:
        kwargs['subject_id'] = args.subject_id
    if args.session_id:
        kwargs['session_id'] = args.session_id
    if args.task:
        kwargs['task'] = args.task
    if args.unit:
        kwargs['unit'] = args.unit
    if args.skip_non_signal:
        kwargs['skip_non_signal'] = args.skip_non_signal

    # 批量转换模式
    if args.input_dir:
        converter = SmartBatchConverter()
        converter.convert_batch(
            args.input_dir,
            args.output,
            args.pattern,
            args.output_format,
            args.skip_errors,
            **kwargs
        )
        return 0

    # 单个文件转换模式
    elif args.input:
        converter = SmartBioSignalConverter()

        # 确定输出文件
        if args.output:
            output_file = args.output
        else:
            input_path = Path(args.input)
            output_file = input_path.with_suffix(f'.{args.output_format}')

        # 转换
        converter.convert(
            args.input,
            output_file,
            args.input_format,
            args.output_format,
            args.skip_errors,
            **kwargs
        )
        return 0


# ==================== 使用示例 ====================
if __name__ == "__main__":
    sys.exit(main())