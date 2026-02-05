# -*- coding: utf-8 -*-
"""
通用数据 I/O 模块
支持格式：edf/bdf (mne), mat (scipy.io), csv (pandas), nwb (pynwb/h5py), h5 (h5py),
numpy (.npz/.npy), pickle

返回值：四层数据字典 { meta, signal, event, processed }
"""
from typing import Dict, Any, Optional, Tuple, List
import os
import logging
import numpy as np
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_data(filepath: str, format: Optional[str] = None) -> Dict[str, Any]:
    """
    根据文件扩展名或显式 format 加载数据并转换为四层数据字典。
    支持自动选择格式：'.edf', '.bdf', '.mat', '.csv', '.nwb', '.h5', '.npz', '.npy', '.pkl', '.pickle'
                      '.json', '.txt', '.snirf', '.nirs', '.xlsx', '.xls'
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    fmt = format.lower() if format else ext.replace('.', '')

    if fmt in ('edf', 'bdf'):
        return _load_edf_bdf(filepath)
    elif fmt in ('mat',):
        return _load_mat(filepath)
    elif fmt in ('csv',):
        return _load_csv(filepath)
    elif fmt in ('nwb',):
        return _load_nwb(filepath)
    elif fmt in ('h5', 'hdf5'):
        return _load_h5(filepath)
    elif fmt in ('npz', 'npy'):
        return _load_numpy(filepath)
    elif fmt in ('pkl', 'pickle'):
        return _load_pickle(filepath)
    elif fmt in ('json',):
        return _load_json(filepath)
    elif fmt in ('txt',):
        return _load_txt(filepath)
    elif fmt in ('snirf', 'nirs'):
        return _load_snirf(filepath)
    elif fmt in ('xlsx', 'xls'):
        return _load_excel(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {fmt} (路径: {filepath})")


# -------------------- 实际加载实现 --------------------

def _load_edf_bdf(filepath: str) -> Dict[str, Any]:
    """
    使用 mne 加载 EDF/BDF，并尝试自动推断模态名称（优先识别 ECG、EOG、EMG、EEG）。
    返回四层数据字典，若检测到特定通道则使用对应模态名，否则用 signal['RAW']。
    """
    try:
        import mne
    except Exception as e:
        raise ImportError("需要安装 mne 才能加载 EDF/BDF 文件，运行: pip install mne") from e

    raw = mne.io.read_raw(filepath, preload=True, verbose=False)
    data = raw.get_data()  # shape (n_channels, n_samples)
    sfreq = raw.info.get('sfreq', None)
    ch_names = raw.ch_names or []

    # 获取通道类型信息
    try:
        ch_types = raw.get_channel_types()
    except Exception:
        ch_types = []

    # 增强模态检测逻辑，支持EEG/EOG/EMG/ECG分离
    modality_groups = _group_channels_by_modality(ch_names, ch_types)

    signal = {}
    for modality, indices in modality_groups.items():
        if not indices:
            continue

        mod_data = data[indices, :]
        mod_ch_names = [ch_names[i] for i in indices]

        # 【修改】获取单位信息
        unit_info = _extract_units(raw, indices)

        signal[modality] = {
            'data': mod_data,
            'sampling_rate': float(sfreq) if sfreq is not None else None,
            'channel_names': mod_ch_names,
            'unit': unit_info,
            'reference': 'unknown',  # 参考电极信息
            'time_offset': 0.0,  # 时间偏移
            'source_file': os.path.basename(filepath)
        }

    # 【修改】规范化meta信息以符合BCI标准格式
    meta = {
        'subject_id': _extract_subject_id(raw.info),  # 提取被试ID
        'session_id': _extract_session_id(filepath),  # 提取session信息
        'task': 'unknown',  # 任务类型
        'modality': list(signal.keys()),  # 使用检测到的模态列表
        'device': raw.info.get('device_info', {}).get('type', 'unknown'),  # 设备信息
        'sampling_rate': float(sfreq) if sfreq is not None else None,
        'n_channels': int(data.shape[0]) if hasattr(data, 'shape') else None,
        'n_samples': int(data.shape[1]) if hasattr(data, 'shape') else None,
        'channel_names': ch_names,  # 所有通道名
        'subject_info': raw.info.get('subject_info', {}),
        'meas_date': str(raw.info.get('meas_date', None)),  # 记录日期
        'file': os.path.basename(filepath),
        'format': 'edf/bdf',
    }

    return {'meta': meta, 'signal': signal, 'event': {}, 'processed': {}}


def _group_channels_by_modality(ch_names: List[str], ch_types: List[str]) -> Dict[str, List[int]]:
    """
    根据通道名和类型将通道分组到不同模态
    返回: {'EEG': [indices], 'EOG': [indices], 'EMG': [indices], 'ECG': [indices]}
    """
    groups = {'EEG': [], 'EOG': [], 'EMG': [], 'ECG': [], 'MISC': []}

    # 定义关键词
    eog_keywords = {'eog', 'heog', 'veog', 'eye'}
    emg_keywords = {'emg', 'muscle', 'flexor', 'extensor'}
    ecg_keywords = {'ecg', 'ekg', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'}

    for i, (name, ch_type) in enumerate(zip(ch_names, ch_types if ch_types else [''] * len(ch_names))):
        name_lower = name.lower()
        type_lower = ch_type.lower() if ch_type else ''

        # 优先使用类型信息
        if 'eog' in type_lower or any(kw in name_lower for kw in eog_keywords):
            groups['EOG'].append(i)
        elif 'emg' in type_lower or any(kw in name_lower for kw in emg_keywords):
            groups['EMG'].append(i)
        elif 'ecg' in type_lower or any(kw in name_lower for kw in ecg_keywords):
            groups['ECG'].append(i)
        elif 'eeg' in type_lower or ch_type == 'eeg':
            groups['EEG'].append(i)
        else:
            # 默认归类到EEG（如果没有明确标识）
            if not any([any(kw in name_lower for kw in eog_keywords | emg_keywords | ecg_keywords)]):
                groups['EEG'].append(i)
            else:
                groups['MISC'].append(i)

    # 移除空组
    return {k: v for k, v in groups.items() if v}


def _extract_units(raw, indices: List[int]) -> str:
    """
    从mne raw对象中提取通道单位信息
    """
    try:
        # 尝试获取第一个通道的单位
        if hasattr(raw, 'info') and 'chs' in raw.info:
            ch_unit = raw.info['chs'][indices[0]].get('unit', 'uV')
            # mne单位常见为FIFF单位常量，需要转换
            unit_map = {107: 'V', -6: 'uV', -3: 'mV'}
            if isinstance(ch_unit, int):
                return unit_map.get(ch_unit, 'uV')
            return 'uV'  # 默认微伏
    except Exception:
        pass
    return 'uV'


def _extract_subject_id(info: Dict) -> str:
    """
    【新增】从mne info中提取被试ID
    """
    subject_info = info.get('subject_info', {})
    if isinstance(subject_info, dict):
        return subject_info.get('id', subject_info.get('his_id', 'unknown'))
    return 'unknown'


def _extract_session_id(filepath: str) -> str:
    """
    【新增】从文件路径中提取session信息
    """
    basename = os.path.basename(filepath)
    # 尝试从文件名提取日期信息
    return os.path.splitext(basename)[0]


def _load_mat(filepath: str) -> Dict[str, Any]:
    """
    使用 scipy.io.loadmat 加载 mat 文件。
    如果找到变量名 data / signal / eeg 等，会尝试映射到 signal 层，否则把原始 dict 放到 processed['raw_mat']。
    """
    try:
        from scipy.io import loadmat
    except Exception as e:
        raise ImportError("需要安装 scipy 才能加载 .mat 文件，运行: pip install scipy") from e

    mat = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    signal = {}
    # 常见变量名映射
    modality_mappings = {
        'EEG': ['eeg', 'EEG', 'data', 'signal'],
        'EOG': ['eog', 'EOG'],
        'EMG': ['emg', 'EMG'],
        'ECG': ['ecg', 'ECG'],
        'fNIRS': ['fnirs', 'NIRS', 'nirs']
    }
    found = False
    for modality, var_names in modality_mappings.items():
        for name in var_names:
            if name in mat:
                arr = mat[name]
                if isinstance(arr, np.ndarray):
                    # 【修改】智能判断数组方向
                    if arr.ndim == 2:
                        # 假设 channels < samples
                        if arr.shape[0] > arr.shape[1]:
                            data = arr.T
                        else:
                            data = arr
                    else:
                        data = arr

                    # 【修改】规范化signal结构
                    signal[modality] = {
                        'data': data,
                        'sampling_rate': _extract_mat_value(mat, ['sfreq', 'fs', 'sampling_rate', 'Fs']),
                        'channel_names': _extract_mat_value(mat, ['channel_names', 'ch_names', 'channels']),
                        'unit': _extract_mat_value(mat, ['unit', 'units'], 'uV'),
                        'reference': _extract_mat_value(mat, ['reference', 'ref'], 'unknown'),
                        'time_offset': 0.0
                    }
                    found = True
                    break

        # 【修改】规范化meta结构
    meta = {
        'subject_id': _extract_mat_value(mat, ['subject_id', 'subject', 'subj'], 'unknown'),
        'session_id': _extract_mat_value(mat, ['session_id', 'session'],
                                         os.path.splitext(os.path.basename(filepath))[0]),
        'task': _extract_mat_value(mat, ['task', 'paradigm'], 'unknown'),
        'modality': list(signal.keys()),
        'device': _extract_mat_value(mat, ['device', 'system'], 'unknown'),
        'sampling_rate': _extract_mat_value(mat, ['sfreq', 'fs', 'sampling_rate']),
        'n_channels': signal[list(signal.keys())[0]]['data'].shape[0] if signal else None,
        'n_samples': signal[list(signal.keys())[0]]['data'].shape[1] if signal else None,
        'file': os.path.basename(filepath),
        'format': 'mat'
    }

    # 【修改】尝试提取event信息
    event = _extract_mat_events(mat)

    if not found:
        meta['note'] = '未检测到常见数据变量，raw mat 内容保存在 processed.raw_mat'
        return {'meta': meta, 'signal': {}, 'event': event, 'processed': {'raw_mat': mat}}
    else:
        return {'meta': meta, 'signal': signal, 'event': event, 'processed': {}}


def _extract_mat_value(mat: Dict, keys: List[str], default=None):
    """
    【新增】从mat字典中提取值，支持多个候选键
    """
    for key in keys:
        if key in mat:
            val = mat[key]
            # 处理matlab数组
            if isinstance(val, np.ndarray):
                if val.size == 1:
                    return val.item()
                elif val.dtype.kind in ['U', 'S', 'O']:  # 字符串类型
                    return [str(v) for v in val.flatten()] if val.size > 1 else str(val.item())
            return val
    return default


def _extract_mat_events(mat: Dict) -> Dict[str, Any]:
    """
    【新增】从mat文件中提取event信息
    """
    event = {}
    event_keys = {
        'event_id': ['event_id', 'event_type', 'events'],
        'event_label': ['event_label', 'event_labels', 'labels'],
        'event_time': ['event_time', 'event_times', 'times'],
        'event_sample': ['event_sample', 'event_samples', 'samples'],
        'duration': ['duration', 'durations']
    }

    for event_field, mat_keys in event_keys.items():
        val = _extract_mat_value(mat, mat_keys)
        if val is not None:
            event[event_field] = val

    return event


def _load_csv(filepath: str) -> Dict[str, Any]:
    """
    使用 pandas 加载 CSV，期望首行为列名，若存在 'time' 或 'timestamp' 列则作为时间轴。
    【修改】支持多模态识别和event信息提取
    """
    try:
        import pandas as pd
    except Exception as e:
        raise ImportError("需要安装 pandas 才能加载 CSV，运行: pip install pandas") from e

    df = pd.read_csv(filepath)

    # 【修改】识别特殊列
    time_cols = [c for c in df.columns if c.lower() in ('time', 'timestamp', 't')]
    event_cols = [c for c in df.columns if c.lower() in ('event', 'event_id', 'label', 'event_label', 'marker')]

    # 【新增】提取event信息
    event = {}
    if event_cols:
        event_col = event_cols[0]
        event_data = df[event_col].dropna()
        if len(event_data) > 0:
            event['event_id'] = event_data.values.tolist()
            if time_cols:
                event_times = df.loc[event_data.index, time_cols[0]].values
                event['event_time'] = event_times.tolist()

    # 移除特殊列，剩余为信号数据
    signal_cols = [c for c in df.columns if c not in time_cols + event_cols]

    # 【修改】根据列名推断模态
    modality_data = {}
    eeg_cols = [c for c in signal_cols if any(kw in c.lower() for kw in ['eeg', 'fp', 'fz', 'cz', 'pz', 'oz'])]
    eog_cols = [c for c in signal_cols if any(kw in c.lower() for kw in ['eog', 'heog', 'veog'])]
    emg_cols = [c for c in signal_cols if any(kw in c.lower() for kw in ['emg'])]

    signal = {}
    time = df[time_cols[0]].values if time_cols else None

    # 【修改】按模态分组
    if eeg_cols:
        modality_data['EEG'] = eeg_cols
    if eog_cols:
        modality_data['EOG'] = eog_cols
    if emg_cols:
        modality_data['EMG'] = emg_cols

    # 如果没有识别到特定模态，使用通用CSV模态
    if not modality_data:
        modality_data['CSV'] = signal_cols

    for modality, cols in modality_data.items():
        data = df[cols].to_numpy().T  # (n_channels, n_samples)

        sampling_rate = None
        if time is not None and len(time) >= 2:
            dt = np.mean(np.diff(time))
            if dt > 0:
                sampling_rate = 1.0 / float(dt)

        signal[modality] = {
            'data': data,
            'sampling_rate': sampling_rate,
            'channel_names': cols,
            'unit': 'unknown',  # 【新增】
            'reference': 'unknown',  # 【新增】
            'time_offset': 0.0,  # 【新增】
            'time': time
        }

    # 【修改】规范化meta
    meta = {
        'subject_id': 'unknown',
        'session_id': os.path.splitext(os.path.basename(filepath))[0],
        'task': 'unknown',
        'modality': list(signal.keys()),
        'device': 'unknown',
        'sampling_rate': signal[list(signal.keys())[0]]['sampling_rate'] if signal else None,
        'n_channels': sum(len(s['channel_names']) for s in signal.values()),
        'n_samples': signal[list(signal.keys())[0]]['data'].shape[1] if signal else None,
        'file': os.path.basename(filepath),
        'format': 'csv'
    }

    return {'meta': meta, 'signal': signal, 'event': event, 'processed': {}}

def _load_nwb(filepath: str) -> Dict[str, Any]:
    """
    尝试使用 pynwb 读取 NWB 文件并提取 acquisition / timeseries 数据
    【修改】规范化输出格式
    """
    try:
        from pynwb import NWBHDF5IO
    except Exception:
        raise ImportError("加载 NWB 需要 pynwb，运行: pip install pynwb")

    io = NWBHDF5IO(filepath, 'r')
    nwbf = io.read()
    signal = {}

    # 遍历 acquisition 中的 timeseries
    for name, ts in nwbf.acquisition.items():
        try:
            data = ts.data
            arr = np.asarray(data)

            # 【修改】规范化signal结构
            signal[name] = {
                'data': arr if arr.ndim == 2 else arr.reshape(1, -1),
                'sampling_rate': getattr(ts, 'rate', None),
                'channel_names': getattr(ts, 'channel_names', None),
                'unit': getattr(ts, 'unit', 'unknown'),  # 【新增】
                'reference': 'unknown',  # 【新增】
                'time_offset': 0.0  # 【新增】
            }
        except Exception:
            continue

    # 【修改】规范化meta
    meta = {
        'subject_id': getattr(nwbf, 'subject', None),
        'session_id': getattr(nwbf, 'session_id', 'unknown'),
        'task': getattr(nwbf, 'session_description', 'unknown'),
        'modality': list(signal.keys()),
        'device': 'unknown',
        'sampling_rate': signal[list(signal.keys())[0]]['sampling_rate'] if signal else None,
        'file': os.path.basename(filepath),
        'format': 'nwb'
    }

    io.close()
    return {'meta': meta, 'signal': signal, 'event': {}, 'processed': {}}


def _load_h5(filepath: str) -> Dict[str, Any]:
    """
    使用 h5py 加载 HDF5，返回 datasets 的字典。
    【修改】规范化输出格式
    """
    try:
        import h5py
    except Exception as e:
        raise ImportError("需要安装 h5py 才能加载 HDF5 文件，运行: pip install h5py") from e

    f = h5py.File(filepath, 'r')
    signal = {}

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            try:
                arr = obj[()]
                if isinstance(arr, (np.ndarray, np.generic)):
                    # 【修改】规范化signal结构
                    arr = np.asarray(arr)
                    signal[name] = {
                        'data': arr if arr.ndim == 2 else arr.reshape(1, -1),
                        'sampling_rate': None,
                        'channel_names': None,
                        'unit': 'unknown',  # 【新增】
                        'reference': 'unknown',  # 【新增】
                        'time_offset': 0.0  # 【新增】
                    }
            except Exception:
                pass

    f.visititems(_visit)

    # 【修改】规范化meta
    meta = {
        'subject_id': 'unknown',
        'session_id': os.path.splitext(os.path.basename(filepath))[0],
        'task': 'unknown',
        'modality': list(signal.keys()),
        'device': 'unknown',
        'file': os.path.basename(filepath),
        'format': 'h5'
    }

    f.close()
    return {'meta': meta, 'signal': signal, 'event': {}, 'processed': {}}


def _load_numpy(filepath: str) -> Dict[str, Any]:
    """
    加载 .npz 或 .npy 文件。
    【修改】规范化输出格式
    """
    ext = os.path.splitext(filepath)[1].lower()
    signal = {}

    if ext == '.npy':
        arr = np.load(filepath, allow_pickle=True)
        # 【修改】规范化signal结构
        arr = np.asarray(arr)
        signal['npy'] = {
            'data': arr if arr.ndim == 2 else arr.reshape(1, -1),
            'sampling_rate': None,
            'channel_names': None,
            'unit': 'unknown',  # 【新增】
            'reference': 'unknown',  # 【新增】
            'time_offset': 0.0  # 【新增】
        }
    else:
        # .npz
        npz = np.load(filepath, allow_pickle=True)
        for k in npz.files:
            arr = npz[k]
            arr = np.asarray(arr)
            # 【修改】规范化signal结构
            signal[k] = {
                'data': arr if arr.ndim == 2 else arr.reshape(1, -1),
                'sampling_rate': None,
                'channel_names': None,
                'unit': 'unknown',  # 【新增】
                'reference': 'unknown',  # 【新增】
                'time_offset': 0.0  # 【新增】
            }

    # 【修改】规范化meta
    meta = {
        'subject_id': 'unknown',
        'session_id': os.path.splitext(os.path.basename(filepath))[0],
        'task': 'unknown',
        'modality': list(signal.keys()),
        'device': 'unknown',
        'file': os.path.basename(filepath),
        'format': 'numpy'
    }

    return {'meta': meta, 'signal': signal, 'event': {}, 'processed': {}}


def _load_pickle(filepath: str) -> Dict[str, Any]:
    """
    反序列化 pickle，若为四层格式直接返回，否则放入 processed.raw_pickle
    """
    import pickle
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and {'meta', 'signal', 'event', 'processed'}.issubset(set(obj.keys())):
        return obj
    else:
        # 【修改】规范化meta
        meta = {
            'subject_id': 'unknown',
            'session_id': os.path.splitext(os.path.basename(filepath))[0],
            'task': 'unknown',
            'modality': [],
            'device': 'unknown',
            'file': os.path.basename(filepath),
            'format': 'pickle',
            'note': 'raw object saved in processed.raw_pickle'
        }
        return {'meta': meta, 'signal': {}, 'event': {}, 'processed': {'raw_pickle': obj}}


def _load_json(filepath: str) -> Dict[str, Any]:
    """
    【新增】加载JSON格式文件
    若为四层格式直接返回，否则尝试解析为信号数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        obj = json.load(f)

    # 检查是否为标准四层格式
    if isinstance(obj, dict) and {'meta', 'signal', 'event', 'processed'}.issubset(set(obj.keys())):
        # 转换numpy数组
        for modality in obj.get('signal', {}):
            if 'data' in obj['signal'][modality]:
                obj['signal'][modality]['data'] = np.array(obj['signal'][modality]['data'])
        return obj
    else:
        # 尝试解析为信号数据
        signal = {}
        if isinstance(obj, dict):
            for key, val in obj.items():
                if isinstance(val, (list, np.ndarray)):
                    arr = np.array(val)
                    signal[key] = {
                        'data': arr if arr.ndim == 2 else arr.reshape(1, -1),
                        'sampling_rate': None,
                        'channel_names': None,
                        'unit': 'unknown',
                        'reference': 'unknown',
                        'time_offset': 0.0
                    }

        meta = {
            'subject_id': obj.get('subject_id', 'unknown') if isinstance(obj, dict) else 'unknown',
            'session_id': obj.get('session_id', os.path.splitext(os.path.basename(filepath))[0]) if isinstance(obj,
                                                                                                               dict) else
            os.path.splitext(os.path.basename(filepath))[0],
            'task': obj.get('task', 'unknown') if isinstance(obj, dict) else 'unknown',
            'modality': list(signal.keys()),
            'device': 'unknown',
            'file': os.path.basename(filepath),
            'format': 'json',
            'note': 'Parsed from JSON' if not signal else None
        }

        return {'meta': meta, 'signal': signal, 'event': {}, 'processed': {'raw_json': obj} if not signal else {}}


def _load_txt(filepath: str) -> Dict[str, Any]:
    """
    【新增】加载TXT格式文件（假设为空格或制表符分隔的数值数据）
    """
    try:
        # 尝试作为数值数据加载
        data = np.loadtxt(filepath)

        # 转换为 (channels, samples) 格式
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.shape[0] > data.shape[1]:
            data = data.T

        signal = {
            'TXT': {
                'data': data,
                'sampling_rate': None,
                'channel_names': [f'Ch{i + 1}' for i in range(data.shape[0])],
                'unit': 'unknown',
                'reference': 'unknown',
                'time_offset': 0.0
            }
        }

        meta = {
            'subject_id': 'unknown',
            'session_id': os.path.splitext(os.path.basename(filepath))[0],
            'task': 'unknown',
            'modality': ['TXT'],
            'device': 'unknown',
            'n_channels': data.shape[0],
            'n_samples': data.shape[1],
            'file': os.path.basename(filepath),
            'format': 'txt'
        }

        return {'meta': meta, 'signal': signal, 'event': {}, 'processed': {}}

    except Exception as e:
        # 如果无法作为数值数据加载，保存为原始文本
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        meta = {
            'subject_id': 'unknown',
            'session_id': os.path.splitext(os.path.basename(filepath))[0],
            'task': 'unknown',
            'modality': [],
            'device': 'unknown',
            'file': os.path.basename(filepath),
            'format': 'txt',
            'note': f'Failed to parse as numeric data: {str(e)}'
        }

        return {'meta': meta, 'signal': {}, 'event': {}, 'processed': {'raw_text': content}}


def _load_snirf(filepath: str) -> Dict[str, Any]:
    """
    【新增】加载SNIRF格式的fNIRS数据
    SNIRF是基于HDF5的标准格式
    """
    try:
        import h5py
    except Exception as e:
        raise ImportError("需要安装 h5py 才能加载 SNIRF 文件，运行: pip install h5py") from e

    f = h5py.File(filepath, 'r')
    signal = {}

    try:
        # SNIRF标准结构: /nirs/data1/dataTimeSeries
        nirs_group = f.get('nirs', f.get('nirs1', None))

        if nirs_group:
            # 提取数据
            for data_key in nirs_group.keys():
                if data_key.startswith('data'):
                    data_group = nirs_group[data_key]

                    if 'dataTimeSeries' in data_group:
                        time_series = np.array(data_group['dataTimeSeries'])

                        # SNIRF格式通常是 (samples, channels)
                        if time_series.ndim == 2 and time_series.shape[0] > time_series.shape[1]:
                            time_series = time_series.T

                        # 获取采样率
                        sampling_rate = None
                        if 'time' in data_group:
                            time_data = np.array(data_group['time'])
                            if len(time_data) >= 2:
                                dt = np.mean(np.diff(time_data))
                                if dt > 0:
                                    sampling_rate = 1.0 / dt

                        signal['fNIRS'] = {
                            'data': time_series,
                            'sampling_rate': sampling_rate,
                            'channel_names': [f'Ch{i + 1}' for i in range(time_series.shape[0])],
                            'unit': 'unknown',
                            'reference': 'unknown',
                            'time_offset': 0.0
                        }

        # 提取元数据
        subject_id = 'unknown'
        if 'nirs' in f and 'metaDataTags' in f['nirs']:
            meta_tags = f['nirs']['metaDataTags']
            if 'SubjectID' in meta_tags:
                subject_id = str(np.array(meta_tags['SubjectID'])[0])

        meta = {
            'subject_id': subject_id,
            'session_id': os.path.splitext(os.path.basename(filepath))[0],
            'task': 'fNIRS',
            'modality': list(signal.keys()),
            'device': 'fNIRS',
            'sampling_rate': signal['fNIRS']['sampling_rate'] if 'fNIRS' in signal else None,
            'file': os.path.basename(filepath),
            'format': 'snirf'
        }

    finally:
        f.close()

    return {'meta': meta, 'signal': signal, 'event': {}, 'processed': {}}


def _load_excel(filepath: str) -> Dict[str, Any]:
    """
    【新增】加载Excel格式文件 (.xlsx, .xls)
    支持多个sheet，每个sheet可能代表不同的模态或数据类型
    第一行作为列名，第一列如果是时间相关则作为时间轴
    """
    try:
        import pandas as pd
    except Exception as e:
        raise ImportError("需要安装 pandas 才能加载 Excel 文件，运行: pip install pandas openpyxl") from e

    # 读取所有sheets
    excel_file = pd.ExcelFile(filepath)
    sheet_names = excel_file.sheet_names

    signal = {}
    event = {}
    meta_info = {}

    for sheet_name in sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet_name)

        # 跳过空sheet
        if df.empty:
            continue

        # 检查sheet名称或内容来判断类型
        sheet_lower = sheet_name.lower()

        # 【新增】识别meta信息sheet
        if any(kw in sheet_lower for kw in ['meta', 'info', 'metadata', 'information']):
            # 尝试解析为key-value对
            if df.shape[1] >= 2:
                for _, row in df.iterrows():
                    key = str(row.iloc[0]).strip()
                    value = row.iloc[1]
                    if pd.notna(key) and pd.notna(value):
                        meta_info[key] = value
            continue

        # 【新增】识别event信息sheet
        if any(kw in sheet_lower for kw in ['event', 'marker', 'trigger', 'stimulus']):
            # 提取event信息
            event_data = {}
            for col in df.columns:
                col_lower = col.lower()
                if any(kw in col_lower for kw in ['id', 'type', 'code']):
                    event_data['event_id'] = df[col].dropna().tolist()
                elif any(kw in col_lower for kw in ['label', 'name', 'description']):
                    event_data['event_label'] = df[col].dropna().tolist()
                elif any(kw in col_lower for kw in ['time', 'onset', 'timestamp']):
                    event_data['event_time'] = df[col].dropna().tolist()
                elif any(kw in col_lower for kw in ['sample', 'index']):
                    event_data['event_sample'] = df[col].dropna().tolist()
                elif any(kw in col_lower for kw in ['duration', 'length']):
                    event_data['duration'] = df[col].dropna().tolist()

            if event_data:
                event.update(event_data)
            continue

        # 【新增】处理信号数据sheet
        # 识别时间列
        time_cols = [c for c in df.columns if any(kw in str(c).lower() for kw in ['time', 'timestamp', 't', 'sample'])]

        # 识别event/marker列
        event_cols = [c for c in df.columns if
                      any(kw in str(c).lower() for kw in ['event', 'marker', 'trigger', 'label'])]

        # 剩余列作为信号数据
        signal_cols = [c for c in df.columns if c not in time_cols + event_cols]

        if not signal_cols:
            continue

        # 提取信号数据
        data = df[signal_cols].to_numpy().T  # (n_channels, n_samples)

        # 提取时间信息
        time = df[time_cols[0]].values if time_cols else None

        # 计算采样率
        sampling_rate = None
        if time is not None and len(time) >= 2:
            dt = np.mean(np.diff(time))
            if dt > 0:
                sampling_rate = 1.0 / float(dt)

        # 【新增】根据sheet名称或列名推断模态
        modality = _infer_modality_from_names(sheet_name, signal_cols)

        signal[modality] = {
            'data': data,
            'sampling_rate': sampling_rate,
            'channel_names': signal_cols,
            'unit': 'unknown',
            'reference': 'unknown',
            'time_offset': 0.0,
            'time': time,
            'source_sheet': sheet_name
        }

        # 【新增】如果该sheet有event信息，也提取
        if event_cols and not event:
            for event_col in event_cols:
                event_data = df[event_col].dropna()
                if len(event_data) > 0:
                    event['event_id'] = event_data.values.tolist()
                    if time_cols:
                        event_times = df.loc[event_data.index, time_cols[0]].values
                        event['event_time'] = event_times.tolist()
                    break

    # 【新增】构建meta信息
    meta = {
        'subject_id': meta_info.get('subject_id', meta_info.get('Subject ID', meta_info.get('SubjectID', 'unknown'))),
        'session_id': meta_info.get('session_id', meta_info.get('Session ID', meta_info.get('SessionID',
                                                                                            os.path.splitext(
                                                                                                os.path.basename(
                                                                                                    filepath))[0]))),
        'task': meta_info.get('task', meta_info.get('Task', 'unknown')),
        'modality': list(signal.keys()),
        'device': meta_info.get('device', meta_info.get('Device', 'unknown')),
        'sampling_rate': signal[list(signal.keys())[0]]['sampling_rate'] if signal else None,
        'n_channels': sum(len(s['channel_names']) for s in signal.values()),
        'n_samples': signal[list(signal.keys())[0]]['data'].shape[1] if signal else None,
        'file': os.path.basename(filepath),
        'format': 'xlsx',
        'sheets': sheet_names
    }

    # 添加其他meta信息
    for key, value in meta_info.items():
        if key not in meta:
            meta[key] = value

    return {'meta': meta, 'signal': signal, 'event': event, 'processed': {}}


def _infer_modality_from_names(sheet_name: str, column_names: List[str]) -> str:
    """
    【新增】根据sheet名称和列名推断模态类型
    """
    combined_text = (sheet_name + ' ' + ' '.join(str(c) for c in column_names)).lower()

    # 定义模态关键词
    modality_keywords = {
        'EEG': ['eeg', 'electroencephalogram', 'fp', 'fz', 'cz', 'pz', 'oz'],
        'EOG': ['eog', 'electrooculogram', 'heog', 'veog', 'eye'],
        'EMG': ['emg', 'electromyogram', 'muscle', 'flexor', 'extensor'],
        'ECG': ['ecg', 'ekg', 'electrocardiogram', 'heart', 'ii', 'iii', 'avr', 'avl', 'avf'],
        'fNIRS': ['fnirs', 'nirs', 'hbo', 'hbr', 'oxy', 'deoxy'],
        'MEG': ['meg', 'magnetoencephalogram'],
        'fMRI': ['fmri', 'bold'],
    }

    # 检查关键词匹配
    for modality, keywords in modality_keywords.items():
        if any(kw in combined_text for kw in keywords):
            return modality

    # 如果没有匹配，使用sheet名称或默认
    if sheet_name and sheet_name.upper() not in ['SHEET1', 'SHEET2', 'SHEET3']:
        return sheet_name.upper()

    return 'DATA'

# -------------------- 保存函数 --------------------

def save_data(data_dict: Dict[str, Any], outpath: str, format: Optional[str] = None) -> None:
    """
    保存四层数据字典到磁盘。
    【修改】支持更多格式: pickle (.pkl/.pickle), numpy (.npz), mat (.mat), json (.json), csv (.csv), xlsx (.xlsx)
    """
    ext = os.path.splitext(outpath)[1].lower()
    fmt = format.lower() if format else ext.replace('.', '')

    if fmt in ('pkl', 'pickle'):
        _save_pickle(data_dict, outpath)
    elif fmt in ('npz', 'numpy'):
        _save_numpy(data_dict, outpath)
    elif fmt in ('mat',):  # 【新增】
        _save_mat(data_dict, outpath)
    elif fmt in ('json',):  # 【新增】
        _save_json(data_dict, outpath)
    elif fmt in ('csv',):  # 【新增】
        _save_csv(data_dict, outpath)
    elif fmt in ('txt',):  # 【新增】
        _save_txt(data_dict, outpath)
    elif fmt in ('xlsx', 'xls'):  # 【新增】
        _save_excel(data_dict, outpath)
    else:
        raise ValueError(f"不支持的保存格式: {fmt}")


def _save_pickle(data_dict: Dict[str, Any], outpath: str) -> None:
    """保存为pickle格式"""
    import pickle
    with open(outpath, 'wb') as f:
        pickle.dump(data_dict, f)
    logger.info(f"已保存 pickle 到 {outpath}")


def _save_numpy(data_dict: Dict[str, Any], outpath: str) -> None:
    """保存为npz格式"""
    save_dict = {}
    for modality, info in data_dict.get('signal', {}).items():
        if isinstance(info, dict) and 'data' in info:
            save_dict[modality] = np.asarray(info['data'])

    # 【新增】也保存meta和event信息
    save_dict['_meta'] = np.array([data_dict.get('meta', {})], dtype=object)
    save_dict['_event'] = np.array([data_dict.get('event', {})], dtype=object)

    np.savez_compressed(outpath, **save_dict)
    logger.info(f"已保存 npz 到 {outpath}")


def _save_mat(data_dict: Dict[str, Any], outpath: str) -> None:
    """
    【新增】保存为MATLAB .mat格式
    """
    try:
        from scipy.io import savemat
    except Exception as e:
        raise ImportError("需要安装 scipy 才能保存 .mat 文件，运行: pip install scipy") from e

    save_dict = {}

    # 保存signal数据
    for modality, info in data_dict.get('signal', {}).items():
        if isinstance(info, dict) and 'data' in info:
            save_dict[modality] = np.asarray(info['data'])
            if 'sampling_rate' in info and info['sampling_rate'] is not None:
                save_dict[f'{modality}_sfreq'] = info['sampling_rate']
            if 'channel_names' in info and info['channel_names'] is not None:
                save_dict[f'{modality}_channels'] = info['channel_names']

    # 保存meta信息
    meta = data_dict.get('meta', {})
    for key, val in meta.items():
        if val is not None and not isinstance(val, (dict, list)):
            save_dict[f'meta_{key}'] = val

    # 保存event信息
    event = data_dict.get('event', {})
    for key, val in event.items():
        if val is not None:
            save_dict[f'event_{key}'] = val

    savemat(outpath, save_dict)
    logger.info(f"已保存 mat 到 {outpath}")


def _save_json(data_dict: Dict[str, Any], outpath: str) -> None:
    """
    【新增】保存为JSON格式（numpy数组转为列表）
    """

    def _convert_to_serializable(obj):
        """递归转换numpy数组为列表"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    serializable_dict = _convert_to_serializable(data_dict)

    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(serializable_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"已保存 json 到 {outpath}")


def _save_csv(data_dict: Dict[str, Any], outpath: str) -> None:
    """
    【新增】保存为CSV格式（仅保存第一个模态的数据）
    """
    try:
        import pandas as pd
    except Exception as e:
        raise ImportError("需要安装 pandas 才能保存 CSV，运行: pip install pandas") from e

    signal = data_dict.get('signal', {})
    if not signal:
        raise ValueError("没有可保存的信号数据")

    # 取第一个模态
    first_modality = list(signal.keys())[0]
    info = signal[first_modality]
    data = info['data']

    # 转换为 (samples, channels)
    if data.shape[0] < data.shape[1]:
        data = data.T

    # 创建DataFrame
    channel_names = info.get('channel_names', [f'Ch{i + 1}' for i in range(data.shape[1])])
    df = pd.DataFrame(data, columns=channel_names)

    # 【新增】添加时间列
    if 'time' in info and info['time'] is not None:
        df.insert(0, 'time', info['time'])
    elif info.get('sampling_rate'):
        time = np.arange(data.shape[0]) / info['sampling_rate']
        df.insert(0, 'time', time)

    # 【新增】添加event标记
    event = data_dict.get('event', {})
    if 'event_sample' in event and 'event_id' in event:
        event_col = np.full(len(df), np.nan)
        for sample, event_id in zip(event['event_sample'], event['event_id']):
            if 0 <= sample < len(event_col):
                event_col[sample] = event_id
        df['event'] = event_col

    df.to_csv(outpath, index=False)
    logger.info(f"已保存 csv 到 {outpath}")


def _save_txt(data_dict: Dict[str, Any], outpath: str) -> None:
    """
    【新增】保存为TXT格式（空格分隔的数值数据）
    """
    signal = data_dict.get('signal', {})
    if not signal:
        raise ValueError("没有可保存的信号数据")

    # 取第一个模态
    first_modality = list(signal.keys())[0]
    data = signal[first_modality]['data']

    # 转换为 (samples, channels)
    if data.shape[0] < data.shape[1]:
        data = data.T

    np.savetxt(outpath, data, fmt='%.6f', delimiter=' ')
    logger.info(f"已保存 txt 到 {outpath}")


def _save_excel(data_dict: Dict[str, Any], outpath: str) -> None:
    """
    【新增】保存为Excel格式，使用多个sheet存储不同信息
    - Meta sheet: 元数据
    - Event sheet: 事件信息
    - 各模态signal sheet: 信号数据
    """
    try:
        import pandas as pd
    except Exception as e:
        raise ImportError("需要安装 pandas 和 openpyxl 才能保存 Excel，运行: pip install pandas openpyxl") from e

    with pd.ExcelWriter(outpath, engine='openpyxl') as writer:
        # 【新增】保存Meta信息
        meta = data_dict.get('meta', {})
        if meta:
            meta_df = pd.DataFrame([
                {'Key': k, 'Value': v}
                for k, v in meta.items()
                if not isinstance(v, (dict, list)) or v == []
            ])
            meta_df.to_excel(writer, sheet_name='Meta', index=False)

        # 【新增】保存Event信息
        event = data_dict.get('event', {})
        if event:
            # 找出最长的列表长度
            max_len = max((len(v) if isinstance(v, list) else 1) for v in event.values())

            # 构建DataFrame
            event_data = {}
            for key, val in event.items():
                if isinstance(val, list):
                    # 填充到相同长度
                    event_data[key] = val + [None] * (max_len - len(val))
                else:
                    event_data[key] = [val] + [None] * (max_len - 1)

            event_df = pd.DataFrame(event_data)
            event_df.to_excel(writer, sheet_name='Events', index=False)

        # 【新增】保存每个模态的信号数据
        signal = data_dict.get('signal', {})
        for modality, info in signal.items():
            if not isinstance(info, dict) or 'data' not in info:
                continue

            data = info['data']

            # 转换为 (samples, channels)
            if data.shape[0] < data.shape[1]:
                data = data.T

            # 获取通道名
            channel_names = info.get('channel_names', [f'Ch{i + 1}' for i in range(data.shape[1])])

            # 创建DataFrame
            df = pd.DataFrame(data, columns=channel_names)

            # 【新增】添加时间列
            if 'time' in info and info['time'] is not None:
                df.insert(0, 'Time', info['time'])
            elif info.get('sampling_rate'):
                time = np.arange(data.shape[0]) / info['sampling_rate']
                df.insert(0, 'Time', time)

            # 使用模态名作为sheet名（限制长度）
            sheet_name = modality[:31]  # Excel sheet名称限制31字符
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        # 【新增】保存processed信息（如果有且不太大）
        processed = data_dict.get('processed', {})
        if processed and len(str(processed)) < 10000:  # 避免过大的数据
            try:
                processed_df = pd.DataFrame([
                    {'Key': k, 'Value': str(v)[:500]}  # 限制长度
                    for k, v in processed.items()
                ])
                processed_df.to_excel(writer, sheet_name='Processed', index=False)
            except Exception:
                pass  # 忽略无法保存的processed数据

    logger.info(f"已保存 excel 到 {outpath}")

# -------------------- 简单检测帮助函数 --------------------

def detect_format_by_magic(filepath: str) -> Tuple[str, str]:
    """
    简单根据文件头检测（扩展名优先），供调试使用
    返回 (ext, mime_hint)
    """
    ext = os.path.splitext(filepath)[1].lower()
    mime = ''
    try:
        with open(filepath, 'rb') as f:
            header = f.read(512)
            if header.startswith(b'MAT'):
                mime = 'mat'
            elif b'HDF' in header[:8] or header.startswith(b'\x89HDF'):
                mime = 'h5'
            elif b'PK' in header[:2]:
                mime = 'zip/npz/pk'
            elif header.startswith(b'\x00\x00'):
                mime = 'binary'
            elif header.startswith(b'{'):  # 【新增】JSON检测
                mime = 'json'
    except Exception:
        pass
    return ext, mime

# -------------------- 工具函数 --------------------

def validate_data_dict(data_dict: Dict[str, Any]) -> bool:
    """
    【新增】验证数据字典是否符合BCI标准格式
    """
    required_keys = {'meta', 'signal', 'event', 'processed'}
    if not all(k in data_dict for k in required_keys):
        return False

    # 验证meta必需字段
    meta_required = {'subject_id', 'session_id', 'task', 'modality', 'device'}
    if not all(k in data_dict['meta'] for k in meta_required):
        return False

    # 验证signal结构
    for modality, info in data_dict.get('signal', {}).items():
        signal_required = {'data', 'sampling_rate', 'channel_names', 'unit', 'reference', 'time_offset'}
        if not all(k in info for k in signal_required):
            return False
        if not isinstance(info['data'], np.ndarray):
            return False

    return True


def print_data_summary(data_dict: Dict[str, Any]) -> None:
    """
    【新增】打印数据字典摘要信息
    """
    print("=" * 60)
    print("数据摘要")
    print("=" * 60)

    # Meta信息
    meta = data_dict.get('meta', {})
    print(f"\n[Meta Information]")
    print(f"  Subject ID: {meta.get('subject_id', 'N/A')}")
    print(f"  Session ID: {meta.get('session_id', 'N/A')}")
    print(f"  Task: {meta.get('task', 'N/A')}")
    print(f"  Modality: {meta.get('modality', 'N/A')}")
    print(f"  Device: {meta.get('device', 'N/A')}")
    print(f"  Format: {meta.get('format', 'N/A')}")

    # Signal信息
    signal = data_dict.get('signal', {})
    print(f"\n[Signal Data]")
    for modality, info in signal.items():
        print(f"  {modality}:")
        print(f"    Shape: {info['data'].shape}")
        print(f"    Sampling Rate: {info.get('sampling_rate', 'N/A')} Hz")
        print(f"    Channels: {len(info.get('channel_names', []))} ({info.get('unit', 'N/A')})")

    # Event信息
    event = data_dict.get('event', {})
    if event:
        print(f"\n[Event Data]")
        print(f"  Events: {len(event.get('event_id', []))}")
        if 'event_label' in event:
            print(f"  Labels: {set(event['event_label'])}")

    print("=" * 60)