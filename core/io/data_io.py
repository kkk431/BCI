# -*- coding: utf-8 -*-
"""
通用数据 I/O 模块
支持格式：edf/bdf (mne), mat (scipy.io), csv (pandas), nwb (pynwb/h5py), h5 (h5py),
numpy (.npz/.npy), pickle

返回值：四层数据字典 { meta, signal, event, processed }
"""
from typing import Dict, Any, Optional, Tuple
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_data(filepath: str, format: Optional[str] = None) -> Dict[str, Any]:
    """
    根据文件扩展名或显式 format 加载数据并转换为四层数据字典。
    支持自动选择格式：'.edf', '.bdf', '.mat', '.csv', '.nwb', '.h5', '.npz', '.npy', '.pkl', '.pickle'
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
    else:
        raise ValueError(f"不支持的文件格式: {fmt} (路径: {filepath})")


# -------------------- 实际加载实现 --------------------

def _load_edf_bdf(filepath: str) -> Dict[str, Any]:
    """
    使用 mne 加载 EDF/BDF，并尝试自动推断模态名称（优先识别 ECG）。
    返回四层数据字典，若检测到 ECG 通道则使用 signal['ECG']，否则用 signal['RAW']。
    """
    try:
        import mne
    except Exception as e:
        raise ImportError("需要安装 mne 才能加载 EDF/BDF 文件，运行: pip install mne") from e

    raw = mne.io.read_raw(filepath, preload=True, verbose=False)
    data = raw.get_data()  # shape (n_channels, n_samples)
    sfreq = raw.info.get('sfreq', None)
    ch_names = raw.ch_names or []
    # mne 返回的通道类型列表
    try:
        ch_types = raw.get_channel_types()
    except Exception:
        ch_types = []

    # 简单的 ECG 关键词检测（包含常见导联名/类型）
    ecg_keywords = {'ecg', 'ekg', 'ii', 'iii', 'i', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'avr', 'avl', 'avf'}
    def _looks_like_ecg(names, types):
        # 检查通道名
        for n in names:
            ns = str(n).lower()
            for kw in ecg_keywords:
                if kw in ns:
                    return True
        # 检查通道类型
        for t in types:
            if isinstance(t, str) and 'ecg' in t.lower():
                return True
        return False

    is_ecg = _looks_like_ecg(ch_names, ch_types)
    mod_name = 'ECG' if is_ecg else 'RAW'

    # unit 信息尽量归一化为字符串或通道类型列表
    unit_info = None
    try:
        # 某些 mne raw.info 中有 chs -> each has 'unit'、'units'
        unit_info = [raw.get_channel_types()[i] for i in range(len(ch_names))] if ch_names else None
    except Exception:
        unit_info = None

    signal = {
        mod_name: {
            'data': data,
            'sampling_rate': float(sfreq) if sfreq is not None else None,
            'channel_names': ch_names,
            'unit': unit_info,
            'source_file': os.path.basename(filepath)
        }
    }

    meta = {
        'n_channels': int(data.shape[0]) if hasattr(data, 'shape') else None,
        'n_samples': int(data.shape[1]) if hasattr(data, 'shape') else None,
        'subject_info': raw.info.get('subject_info', {}),
        'sfreq': sfreq,
        'device': raw.info.get('meas_date', None),
        'file': os.path.basename(filepath),
        'format': 'edf/bdf',
        'inferred_modality': mod_name
    }

    return {'meta': meta, 'signal': signal, 'event': {}, 'processed': {}}


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
    candidates = ['data', 'signal', 'EEG', 'eeg', 'emg', 'ECG', 'ecg', 'fnirs']
    found = False
    for name in candidates:
        if name in mat:
            arr = mat[name]
            # 期望 ndarray (channels, samples) 或 (samples, channels)
            if isinstance(arr, np.ndarray):
                if arr.ndim == 2 and arr.shape[0] <= arr.shape[1]:
                    data = arr
                elif arr.ndim == 2:
                    data = arr
                else:
                    data = arr
                signal[name] = {'data': data, 'sampling_rate': mat.get('sfreq', mat.get('fs', None)),
                                'channel_names': mat.get('channel_names', None)}
                found = True

    meta = {'file': os.path.basename(filepath), 'format': 'mat'}
    if not found:
        # 不做断言，保存原始结构以便后续手动处理
        meta['note'] = '未检测到常见数据变量，raw mat 内容保存在 processed.raw_mat'
        return {'meta': meta, 'signal': {}, 'event': {}, 'processed': {'raw_mat': mat}}
    else:
        return {'meta': meta, 'signal': signal, 'event': {}, 'processed': {}}


def _load_csv(filepath: str) -> Dict[str, Any]:
    """
    使用 pandas 加载 CSV，期望首行为列名，若存在 'time' 或 'timestamp' 列则作为时间轴。
    默认把数值列视为通道，结果为 channels x samples 数组，放到 signal['CSV']。
    """
    try:
        import pandas as pd
    except Exception as e:
        raise ImportError("需要安装 pandas 才能加载 CSV，运行: pip install pandas") from e

    df = pd.read_csv(filepath)
    # 寻找时间列
    time_cols = [c for c in df.columns if c.lower() in ('time', 'timestamp', 't')]
    if time_cols:
        time = df[time_cols[0]].values
        signal_cols = [c for c in df.columns if c not in time_cols]
    else:
        time = None
        signal_cols = list(df.columns)

    data = df[signal_cols].to_numpy().T  # (n_channels, n_samples)
    sampling_rate = None
    if time is not None and len(time) >= 2:
        # 通过时间列估算采样率（若为秒）
        dt = np.mean(np.diff(time))
        if dt > 0:
            sampling_rate = 1.0 / float(dt)

    signal = {
        'CSV': {
            'data': data,
            'sampling_rate': sampling_rate,
            'channel_names': signal_cols,
            'time': time
        }
    }
    meta = {'file': os.path.basename(filepath), 'format': 'csv'}
    return {'meta': meta, 'signal': signal, 'event': {}, 'processed': {}}


def _load_nwb(filepath: str) -> Dict[str, Any]:
    """
    尝试使用 pynwb 读取 NWB 文件并提取 acquisition / timeseries 数据
    """
    try:
        from pynwb import NWBHDF5IO
    except Exception:
        # 回退到 h5py 提示
        raise ImportError("加载 NWB 需要 pynwb，运行: pip install pynwb")

    io = NWBHDF5IO(filepath, 'r')
    nwbf = io.read()
    signal = {}
    # 遍历 acquisition 中的 timeseries
    for name, ts in nwbf.acquisition.items():
        try:
            data = ts.data
            # data 可能是 numpy 或 h5 源，转换为 ndarray
            arr = np.asarray(data)
            signal[name] = {
                'data': arr,
                'sampling_rate': getattr(ts, 'rate', None),
                'channel_names': getattr(ts, 'channel_names', None)
            }
        except Exception:
            continue

    meta = {'file': os.path.basename(filepath), 'format': 'nwb'}
    io.close()
    return {'meta': meta, 'signal': signal, 'event': {}, 'processed': {}}


def _load_h5(filepath: str) -> Dict[str, Any]:
    """
    使用 h5py 加载 HDF5，返回 datasets 的字典。不会尝试智能映射，但会将数值数组转成 numpy。
    """
    try:
        import h5py
    except Exception as e:
        raise ImportError("需要安装 h5py 才能加载 HDF5 文件，运行: pip install h5py") from e

    f = h5py.File(filepath, 'r')
    signal = {}
    # 简单遍历顶层可数值数据集
    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            try:
                arr = obj[()]
                # 仅处理数值数组
                if isinstance(arr, (np.ndarray, np.generic)):
                    signal[name] = {'data': np.asarray(arr)}
            except Exception:
                pass

    f.visititems(_visit)
    meta = {'file': os.path.basename(filepath), 'format': 'h5'}
    f.close()
    return {'meta': meta, 'signal': signal, 'event': {}, 'processed': {}}


def _load_numpy(filepath: str) -> Dict[str, Any]:
    """
    加载 .npz 或 .npy 文件。
    对 .npz 尝试把每个数组映射为一个 signal 模态。
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.npy':
        arr = np.load(filepath, allow_pickle=True)
        signal = {'npy': {'data': np.asarray(arr), 'sampling_rate': None}}
    else:
        # .npz
        npz = np.load(filepath, allow_pickle=True)
        signal = {}
        for k in npz.files:
            arr = npz[k]
            signal[k] = {'data': np.asarray(arr), 'sampling_rate': None}
    meta = {'file': os.path.basename(filepath), 'format': 'numpy'}
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
        meta = {'file': os.path.basename(filepath), 'format': 'pickle', 'note': 'raw object saved in processed.raw_pickle'}
        return {'meta': meta, 'signal': {}, 'event': {}, 'processed': {'raw_pickle': obj}}


# -------------------- 保存函数 --------------------

def save_data(data_dict: Dict[str, Any], outpath: str, format: Optional[str] = None) -> None:
    """
    保存四层数据字典到磁盘。支持 pickle (.pkl/.pickle) 和 numpy (.npz)。
    若指定 format 会覆盖扩展名判断。
    """
    ext = os.path.splitext(outpath)[1].lower()
    fmt = format.lower() if format else ext.replace('.', '')

    if fmt in ('pkl', 'pickle'):
        import pickle
        with open(outpath, 'wb') as f:
            pickle.dump(data_dict, f)
        logger.info(f"已保存 pickle 到 {outpath}")
    elif fmt in ('npz', 'numpy'):
        # 将每个 signal 模态按名字保存为 npz 中的数组（尽量保存 data 字段）
        save_dict = {}
        for modality, info in data_dict.get('signal', {}).items():
            if isinstance(info, dict) and 'data' in info:
                save_dict[modality] = np.asarray(info['data'])
        np.savez_compressed(outpath, **save_dict)
        logger.info(f"已保存 npz 到 {outpath}")
    else:
        raise ValueError(f"不支持的保存格式: {fmt}")


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
    except Exception:
        pass
    return ext, mime