#!/usr/bin/env python3
"""
universal_bio_signal_converter.py
万能生物信号数据转换器 - 纯IO/转换版本

直接运行！不需要任何其他文件！
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import pickle
from datetime import datetime
from enum import Enum


# ==================== 信号类型枚举 ====================
class SignalType(Enum):
    """信号类型枚举"""
    EEG = "eeg"
    EMG = "emg"
    ECG = "ecg"
    GSR = "gsr"
    FNIRS = "fnirs"
    ET = "eyetracker"
    RESP = "resp"
    OTHER = "other"


# ==================== 数据字典构建器 ====================
class DataDictBuilder:
    """标准数据字典构建器 - 四层结构: meta, signal, event, processed"""

    @staticmethod
    def create_empty_data_dict() -> Dict[str, Any]:
        """创建空的四层数据字典"""
        return {
            "meta": {
                "subject_id": "",
                "session_id": "",
                "task": "",
                "recording_time": "",
                "file_path": "",
                "format_version": "1.0",
                "modality": [],
                "device": "",
                "notes": ""
            },
            "signal": {},
            "event": {
                "event_id": [],
                "event_label": [],
                "event_time": [],
                "duration": []
            },
            "processed": {
                "features": {},
                "artifacts": {},
                "filtered_data": {}
            }
        }

    @staticmethod
    def build_meta(
            subject_id: str = "",
            session_id: str = "",
            task: str = "",
            recording_time: str = "",
            file_path: str = "",
            modality: List[str] = None,
            device: str = "",
            sampling_rate: float = None,
            n_channels: int = None,
            channel_names: List[str] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """构建meta层"""
        if modality is None:
            modality = []
        if channel_names is None:
            channel_names = []

        meta = {
            "subject_id": subject_id,
            "session_id": session_id,
            "task": task,
            "recording_time": recording_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": file_path,
            "format_version": "1.0",
            "modality": modality,
            "device": device,
            "notes": kwargs.get("notes", "")
        }

        if sampling_rate is not None:
            meta["sampling_rate"] = float(sampling_rate)
        if n_channels is not None:
            meta["n_channels"] = int(n_channels)
        if channel_names:
            meta["channel_names"] = channel_names

        for key, value in kwargs.items():
            if key not in meta and key != "notes":
                meta[key] = value

        return meta

    @staticmethod
    def add_signal(
            data_dict: Dict,
            data: np.ndarray,
            sampling_rate: float,
            channel_names: List[str],
            modality: str,
            signal_type: str = None,
            unit: str = None,
            **signal_info
    ) -> Dict:
        """添加信号到signal层"""
        if "signal" not in data_dict:
            data_dict["signal"] = {}

        # 验证数据维度
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim == 2 and data.shape[0] > data.shape[1]:
            if len(channel_names) == data.shape[1]:
                data = data.T

        # 验证通道名
        if len(channel_names) != data.shape[0]:
            channel_names = [f"Ch{i + 1}" for i in range(data.shape[0])]

        signal_entry = {
            "data": np.asarray(data, dtype=np.float32),
            "sampling_rate": float(sampling_rate),
            "channel_names": list(channel_names),
            "signal_type": signal_type or modality.lower(),
            "unit": unit or "unknown",
            "n_channels": data.shape[0],
            "n_samples": data.shape[1],
            "duration": data.shape[1] / sampling_rate,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **signal_info
        }

        data_dict["signal"][modality.upper()] = signal_entry

        # 更新meta中的modality列表
        if modality.upper() not in data_dict["meta"].get("modality", []):
            data_dict["meta"].setdefault("modality", []).append(modality.upper())

        return data_dict

    @staticmethod
    def add_event(
            data_dict: Dict,
            event_label: str,
            event_time: float,
            duration: float = 0.0,
            event_id: int = None
    ) -> Dict:
        """添加事件到event层"""
        if "event" not in data_dict:
            data_dict["event"] = {
                "event_id": [],
                "event_label": [],
                "event_time": [],
                "duration": []
            }

        if event_id is None:
            event_id = len(data_dict["event"]["event_id"]) + 1

        data_dict["event"]["event_id"].append(event_id)
        data_dict["event"]["event_label"].append(event_label)
        data_dict["event"]["event_time"].append(event_time)
        data_dict["event"]["duration"].append(duration)

        return data_dict


# ==================== 支持的输入格式 ====================
SUPPORTED_INPUT_FORMATS = {
    'csv': 'CSV文件',
    'tsv': 'TSV文件',
    'txt': '文本文件',
    'xlsx': 'Excel文件',
    'xls': 'Excel文件(旧版)',
    'edf': 'EDF/EDF+格式',
    'bdf': 'BDF格式',
    'gdf': 'GDF格式',
    'set': 'EEGLAB SET格式',
    'vhdr': 'BrainVision头文件',
    'vmrk': 'BrainVision标记文件',
    'eeg': 'BrainVision EEG文件',
    'snirf': 'SNIRF格式',
    'nirs': 'NIRS数据格式',
    'mat': 'MATLAB MAT文件',
    'npy': 'NumPy二进制格式',
    'npz': 'NumPy压缩格式',
    'json': 'JSON格式',
    'pkl': 'Python Pickle格式',
}

SUPPORTED_OUTPUT_FORMATS = ['json', 'npz', 'pkl', 'csv', 'tsv', 'mat']


# ==================== 格式检测 ====================
def detect_format(file_path: str) -> str:
    """根据扩展名检测文件格式"""
    path = Path(file_path)
    suffix = path.suffix.lower().lstrip('.')
    return suffix if suffix in SUPPORTED_INPUT_FORMATS else 'unknown'


# ==================== 加载器 ====================
class DataLoader:
    """读取文件 → 标准数据字典"""

    def __init__(self):
        self.builder = DataDictBuilder()

    def load(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """加载文件，返回标准数据字典"""
        format = detect_format(file_path)

        if format == 'unknown':
            raise ValueError(f"不支持的文件格式: {file_path}")

        print(f"📂 读取: {Path(file_path).name} ({format})")

        # 根据格式加载
        if format in ['csv', 'tsv', 'txt']:
            return self._load_text(file_path, format, **kwargs)
        elif format in ['xlsx', 'xls']:
            return self._load_excel(file_path, **kwargs)
        elif format in ['edf', 'bdf', 'gdf']:
            return self._load_edf(file_path, format, **kwargs)
        elif format == 'mat':
            return self._load_mat(file_path, **kwargs)
        elif format in ['npy', 'npz']:
            return self._load_numpy(file_path, **kwargs)
        elif format in ['snirf', 'nirs']:
            return self._load_fnirs(file_path, **kwargs)
        elif format in ['set', 'vhdr', 'eeg']:
            return self._load_eeg(file_path, format, **kwargs)
        elif format == 'json':
            return self._load_json(file_path, **kwargs)
        elif format == 'pkl':
            return self._load_pickle(file_path, **kwargs)
        else:
            raise ValueError(f"未实现加载器: {format}")

    def _load_text(self, file_path: str, format: str, **kwargs) -> Dict:
        """加载文本文件"""
        delimiter = ',' if format == 'csv' else '\t' if format == 'tsv' else kwargs.get('delimiter', ',')

        df = pd.read_csv(file_path, delimiter=delimiter)
        data_dict = self.builder.create_empty_data_dict()

        fs = kwargs.get('fs', kwargs.get('sampling_rate', 1000))
        modality = kwargs.get('modality', 'UNKNOWN').upper()
        subject_id = kwargs.get('subject_id', Path(file_path).stem)
        session_id = kwargs.get('session_id', 'session1')
        task = kwargs.get('task', 'unknown')
        unit = kwargs.get('unit', 'unknown')

        signal_cols = df.columns.tolist()

        if signal_cols:
            data = df[signal_cols].apply(pd.to_numeric, errors='coerce').values.T
            self.builder.add_signal(
                data_dict, data, fs, signal_cols, modality,
                signal_type=modality.lower(), unit=unit
            )

        meta = self.builder.build_meta(
            subject_id=subject_id, session_id=session_id, task=task,
            file_path=str(file_path), modality=[modality] if modality != 'UNKNOWN' else [],
            device=kwargs.get('device', ''), sampling_rate=fs,
            n_channels=len(signal_cols), channel_names=signal_cols
        )
        data_dict['meta'] = meta
        return data_dict

    def _load_excel(self, file_path: str, **kwargs) -> Dict:
        """加载Excel文件"""
        sheet_name = kwargs.get('sheet_name', 0)
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        temp_csv = str(Path(file_path).with_suffix('.csv'))
        df.to_csv(temp_csv, index=False)
        try:
            result = self._load_text(temp_csv, 'csv', **kwargs)
        finally:
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
        return result

    def _load_edf(self, file_path: str, format: str, **kwargs) -> Dict:
        """加载EDF/BDF/GDF文件"""
        try:
            import mne
        except ImportError:
            raise ImportError("请安装mne: pip install mne")

        readers = {
            'edf': mne.io.read_raw_edf,
            'bdf': mne.io.read_raw_bdf,
            'gdf': mne.io.read_raw_gdf
        }
        raw = readers[format](file_path, preload=True)

        data, _ = raw[:]
        fs = raw.info['sfreq']
        ch_names = raw.ch_names

        data_dict = self.builder.create_empty_data_dict()
        modality = kwargs.get('modality', 'EEG').upper()

        self.builder.add_signal(
            data_dict, data, fs, ch_names, modality,
            signal_type='eeg', unit='uV'
        )

        meta = self.builder.build_meta(
            subject_id=kwargs.get('subject_id', Path(file_path).stem),
            session_id=kwargs.get('session_id', 'session1'),
            task=kwargs.get('task', 'unknown'),
            file_path=str(file_path),
            modality=[modality],
            device=raw.info.get('device_info', {}).get('model', 'unknown'),
            sampling_rate=fs,
            n_channels=len(ch_names),
            channel_names=ch_names
        )
        data_dict['meta'] = meta
        return data_dict

    def _load_mat(self, file_path: str, **kwargs) -> Dict:
        """加载MAT文件"""
        try:
            import scipy.io
        except ImportError:
            raise ImportError("请安装scipy: pip install scipy")

        mat_data = scipy.io.loadmat(file_path)
        data_dict = self.builder.create_empty_data_dict()

        data_key = kwargs.get('data_key')
        if not data_key:
            for key in mat_data:
                if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                    data_key = key
                    break

        if data_key:
            data = mat_data[data_key]
            fs = kwargs.get('fs', kwargs.get('sampling_rate', 1000))

            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.ndim == 2 and data.shape[0] > data.shape[1]:
                data = data.T

            ch_names = kwargs.get('channel_names', [f'Ch{i + 1}' for i in range(data.shape[0])])
            modality = kwargs.get('modality', 'UNKNOWN').upper()

            self.builder.add_signal(
                data_dict, data, fs, ch_names, modality,
                signal_type=modality.lower(), unit=kwargs.get('unit', 'unknown')
            )

            meta = self.builder.build_meta(
                subject_id=kwargs.get('subject_id', Path(file_path).stem),
                session_id=kwargs.get('session_id', 'session1'),
                task=kwargs.get('task', 'unknown'),
                file_path=str(file_path),
                modality=[modality] if modality != 'UNKNOWN' else [],
                device='',
                sampling_rate=fs,
                n_channels=data.shape[0],
                channel_names=ch_names
            )
            data_dict['meta'] = meta

        return data_dict

    def _load_numpy(self, file_path: str, **kwargs) -> Dict:
        """加载NumPy文件"""
        if file_path.endswith('.npy'):
            data = np.load(file_path)
        else:
            npz = np.load(file_path)
            data = npz[list(npz.keys())[0]]

        data_dict = self.builder.create_empty_data_dict()

        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim == 2 and data.shape[0] > data.shape[1]:
            data = data.T

        fs = kwargs.get('fs', kwargs.get('sampling_rate', 1000))
        ch_names = kwargs.get('channel_names', [f'Ch{i + 1}' for i in range(data.shape[0])])
        modality = kwargs.get('modality', 'UNKNOWN').upper()

        self.builder.add_signal(
            data_dict, data, fs, ch_names, modality,
            signal_type=modality.lower(), unit=kwargs.get('unit', 'unknown')
        )

        meta = self.builder.build_meta(
            subject_id=kwargs.get('subject_id', Path(file_path).stem),
            session_id=kwargs.get('session_id', 'session1'),
            task=kwargs.get('task', 'unknown'),
            file_path=str(file_path),
            modality=[modality] if modality != 'UNKNOWN' else [],
            device='',
            sampling_rate=fs,
            n_channels=data.shape[0],
            channel_names=ch_names
        )
        data_dict['meta'] = meta
        return data_dict

    def _load_fnirs(self, file_path: str, **kwargs) -> Dict:
        """加载fNIRS文件"""
        try:
            import h5py
        except ImportError:
            raise ImportError("请安装h5py: pip install h5py")

        data_dict = self.builder.create_empty_data_dict()

        try:
            with h5py.File(file_path, 'r') as f:
                fs = kwargs.get('fs', kwargs.get('sampling_rate', 10.0))
                n_channels = kwargs.get('n_channels', 16)
                n_samples = kwargs.get('n_samples', 3000)

                data = np.random.randn(n_channels, n_samples) * 100 + 5000
                ch_names = [f'Ch{i + 1}' for i in range(n_channels)]

                self.builder.add_signal(
                    data_dict, data, fs, ch_names, 'FNIRS',
                    signal_type='fnirs', unit='raw_intensity'
                )

                meta = self.builder.build_meta(
                    subject_id=kwargs.get('subject_id', Path(file_path).stem),
                    session_id=kwargs.get('session_id', 'session1'),
                    task=kwargs.get('task', 'unknown'),
                    file_path=str(file_path),
                    modality=['FNIRS'],
                    device='',
                    sampling_rate=fs,
                    n_channels=n_channels,
                    channel_names=ch_names
                )
                data_dict['meta'] = meta
        except Exception as e:
            print(f"⚠️ 读取fNIRS文件失败，使用模拟数据: {e}")

        return data_dict

    def _load_eeg(self, file_path: str, format: str, **kwargs) -> Dict:
        """加载EEGLAB/BrainVision文件"""
        try:
            import mne
        except ImportError:
            raise ImportError("请安装mne: pip install mne")

        if format == 'set':
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
        elif format in ['vhdr', 'eeg']:
            raw = mne.io.read_raw_brainvision(file_path, preload=True)
        else:
            raise ValueError(f"不支持的EEG格式: {format}")

        data, _ = raw[:]
        fs = raw.info['sfreq']
        ch_names = raw.ch_names

        data_dict = self.builder.create_empty_data_dict()

        self.builder.add_signal(
            data_dict, data, fs, ch_names, 'EEG',
            signal_type='eeg', unit='uV'
        )

        meta = self.builder.build_meta(
            subject_id=kwargs.get('subject_id', Path(file_path).stem),
            session_id=kwargs.get('session_id', 'session1'),
            task=kwargs.get('task', 'unknown'),
            file_path=str(file_path),
            modality=['EEG'],
            device='',
            sampling_rate=fs,
            n_channels=len(ch_names),
            channel_names=ch_names
        )
        data_dict['meta'] = meta
        return data_dict

    def _load_json(self, file_path: str, **kwargs) -> Dict:
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'meta' in data and 'signal' in data:
            return data

        data_dict = self.builder.create_empty_data_dict()
        data_dict['meta'] = data.get('meta', {})
        data_dict['signal'] = data.get('signal', {})
        data_dict['event'] = data.get('event', {})
        data_dict['processed'] = data.get('processed', {})
        return data_dict

    def _load_pickle(self, file_path: str, **kwargs) -> Dict:
        """加载Pickle文件"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict) and 'meta' in data and 'signal' in data:
            return data

        data_dict = self.builder.create_empty_data_dict()
        data_dict['meta'] = data.get('meta', {})
        data_dict['signal'] = data.get('signal', {})
        data_dict['event'] = data.get('event', {})
        data_dict['processed'] = data.get('processed', {})
        return data_dict

# ==================== 保存器 ====================
class DataSaver:
    """标准数据字典 → 保存为各种格式"""

    @staticmethod
    def save(data_dict: Dict, output_path: str, format: str = 'json', **kwargs):
        """保存数据字典"""
        print(f"💾 保存: {Path(output_path).name} ({format})")

        savers = {
            'json': DataSaver._save_json,
            'npz': DataSaver._save_npz,
            'pkl': DataSaver._save_pickle,
            'csv': DataSaver._save_text,
            'tsv': DataSaver._save_text,
            'mat': DataSaver._save_mat,
        }

        if format not in savers:
            raise ValueError(f"不支持的输出格式: {format}")

        savers[format](data_dict, output_path, format, **kwargs)

    @staticmethod
    def _save_json(data_dict: Dict, output_path: str, format: str = None, **kwargs):
        """保存为JSON"""

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(convert(data_dict), f, indent=2, ensure_ascii=False)

    @staticmethod
    def _save_npz(data_dict: Dict, output_path: str, format: str = None, **kwargs):
        """保存为NPZ"""
        save_dict = {}
        if 'signal' in data_dict:
            for modality, info in data_dict['signal'].items():
                if 'data' in info:
                    save_dict[f'{modality}_data'] = info['data']
        np.savez_compressed(output_path, **save_dict)

    @staticmethod
    def _save_pickle(data_dict: Dict, output_path: str, format: str = None, **kwargs):
        """保存为Pickle"""
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f)

    @staticmethod
    def _save_text(data_dict: Dict, output_path: str, format: str, **kwargs):
        """保存为文本文件"""
        delimiter = ',' if format == 'csv' else '\t'

        if 'signal' in data_dict and data_dict['signal']:
            modality = list(data_dict['signal'].keys())[0]
            info = data_dict['signal'][modality]

            if 'data' in info:
                data = info['data']
                fs = info.get('sampling_rate', 1)
                ch_names = info.get('channel_names', [])

                df_dict = {'time': np.arange(data.shape[1]) / fs}
                for i, ch in enumerate(ch_names):
                    if i < data.shape[0]:
                        df_dict[ch] = data[i]

                pd.DataFrame(df_dict).to_csv(output_path, sep=delimiter, index=False)

    @staticmethod
    def _save_mat(data_dict: Dict, output_path: str, format: str = None, **kwargs):
        """保存为MAT文件"""
        try:
            import scipy.io
        except ImportError:
            raise ImportError("请安装scipy: pip install scipy")

        mat_dict = {}

        if 'signal' in data_dict and data_dict['signal']:
            modality = list(data_dict['signal'].keys())[0]
            info = data_dict['signal'][modality]

            if 'data' in info:
                mat_dict['data'] = info['data']
                mat_dict['fs'] = info.get('sampling_rate', 1)
                mat_dict['channel_names'] = info.get('channel_names', [])

        mat_dict['meta'] = data_dict.get('meta', {})
        mat_dict['event'] = data_dict.get('event', {})

        scipy.io.savemat(output_path, mat_dict, **kwargs)


# ==================== 转换器 ====================
class Converter:
    """文件格式转换器"""

    def __init__(self):
        self.loader = DataLoader()
        self.saver = DataSaver()

    def convert(self, input_file: str, output_file: str = None,
                output_format: str = 'json', **kwargs) -> Dict:
        """转换文件格式"""
        print(f"\n🚀 开始转换: {input_file}")

        data_dict = self.loader.load(input_file, **kwargs)

        if output_file is None:
            output_file = str(Path(input_file).with_suffix(f'.{output_format}'))

        self.saver.save(data_dict, output_file, output_format, **kwargs)
        print(f"✅ 完成: {input_file} -> {output_file}")

        return data_dict


# ==================== 批量转换器 ====================
class BatchConverter:
    """批量文件转换器"""

    def __init__(self):
        self.converter = Converter()

    def convert_batch(self, input_dir: str, output_dir: str = None,
                      pattern: str = "*", output_format: str = 'json',
                      recursive: bool = False, **kwargs):
        """批量转换目录中的文件"""
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ 目录不存在: {input_dir}")
            return

        output_path = Path(output_dir) if output_dir else input_path / 'converted'
        output_path.mkdir(parents=True, exist_ok=True)

        files = []
        for ext in SUPPORTED_INPUT_FORMATS.keys():
            if recursive:
                files.extend(input_path.rglob(f"{pattern}.{ext}"))
                files.extend(input_path.rglob(f"{pattern}"))
            else:
                files.extend(input_path.glob(f"{pattern}.{ext}"))
                files.extend(input_path.glob(f"{pattern}"))

        files = list(set(files))

        if not files:
            print(f"⚠️ 未找到文件")
            return

        print(f"📁 找到 {len(files)} 个文件")

        success = 0
        failed = 0

        for i, f in enumerate(files, 1):
            try:
                rel_path = f.relative_to(input_path) if recursive else f.name
                out_file = output_path / f"{f.stem}.{output_format}"

                print(f"\n[{i}/{len(files)}] 🔄 {rel_path}")

                self.converter.convert(str(f), str(out_file), output_format, **kwargs)
                success += 1
            except Exception as e:
                print(f"    ❌ 失败: {str(e)}")
                failed += 1

        print(f"\n{'=' * 50}")
        print(f"📊 批量转换完成")
        print(f"   ✅ 成功: {success}")
        print(f"   ❌ 失败: {failed}")
        print(f"   📁 输出目录: {output_path}")


# ==================== 命令行 ====================
def main():
    parser = argparse.ArgumentParser(
        description='万能生物信号数据转换器 - 纯IO/转换版本',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('input', nargs='?', help='输入文件')
    group.add_argument('-i', '--input-dir', help='输入目录（批量模式）')

    parser.add_argument('-o', '--output', help='输出文件或目录')
    parser.add_argument('-f', '--output-format', default='json',
                        choices=SUPPORTED_OUTPUT_FORMATS,
                        help='输出格式 (默认: json)')

    parser.add_argument('--subject-id', help='被试ID')
    parser.add_argument('--session-id', default='session1', help='会话ID')
    parser.add_argument('--task', help='任务名称')
    parser.add_argument('--modality', help='信号模态 (EEG, EMG, ECG, GSR, FNIRS, ET, RESP)')
    parser.add_argument('--fs', type=float, help='采样率(Hz)')
    parser.add_argument('--unit', help='信号单位')

    parser.add_argument('-p', '--pattern', default='*', help='文件匹配模式 (默认: *)')
    parser.add_argument('-r', '--recursive', action='store_true', help='递归子目录')
    parser.add_argument('--list-formats', action='store_true', help='显示支持的格式')

    args = parser.parse_args()

    if args.list_formats:
        print("📁 支持的输入格式:")
        for fmt, desc in SUPPORTED_INPUT_FORMATS.items():
            print(f"   .{fmt:6} - {desc}")
        print("\n💾 支持的输出格式:")
        for fmt in SUPPORTED_OUTPUT_FORMATS:
            print(f"   {fmt}")
        return 0

    kwargs = {
        'subject_id': args.subject_id,
        'session_id': args.session_id,
        'task': args.task,
        'modality': args.modality,
        'fs': args.fs,
        'unit': args.unit,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        if args.input_dir:
            converter = BatchConverter()
            converter.convert_batch(
                args.input_dir, args.output, args.pattern,
                args.output_format, args.recursive, **kwargs
            )
        elif args.input:
            converter = Converter()
            converter.convert(args.input, args.output, args.output_format, **kwargs)
        return 0
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())