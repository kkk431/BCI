#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
优化版BDF/EDF读取器
输出符合四层结构的 data_dict
"""

import mne
import numpy as np
import os
from datetime import datetime
from typing import Dict, List


class BDFReader:
    """读取BDF/EDF文件，转换为统一数据字典格式"""

    def __init__(self, file_path: str, subject_id: str = None, session_id: str = None):
        """
        初始化BDF读取器

        参数:
            file_path: BDF/EDF文件路径
            subject_id: 被试ID（可选，从文件名自动提取）
            session_id: 会话ID（可选，从文件名自动提取）
        """
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self.file_dir = os.path.dirname(file_path)

        # 如果没有提供，从文件名提取元信息
        if subject_id is None:
            self.subject_id = self._extract_subject_id()
        else:
            self.subject_id = subject_id

        if session_id is None:
            self.session_id = self._extract_session_id()
        else:
            self.session_id = session_id

    def _extract_subject_id(self) -> str:
        """从文件名提取被试ID"""
        # 简单示例：假设文件名为 "S001_session1_data.bdf"
        import re
        match = re.search(r'([Ss]\d+)', self.filename)
        return match.group(1) if match else 'unknown_subject'

    def _extract_session_id(self) -> str:
        """从文件名提取会话ID"""
        import re
        match = re.search(r'session[_\s]*(\d+)', self.filename, re.IGNORECASE)
        if match:
            return f"session_{match.group(1)}"
        # 如果没有找到，使用文件修改时间
        mtime = os.path.getmtime(self.file_path)
        return datetime.fromtimestamp(mtime).strftime('%Y%m%d_%H%M%S')

    def read_to_data_dict(self) -> Dict:
        """
        读取BDF/EDF文件，转换为四层结构data_dict

        返回:
            符合统一格式的数据字典
        """
        # 1. 读取原始数据
        raw = mne.io.read_raw_bdf(self.file_path, preload=True)

        # 2. 构建meta层
        data_dict = self._create_meta_layer(raw)

        # 3. 构建signal层
        data_dict['signal'] = self._create_signal_layer(raw)

        # 4. 构建event层
        data_dict['event'] = self._create_event_layer(raw)

        # 5. 初始化processed层
        data_dict['processed'] = {}

        # 6. 添加系统信息
        data_dict['system'] = {
            'source_file': self.file_path,
            'read_time': datetime.now().isoformat(),
            'file_size_mb': os.path.getsize(self.file_path) / (1024 * 1024)
        }

        return data_dict

    def _create_meta_layer(self, raw) -> Dict:
        """创建meta层信息"""
        # 从raw.info提取信息
        info = raw.info

        return {
            'subject_id': self.subject_id,
            'session_id': self.session_id,
            'task': 'unknown',  # 可以从文件名或注释中提取
            'modality': ['EEG'],  # 默认，可以检测是否有EOG/EMG
            'device': 'Neuracle',  # 假设
            'sampling_rate': float(info['sfreq']),
            'n_channels': info['nchan'],
            'channel_names': info['ch_names'],
            'recording_date': info['meas_date'].isoformat() if info['meas_date'] else None,
            'file_creation_time': datetime.now().isoformat(),
            'montage': str(info.get('dig', 'unknown')),
            'highpass': float(info['highpass']),
            'lowpass': float(info['lowpass'])
        }

    def _create_signal_layer(self, raw) -> Dict:
        """创建signal层，支持多模态分离"""
        signals = {}

        # 获取所有数据
        data, times = raw[:]

        # 分离不同模态（简化版，实际需要根据通道名判断）
        eeg_channels = []
        eog_channels = []
        ecg_channels = []
        emg_channels = []
        trigger_channels = []

        for i, ch_name in enumerate(raw.info['ch_names']):
            ch_name_lower = ch_name.lower()

            if 'eog' in ch_name_lower or 'v' in ch_name_lower or 'h' in ch_name_lower:
                eog_channels.append(i)
            elif 'ecg' in ch_name_lower:
                ecg_channels.append(i)
            elif 'emg' in ch_name_lower:
                emg_channels.append(i)
            elif 'trig' in ch_name_lower or 'stim' in ch_name_lower:
                trigger_channels.append(i)
            else:
                eeg_channels.append(i)

        # EEG信号
        if eeg_channels:
            signals['EEG'] = {
                'data': data[eeg_channels, :],
                'sampling_rate': float(raw.info['sfreq']),
                'unit': 'uV',  # BDF默认单位
                'channel_names': [raw.info['ch_names'][i] for i in eeg_channels],
                'reference': self._detect_reference(raw),
                'time_offset': 0.0,
                'montage': 'standard_1020'  # 假设
            }

        # EOG信号
        if eog_channels:
            signals['EOG'] = {
                'data': data[eog_channels, :],
                'sampling_rate': float(raw.info['sfreq']),
                'unit': 'uV',
                'channel_names': [raw.info['ch_names'][i] for i in eog_channels],
                'time_offset': 0.0
            }

        # 触发通道
        if trigger_channels:
            signals['TRIGGER'] = {
                'data': data[trigger_channels, :],
                'sampling_rate': float(raw.info['sfreq']),
                'unit': 'digital',
                'channel_names': [raw.info['ch_names'][i] for i in trigger_channels],
                'time_offset': 0.0
            }

        return signals

    def _detect_reference(self, raw) -> str:
        """检测参考电极类型"""
        # 简化检测逻辑
        ch_names = [ch.lower() for ch in raw.info['ch_names']]

        if any('ref' in ch or 'm1' in ch or 'm2' in ch for ch in ch_names):
            return 'mastoid'
        elif any('avg' in ch or 'average' in ch for ch in ch_names):
            return 'average'
        else:
            return 'unknown'

    def _create_event_layer(self, raw) -> Dict:
        """创建event层，解析注释和触发"""
        events = {
            'annotations': [],
            'triggers': [],
            'stimuli': []
        }

        try:
            # 1. 从注释中获取事件
            if raw.annotations:
                for ann in raw.annotations:
                    events['annotations'].append({
                        'onset': float(ann['onset']),
                        'duration': float(ann['duration']),
                        'description': ann['description'],
                        'type': 'annotation'
                    })

            # 2. 从触发通道获取事件（如果存在）
            trigger_data = self._extract_trigger_events(raw)
            if trigger_data:
                events['triggers'] = trigger_data

            # 3. 查找刺激事件
            events['stimuli'] = self._extract_stimulus_events(raw)

        except Exception as e:
            print(f"事件解析错误: {e}")

        # 转换为DataFrame友好格式
        events['dataframe_format'] = self._events_to_dataframe_format(events)

        return events

    def _extract_trigger_events(self, raw) -> List:
        """从触发通道提取事件"""
        triggers = []

        # 查找可能的触发通道
        trigger_ch_idx = []
        for i, ch_name in enumerate(raw.info['ch_names']):
            if 'TRIG' in ch_name.upper() or 'STIM' in ch_name.upper():
                trigger_ch_idx.append(i)

        if not trigger_ch_idx:
            return triggers

        # 获取触发通道数据
        trigger_data, _ = raw[trigger_ch_idx, :]

        # 简单阈值检测触发
        for ch_idx, global_idx in enumerate(trigger_ch_idx):
            ch_data = trigger_data[ch_idx, :]

            # 找到超过阈值的点
            threshold = np.max(ch_data) * 0.5
            trigger_samples = np.where(ch_data > threshold)[0]

            if len(trigger_samples) > 0:
                # 合并连续的触发
                diff_samples = np.diff(trigger_samples)
                break_points = np.where(diff_samples > 1)[0] + 1

                trigger_groups = np.split(trigger_samples, break_points)

                for group in trigger_groups:
                    if len(group) > 0:
                        onset_sample = group[0]
                        duration_samples = len(group)

                        triggers.append({
                            'onset_sample': int(onset_sample),
                            'onset_time': float(onset_sample / raw.info['sfreq']),
                            'duration_samples': int(duration_samples),
                            'channel': raw.info['ch_names'][global_idx],
                            'value': float(np.mean(ch_data[group])),
                            'type': 'trigger'
                        })

        return triggers

    def _extract_stimulus_events(self, raw) -> List:
        """从注释中提取刺激事件"""
        stimuli = []

        if not raw.annotations:
            return stimuli

        stimulus_keywords = ['stim', 'trial', 'onset', 'target', 'cue']

        for ann in raw.annotations:
            desc_lower = ann['description'].lower()

            if any(keyword in desc_lower for keyword in stimulus_keywords):
                stimuli.append({
                    'onset': float(ann['onset']),
                    'duration': float(ann['duration']),
                    'description': ann['description'],
                    'type': 'stimulus'
                })

        return stimuli

    def _events_to_dataframe_format(self, events_dict: Dict) -> Dict:
        """将事件转换为DataFrame友好格式"""
        all_events = []

        # 合并所有事件类型
        for event_type in ['annotations', 'triggers', 'stimuli']:
            all_events.extend(events_dict.get(event_type, []))

        if not all_events:
            return {'onset': [], 'duration': [], 'trial_type': [], 'value': []}

        # 排序
        all_events.sort(key=lambda x: x.get('onset', x.get('onset_time', 0)))

        # 构建DataFrame格式
        return {
            'onset': [e.get('onset', e.get('onset_time', 0)) for e in all_events],
            'duration': [e.get('duration', e.get('duration_samples', 0) / self.meta['sampling_rate'])
                         for e in all_events],
            'trial_type': [e.get('type', 'unknown') for e in all_events],
            'value': [e.get('value', 1.0) for e in all_events],
            'description': [e.get('description', '') for e in all_events]
        }

    def save_as_unified_format(self, output_path: str = None) -> str:
        """
        读取并保存为统一格式文件

        参数:
            output_path: 输出文件路径（可选）

        返回:
            保存的文件路径
        """
        # 1. 读取数据
        data_dict = self.read_to_data_dict()

        # 2. 确定输出路径
        if output_path is None:
            base_name = os.path.splitext(self.filename)[0]
            output_path = os.path.join(self.file_dir, f"{base_name}_unified.h5")

        # 3. 保存为HDF5（使用之前的保存方法）
        from core.io.trigger_system.data_server import UnifiedDataServer
        # 创建临时服务器实例用于保存
        temp_server = UnifiedDataServer(
            device_config={'device_type': 'Neuracle'},
            data_dict=data_dict,
            buffer_seconds=1.0
        )

        temp_server.save_data(output_path, format='hdf5')

        print(f"文件已保存为统一格式: {output_path}")
        return output_path


# 向后兼容的包装函数
def readbdfdata_to_dict(filename, pathname):
    """
    向后兼容的函数，返回符合四层结构的data_dict

    参数:
        filename: 文件名列表
        pathname: 路径列表

    返回:
        统一格式的data_dict
    """
    # 构建完整文件路径
    if isinstance(filename, list) and len(filename) > 0:
        file_path = os.path.join(pathname[0], filename[0])
    else:
        file_path = os.path.join(pathname, filename)

    # 创建读取器并读取
    reader = BDFReader(file_path)
    return reader.read_to_data_dict()


# 测试代码
if __name__ == "__main__":
    # 示例使用
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # 如果没有提供参数，假设有一个测试文件
        file_path = "test_data.bdf"
        print(f"警告: 使用默认文件路径 {file_path}")
        print("请提供BDF文件路径作为参数: python bdf_reader_v2.py your_file.bdf")

    if os.path.exists(file_path):
        # 创建读取器
        reader = BDFReader(file_path)

        # 读取数据
        print(f"读取文件: {file_path}")
        data_dict = reader.read_to_data_dict()

        # 显示摘要
        print("\n数据摘要:")
        print(f"  被试ID: {data_dict['meta']['subject_id']}")
        print(f"  采样率: {data_dict['meta']['sampling_rate']} Hz")
        print(f"  通道数: {data_dict['meta']['n_channels']}")

        print(f"\n信号模态: {list(data_dict['signal'].keys())}")
        for modality, info in data_dict['signal'].items():
            print(f"  {modality}: {info['data'].shape[0]}通道, {info['data'].shape[1]}样本")

        print(f"\n事件数量:")
        total_events = len(data_dict['event']['dataframe_format']['onset'])
        print(f"  总共: {total_events}个事件")

        # 保存为统一格式
        output_file = reader.save_as_unified_format()
        print(f"\n已保存为统一格式: {output_file}")

    else:
        print(f"文件不存在: {file_path}")
        print("请确保文件路径正确")