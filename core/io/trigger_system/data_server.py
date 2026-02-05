#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
优化版数据接收服务器
基于统一四层结构 data_dict 格式
支持多模态信号、事件管理、实时处理
"""

import socket
from struct import unpack
import numpy as np
from threading import Lock, Thread, Event
import select
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import warnings


# ==================== 环形缓冲区类 (增强版) ====================
class EnhancedRingBuffer:
    """增强版环形缓冲区，支持多模态信号独立存储"""

    def __init__(self, buffer_config: Dict[str, Dict]):
        """
        初始化多模态环形缓冲区

        参数:
            buffer_config: 缓冲区配置字典
                示例: {
                    'EEG': {'n_chan': 64, 'n_points': 3000},
                    'EOG': {'n_chan': 2, 'n_points': 3000},
                    'TRIGGER': {'n_chan': 1, 'n_points': 3000}
                }
        """
        self.buffers = {}
        self.pointers = {}
        self.update_counts = {}

        for modality, config in buffer_config.items():
            n_chan = config['n_chan']
            n_points = config['n_points']
            self.buffers[modality] = np.zeros((n_chan, n_points))
            self.pointers[modality] = 0
            self.update_counts[modality] = 0

    def append_buffer(self, modality: str, data: np.ndarray):
        """向指定模态的缓冲区追加数据"""
        if modality not in self.buffers:
            raise ValueError(f"未知的模态类型: {modality}")

        buffer = self.buffers[modality]
        n = data.shape[1]
        current_ptr = self.pointers[modality]
        n_points = buffer.shape[1]

        # 环形写入
        indices = np.mod(np.arange(current_ptr, current_ptr + n), n_points)
        buffer[:, indices] = data

        # 更新指针和计数
        self.pointers[modality] = (current_ptr + n) % n_points
        self.update_counts[modality] += n

    def get_data(self, modality: str, n_samples: Optional[int] = None) -> np.ndarray:
        """获取指定模态的数据（时间顺序）"""
        if modality not in self.buffers:
            raise ValueError(f"未知的模态类型: {modality}")

        buffer = self.buffers[modality]
        current_ptr = self.pointers[modality]

        if n_samples is None or n_samples > buffer.shape[1]:
            n_samples = buffer.shape[1]

        # 获取最近n_samples个点（时间顺序）
        indices = np.mod(np.arange(current_ptr - n_samples, current_ptr), buffer.shape[1])
        return buffer[:, indices]

    def get_all_data(self, modality: str) -> np.ndarray:
        """获取指定模态的全部数据（按时间顺序重新排列）"""
        buffer = self.buffers[modality]
        current_ptr = self.pointers[modality]

        # 拼接数据以获得时间线性顺序
        return np.hstack([buffer[:, current_ptr:], buffer[:, :current_ptr]])

    def reset_modality(self, modality: str):
        """重置指定模态的缓冲区"""
        if modality in self.buffers:
            self.buffers[modality][:] = 0
            self.pointers[modality] = 0
            self.update_counts[modality] = 0

    def reset_all(self):
        """重置所有缓冲区"""
        for modality in self.buffers:
            self.reset_modality(modality)


# ==================== 事件管理器类 ====================
class EventManager:
    """事件管理器，负责事件的存储、查询和同步"""

    def __init__(self, sampling_rate: float):
        self.sampling_rate = sampling_rate
        self.events = []  # 存储事件字典的列表
        self.event_counter = 0
        self.lock = Lock()

        # 事件类型定义
        self.event_types = {
            'TRIGGER': '硬件触发',
            'STIMULUS': '刺激呈现',
            'RESPONSE': '被试反应',
            'MARKER': '实验标记',
            'SYSTEM': '系统事件',
            'ARTIFACT': '伪迹标记'
        }

    def add_event(self,
                  event_type: str,
                  sample_index: int,
                  value: Any = None,
                  description: str = "",
                  metadata: Dict = None) -> int:
        """
        添加事件

        参数:
            event_type: 事件类型
            sample_index: 采样点索引（相对于数据开始）
            value: 事件值
            description: 事件描述
            metadata: 额外元数据

        返回:
            事件ID
        """
        with self.lock:
            event_id = self.event_counter
            timestamp = sample_index / self.sampling_rate

            event = {
                'id': event_id,
                'type': event_type,
                'sample_index': sample_index,
                'timestamp': timestamp,
                'value': value,
                'description': description,
                'metadata': metadata or {}
            }

            self.events.append(event)
            self.event_counter += 1

            # 按时间排序
            self.events.sort(key=lambda x: x['sample_index'])

            return event_id

    def get_events_in_range(self,
                            start_sample: int,
                            end_sample: int,
                            event_type: Optional[str] = None) -> List[Dict]:
        """获取指定时间范围内的事件"""
        with self.lock:
            filtered = [
                event for event in self.events
                if start_sample <= event['sample_index'] <= end_sample
            ]

            if event_type:
                filtered = [e for e in filtered if e['type'] == event_type]

            return filtered

    def get_latest_events(self, n: int = 10) -> List[Dict]:
        """获取最近的n个事件"""
        with self.lock:
            return self.events[-n:] if self.events else []

    def clear_events(self):
        """清空所有事件"""
        with self.lock:
            self.events.clear()
            self.event_counter = 0

    def to_dataframe_format(self):
        """转换为DataFrame友好格式（用于保存）"""
        with self.lock:
            if not self.events:
                return {'onset': [], 'duration': [], 'trial_type': [], 'value': []}

            return {
                'onset': [e['timestamp'] for e in self.events],
                'duration': [0] * len(self.events),  # 瞬时事件
                'trial_type': [e['type'] for e in self.events],
                'value': [e['value'] for e in self.events],
                'description': [e['description'] for e in self.events]
            }


# ==================== 数据处理管道类 ====================
class ProcessingPipeline:
    """实时数据处理管道"""

    def __init__(self):
        self.processors = []
        self.enabled = True

    def add_processor(self, name: str, processor_func, config: Dict = None):
        """添加处理器"""
        self.processors.append({
            'name': name,
            'func': processor_func,
            'config': config or {},
            'enabled': True
        })

    def process(self, data_dict: Dict, modality: str = 'EEG'):
        """执行处理管道"""
        if not self.enabled:
            return data_dict

        processed_data = data_dict.copy()

        for processor in self.processors:
            if processor['enabled']:
                try:
                    if 'signal' in processed_data and modality in processed_data['signal']:
                        result = processor['func'](
                            processed_data['signal'][modality]['data'],
                            **processor['config']
                        )
                        # 存储处理结果
                        if 'processed' not in processed_data:
                            processed_data['processed'] = {}

                        proc_key = f"{modality}_{processor['name']}"
                        processed_data['processed'][proc_key] = result
                except Exception as e:
                    warnings.warn(f"处理器 {processor['name']} 失败: {str(e)}")

        return processed_data

    def enable_processor(self, name: str, enabled: bool = True):
        """启用/禁用处理器"""
        for processor in self.processors:
            if processor['name'] == name:
                processor['enabled'] = enabled
                break

    def clear_processors(self):
        """清空所有处理器"""
        self.processors.clear()


# ==================== 统一数据服务器类 ====================
class UnifiedDataServer(Thread):
    """
    统一数据服务器
    基于四层结构 data_dict 格式
    """

    def __init__(self,
                 device_config: Dict[str, Any],
                 data_dict: Dict[str, Any],
                 buffer_seconds: float = 5.0):
        """
        初始化

        参数:
            device_config: 设备配置
            data_dict: 统一数据字典（四层结构）
            buffer_seconds: 缓冲区秒数
        """
        Thread.__init__(self)
        self.device_config = device_config
        self.unified_data = data_dict
        self.buffer_seconds = buffer_seconds

        # 设备信息
        self.device = device_config.get('device_type', 'Neuracle')
        self.host = device_config.get('host', '127.0.0.1')
        self.port = device_config.get('port', 8712)

        # 从 meta 获取配置
        meta = data_dict.get('meta', {})
        self.sampling_rate = meta.get('sampling_rate', 1000)
        self.n_channels = meta.get('n_channels', 64)

        # 信号配置
        signal_config = data_dict.get('signal', {})
        self.modalities = list(signal_config.keys())

        # 初始化组件
        self._init_components()

        # 状态变量
        self.is_connected = False
        self.is_recording = False
        self.total_samples = 0

        # 性能监控
        self.data_rate = 0
        self.last_update_time = time.time()

    def _init_components(self):
        """初始化各组件"""
        # 1. 初始化环形缓冲区配置
        buffer_config = {}
        signal_config = self.unified_data.get('signal', {})

        for modality, config in signal_config.items():
            n_chan = config.get('data', np.zeros((1, 1))).shape[0]
            n_points = int(self.buffer_seconds * config.get('sampling_rate', self.sampling_rate))
            buffer_config[modality] = {'n_chan': n_chan, 'n_points': n_points}

        # 添加触发通道
        buffer_config['TRIGGER'] = {'n_chan': 1, 'n_points': int(self.buffer_seconds * self.sampling_rate)}

        # 创建增强环形缓冲区
        self.ring_buffer = EnhancedRingBuffer(buffer_config)

        # 2. 初始化事件管理器
        self.event_manager = EventManager(self.sampling_rate)

        # 3. 初始化处理管道
        self.processing_pipeline = ProcessingPipeline()

        # 4. Socket 相关
        self.sock = None
        self.shutdown_flag = Event()
        self.shutdown_flag.set()
        self.socket_lock = Lock()

        # 5. 临时缓冲区
        self.raw_buffer = b''

    def connect(self) -> bool:
        """连接到设备服务器"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)

            print(f"正在连接到 {self.host}:{self.port}...")
            self.sock.connect((self.host, self.port))

            # 设置非阻塞
            self.sock.setblocking(False)

            self.is_connected = True
            print("连接成功")

            # 更新连接状态
            if 'system' not in self.unified_data:
                self.unified_data['system'] = {}
            self.unified_data['system']['connection_status'] = 'connected'
            self.unified_data['system']['connection_time'] = datetime.now().isoformat()

            return True

        except Exception as e:
            print(f"连接失败: {str(e)}")
            self.is_connected = False
            return False

    def disconnect(self):
        """断开连接"""
        with self.socket_lock:
            if self.sock:
                self.sock.close()
                self.sock = None

            self.is_connected = False

            # 更新状态
            if 'system' in self.unified_data:
                self.unified_data['system']['connection_status'] = 'disconnected'
                self.unified_data['system']['disconnection_time'] = datetime.now().isoformat()

    def start_recording(self):
        """开始记录"""
        self.is_recording = True

        # 初始化记录数据
        if 'record' not in self.unified_data:
            self.unified_data['record'] = {
                'start_time': datetime.now().isoformat(),
                'samples_recorded': 0,
                'data': {}
            }

        # 为每个模态初始化记录数组
        for modality in self.modalities:
            if modality in self.unified_data.get('signal', {}):
                self.unified_data['record']['data'][modality] = []

        print("开始记录数据")

    def stop_recording(self):
        """停止记录"""
        self.is_recording = False

        if 'record' in self.unified_data:
            self.unified_data['record']['end_time'] = datetime.now().isoformat()
            self.unified_data['record']['duration'] = (
                    datetime.fromisoformat(self.unified_data['record']['end_time']) -
                    datetime.fromisoformat(self.unified_data['record']['start_time'])
            ).total_seconds()

        print("停止记录数据")

    def run(self):
        """主线程运行方法"""
        if not self.is_connected:
            print("未连接，无法启动数据接收")
            return

        print("启动数据接收线程")

        while self.shutdown_flag.is_set():
            try:
                # 使用select监听socket
                ready_to_read, _, _ = select.select([self.sock], [], [], 1.0)

                if ready_to_read:
                    with self.socket_lock:
                        if not self.sock:
                            break

                        # 接收数据
                        raw_data = self.sock.recv(4096)

                        if raw_data:
                            self._process_raw_data(raw_data)

                            # 更新数据率
                            current_time = time.time()
                            time_diff = current_time - self.last_update_time
                            if time_diff > 0:
                                self.data_rate = len(raw_data) / time_diff / 1024  # KB/s
                            self.last_update_time = current_time

                # 定期更新统一数据字典
                self._update_unified_data()

                # 定期保存检查点（如果正在记录）
                if self.is_recording and self.total_samples % 10000 == 0:
                    self._save_checkpoint()

            except socket.timeout:
                continue
            except Exception as e:
                print(f"数据接收错误: {str(e)}")
                if not self.shutdown_flag.is_set():
                    break

    def _process_raw_data(self, raw_data: bytes):
        """处理原始数据"""
        # 1. 拼接缓冲区
        self.raw_buffer += raw_data

        # 2. 解析数据（根据设备类型）
        if 'Neuracle' in self.device:
            data_arrays, events = self._parse_neuracle_data()
        elif 'Neuroscan' in self.device:
            data_arrays, events = self._parse_neuroscan_data()
        elif 'DSI' in self.device:
            data_arrays, events = self._parse_dsi_data()
        else:
            # 默认解析
            data_arrays, events = self._parse_default_data()

        # 3. 更新环形缓冲区
        if data_arrays:
            for modality, data in data_arrays.items():
                if modality in self.ring_buffer.buffers:
                    self.ring_buffer.append_buffer(modality, data)

            # 更新总样本数
            sample_count = next(iter(data_arrays.values())).shape[1]
            self.total_samples += sample_count

            # 4. 添加事件
            for event in events:
                self.event_manager.add_event(
                    event_type=event.get('type', 'UNKNOWN'),
                    sample_index=self.total_samples,
                    value=event.get('value'),
                    description=event.get('description', '')
                )

            # 5. 如果正在记录，保存数据
            if self.is_recording and 'record' in self.unified_data:
                for modality, data in data_arrays.items():
                    if modality in self.unified_data['record']['data']:
                        # 转换为列表存储（实际使用时可能要考虑内存优化）
                        self.unified_data['record']['data'][modality].append(data.tolist())
                        self.unified_data['record']['samples_recorded'] += sample_count

    def _parse_neuracle_data(self):
        """解析Neuracle设备数据"""
        data_arrays = {}
        events = []

        # 假设EEG是主要信号
        if 'EEG' in self.modalities:
            # 解析逻辑（简化示例）
            n_bytes_per_sample = self.n_channels * 4  # float32
            n_complete_samples = len(self.raw_buffer) // n_bytes_per_sample

            if n_complete_samples > 0:
                # 解析完整样本
                format_str = '<' + ('f' * self.n_channels) * n_complete_samples
                parsed = unpack(format_str, self.raw_buffer[:n_complete_samples * n_bytes_per_sample])

                # 转换为numpy数组
                data = np.array(parsed).reshape(n_complete_samples, self.n_channels).T
                data_arrays['EEG'] = data

                # 移除已处理的数据
                self.raw_buffer = self.raw_buffer[n_complete_samples * n_bytes_per_sample:]

        return data_arrays, events

    def _parse_neuroscan_data(self):
        """解析Neuroscan设备数据"""
        data_arrays = {}
        events = []

        # 简化解析逻辑
        header_size = 12
        samples_per_packet = 40
        bytes_per_sample = self.n_channels * 4

        packet_size = header_size + samples_per_packet * bytes_per_sample

        while len(self.raw_buffer) >= packet_size:
            # 跳过包头
            packet_data = self.raw_buffer[header_size:packet_size]

            # 解析数据
            format_str = '>' + ('i' * self.n_channels * samples_per_packet)
            parsed = unpack(format_str, packet_data)

            # 转换为numpy数组并应用缩放
            data = np.array(parsed).reshape(samples_per_packet, self.n_channels).T
            data = data * 0.14827  # Neuroscan特定缩放

            if 'EEG' not in data_arrays:
                data_arrays['EEG'] = data
            else:
                data_arrays['EEG'] = np.hstack([data_arrays['EEG'], data])

            # 移除已处理的数据包
            self.raw_buffer = self.raw_buffer[packet_size:]

        return data_arrays, events

    def _parse_dsi_data(self):
        """解析DSI设备数据"""
        data_arrays = {}
        events = []

        # DSI数据包解析逻辑
        token = b'@ABCD'
        token_len = len(token)

        i = 0
        while i + 12 < len(self.raw_buffer):
            if self.raw_buffer[i:i + token_len] == token:
                # 找到数据包
                packet_type = self.raw_buffer[i + 5]
                packet_length = self.raw_buffer[i + 6] * 256 + self.raw_buffer[i + 7]

                if i + 12 + packet_length <= len(self.raw_buffer):
                    if packet_type == 1:  # 数据包
                        # 解析数据
                        data_start = i + 23
                        data_end = i + 12 + packet_length
                        data_bytes = self.raw_buffer[data_start:data_end]

                        n_floats = len(data_bytes) // 4
                        if n_floats > 0:
                            format_str = '>' + ('f' * n_floats)
                            parsed = unpack(format_str, data_bytes)
                            data = np.array(parsed).reshape(n_floats // self.n_channels, self.n_channels).T

                            if 'EEG' not in data_arrays:
                                data_arrays['EEG'] = data
                            else:
                                data_arrays['EEG'] = np.hstack([data_arrays['EEG'], data])

                    i += 12 + packet_length
                else:
                    break  # 不完整的数据包
            else:
                i += 1

        # 保留未处理的数据
        self.raw_buffer = self.raw_buffer[i:]

        return data_arrays, events

    def _parse_default_data(self):
        """默认数据解析"""
        # 简单解析为EEG数据
        data_arrays = {}
        events = []

        n_bytes_per_sample = self.n_channels * 4
        n_complete_samples = len(self.raw_buffer) // n_bytes_per_sample

        if n_complete_samples > 0:
            format_str = '<' + ('f' * self.n_channels) * n_complete_samples
            parsed = unpack(format_str, self.raw_buffer[:n_complete_samples * n_bytes_per_sample])

            data = np.array(parsed).reshape(n_complete_samples, self.n_channels).T
            data_arrays['EEG'] = data

            self.raw_buffer = self.raw_buffer[n_complete_samples * n_bytes_per_sample:]

        return data_arrays, events

    def _update_unified_data(self):
        """更新统一数据字典"""
        # 1. 更新signal层
        signal_data = {}

        for modality in self.modalities:
            if modality in self.ring_buffer.buffers:
                # 获取最新数据（最近1秒）
                n_samples = int(self.unified_data['signal'][modality].get('sampling_rate', self.sampling_rate))
                latest_data = self.ring_buffer.get_data(modality, n_samples)

                signal_data[modality] = {
                    'data': latest_data,
                    'sampling_rate': self.unified_data['signal'][modality].get('sampling_rate', self.sampling_rate),
                    'unit': self.unified_data['signal'][modality].get('unit', 'uV'),
                    'channel_names': self.unified_data['signal'][modality].get('channel_names', []),
                    'time_offset': self.unified_data['signal'][modality].get('time_offset', 0.0),
                    'latest_timestamp': time.time()
                }

        self.unified_data['signal'] = signal_data

        # 2. 更新event层
        self.unified_data['event'] = {
            'events': self.event_manager.to_dataframe_format(),
            'latest_events': self.event_manager.get_latest_events(20),
            'total_events': len(self.event_manager.events)
        }

        # 3. 更新processed层（通过处理管道）
        if self.processing_pipeline.enabled:
            processed_data = self.processing_pipeline.process(self.unified_data)
            self.unified_data['processed'] = processed_data.get('processed', {})

        # 4. 更新系统状态
        if 'system' not in self.unified_data:
            self.unified_data['system'] = {}

        self.unified_data['system'].update({
            'is_connected': self.is_connected,
            'is_recording': self.is_recording,
            'total_samples': self.total_samples,
            'data_rate_kbps': self.data_rate,
            'buffer_usage': {mod: self.ring_buffer.update_counts.get(mod, 0)
                             for mod in self.modalities},
            'update_time': datetime.now().isoformat()
        })

    def _save_checkpoint(self):
        """保存检查点（用于长时间记录）"""
        if not self.is_recording:
            return

        checkpoint_file = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # 简化保存（实际使用时可能需要更高效的格式如HDF5）
        checkpoint_data = {
            'meta': self.unified_data.get('meta', {}),
            'record_summary': {
                'samples_recorded': self.total_samples,
                'duration': self.total_samples / self.sampling_rate,
                'modalities': list(self.unified_data.get('record', {}).get('data', {}).keys())
            },
            'timestamp': datetime.now().isoformat()
        }

        try:
            with open(checkpoint_file, 'w') as f:
                # 注意：实际数据可能太大，这里只保存元数据
                json.dump(checkpoint_data, f, indent=2)

            print(f"检查点已保存: {checkpoint_file}")
        except Exception as e:
            print(f"保存检查点失败: {str(e)}")

    def add_event_manually(self,
                           event_type: str,
                           value: Any = None,
                           description: str = ""):
        """手动添加事件"""
        event_id = self.event_manager.add_event(
            event_type=event_type,
            sample_index=self.total_samples,
            value=value,
            description=description
        )

        print(f"添加事件 [{event_type}]: {description}")
        return event_id

    def get_realtime_data(self, modality: str = 'EEG', n_seconds: float = 1.0) -> np.ndarray:
        """获取实时数据"""
        if modality not in self.unified_data.get('signal', {}):
            raise ValueError(f"未知的模态: {modality}")

        sampling_rate = self.unified_data['signal'][modality].get('sampling_rate', self.sampling_rate)
        n_samples = int(n_seconds * sampling_rate)

        return self.ring_buffer.get_data(modality, n_samples)

    def get_data_summary(self) -> Dict:
        """获取数据摘要"""
        return {
            'total_samples': self.total_samples,
            'recording_time': self.total_samples / self.sampling_rate,
            'modalities': self.modalities,
            'event_count': len(self.event_manager.events),
            'connection_status': 'connected' if self.is_connected else 'disconnected',
            'recording_status': 'recording' if self.is_recording else 'idle',
            'data_rate_kbps': self.data_rate
        }

    def save_data(self, filename: str, format: str = 'hdf5'):
        """保存数据到文件"""
        if format == 'hdf5':
            self._save_hdf5(filename)
        elif format == 'numpy':
            self._save_numpy(filename)
        elif format == 'json':
            self._save_json(filename)
        else:
            raise ValueError(f"不支持的格式: {format}")

    def _save_hdf5(self, filename: str):
        """保存为HDF5格式"""
        try:
            import h5py

            with h5py.File(filename, 'w') as f:
                # 保存meta
                meta_grp = f.create_group('meta')
                for key, value in self.unified_data.get('meta', {}).items():
                    if isinstance(value, (list, np.ndarray)):
                        meta_grp.create_dataset(key, data=value)
                    else:
                        meta_grp.attrs[key] = value

                # 保存signal
                signal_grp = f.create_group('signal')
                for modality, data_dict in self.unified_data.get('signal', {}).items():
                    mod_grp = signal_grp.create_group(modality)
                    if 'data' in data_dict and isinstance(data_dict['data'], np.ndarray):
                        mod_grp.create_dataset('data', data=data_dict['data'], compression='gzip')
                    for key, value in data_dict.items():
                        if key != 'data':
                            if isinstance(value, (list, np.ndarray)):
                                mod_grp.create_dataset(key, data=value)
                            else:
                                mod_grp.attrs[key] = value

                # 保存event
                event_grp = f.create_group('event')
                event_data = self.event_manager.to_dataframe_format()
                for key, value in event_data.items():
                    event_grp.create_dataset(key, data=value)

                # 保存record（如果存在）
                if 'record' in self.unified_data:
                    record_grp = f.create_group('record')
                    for key, value in self.unified_data['record'].items():
                        if key != 'data':
                            record_grp.attrs[key] = value

                print(f"数据已保存为HDF5: {filename}")

        except ImportError:
            print("h5py未安装，无法保存为HDF5格式")
            self._save_numpy(filename)

    def _save_numpy(self, filename: str):
        """保存为NumPy格式"""
        save_data = {
            'meta': self.unified_data.get('meta', {}),
            'signal_data': {mod: data_dict['data']
                            for mod, data_dict in self.unified_data.get('signal', {}).items()
                            if 'data' in data_dict},
            'event_data': self.event_manager.to_dataframe_format(),
            'record_data': self.unified_data.get('record', {})
        }

        np.savez_compressed(filename, **save_data)
        print(f"数据已保存为NumPy: {filename}")

    def _save_json(self, filename: str):
        """保存为JSON格式（仅元数据和摘要）"""
        summary_data = {
            'meta': self.unified_data.get('meta', {}),
            'summary': self.get_data_summary(),
            'event_summary': {
                'total_events': len(self.event_manager.events),
                'event_types': list(set(e['type'] for e in self.event_manager.events))
            },
            'signal_info': {
                mod: {k: v for k, v in data_dict.items() if k != 'data'}
                for mod, data_dict in self.unified_data.get('signal', {}).items()
            }
        }

        with open(filename, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"摘要已保存为JSON: {filename}")

    def stop(self):
        """停止服务器"""
        self.shutdown_flag.clear()
        self.disconnect()
        self.stop_recording()
        print("数据服务器已停止")


# ==================== 辅助函数 ====================
def create_default_data_dict() -> Dict[str, Any]:
    """创建默认的四层结构数据字典"""
    return {
        'meta': {
            'subject_id': 'S001',
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'task': 'resting_state',
            'modality': ['EEG'],
            'device': 'Neuracle',
            'sampling_rate': 1000,
            'n_channels': 64,
            'channel_names': [f'CH{i + 1}' for i in range(64)],
            'creation_time': datetime.now().isoformat(),
            'version': '1.0'
        },

        'signal': {
            'EEG': {
                'data': np.zeros((64, 0)),  # 初始为空
                'sampling_rate': 1000,
                'unit': 'uV',
                'channel_names': [f'CH{i + 1}' for i in range(64)],
                'reference': 'average',
                'time_offset': 0.0
            }
        },

        'event': {
            'events': {
                'onset': [],
                'duration': [],
                'trial_type': [],
                'value': [],
                'description': []
            },
            'latest_events': [],
            'total_events': 0
        },

        'processed': {},

        'system': {
            'status': 'initialized',
            'connection_status': 'disconnected'
        }
    }


def add_signal_modality(data_dict: Dict,
                        modality: str,
                        n_channels: int,
                        sampling_rate: float,
                        channel_names: List[str] = None,
                        unit: str = 'uV',
                        reference: str = 'average') -> Dict:
    """向数据字典添加新的信号模态"""
    if channel_names is None:
        channel_names = [f'{modality}_CH{i + 1}' for i in range(n_channels)]

    data_dict['signal'][modality] = {
        'data': np.zeros((n_channels, 0)),
        'sampling_rate': sampling_rate,
        'unit': unit,
        'channel_names': channel_names,
        'reference': reference,
        'time_offset': 0.0
    }

    # 更新meta中的modality列表
    if 'modality' in data_dict['meta']:
        if modality not in data_dict['meta']['modality']:
            data_dict['meta']['modality'].append(modality)

    return data_dict


# ==================== 示例使用代码 ====================
if __name__ == "__main__":
    print("=== 统一数据服务器示例 ===")

    # 1. 创建数据字典
    data_dict = create_default_data_dict()

    # 2. 添加EOG模态
    data_dict = add_signal_modality(
        data_dict,
        modality='EOG',
        n_channels=2,
        sampling_rate=1000,
        channel_names=['HEOG', 'VEOG'],
        unit='uV'
    )

    # 3. 设备配置
    device_config = {
        'device_type': 'Neuracle',
        'host': '127.0.0.1',
        'port': 8712
    }

    # 4. 创建服务器实例
    server = UnifiedDataServer(
        device_config=device_config,
        data_dict=data_dict,
        buffer_seconds=3.0
    )


    # 5. 添加实时处理（示例：带通滤波）
    def bandpass_filter(data, lowcut=1.0, highcut=40.0, fs=1000.0):
        """简单的带通滤波（示例函数）"""
        # 这里应实现实际的滤波逻辑
        # 为了示例，我们返回原始数据
        return data


    server.processing_pipeline.add_processor(
        name='bandpass_filter',
        processor_func=bandpass_filter,
        config={'lowcut': 1.0, 'highcut': 40.0, 'fs': 1000.0}
    )

    # 6. 连接设备（实际使用时需要真实设备）
    # connected = server.connect()
    # if connected:
    #     server.start()
    #
    #     # 等待一段时间
    #     time.sleep(10)
    #
    #     # 开始记录
    #     server.start_recording()
    #     time.sleep(5)
    #
    #     # 添加事件
    #     server.add_event_manually('STIMULUS', value=1, description="视觉刺激开始")
    #
    #     time.sleep(5)
    #     server.stop_recording()
    #     server.stop()
    #
    #     # 保存数据
    #     server.save_data('test_data.h5', format='hdf5')
    # else:
    #     print("无法连接到设备")

    print("示例代码结束")
    print("数据字典结构:")
    print(json.dumps({k: type(v).__name__ for k, v in data_dict.items()}, indent=2))
