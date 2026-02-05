#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
高级事件管理器
支持事件映射、时序控制、性能监控
"""

import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import IntEnum
import threading
import queue
import json
from datetime import datetime
import numpy as np
from collections import deque


# ==================== 事件定义 ====================
class EventPriority(IntEnum):
    """事件优先级"""
    CRITICAL = 0  # 关键事件（实验开始/结束）
    HIGH = 1  # 重要事件（刺激呈现）
    NORMAL = 2  # 普通事件（反应记录）
    LOW = 3  # 低优先级事件（调试信息）


@dataclass
class TriggerEvent:
    """触发事件数据结构"""
    value: int  # 事件值
    timestamp: float  # 发送时间戳
    description: str  # 事件描述
    priority: EventPriority  # 事件优先级
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加元数据
    sample_index: Optional[int] = None  # 对应的样本索引（事后填充）
    hardware_sent: bool = False  # 是否已发送到硬件
    id: str = None  # 事件唯一ID

    def __post_init__(self):
        if self.id is None:
            self.id = f"event_{int(self.timestamp * 1000)}_{self.value}"


# ==================== 事件缓冲队列 ====================
class EventBuffer:
    """高性能事件缓冲队列"""

    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.event_counter = 0
        self._index_map = {}  # ID -> 位置索引（快速查找）

    def add_event(self, event: TriggerEvent) -> str:
        """添加事件到缓冲区"""
        with self.lock:
            self.buffer.append(event)
            self._index_map[event.id] = len(self.buffer) - 1
            self.event_counter += 1
            return event.id

    def get_event(self, event_id: str) -> Optional[TriggerEvent]:
        """根据ID获取事件"""
        with self.lock:
            idx = self._index_map.get(event_id)
            if idx is not None and idx < len(self.buffer):
                return self.buffer[idx]
            return None

    def get_events_in_range(self, start_time: float, end_time: float) -> List[TriggerEvent]:
        """获取时间范围内的事件"""
        with self.lock:
            return [
                event for event in self.buffer
                if start_time <= event.timestamp <= end_time
            ]

    def get_recent_events(self, count: int = 100) -> List[TriggerEvent]:
        """获取最近的事件"""
        with self.lock:
            return list(self.buffer)[-count:] if self.buffer else []

    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()
            self._index_map.clear()
            self.event_counter = 0


# ==================== 事件映射配置 ====================
class EventMapping:
    """智能事件映射管理器"""

    def __init__(self, config_file: str = None):
        self.mappings: Dict[int, Dict] = {}
        self.reverse_mappings: Dict[str, int] = {}
        self.category_groups: Dict[str, List[int]] = {}

        if config_file:
            self.load_config(config_file)
        else:
            self._load_default_mappings()

    def _load_default_mappings(self):
        """加载默认事件映射"""
        # 实验控制
        self.add_mapping(255, "实验开始", "experiment_control")
        self.add_mapping(254, "实验结束", "experiment_control")
        self.add_mapping(253, "试次开始", "experiment_control")
        self.add_mapping(252, "试次结束", "experiment_control")

        # 视觉刺激
        for i in range(1, 51):
            self.add_mapping(100 + i, f"视觉刺激_{i}", "visual_stimulus")

        # 听觉刺激
        for i in range(1, 51):
            self.add_mapping(200 + i, f"听觉刺激_{i}", "auditory_stimulus")

        # 被试反应
        self.add_mapping(301, "反应正确", "response")
        self.add_mapping(302, "反应错误", "response")
        self.add_mapping(303, "反应超时", "response")

        # 系统事件
        self.add_mapping(401, "数据记录开始", "system")
        self.add_mapping(402, "数据记录结束", "system")

    def add_mapping(self, value: int, description: str, category: str = "custom"):
        """添加事件映射"""
        self.mappings[value] = {
            'description': description,
            'category': category,
            'value': value
        }
        self.reverse_mappings[description] = value

        if category not in self.category_groups:
            self.category_groups[category] = []
        self.category_groups[category].append(value)

    def get_description(self, value: int) -> str:
        """获取事件描述"""
        mapping = self.mappings.get(value)
        if mapping:
            return mapping['description']
        return f"未定义事件_{value}"

    def get_value(self, description: str) -> Optional[int]:
        """根据描述获取事件值"""
        return self.reverse_mappings.get(description)

    def get_events_by_category(self, category: str) -> List[int]:
        """获取指定类别的事件值列表"""
        return self.category_groups.get(category, []).copy()

    def save_config(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'mappings': self.mappings,
                'category_groups': self.category_groups
            }, f, indent=2, ensure_ascii=False)

    def load_config(self, filepath: str):
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.mappings = data['mappings']
            self.category_groups = data['category_groups']

            # 重建反向映射
            self.reverse_mappings.clear()
            for value, info in self.mappings.items():
                self.reverse_mappings[info['description']] = int(value)


# ==================== 高性能事件管理器 ====================
class HighPerformanceTriggerManager:
    """
    高性能事件管理器
    特性：
    1. 多线程安全
    2. 事件缓冲
    3. 性能监控
    4. 错误恢复
    5. 实时统计
    """

    def __init__(self, trigger_box=None, data_server=None):
        self.trigger_box = trigger_box
        self.data_server = data_server

        # 事件管理组件
        self.event_mapping = EventMapping()
        self.event_buffer = EventBuffer(max_size=50000)

        # 异步发送队列
        self.send_queue = queue.PriorityQueue(maxsize=1000)
        self._send_thread = None
        self._stop_sending = threading.Event()

        # 统计信息
        self.stats = {
            'total_events_sent': 0,
            'total_hardware_events': 0,
            'total_software_events': 0,
            'hardware_errors': 0,
            'avg_latency_ms': 0.0,
            'peak_latency_ms': 0.0,
            'events_by_category': {},
            'recent_latencies': deque(maxlen=100)
        }

        # 时序控制
        self._last_event_time = 0
        self._min_event_interval = 0.001  # 1ms最小间隔

        # 启动发送线程
        self._start_send_thread()

    def _start_send_thread(self):
        """启动异步发送线程"""
        if self._send_thread is None:
            self._stop_sending.clear()
            self._send_thread = threading.Thread(
                target=self._send_worker,
                daemon=True,
                name="TriggerSendThread"
            )
            self._send_thread.start()

    def _send_worker(self):
        """发送工作线程"""
        while not self._stop_sending.is_set():
            try:
                # 从优先队列获取事件（阻塞但可超时）
                priority, (timestamp, event) = self.send_queue.get(timeout=0.1)

                # 检查时间间隔
                current_time = time.time()
                time_since_last = current_time - self._last_event_time

                if time_since_last < self._min_event_interval:
                    # 等待最小间隔
                    time.sleep(self._min_event_interval - time_since_last)

                # 发送事件
                self._send_event_internal(event)
                self._last_event_time = time.time()

                self.send_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"发送线程错误: {e}")
                time.sleep(0.01)

    def _send_event_internal(self, event: TriggerEvent):
        """内部发送事件（线程安全）"""
        start_time = time.time()

        try:
            # 发送到硬件（如果可用）
            hardware_success = False
            if self.trigger_box:
                hardware_success = self.trigger_box.send_event(event.value)
                event.hardware_sent = hardware_success

                if hardware_success:
                    self.stats['total_hardware_events'] += 1
                else:
                    self.stats['hardware_errors'] += 1

            # 记录到数据服务器（如果可用）
            if self.data_server:
                event.sample_index = self.data_server.total_samples

                self.data_server.add_event_manually(
                    event_type="TRIGGER",
                    value=event.value,
                    description=event.description
                )

            # 添加到缓冲区
            self.event_buffer.add_event(event)

            # 更新统计
            self.stats['total_events_sent'] += 1
            self.stats['total_software_events'] += 1

            # 计算延迟
            latency = (time.time() - start_time) * 1000  # 转毫秒
            self.stats['recent_latencies'].append(latency)
            self.stats['avg_latency_ms'] = np.mean(self.stats['recent_latencies'])
            self.stats['peak_latency_ms'] = max(
                self.stats['peak_latency_ms'],
                latency
            )

            # 按类别统计
            category = self.event_mapping.mappings.get(event.value, {}).get('category', 'unknown')
            if category not in self.stats['events_by_category']:
                self.stats['events_by_category'][category] = 0
            self.stats['events_by_category'][category] += 1

        except Exception as e:
            print(f"发送事件失败: {e}")

    def send_event(self,
                   value: int,
                   description: str = None,
                   priority: EventPriority = EventPriority.NORMAL,
                   metadata: Dict = None,
                   async_send: bool = True) -> str:
        """
        发送事件

        参数:
            value: 事件值
            description: 事件描述（如为None则从映射获取）
            priority: 事件优先级
            metadata: 附加元数据
            async_send: 是否异步发送

        返回:
            事件ID
        """
        # 获取描述
        if description is None:
            description = self.event_mapping.get_description(value)

        # 创建事件对象
        event = TriggerEvent(
            value=value,
            timestamp=time.time(),
            description=description,
            priority=priority,
            metadata=metadata or {}
        )

        if async_send:
            # 添加到异步队列
            self.send_queue.put((priority.value, (event.timestamp, event)))
            return event.id
        else:
            # 同步发送
            self._send_event_internal(event)
            return event.id

    def send_event_by_description(self,
                                  description: str,
                                  priority: EventPriority = EventPriority.NORMAL,
                                  **kwargs) -> Optional[str]:
        """通过描述发送事件"""
        value = self.event_mapping.get_value(description)
        if value is not None:
            return self.send_event(value, description, priority, **kwargs)
        return None

    def send_stimulus(self,
                      stimulus_type: str,
                      stimulus_id: int,
                      category: str = "visual_stimulus",
                      **kwargs) -> str:
        """发送刺激事件"""
        base_values = {
            'visual': 100,
            'auditory': 200,
            'tactile': 300,
            'response': 400
        }

        base_value = base_values.get(stimulus_type, 0)
        event_value = base_value + stimulus_id

        description = f"{stimulus_type}_刺激_{stimulus_id}"

        # 确保有映射
        if event_value not in self.event_mapping.mappings:
            self.event_mapping.add_mapping(event_value, description, category)

        return self.send_event(event_value, description, **kwargs)

    def start_experiment(self, experiment_name: str = None) -> str:
        """开始实验"""
        metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat()
        }
        return self.send_event(
            255,
            "实验开始",
            EventPriority.CRITICAL,
            metadata
        )

    def end_experiment(self) -> str:
        """结束实验"""
        return self.send_event(
            254,
            "实验结束",
            EventPriority.CRITICAL
        )

    def start_trial(self, trial_number: int) -> str:
        """开始试次"""
        metadata = {'trial_number': trial_number}
        return self.send_event(
            253,
            f"试次开始_{trial_number}",
            EventPriority.HIGH,
            metadata
        )

    def end_trial(self, trial_number: int) -> str:
        """结束试次"""
        metadata = {'trial_number': trial_number}
        return self.send_event(
            252,
            f"试次结束_{trial_number}",
            EventPriority.HIGH,
            metadata
        )

    def record_response(self,
                        correct: bool,
                        reaction_time: float = None,
                        trial_number: int = None) -> str:
        """记录反应"""
        event_value = 301 if correct else 302
        description = "反应正确" if correct else "反应错误"

        metadata = {
            'correct': correct,
            'reaction_time': reaction_time,
            'trial_number': trial_number
        }

        return self.send_event(
            event_value,
            description,
            EventPriority.NORMAL,
            metadata
        )

    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        stats = self.stats.copy()

        # 添加实时信息
        stats['queue_size'] = self.send_queue.qsize()
        stats['buffer_size'] = len(self.event_buffer.buffer)
        stats['is_hardware_connected'] = self.trigger_box is not None

        if self.trigger_box:
            box_stats = self.trigger_box.get_performance_stats()
            stats['hardware_stats'] = box_stats

        return stats

    def get_recent_events(self, count: int = 20) -> List[Dict]:
        """获取最近事件"""
        events = self.event_buffer.get_recent_events(count)
        return [
            {
                'id': e.id,
                'value': e.value,
                'description': e.description,
                'timestamp': e.timestamp,
                'hardware_sent': e.hardware_sent,
                'sample_index': e.sample_index
            }
            for e in events
        ]

    def wait_for_queue_empty(self, timeout: float = 5.0) -> bool:
        """等待发送队列清空"""
        start_time = time.time()
        while not self.send_queue.empty():
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.01)
        return True

    def shutdown(self):
        """关闭管理器"""
        # 停止发送线程
        self._stop_sending.set()
        if self._send_thread:
            self._send_thread.join(timeout=2.0)

        # 等待队列清空
        self.wait_for_queue_empty(timeout=2.0)

        # 断开硬件连接
        if self.trigger_box:
            self.trigger_box.disconnect()


# ==================== 上下文管理器 ====================
class TriggerManagerContext:
    """触发管理器上下文"""

    def __init__(self, port: str = None, data_server=None):
        self.port = port
        self.data_server = data_server
        self.manager: Optional[HighPerformanceTriggerManager] = None

    def __enter__(self) -> HighPerformanceTriggerManager:
        from trigger_core import create_trigger_box

        # 创建触发盒
        trigger_box = create_trigger_box(self.port)

        # 创建管理器
        self.manager = HighPerformanceTriggerManager(
            trigger_box=trigger_box,
            data_server=self.data_server
        )

        return self.manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.manager:
            self.manager.shutdown()