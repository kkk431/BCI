#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
高性能软件触发模拟器
用于无硬件时的开发和测试
"""

import time
import threading
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np


# ==================== 软件模拟器 ====================
class SoftwareTriggerEmulator:
    """
    软件触发模拟器
    模拟硬件触发盒行为，用于开发和测试
    """

    def __init__(self, latency_mean: float = 0.001, latency_std: float = 0.0005):
        """
        参数:
            latency_mean: 平均模拟延迟（秒）
            latency_std: 延迟标准差
        """
        self.latency_mean = latency_mean
        self.latency_std = latency_std
        self._is_connected = True
        self._stats = {
            'events_sent': 0,
            'total_latency': 0.0,
            'last_event_time': 0.0
        }
        self._lock = threading.RLock()
        self._callbacks = []

    def send_event(self, event_data: int, **kwargs) -> bool:
        """发送事件（模拟）"""
        with self._lock:
            # 模拟延迟
            if self.latency_std > 0:
                latency = np.random.normal(self.latency_mean, self.latency_std)
                latency = max(0.0001, min(latency, 0.01))  # 限制范围
                time.sleep(latency)
            else:
                latency = self.latency_mean
                if latency > 0:
                    time.sleep(latency)

            # 更新统计
            self._stats['events_sent'] += 1
            self._stats['total_latency'] += latency
            self._stats['last_event_time'] = time.time()

            # 触发回调
            for callback in self._callbacks:
                try:
                    callback(event_data, latency)
                except:
                    pass

            return True

    def send_event_batch(self, events: list, **kwargs) -> list:
        """批量发送事件"""
        return [self.send_event(e) for e in events]

    async def send_event_async(self, event_data: int) -> bool:
        """异步发送事件"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.send_event, event_data)

    def connect(self) -> bool:
        """模拟连接"""
        self._is_connected = True
        return True

    def disconnect(self):
        """模拟断开"""
        self._is_connected = False

    def get_device_info(self) -> Dict:
        """获取模拟设备信息"""
        return {
            'hardware_version': 1,
            'firmware_version': 1,
            'sensor_count': 8,
            'device_id': 123456,
            'is_emulator': True
        }

    def get_sensor_info(self) -> list:
        """获取模拟传感器信息"""
        return [
            {'type': 1, 'type_name': '数字输入', 'number': 1, 'id': 0},
            {'type': 2, 'type_name': '光传感器', 'number': 1, 'id': 1},
            {'type': 5, 'type_name': '按键', 'number': 1, 'id': 2},
            {'type': 9, 'type_name': '调试', 'number': 1, 'id': 3}
        ]

    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        with self._lock:
            stats = self._stats.copy()
            if stats['events_sent'] > 0:
                stats['avg_latency'] = stats['total_latency'] / stats['events_sent']
            else:
                stats['avg_latency'] = 0
            return stats

    def add_callback(self, callback):
        """添加事件回调"""
        self._callbacks.append(callback)

    def reset_statistics(self):
        """重置统计"""
        with self._lock:
            self._stats = {
                'events_sent': 0,
                'total_latency': 0.0,
                'last_event_time': 0.0
            }


# ==================== 智能模拟器选择 ====================
def create_trigger_interface(use_hardware: bool = True, port: str = None):
    """
    智能创建触发接口

    参数:
        use_hardware: 是否使用真实硬件
        port: 串口路径

    返回:
        硬件或软件触发接口
    """
    if use_hardware:
        from trigger_core import create_trigger_box
        hardware = create_trigger_box(port)
        if hardware:
            return hardware

    # 回退到软件模拟器
    print("使用软件触发模拟器（硬件不可用或未指定）")
    return SoftwareTriggerEmulator()


# ==================== 性能测试工具 ====================
class TriggerBenchmark:
    """触发性能测试工具"""

    @staticmethod
    def benchmark_latency(trigger_interface, num_events: int = 100) -> Dict:
        """测试延迟性能"""
        latencies = []

        for i in range(num_events):
            start_time = time.perf_counter()
            trigger_interface.send_event(i % 255 + 1)
            end_time = time.perf_counter()

            latencies.append((end_time - start_time) * 1000)  # 转毫秒

        return {
            'num_events': num_events,
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99)
        }

    @staticmethod
    def benchmark_throughput(trigger_interface, duration: float = 1.0) -> Dict:
        """测试吞吐量"""
        events_sent = 0
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration:
            trigger_interface.send_event(1)
            events_sent += 1

        end_time = time.perf_counter()
        actual_duration = end_time - start_time

        return {
            'duration_sec': actual_duration,
            'events_sent': events_sent,
            'events_per_second': events_sent / actual_duration,
            'avg_interval_ms': (actual_duration / events_sent) * 1000
        }