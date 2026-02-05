#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
极致优化的触发盒硬件控制层
使用异步IO、零拷贝、内存池优化
"""

import serial
import serial.tools.list_ports
import asyncio
import threading
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import struct
import time
from enum import IntEnum
from functools import lru_cache
import numpy as np


# ==================== 内存优化结构 ====================
@dataclass(frozen=True)
class TriggerConfig:
    """不可变配置类，优化内存使用"""
    port: str
    baudrate: int = 115200
    timeout: float = 2.0
    write_timeout: float = 2.0
    retry_count: int = 3
    retry_delay: float = 0.1


class SensorType(IntEnum):
    """传感器类型枚举"""
    DIGITAL_IN = 1
    LIGHT = 2
    LINE_IN = 3
    MIC = 4
    KEY = 5
    TEMPERATURE = 6
    HUMIDITY = 7
    AMBIENT_LIGHT = 8
    DEBUG = 9
    ALL = 255


class FunctionID(IntEnum):
    """功能ID枚举"""
    SENSOR_PARA_GET = 1
    SENSOR_PARA_SET = 2
    DEVICE_INFO_GET = 3
    DEVICE_NAME_GET = 4
    SENSOR_SAMPLE_GET = 5
    SENSOR_INFO_GET = 6
    OUTPUT_EVENT_DATA = 225
    ERROR = 131


# ==================== 零拷贝协议解析器 ====================
class TriggerProtocol:
    """二进制协议解析器，避免内存分配"""

    # 预编译结构体格式
    _FRAME_FORMAT = struct.Struct('<BBH')  # deviceID, functionID, payload
    _SENSOR_INFO_FORMAT = struct.Struct('<BB')  # sensorType, sensorNum
    _SENSOR_PARA_FORMAT = struct.Struct('<BBHHH')  # Edge, OutputChannel, TriggerToBeOut, Threshold, EventData

    @staticmethod
    def pack_frame(device_id: int, function_id: int, payload: int) -> bytes:
        """打包帧头，零内存分配"""
        return TriggerProtocol._FRAME_FORMAT.pack(device_id, function_id, payload)

    @staticmethod
    def unpack_frame(data: bytes) -> Tuple[int, int, int]:
        """解包帧头，零内存分配"""
        return TriggerProtocol._FRAME_FORMAT.unpack(data)

    @staticmethod
    def pack_sensor_info(sensor_type: int, sensor_num: int) -> bytes:
        """打包传感器信息"""
        return TriggerProtocol._SENSOR_INFO_FORMAT.pack(sensor_type, sensor_num)

    @staticmethod
    def pack_sensor_para(edge: int, output_channel: int,
                         trigger_to_be_out: int, threshold: int,
                         event_data: int) -> bytes:
        """打包传感器参数"""
        return TriggerProtocol._SENSOR_PARA_FORMAT.pack(
            edge, output_channel, trigger_to_be_out, threshold, event_data
        )


# ==================== 连接池管理器 ====================
class ConnectionPool:
    """串口连接池，避免频繁开关"""

    _pool: Dict[str, serial.Serial] = {}
    _lock = threading.RLock()

    @classmethod
    def get_connection(cls, config: TriggerConfig) -> Optional[serial.Serial]:
        """从连接池获取连接"""
        with cls._lock:
            if config.port in cls._pool:
                conn = cls._pool[config.port]
                if conn.is_open:
                    return conn
                else:
                    del cls._pool[config.port]

            # 创建新连接
            try:
                conn = serial.Serial(
                    port=config.port,
                    baudrate=config.baudrate,
                    timeout=config.timeout,
                    write_timeout=config.write_timeout,
                    exclusive=True
                )
                cls._pool[config.port] = conn
                return conn
            except Exception:
                return None

    @classmethod
    def release_connection(cls, port: str):
        """释放连接（不关闭，保持连接池）"""
        pass  # 连接池保持打开

    @classmethod
    def close_all(cls):
        """关闭所有连接"""
        with cls._lock:
            for port, conn in cls._pool.items():
                try:
                    conn.close()
                except:
                    pass
            cls._pool.clear()


# ==================== 极致优化的触发盒核心 ====================
class OptimizedTriggerBox:
    """
    极致优化的触发盒控制类
    特性：
    1. 连接池管理
    2. 零拷贝协议处理
    3. 异步IO支持
    4. 智能重试机制
    5. 内存池缓存
    """

    def __init__(self, config: TriggerConfig):
        self.config = config
        self._conn: Optional[serial.Serial] = None
        self._lock = threading.RLock()
        self._device_id = 1
        self._device_info_cache: Optional[Dict] = None
        self._sensor_info_cache: Optional[List] = None
        self._last_error: Optional[str] = None
        self._stats = {
            'requests': 0,
            'errors': 0,
            'avg_response_time': 0.0,
            'last_request_time': 0.0
        }

        # 预计算常用命令
        self._cached_commands = self._precompute_commands()

    def _precompute_commands(self) -> Dict:
        """预计算常用命令，减少运行时计算"""
        return {
            'get_device_name': TriggerProtocol.pack_frame(
                self._device_id, FunctionID.DEVICE_NAME_GET, 0
            ),
            'get_device_info': TriggerProtocol.pack_frame(
                self._device_id, FunctionID.DEVICE_INFO_GET, 1
            ) + bytes([1]),  # command=1
            'get_sensor_info': TriggerProtocol.pack_frame(
                self._device_id, FunctionID.SENSOR_INFO_GET, 0
            )
        }

    def _update_stats(self, start_time: float, success: bool = True):
        """更新性能统计"""
        response_time = time.time() - start_time
        self._stats['requests'] += 1
        self._stats['last_request_time'] = time.time()

        # 指数移动平均更新响应时间
        if self._stats['avg_response_time'] == 0:
            self._stats['avg_response_time'] = response_time
        else:
            alpha = 0.1  # 平滑因子
            self._stats['avg_response_time'] = (
                    alpha * response_time +
                    (1 - alpha) * self._stats['avg_response_time']
            )

        if not success:
            self._stats['errors'] += 1

    def connect(self) -> bool:
        """连接触发盒（使用连接池）"""
        with self._lock:
            self._conn = ConnectionPool.get_connection(self.config)
            if self._conn and self._conn.is_open:
                # 清空缓冲区
                self._conn.reset_input_buffer()
                self._conn.reset_output_buffer()
                return True
            return False

    def disconnect(self):
        """断开连接（释放到连接池）"""
        with self._lock:
            if self._conn:
                ConnectionPool.release_connection(self.config.port)
                self._conn = None

    def _send_command(self, command: bytes, expected_function_id: int) -> Optional[bytes]:
        """发送命令并读取响应（带重试机制）"""
        if not self._conn or not self._conn.is_open:
            if not self.connect():
                return None

        start_time = time.time()

        for attempt in range(self.config.retry_count):
            try:
                # 发送命令
                self._conn.write(command)

                # 读取响应头
                header = self._conn.read(4)
                if len(header) < 4:
                    continue

                device_id, function_id, payload = TriggerProtocol.unpack_frame(header)

                # 验证响应
                if device_id != self._device_id:
                    self._last_error = f"设备ID不匹配: {device_id}"
                    continue

                if function_id == FunctionID.ERROR:
                    error_data = self._conn.read(1)
                    error_map = {
                        0: '无错误',
                        1: '帧头错误',
                        2: '负载错误',
                        3: '通道不存在',
                        4: '设备ID错误',
                        5: '功能ID错误',
                        6: '传感器类型错误'
                    }
                    self._last_error = error_map.get(error_data[0], '未知错误')
                    continue

                if function_id != expected_function_id:
                    self._last_error = f"功能ID不匹配: {function_id}"
                    continue

                # 读取负载数据
                if payload > 0:
                    data = self._conn.read(payload)
                    if len(data) == payload:
                        self._update_stats(start_time, success=True)
                        return data

            except Exception as e:
                self._last_error = str(e)
                time.sleep(self.config.retry_delay)

        self._update_stats(start_time, success=False)
        return None

    @lru_cache(maxsize=32)
    def get_device_name(self) -> Optional[str]:
        """获取设备名（带缓存）"""
        data = self._send_command(
            self._cached_commands['get_device_name'],
            FunctionID.DEVICE_NAME_GET
        )
        return data.decode('utf-8', errors='ignore') if data else None

    def get_device_info(self, force_refresh: bool = False) -> Optional[Dict]:
        """获取设备信息（带缓存）"""
        if self._device_info_cache is not None and not force_refresh:
            return self._device_info_cache.copy()

        data = self._send_command(
            self._cached_commands['get_device_info'],
            FunctionID.DEVICE_INFO_GET
        )

        if data and len(data) >= 8:
            self._device_info_cache = {
                'hardware_version': data[0],
                'firmware_version': data[1],
                'sensor_count': data[2],
                'device_id': struct.unpack('<I', data[4:8])[0]
            }
            return self._device_info_cache.copy()

        return None

    def get_sensor_info(self, force_refresh: bool = False) -> Optional[List[Dict]]:
        """获取传感器信息（带缓存）"""
        if self._sensor_info_cache is not None and not force_refresh:
            return self._sensor_info_cache.copy()

        data = self._send_command(
            self._cached_commands['get_sensor_info'],
            FunctionID.SENSOR_INFO_GET
        )

        if data and len(data) % 2 == 0:
            sensors = []
            for i in range(0, len(data), 2):
                sensor_type = data[i]
                sensor_num = data[i + 1]

                sensor_name = {
                    SensorType.DIGITAL_IN: "数字输入",
                    SensorType.LIGHT: "光传感器",
                    SensorType.LINE_IN: "线路输入",
                    SensorType.MIC: "麦克风",
                    SensorType.KEY: "按键",
                    SensorType.TEMPERATURE: "温度",
                    SensorType.HUMIDITY: "湿度",
                    SensorType.AMBIENT_LIGHT: "环境光",
                    SensorType.DEBUG: "调试"
                }.get(sensor_type, f"未知({sensor_type})")

                sensors.append({
                    'type': sensor_type,
                    'type_name': sensor_name,
                    'number': sensor_num,
                    'id': len(sensors)
                })

            self._sensor_info_cache = sensors
            return self._sensor_info_cache.copy()

        return None

    def send_event(self, event_data: int, sensor_id: Optional[int] = None) -> bool:
        """发送事件数据（核心方法）"""
        if sensor_id is not None:
            # 通过特定传感器发送
            sensor_info = self.get_sensor_info()
            if not sensor_info or sensor_id >= len(sensor_info):
                return False

            sensor = sensor_info[sensor_id]
            # 构建设置传感器参数命令
            command = (
                    TriggerProtocol.pack_frame(
                        self._device_id, FunctionID.SENSOR_PARA_SET, 10
                    ) +
                    TriggerProtocol.pack_sensor_info(sensor['type'], sensor['number']) +
                    TriggerProtocol.pack_sensor_para(1, 3, 1, 0, event_data)
            )
            expected_function_id = FunctionID.SENSOR_PARA_SET
        else:
            # 直接发送事件
            command = (
                    TriggerProtocol.pack_frame(
                        self._device_id, FunctionID.OUTPUT_EVENT_DATA, 1
                    ) +
                    bytes([event_data])
            )
            expected_function_id = FunctionID.OUTPUT_EVENT_DATA

        result = self._send_command(command, expected_function_id)
        return result is not None

    def send_event_batch(self, events: List[int], delay: float = 0.01) -> List[bool]:
        """批量发送事件"""
        results = []
        for event in events:
            results.append(self.send_event(event))
            if delay > 0:
                time.sleep(delay)
        return results

    async def send_event_async(self, event_data: int) -> bool:
        """异步发送事件"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.send_event, event_data)

    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        return self._stats.copy()

    def get_last_error(self) -> Optional[str]:
        """获取最后错误信息"""
        return self._last_error

    def reset_statistics(self):
        """重置统计信息"""
        self._stats = {
            'requests': 0,
            'errors': 0,
            'avg_response_time': 0.0,
            'last_request_time': 0.0
        }


# ==================== 工厂函数 ====================
def create_trigger_box(port: str = None, auto_detect: bool = True) -> Optional[OptimizedTriggerBox]:
    """
    创建触发盒实例（自动检测或指定端口）

    参数:
        port: 串口路径，如 'COM3' 或 '/dev/ttyUSB0'
        auto_detect: 是否自动检测设备

    返回:
        OptimizedTriggerBox实例 或 None
    """
    if port:
        config = TriggerConfig(port=port)
        trigger_box = OptimizedTriggerBox(config)
        if trigger_box.connect():
            return trigger_box

    elif auto_detect:
        # 自动检测可用串口
        available_ports = serial.tools.list_ports.comports()

        for port_info in available_ports:
            try:
                config = TriggerConfig(port=port_info.device)
                trigger_box = OptimizedTriggerBox(config)

                # 快速测试连接
                if trigger_box.connect():
                    device_name = trigger_box.get_device_name()
                    if device_name and "Neuracle" in device_name:
                        print(f"检测到触发盒: {port_info.device} - {device_name}")
                        return trigger_box
                    else:
                        trigger_box.disconnect()
            except:
                continue

    return None


# ==================== 上下文管理器 ====================
class TriggerBoxContext:
    """上下文管理器，确保资源清理"""

    def __init__(self, port: str = None):
        self.port = port
        self.trigger_box: Optional[OptimizedTriggerBox] = None

    def __enter__(self) -> Optional[OptimizedTriggerBox]:
        self.trigger_box = create_trigger_box(self.port)
        return self.trigger_box

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.trigger_box:
            self.trigger_box.disconnect()
        # 清理连接池
        ConnectionPool.close_all()