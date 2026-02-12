#! /usr/bin/env python
#  -*- coding:utf-8 -*-
"""
trigger_core.py：触发盒硬件控制核心层
参考BRAINFUSION TriggerBox源码（Neuracle触发盒驱动），保留核心硬件交互逻辑，简化冗余代码
核心功能：串口连接、设备校验、触发信号发送/读取、传感器参数配置
"""
import serial
import serial.tools.list_ports
from ctypes import *
from typing import Optional, Dict, List


# ===================== 1. 数据帧结构定义（完全参考BRAINFUSION源码） =====================
class PackageTriggerBoxBaseFrame(Structure):
    """参考原代码：触发盒基础帧结构（设备ID+功能ID+有效载荷长度）"""
    _fields_ = [('deviceID', c_ubyte), ('functionID', c_ubyte), ('payload', c_ushort)]
    _pack_ = 1  # 1字节对齐，匹配硬件协议


class PackageSensorInfo(Structure):
    """参考原代码：传感器信息结构（类型+编号）"""
    _fields_ = [('sensorType', c_ubyte), ('sensorNum', c_ubyte)]
    _pack_ = 1


class PackageSensorPara(Structure):
    """参考原代码：传感器参数结构（触发边沿、输出通道、触发值、阈值、事件数据）"""
    _fields_ = [('Edge', c_ubyte), ('OutputChannel', c_ubyte), ('TriggerToBeOut', c_ushort),
                ('Threshold', c_ushort), ('EventData', c_ushort)]
    _pack_ = 1


class PackageGetDeviceInfo(Structure):
    """参考原代码：获取设备信息指令结构"""
    _fields_ = [('frame', PackageTriggerBoxBaseFrame), ('command', c_ubyte)]
    _pack_ = 1


class PackageGetSensorPara(Structure):
    """参考原代码：读取传感器参数指令结构"""
    _fields_ = [('frame', PackageTriggerBoxBaseFrame), ('sensorInfo', PackageSensorInfo)]
    _pack_ = 1


class PackageSetSensorPara(Structure):
    """参考原代码：设置传感器参数指令结构"""
    _fields_ = [('frame', PackageTriggerBoxBaseFrame), ('sensorInfo', PackageSensorInfo),
                ('sensorPara', PackageSensorPara)]
    _pack_ = 1


# ===================== 2. 核心触发盒控制类（参考+优化原代码TriggerBox类） =====================
class TriggerCore:
    # 参考原代码：功能ID常量（触发盒指令类型）
    FUNCTION_ID_SENSOR_PARA_GET = 1  # 读取传感器参数
    FUNCTION_ID_SENSOR_PARA_SET = 2  # 设置传感器参数
    FUNCTION_ID_DEVICE_INFO_GET = 3  # 读取设备信息
    FUNCTION_ID_DEVICE_NAME_GET = 4  # 读取设备名称
    FUNCTION_ID_SENSOR_SAMPLE_GET = 5  # 读取传感器采样值
    FUNCTION_ID_SENSOR_INFO_GET = 6  # 读取传感器信息
    FUNCTION_ID_OUTPUT_EVENT_DATA = 225  # 输出事件数据（核心触发功能）
    FUNCTION_ID_ERROR = 131  # 错误响应

    # 参考原代码：传感器类型常量+映射（数字→可读名称）
    SENSOR_TYPE_DIGITAL_IN = 1
    SENSOR_TYPE_LIGHT = 2
    SENSOR_TYPE_LINE_IN = 3
    SENSOR_TYPE_MIC = 4
    SENSOR_TYPE_KEY = 5
    SENSOR_TYPE_TEMPERATURE = 6
    SENSOR_TYPE_HUMIDITY = 7
    SENSOR_TYPE_AMBIENT_LIGHT = 8
    SENSOR_TYPE_DEBUG = 9
    SENSOR_TYPE_ALL = 255

    SENSOR_TYPE_MAP = {
        SENSOR_TYPE_DIGITAL_IN: 'DigitalIN',
        SENSOR_TYPE_LIGHT: 'Light',
        SENSOR_TYPE_LINE_IN: 'LineIN',
        SENSOR_TYPE_MIC: 'Mic',
        SENSOR_TYPE_KEY: 'Key',
        SENSOR_TYPE_TEMPERATURE: 'Temperature',
        SENSOR_TYPE_HUMIDITY: 'Humidity',
        SENSOR_TYPE_AMBIENT_LIGHT: 'Ambientlight',
        SENSOR_TYPE_DEBUG: 'Debug'
    }

    def __init__(self, serial_name: str, baudrate: int = 115200, timeout: float = 60):
        """
        初始化触发盒（参考原代码__init__，优化为可配置参数）
        :param serial_name: 串口名（如COM3、/dev/ttyUSB0）
        :param baudrate: 波特率（原代码固定115200，此处可配置）
        :param timeout: 串口超时时间（原代码固定60s，此处可配置）
        """
        self.serial_name = serial_name
        self.baudrate = baudrate
        self.timeout = timeout
        self._device_id = 1  # 触发盒设备ID（原代码固定为1）
        self._serial_handle: Optional[serial.Serial] = None  # 串口句柄
        self._device_name: Optional[str] = None  # 设备名称
        self._device_info: Optional[Dict] = None  # 设备信息（版本、传感器数量等）
        self._sensor_info: List[Dict] = []  # 传感器信息列表

        # 参考原代码：初始化时自动校验设备、读取基础信息
        self._init_device()

    def _init_device(self):
        """参考原代码validate_device，封装初始化流程"""
        if not self.check_online():
            raise RuntimeError(f"触发盒串口{self.serial_name}未在线！")

        # 打开串口（参考原代码validate_device）
        try:
            self._serial_handle = serial.Serial(
                port=self.serial_name,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
        except Exception as e:
            raise RuntimeError(f"打开串口{self.serial_name}失败：{str(e)}")

        if not self._serial_handle.is_open:
            raise RuntimeError(f"串口{self.serial_name}打开后未处于就绪状态！")

        print(f"✅ 串口{self.serial_name}打开成功")

        # 读取设备基础信息（参考原代码__init__）
        self._device_name = self.get_device_name()
        self._device_info = self.get_device_info()
        self._sensor_info = self.get_sensor_info()
        print(f"📌 设备名称：{self._device_name}")
        print(f"📌 设备信息：{self._device_info}")

    def refresh_serial_list(self) -> List:
        """参考原代码：刷新串口列表"""
        return list(serial.tools.list_ports.comports())

    def check_online(self) -> bool:
        """参考原代码：检查目标串口是否在线"""
        port_list = self.refresh_serial_list()
        if not port_list:
            print("❌ 未检测到任何在线串口！")
            return False

        for port in port_list:
            if port.device == self.serial_name:
                print(f"✅ 目标串口[{self.serial_name}]在线（描述：{port.description}）")
                return True

        print(f"❌ 目标串口[{self.serial_name}]未在线！")
        print("📋 在线串口列表：")
        for port in port_list:
            print(f"  - {port.device} : {port.description}")
        return False

    def get_device_name(self) -> str:
        """参考原代码：读取触发盒设备名称"""
        cmd = PackageTriggerBoxBaseFrame()
        cmd.deviceID = self._device_id
        cmd.functionID = self.FUNCTION_ID_DEVICE_NAME_GET
        cmd.payload = 0

        self.send(cmd)
        data = self.read(cmd.functionID)
        device_name = str(data).strip()
        return device_name

    def get_device_info(self) -> Dict:
        """参考原代码：读取触发盒硬件信息（版本、传感器数量、设备ID）"""
        cmd = PackageGetDeviceInfo()
        cmd.command = 1
        cmd.frame.deviceID = self._device_id
        cmd.frame.functionID = self.FUNCTION_ID_DEVICE_INFO_GET
        cmd.frame.payload = 1

        self.send(cmd)
        data = self.read(cmd.frame.functionID)

        # 参考原代码解析逻辑，修复原代码索引错误（原代码data[0]实际应为data[4]）
        hardware_version = data[4] if len(data) >= 8 else 0
        firmware_version = data[5] if len(data) >= 8 else 0
        sensor_sum = data[6] if len(data) >= 8 else 0
        device_id = (data[8] << 24) | (data[9] << 16) | (data[10] << 8) | data[11] if len(data) >= 12 else 0

        device_info = {
            "HardwareVersion": hardware_version,
            "FirmwareVersion": firmware_version,
            "SensorSum": sensor_sum,
            "DeviceID": device_id
        }
        return device_info

    def _get_sensor_type_string(self, sensor_type: int) -> str:
        """参考原代码：传感器类型数字转可读名称"""
        return self.SENSOR_TYPE_MAP.get(sensor_type, "Undefined")

    def _get_sensor_type_num(self, type_string: str) -> int:
        """参考原代码_sensor_type：传感器名称转数字"""
        type_map_reverse = {v: k for k, v in self.SENSOR_TYPE_MAP.items()}
        if type_string not in type_map_reverse:
            raise ValueError(f"未定义的传感器类型：{type_string}")
        return type_map_reverse[type_string]

    def get_sensor_info(self) -> List[Dict]:
        """参考原代码：读取所有传感器信息"""
        cmd = PackageTriggerBoxBaseFrame()
        cmd.deviceID = self._device_id
        cmd.functionID = self.FUNCTION_ID_SENSOR_INFO_GET
        cmd.payload = 0

        self.send(cmd)
        data = self.read(cmd.functionID)

        if len(data) % 2 != 0:
            raise RuntimeError(f"传感器信息响应长度错误（{len(data)}），应为偶数！")

        sensor_info = []
        for i in range(int(len(data) / 2)):
            sensor_type = data[i * 2]
            sensor_num = data[i * 2 + 1]
            sensor_type_str = self._get_sensor_type_string(sensor_type)
            sensor_info.append({
                "Type": sensor_type_str,
                "Number": sensor_num,
                "TypeNum": sensor_type
            })
            print(f"🔍 传感器{i}：类型={sensor_type_str}，编号={sensor_num}")

        return sensor_info

    def get_sensor_para(self, sensor_id: int) -> PackageSensorPara:
        """参考原代码：读取指定传感器参数"""
        if sensor_id >= len(self._sensor_info):
            raise IndexError(f"传感器ID{sensor_id}超出范围（共{len(self._sensor_info)}个传感器）")

        sensor = self._sensor_info[sensor_id]
        cmd = PackageGetSensorPara()
        cmd.sensorInfo.sensorType = sensor["TypeNum"]
        cmd.sensorInfo.sensorNum = sensor["Number"]
        cmd.frame.deviceID = self._device_id
        cmd.frame.functionID = self.FUNCTION_ID_SENSOR_PARA_GET
        cmd.frame.payload = 2

        self.send(cmd)
        data = self.read(cmd.frame.functionID)

        # 解析传感器参数（参考原代码）
        sensor_para = PackageSensorPara()
        sensor_para.Edge = data[0]
        sensor_para.OutputChannel = data[1]
        sensor_para.TriggerToBeOut = data[2] | (data[3] << 8)
        sensor_para.Threshold = data[4] | (data[5] << 8)
        sensor_para.EventData = data[6] | (data[7] << 8)

        print(f"📌 传感器{sensor_id}参数：")
        print(f"  - 触发边沿：{sensor_para.Edge}")
        print(f"  - 输出通道：{sensor_para.OutputChannel}")
        print(f"  - 触发值：{sensor_para.TriggerToBeOut}")
        print(f"  - 阈值：{sensor_para.Threshold}")
        print(f"  - 事件数据：{sensor_para.EventData}")
        return sensor_para

    def set_sensor_para(self, sensor_id: int, sensor_para: PackageSensorPara):
        """参考原代码：设置指定传感器参数（修复原代码FUNCTION_ID错误）"""
        if sensor_id >= len(self._sensor_info):
            raise IndexError(f"传感器ID{sensor_id}超出范围（共{len(self._sensor_info)}个传感器）")

        sensor = self._sensor_info[sensor_id]
        cmd = PackageSetSensorPara()
        cmd.frame.deviceID = self._device_id
        # 修复原代码BUG：原代码用了FUNCTION_ID_OUTPUT_EVENT_DATA，应改为FUNCTION_ID_SENSOR_PARA_SET
        cmd.frame.functionID = self.FUNCTION_ID_SENSOR_PARA_SET
        cmd.frame.payload = 10
        cmd.sensorInfo.sensorType = sensor["TypeNum"]
        cmd.sensorInfo.sensorNum = sensor["Number"]
        cmd.sensorPara = sensor_para

        self.send(cmd)
        data = self.read(cmd.frame.functionID)

        if data[0] == sensor["TypeNum"] and data[1] == sensor["Number"]:
            print(f"✅ 传感器{sensor_id}参数设置成功")
        else:
            raise RuntimeError(f"传感器{sensor_id}参数设置失败！响应：{data}")

    def output_event_data(self, event_data: int, trigger_to_be_out: int = 1):
        """参考原代码：输出触发事件数据（核心触发功能）"""
        cmd = PackageGetDeviceInfo()
        cmd.command = event_data
        cmd.frame.deviceID = self._device_id
        cmd.frame.functionID = self.FUNCTION_ID_OUTPUT_EVENT_DATA
        cmd.frame.payload = 1

        self.send(cmd)
        data = self.read(cmd.frame.functionID)

        if data[0] != self.FUNCTION_ID_OUTPUT_EVENT_DATA:
            raise RuntimeError(f"发送触发事件失败！响应功能ID：{data[0]}")
        print(f"✅ 触发事件发送成功（事件数据：{event_data}，触发值：{trigger_to_be_out}）")

    def send(self, data):
        """参考原代码：底层串口发送指令"""
        if not self._serial_handle or not self._serial_handle.is_open:
            raise RuntimeError("串口未打开，无法发送数据！")

        self._serial_handle.flushInput()  # 清空接收缓冲区
        self._serial_handle.write(bytes(data))  # 转换为字节发送

    def read(self, function_id: int) -> bytes:
        """参考原代码：底层串口读取响应，增加异常处理"""
        if not self._serial_handle or not self._serial_handle.is_open:
            raise RuntimeError("串口未打开，无法读取数据！")

        self._serial_handle.flushOutput()  # 清空发送缓冲区
        # 读取响应头（4字节：deviceID+functionID+payload）
        header = self._serial_handle.read(4)
        if len(header) != 4:
            raise RuntimeError(f"响应头读取失败（长度：{len(header)}）")

        # 校验设备ID（参考原代码）
        if header[0] != self._device_id:
            raise RuntimeError(f"设备ID不匹配！请求：{self._device_id}，响应：{header[0]}")

        # 校验功能ID（参考原代码）
        if header[1] != function_id:
            if header[1] == self.FUNCTION_ID_ERROR:
                error_type = self._serial_handle.read(1)[0]
                error_msg = {
                    0: '无错误',
                    1: '帧头错误',
                    2: '载荷长度错误',
                    3: '通道不存在',
                    4: '设备ID错误',
                    5: '功能ID错误',
                    6: '传感器类型错误'
                }.get(error_type, f"未知错误（{error_type}）")
                raise RuntimeError(f"触发盒响应错误：{error_msg}")
            else:
                raise RuntimeError(f"功能ID不匹配！请求：{function_id}，响应：{header[1]}")

        # 读取有效载荷（参考原代码）
        payload_len = header[2] | (header[3] << 8)
        payload = self._serial_handle.read(payload_len)
        if len(payload) != payload_len:
            raise RuntimeError(f"载荷读取失败！预期：{payload_len}字节，实际：{len(payload)}字节")

        return payload

    def close_serial(self):
        """参考原代码：关闭串口"""
        if self._serial_handle and self._serial_handle.is_open:
            self._serial_handle.close()
            print(f"✅ 串口{self.serial_name}已关闭")


# ===================== 测试代码（可直接运行验证） =====================
if __name__ == "__main__":
    # 替换为你的触发盒串口名（如COM3、/dev/ttyUSB0）
    SERIAL_NAME = "COM3"

    try:
        # 初始化触发盒
        trigger_core = TriggerCore(serial_name=SERIAL_NAME)

        # 示例1：读取第一个传感器参数
        sensor_para = trigger_core.get_sensor_para(sensor_id=0)

        # 示例2：发送触发事件（事件数据=100，触发值=1）
        trigger_core.output_event_data(event_data=100, trigger_to_be_out=1)

        # 关闭串口
        trigger_core.close_serial()
    except Exception as e:
        print(f"❌ 测试失败：{str(e)}")