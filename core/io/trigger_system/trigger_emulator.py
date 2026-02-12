#! /usr/bin/env python
#  -*- coding:utf-8 -*-
"""
trigger_emulator.py：触发盒软件模拟器
完全对齐trigger_core.py的类/方法/参数，无硬件依赖，模拟触发盒所有核心功能
核心功能：模拟串口连接、设备信息返回、触发事件发送、传感器参数读写，记录触发历史
"""
import time
from typing import Optional, Dict, List, Any
from ctypes import c_ubyte, c_ushort

# ===================== 1. 模拟常量（和trigger_core.py完全一致） =====================
# 功能ID常量（和TriggerCore对齐）
FUNCTION_ID_SENSOR_PARA_GET = 1
FUNCTION_ID_SENSOR_PARA_SET = 2
FUNCTION_ID_DEVICE_INFO_GET = 3
FUNCTION_ID_DEVICE_NAME_GET = 4
FUNCTION_ID_SENSOR_INFO_GET = 6
FUNCTION_ID_OUTPUT_EVENT_DATA = 225
FUNCTION_ID_ERROR = 131

# 传感器类型常量+映射（和TriggerCore对齐，参考BRAINFUSION）
SENSOR_TYPE_DIGITAL_IN = 1
SENSOR_TYPE_LIGHT = 2
SENSOR_TYPE_LINE_IN = 3
SENSOR_TYPE_MIC = 4
SENSOR_TYPE_KEY = 5
SENSOR_TYPE_TEMPERATURE = 6
SENSOR_TYPE_HUMIDITY = 7
SENSOR_TYPE_AMBIENT_LIGHT = 8
SENSOR_TYPE_DEBUG = 9

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
SENSOR_TYPE_REVERSE_MAP = {v: k for k, v in SENSOR_TYPE_MAP.items()}


# ===================== 2. 模拟传感器参数类（和trigger_core的PackageSensorPara对齐） =====================
# 无需依赖ctypes，用普通类模拟，保证属性一致即可
class MockSensorPara:
    def __init__(self):
        self.Edge = 1  # 触发边沿：1=上升沿
        self.OutputChannel = 3  # 输出通道：3（参考BRAINFUSION源码默认值）
        self.TriggerToBeOut = 1  # 触发值：1（默认）
        self.Threshold = 0  # 阈值：0（默认）
        self.EventData = 0  # 事件数据：0（默认）


# ===================== 3. 核心模拟器类（和TriggerCore接口1:1对齐） =====================
class TriggerEmulator:
    def __init__(self, serial_name: str, baudrate: int = 115200, timeout: float = 60):
        """
        初始化触发盒模拟器（参数和TriggerCore完全一致，仅做占位）
        :param serial_name: 模拟串口名（如COM3）
        :param baudrate: 模拟波特率（和硬件一致）
        :param timeout: 模拟超时时间（和硬件一致）
        """
        self.serial_name = serial_name
        self.baudrate = baudrate
        self.timeout = timeout
        self._device_id = 1  # 模拟设备ID（和BRAINFUSION一致）
        self._is_open = False  # 模拟串口打开状态
        self._device_name = "Neuracle_TriggerBox_Mock"  # 模拟设备名称
        self._device_info: Dict = {}  # 模拟设备信息
        self._sensor_info: List[Dict] = []  # 模拟传感器信息（参考BRAINFUSION源码的传感器列表）
        self._sensor_para_list: List[MockSensorPara] = []  # 模拟传感器参数列表
        self.trigger_history: List[Dict] = []  # 核心：记录所有模拟触发事件的历史，方便后续解析
        self._init_device()  # 模拟设备初始化

    def _init_device(self):
        """模拟设备初始化（对应trigger_core的_init_device）"""
        # 模拟打开串口
        self._is_open = True
        print(f"✅ [模拟器] 模拟打开串口{self.serial_name}成功（波特率：{self.baudrate}）")
        # 生成模拟设备信息（参考BRAINFUSION源码的设备信息格式）
        self._device_info = {
            "HardwareVersion": 1,
            "FirmwareVersion": 2,
            "SensorSum": 9,  # 模拟9个传感器（参考BRAINFUSION源码的传感器数量）
            "DeviceID": 12345678
        }
        # 生成模拟传感器信息（参考BRAINFUSION源码的传感器列表，还原原代码的"双Light/双LineIN"特性）
        mock_sensor_list = [
            ("Light", 1), ("Light", 2), ("LineIN", 1), ("LineIN", 2),
            ("Ambientlight", 1), ("Mic", 1), ("Humidity", 1),
            ("Temperature", 1), ("Debug", 1)
        ]
        for idx, (sensor_type, sensor_num) in enumerate(mock_sensor_list):
            self._sensor_info.append({
                "Type": sensor_type,
                "Number": sensor_num,
                "TypeNum": SENSOR_TYPE_REVERSE_MAP[sensor_type]
            })
            # 为每个传感器生成默认参数
            self._sensor_para_list.append(MockSensorPara())
        # 打印模拟设备信息
        print(f"📌 [模拟器] 设备名称：{self._device_name}")
        print(f"📌 [模拟器] 设备信息：{self._device_info}")
        print(f"🔍 [模拟器] 生成{len(self._sensor_info)}个模拟传感器（和BRAINFUSION源码一致）")

    def refresh_serial_list(self) -> List[Dict]:
        """模拟刷新串口列表（对应trigger_core的refresh_serial_list）"""
        return [{"device": self.serial_name, "description": "Neuracle TriggerBox (Mock)"}]

    def check_online(self) -> bool:
        """模拟检查串口在线（对应trigger_core的check_online）"""
        print(f"✅ [模拟器] 目标串口[{self.serial_name}]模拟在线")
        return True

    def get_device_name(self) -> str:
        """模拟获取设备名称（对应trigger_core的get_device_name）"""
        return self._device_name

    def get_device_info(self) -> Dict:
        """模拟获取设备信息（对应trigger_core的get_device_info）"""
        return self._device_info

    def _get_sensor_type_string(self, sensor_type: int) -> str:
        """模拟传感器类型数字转名称（和trigger_core一致）"""
        return SENSOR_TYPE_MAP.get(sensor_type, "Undefined")

    def _get_sensor_type_num(self, type_string: str) -> int:
        """模拟传感器类型名称转数字（和trigger_core一致）"""
        if type_string not in SENSOR_TYPE_REVERSE_MAP:
            raise ValueError(f"[模拟器] 未定义的传感器类型：{type_string}")
        return SENSOR_TYPE_REVERSE_MAP[type_string]

    def get_sensor_info(self) -> List[Dict]:
        """模拟获取传感器信息（对应trigger_core的get_sensor_info）"""
        for idx, sensor in enumerate(self._sensor_info):
            print(f"🔍 [模拟器] 传感器{idx}：类型={sensor['Type']}，编号={sensor['Number']}")
        return self._sensor_info

    def get_sensor_para(self, sensor_id: int) -> MockSensorPara:
        """模拟获取传感器参数（对应trigger_core的get_sensor_para）"""
        if sensor_id >= len(self._sensor_para_list):
            raise IndexError(f"[模拟器] 传感器ID{sensor_id}超出范围（共{len(self._sensor_para_list)}个）")
        para = self._sensor_para_list[sensor_id]
        print(f"📌 [模拟器] 传感器{sensor_id}模拟参数：")
        print(f"  - 触发边沿：{para.Edge} | 输出通道：{para.OutputChannel} | 触发值：{para.TriggerToBeOut}")
        print(f"  - 阈值：{para.Threshold} | 事件数据：{para.EventData}")
        return para

    def set_sensor_para(self, sensor_id: int, sensor_para: MockSensorPara):
        """模拟设置传感器参数（对应trigger_core的set_sensor_para）"""
        if sensor_id >= len(self._sensor_para_list):
            raise IndexError(f"[模拟器] 传感器ID{sensor_id}超出范围（共{len(self._sensor_para_list)}个）")
        self._sensor_para_list[sensor_id] = sensor_para
        print(f"✅ [模拟器] 传感器{sensor_id}参数模拟设置成功")

    def output_event_data(self, event_data: int, trigger_to_be_out: int = 1):
        """
        模拟发送触发事件（核心功能，对应trigger_core的output_event_data）
        记录触发事件到trigger_history，包含「时间/事件数据/触发值」，方便后续解析
        """
        if not self._is_open:
            raise RuntimeError("[模拟器] 串口未打开，无法发送模拟触发事件！")
        # 记录触发历史（核心：后续trigger_manager从这里解析事件）
        self.trigger_history.append({
            "timestamp": time.time(),  # 触发时间戳（秒）
            "event_data": event_data,  # 事件数据（核心，如1=握拳、2=张开）
            "trigger_to_be_out": trigger_to_be_out,  # 触发值
            "status": "mock_success"  # 模拟触发状态
        })
        print(f"✅ [模拟器] 触发事件模拟发送成功！事件数据：{event_data}，触发值：{trigger_to_be_out}")
        print(f"📝 [模拟器] 触发历史累计：{len(self.trigger_history)}条")

    def send(self, data: Any):
        """模拟串口发送（对应trigger_core的send，仅做占位，无实际操作）"""
        if not self._is_open:
            raise RuntimeError("[模拟器] 串口未打开，无法模拟发送！")
        # 模拟发送，无需实际操作
        pass

    def read(self, function_id: int) -> bytes:
        """
        模拟串口读取（对应trigger_core的read，返回模拟字节数据）
        适配trigger_core的响应格式，保证上层调用无感知
        """
        if not self._is_open:
            raise RuntimeError("[模拟器] 串口未打开，无法模拟读取！")
        # 模拟不同功能ID的响应字节（仅保证格式正确，满足上层解析）
        if function_id == FUNCTION_ID_DEVICE_NAME_GET:
            return b"Neuracle_TriggerBox_Mock"
        elif function_id == FUNCTION_ID_DEVICE_INFO_GET:
            return b"\x00\x00\x00\x00\x01\x02\x09\x00\x00\x89\xabc\xdef"
        elif function_id == FUNCTION_ID_SENSOR_INFO_GET:
            return b"\x02\x01\x02\x02\x03\x01\x03\x02\x08\x01\x04\x01\x07\x01\x06\x01\x09\x01"
        elif function_id == FUNCTION_ID_OUTPUT_EVENT_DATA:
            return bytes([FUNCTION_ID_OUTPUT_EVENT_DATA])
        else:
            return b"\x00" * 8  # 默认返回8字节空数据

    def close_serial(self):
        """模拟关闭串口（对应trigger_core的close_serial）"""
        self._is_open = False
        self.trigger_history = []  # 清空触发历史
        print(f"✅ [模拟器] 串口{self.serial_name}模拟关闭成功")


# ===================== 测试代码（和trigger_core.py测试逻辑一致，可直接运行） =====================
if __name__ == "__main__":
    # 模拟串口名（和硬件一致）
    SERIAL_NAME = "COM3"

    try:
        # 初始化模拟器（和初始化硬件TriggerCore的代码完全一样）
        trigger_emulator = TriggerEmulator(serial_name=SERIAL_NAME)

        # 示例1：读取第一个传感器参数
        print("\n--- 模拟读取传感器0参数 ---")
        sensor_para = trigger_emulator.get_sensor_para(sensor_id=0)

        # 示例2：模拟发送多个触发事件（贴合BCI/EMG任务，如1=握拳、2=张开、3=捏指）
        print("\n--- 模拟发送触发事件 ---")
        trigger_emulator.output_event_data(event_data=1, trigger_to_be_out=1)  # 握拳
        time.sleep(2)  # 模拟任务间隔
        trigger_emulator.output_event_data(event_data=2, trigger_to_be_out=1)  # 张开
        time.sleep(2)
        trigger_emulator.output_event_data(event_data=3, trigger_to_be_out=1)  # 捏指

        # 示例3：查看触发历史（后续trigger_manager会解析这个历史）
        print("\n--- 查看模拟触发历史 ---")
        for idx, trigger in enumerate(trigger_emulator.trigger_history):
            print(f"触发{idx}：时间={trigger['timestamp']:.2f}s，事件数据={trigger['event_data']}")

        # 关闭模拟器
        print("\n--- 模拟关闭串口 ---")
        trigger_emulator.close_serial()

    except Exception as e:
        print(f"❌ [模拟器] 测试失败：{str(e)}")