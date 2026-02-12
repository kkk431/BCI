#! /usr/bin/env python
#  -*- coding:utf-8 -*-
"""
trigger_manager.py：触发事件管理层
核心功能：
1. 解析触发核心/模拟器的原始触发历史，过滤无效触发
2. 自定义触发值→BCI任务事件标签的映射（如1=握拳、2=张开）
3. 时间戳校准：将秒级触发时间转成脑电/EMG数据的采样点索引（关键同步步骤）
4. 输出标准化事件字典，可直接嵌入标准化数据字典的event字段
"""
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime


class TriggerManager:
    def __init__(self, sampling_rate: float = 2000.0, custom_trigger_map: Optional[Dict[int, str]] = None):
        """
        初始化触发事件管理器
        :param sampling_rate: 脑电/EMG数据的采样率（默认2000Hz适配NinaPro EMG，EEG可设256/512）
        :param custom_trigger_map: 自定义触发值→事件标签映射，如{1:'握拳',2:'张开'}，不传则用默认映射
        """
        # 修复语法错误：先校验采样率，再赋值
        if sampling_rate <= 0:
            raise ValueError("采样率必须大于0！")
        self.sampling_rate = sampling_rate
        # 触发值-事件标签映射（默认适配NinaPro手部运动，支持用户自定义覆盖）
        self.trigger_map = custom_trigger_map if custom_trigger_map else self._get_default_trigger_map()
        # 解析后的标准化事件字典（最终输出格式）
        self.standard_event = {
            "event_count": 0,  # 有效触发事件数量
            "event_timestamp": [],  # 触发事件的绝对时间戳（秒，UTC）
            "event_rel_time": [],  # 触发事件的相对时间（秒，以第一个事件为0点，贴合数据采集时序）
            "event_sample": [],  # 触发事件对应的数采采样点索引（核心，用于数据同步）
            "event_value": [],  # 原始触发值（如1/2/3）
            "event_label": []  # 解析后的事件标签（如握拳/张开/休息，贴合业务）
        }

    def _get_default_trigger_map(self) -> Dict[int, str]:
        """
        默认触发值-事件标签映射（适配NinaPro手部运动数据集，可直接修改/自定义覆盖）
        可根据你的实际任务场景调整：如EEG运动想象（1=左手/2=右手/3=脚）
        """
        return {
            0: "无动作",
            1: "握拳",
            2: "手掌张开",
            3: "捏指（食指+拇指）",
            4: "捏指（中指+拇指）",
            5: "捏指（无名指+拇指）",
            6: "捏指（小指+拇指）",
            7: "休息",
            8: "手腕上抬",
            9: "手腕下压"
        }

    def update_trigger_map(self, new_trigger_map: Dict[int, str]):
        """更新触发值-事件标签映射，适配不同任务场景"""
        if not isinstance(new_trigger_map, dict) or not all(isinstance(k, int) for k in new_trigger_map.keys()):
            raise TypeError("触发映射表必须是「int:str」的字典（如{1:'握拳'}）")
        self.trigger_map.update(new_trigger_map)
        print(f"✅ 触发映射表更新成功，当前映射：{self.trigger_map}")

    def _filter_invalid_triggers(self, trigger_history: List[Dict]) -> List[Dict]:
        """过滤无效触发历史（空值/无event_data/触发值不在映射表的记录）"""
        if not isinstance(trigger_history, list) or len(trigger_history) == 0:
            raise ValueError("触发历史必须是非空列表！")
        # 过滤规则：有event_data字段 + 触发值是整数 + 触发值在映射表中
        valid_triggers = [
            t for t in trigger_history
            if "event_data" in t and isinstance(t["event_data"], int) and t["event_data"] in self.trigger_map
        ]
        invalid_count = len(trigger_history) - len(valid_triggers)
        if invalid_count > 0:
            print(f"⚠️  过滤{invalid_count}条无效触发记录，剩余{len(valid_triggers)}条有效记录")
        return valid_triggers

    def parse_trigger_history(self, trigger_history: List[Dict], reset: bool = True):
        """
        核心方法：解析触发历史，生成标准化事件字典
        :param trigger_history: trigger_core/trigger_emulator的trigger_history触发历史列表
        :param reset: 是否重置之前的解析结果（默认True，每次解析生成新结果）
        """
        # 重置标准化事件字典（避免多次解析数据叠加）
        if reset:
            self.standard_event = self.standard_event.fromkeys(self.standard_event, [])

        # 步骤1：过滤无效触发记录
        valid_triggers = self._filter_invalid_triggers(trigger_history)
        if len(valid_triggers) == 0:
            print("⚠️  无有效触发记录，标准化事件字典为空")
            return self.standard_event

        # 步骤2：提取有效触发的核心字段
        event_timestamps = [t["timestamp"] for t in valid_triggers]  # 绝对时间戳
        event_values = [t["event_data"] for t in valid_triggers]  # 原始触发值
        # 步骤3：计算相对时间（以第一个事件为0点，贴合数据采集的时序起点）
        base_time = event_timestamps[0]
        event_rel_times = [round(ts - base_time, 3) for ts in event_timestamps]
        # 步骤4：时间戳→采样点索引（核心同步步骤：采样点=相对时间×采样率，取整）
        event_samples = [int(round(rt * self.sampling_rate)) for rt in event_rel_times]
        # 步骤5：解析触发值为业务标签
        event_labels = [self.trigger_map[v] for v in event_values]

        # 步骤6：填充标准化事件字典
        self.standard_event.update({
            "event_count": len(valid_triggers),
            "event_timestamp": event_timestamps,
            "event_rel_time": event_rel_times,
            "event_sample": event_samples,
            "event_value": event_values,
            "event_label": event_labels
        })

        # 打印解析结果摘要（方便调试）
        print("\n" + "=" * 60)
        print(f"✅ 触发历史解析完成，共解析{len(valid_triggers)}条有效事件")
        print(f"📌 数据采样率：{self.sampling_rate}Hz | 时间戳精度：3位小数")
        print(f"📌 首个事件：{event_labels[0]}（相对时间：{event_rel_times[0]}s，采样点：{event_samples[0]}）")
        print("=" * 60)
        return self.standard_event

    def get_standard_event(self) -> Dict:
        """获取解析后的标准化事件字典（可直接嵌入BCI标准化数据字典）"""
        return self.standard_event

    def save_event_to_txt(self, save_path: str, encoding: str = "utf-8"):
        """将标准化事件字典保存为TXT文件，方便后续分析/查看"""
        if self.standard_event["event_count"] == 0:
            raise RuntimeError("无解析后的事件数据，无法保存！")
        with open(save_path, "w", encoding=encoding) as f:
            f.write(f"BCI触发事件解析结果 | 采样率：{self.sampling_rate}Hz | 生成时间：{datetime.now()}\n")
            f.write("=" * 80 + "\n")
            f.write(f"事件数量\t相对时间(s)\t采样点索引\t原始触发值\t事件标签\n")
            f.write("=" * 80 + "\n")
            for i in range(self.standard_event["event_count"]):
                f.write(f"{i + 1}\t\t{self.standard_event['event_rel_time'][i]}\t\t")
                f.write(f"{self.standard_event['event_sample'][i]}\t\t{self.standard_event['event_value'][i]}\t\t")
                f.write(f"{self.standard_event['event_label'][i]}\n")
        print(f"✅ 标准化事件数据已保存至：{save_path}")


# ===================== 测试代码（联动前两个文件，完整验证解析流程） =====================
if __name__ == "__main__":
    # 步骤1：导入模拟器（硬件层同理，只需替换为TriggerCore）
    from trigger_emulator import TriggerEmulator
    import time

    # 步骤2：初始化模拟器，生成触发历史
    SERIAL_NAME = "COM3"
    emulator = TriggerEmulator(serial_name=SERIAL_NAME)
    # 模拟发送贴合NinaPro的手部运动触发事件
    emulator.output_event_data(event_data=1, trigger_to_be_out=1)  # 握拳
    time.sleep(2)
    emulator.output_event_data(event_data=2, trigger_to_be_out=1)  # 手掌张开
    time.sleep(2)
    emulator.output_event_data(event_data=7, trigger_to_be_out=1)  # 休息
    time.sleep(2)
    emulator.output_event_data(event_data=8, trigger_to_be_out=1)  # 手腕上抬

    # 步骤3：初始化触发管理器（默认2000Hz适配NinaPro，可自定义映射）
    # 自定义映射示例：适配EEG运动想象
    # custom_map = {1:'左手运动', 2:'右手运动', 3:'脚部运动', 7:'休息'}
    # manager = TriggerManager(sampling_rate=256.0, custom_trigger_map=custom_map)
    manager = TriggerManager(sampling_rate=2000.0)

    # 步骤4：解析模拟器的触发历史
    manager.parse_trigger_history(emulator.trigger_history)

    # 步骤5：获取标准化事件字典（可直接嵌入你的BCI标准化数据字典）
    standard_event = manager.get_standard_event()
    print("\n📋 解析后的标准化事件字典：")
    for k, v in standard_event.items():
        print(f"  {k}: {v}")

    # 步骤6：（可选）保存事件数据到TXT
    manager.save_event_to_txt("bci_trigger_events.txt")

    # 步骤7：关闭模拟器
    emulator.close_serial()