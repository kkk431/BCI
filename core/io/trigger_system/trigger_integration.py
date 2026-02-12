#! /usr/bin/env python
#  -*- coding:utf-8 -*-
"""
trigger_integration.py：触发系统与BCI数据集成层（项目正式版）
核心功能：
1. 无缝对接项目真实的data.io.py，读取EDF/MAT格式的BCI标准化数据
2. 一键切换物理触发盒(trigger_core)/软件模拟器(trigger_emulator)
3. 串联触发事件发送→解析→整合全流程
4. 支持单文件/批量文件触发事件集成，兼容EEG/EMG所有BCI数据
5. 输出带触发事件的完整标准化BCI数据字典，可直接用于后续预处理/特征提取
适配说明：
- 严格适配项目目录结构（trigger_system在IO文件夹下，导入上一级的data.io.py）
- 兼容项目data.io.py的输出格式
- 无额外依赖，直接沿用项目现有环境
"""
import os
import time
from typing import List, Dict, Optional, Union
import sys

# ===================== 核心修正：适配项目IO目录结构 =====================
# 当前文件路径：BCI/core/io/trigger_system/trigger_integration.py
# 需要导入上一级目录（BCI/core/io/）的data.io.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.io import load_data  # 正确导入项目真实的I/O模块

# 导入触发系统核心模块（同目录下的另外三个文件）
from trigger_core import TriggerCore
from trigger_emulator import TriggerEmulator
from trigger_manager import TriggerManager

# 配置日志
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def integrate_trigger_with_bci_data(
        data_path: str,
        serial_name: str = "COM3",
        use_emulator: bool = False,  # 项目默认用物理触发盒，测试时改True
        sampling_rate: float = 2000.0,
        custom_trigger_map: Optional[Dict[int, str]] = None,
        trigger_event_list: List[int] = None,
        event_interval: float = 2.0  # 事件间隔（秒），贴合实验设计
) -> Dict:
    """
    核心单文件集成函数（项目正式版）
    :param data_path: 真实BCI数据文件路径（EDF/MAT，必填）
    :param serial_name: 触发盒真实串口名（如COM3、/dev/ttyUSB0）
    :param use_emulator: True=测试/无硬件，False=真实触发盒（项目默认）
    :param sampling_rate: 数据采样率（默认2000Hz适配NinaPro EMG，EEG改256/512）
    :param custom_trigger_map: 项目的触发值-事件标签映射（如{1:'握拳',2:'张开'}）
    :param trigger_event_list: 项目的触发事件序列（如[1,7,2,7]）
    :param event_interval: 触发事件发送间隔（秒），默认2s
    :return: 带event字段的标准化BCI数据字典（可直接用于预处理）
    """
    # 步骤0：真实文件校验（项目版必须保留）
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"BCI数据文件不存在：{data_path}")

    # 默认触发序列（可替换为项目真实序列）
    default_event_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # NinaPro全动作
    trigger_event_list = trigger_event_list if trigger_event_list else default_event_list
    if not isinstance(trigger_event_list, list) or not all(isinstance(x, int) for x in trigger_event_list):
        raise TypeError("触发事件序列必须是整数列表，如[1,7,2]")

    logger.info("=" * 80)
    logger.info(f"📌 项目正式版 | 集成BCI数据与触发事件 | 文件：{os.path.basename(data_path)}")
    logger.info(
        f"🔧 配置：{'物理触发盒' if not use_emulator else '模拟器'} | 串口：{serial_name} | 采样率：{sampling_rate}Hz")
    logger.info("=" * 80)

    try:
        # 步骤1：读取项目真实的标准化BCI数据（调用data.io.py的load_data）
        logger.info("\n📂 读取真实BCI数据...")
        bci_standard_data = load_data(data_path)
        # 适配EEG/EMG双模态
        data_key = "eeg_data" if "eeg_data" in bci_standard_data else "emg_data"
        data_shape = bci_standard_data[data_key].shape
        logger.info(f"✅ 真实数据读取成功 | 数据形状：{data_shape} | 模态：{data_key[:3].upper()}")

        # 自动适配数据内置采样率（优先级最高）
        if "sampling_rate" in bci_standard_data:
            sampling_rate = bci_standard_data["sampling_rate"]
            logger.info(f"📌 自动适配数据采样率：{sampling_rate}Hz")

        # 步骤2：初始化触发系统（硬件/模拟器，接口完全对齐）
        logger.info("\n🔌 初始化触发系统...")
        trigger = TriggerEmulator(serial_name=serial_name, baudrate=115200) if use_emulator else TriggerCore(
            serial_name=serial_name, baudrate=115200)
        if use_emulator:
            trigger.open_serial()  # 模拟器无需真实串口，直接打开
        else:
            if not trigger.open_serial():
                raise ConnectionError(f"物理触发盒串口{serial_name}连接失败，请检查串口名和硬件供电")
        logger.info("✅ 触发系统初始化成功")

        # 步骤3：发送触发事件序列（贴合项目实验节奏）
        logger.info(f"\n📤 发送触发事件序列 | 序列：{trigger_event_list} | 间隔：{event_interval}s")
        for event_data in trigger_event_list:
            trigger.output_event_data(event_data=event_data, trigger_to_be_out=1)
            time.sleep(event_interval)
        logger.info(f"✅ 触发事件发送完成 | 累计：{len(trigger_event_list)}个")

        # 步骤4：解析触发历史为标准化事件
        logger.info("\n📝 解析触发历史...")
        manager = TriggerManager(sampling_rate=sampling_rate, custom_trigger_map=custom_trigger_map)
        standard_event = manager.parse_trigger_history(trigger.trigger_history)

        # 步骤5：核心整合：将标准化事件嵌入项目的标准化数据字典
        bci_standard_data["event"] = standard_event
        logger.info("\n✅ 触发事件与BCI数据整合完成！")
        logger.info(f"📋 整合后字段：{list(bci_standard_data.keys())}")

        # 步骤6：关闭串口（释放硬件资源）
        logger.info("\n🔌 关闭触发系统串口...")
        trigger.close_serial()
        logger.info("✅ 串口关闭成功")

    except Exception as e:
        raise RuntimeError(f"集成失败：{str(e)}") from e

    logger.info("=" * 80)
    logger.info(f"🎉 集成完成 | 数据可直接用于预处理/特征提取")
    logger.info("=" * 80)
    return bci_standard_data


def batch_integrate_trigger(
        data_dir: str,
        serial_name: str = "COM3",
        use_emulator: bool = False,
        sampling_rate: float = 2000.0,
        custom_trigger_map: Optional[Dict[int, str]] = None,
        trigger_event_list: List[int] = None,
        event_interval: float = 2.0
) -> List[Dict]:
    """
    批量集成函数（项目正式版）：处理文件夹下所有EDF/MAT文件
    :param data_dir: 真实BCI数据文件夹路径（必填）
    :return: 带触发事件的标准化数据字典列表
    """
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"数据文件夹不存在：{data_dir}")

    # 支持项目的真实数据格式（EDF/MAT）
    support_formats = (".edf", ".mat", ".EDF", ".MAT")
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(support_formats)]
    if not data_files:
        raise FileNotFoundError(f"文件夹{data_dir}下未找到EDF/MAT格式文件")

    logger.info(f"📁 批量处理 | 文件夹：{data_dir} | 待处理文件数：{len(data_files)}")
    batch_result = []
    for idx, file_path in enumerate(data_files, 1):
        logger.info(f"\n===== 处理第{idx}/{len(data_files)}个文件 =====")
        try:
            data = integrate_trigger_with_bci_data(
                data_path=file_path,
                serial_name=serial_name,
                use_emulator=use_emulator,
                sampling_rate=sampling_rate,
                custom_trigger_map=custom_trigger_map,
                trigger_event_list=trigger_event_list,
                event_interval=event_interval
            )
            batch_result.append(data)
        except Exception as e:
            logger.error(f"❌ 第{idx}个文件处理失败：{str(e)}")
            continue
    logger.info(f"\n📊 批量集成完成 | 成功：{len(batch_result)}个 | 失败：{len(data_files) - len(batch_result)}个")
    return batch_result


# ===================== 项目使用示例（直接替换配置即可运行） =====================
if __name__ == "__main__":
    # ************************** 项目真实配置（仅需修改这里）**************************
    # 1. 基础配置
    SERIAL_NAME = "COM3"  # 触发盒真实串口名（设备管理器查看）
    USE_EMULATOR = True  # 测试阶段改True（无硬件），项目运行改False（用硬件）
    SAMPLING_RATE = 2000.0  # 数据采样率（EMG=2000，EEG=256/512）
    EVENT_INTERVAL = 2.0  # 实验事件间隔（秒）

    # 2. 触发映射（替换为项目真实的触发值-标签对应关系）
    PROJECT_TRIGGER_MAP = {
        1: "握拳",
        2: "手掌张开",
        3: "食指捏指",
        4: "中指捏指",
        5: "无名指捏指",
        6: "小指捏指",
        7: "休息",
        8: "手腕上抬",
        9: "手腕下压"
    }

    # 3. 触发事件序列（替换为项目真实的实验序列）
    PROJECT_EVENT_LIST = [1, 7, 2, 7, 3, 7]  # 示例：握拳→休息→张开→休息→食指捏指→休息

    # 4. 真实数据路径（二选一：单文件/批量）
    # 单文件路径（替换为项目中的真实EDF/MAT文件路径）
    SINGLE_DATA_PATH = r"E:\BCI\core\io\data\S1_A1_E1.mat"  # 示例路径，需替换
    # 批量文件夹路径（替换为项目的数据集文件夹路径）
    BATCH_DATA_DIR = r"E:\BCI\core\io\data"  # 示例路径，需替换

    # ************************** 单文件集成（项目常用）**************************
    logger.info("===== 项目正式版 | 单文件集成 =====")
    bci_data = integrate_trigger_with_bci_data(
        data_path=SINGLE_DATA_PATH,
        serial_name=SERIAL_NAME,
        use_emulator=USE_EMULATOR,
        sampling_rate=SAMPLING_RATE,
        custom_trigger_map=PROJECT_TRIGGER_MAP,
        trigger_event_list=PROJECT_EVENT_LIST,
        event_interval=EVENT_INTERVAL
    )

    # 验证集成结果（可直接接入项目预处理模块）
    logger.info("\n📌 项目数据验证：")
    logger.info(f"  采样率：{bci_data['sampling_rate']}Hz")
    logger.info(f"  有效事件数：{bci_data['event']['event_count']}")
    logger.info(f"  事件标签：{bci_data['event']['event_label']}")
    logger.info(f"  事件采样点：{bci_data['event']['event_sample']}")

    # ************************** 批量集成（按需解开注释）**************************
    # logger.info("\n===== 项目正式版 | 批量集成 =====")
    # batch_data = batch_integrate_trigger(
    #     data_dir=BATCH_DATA_DIR,
    #     serial_name=SERIAL_NAME,
    #     use_emulator=USE_EMULATOR,
    #     sampling_rate=SAMPLING_RATE,
    #     custom_trigger_map=PROJECT_TRIGGER_MAP,
    #     trigger_event_list=PROJECT_EVENT_LIST,
    #     event_interval=EVENT_INTERVAL
    # )