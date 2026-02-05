#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
触发系统与数据服务器的无缝集成
提供统一的API接口
"""

from typing import Optional, Dict, Any
import time
from datetime import datetime

# 导入之前的模块
from trigger_core import OptimizedTriggerBox, TriggerBoxContext
from trigger_manager import HighPerformanceTriggerManager, TriggerManagerContext, EventPriority
from trigger_emulator import SoftwareTriggerEmulator, create_trigger_interface


# ==================== 统一集成接口 ====================
class UnifiedTriggerSystem:
    """
    统一触发系统
    提供硬件/软件无缝切换的统一接口
    """

    def __init__(self, data_server=None, config: Dict = None):
        """
        参数:
            data_server: 数据服务器实例
            config: 配置字典
        """
        self.data_server = data_server
        self.config = config or {}

        # 创建触发接口
        use_hardware = self.config.get('use_hardware', True)
        port = self.config.get('port')

        self.trigger_interface = create_trigger_interface(
            use_hardware=use_hardware,
            port=port
        )

        # 创建事件管理器
        self.manager = HighPerformanceTriggerManager(
            trigger_box=self.trigger_interface,
            data_server=data_server
        )

        # 状态跟踪
        self._experiment_active = False
        self._current_trial = 0

    def send_trigger(self,
                     value: int,
                     description: str = None,
                     **kwargs) -> str:
        """
        发送触发（统一接口）

        参数:
            value: 触发值
            description: 描述
            **kwargs: 其他参数

        返回:
            事件ID
        """
        return self.manager.send_event(value, description, **kwargs)

    def send_stimulus(self,
                      stimulus_type: str,
                      stimulus_id: int = 1,
                      **kwargs) -> str:
        """
        发送刺激事件

        参数:
            stimulus_type: 刺激类型（visual, auditory, tactile）
            stimulus_id: 刺激ID
            **kwargs: 其他参数

        返回:
            事件ID
        """
        return self.manager.send_stimulus(stimulus_type, stimulus_id, **kwargs)

    def start_experiment(self, name: str = None) -> str:
        """开始实验"""
        self._experiment_active = True
        self._current_trial = 0

        if self.data_server:
            self.data_server.start_recording()

        return self.manager.start_experiment(name)

    def end_experiment(self) -> str:
        """结束实验"""
        self._experiment_active = False

        if self.data_server:
            self.data_server.stop_recording()

        return self.manager.end_experiment()

    def start_trial(self, trial_number: int = None) -> str:
        """开始试次"""
        if trial_number is None:
            self._current_trial += 1
            trial_number = self._current_trial

        return self.manager.start_trial(trial_number)

    def end_trial(self, trial_number: int = None) -> str:
        """结束试次"""
        if trial_number is None:
            trial_number = self._current_trial

        return self.manager.end_trial(trial_number)

    def record_response(self,
                        correct: bool,
                        reaction_time: float = None,
                        trial_number: int = None) -> str:
        """记录反应"""
        if trial_number is None:
            trial_number = self._current_trial

        return self.manager.record_response(
            correct=correct,
            reaction_time=reaction_time,
            trial_number=trial_number
        )

    def get_status(self) -> Dict:
        """获取系统状态"""
        status = {
            'hardware_connected': not isinstance(self.trigger_interface, SoftwareTriggerEmulator),
            'experiment_active': self._experiment_active,
            'current_trial': self._current_trial,
            'data_server_connected': self.data_server is not None,
            'timestamp': datetime.now().isoformat()
        }

        # 添加性能统计
        perf_stats = self.manager.get_performance_stats()
        status.update({
            'performance': perf_stats,
            'recent_events': self.manager.get_recent_events(10)
        })

        return status

    def wait_for_pending_events(self, timeout: float = 2.0) -> bool:
        """等待所有待处理事件完成"""
        return self.manager.wait_for_queue_empty(timeout)

    def save_event_log(self, filepath: str):
        """保存事件日志"""
        import json

        events = self.manager.event_buffer.get_recent_events(10000)

        log_data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_events': len(events),
                'system_status': self.get_status()
            },
            'events': [
                {
                    'id': e.id,
                    'value': e.value,
                    'description': e.description,
                    'timestamp': e.timestamp,
                    'sample_index': e.sample_index,
                    'hardware_sent': e.hardware_sent,
                    'metadata': e.metadata
                }
                for e in events
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

    def shutdown(self):
        """关闭系统"""
        # 等待待处理事件
        self.wait_for_pending_events(timeout=1.0)

        # 关闭管理器
        self.manager.shutdown()


# ==================== 实验协议模板 ====================
class ExperimentProtocol:
    """实验协议模板"""

    @staticmethod
    def oddball_protocol(trigger_system: UnifiedTriggerSystem,
                         num_trials: int = 100,
                         oddball_prob: float = 0.2,
                         trial_duration: float = 2.0):
        """
        Oddball实验协议

        参数:
            trigger_system: 触发系统
            num_trials: 试次数
            oddball_prob: oddball刺激概率
            trial_duration: 试次时长
        """
        import random

        # 开始实验
        trigger_system.start_experiment("Oddball实验")

        for trial in range(1, num_trials + 1):
            # 开始试次
            trigger_system.start_trial(trial)

            # 确定刺激类型
            is_oddball = random.random() < oddball_prob

            if is_oddball:
                # Oddball刺激
                trigger_system.send_stimulus('visual', 2)  # oddball
                # 这里应该呈现oddball刺激
            else:
                # 标准刺激
                trigger_system.send_stimulus('visual', 1)  # 标准
                # 这里应该呈现标准刺激

            # 等待刺激呈现时间
            time.sleep(0.5)  # 刺激呈现500ms

            # 记录反应（这里需要实际收集反应）
            # 假设反应正确率为80%
            correct = random.random() < 0.8
            reaction_time = random.uniform(0.2, 0.8) if correct else None

            trigger_system.record_response(
                correct=correct,
                reaction_time=reaction_time,
                trial_number=trial
            )

            # 结束试次
            trigger_system.end_trial(trial)

            # 试次间隔
            time.sleep(trial_duration - 0.5)  # 减去刺激呈现时间

        # 结束实验
        trigger_system.end_experiment()

    @staticmethod
    def motor_imagery_protocol(trigger_system: UnifiedTriggerSystem,
                               num_trials_per_class: int = 30):
        """
        运动想象实验协议

        参数:
            trigger_system: 触发系统
            num_trials_per_class: 每类试次数
        """
        # 运动想象类别
        classes = ['left_hand', 'right_hand', 'feet', 'tongue']

        # 开始实验
        trigger_system.start_experiment("运动想象实验")

        trial_counter = 0

        for class_idx, class_name in enumerate(classes):
            for trial in range(num_trials_per_class):
                trial_counter += 1

                # 开始试次
                trigger_system.start_trial(trial_counter)

                # 准备阶段（注视十字）
                trigger_system.send_trigger(10, f"准备阶段_{class_name}")
                time.sleep(2.0)

                # 提示阶段（显示提示）
                trigger_system.send_trigger(20 + class_idx, f"提示_{class_name}")
                time.sleep(1.0)

                # 运动想象阶段
                trigger_system.send_trigger(30 + class_idx, f"想象_{class_name}")
                time.sleep(4.0)

                # 休息阶段
                trigger_system.send_trigger(40, "休息阶段")
                time.sleep(2.0)

                # 结束试次
                trigger_system.end_trial(trial_counter)

        # 结束实验
        trigger_system.end_experiment()


# ==================== 快速使用示例 ====================
def quick_example():
    """快速使用示例"""
    # 导入数据服务器
    from data_server import UnifiedDataServer, create_default_data_dict

    # 1. 创建数据服务器
    data_dict = create_default_data_dict()
    data_server = UnifiedDataServer(
        device_config={'device_type': 'Neuracle'},
        data_dict=data_dict,
        buffer_seconds=5.0
    )

    # 2. 创建触发系统（自动检测硬件）
    trigger_system = UnifiedTriggerSystem(
        data_server=data_server,
        config={'use_hardware': True}  # 自动检测硬件
    )

    # 3. 获取状态
    status = trigger_system.get_status()
    print(f"硬件连接: {status['hardware_connected']}")

    # 4. 运行简单测试
    trigger_system.start_experiment("测试实验")

    for i in range(5):
        trigger_system.start_trial(i + 1)
        trigger_system.send_stimulus('visual', i + 1)
        time.sleep(0.5)
        trigger_system.record_response(correct=True, trial_number=i + 1)
        trigger_system.end_trial(i + 1)
        time.sleep(1.0)

    trigger_system.end_experiment()

    # 5. 保存数据
    data_server.save_data("test_experiment.h5", format='hdf5')
    trigger_system.save_event_log("test_events.json")

    # 6. 显示统计
    stats = trigger_system.get_status()
    print(f"\n实验统计:")
    print(f"  发送事件: {stats['performance']['total_events_sent']}")
    print(f"  平均延迟: {stats['performance']['avg_latency_ms']:.2f}ms")

    # 7. 清理
    trigger_system.shutdown()
    data_server.stop()


if __name__ == "__main__":
    quick_example()