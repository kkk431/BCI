#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
示例使用代码
展示如何使用优化版数据服务器
"""

import time
import numpy as np
from core.io.trigger_system.data_server import (
    UnifiedDataServer,
    create_default_data_dict,
    add_signal_modality
)


def main():
    """主示例函数"""

    print("=" * 60)
    print("优化版数据服务器 - 使用示例")
    print("=" * 60)

    # ========== 1. 创建数据字典 ==========
    print("\n1. 创建数据字典...")

    # 创建基础数据字典
    data_dict = create_default_data_dict()

    # 自定义元信息
    data_dict['meta'].update({
        'subject_id': 'TEST_001',
        'task': 'motor_imagery',
        'experimenter': '研究员_张三',
        'notes': '测试运行'
    })

    # 添加多模态信号
    print("  添加EEG信号...")
    # EEG已经在create_default_data_dict中创建

    print("  添加EOG信号...")
    data_dict = add_signal_modality(
        data_dict,
        modality='EOG',
        n_channels=2,
        sampling_rate=1000,
        channel_names=['HEOG', 'VEOG']
    )

    print("  添加EMG信号...")
    data_dict = add_signal_modality(
        data_dict,
        modality='EMG',
        n_channels=4,
        sampling_rate=2000,
        channel_names=['Biceps_L', 'Triceps_L', 'Biceps_R', 'Triceps_R'],
        unit='mV'
    )

    print(f"  数据字典创建完成，包含模态: {list(data_dict['signal'].keys())}")

    # ========== 2. 配置和创建服务器 ==========
    print("\n2. 创建数据服务器...")

    device_config = {
        'device_type': 'Neuracle',
        'host': '127.0.0.1',
        'port': 8712,
        'timeout': 10
    }

    server = UnifiedDataServer(
        device_config=device_config,
        data_dict=data_dict,
        buffer_seconds=5.0  # 5秒缓冲区
    )

    print(f"  服务器创建完成，缓冲区: {server.buffer_seconds}秒")

    # ========== 3. 添加实时处理器 ==========
    print("\n3. 配置实时处理管道...")

    # 示例处理器1: 简单归一化
    def normalize_data(data, method='zscore'):
        """数据归一化"""
        if method == 'zscore':
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            std[std == 0] = 1  # 避免除零
            return (data - mean) / std
        else:
            return data

    server.processing_pipeline.add_processor(
        name='normalization',
        processor_func=normalize_data,
        config={'method': 'zscore'}
    )

    # 示例处理器2: 伪迹检测
    def artifact_detection(data, threshold=100):
        """简单伪迹检测"""
        artifacts = np.any(np.abs(data) > threshold, axis=0)
        return {
            'artifact_mask': artifacts,
            'artifact_count': np.sum(artifacts),
            'max_amplitude': np.max(np.abs(data))
        }

    server.processing_pipeline.add_processor(
        name='artifact_detection',
        processor_func=artifact_detection,
        config={'threshold': 100}
    )

    print(f"  已添加处理器: {[p['name'] for p in server.processing_pipeline.processors]}")

    # ========== 4. 模拟数据接收（实际使用时应连接真实设备） ==========
    print("\n4. 模拟数据接收...")

    # 注意：这里模拟数据，实际使用时需要真实设备连接
    print("  警告：这是模拟模式，实际使用时请连接真实设备")
    print("  要连接真实设备，请取消注释 server.connect() 和 server.start()")

    # 模拟一些数据更新
    print("\n5. 模拟数据更新循环...")

    # 创建模拟数据
    n_eeg_channels = data_dict['signal']['EEG']['data'].shape[0]
    n_eog_channels = data_dict['signal']['EOG']['data'].shape[0]
    n_emg_channels = data_dict['signal']['EMG']['data'].shape[0]

    for i in range(10):  # 模拟10次更新
        print(f"\n  更新 #{i + 1}")

        # 生成模拟数据
        eeg_data = np.random.randn(n_eeg_channels, 100) * 10  # 100个样本点
        eog_data = np.random.randn(n_eog_channels, 100) * 5
        emg_data = np.random.randn(n_emg_channels, 50) * 20  # EMG采样率不同

        # 手动添加到缓冲区（模拟数据接收）
        server.ring_buffer.append_buffer('EEG', eeg_data)
        server.ring_buffer.append_buffer('EOG', eog_data)
        server.ring_buffer.append_buffer('EMG', emg_data)

        # 更新总样本数
        server.total_samples += 100

        # 添加一些事件
        if i % 3 == 0:
            event_id = server.add_event_manually(
                event_type='STIMULUS',
                value=i,
                description=f"刺激呈现 #{i // 3 + 1}"
            )
            print(f"    添加事件: 刺激呈现 #{i // 3 + 1} (ID: {event_id})")

        # 更新统一数据字典
        server._update_unified_data()

        # 显示状态
        summary = server.get_data_summary()
        print(f"    总样本数: {summary['total_samples']}")
        print(f"    事件数: {summary['event_count']}")

        # 检查处理结果
        if 'processed' in server.unified_data:
            proc_keys = list(server.unified_data['processed'].keys())
            print(f"    处理结果: {len(proc_keys)} 个")
            for key in proc_keys[:2]:  # 只显示前两个
                if isinstance(server.unified_data['processed'][key], dict):
                    print(f"      {key}: {list(server.unified_data['processed'][key].keys())}")

        time.sleep(0.5)

    # ========== 5. 数据查询和操作示例 ==========
    print("\n6. 数据查询示例...")

    # 获取实时数据
    print("\n  获取实时数据:")
    eeg_realtime = server.get_realtime_data('EEG', n_seconds=1.0)
    print(f"    EEG实时数据形状: {eeg_realtime.shape}")

    # 获取事件
    print("\n  获取最近事件:")
    latest_events = server.event_manager.get_latest_events(5)
    for evt in latest_events:
        print(f"    [{evt['type']}] {evt['description']} (样本: {evt['sample_index']})")

    # ========== 6. 数据保存 ==========
    print("\n7. 保存数据...")

    # 开始记录
    print("  开始记录...")
    server.start_recording()

    # 模拟一些记录数据
    for i in range(3):
        eeg_data = np.random.randn(n_eeg_channels, 50)
        server.ring_buffer.append_buffer('EEG', eeg_data)
        server.total_samples += 50

        # 更新并记录
        server._update_unified_data()
        time.sleep(0.2)

    print("  停止记录...")
    server.stop_recording()

    # 保存为不同格式
    print("\n  保存数据文件:")

    # HDF5格式（推荐）
    try:
        server.save_data('example_data.h5', format='hdf5')
        print("    ✓ 保存为HDF5格式: example_data.h5")
    except Exception as e:
        print(f"    ✗ HDF5保存失败: {e}")

    # NumPy格式
    server.save_data('example_data.npz', format='numpy')
    print("    ✓ 保存为NumPy格式: example_data.npz")

    # JSON格式（仅摘要）
    server.save_data('example_summary.json', format='json')
    print("    ✓ 保存为JSON摘要: example_summary.json")

    # ========== 7. 数据字典结构展示 ==========
    print("\n8. 最终数据字典结构:")

    def print_dict_structure(d, indent=0):
        """递归打印字典结构"""
        for key, value in d.items():
            print("  " * indent + f"├─ {key}: ", end="")
            if isinstance(value, dict):
                print(f"dict({len(value)} items)")
                print_dict_structure(value, indent + 1)
            elif isinstance(value, list):
                print(f"list({len(value)} items)")
                if value and isinstance(value[0], dict):
                    print_dict_structure(value[0], indent + 1)
            elif isinstance(value, np.ndarray):
                print(f"ndarray{value.shape}")
            else:
                print(f"{type(value).__name__}")

    print_dict_structure(server.unified_data)

    # ========== 8. 清理 ==========
    print("\n9. 清理...")
    server.stop()

    print("\n" + "=" * 60)
    print("示例运行完成!")
    print("=" * 60)

    # 显示文件大小
    import os
    for filename in ['example_data.h5', 'example_data.npz', 'example_summary.json']:
        if os.path.exists(filename):
            size = os.path.getsize(filename) / 1024
            print(f"{filename}: {size:.1f} KB")


if __name__ == "__main__":
    main()