# -*- coding: utf-8 -*-
"""
fNIRS预处理测试脚本
用于测试fnirs_preprocessing.py模块是否能够处理BIDS格式的fNIRS数据
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import h5py  # 用于读取SNIRF文件
import mne  # 可选的替代SNIRF读取方式

# 添加当前目录到路径，确保可以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入预处理模块
try:
    from preprocessing import PreprocessingConfig, FilterType, DetrendMethod
    from fnirs_preprocessing import fNIRSPreprocessor, fNIRSConfig, OpticalModel, MotionCorrectionMethod
    print("成功导入预处理模块")
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保fnirs_preprocessing.py和preprocessing.py在当前目录")
    sys.exit(1)


# ====================== 数据加载函数 ======================

def load_bids_fNIRS_data(subject_dir, task_name="task-1"):
    """
    加载BIDS格式的fNIRS数据
    
    Args:
        subject_dir: 被试目录路径
        task_name: 任务名称，如"task-1"
        
    Returns:
        四层结构的数据字典
    """
    subject_dir = Path(subject_dir)
    nirs_dir = subject_dir / "nirs"
    
    if not nirs_dir.exists():
        raise FileNotFoundError(f"nirs目录不存在: {nirs_dir}")
    
    # 1. 加载元数据
    meta = {
        "subject_id": subject_dir.name,
        "task": task_name,
        "modality": ["fnirs"],
        "data_format": "BIDS",
        "source_directory": str(subject_dir)
    }
    
    # 2. 加载通道信息
    channels_file = nirs_dir / f"{subject_dir.name}_{task_name}_channels.tsv"
    if channels_file.exists():
        channels_df = pd.read_csv(channels_file, sep='\t')
        meta["n_channels"] = len(channels_df)
        meta["channel_names"] = channels_df["name"].tolist() if "name" in channels_df.columns else [f"CH{i}" for i in range(len(channels_df))]
        
        # 提取波长信息
        if "wavelength_nominal" in channels_df.columns:
            # 从通道信息中提取唯一的波长
            wavelengths = channels_df["wavelength_nominal"].unique()
            meta["wavelengths"] = sorted(wavelengths.tolist())
        else:
            # 默认波长
            meta["wavelengths"] = [730.0, 850.0]
    else:
        print(f"警告: 通道文件不存在: {channels_file}")
        meta["n_channels"] = 20  # 假设值
        meta["channel_names"] = [f"CH{i}" for i in range(20)]
        meta["wavelengths"] = [730.0, 850.0]
    
    # 3. 加载事件信息
    events_file = nirs_dir / f"{subject_dir.name}_{task_name}_events.tsv"
    events_info = {}
    if events_file.exists():
        events_df = pd.read_csv(events_file, sep='\t')
        events_info = {
            "event_id": events_df.get("trial_type", np.arange(len(events_df))).tolist(),
            "event_label": events_df.get("trial_type", ["event"] * len(events_df)).tolist(),
            "event_time": events_df.get("onset", np.arange(0, len(events_df)*10, 10)).tolist(),
            "event_sample": events_df.get("sample", np.arange(0, len(events_df)*100, 100)).tolist()
        }
    else:
        print(f"警告: 事件文件不存在: {events_file}")
        # 创建模拟事件
        events_info = {
            "event_id": [1, 2, 1],
            "event_label": ["stimulus", "rest", "stimulus"],
            "event_time": [10.0, 30.0, 50.0],
            "event_sample": [100, 300, 500]
        }
    
    # 4. 加载SNIRF数据文件
    snirf_file = nirs_dir / f"{subject_dir.name}_{task_name}_nirs.snirf"
    
    if snirf_file.exists():
        print(f"加载SNIRF文件: {snirf_file}")
        # 方法1: 使用h5py直接读取SNIRF文件
        try:
            data, sampling_rate, additional_info = load_snirf_h5py(snirf_file)
            meta["sampling_rate"] = sampling_rate
            meta["device"] = additional_info.get("device", "Unknown")
            
            # 如果有距离信息，提取它
            if "distances" in additional_info:
                meta["distances"] = additional_info["distances"]
            
        except Exception as e:
            print(f"h5py读取SNIRF失败: {e}")
            print("将使用模拟数据")
            data, sampling_rate = create_simulated_fNIRS_data()
            meta["sampling_rate"] = sampling_rate
            meta["device"] = "Simulated"
    else:
        print(f"警告: SNIRF文件不存在: {snirf_file}")
        print("将使用模拟数据")
        data, sampling_rate = create_simulated_fNIRS_data()
        meta["sampling_rate"] = sampling_rate
        meta["device"] = "Simulated"
    
    # 5. 加载额外的JSON元数据
    json_file = nirs_dir / f"{subject_dir.name}_{task_name}_nirs.json"
    if json_file.exists():
        with open(json_file, 'r') as f:
            json_meta = json.load(f)
        meta.update(json_meta)
    
    # 6. 构建四层数据字典
    data_dict = {
        "meta": meta,
        "signal": {
            "fnirs": {
                "data": data,
                "sampling_rate": meta["sampling_rate"],
                "unit": "V",  # 假设单位是伏特
                "channel_names": meta["channel_names"][:data.shape[0]],
                "wavelengths": meta["wavelengths"],
            }
        },
        "event": events_info,
        "processed": {}
    }
    
    # 添加距离信息（如果可用）
    if "distances" in meta:
        data_dict["signal"]["fnirs"]["distances"] = meta["distances"]
    
    print(f"数据加载完成: 形状={data.shape}, 采样率={meta['sampling_rate']}Hz, 通道数={data.shape[0]}, 波长数={len(meta['wavelengths'])}")
    
    return data_dict


def load_snirf_h5py(snirf_file):
    """
    使用h5py读取SNIRF文件
    
    Args:
        snirf_file: SNIRF文件路径
        
    Returns:
        data: fNIRS数据数组
        sampling_rate: 采样率
        additional_info: 附加信息字典
    """
    try:
        with h5py.File(snirf_file, 'r') as f:
            # 获取数据
            if 'nirs' in f and 'data1' in f['nirs'] and 'dataTimeSeries' in f['nirs']['data1']:
                data_ts = f['nirs']['data1']['dataTimeSeries'][:]
                print(f"从SNIRF文件读取数据形状: {data_ts.shape}")
                
                # 获取测量列表信息
                if 'measurementList1' in f['nirs']['data1']:
                    ml = f['nirs']['data1']['measurementList1']
                    
                    # 获取波长信息
                    wavelengths = []
                    if 'wavelengthActual' in ml:
                        wavelengths = ml['wavelengthActual'][:]
                        print(f"波长: {wavelengths}")
                    
                    # 获取通道信息
                    n_channels = data_ts.shape[1]
                    
                    # 尝试获取源探测器距离
                    distances = []
                    if 'sourcePos' in f['nirs'] and 'detectorPos' in f['nirs']:
                        source_pos = f['nirs']['sourcePos'][:]
                        detector_pos = f['nirs']['detectorPos'][:]
                        # 这里需要根据测量列表中的源探测器索引计算距离
                        # 简化处理：假设每个测量对应一个源-探测器对
                        for i in range(n_channels):
                            if i < len(source_pos) and i < len(detector_pos):
                                dist = np.linalg.norm(source_pos[i] - detector_pos[i])
                                distances.append(dist)
                    
                    # 获取采样率
                    sampling_rate = 10.0  # 默认值
                    if 'time' in f['nirs']['data1'] and len(f['nirs']['data1']['time']) > 1:
                        time = f['nirs']['data1']['time'][:]
                        if len(time) > 1:
                            sampling_rate = 1.0 / (time[1] - time[0])
                    
                    # 重塑数据为三维数组 (channels, wavelengths, samples)
                    # 注意: 这取决于SNIRF文件的具体结构
                    n_wavelengths = len(wavelengths) if len(wavelengths) > 0 else 2
                    n_samples = data_ts.shape[0]
                    
                    # 假设数据已经是按波长和通道排列的
                    if data_ts.shape[1] % n_wavelengths == 0:
                        n_channels_actual = data_ts.shape[1] // n_wavelengths
                        # 重塑为 (channels, wavelengths, samples)
                        data_3d = data_ts.T.reshape(n_channels_actual, n_wavelengths, n_samples)
                        print(f"重塑数据为三维: {data_3d.shape}")
                        return data_3d, sampling_rate, {"wavelengths": wavelengths, "distances": distances}
                    else:
                        # 如果不能重塑，保持二维
                        print(f"保持数据为二维: {data_ts.T.shape}")
                        return data_ts.T, sampling_rate, {"wavelengths": wavelengths, "distances": distances}
            
            # 如果上述路径不存在，尝试其他路径
            print("使用备用路径读取数据")
            # 这里可以添加其他读取逻辑
            
            # 如果都无法读取，返回模拟数据
            return create_simulated_fNIRS_data()
            
    except Exception as e:
        print(f"读取SNIRF文件时出错: {e}")
        # 返回模拟数据作为后备
        return create_simulated_fNIRS_data()


def create_simulated_fNIRS_data():
    """
    创建模拟的fNIRS数据用于测试
    
    Returns:
        data: 模拟的fNIRS数据 (channels, wavelengths, samples)
        sampling_rate: 采样率
    """
    print("生成模拟fNIRS数据")
    
    # 模拟参数
    sampling_rate = 10.0  # Hz
    duration = 60.0  # 秒
    n_samples = int(duration * sampling_rate)
    n_channels = 16
    n_wavelengths = 2  # 730nm和850nm
    
    # 时间轴
    t = np.arange(n_samples) / sampling_rate
    
    # 创建基础信号
    data = np.zeros((n_channels, n_wavelengths, n_samples))
    
    for ch in range(n_channels):
        for wl in range(n_wavelengths):
            # 基础血流动力学响应
            hrf = np.zeros(n_samples)
            
            # 添加事件相关的HRF响应
            for event_time in [10.0, 30.0, 50.0]:
                event_sample = int(event_time * sampling_rate)
                # HRF模型 (双伽马函数)
                hrf_response = 6 * (t - event_time/sampling_rate)**5 * np.exp(-(t - event_time/sampling_rate)) / 120
                hrf += np.roll(hrf_response, event_sample)[:n_samples]
            
            # 添加生理噪声 (心跳 ~1Hz, 呼吸 ~0.3Hz)
            cardiac = 0.1 * np.sin(2 * np.pi * 1.0 * t)
            respiration = 0.05 * np.sin(2 * np.pi * 0.3 * t)
            
            # 添加慢漂移
            drift = 0.01 * np.sin(2 * np.pi * 0.01 * t)
            
            # 添加随机噪声
            noise = 0.02 * np.random.randn(n_samples)
            
            # 合并信号
            signal = hrf + cardiac + respiration + drift + noise
            
            # 对不同波长应用不同的增益
            if wl == 0:  # 730nm
                signal = signal * 1.2 + 0.5
            else:  # 850nm
                signal = signal * 1.0 + 0.6
            
            # 添加随机通道变化
            signal = signal * (0.9 + 0.2 * np.random.rand())
            
            # 添加运动伪影（模拟）
            if ch % 4 == 0:  # 每4个通道添加一个运动伪影
                motion_start = int(20 * sampling_rate)
                motion_end = int(22 * sampling_rate)
                signal[motion_start:motion_end] += 0.5 * np.random.randn(motion_end - motion_start)
            
            data[ch, wl, :] = signal
    
    print(f"生成模拟数据: 形状={data.shape}, 采样率={sampling_rate}Hz")
    
    return data, sampling_rate


# ====================== 测试函数 ======================

def test_fNIRS_preprocessing(subject_dir, task_name="task-1"):
    """
    测试fNIRS预处理流程
    
    Args:
        subject_dir: 被试数据目录
        task_name: 任务名称
    """
    print("=" * 60)
    print(f"开始测试fNIRS预处理: {subject_dir}, 任务: {task_name}")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载BIDS格式fNIRS数据...")
    try:
        data_dict = load_bids_fNIRS_data(subject_dir, task_name)
        print("✓ 数据加载成功")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return None
    
    # 2. 创建fNIRS预处理配置
    print("\n2. 创建fNIRS预处理配置...")
    fnirs_config = fNIRSConfig(
        # 通用预处理参数
        lowcut=0.01,  # 血流动力学响应的低频截止
        highcut=0.5,   # 血流动力学响应的高频截止
        filter_type=FilterType.BUTTERWORTH,
        filter_order=4,
        detrend_method=DetrendMethod.LINEAR,
        remove_baseline=True,
        normalize_method="zscore",
        remove_outliers=True,
        outlier_threshold=3.0,
        
        # fNIRS特有参数
        optical_model=OpticalModel.MODIFIED_BEER_LAMBERT,
        motion_correction_method=MotionCorrectionMethod.SPLINE,
        motion_correction_threshold=3.0,
        use_channel_quality_assessment=True,
        snr_threshold=15.0,
        intensity_cv_threshold=0.3,
        use_short_channel_regression=False,  # 需要距离信息
        remove_physiological_noise=True,
        baseline_correction_window=(-5.0, 0.0),
        use_percentage_baseline=False
    )
    
    # 3. 创建fNIRS预处理器
    print("\n3. 创建fNIRS预处理器...")
    fnirs_processor = fNIRSPreprocessor(fnirs_config)
    print("✓ 预处理器创建成功")
    
    # 4. 执行fNIRS预处理
    print("\n4. 执行fNIRS预处理...")
    try:
        processed_data = fnirs_processor.process_fNIRS(
            data_dict,
            modality="fnirs",
            return_hb_types=["HbO", "HbR"]
        )
        print("✓ fNIRS预处理成功完成")
    except Exception as e:
        print(f"✗ fNIRS预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 5. 显示处理结果信息
    print("\n5. 预处理结果信息:")
    
    # 检查处理历史
    if "processed" in processed_data and "fNIRS_processing" in processed_data["processed"]:
        processing_info = processed_data["processed"]["fNIRS_processing"]["fnirs"]
        
        if "steps" in processing_info:
            for i, step_record in enumerate(processing_info["steps"]):
                print(f"  步骤{i+1}: {step_record.get('step', 'unknown')}")
                if "n_good_channels" in step_record:
                    print(f"    通过质量检测的通道: {step_record['n_good_channels']}/{step_record['n_total_channels']}")
                if "n_motion_events" in step_record:
                    print(f"    检测到的运动事件: {step_record['n_motion_events']}")
    
    # 检查生成的信号
    print("\n6. 生成的信号模态:")
    for modality in processed_data["signal"]:
        signal_info = processed_data["signal"][modality]
        data_shape = signal_info["data"].shape
        print(f"  {modality}: 形状={data_shape}, 采样率={signal_info['sampling_rate']}Hz")
    
    return processed_data


def visualize_results(processed_data):
    """
    可视化预处理结果
    
    Args:
        processed_data: 处理后的数据字典
    """
    print("\n7. 生成可视化...")
    
    try:
        # 尝试使用预处理模块自带的可视化功能
        if hasattr(processed_data.get("processed", {}).get("fNIRS_processing", {}).get("fnirs", {}), "visualize"):
            # 如果有可视化方法，使用它
            pass
        else:
            # 否则创建自定义可视化
            create_custom_visualization(processed_data)
    except Exception as e:
        print(f"可视化失败: {e}")
        # 尝试简单的可视化
        try:
            create_simple_visualization(processed_data)
        except Exception as e2:
            print(f"简单可视化也失败: {e2}")


def create_custom_visualization(data_dict):
    """
    创建自定义可视化
    
    Args:
        data_dict: 处理后的数据字典
    """
    import matplotlib.pyplot as plt
    
    # 检查是否有HbO和HbR信号
    if "HbO" not in data_dict["signal"] or "HbR" not in data_dict["signal"]:
        print("没有找到HbO或HbR信号")
        return
    
    hbo_data = data_dict["signal"]["HbO"]["data"]
    hbr_data = data_dict["signal"]["HbR"]["data"]
    sampling_rate = data_dict["signal"]["HbO"]["sampling_rate"]
    
    n_channels = min(4, hbo_data.shape[0])  # 显示前4个通道
    n_samples = hbo_data.shape[1]
    time_axis = np.arange(n_samples) / sampling_rate
    
    # 创建图形
    fig, axes = plt.subplots(n_channels, 2, figsize=(14, 3 * n_channels))
    fig.suptitle('HbOandHbR', fontsize=16)
    
    if n_channels == 1:
        axes = axes.reshape(1, -1)
    
    for ch in range(n_channels):
        # HbO信号
        axes[ch, 0].plot(time_axis, hbo_data[ch, :])
        axes[ch, 0].set_title(f'channal {ch+1} - HbO')
        axes[ch, 0].set_xlabel('time (s )')
        axes[ch, 0].set_ylabel('contribution (μM)')
        axes[ch, 0].grid(True, alpha=0.3)
        
        # HbR信号
        axes[ch, 1].plot(time_axis, hbr_data[ch, :])
        axes[ch, 1].set_title(f'channel {ch+1} - HbR')
        axes[ch, 1].set_xlabel('time (s )')
        axes[ch, 1].set_ylabel('contribution (μM)')
        axes[ch, 1].grid(True, alpha=0.3)
        
        # 标记事件位置
        if "event" in data_dict and "event_time" in data_dict["event"]:
            for event_time in data_dict["event"]["event_time"]:
                axes[ch, 0].axvline(x=event_time, color='r', linestyle='--', alpha=0.5)
                axes[ch, 1].axvline(x=event_time, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('fnirs_preprocessing_results.png', dpi=150, bbox_inches='tight')
    print("✓ 可视化已保存为 'fnirs_preprocessing_results.png'")
    plt.show()


def create_simple_visualization(data_dict):
    """
    创建简单可视化
    
    Args:
        data_dict: 处理后的数据字典
    """
    import matplotlib.pyplot as plt
    
    # 只显示第一个通道的HbO信号
    if "HbO" in data_dict["signal"]:
        hbo_data = data_dict["signal"]["HbO"]["data"]
        sampling_rate = data_dict["signal"]["HbO"]["sampling_rate"]
        
        plt.figure(figsize=(12, 4))
        plt.plot(np.arange(hbo_data.shape[1]) / sampling_rate, hbo_data[0, :])
        plt.title('channel 1 - HbOsignal')
        plt.xlabel('time (s )')
        plt.ylabel('contribution (μM)')
        plt.grid(True, alpha=0.3)
        plt.savefig('fnirs_hbo_channel1.png', dpi=150, bbox_inches='tight')
        print("✓ 简单可视化已保存为 'fnirs_hbo_channel1.png'")
        plt.show()


def save_processed_data(processed_data, output_dir="output"):
    """
    保存处理后的数据
    
    Args:
        processed_data: 处理后的数据字典
        output_dir: 输出目录
    """
    print(f"\n8. 保存处理结果到目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为numpy文件
    np.savez_compressed(
        os.path.join(output_dir, "processed_fNIRS_data.npz"),
        hbo=processed_data["signal"]["HbO"]["data"],
        hbr=processed_data["signal"]["HbR"]["data"],
        sampling_rate=processed_data["signal"]["HbO"]["sampling_rate"]
    )
    print(f"✓ 数据已保存为: {output_dir}/processed_fNIRS_data.npz")
    
    # 保存元数据为JSON
    meta_data = {
        "subject_id": processed_data["meta"].get("subject_id", "unknown"),
        "task": processed_data["meta"].get("task", "unknown"),
        "sampling_rate": processed_data["signal"]["HbO"]["sampling_rate"],
        "n_channels": processed_data["signal"]["HbO"]["data"].shape[0],
        "n_samples": processed_data["signal"]["HbO"]["data"].shape[1],
        "processing_steps": processed_data["processed"].get("fNIRS_processing", {}).get("fnirs", {}).get("steps", [])
    }
    
    with open(os.path.join(output_dir, "processing_metadata.json"), 'w') as f:
        json.dump(meta_data, f, indent=2, default=str)
    print(f"✓ 元数据已保存为: {output_dir}/processing_metadata.json")
    
    # 保存通道平均值的时间序列（便于检查）
    hbo_mean = np.mean(processed_data["signal"]["HbO"]["data"], axis=0)
    hbr_mean = np.mean(processed_data["signal"]["HbR"]["data"], axis=0)
    
    time_series_df = pd.DataFrame({
        "time": np.arange(len(hbo_mean)) / processed_data["signal"]["HbO"]["sampling_rate"],
        "HbO_mean": hbo_mean,
        "HbR_mean": hbr_mean
    })
    
    time_series_df.to_csv(os.path.join(output_dir, "channel_average_timeseries.csv"), index=False)
    print(f"✓ 时间序列已保存为: {output_dir}/channel_average_timeseries.csv")


# ====================== 主测试函数 ======================

def main():
    """
    主测试函数
    """
    print("=" * 60)
    print("fNIRS预处理模块测试")
    print("=" * 60)
    
    # 测试选项
    test_option = input("选择测试模式:\n1. 使用BIDS数据目录\n2. 使用模拟数据\n请输入选项 (1或2): ").strip()
    
    if test_option == "1":
        # 使用实际的BIDS数据目录
        data_dir = input("请输入BIDS数据目录路径 (例如: D:/BCI/fnirs_dataset/sub-1): ").strip()
        task_name = input("请输入任务名称 (例如: task-1, 按回车使用默认值): ").strip()
        
        if not task_name:
            task_name = "task-1"
        
        if not os.path.exists(data_dir):
            print(f"错误: 目录不存在: {data_dir}")
            print("将使用模拟数据模式")
            test_option = "2"
        else:
            # 测试实际数据
            processed_data = test_fNIRS_preprocessing(data_dir, task_name)
    else:
        # 使用模拟数据
        print("\n使用模拟数据进行测试...")
        
        # 创建模拟数据字典
        sampling_rate = 10.0
        duration = 60.0
        n_samples = int(duration * sampling_rate)
        n_channels = 16
        
        # 生成模拟fNIRS数据 (三维: channels, wavelengths, samples)
        fnirs_data = np.random.randn(n_channels, 2, n_samples) * 0.1 + 1.0
        
        # 添加一些结构
        for ch in range(n_channels):
            # 添加一些周期性变化
            t = np.arange(n_samples) / sampling_rate
            fnirs_data[ch, 0, :] += 0.1 * np.sin(2 * np.pi * 0.1 * t)  # 低频振荡
            fnirs_data[ch, 1, :] += 0.08 * np.sin(2 * np.pi * 0.1 * t)  # 低频振荡
            
            # 添加事件相关响应
            for event_time in [10.0, 30.0, 50.0]:
                event_sample = int(event_time * sampling_rate)
                response_duration = int(5 * sampling_rate)
                if event_sample + response_duration < n_samples:
                    fnirs_data[ch, 0, event_sample:event_sample+response_duration] += 0.2
                    fnirs_data[ch, 1, event_sample:event_sample+response_duration] -= 0.15
        
        data_dict = {
            "meta": {
                "subject_id": "simulated",
                "task": "simulation",
                "modality": ["fnirs"],
                "sampling_rate": sampling_rate,
                "n_channels": n_channels,
                "channel_names": [f"CH{i}" for i in range(n_channels)],
                "wavelengths": [730.0, 850.0],
                "distances": np.random.uniform(2.0, 4.0, n_channels)  # 模拟距离
            },
            "signal": {
                "fnirs": {
                    "data": fnirs_data,
                    "sampling_rate": sampling_rate,
                    "unit": "V",
                    "channel_names": [f"CH{i}" for i in range(n_channels)],
                    "wavelengths": [730.0, 850.0],
                    "distances": np.random.uniform(2.0, 4.0, n_channels)
                }
            },
            "event": {
                "event_id": [1, 2, 1],
                "event_label": ["stimulus", "rest", "stimulus"],
                "event_time": [10.0, 30.0, 50.0],
                "event_sample": [100, 300, 500]
            },
            "processed": {}
        }
        
        print(f"模拟数据创建完成: 形状={fnirs_data.shape}, 采样率={sampling_rate}Hz")
        
        # 创建fNIRS预处理配置
        fnirs_config = fNIRSConfig(
            lowcut=0.01,
            highcut=0.5,
            filter_type=FilterType.BUTTERWORTH,
            filter_order=4,
            detrend_method=DetrendMethod.LINEAR,
            remove_baseline=True,
            normalize_method="zscore",
            
            # fNIRS特有参数
            motion_correction_method=MotionCorrectionMethod.SPLINE,
            motion_correction_threshold=3.0,
            use_channel_quality_assessment=True,
            snr_threshold=15.0,
            use_short_channel_regression=True,
            short_channel_distance_threshold=3.0,
            remove_physiological_noise=True,
            baseline_correction_window=(-5.0, 0.0)
        )
        
        # 创建预处理器
        fnirs_processor = fNIRSPreprocessor(fnirs_config)
        
        # 执行预处理
        print("\n执行fNIRS预处理...")
        try:
            processed_data = fnirs_processor.process_fNIRS(
                data_dict,
                modality="fnirs",
                return_hb_types=["HbO", "HbR"]
            )
            print("✓ fNIRS预处理成功完成")
        except Exception as e:
            print(f"✗ fNIRS预处理失败: {e}")
            import traceback
            traceback.print_exc()
            return
    
    if processed_data is None:
        print("预处理失败，无法继续")
        return
    
    # 可视化结果
    visualize_results(processed_data)
    
    # 保存结果
    save_option = input("\n是否保存处理结果? (y/n): ").strip().lower()
    if save_option == 'y':
        output_dir = input("请输入输出目录 (按回车使用默认值 'output'): ").strip()
        if not output_dir:
            output_dir = "output"
        save_processed_data(processed_data, output_dir)
    
    # 显示预处理历史
    print("\n9. 预处理历史记录:")
    for i, record in enumerate(fnirs_processor.history):
        print(f"\n处理记录 {i+1}:")
        print(f"  模态: {record.get('modality', 'unknown')}")
        print(f"  步骤数: {len(record.get('steps', []))}")
        for j, step in enumerate(record.get('steps', [])):
            print(f"    步骤{j+1}: {step.get('step', 'unknown')}")
    
    print("\n" + "=" * 60)
    print("fNIRS预处理测试完成!")
    print("=" * 60)


# ====================== 直接测试函数 ======================

def quick_test():
    """
    快速测试函数，直接运行而不需要用户输入
    """
    print("执行快速测试...")
    
    # 创建模拟数据
    sampling_rate = 10.0
    n_samples = 600
    n_channels = 8
    
    fnirs_data = np.random.randn(n_channels, 2, n_samples) * 0.1 + 1.0
    
    # 添加一些结构
    t = np.arange(n_samples) / sampling_rate
    for ch in range(n_channels):
        fnirs_data[ch, 0, :] += 0.1 * np.sin(2 * np.pi * 0.1 * t)
        fnirs_data[ch, 1, :] += 0.08 * np.sin(2 * np.pi * 0.1 * t)
    
    data_dict = {
        "meta": {
            "subject_id": "test",
            "task": "test_task",
            "sampling_rate": sampling_rate,
            "n_channels": n_channels,
            "channel_names": [f"CH{i}" for i in range(n_channels)],
            "wavelengths": [730.0, 850.0]
        },
        "signal": {
            "fnirs": {
                "data": fnirs_data,
                "sampling_rate": sampling_rate,
                "unit": "V",
                "channel_names": [f"CH{i}" for i in range(n_channels)],
                "wavelengths": [730.0, 850.0]
            }
        },
        "event": {
            "event_time": [10.0, 30.0, 50.0]
        },
        "processed": {}
    }
    
    # 配置和处理器
    fnirs_config = fNIRSConfig(
        lowcut=0.01,
        highcut=0.5,
        motion_correction_method=MotionCorrectionMethod.SPLINE,
        use_channel_quality_assessment=True
    )
    
    fnirs_processor = fNIRSPreprocessor(fnirs_config)
    
    # 执行预处理
    try:
        processed_data = fnirs_processor.process_fNIRS(
            data_dict,
            modality="fnirs",
            return_hb_types=["HbO", "HbR"]
        )
        print("✓ 快速测试成功!")
        
        # 显示结果摘要
        print(f"\n处理结果摘要:")
        print(f"  HbO形状: {processed_data['signal']['HbO']['data'].shape}")
        print(f"  HbR形状: {processed_data['signal']['HbR']['data'].shape}")
        print(f"  处理步骤数: {len(fnirs_processor.history[0]['steps'])}")
        
        return True
    except Exception as e:
        print(f"✗ 快速测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ====================== 运行测试 ======================

if __name__ == "__main__":
    print("fNIRS预处理测试脚本")
    print("请确保fnirs_preprocessing.py和preprocessing.py在同一目录")
    
    # 运行快速测试
    quick_test_result = quick_test()
    
    if quick_test_result:
        print("\n快速测试通过，现在运行完整测试...")
        # 询问是否运行完整测试
        run_full_test = input("\n是否运行完整测试? (y/n): ").strip().lower()
        if run_full_test == 'y':
            main()
        else:
            print("测试完成。")
    else:
        print("快速测试失败，请检查模块代码。")