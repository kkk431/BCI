# -*- coding: utf-8 -*-
"""
Multimodal BCI Feature Extraction Pipeline
===================================================
文件名: multimodal_pipeline.py
描述:
    多模态脑机接口特征提取的总调度模块。
    用于解析统一的“四层字典数据结构（meta, signal, event, processed）”，
    自动处理多模态信号并发、采样率差异、事件时间戳对齐等问题，
    并自动路由调用底层的 EEG, fNIRS, EMG, ECG 特征提取器。

依赖:
    需要与以下文件在同一目录下：
    - common_features.py
    - eeg_features.py
    - fnirs_features.py
    - emg_features.py
    - ecg_features.py

作者: BCI Team
版本: 1.0.0
"""

import numpy as np
import warnings

# =========================================================================
# 导入各个模态的底层特征提取模块
# =========================================================================
try:
    from eeg_features import EEGFeatureExtractor
    from fnirs_features import FNIRSFeatureExtractor
    from emg_features import extract_emg_features
    from ecg_features import extract_ecg_features
except ImportError as e:
    warnings.warn(f"导入底层特征提取模块失败，请确保所有模块在同一目录下。详细错误: {e}")


class BCIDataProcessor:
    """
    数据格式处理器：专门用于解析四层字典数据，并解决多模态的时间对齐问题。
    """

    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        self._validate_and_init_structure()

    def _validate_and_init_structure(self):
        """验证四层结构，若缺失则初始化为空，防止后续抛出 KeyError"""
        required_layers = ['meta', 'signal', 'event', 'processed']
        for layer in required_layers:
            if layer not in self.data_dict:
                self.data_dict[layer] = {}

        # 确保 processed 层包含标准结果容器
        for sub_layer in ['preprocessing', 'epoch', 'feature']:
            if sub_layer not in self.data_dict['processed']:
                self.data_dict['processed'][sub_layer] = {}

    def get_modalities(self) -> list:
        """获取当前字典中实际存在的所有信号模态 (例如 ['EEG', 'EMG'])"""
        return list(self.data_dict.get('signal', {}).keys())

    def get_modality_data(self, modality: str) -> dict:
        """
        获取特定模态的预处理数据，并自动完成事件时间的对齐（根据独立采样率和时间偏移）。
        """
        if modality not in self.data_dict['signal']:
            raise ValueError(f"模态 {modality} 不在数据字典中。")

        sig_info = self.data_dict['signal'][modality]
        data = sig_info.get('data')
        fs = sig_info.get('sampling_rate', 1000)
        ch_names = sig_info.get('channel_names', [])
        time_offset = sig_info.get('time_offset', 0.0)

        # 核心：多模态事件对齐
        # 将统一的绝对事件时间 (event_time)，转换为当前模态特有的样本点 (sample_index)
        aligned_events = self._align_events(fs, time_offset)

        return {
            'data': data,
            'sampling_rate': fs,
            'channel_names': ch_names,
            'events': aligned_events,
            'time_offset': time_offset
        }

    def _align_events(self, fs: float, time_offset: float) -> list:
        """
        根据当前模态的采样率和时间偏移量，计算事件在当前信号数组中的真实样本点索引。
        """
        aligned_events = []
        event_dict = self.data_dict.get('event', {})

        if not event_dict or 'event_time' not in event_dict:
            return aligned_events

        e_times = event_dict.get('event_time', [])
        e_labels = event_dict.get('event_label', [])
        e_ids = event_dict.get('event_id', [])
        e_durations = event_dict.get('duration', [])

        for i, t in enumerate(e_times):
            # 样本点 = (事件发生时间 - 该设备的相对偏移) * 该设备采样率
            sample_idx = int(max(0, t - time_offset) * fs)
            aligned_events.append({
                'event_time': t,
                'event_sample': sample_idx,
                'event_id': e_ids[i] if i < len(e_ids) else None,
                'event_label': e_labels[i] if i < len(e_labels) else None,
                'duration': e_durations[i] if i < len(e_durations) else 0.0
            })

        return aligned_events

    def save_features(self, modality: str, features: dict):
        """将提取完毕的特征标准、规范地写回 processed -> feature 层"""
        self.data_dict['processed']['feature'][modality] = features
        return self.data_dict


class MultimodalFeaturePipeline:
    """
    多模态特征提取主管线
    """

    def __init__(self, data_dict: dict):
        # 1. 挂载数据处理器
        self.processor = BCIDataProcessor(data_dict)
        self.data_dict = self.processor.data_dict

    def run_pipeline(self) -> dict:
        """
        执行完整的特征提取流水线
        """
        modalities = self.processor.get_modalities()
        print(f"[*] 启动多模态调度管线，检测到 {len(modalities)} 种模态: {modalities}")

        # 2. 遍历字典中包含的所有模态，分别调度
        for mod in modalities:
            print(f"\n[->] 正在调度特征提取模态: {mod}")

            try:
                # 获取处理后的模态数据和对齐后的事件
                mod_info = self.processor.get_modality_data(mod)
                data = mod_info['data']
                fs = mod_info['sampling_rate']
                ch_names = mod_info['channel_names']
                events = mod_info['events']

                # 提取专属于该模态的 Event Sample 索引数组（供 ERP/HRF 等特征计算使用）
                event_samples = [ev['event_sample'] for ev in events]

                features = {}

                # 3. 根据模态类型，调用对应的底层特征提取算法
                if mod == 'EEG':
                    extractor = EEGFeatureExtractor(fs=fs, channel_names=ch_names)
                    features = extractor.extract_all_eeg_features(
                        data=data,
                        extract_erp=(len(event_samples) > 0),
                        event_times=event_samples,
                        verbose=False
                    )

                elif mod == 'fNIRS':
                    extractor = FNIRSFeatureExtractor(fs=fs)
                    # 如果数据是2D(多通道)，分别提取并计算通道相关性
                    if hasattr(data, 'ndim') and data.ndim == 2:
                        if data.shape[0] >= 2:
                            features.update(extractor.compute_channel_correlation(data))
                        # 逐通道提取通用特征
                        for i in range(data.shape[0]):
                            ch_name = ch_names[i] if i < len(ch_names) else f"ch{i}"
                            ch_feat = extractor.extract_all_features(data[i])
                            # 给每个通道特征加上前缀避免键值冲突
                            features.update({f"{ch_name}_{k}": v for k, v in ch_feat.items()})
                    else:
                        features = extractor.extract_all_features(data)

                elif mod == 'EMG':
                    # EMG文件原生支持处理整个data_dict，只需传入fs保证一致性
                    features = extract_emg_features(self.data_dict, sampling_rate=fs)

                elif mod == 'ECG':
                    # ECG文件原生支持处理整个data_dict，只需传入fs保证一致性
                    features = extract_ecg_features(self.data_dict, sampling_rate=fs)

                else:
                    print(f"  [!] 警告: 暂无匹配的提取器以处理未知模态: {mod}，已跳过。")
                    continue

                # 4. 利用解析器将特征标准化并写回原字典
                self.processor.save_features(mod, features)

                # 兼容不同文件的返回格式（扁平字典或嵌套字典）统计数量
                feat_count = sum(len(v) if isinstance(v, dict) else 1 for v in features.values())
                print(f"  [√] {mod} 特征提取完成！提取特征组/特征数量: {feat_count}")

            except Exception as e:
                print(f"  [X] {mod} 特征提取过程中发生异常: {e}")

        print("\n[*] 多模态特征提取流水线执行完毕！结果已更新至 data_dict['processed']['feature'] 层。")
        return self.data_dict


# =============================================================================
# 使用示例 (Usage Example)
# =============================================================================
if __name__ == "__main__":
    print("=== 测试多模态总调度模块 ===")

    # 构建一个模拟的四层字典数据结构（包含 EEG 和 EMG 两种模态）
    simulated_eeg_data = np.random.normal(0, 1, (2, 5000))  # 2通道，5秒 (1000Hz)
    simulated_emg_data = np.random.normal(0, 1, (1, 10000))  # 1通道，5秒 (2000Hz)

    sample_data_dict = {
        "meta": {
            "subject_id": "S01",
            "task": "motor_imagery"
        },
        "signal": {
            "EEG": {
                "data": simulated_eeg_data,
                "sampling_rate": 1000,
                "channel_names": ["C3", "C4"],
                "time_offset": 0.0,
            },
            "EMG": {
                "data": simulated_emg_data,
                "sampling_rate": 2000,
                "channel_names": ["Right_Arm"],
                "time_offset": 0.002,  # 模拟设备同步的时间延迟
            }
        },
        "event": {
            "event_id": [1, 2],
            "event_label": ["left_hand", "right_hand"],
            "event_time": [1.0, 3.5],  # 物理绝对时间 (秒)
            "duration": [2.0, 2.0]
        },
        # 即使 processed 没传，我们的调度器也会自动初始化它
        "processed": {}
    }

    # 实例化流水线并运行
    pipeline = MultimodalFeaturePipeline(sample_data_dict)
    processed_dict = pipeline.run_pipeline()

    # 验证输出结果
    print("\n--- 输出结构验证 ---")
    print("提取出的模态特征分支:", list(processed_dict['processed']['feature'].keys()))
    print("EEG 的第一项特征示例:", list(processed_dict['processed']['feature']['EEG'].items())[0])

    # 验证事件对齐机制是否生效
    print("\n--- 事件对齐机制验证 ---")
    eeg_events = pipeline.processor.get_modality_data("EEG")['events']
    emg_events = pipeline.processor.get_modality_data("EMG")['events']
    print(f"原始物理时间: 1.0秒")
    print(f"  -> EEG 对齐样本点 (1000Hz, 偏0s): {eeg_events[0]['event_sample']}")
    print(f"  -> EMG 对齐样本点 (2000Hz, 偏0.002s): {emg_events[0]['event_sample']}")
