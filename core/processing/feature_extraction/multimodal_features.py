# -*- coding: utf-8 -*-
"""
Multimodal BCI Feature Extraction Pipeline
===================================================
文件名: multimodal_pipeline.py
描述:
    支持“按需提取”的多模态脑机接口特征提取总调度模块。
"""

import numpy as np
import warnings

try:
    from eeg_features import EEGFeatureExtractor
    from fnirs_features import FNIRSFeatureExtractor
    # 精确导入 EMG 的细分函数
    from emg_features import (extract_time_domain_features as emg_time,
                              extract_frequency_domain_features as emg_freq,
                              extract_time_frequency_features as emg_tf,
                              extract_nonlinear_features as emg_nl)
    # 精确导入 ECG 的细分函数
    from ecg_features import (detect_r_peaks,
                              extract_morphological_features as ecg_morph,
                              extract_hrv_time_features as ecg_hrv_t,
                              extract_hrv_frequency_features as ecg_hrv_f,
                              extract_time_frequency_features as ecg_tf,
                              extract_nonlinear_features as ecg_nl)
except ImportError as e:
    warnings.warn(f"导入底层特征提取模块失败，详细错误: {e}")


class BCIDataProcessor:
    """数据格式处理器：专门用于解析四层字典数据，并解决时间对齐问题。"""

    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        self._validate_and_init_structure()

    def _validate_and_init_structure(self):
        required_layers = ['meta', 'signal', 'event', 'processed']
        for layer in required_layers:
            if layer not in self.data_dict:
                self.data_dict[layer] = {}
        for sub_layer in ['preprocessing', 'epoch', 'feature']:
            if sub_layer not in self.data_dict['processed']:
                self.data_dict['processed'][sub_layer] = {}

    def get_modalities(self) -> list:
        return list(self.data_dict.get('signal', {}).keys())

    def get_modality_data(self, modality: str) -> dict:
        if modality not in self.data_dict['signal']:
            raise ValueError(f"模态 {modality} 不在数据字典中。")
        sig_info = self.data_dict['signal'][modality]
        fs = sig_info.get('sampling_rate', 1000)
        time_offset = sig_info.get('time_offset', 0.0)
        aligned_events = self._align_events(fs, time_offset)
        return {
            'data': sig_info.get('data'),
            'sampling_rate': fs,
            'channel_names': sig_info.get('channel_names', []),
            'events': aligned_events,
            'time_offset': time_offset
        }

    def _align_events(self, fs: float, time_offset: float) -> list:
        aligned_events = []
        event_dict = self.data_dict.get('event', {})
        if not event_dict or 'event_time' not in event_dict:
            return aligned_events

        e_times = event_dict.get('event_time', [])
        for i, t in enumerate(e_times):
            sample_idx = int(max(0, t - time_offset) * fs)
            aligned_events.append({'event_time': t, 'event_sample': sample_idx})
        return aligned_events

    def save_features(self, modality: str, features: dict):
        # 增量更新特征，防止覆盖别的模块特征
        if modality not in self.data_dict['processed']['feature']:
            self.data_dict['processed']['feature'][modality] = {}
        self.data_dict['processed']['feature'][modality].update(features)
        return self.data_dict


class MultimodalFeaturePipeline:
    def __init__(self, data_dict: dict, selected_features: dict = None):
        """
        :param data_dict: 四层字典数据
        :param selected_features: 字典格式，如 {"EMG": ["time_domain", "nonlinear"]}
        """
        self.processor = BCIDataProcessor(data_dict)
        self.data_dict = self.processor.data_dict
        # 如果没有传入，默认提取所有
        self.selected_features = selected_features or {}

    def run_pipeline(self) -> dict:
        modalities = self.processor.get_modalities()

        # 如果 selected_features 不为空，则只处理在配置中被指定的模态
        if self.selected_features:
            modalities = [m for m in modalities if m in self.selected_features]

        for mod in modalities:
            try:
                mod_info = self.processor.get_modality_data(mod)
                data = mod_info['data']
                fs = mod_info['sampling_rate']
                ch_names = mod_info['channel_names']
                event_samples = [ev['event_sample'] for ev in mod_info['events']]

                # 获取用户选择的该模态特征集
                selected_cats = self.selected_features.get(mod, [])
                features = {}

                # ================= EEG 分支 =================
                if mod == 'EEG':
                    extractor = EEGFeatureExtractor(fs=fs, channel_names=ch_names)
                    # 1. 逐通道特征
                    for ch_idx in range(data.shape[0]):
                        ch_data = data[ch_idx]
                        ch_name = ch_names[ch_idx] if ch_names else f"ch{ch_idx}"
                        if 'time_domain' in selected_cats:
                            features.update({f"{ch_name}_{k}": v for k, v in
                                             extractor.compute_time_domain_features(ch_data).items()})
                        if 'freq_domain' in selected_cats:
                            features.update({f"{ch_name}_{k}": v for k, v in
                                             extractor.compute_freq_domain_features(ch_data).items()})
                        if 'wavelet' in selected_cats:
                            features.update(
                                {f"{ch_name}_{k}": v for k, v in extractor.compute_wavelet_features(ch_data).items()})
                        if 'nonlinear' in selected_cats:
                            features.update(
                                {f"{ch_name}_{k}": v for k, v in extractor.compute_nonlinear_features(ch_data).items()})
                    # 2. 跨通道特征
                    if 'band_power' in selected_cats:
                        features.update(extractor.extract_band_powers(data))
                    if 'erp' in selected_cats and event_samples:
                        features.update(extractor.extract_erp_features(data, event_samples))
                    if 'connectivity' in selected_cats and data.shape[0] > 1:
                        conn = extractor.compute_connectivity_features(data, method='coherence', freq_band=(8, 13))
                        features.update({f"{k}_alpha": v for k, v in conn.items()})
                    if 'spatial' in selected_cats and data.shape[0] > 1:
                        spat = extractor.compute_topographic_features(data, freq_band=(8, 13))
                        features.update({f"{k}_alpha": v for k, v in spat.items()})

                # ================= fNIRS 分支 =================
                elif mod == 'fNIRS':
                    extractor = FNIRSFeatureExtractor(fs=fs)
                    # 通用单通道提取
                    for ch_idx in range(data.shape[0]):
                        ch_data = data[ch_idx]
                        ch_name = ch_names[ch_idx] if ch_names else f"ch{ch_idx}"
                        if 'time_domain' in selected_cats:
                            features.update({f"{ch_name}_{k}": v for k, v in
                                             extractor.compute_time_domain_features(ch_data).items()})
                        if 'freq_domain' in selected_cats:
                            features.update({f"{ch_name}_{k}": v for k, v in
                                             extractor.compute_freq_domain_features(ch_data).items()})
                        if 'wavelet' in selected_cats:
                            features.update(
                                {f"{ch_name}_{k}": v for k, v in extractor.compute_wavelet_features(ch_data).items()})
                        if 'nonlinear' in selected_cats:
                            features.update(
                                {f"{ch_name}_{k}": v for k, v in extractor.compute_nonlinear_features(ch_data).items()})
                    # 特定特征
                    if 'hbo_hbr' in selected_cats and data.shape[0] >= 2:
                        features.update(extractor.extract_hbo_hbr_features(data[0], data[1]))
                    if 'channel_correlation' in selected_cats and data.shape[0] >= 2:
                        features.update(extractor.compute_channel_correlation(data))

                # ================= EMG 分支 =================
                elif mod == 'EMG':
                    emg_signal = data[0] if data.ndim == 2 else data
                    if 'time_domain' in selected_cats:
                        features['time_domain'] = emg_time(emg_signal)
                    if 'freq_domain' in selected_cats:
                        features['frequency_domain'] = emg_freq(emg_signal, fs)
                    if 'wavelet' in selected_cats:  # EMG里的时频对应特征
                        features['time_frequency'] = emg_tf(emg_signal, fs)
                    if 'nonlinear' in selected_cats:
                        features['nonlinear'] = emg_nl(emg_signal, fs)

                # ================= ECG 分支 =================
                elif mod == 'ECG':
                    ecg_signal = data[0] if data.ndim == 2 else data
                    # R波是其他特征的基础，只要提取ECG特征，就必须先检波
                    r_peaks = detect_r_peaks(ecg_signal, fs)
                    rr_intervals = np.diff(r_peaks) / fs * 1000

                    if 'morphological' in selected_cats:
                        features['morphological'] = ecg_morph(ecg_signal, r_peaks, fs)
                    if 'hrv_time' in selected_cats:
                        features['hrv_time'] = ecg_hrv_t(rr_intervals)
                    if 'hrv_frequency' in selected_cats:
                        features['hrv_frequency'] = ecg_hrv_f(rr_intervals, 4.0)
                    if 'wavelet' in selected_cats:
                        features['time_frequency'] = ecg_tf(ecg_signal, fs)
                    if 'nonlinear' in selected_cats:
                        features['nonlinear'] = ecg_nl(rr_intervals)

                # 保存特征
                if features:
                    self.processor.save_features(mod, features)

            except Exception as e:
                print(f"[X] 提取模态 {mod} 发生异常: {e}")

        return self.data_dict
