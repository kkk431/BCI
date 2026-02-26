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
    from core.processing.feature_extraction.eeg_features import EEGFeatureExtractor
    from core.processing.feature_extraction.fnirs_features import FNIRSFeatureExtractor
    from core.processing.feature_extraction.emg_features import EMGFeatureExtractor
    from core.processing.feature_extraction.ecg_features import ECGFeatureExtractor
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
                    # 实例化EMG特征提取器
                    extractor = EMGFeatureExtractor(sampling_rate=fs)

                    # 为EMG创建数据字典（支持多通道）
                    emg_data_dict = {
                        'signal': {
                            'EMG': {
                                'data': data,
                                'sampling_rate': fs,
                                'channel_names': ch_names
                            }
                        }
                    }

                    # 使用extract_emg_features提取所有特征
                    all_features = extractor.extract_emg_features(emg_data_dict)

                    # 根据用户选择过滤特征类别
                    for cat in selected_cats:
                        if cat == 'time_domain' and 'time_domain' in all_features:
                            # 为时域特征添加前缀
                            time_feats = {f"emg_time_{k}": v for k, v in all_features['time_domain'].items()}
                            features.update(time_feats)
                        elif cat == 'freq_domain' and 'frequency_domain' in all_features:
                            freq_feats = {f"emg_freq_{k}": v for k, v in all_features['frequency_domain'].items()}
                            features.update(freq_feats)
                        elif cat == 'wavelet' and 'wavelet' not in all_features:  # EMG没有单独的wavelet
                            # EMG的wavelet特征实际上在frequency_domain中已经包含
                            pass
                        elif cat == 'nonlinear' and 'nonlinear' in all_features:
                            nl_feats = {f"emg_nl_{k}": v for k, v in all_features['nonlinear'].items()}
                            features.update(nl_feats)

                    # 如果用户选择了通用特征，添加common特征
                    if 'common' in all_features:
                        common_feats = {f"emg_common_{k}": v for k, v in all_features['common'].items()}
                        features.update(common_feats)

                # ================= ECG 分支 =================
                elif mod == 'ECG':
                    # 实例化ECG特征提取器
                    extractor = ECGFeatureExtractor(sampling_rate=fs)

                    # 为ECG创建数据字典（支持多通道）
                    ecg_data_dict = {
                        'signal': {
                            'ECG': {
                                'data': data,
                                'sampling_rate': fs,
                                'channel_names': ch_names
                            }
                        }
                    }

                    # 使用extract_ecg_features提取所有特征
                    all_features = extractor.extract_ecg_features(ecg_data_dict)

                    # 根据用户选择过滤特征类别
                    for cat in selected_cats:
                        if cat == 'morphological' and 'morphological' in all_features:
                            morph_feats = {f"ecg_morph_{k}": v for k, v in all_features['morphological'].items()}
                            features.update(morph_feats)
                        elif cat == 'hrv_time' and 'hrv_time' in all_features:
                            hrv_t_feats = {f"ecg_hrv_t_{k}": v for k, v in all_features['hrv_time'].items()}
                            features.update(hrv_t_feats)
                        elif cat == 'hrv_frequency' and 'hrv_frequency' in all_features:
                            hrv_f_feats = {f"ecg_hrv_f_{k}": v for k, v in all_features['hrv_frequency'].items()}
                            features.update(hrv_f_feats)
                        elif cat == 'wavelet' and 'wavelet' not in all_features:  # ECG没有单独的wavelet
                            # ECG的wavelet特征可能在其他地方
                            pass
                        elif cat == 'nonlinear' and 'hrv_nonlinear' in all_features:
                            nl_feats = {f"ecg_hrv_nl_{k}": v for k, v in all_features['hrv_nonlinear'].items()}
                            features.update(nl_feats)

                    # 如果用户选择了通用特征，添加common特征
                    if 'common' in all_features:
                        common_feats = {f"ecg_common_{k}": v for k, v in all_features['common'].items()}
                        features.update(common_feats)

                # 保存特征
                if features:
                    self.processor.save_features(mod, features)
                    print(f"模态 {mod} 特征提取完成，共 {len(features)} 个特征")

            except Exception as e:
                print(f"提取模态 {mod} 发生异常: {e}")
                import traceback
                traceback.print_exc()

        return self.data_dict