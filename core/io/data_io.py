"""
universal_biosignal_processor.py
万能生物信号数据处理器
支持所有生物信号：EEG、fNIRS、EMG、ECG、EOG、GSR、RESP、ET等
基于标准化的四层 data_dict 格式。
数据结构参考：https://xcnmvog3p8wo.feishu.cn/wiki/NIo8wyMfqiaZSzkm8ERcV7vOnie
"""

import os
import json
import numpy as np
import pandas as pd
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from pathlib import Path
from enum import Enum
import hashlib

# ==================== 信号类型枚举 ====================
class SignalType(Enum):
    """信号类型枚举"""
    EEG = "eeg"           # 脑电
    FNIRS = "fnirs"       # 近红外
    EMG = "emg"           # 肌电
    ECG = "ecg"           # 心电
    EOG = "eog"           # 眼电
    GSR = "gsr"           # 皮肤电（皮电）
    RESP = "resp"         # 呼吸
    ET = "eyetracker"     # 眼动
    PPG = "ppg"           # 光电容积脉搏波
    TEMP = "temperature"  # 体温
    BVP = "bvp"           # 血容量脉搏波
    ACC = "accelerometer" # 加速度计
    GYRO = "gyroscope"    # 陀螺仪
    MAG = "magnetometer"  # 磁力计
    OTHER = "other"       # 其他

# ==================== 信号特征数据库 ====================
SIGNAL_CHARACTERISTICS = {
    SignalType.EEG: {
        "typical_fs": [250, 500, 1000, 2000],
        "frequency_range": (0.5, 100),
        "amplitude_range": (-200, 200),  # uV
        "typical_units": ["uV", "V"],
        "reference_types": ["average", "mastoid", "cz", "linked_mastoids"],
        "artifact_types": ["blink", "eye_movement", "muscle", "heart", "line_noise"]
    },
    SignalType.EMG: {
        "typical_fs": [1000, 2000, 5000],
        "frequency_range": (20, 500),
        "amplitude_range": (-5000, 5000),  # mV级别
        "typical_units": ["mV", "uV"],
        "muscle_types": ["biceps", "triceps", "deltoid", "quadriceps", "gastrocnemius"],
        "processing_steps": ["rectification", "envelope", "rms"]
    },
    SignalType.ECG: {
        "typical_fs": [250, 500, 1000],
        "frequency_range": (0.05, 100),
        "amplitude_range": (-5, 5),  # mV级别
        "typical_units": ["mV", "uV"],
        "lead_types": ["I", "II", "III", "aVR", "aVL", "aVF", "V1-V6"],
        "features": ["R_peaks", "RR_intervals", "heart_rate", "QRS_complex"]
    },
    SignalType.EOG: {
        "typical_fs": [250, 500, 1000],
        "frequency_range": (0.1, 35),
        "amplitude_range": (-1000, 1000),  # uV级别
        "typical_units": ["uV"],
        "directions": ["horizontal", "vertical", "radial"],
        "event_types": ["saccade", "fixation", "blink", "smooth_pursuit"]
    },
    SignalType.GSR: {
        "typical_fs": [10, 20, 100],
        "frequency_range": (0, 5),
        "amplitude_range": (0, 100),  # 皮肤电导(uS)
        "typical_units": ["uS", "S"],
        "components": ["tonic", "phasic"],
        "features": ["skin_conductance_level", "skin_conductance_response"]
    },
    SignalType.RESP: {
        "typical_fs": [10, 50, 100],
        "frequency_range": (0.1, 2),
        "amplitude_range": (-5, 5),  # V级别
        "typical_units": ["V", "mV"],
        "sensor_types": ["thermistor", "strain_gauge", "impedance", "pressure"],
        "features": ["respiration_rate", "tidal_volume", "inspiration_time"]
    },
    SignalType.ET: {
        "typical_fs": [30, 60, 120, 250, 1000],
        "data_columns": ["x", "y", "pupil", "timestamp"],
        "units": ["pixel", "degree", "mm"],
        "event_types": ["fixation", "saccade", "blink", "smooth_pursuit"],
        "metrics": ["pupil_diameter", "gaze_x", "gaze_y", "velocity", "acceleration"]
    },
    SignalType.FNIRS: {
        "typical_fs": [10, 50, 100],
        "wavelengths": [760, 780, 805, 830, 850],
        "units": ["optical_density", "mmol/L"],
        "source_detector_distances": [15, 25, 30, 35],  # mm
        "hemoglobin_types": ["HbO", "HbR", "HbT"]
    }
}

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 数据字典构建器 ====================
class DataDictBuilder:
    """
    标准数据字典构建器
    基于四层结构：meta, signal, event, processed
    """

    @staticmethod
    def create_empty_data_dict() -> Dict[str, Any]:
        """创建空的四层数据字典"""
        return {
            "meta": {
                "subject_id": "",
                "session_id": "",
                "task": "",
                "recording_time": "",
                "file_path": "",
                "format_version": "2.0",
                "modality": [],
                "device": "",
                "notes": ""
            },
            "signal": {},
            "event": {
                "event_id": [],
                "event_label": [],
                "event_time": [],
                "duration": []
            },
            "processed": {
                "features": {},
                "artifacts": {},
                "filtered_data": {}
            }
        }

    @staticmethod
    def build_meta(subject_id: str = "", session_id: str = "",
                   task: str = "", recording_time: str = "",
                   file_path: str = "", modality: List[str] = None,
                   device: str = "", sampling_rate: float = None,
                   n_channels: int = None, channel_names: List[str] = None,
                   **kwargs) -> Dict[str, Any]:
        """构建meta层"""
        if modality is None:
            modality = []
        if channel_names is None:
            channel_names = []

        meta = {
            "subject_id": subject_id,
            "session_id": session_id,
            "task": task,
            "recording_time": recording_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": file_path,
            "format_version": "2.0",
            "modality": modality,
            "device": device,
            "notes": kwargs.get("notes", "")
        }

        # 添加技术信息
        if sampling_rate is not None:
            meta["sampling_rate"] = float(sampling_rate)
        if n_channels is not None:
            meta["n_channels"] = int(n_channels)
        if channel_names:
            meta["channel_names"] = channel_names

        # 添加额外参数
        for key, value in kwargs.items():
            if key not in meta and key != "notes":
                meta[key] = value

        return meta

    @staticmethod
    def add_signal(data_dict: Dict, data: np.ndarray, sampling_rate: float,
                   channel_names: List[str], modality: str,
                   signal_type: str = None, unit: str = None,
                   **signal_info) -> Dict:
        """添加信号到signal层"""
        if "signal" not in data_dict:
            data_dict["signal"] = {}

        # 验证数据维度
        if data.ndim == 1:
            data = data.reshape(1, -1)  # 转为(1, n_samples)
        elif data.ndim == 2 and data.shape[0] > data.shape[1]:
            # 如果通道数多于样本数，可能维度不对，尝试转置
            if len(channel_names) == data.shape[1]:
                data = data.T

        # 验证通道名数量匹配
        if len(channel_names) != data.shape[0]:
            logger.warning(f"通道名数量({len(channel_names)})与数据通道数({data.shape[0]})不匹配")
            channel_names = [f"Ch{i+1}" for i in range(data.shape[0])]

        # 构建信号信息
        signal_entry = {
            "data": np.asarray(data, dtype=np.float32),
            "sampling_rate": float(sampling_rate),
            "channel_names": list(channel_names),
            "signal_type": signal_type or modality.lower(),
            "unit": unit or "unknown",
            "n_channels": data.shape[0],
            "n_samples": data.shape[1],
            "duration": data.shape[1] / sampling_rate,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **signal_info
        }

        # 添加到signal层
        data_dict["signal"][modality.upper()] = signal_entry

        # 更新meta层的modality列表
        if modality.upper() not in data_dict["meta"].get("modality", []):
            data_dict["meta"].setdefault("modality", []).append(modality.upper())

        return data_dict

    @staticmethod
    def add_event(data_dict: Dict, event_label: Union[str, List[str]],
                  event_time: Union[float, List[float]],
                  duration: Union[float, List[float]] = None,
                  event_id: Union[int, List[int]] = None) -> Dict:
        """添加事件到event层"""
        if "event" not in data_dict:
            data_dict["event"] = {
                "event_id": [],
                "event_label": [],
                "event_time": [],
                "duration": []
            }

        # 处理标量输入
        if isinstance(event_label, str):
            event_label = [event_label]
        if isinstance(event_time, (int, float)):
            event_time = [float(event_time)]
        if isinstance(duration, (int, float)):
            duration = [float(duration)]
        elif duration is None:
            duration = [0.0] * len(event_label)

        # 生成事件ID
        if event_id is None:
            start_id = len(data_dict["event"]["event_id"]) + 1
            event_id = list(range(start_id, start_id + len(event_label)))
        elif isinstance(event_id, int):
            event_id = [event_id]

        # 添加到event层
        data_dict["event"]["event_id"].extend(event_id)
        data_dict["event"]["event_label"].extend(event_label)
        data_dict["event"]["event_time"].extend(event_time)
        data_dict["event"]["duration"].extend(duration)

        return data_dict

    @staticmethod
    def add_processed_data(data_dict: Dict, processed_name: str,
                          data: np.ndarray, processing_steps: List[str] = None,
                          parameters: Dict = None) -> Dict:
        """添加处理后的数据到processed层"""
        if "processed" not in data_dict:
            data_dict["processed"] = {"filtered_data": {}, "features": {}, "artifacts": {}}

        if processed_name not in data_dict["processed"]["filtered_data"]:
            data_dict["processed"]["filtered_data"][processed_name] = {}

        data_dict["processed"]["filtered_data"][processed_name] = {
            "data": np.asarray(data, dtype=np.float32),
            "processing_steps": processing_steps or [],
            "parameters": parameters or {},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return data_dict

    @staticmethod
    def add_features(data_dict: Dict, feature_name: str,
                     features: Dict[str, Any], modality: str = None) -> Dict:
        """添加特征到processed层"""
        if "processed" not in data_dict:
            data_dict["processed"] = {"filtered_data": {}, "features": {}, "artifacts": {}}

        if "features" not in data_dict["processed"]:
            data_dict["processed"]["features"] = {}

        if modality:
            if modality not in data_dict["processed"]["features"]:
                data_dict["processed"]["features"][modality] = {}
            data_dict["processed"]["features"][modality][feature_name] = features
        else:
            data_dict["processed"]["features"][feature_name] = features

        return data_dict

    @staticmethod
    def detect_signal_type(channel_names: List[str], data: np.ndarray = None,
                          sampling_rate: float = None) -> SignalType:
        """自动检测信号类型"""
        channel_names_lower = [name.lower() for name in channel_names]

        # 基于通道名关键词检测
        keyword_mapping = {
            SignalType.EEG: ['eeg', 'f', 'c', 'p', 'o', 't', 'a', 'fp', 'cz', 'pz', 'oz'],
            SignalType.ECG: ['ecg', 'ekg', 'heart', 'cardio'],
            SignalType.EMG: ['emg', 'muscle', 'biceps', 'triceps', 'deltoid'],
            SignalType.EOG: ['eog', 'heog', 'veog', 'eye', 'blink'],
            SignalType.GSR: ['gsr', 'eda', 'skin', 'conductance', 'sc'],
            SignalType.RESP: ['resp', 'breath', 'thoracic', 'abdominal'],
            SignalType.FNIRS: ['fnirs', 'nirs', 'hbo', 'hbr', 'source', 'detector'],
            SignalType.ET: ['gaze', 'pupil', 'eyetrack', 'fixation', 'saccade'],
            SignalType.PPG: ['ppg', 'pulse', 'blood', 'volume'],
            SignalType.ACC: ['acc', 'accelerometer', 'accel'],
            SignalType.GYRO: ['gyro', 'gyroscope'],
            SignalType.MAG: ['mag', 'magnetometer']
        }

        for signal_type, keywords in keyword_mapping.items():
            for channel_name in channel_names_lower:
                if any(keyword in channel_name for keyword in keywords):
                    return signal_type

        # 如果数据可用，基于数据特征检测
        if data is not None and sampling_rate is not None:
            try:
                from scipy.signal import welch

                if data.ndim == 1:
                    data = data.reshape(1, -1)

                # 分析信号频谱
                freqs, psd = welch(data[0], fs=sampling_rate, nperseg=min(1024, data.shape[1]))

                # ECG检测：低频能量高，有典型心率频率
                if sampling_rate > 100:  # ECG通常采样率较高
                    hr_band = (0.8, 3.0)  # 心率对应的频率范围(48-180 BPM)
                    hr_power = np.sum(psd[(freqs >= hr_band[0]) & (freqs <= hr_band[1])])
                    total_power = np.sum(psd)
                    if hr_power / total_power > 0.2:
                        return SignalType.ECG

                # RESP检测：非常低频(0.1-0.5 Hz)
                resp_band = (0.1, 0.5)
                resp_power = np.sum(psd[(freqs >= resp_band[0]) & (freqs <= resp_band[1])])
                total_power = np.sum(psd)
                if resp_power / total_power > 0.3:
                    return SignalType.RESP

            except Exception as e:
                logger.debug(f"频谱分析失败: {e}")

        return SignalType.OTHER

# ==================== 信号处理器基类 ====================
class SignalProcessor:
    """信号处理器基类"""

    def __init__(self):
        self.builder = DataDictBuilder()

    def process(self, data_dict: Dict) -> Dict:
        """处理信号（子类需重写）"""
        raise NotImplementedError("子类必须实现process方法")

    def extract_features(self, data: np.ndarray, fs: float) -> Dict[str, Any]:
        """提取特征（子类可重写）"""
        return {}

    def detect_artifacts(self, data: np.ndarray, fs: float) -> Dict[str, Any]:
        """检测伪迹（子类可重写）"""
        return {}

# ==================== EEG信号处理器 ====================
class EEGProcessor(SignalProcessor):
    """EEG信号处理器"""

    def __init__(self, reference_method: str = "average",
                 notch_freq: float = 50.0, bandpass_range: Tuple[float, float] = (0.5, 45.0)):
        super().__init__()
        self.reference_method = reference_method
        self.notch_freq = notch_freq
        self.bandpass_range = bandpass_range

    def process(self, data_dict: Dict) -> Dict:
        """处理EEG信号"""
        if "EEG" not in data_dict.get("signal", {}):
            return data_dict

        eeg_info = data_dict["signal"]["EEG"]
        data = eeg_info["data"]
        fs = eeg_info["sampling_rate"]

        # 1. 重参考
        if self.reference_method == "average":
            ref_data = data - np.mean(data, axis=0, keepdims=True)
        else:
            ref_data = data  # 其他参考方法需要具体实现

        # 2. 滤波
        filtered_data = self._apply_filters(ref_data, fs)

        # 3. 提取特征
        features = self.extract_features(filtered_data, fs)

        # 4. 检测伪迹
        artifacts = self.detect_artifacts(filtered_data, fs)

        # 保存处理结果
        self.builder.add_processed_data(data_dict, "EEG_filtered", filtered_data,
                                       processing_steps=["rereference", "filtering"],
                                       parameters={
                                           "reference_method": self.reference_method,
                                           "notch_freq": self.notch_freq,
                                           "bandpass_range": self.bandpass_range
                                       })

        self.builder.add_features(data_dict, "EEG_features", features, "EEG")
        self.builder.add_features(data_dict, "EEG_artifacts", artifacts, "EEG")

        return data_dict

    def _apply_filters(self, data: np.ndarray, fs: float) -> np.ndarray:
        """应用滤波器"""
        from scipy.signal import butter, filtfilt, iirnotch

        # 陷波滤波器（去除工频干扰）
        if self.notch_freq > 0:
            w0 = self.notch_freq / (fs / 2)
            if 0 < w0 < 1:
                b, a = iirnotch(w0, 30)
                data = filtfilt(b, a, data, axis=1)

        # 带通滤波器
        nyq = fs / 2
        low = self.bandpass_range[0] / nyq
        high = self.bandpass_range[1] / nyq
        if 0 < low < 1 and 0 < high < 1:
            b, a = butter(4, [low, high], btype='band')
            data = filtfilt(b, a, data, axis=1)

        return data

    def extract_features(self, data: np.ndarray, fs: float) -> Dict[str, Any]:
        """提取EEG特征"""
        features = {}

        # 频带功率
        band_definitions = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45)
        }

        from scipy.signal import welch
        for band_name, (low_freq, high_freq) in band_definitions.items():
            band_powers = []
            for ch_idx in range(data.shape[0]):
                freqs, psd = welch(data[ch_idx], fs=fs, nperseg=min(1024, data.shape[1]))
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.sum(psd[band_mask])
                band_powers.append(band_power)

            features[f"{band_name}_power"] = band_powers
            features[f"{band_name}_power_mean"] = float(np.mean(band_powers))

        # 统计特征
        features["mean_amplitude"] = np.mean(data, axis=1).tolist()
        features["std_amplitude"] = np.std(data, axis=1).tolist()
        features["variance"] = np.var(data, axis=1).tolist()

        return features

    def detect_artifacts(self, data, fs):
        """
        检测EEG伪迹

        参数:
            data: EEG数据 (channels x samples)
            fs: 采样率
        """
        artifacts = {}

        # 这里有问题：方法参数中没有data_dict，但代码中使用了data_dict
        # 应该使用self.current_data_dict或者从其他地方获取

        try:
            # 原来的错误代码：
            # frontal_channels = [i for i, name in enumerate(data_dict.get("signal", {}).get("EEG", {}).get("channel_names", []))

            # 修复方案1：如果确实需要channel_names，可以从其他地方获取
            # 但detect_artifacts方法只接收data和fs，没有channel_names

            # 简单的修复：先不进行通道特定检测
            n_channels = data.shape[0]

            # 基本伪迹检测（不依赖通道名）
            for i in range(n_channels):
                channel_data = data[i]

                # 检测振幅异常
                threshold = np.median(np.abs(channel_data)) * 5
                large_amp_idx = np.where(np.abs(channel_data) > threshold)[0]

                if len(large_amp_idx) > 0:
                    if f'ch{i}' not in artifacts:
                        artifacts[f'ch{i}'] = []
                    artifacts[f'ch{i}'].append({
                        'type': 'amplitude',
                        'indices': large_amp_idx.tolist(),
                        'percentage': len(large_amp_idx) / len(channel_data) * 100
                    })

            return artifacts

        except Exception as e:
            print(f"EEG伪迹检测错误: {e}")
            return {}

# ==================== EMG信号处理器 ====================
class EMGProcessor(SignalProcessor):
    """EMG信号处理器"""

    def __init__(self, envelope_cutoff: float = 5.0, rms_window: float = 0.1):
        super().__init__()
        self.envelope_cutoff = envelope_cutoff
        self.rms_window = rms_window

    def process(self, data_dict: Dict) -> Dict:
        """处理EMG信号"""
        if "EMG" not in data_dict.get("signal", {}):
            return data_dict

        emg_info = data_dict["signal"]["EMG"]
        data = emg_info["data"]
        fs = emg_info["sampling_rate"]

        # 1. 带通滤波 (20-500 Hz)
        filtered_data = self._bandpass_filter(data, fs, lowcut=20, highcut=500)

        # 2. 整流
        rectified = np.abs(filtered_data)

        # 3. 包络提取
        envelope = self._extract_envelope(rectified, fs)

        # 4. RMS计算
        rms = self._calculate_rms(filtered_data, fs)

        # 5. 提取特征
        features = self.extract_features(filtered_data, fs)

        # 保存处理结果
        self.builder.add_processed_data(data_dict, "EMG_filtered", filtered_data,
                                       processing_steps=["bandpass_filter"])

        self.builder.add_processed_data(data_dict, "EMG_rectified", rectified,
                                       processing_steps=["rectification"])

        self.builder.add_processed_data(data_dict, "EMG_envelope", envelope,
                                       processing_steps=["envelope_extraction"],
                                       parameters={"cutoff_freq": self.envelope_cutoff})

        self.builder.add_processed_data(data_dict, "EMG_RMS", rms,
                                       processing_steps=["rms_calculation"],
                                       parameters={"window_size": self.rms_window})

        self.builder.add_features(data_dict, "EMG_features", features, "EMG")

        return data_dict

    def _bandpass_filter(self, data: np.ndarray, fs: float,
                        lowcut: float, highcut: float) -> np.ndarray:
        """带通滤波器"""
        from scipy.signal import butter, filtfilt

        nyq = fs / 2
        low = lowcut / nyq
        high = highcut / nyq

        if low < 1 and high < 1:
            b, a = butter(4, [low, high], btype='band')
            return filtfilt(b, a, data, axis=1)
        return data

    def _extract_envelope(self, data: np.ndarray, fs: float) -> np.ndarray:
        """提取包络线"""
        from scipy.signal import butter, filtfilt

        nyq = fs / 2
        cutoff = self.envelope_cutoff / nyq

        if cutoff < 1:
            b, a = butter(4, cutoff, btype='low')
            return filtfilt(b, a, data, axis=1)
        return data

    def _calculate_rms(self, data: np.ndarray, fs: float) -> np.ndarray:
        """计算RMS"""
        window_size = int(self.rms_window * fs)
        if window_size < 1:
            window_size = 1

        rms_signal = np.zeros_like(data)
        for ch in range(data.shape[0]):
            squared = data[ch] ** 2
            window = np.ones(window_size) / window_size
            rms_signal[ch] = np.sqrt(np.convolve(squared, window, mode='same'))

        return rms_signal

    def extract_features(self, data: np.ndarray, fs: float) -> Dict[str, Any]:
        """提取EMG特征"""
        features = {}

        # 幅度特征
        features["mean_amplitude"] = float(np.mean(np.abs(data)))
        features["max_amplitude"] = float(np.max(np.abs(data)))
        features["std_amplitude"] = float(np.std(data))

        # 过零率
        zero_crossings = np.sum(np.diff(np.signbit(data)) != 0, axis=1)
        features["zero_crossing_rate"] = (zero_crossings / data.shape[1] * fs).tolist()

        # 中值频率
        from scipy.signal import welch
        for ch_idx in range(min(3, data.shape[0])):  # 只计算前3个通道
            freqs, psd = welch(data[ch_idx], fs=fs)
            cumulative_sum = np.cumsum(psd)
            median_freq_idx = np.argmax(cumulative_sum >= cumulative_sum[-1] / 2)
            features[f"median_freq_ch{ch_idx+1}"] = float(freqs[median_freq_idx])

        # 肌肉激活检测
        envelope = self._extract_envelope(np.abs(data), fs)
        activation_threshold = np.mean(envelope) + 2 * np.std(envelope)
        activation_mask = envelope > activation_threshold
        features["activation_percentage"] = float(np.mean(activation_mask) * 100)

        return features

# ==================== ECG信号处理器 ====================
class ECGProcessor(SignalProcessor):
    """ECG信号处理器"""

    def __init__(self, qrs_window: float = 0.15, hr_min: float = 40, hr_max: float = 180):
        super().__init__()
        self.qrs_window = qrs_window
        self.hr_min = hr_min
        self.hr_max = hr_max

    def process(self, data_dict: Dict) -> Dict:
        """处理ECG信号"""
        if "ECG" not in data_dict.get("signal", {}):
            return data_dict

        ecg_info = data_dict["signal"]["ECG"]
        data = ecg_info["data"]
        fs = ecg_info["sampling_rate"]

        # 选择最佳通道
        if data.shape[0] > 1:
            channel_powers = np.std(data, axis=1)
            best_channel = np.argmax(channel_powers)
            ecg_signal = data[best_channel]
        else:
            ecg_signal = data[0]

        # 1. 滤波
        filtered_signal = self._filter_ecg(ecg_signal, fs)

        # 2. R峰检测
        r_peaks = self._detect_r_peaks(filtered_signal, fs)

        # 3. 计算心率变异性
        hrv_features = self._calculate_hrv(r_peaks, fs)

        # 4. 提取特征
        features = self.extract_features(filtered_signal, fs, r_peaks)

        # 保存处理结果
        self.builder.add_processed_data(data_dict, "ECG_filtered", filtered_signal.reshape(1, -1),
                                       processing_steps=["bandpass_filter"])

        self.builder.add_features(data_dict, "ECG_features", features, "ECG")
        self.builder.add_features(data_dict, "ECG_HRV", hrv_features, "ECG")

        data_dict.setdefault("processed", {}).setdefault("artifacts", {})["ECG_R_peaks"] = r_peaks.tolist()

        return data_dict

    def _filter_ecg(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """ECG滤波"""
        from scipy.signal import butter, filtfilt

        # 带通滤波 0.5-40 Hz
        nyq = fs / 2
        low = 0.5 / nyq
        high = 40 / nyq

        if low < 1 and high < 1:
            b, a = butter(4, [low, high], btype='band')
            return filtfilt(b, a, signal)
        return signal

    def _detect_r_peaks(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """检测R峰"""
        from scipy.signal import find_peaks

        # 使用Pan-Tompkins算法简化版
        diff = np.diff(signal)
        squared = diff ** 2
        window = int(self.qrs_window * fs)
        integrated = np.convolve(squared, np.ones(window)/window, mode='same')

        # 找峰值
        min_distance = int(0.6 * fs)  # 最小RR间期
        height_threshold = np.mean(integrated) + 2 * np.std(integrated)

        peaks, _ = find_peaks(integrated, distance=min_distance, height=height_threshold)

        return peaks

    def _calculate_hrv(self, r_peaks: np.ndarray, fs: float) -> Dict[str, Any]:
        """计算心率变异性"""
        if len(r_peaks) < 2:
            return {}

        # RR间期（秒）
        rr_intervals = np.diff(r_peaks) / fs

        # 心率（BPM）
        heart_rate = 60 / rr_intervals

        # 去除异常值
        valid_mask = (heart_rate >= self.hr_min) & (heart_rate <= self.hr_max)
        rr_intervals_clean = rr_intervals[valid_mask]

        if len(rr_intervals_clean) < 2:
            return {}

        hrv_features = {
            "mean_hr": float(np.mean(heart_rate[valid_mask])),
            "std_hr": float(np.std(heart_rate[valid_mask])),
            "mean_rr": float(np.mean(rr_intervals_clean) * 1000),  # 转为毫秒
            "std_rr": float(np.std(rr_intervals_clean) * 1000),
            "rmssd": float(np.sqrt(np.mean(np.diff(rr_intervals_clean)**2)) * 1000),
            "nn50": int(np.sum(np.abs(np.diff(rr_intervals_clean)) > 0.05)),
            "pnn50": float(np.sum(np.abs(np.diff(rr_intervals_clean)) > 0.05) / len(rr_intervals_clean) * 100)
        }

        return hrv_features

    def extract_features(self, signal: np.ndarray, fs: float, r_peaks: np.ndarray) -> Dict[str, Any]:
        """提取ECG特征"""
        features = {
            "mean_amplitude": float(np.mean(np.abs(signal))),
            "max_amplitude": float(np.max(np.abs(signal))),
            "r_peak_count": len(r_peaks),
            "signal_noise_ratio": float(np.std(signal) / np.std(signal[:min(1000, len(signal))]))
        }

        return features

# ==================== GSR信号处理器 ====================
class GSRProcessor(SignalProcessor):
    """皮肤电信号处理器"""

    def __init__(self, tonic_cutoff: float = 0.05, scr_threshold: float = 0.05):
        super().__init__()
        self.tonic_cutoff = tonic_cutoff
        self.scr_threshold = scr_threshold

    def process(self, data_dict: Dict) -> Dict:
        """处理GSR信号"""
        if "GSR" not in data_dict.get("signal", {}):
            return data_dict

        gsr_info = data_dict["signal"]["GSR"]
        data = gsr_info["data"]
        fs = gsr_info["sampling_rate"]

        gsr_signal = data[0] if data.ndim > 1 else data

        # 1. 分解为tonic和phasic成分
        tonic, phasic = self._decompose_gsr(gsr_signal, fs)

        # 2. 检测SCR事件
        scr_events = self._detect_scr(phasic, fs)

        # 3. 提取特征
        features = self.extract_features(gsr_signal, tonic, phasic, scr_events, fs)

        # 保存处理结果
        self.builder.add_processed_data(data_dict, "GSR_tonic", tonic.reshape(1, -1),
                                       processing_steps=["tonic_component_extraction"])

        self.builder.add_processed_data(data_dict, "GSR_phasic", phasic.reshape(1, -1),
                                       processing_steps=["phasic_component_extraction"])

        self.builder.add_features(data_dict, "GSR_features", features, "GSG")
        self.builder.add_features(data_dict, "GSR_SCR_events", scr_events, "GSR")

        return data_dict

    def _decompose_gsr(self, signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """分解GSR为tonic和phasic成分"""
        from scipy.signal import savgol_filter

        # tonic成分：低频部分
        window_length = min(int(10 * fs), len(signal))
        if window_length % 2 == 0:
            window_length += 1

        if window_length > 3:
            tonic = savgol_filter(signal, window_length=window_length, polyorder=3)
        else:
            tonic = signal

        # phasic成分：剩余部分
        phasic = signal - tonic

        return tonic, phasic

    def _detect_scr(self, phasic: np.ndarray, fs: float) -> List[Dict[str, Any]]:
        """检测皮肤电导反应(SCR)"""
        from scipy.signal import find_peaks

        scr_events = []

        peaks, properties = find_peaks(phasic,
                                      height=self.scr_threshold,
                                      distance=int(1 * fs))  # 最少1秒间隔

        for i, peak in enumerate(peaks):
            amplitude = properties['peak_heights'][i]

            # 找到起始点
            start_idx = peak
            while start_idx > 0 and phasic[start_idx] > phasic[start_idx - 1]:
                start_idx -= 1

            # 找到结束点
            end_idx = peak
            while end_idx < len(phasic) - 1 and phasic[end_idx] > phasic[end_idx + 1]:
                end_idx += 1

            scr_events.append({
                "onset": float(start_idx / fs),
                "peak": float(peak / fs),
                "offset": float(end_idx / fs),
                "amplitude": float(amplitude),
                "rise_time": float((peak - start_idx) / fs),
                "recovery_time": float((end_idx - peak) / fs) if end_idx > peak else 0
            })

        return scr_events

    def extract_features(self, signal: np.ndarray, tonic: np.ndarray,
                        phasic: np.ndarray, scr_events: List[Dict], fs: float) -> Dict[str, Any]:
        """提取GSR特征"""
        features = {
            "mean_tonic": float(np.mean(tonic)),
            "std_tonic": float(np.std(tonic)),
            "mean_phasic": float(np.mean(np.abs(phasic))),
            "scr_count": len(scr_events),
            "scr_rate": len(scr_events) / (len(signal) / fs) * 60 if len(signal) > 0 else 0,  # 每分钟SCR次数
            "mean_scr_amplitude": float(np.mean([e["amplitude"] for e in scr_events])) if scr_events else 0,
            "mean_scr_rise_time": float(np.mean([e["rise_time"] for e in scr_events])) if scr_events else 0
        }

        return features

# ==================== fNIRS信号处理器 ====================
class fNIRSProcessor(SignalProcessor):
    """fNIRS信号处理器"""

    def __init__(self, wavelengths: List[float] = None,
                 source_detector_distance: float = 30.0):
        super().__init__()
        self.wavelengths = wavelengths or [760, 850]
        self.source_detector_distance = source_detector_distance

    def process(self, data_dict: Dict) -> Dict:
        """处理fNIRS信号"""
        if "FNIRS" not in data_dict.get("signal", {}):
            return data_dict

        fnirs_info = data_dict["signal"]["FNIRS"]
        data = fnirs_info["data"]
        fs = fnirs_info["sampling_rate"]
        channel_names = fnirs_info["channel_names"]

        # 1. 分离不同波长的数据
        wavelength_data = self._separate_wavelengths(data, channel_names)

        # 2. 转换为光密度
        optical_density = self._convert_to_optical_density(wavelength_data)

        # 3. 计算血红蛋白浓度
        hemoglobin_concentration = self._calculate_hemoglobin_concentration(optical_density)

        # 4. 提取特征
        features = self.extract_features(hemoglobin_concentration, fs)

        # 保存处理结果
        self.builder.add_processed_data(data_dict, "FNIRS_optical_density", optical_density,
                                       processing_steps=["optical_density_conversion"])

        self.builder.add_processed_data(data_dict, "FNIRS_hemoglobin", hemoglobin_concentration,
                                       processing_steps=["hemoglobin_calculation"])

        self.builder.add_features(data_dict, "FNIRS_features", features, "FNIRS")

        return data_dict

    def _separate_wavelengths(self, data: np.ndarray, channel_names: List[str]) -> Dict[float, np.ndarray]:
        """按波长分离数据"""
        wavelength_data = {}

        for i, channel_name in enumerate(channel_names):
            # 从通道名中提取波长信息
            for wavelength in self.wavelengths:
                if str(wavelength) in channel_name or f"WL{wavelength}" in channel_name:
                    if wavelength not in wavelength_data:
                        wavelength_data[wavelength] = []
                    wavelength_data[wavelength].append(data[i])
                    break

        # 转换为数组
        for wavelength in wavelength_data:
            wavelength_data[wavelength] = np.array(wavelength_data[wavelength])

        return wavelength_data

    def _convert_to_optical_density(self, wavelength_data: Dict[float, np.ndarray]) -> Dict[float, np.ndarray]:
        """转换为光密度"""
        optical_density = {}

        for wavelength, data in wavelength_data.items():
            # 假设输入是原始强度值
            # OD = -log(I/I0)，这里I0用均值近似
            I0 = np.mean(data, axis=1, keepdims=True)
            od = -np.log(data / I0)
            optical_density[wavelength] = od

        return optical_density

    def _calculate_hemoglobin_concentration(self, optical_density: Dict[float, np.ndarray]) -> Dict[str, np.ndarray]:
        """计算血红蛋白浓度"""
        # 简化的MBLL（修正的比尔-朗伯定律）实现
        # 需要消光系数矩阵，这里使用示例值
        extinction_coefficients = {
            760: {"HbO": 0.38, "HbR": 1.06},  # 示例值，单位：cm^-1 * mM^-1
            850: {"HbO": 0.87, "HbR": 0.69}
        }

        # 确保有足够的波长
        available_wavelengths = list(optical_density.keys())
        if len(available_wavelengths) < 2:
            logger.warning("需要至少2个波长来计算血红蛋白浓度")
            return {}

        # 选择前两个波长
        wl1 = available_wavelengths[0]
        wl2 = available_wavelengths[1]

        # 提取消光系数
        epsilon = np.array([
            [extinction_coefficients.get(wl1, {}).get("HbO", 0.38), extinction_coefficients.get(wl1, {}).get("HbR", 1.06)],
            [extinction_coefficients.get(wl2, {}).get("HbO", 0.87), extinction_coefficients.get(wl2, {}).get("HbR", 0.69)]
        ])

        # 路径长度因子（DPF * 距离）
        dpf = 6.0  # 微分路径长度因子，示例值
        pathlength = dpf * self.source_detector_distance / 10  # 转为cm

        # 计算浓度
        od1 = optical_density[wl1]
        od2 = optical_density[wl2]

        hemoglobin_concentration = {}

        for ch in range(od1.shape[0]):
            # 解线性方程组
            A = epsilon * pathlength
            b = np.array([od1[ch], od2[ch]])

            # 使用最小二乘法
            try:
                concentration = np.linalg.lstsq(A, b, rcond=None)[0]
                if ch == 0:
                    hemoglobin_concentration["HbO"] = [concentration[0]]
                    hemoglobin_concentration["HbR"] = [concentration[1]]
                else:
                    hemoglobin_concentration["HbO"].append(concentration[0])
                    hemoglobin_concentration["HbR"].append(concentration[1])
            except:
                if ch == 0:
                    hemoglobin_concentration["HbO"] = [0]
                    hemoglobin_concentration["HbR"] = [0]
                else:
                    hemoglobin_concentration["HbO"].append(0)
                    hemoglobin_concentration["HbR"].append(0)

        # 转换为数组
        hemoglobin_concentration["HbO"] = np.array(hemoglobin_concentration["HbO"])
        hemoglobin_concentration["HbR"] = np.array(hemoglobin_concentration["HbR"])
        hemoglobin_concentration["HbT"] = hemoglobin_concentration["HbO"] + hemoglobin_concentration["HbR"]

        return hemoglobin_concentration

    def extract_features(self, hemoglobin_concentration: Dict[str, np.ndarray], fs: float) -> Dict[str, Any]:
        """提取fNIRS特征"""
        features = {}

        if not hemoglobin_concentration:
            return features

        for hb_type, data in hemoglobin_concentration.items():
            if len(data.shape) == 1:
                data = data.reshape(1, -1)

            features[f"{hb_type}_mean"] = np.mean(data, axis=1).tolist()
            features[f"{hb_type}_std"] = np.std(data, axis=1).tolist()

            # 计算激活（相对于基线的变化）
            if data.shape[1] > int(5 * fs):  # 至少有5秒数据
                baseline = np.mean(data[:, :int(5 * fs)], axis=1, keepdims=True)
                activation = np.mean(data[:, int(5 * fs):], axis=1) - baseline.flatten()
                features[f"{hb_type}_activation"] = activation.tolist()

        return features

# ==================== 眼动信号处理器 ====================
class EyeTrackerProcessor(SignalProcessor):
    """眼动信号处理器"""

    def __init__(self, screen_resolution: Tuple[int, int] = (1920, 1080),
                 fixation_threshold: float = 10.0,  # 像素/秒
                 saccade_threshold: float = 30.0):  # 像素/秒
        super().__init__()
        self.screen_resolution = screen_resolution
        self.fixation_threshold = fixation_threshold
        self.saccade_threshold = saccade_threshold

    def process(self, data_dict: Dict) -> Dict:
        """处理眼动信号"""
        if "ET" not in data_dict.get("signal", {}):
            return data_dict

        et_info = data_dict["signal"]["ET"]
        data = et_info["data"]
        fs = et_info["sampling_rate"]
        channel_names = et_info["channel_names"]

        # 1. 检测眼动事件
        events = self._detect_eye_events(data, fs, channel_names)

        # 2. 提取特征
        features = self.extract_features(data, events, fs)

        # 3. 检测伪迹
        artifacts = self.detect_artifacts(data, fs)

        # 保存处理结果
        self.builder.add_features(data_dict, "ET_events", events, "ET")
        self.builder.add_features(data_dict, "ET_features", features, "ET")
        self.builder.add_features(data_dict, "ET_artifacts", artifacts, "ET")

        return data_dict

    def _detect_eye_events(self, data: np.ndarray, fs: float,
                          channel_names: List[str]) -> Dict[str, List[Dict]]:
        """检测眼动事件"""
        events = {
            "fixations": [],
            "saccades": [],
            "blinks": []
        }

        # 寻找X和Y坐标通道
        x_idx, y_idx, pupil_idx = -1, -1, -1
        for i, name in enumerate(channel_names):
            name_lower = name.lower()
            if 'x' in name_lower or 'gaze_x' in name_lower:
                x_idx = i
            elif 'y' in name_lower or 'gaze_y' in name_lower:
                y_idx = i
            elif 'pupil' in name_lower:
                pupil_idx = i

        if x_idx == -1 or y_idx == -1:
            logger.warning("未找到X或Y坐标通道")
            return events

        # 提取坐标
        x = data[x_idx]
        y = data[y_idx]

        # 计算速度
        dx = np.diff(x)
        dy = np.diff(y)
        velocity = np.sqrt(dx**2 + dy**2) * fs  # 像素/秒

        # 检测扫视（高速运动）
        saccade_mask = velocity > self.saccade_threshold

        # 检测注视（低速运动）
        fixation_mask = velocity < self.fixation_threshold

        # 检测眨眼（瞳孔数据缺失或为0）
        blink_mask = np.zeros(len(x), dtype=bool)
        if pupil_idx != -1:
            pupil = data[pupil_idx]
            blink_mask = (pupil == 0) | np.isnan(pupil)

        # 将连续的区域转换为事件
        events["fixations"] = self._clusters_to_events(fixation_mask, fs, "fixation")
        events["saccades"] = self._clusters_to_events(saccade_mask, fs, "saccade")
        events["blinks"] = self._clusters_to_events(blink_mask, fs, "blink")

        # 计算扫视幅度
        for saccade in events["saccades"]:
            start_idx = int(saccade["start"] * fs)
            end_idx = int(saccade["end"] * fs)
            if start_idx < len(x) and end_idx < len(x):
                dx_event = x[end_idx] - x[start_idx]
                dy_event = y[end_idx] - y[start_idx]
                saccade["amplitude"] = float(np.sqrt(dx_event**2 + dy_event**2))
                saccade["angle"] = float(np.arctan2(dy_event, dx_event) * 180 / np.pi)

        # 计算注视位置
        for fixation in events["fixations"]:
            start_idx = int(fixation["start"] * fs)
            end_idx = int(fixation["end"] * fs)
            if start_idx < len(x) and end_idx < len(x):
                fixation["x_mean"] = float(np.mean(x[start_idx:end_idx]))
                fixation["y_mean"] = float(np.mean(y[start_idx:end_idx]))
                fixation["x_std"] = float(np.std(x[start_idx:end_idx]))
                fixation["y_std"] = float(np.std(y[start_idx:end_idx]))

        return events

    def _clusters_to_events(self, mask: np.ndarray, fs: float,
                           event_type: str) -> List[Dict[str, Any]]:
        """将连续的mask区域转换为事件"""
        events = []

        # 找到区域变化点
        diff_mask = np.diff(mask.astype(int))
        start_indices = np.where(diff_mask == 1)[0] + 1
        end_indices = np.where(diff_mask == -1)[0]

        # 处理起始和结束
        if mask[0]:
            start_indices = np.insert(start_indices, 0, 0)
        if mask[-1]:
            end_indices = np.append(end_indices, len(mask) - 1)

        # 创建事件
        for start_idx, end_idx in zip(start_indices, end_indices):
            if end_idx - start_idx >= int(0.05 * fs):  # 最少50ms
                events.append({
                    "type": event_type,
                    "start": float(start_idx / fs),
                    "end": float(end_idx / fs),
                    "duration": float((end_idx - start_idx) / fs),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx)
                })

        return events

    def extract_features(self, data: np.ndarray, events: Dict[str, List[Dict]],
                        fs: float) -> Dict[str, Any]:
        """提取眼动特征"""
        features = {}

        # 基本统计
        features["sample_count"] = data.shape[1]
        features["recording_duration"] = data.shape[1] / fs if fs > 0 else 0

        # 事件计数
        for event_type in ["fixations", "saccades", "blinks"]:
            features[f"{event_type}_count"] = len(events.get(event_type, []))

        # 注视特征
        fixations = events.get("fixations", [])
        if fixations:
            features["fixation_duration_mean"] = float(np.mean([f["duration"] for f in fixations]))
            features["fixation_duration_std"] = float(np.std([f["duration"] for f in fixations]))
            features["fixation_rate"] = len(fixations) / features["recording_duration"] if features["recording_duration"] > 0 else 0

        # 扫视特征
        saccades = events.get("saccades", [])
        if saccades:
            features["saccade_duration_mean"] = float(np.mean([s["duration"] for s in saccades]))
            features["saccade_amplitude_mean"] = float(np.mean([s.get("amplitude", 0) for s in saccades]))
            features["saccade_velocity_mean"] = float(np.mean([s.get("amplitude", 0) / s["duration"] if s["duration"] > 0 else 0 for s in saccades]))

        # 眨眼特征
        blinks = events.get("blinks", [])
        if blinks:
            features["blink_duration_mean"] = float(np.mean([b["duration"] for b in blinks]))
            features["blink_rate"] = len(blinks) / features["recording_duration"] * 60 if features["recording_duration"] > 0 else 0

        return features

    def detect_artifacts(self, data: np.ndarray, fs: float) -> Dict[str, Any]:
        """检测眼动伪迹"""
        artifacts = {
            "data_loss": [],
            "out_of_bounds": []
        }

        # 检查数据丢失（NaN或0值）
        nan_mask = np.any(np.isnan(data), axis=0)
        zero_mask = np.any(data == 0, axis=0)
        data_loss_mask = nan_mask | zero_mask

        if np.any(data_loss_mask):
            loss_events = self._clusters_to_events(data_loss_mask, fs, "data_loss")
            artifacts["data_loss"] = loss_events

        return artifacts

# ==================== 呼吸信号处理器 ====================
class RespiratoryProcessor(SignalProcessor):
    """呼吸信号处理器"""

    def __init__(self, resp_band: Tuple[float, float] = (0.1, 2.0)):
        super().__init__()
        self.resp_band = resp_band

    def process(self, data_dict: Dict) -> Dict:
        """处理呼吸信号"""
        if "RESP" not in data_dict.get("signal", {}):
            return data_dict

        resp_info = data_dict["signal"]["RESP"]
        data = resp_info["data"]
        fs = resp_info["sampling_rate"]

        resp_signal = data[0] if data.ndim > 1 else data

        # 1. 滤波
        filtered_signal = self._filter_respiratory(resp_signal, fs)

        # 2. 检测呼吸周期
        peaks, valleys = self._detect_respiratory_cycles(filtered_signal, fs)

        # 3. 计算呼吸特征
        features = self.extract_features(filtered_signal, peaks, valleys, fs)

        # 保存处理结果
        self.builder.add_processed_data(data_dict, "RESP_filtered", filtered_signal.reshape(1, -1),
                                       processing_steps=["bandpass_filter"])

        self.builder.add_features(data_dict, "RESP_features", features, "RESP")

        # 保存检测到的峰谷
        data_dict.setdefault("processed", {}).setdefault("artifacts", {})["RESP_peaks"] = peaks.tolist()
        data_dict.setdefault("processed", {}).setdefault("artifacts", {})["RESP_valleys"] = valleys.tolist()

        return data_dict

    def _filter_respiratory(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """呼吸信号滤波"""
        from scipy.signal import butter, filtfilt

        nyq = fs / 2
        low = self.resp_band[0] / nyq
        high = self.resp_band[1] / nyq

        if low < 1 and high < 1:
            b, a = butter(4, [low, high], btype='band')
            return filtfilt(b, a, signal)
        return signal

    def _detect_respiratory_cycles(self, signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """检测呼吸周期"""
        from scipy.signal import find_peaks

        # 检测吸气峰
        peaks, _ = find_peaks(signal,
                             distance=int(1 * fs),  # 最少1秒间隔
                             prominence=np.std(signal) * 0.5)

        # 检测呼气谷
        valleys, _ = find_peaks(-signal,
                               distance=int(1 * fs),
                               prominence=np.std(signal) * 0.5)

        return peaks, valleys

    def extract_features(self, signal: np.ndarray, peaks: np.ndarray,
                        valleys: np.ndarray, fs: float) -> Dict[str, Any]:
        """提取呼吸特征"""
        features = {}

        # 呼吸率
        if len(peaks) > 1:
            breath_durations = np.diff(peaks) / fs
            respiration_rate = 60 / np.mean(breath_durations)  # 次/分钟

            features["respiration_rate_bpm"] = float(respiration_rate)
            features["breath_duration_mean"] = float(np.mean(breath_durations))
            features["breath_duration_std"] = float(np.std(breath_durations))
            features["breath_count"] = len(peaks)

        # 幅度特征
        if len(peaks) > 0 and len(valleys) > 0:
            # 对齐峰谷
            n_cycles = min(len(peaks), len(valleys))
            tidal_volumes = []

            for i in range(n_cycles):
                peak_val = signal[peaks[i]]
                valley_val = signal[valleys[i]] if i < len(valleys) else signal[0]
                tidal_volumes.append(peak_val - valley_val)

            if tidal_volumes:
                features["tidal_volume_mean"] = float(np.mean(tidal_volumes))
                features["tidal_volume_std"] = float(np.std(tidal_volumes))

        # 信号质量指标
        features["signal_mean"] = float(np.mean(signal))
        features["signal_std"] = float(np.std(signal))
        features["snr"] = float(np.std(signal) / (np.std(signal[:min(1000, len(signal))]) + 1e-10))

        # 呼吸规律性（变异系数）
        if len(peaks) > 2:
            rr_intervals = np.diff(peaks) / fs
            cv = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
            features["respiratory_variability"] = float(cv)

        return features

# ==================== 万能生物信号处理器 ====================
class UniversalBioSignalProcessor:
    """
    万能生物信号处理器
    集成所有信号类型的处理功能
    """

    def __init__(self, auto_process: bool = True):
        self.builder = DataDictBuilder()
        self.auto_process = auto_process

        # 信号处理器映射
        self.processors = {
            SignalType.EEG: EEGProcessor(),
            SignalType.EMG: EMGProcessor(),
            SignalType.ECG: ECGProcessor(),
            SignalType.GSR: GSRProcessor(),
            SignalType.ET: EyeTrackerProcessor(),
            SignalType.RESP: RespiratoryProcessor(),
            SignalType.FNIRS: fNIRSProcessor(),
            SignalType.EOG: EEGProcessor(),  # EOG可以使用EEG处理器
            SignalType.PPG: ECGProcessor(),  # PPG可以使用ECG处理器
        }

    def process(self, data_dict: Dict, signal_type: Union[str, SignalType] = None) -> Dict:
        """
        处理数据字典中的信号

        参数:
        - data_dict: 输入数据字典
        - signal_type: 信号类型（可选，自动检测）

        返回:
        - 处理后的数据字典
        """
        if not data_dict or "signal" not in data_dict:
            logger.warning("数据字典中没有信号数据")
            return data_dict

        # 1. 确定要处理的信号类型
        if signal_type is None:
            # 自动检测信号类型
            for modality, signal_info in data_dict["signal"].items():
                signal_type_str = signal_info.get("signal_type", "").lower()
                try:
                    sig_type = SignalType(signal_type_str)
                    signal_type = sig_type
                    break
                except ValueError:
                    # 尝试从modality推断
                    modality_lower = modality.lower()
                    for sig_enum in SignalType:
                        if sig_enum.value in modality_lower:
                            signal_type = sig_enum
                            break

        if isinstance(signal_type, str):
            try:
                signal_type = SignalType(signal_type.lower())
            except ValueError:
                logger.warning(f"未知信号类型: {signal_type}")
                return data_dict

        # 2. 调用对应的处理器
        if signal_type in self.processors:
            processor = self.processors[signal_type]
            logger.info(f"使用{signal_type.value.upper()}处理器")
            return processor.process(data_dict)
        else:
            logger.warning(f"没有找到{signal_type.value}的处理器")
            return data_dict

    def process_all(self, data_dict: Dict) -> Dict:
        """处理所有信号"""
        if not data_dict or "signal" not in data_dict:
            return data_dict

        for modality, signal_info in data_dict["signal"].items():
            signal_type_str = signal_info.get("signal_type", "").lower()

            try:
                signal_type = SignalType(signal_type_str)
            except ValueError:
                # 尝试从modality推断
                modality_lower = modality.lower()
                for sig_enum in SignalType:
                    if sig_enum.value in modality_lower:
                        signal_type = sig_enum
                        break
                else:
                    continue

            if signal_type in self.processors:
                logger.info(f"处理{modality}信号 ({signal_type.value})")
                processor = self.processors[signal_type]
                data_dict = processor.process(data_dict)

        return data_dict

    def create_pipeline(self, processing_steps: List[Dict]) -> callable:
        """
        创建处理流水线

        参数:
        - processing_steps: 处理步骤列表，每个步骤是一个字典，包含:
            - "type": 处理器类型
            - "parameters": 处理器参数

        返回:
        - 处理函数
        """
        def pipeline(data_dict):
            for step in processing_steps:
                step_type = step.get("type")
                parameters = step.get("parameters", {})

                if step_type == "eeg":
                    processor = EEGProcessor(**parameters)
                elif step_type == "emg":
                    processor = EMGProcessor(**parameters)
                elif step_type == "ecg":
                    processor = ECGProcessor(**parameters)
                elif step_type == "gsr":
                    processor = GSRProcessor(**parameters)
                elif step_type == "fnirs":
                    processor = fNIRSProcessor(**parameters)
                elif step_type == "eyetracker":
                    processor = EyeTrackerProcessor(**parameters)
                elif step_type == "respiratory":
                    processor = RespiratoryProcessor(**parameters)
                else:
                    logger.warning(f"未知处理器类型: {step_type}")
                    continue

                data_dict = processor.process(data_dict)

            return data_dict

        return pipeline

# ==================== 工具函数 ====================
def save_data_dict(data_dict: Dict, file_path: str, format: str = "npz") -> None:
    """
    保存数据字典到文件

    参数:
    - data_dict: 数据字典
    - file_path: 文件路径
    - format: 文件格式，支持"npz"、"json"、"h5"
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "npz":
        # 转换为可保存的格式
        save_dict = {}
        for key, value in data_dict.items():
            if key == "signal":
                for modality, sig_info in value.items():
                    # 保存数据和元数据分开
                    save_dict[f"signal_{modality}_data"] = sig_info.get("data", np.array([]))
                    save_dict[f"signal_{modality}_meta"] = {k: v for k, v in sig_info.items() if k != "data"}
            else:
                save_dict[key] = value

        np.savez_compressed(file_path, **save_dict)

    elif format == "json":
        # 转换numpy数组为列表
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj

        json_dict = convert_for_json(data_dict)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=2)

    elif format == "h5":
        import h5py
        with h5py.File(file_path, 'w') as f:
            # 保存meta层
            if "meta" in data_dict:
                meta_group = f.create_group("meta")
                for key, value in data_dict["meta"].items():
                    if isinstance(value, (list, np.ndarray)):
                        meta_group.create_dataset(key, data=value)
                    else:
                        meta_group.attrs[key] = value

            # 保存signal层
            if "signal" in data_dict:
                signal_group = f.create_group("signal")
                for modality, sig_info in data_dict["signal"].items():
                    modality_group = signal_group.create_group(modality)
                    for key, value in sig_info.items():
                        if key == "data" and isinstance(value, np.ndarray):
                            modality_group.create_dataset("data", data=value)
                        elif isinstance(value, (list, np.ndarray)):
                            modality_group.create_dataset(key, data=value)
                        else:
                            modality_group.attrs[key] = value

    logger.info(f"数据字典已保存到: {file_path}")

def load_data_dict(file_path: str, format: str = None) -> Dict:
    """
    从文件加载数据字典

    参数:
    - file_path: 文件路径
    - format: 文件格式（自动检测）

    返回:
    - 数据字典
    """
    file_path = Path(file_path)

    if format is None:
        if file_path.suffix == ".npz":
            format = "npz"
        elif file_path.suffix == ".json":
            format = "json"
        elif file_path.suffix in [".h5", ".hdf5"]:
            format = "h5"
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    if format == "npz":
        data = np.load(file_path, allow_pickle=True)

        # 重建数据字典
        data_dict = {}
        signal_data = {}

        for key in data.files:
            if key.startswith("signal_") and key.endswith("_data"):
                modality = key[7:-5]  # 提取modality名
                meta_key = f"signal_{modality}_meta"
                if meta_key in data:
                    meta = data[meta_key].item()
                    meta["data"] = data[key]
                    signal_data[modality] = meta
            elif key.startswith("signal_") and key.endswith("_meta"):
                continue  # 已经在上面处理了
            else:
                data_dict[key] = data[key].item() if data[key].dtype == object else data[key]

        if signal_data:
            data_dict["signal"] = signal_data

        return data_dict

    elif format == "json":
        with open(file_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)

        # 转换列表为numpy数组
        def convert_from_json(obj):
            if isinstance(obj, dict):
                return {k: convert_from_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                # 检查是否为数值列表
                if all(isinstance(item, (int, float)) for item in obj):
                    return np.array(obj)
                else:
                    return [convert_from_json(item) for item in obj]
            else:
                return obj

        return convert_from_json(data_dict)

    elif format == "h5":
        import h5py
        with h5py.File(file_path, 'r') as f:
            data_dict = {}

            # 加载meta层
            if "meta" in f:
                meta_group = f["meta"]
                meta = {}
                for key in meta_group.attrs:
                    meta[key] = meta_group.attrs[key]
                for key in meta_group:
                    if key in meta_group:
                        meta[key] = meta_group[key][()]
                data_dict["meta"] = meta

            # 加载signal层
            if "signal" in f:
                signal_group = f["signal"]
                signal = {}
                for modality in signal_group:
                    modality_group = signal_group[modality]
                    sig_info = {}
                    for key in modality_group.attrs:
                        sig_info[key] = modality_group.attrs[key]
                    for key in modality_group:
                        if key in modality_group:
                            sig_info[key] = modality_group[key][()]
                    signal[modality] = sig_info
                data_dict["signal"] = signal

        return data_dict

# ==================== 使用示例 ====================
if __name__ == "__main__":
    print("=== 万能生物信号处理器 ===\n")

    # 创建数据字典
    builder = DataDictBuilder()
    data_dict = builder.create_empty_data_dict()

    # 示例1: 添加EEG信号
    print("1. 创建EEG信号示例:")
    fs = 250
    n_samples = 10000
    n_channels = 32
    eeg_data = np.random.randn(n_channels, n_samples) * 50  # 模拟EEG数据
    channel_names = [f"EEG_{i+1}" for i in range(n_channels)]

    builder.add_signal(data_dict, eeg_data, fs, channel_names, "EEG",
                      signal_type="eeg", unit="uV")

    print(f"   EEG信号: {n_channels}通道, {n_samples}样本, {fs}Hz采样率")

    # 示例2: 添加事件
    print("\n2. 添加事件:")
    builder.add_event(data_dict, ["Stimulus", "Response"], [1.5, 3.2], [0.1, 0.2])
    print(f"   添加了{len(data_dict['event']['event_label'])}个事件")

    # 示例3: 处理信号
    print("\n3. 处理信号:")
    processor = UniversalBioSignalProcessor()
    processed_data = processor.process(data_dict, "eeg")

    if "processed" in processed_data and "EEG_features" in processed_data["processed"]["features"]:
        features = processed_data["processed"]["features"]["EEG_features"]
        print(f"   提取了{len(features)}个EEG特征")
        print(f"   Alpha功率: {features.get('alpha_power_mean', 'N/A')}")

    # 示例4: 保存和加载
    print("\n4. 保存和加载:")
    save_data_dict(processed_data, "test_data.npz", "npz")
    loaded_data = load_data_dict("test_data.npz")
    print(f"   保存并重新加载了数据字典")

    # 示例5: 创建处理流水线
    print("\n5. 创建处理流水线:")
    pipeline_steps = [
        {"type": "eeg", "parameters": {"reference_method": "average", "notch_freq": 50.0}},
        {"type": "emg", "parameters": {"envelope_cutoff": 5.0, "rms_window": 0.1}}
    ]

    pipeline = processor.create_pipeline(pipeline_steps)
    print(f"   创建了包含{len(pipeline_steps)}个步骤的处理流水线")

    # 清理测试文件
    if Path("test_data.npz").exists():
        Path("test_data.npz").unlink()

    print("\n=== 示例完成 ===")