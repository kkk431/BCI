"""
universal_biosignal_processor.py
万能生物信号数据处理器
支持所有生物信号：EEG、fNIRS、EMG、ECG、EOG、GSR、RESP、ET等
基于标准化的四层 data_dict 格式。
数据结构参考：https://xcnmvog3p8wo.feishu.cn/wiki/NIo8wyMfqiaZSzkm8ERcV7vOnie
"""

import numpy as np
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from enum import Enum

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

        try:
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

        return tonic