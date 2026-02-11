# -*- coding: utf-8 -*-
"""
fNIRS Feature Extraction Module for Multimodal BCI
===================================================
文件名: fnirs_features.py
描述: 
    本模块为功能性近红外光谱（fNIRS）信号提供专用的特征提取方法。
    包含双信号源（HbO/HbR）分析、血流动力学响应形态特征、通道间相关性等。
    继承自 common_features.CommonFeatureExtractor，可复用通用特征。

依赖:
    numpy, scipy, PyWavelets (通过父类)

作者: Wan-Jingyu
版本: 1.0.1
"""

import numpy as np
import warnings
from common_features import CommonFeatureExtractor

class FNIRSFeatureExtractor(CommonFeatureExtractor):
    """
    fNIRS专用特征提取器。
    继承自通用特征提取器，并增加fNIRS特有的分析方法。
    """

    def __init__(self, fs: float):
        """
        初始化fNIRS特征提取器。

        Args:
            fs (float): 信号的采样频率 (Hz)。fNIRS典型值为1~20 Hz。
        """
        super().__init__(fs)

    # =========================================================================
    # 3.1 双信号源特征 (HbO, HbR, HbT, HbO-HbR)
    # =========================================================================

    def extract_hbo_hbr_features(self, hbo: np.ndarray, hbr: np.ndarray) -> dict:
        """
        提取氧合血红蛋白(HbO)和脱氧血红蛋白(HbR)的双信号源特征。
        包括HbO、HbR、总血红蛋白(HbT=HbO+HbR)以及差值(HbO-HbR)的时域统计特征。

        Args:
            hbo (np.ndarray): 1D HbO浓度变化信号。
            hbr (np.ndarray): 1D HbR浓度变化信号。

        Returns:
            dict: 包含各衍生信号的均值、标准差、最大/最小值、峰峰值、均方根等特征。
        """
        features = {}

        hbo = np.asarray(hbo, dtype=np.float64)
        hbr = np.asarray(hbr, dtype=np.float64)
        if len(hbo) != len(hbr):
            raise ValueError("HbO和HbR信号长度必须一致。")
        if len(hbo) == 0:
            warnings.warn("输入信号为空，返回空特征。")
            return {}

        # 1. HbO特征
        features.update(self._add_prefix(
            self.compute_time_domain_features(hbo), 'hbo_'))

        # 2. HbR特征
        features.update(self._add_prefix(
            self.compute_time_domain_features(hbr), 'hbr_'))

        # 3. 总血红蛋白 HbT = HbO + HbR
        hbt = hbo + hbr
        features.update(self._add_prefix(
            self.compute_time_domain_features(hbt), 'hbt_'))

        # 4. 差值 HbO-HbR
        hbo_hbr_diff = hbo - hbr
        features.update(self._add_prefix(
            self.compute_time_domain_features(hbo_hbr_diff), 'hbo_hbr_diff_'))

        return features

    # =========================================================================
    # 3.2 血流动力学响应形态特征 (Hemodynamic Response Function, HRF)
    # =========================================================================

    def extract_hrf_features(self, signal: np.ndarray, fs: float = None,
                             baseline_window: tuple = None,
                             task_window: tuple = None,
                             t: np.ndarray = None) -> dict:
        """
        提取任务诱发血流动力学响应的形态特征。
        典型fNIRS实验为“基线-任务-恢复”块设计，本函数针对单个任务块提取响应形态。

        Args:
            signal (np.ndarray): 1D HbO或HbR信号（通常使用HbO，信噪比更高）。
            fs (float, optional): 信号采样率，若未提供则使用self.fs。
            baseline_window (tuple): 基线时间段 (start_sec, end_sec)，相对时间。
            task_window (tuple): 任务时间段 (start_sec, end_sec)，相对时间。
            t (np.ndarray, optional): 时间轴（秒），长度需与signal相同。
                                      若为None，则假设信号从0开始，间隔1/fs秒。

        Returns:
            dict: 包含峰幅值、达峰时间、上升斜率、下降斜率、曲线下面积、半高全宽等。
        """
        if fs is None:
            fs = self.fs

        n = len(signal)
        if t is None:
            t = np.arange(n) / fs
        else:
            t = np.asarray(t)
            if len(t) != n:
                raise ValueError("时间轴t的长度必须与信号长度一致。")

        if baseline_window is None:
            baseline_window = (-10, 0)   # 假设任务前10秒为基线
        if task_window is None:
            task_window = (0, 20)        # 假设任务持续20秒

        baseline_mask = (t >= baseline_window[0]) & (t < baseline_window[1])
        task_mask = (t >= task_window[0]) & (t < task_window[1])

        baseline_signal = signal[baseline_mask]
        task_signal = signal[task_mask]
        task_time = t[task_mask]

        if len(baseline_signal) == 0 or len(task_signal) == 0:
            warnings.warn("基线或任务窗口内无数据点，返回空特征。")
            return {}

        baseline_mean = np.mean(baseline_signal)
        relative_task = task_signal - baseline_mean

        features = {}

        # --- 峰幅值 ---
        peak_amplitude = np.max(relative_task)
        features['hrf_peak_amplitude'] = peak_amplitude

        # --- 达到峰值时间 ---
        peak_idx = np.argmax(relative_task)
        features['hrf_time_to_peak'] = task_time[peak_idx] - task_window[0]

        # --- 响应上升斜率 ---
        if peak_idx >= 1:
            up_slope = (relative_task[peak_idx] - relative_task[0]) / (task_time[peak_idx] - task_time[0] + 1e-10)
        else:
            up_slope = 0.0
        features['hrf_upslope'] = up_slope

        # --- 响应下降斜率 ---
        if peak_idx < len(relative_task) - 1:
            down_slope = (relative_task[-1] - relative_task[peak_idx]) / (task_time[-1] - task_time[peak_idx] + 1e-10)
        else:
            down_slope = 0.0
        features['hrf_downslope'] = down_slope

        # --- 曲线下面积  ---
        auc = np.trapezoid(relative_task, task_time)   # 已使用兼容函数
        features['hrf_auc'] = auc

        # --- 半高全宽 ---
        half_max = peak_amplitude / 2.0
        rising_cross = None
        falling_cross = None
        for i in range(peak_idx):
            if relative_task[i] <= half_max <= relative_task[i+1]:
                t1, t2 = task_time[i], task_time[i+1]
                v1, v2 = relative_task[i], relative_task[i+1]
                rising_cross = t1 + (half_max - v1) * (t2 - t1) / (v2 - v1 + 1e-10)
                break
        for i in range(peak_idx, len(relative_task)-1):
            if relative_task[i] >= half_max >= relative_task[i+1]:
                t1, t2 = task_time[i], task_time[i+1]
                v1, v2 = relative_task[i], relative_task[i+1]
                falling_cross = t1 + (half_max - v1) * (t2 - t1) / (v2 - v1 + 1e-10)
                break
        if rising_cross is not None and falling_cross is not None:
            fwhm = falling_cross - rising_cross
        else:
            fwhm = np.nan
        features['hrf_fwhm'] = fwhm

        return features

    # =========================================================================
    # 3.3 通道间相关性特征
    # =========================================================================

    def compute_channel_correlation(self, data_2d: np.ndarray,
                                    ch_names: list = None,
                                    left_indices: list = None,
                                    right_indices: list = None,
                                    anterior_indices: list = None,
                                    posterior_indices: list = None) -> dict:
        """
        计算fNIRS多通道数据中的通道间相关性。
        可分别计算同侧半球、对侧半球、前后脑区等分组的平均相关系数。
        """
        features = {}
        data_2d = np.asarray(data_2d)
        n_ch, n_t = data_2d.shape
        if n_ch < 2:
            warnings.warn("通道数小于2，无法计算相关性。")
            return features

        corr_matrix = np.corrcoef(data_2d)
        np.fill_diagonal(corr_matrix, 0)

        # 全局平均相关系数
        global_mean_corr = np.sum(corr_matrix) / (n_ch * (n_ch - 1))
        features['global_mean_correlation'] = global_mean_corr

        # 同侧半球
        if left_indices is not None and len(left_indices) >= 2:
            left_corr = corr_matrix[np.ix_(left_indices, left_indices)]
            np.fill_diagonal(left_corr, 0)
            features['left_hemisphere_mean_corr'] = np.sum(left_corr) / (len(left_indices)*(len(left_indices)-1))
        if right_indices is not None and len(right_indices) >= 2:
            right_corr = corr_matrix[np.ix_(right_indices, right_indices)]
            np.fill_diagonal(right_corr, 0)
            features['right_hemisphere_mean_corr'] = np.sum(right_corr) / (len(right_indices)*(len(right_indices)-1))

        # 对侧半球间
        if left_indices is not None and right_indices is not None:
            cross_corr = corr_matrix[np.ix_(left_indices, right_indices)]
            features['cross_hemisphere_mean_corr'] = np.mean(cross_corr)

        # 前后脑区
        if anterior_indices is not None and len(anterior_indices) >= 2:
            ant_corr = corr_matrix[np.ix_(anterior_indices, anterior_indices)]
            np.fill_diagonal(ant_corr, 0)
            features['anterior_mean_corr'] = np.sum(ant_corr) / (len(anterior_indices)*(len(anterior_indices)-1))
        if posterior_indices is not None and len(posterior_indices) >= 2:
            post_corr = corr_matrix[np.ix_(posterior_indices, posterior_indices)]
            np.fill_diagonal(post_corr, 0)
            features['posterior_mean_corr'] = np.sum(post_corr) / (len(posterior_indices)*(len(posterior_indices)-1))
        if anterior_indices is not None and posterior_indices is not None:
            ant_post_corr = corr_matrix[np.ix_(anterior_indices, posterior_indices)]
            features['anterior_posterior_mean_corr'] = np.mean(ant_post_corr)

        return features

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def _add_prefix(self, feat_dict: dict, prefix: str) -> dict:
        return {prefix + key: val for key, val in feat_dict.items()}


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    fs = 5.0
    t = np.arange(0, 60, 1/fs)

    def hrf_model(t, onset=10, peak_delay=6, undershoot_delay=16):
        return (np.maximum(0, (t-onset)/2) * np.exp(-(t-onset-peak_delay)**2/8) -
                0.2 * np.maximum(0, (t-onset-undershoot_delay)/2) * np.exp(-(t-onset-undershoot_delay)**2/8))

    hbo_signal = hrf_model(t, onset=10) * 5 + np.random.normal(0, 0.1, len(t))
    hbr_signal = -hrf_model(t, onset=10) * 2 + np.random.normal(0, 0.05, len(t))

    fnirs_extractor = FNIRSFeatureExtractor(fs=fs)

    print("=== 双信号源特征 ===")
    dual_features = fnirs_extractor.extract_hbo_hbr_features(hbo_signal, hbr_signal)
    for k, v in list(dual_features.items())[:8]:
        print(f"{k}: {v:.4f}")

    print("\n=== 血流动力学响应特征 ===")
    hrf_features = fnirs_extractor.extract_hrf_features(
        hbo_signal, fs=fs,
        baseline_window=(0, 10),
        task_window=(10, 30)
    )
    for k, v in hrf_features.items():
        print(f"{k}: {v:.4f}")

    print("\n=== 通道间相关性特征 ===")
    n_ch = 4
    ch_data = np.zeros((n_ch, len(t)))
    for i in range(n_ch):
        ch_data[i] = hbo_signal * (1 + 0.1*i) + np.random.normal(0, 0.1, len(t))
    left_idx = [0, 1]
    right_idx = [2, 3]
    corr_feats = fnirs_extractor.compute_channel_correlation(
        ch_data, left_indices=left_idx, right_indices=right_idx)
    for k, v in corr_feats.items():
        print(f"{k}: {v:.4f}")