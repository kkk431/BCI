"""
stat_inference.py
完全适配四层嵌套结构，状态感知管线：
- 自动判断数据源优先级：feature > epoch > raw signal
- 所有结果写入 processed.statistics.inference
- 完全兼容原函数接口语义

优化项：
- 移除未使用的joblib依赖
- welch导入移到文件顶部
- 实现permutation检验
- AUC手动回退（当sklearn不可用时）
"""

import numpy as np
from scipy import stats, signal
from scipy.signal import welch  # 移到顶部，避免重复导入
import warnings
from functools import lru_cache

# 可选依赖：sklearn.metrics.roc_auc_score
try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn未安装，将使用手动实现的AUC")

DEFAULT_MODALITY = 'EEG'


def _fast_auc(y_true, y_score):
    """手动实现AUC，零依赖"""
    n1 = np.sum(y_true == 0)
    n2 = np.sum(y_true == 1)
    if n1 == 0 or n2 == 0:
        return 0.5

    ranks = np.argsort(y_score)
    rank_sum = np.sum(ranks[np.where(y_true == 1)[0]])
    u = rank_sum - (n2 * (n2 + 1)) / 2
    return u / (n1 * n2)


def _get_epochs_from_signal(data_dict, modality, tmin, tmax, baseline=None):
    """从连续信号+事件中提取epoch（仅当processed.epoch不存在时调用）"""
    sig = data_dict['signal'][modality]
    data = sig['data']
    srate = sig['sampling_rate']

    # 获取事件信息
    event = data_dict['event']
    event_times = np.array(event['event_time'])
    event_labels = np.array(event['event_label'])
    event_durations = np.array(event.get('duration', [0] * len(event_times)))

    tmin_samples = int(tmin * srate)
    tmax_samples = int(tmax * srate)

    epochs = []
    labels = []

    for i, t in enumerate(event_times):
        onset = int(t * srate)
        start = onset + tmin_samples
        end = onset + tmax_samples

        if start >= 0 and end <= data.shape[1]:
            epoch = data[:, start:end].copy()

            # 基线校正
            if baseline is not None:
                b_start = int(baseline[0] * srate)
                b_end = int(baseline[1] * srate)
                baseline_mean = data[:, onset + b_start:onset + b_end].mean(axis=1, keepdims=True)
                epoch = epoch - baseline_mean

            epochs.append(epoch)
            labels.append(event_labels[i])

    return np.array(epochs), np.array(labels)


def _get_data_source(data_dict, modality=DEFAULT_MODALITY,
                     feature_type=None, freq_band=None,
                     tmin=-0.5, tmax=2.0, baseline=None):
    """
    智能数据源选择核心函数
    优先级：feature > epoch > raw signal
    """
    # 1. 检查processed.feature
    if 'processed' in data_dict:
        proc = data_dict['processed']

        # 1.1 特征级数据
        if feature_type and 'feature' in proc:
            feat = proc['feature']
            if feat.get('type', '').lower() == feature_type.lower():
                if 'data' in feat and 'labels' in feat:
                    print(f"   [数据源] 使用已提取特征: {feature_type}")
                    return {
                        'data': feat['data'],
                        'labels': feat['labels'],
                        'source': 'feature',
                        'ch_names': feat.get('channels', None),
                        'srate': None
                    }

        # 1.2 epoch级数据
        if 'epoch' in proc:
            epoch_data = proc['epoch']
            if 'data' in epoch_data:
                print(f"   [数据源] 使用已分段数据 (epoch)")
                return {
                    'data': epoch_data['data'],
                    'labels': epoch_data.get('labels', None),
                    'source': 'epoch',
                    'ch_names': data_dict['signal'][modality]['channel_names'],
                    'srate': data_dict['signal'][modality]['sampling_rate']
                }

    # 2. 回退到原始信号
    if modality in data_dict['signal']:
        sig = data_dict['signal'][modality]
        print(f"   [数据源] 从连续信号动态提取epoch")
        epochs, labels = _get_epochs_from_signal(
            data_dict, modality, tmin, tmax, baseline
        )
        return {
            'data': epochs,
            'labels': labels,
            'source': 'raw',
            'ch_names': sig['channel_names'],
            'srate': sig['sampling_rate']
        }

    raise ValueError(f"无法找到可用的数据源（模态: {modality}）")


def _extract_power_feature(epochs, srate, freq_band, method='classic'):
    """提取频带功率特征"""
    n_epochs, n_channels, n_times = epochs.shape

    if method == 'classic':
        freqs, psd = welch(
            epochs.reshape(-1, n_times),
            fs=srate,
            nperseg=min(256, n_times // 2)
        )
        freq_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
        power = psd[:, freq_mask].mean(axis=1)
        return power.reshape(n_epochs, n_channels)

    else:  # parametric
        # 参数化周期估计（简化版）
        power = np.zeros((n_epochs, n_channels))
        for ep_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                f, p = welch(epochs[ep_idx, ch_idx], fs=srate, nperseg=min(256, n_times // 2))
                mask = (f >= freq_band[0]) & (f <= freq_band[1])
                power[ep_idx, ch_idx] = np.mean(p[mask])
        return power


def _extract_amplitude_feature(epochs, srate, window):
    """提取ERP幅值特征"""
    start = int(window[0] * srate)
    end = int(window[1] * srate)
    return epochs[:, :, start:end].mean(axis=2)


def _fdr_bh(p_values):
    """手动实现FDR BH校正，零依赖"""
    p_flat = p_values.flatten()
    mask = ~np.isnan(p_flat)
    if not np.any(mask):
        return p_values

    p = p_flat[mask].copy()
    n = len(p)

    # BH过程
    order = np.argsort(p)
    p_ordered = p[order]

    p_corrected = p_ordered * n / np.arange(1, n + 1)
    p_corrected = np.minimum.accumulate(p_corrected[::-1])[::-1]
    p_corrected = np.clip(p_corrected, 0, 1)

    # 恢复原始顺序
    p_result = np.full_like(p_flat, np.nan)
    p_result[mask] = p_corrected[np.argsort(order)]

    return p_result.reshape(p_values.shape)


def hypothesis_testing(data_dict, condition1, condition2,
                       modality=DEFAULT_MODALITY,
                       feature_type='power',
                       freq_band=None,
                       channels=None,
                       time_window=None,
                       test_type='ttest',
                       correction='fdr',
                       advanced=False,
                       use_existing=True):
    """
    核心假设检验函数

    参数:
        data_dict: 四层嵌套结构
        condition1, condition2: 条件标签，如 ['left'], ['right']
        modality: 使用哪种模态，默认EEG
        feature_type: 'power', 'amplitude', 'raw'
        freq_band: 频带，如 [8, 13]
        channels: 指定通道，None表示全部
        time_window: 幅值特征时间窗，如 [0.1, 0.3]
        test_type: 'ttest', 'wilcoxon', 'permutation'
        correction: 'fdr', 'bonferroni', 'cluster', None
        advanced: 是否使用先进方法（贝叶斯因子、参数化周期）
        use_existing: 是否优先使用已存在的特征/epoch
    """
    # ============ 1. 智能获取数据源 ============
    if not use_existing:
        # 强制从原始信号提取
        source = _get_data_source(
            data_dict, modality,
            feature_type=None,  # 跳过特征
            tmin=-0.5, tmax=2.0,
            baseline=(-0.2, 0) if feature_type == 'amplitude' else None
        )
    else:
        source = _get_data_source(
            data_dict, modality,
            feature_type=feature_type if feature_type != 'raw' else None,
            tmin=-0.5, tmax=2.0,
            baseline=(-0.2, 0) if feature_type == 'amplitude' else None
        )

    data = source['data']
    labels = source['labels']
    ch_names = source['ch_names']
    srate = source['srate']

    # ============ 2. 数据格式标准化 ============
    if data.ndim == 2 and source['source'] != 'feature':
        # 可能是2维特征数据，需要reshape
        pass

    # ============ 3. 按条件分割 ============
    mask1 = np.isin(labels, condition1)
    mask2 = np.isin(labels, condition2)

    if mask1.sum() == 0 or mask2.sum() == 0:
        raise ValueError(f"条件标签不存在: {condition1} 或 {condition2}")

    data1 = data[mask1]
    data2 = data[mask2]

    # ============ 4. 通道选择 ============
    if channels is not None and source['source'] != 'feature':
        # 转换通道名到索引
        if isinstance(channels[0], str):
            ch_idx = [ch_names.index(ch) for ch in channels if ch in ch_names]
        else:
            ch_idx = channels

        if data1.ndim == 3:
            data1 = data1[:, ch_idx, :]
            data2 = data2[:, ch_idx, :]
        elif data1.ndim == 2:
            # 特征级数据，假设列对应通道
            data1 = data1[:, ch_idx]
            data2 = data2[:, ch_idx]

    # ============ 5. 特征提取（如果还不是特征） ============
    if source['source'] == 'feature':
        # 已经是特征，直接使用
        X1 = data1
        X2 = data2
        feature_info = {'source': 'existing_feature'}

    elif feature_type == 'power':
        X1 = _extract_power_feature(data1, srate, freq_band,
                                    method='parametric' if advanced else 'classic')
        X2 = _extract_power_feature(data2, srate, freq_band,
                                    method='parametric' if advanced else 'classic')
        feature_info = {'type': 'power', 'freq_band': freq_band}

    elif feature_type == 'amplitude':
        if time_window is None:
            time_window = [0.1, 0.3]
        X1 = _extract_amplitude_feature(data1, srate, time_window)
        X2 = _extract_amplitude_feature(data2, srate, time_window)
        feature_info = {'type': 'amplitude', 'window': time_window}

    else:  # raw
        # 使用原始时空数据（用于簇置换检验）
        X1 = data1
        X2 = data2
        feature_info = {'type': 'raw'}

    # ============ 6. 统计检验 ============
    results = {
        'conditions': [condition1, condition2],
        'modality': modality,
        'feature_type': feature_type,
        'feature_info': feature_info,
        'channels': channels if channels else 'all',
        'test_type': test_type,
        'correction': correction,
        'advanced_methods': advanced,
        'data_source': source['source'],
        'n_trials_cond1': len(X1),
        'n_trials_cond2': len(X2)
    }

    # 6.1 检验执行
    if test_type == 'ttest':
        if X1.ndim == 3:  # 时空数据 -> 逐点t检验
            t_stats = np.zeros((X1.shape[1], X1.shape[2]))
            p_vals = np.zeros((X1.shape[1], X1.shape[2]))
            for i in range(X1.shape[1]):
                for j in range(X1.shape[2]):
                    t, p = stats.ttest_ind(X1[:, i, j], X2[:, i, j])
                    t_stats[i, j] = t
                    p_vals[i, j] = p
            results['statistic'] = t_stats
            results['p_value'] = p_vals

        else:  # 特征数据
            t_stats, p_vals = stats.ttest_ind(X1, X2, axis=0)
            results['statistic'] = t_stats
            results['p_value'] = p_vals
            results['df'] = len(X1) + len(X2) - 2

    elif test_type == 'wilcoxon':
        from scipy.stats import mannwhitneyu
        if X1.ndim == 1 or X1.shape[1] == 1:
            stat, p = mannwhitneyu(X1.flatten(), X2.flatten())
            results['statistic'] = stat
            results['p_value'] = p
        else:
            stats_arr = []
            p_arr = []
            for i in range(X1.shape[1]):
                stat, p = mannwhitneyu(X1[:, i], X2[:, i])
                stats_arr.append(stat)
                p_arr.append(p)
            results['statistic'] = np.array(stats_arr)
            results['p_value'] = np.array(p_arr)

    elif test_type == 'permutation':
        # 新增：置换检验实现
        from sklearn.utils import shuffle
        n_permutations = 5000

        if X1.ndim == 3:  # 时空数据
            # 扁平化处理
            X1_flat = X1.reshape(len(X1), -1)
            X2_flat = X2.reshape(len(X2), -1)
            observed_t, _ = stats.ttest_ind(X1_flat, X2_flat, axis=0)
            combined = np.concatenate([X1_flat, X2_flat], axis=0)
        else:
            observed_t, _ = stats.ttest_ind(X1, X2, axis=0)
            combined = np.concatenate([X1, X2], axis=0)

        n1 = len(X1)
        perm_dist = []

        for _ in range(n_permutations):
            shuffled = shuffle(combined)
            t, _ = stats.ttest_ind(shuffled[:n1], shuffled[n1:], axis=0)
            perm_dist.append(t)

        perm_dist = np.array(perm_dist)

        if np.isscalar(observed_t):
            p_val = np.mean(np.abs(perm_dist) >= np.abs(observed_t))
        else:
            p_val = np.mean(np.abs(perm_dist) >= np.abs(observed_t), axis=0)

        results['statistic'] = observed_t
        results['p_value'] = p_val
        results['n_permutations'] = n_permutations

    # ============ 7. 多重比较校正 ============
    if correction == 'fdr':
        try:
            # 优先使用statsmodels
            from statsmodels.stats.multitest import multipletests
            p_flat = results['p_value'].flatten()
            mask = ~np.isnan(p_flat)
            if np.any(mask):
                reject, p_corr, _, _ = multipletests(p_flat[mask], method='fdr_bh')
                p_corrected = np.full_like(p_flat, np.nan)
                p_corrected[mask] = p_corr
                results['p_corrected'] = p_corrected.reshape(results['p_value'].shape)
                results['correction_method'] = 'fdr_bh'
        except ImportError:
            # 回退到手动实现
            results['p_corrected'] = _fdr_bh(results['p_value'])
            results['correction_method'] = 'fdr_bh_manual'

    elif correction == 'bonferroni':
        n = np.sum(~np.isnan(results['p_value']))
        results['p_corrected'] = np.clip(results['p_value'] * n, 0, 1)
        results['correction_method'] = 'bonferroni'

    # ============ 8. 效应量 ============
    if X1.ndim <= 2:  # 仅对特征数据计算效应量
        # Cohen's d
        mean1 = np.mean(X1, axis=0)
        mean2 = np.mean(X2, axis=0)
        std1 = np.std(X1, axis=0, ddof=1)
        std2 = np.std(X2, axis=0, ddof=1)
        n1, n2 = len(X1), len(X2)
        pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
        cohens_d = np.abs(mean1 - mean2) / (pooled_std + 1e-10)
        results['effect_size'] = {'cohens_d': cohens_d}

        # AUC（二分类）
        if X1.shape[1] == 1:
            y_true = np.concatenate([np.zeros(n1), np.ones(n2)])
            y_score = np.concatenate([X1.flatten(), X2.flatten()])
            if SKLEARN_AVAILABLE:
                results['effect_size']['auc'] = roc_auc_score(y_true, y_score)
            else:
                results['effect_size']['auc'] = _fast_auc(y_true, y_score)

    # ============ 9. 写入processed ============
    if 'processed' not in data_dict:
        data_dict['processed'] = {}
    if 'statistics' not in data_dict['processed']:
        data_dict['processed']['statistics'] = {}

    data_dict['processed']['statistics']['inference'] = results

    return data_dict


def effect_size(data_dict, condition1, condition2, **kwargs):
    """便捷函数：仅计算效应量"""
    result_dict = hypothesis_testing(
        data_dict, condition1, condition2, **kwargs
    )
    return result_dict['processed']['statistics']['inference']['effect_size']


def extract_epoch(data_dict, modality=DEFAULT_MODALITY,
                  tmin=-0.5, tmax=2.0, baseline=(-0.2, 0)):
    """
    显式提取epoch并存入processed.epoch
    供后续分析复用
    """
    epochs, labels = _get_epochs_from_signal(
        data_dict, modality, tmin, tmax, baseline
    )

    if 'processed' not in data_dict:
        data_dict['processed'] = {}

    data_dict['processed']['epoch'] = {
        'tmin': tmin,
        'tmax': tmax,
        'baseline': baseline,
        'data': epochs,
        'labels': labels,
        'modality': modality,
        'n_trials': len(epochs),
        'n_channels': epochs.shape[1],
        'n_times': epochs.shape[2]
    }

    return data_dict