"""
stat_inference.py
================================================================================
脑机接口统计检验模块 —— 纯统计推断，绝不包含任何特征提取代码！

核心功能：
    1. 假设检验：t检验、Wilcoxon、置换检验
    2. 时空聚类置换检验（EEG/fNIRS多重比较黄金标准）
    3. 效应量：Cohen's d, AUC, η², partial η²
    4. 多重比较校正：FDR, Bonferroni

设计哲学：
    - 纯统计：只接收【已提取的特征矩阵】或【已分段的时空数据】
    - 绝不导入任何特征提取模块，绝不动态提取特征
    - 绝不假设任何特征提取函数的存在
    - 所有数据必须由外部准备好后传入，本模块只做统计计算

输入格式：
    - 特征数据：必须通过 feature_key 从 data_dict['processed']['features'] 获取
    - 时空数据：必须通过 epoch_key 从 data_dict['processed']['epochs'] 获取

依赖：
    - 必需：numpy, scipy
    - 可选：joblib（无时自动降级单核）

版本: 2.0.0
最后更新: 2024
================================================================================
"""

import numpy as np
from scipy import stats, ndimage
from scipy.stats import mannwhitneyu
import warnings
import logging

# joblib可选导入（仅用于并行加速，非必需）
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 1. 基础工具函数（零依赖）
# ============================================================================

def _fast_auc(y_true, y_score):
    """手动实现AUC，零依赖版本。"""
    n1 = np.sum(y_true == 0)
    n2 = np.sum(y_true == 1)
    if n1 == 0 or n2 == 0:
        return 0.5
    ranks = np.argsort(y_score)
    rank_sum = np.sum(ranks[np.where(y_true == 1)[0]])
    u = rank_sum - (n2 * (n2 + 1)) / 2
    return u / (n1 * n2)


def _fdr_bh(p_values):
    """手动实现FDR BH校正，零依赖版本。"""
    p_flat = p_values.flatten()
    mask = ~np.isnan(p_flat)
    if not np.any(mask):
        return p_values

    p = p_flat[mask].copy()
    n = len(p)
    order = np.argsort(p)
    p_ordered = p[order]
    p_corrected = p_ordered * n / np.arange(1, n + 1)
    p_corrected = np.minimum.accumulate(p_corrected[::-1])[::-1]
    p_corrected = np.clip(p_corrected, 0, 1)

    p_result = np.full_like(p_flat, np.nan)
    p_result[mask] = p_corrected[np.argsort(order)]
    return p_result.reshape(p_values.shape)


# ============================================================================
# 2. 假设检验核心函数（纯数学计算）
# ============================================================================

def ttest_independent(X1, X2, equal_var=True, tail=0):
    """
    独立样本t检验。

    Parameters
    ----------
    X1, X2 : ndarray
        两组样本数据，shape (n1, n_features) 或 (n1,)
    equal_var : bool, default=True
        是否假设方差齐性
    tail : {0, 1, -1}, default=0
        检验尾型：0-双尾，1-单尾(X1>X2)，-1-单尾(X1<X2)

    Returns
    -------
    t_stat : float or ndarray
        t统计量
    p_val : float or ndarray
        p值
    df : int
        自由度
    """
    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)

    n1, n2 = len(X1), len(X2)
    df = n1 + n2 - 2

    mean1, mean2 = np.mean(X1, axis=0), np.mean(X2, axis=0)
    var1, var2 = np.var(X1, axis=0, ddof=1), np.var(X2, axis=0, ddof=1)

    if equal_var:
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / df
        se = np.sqrt(pooled_var * (1/n1 + 1/n2))
    else:
        se = np.sqrt(var1/n1 + var2/n2)
        df = ((var1/n1 + var2/n2)**2) / (
            (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
        )

    t_stat = (mean1 - mean2) / (se + 1e-10)

    if tail == 0:
        p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
    elif tail == 1:
        p_val = 1 - stats.t.cdf(t_stat, df)
    else:
        p_val = stats.t.cdf(t_stat, df)

    if t_stat.shape[0] == 1:
        return t_stat[0], p_val[0], df
    return t_stat, p_val, df


def wilcoxon_test(X1, X2):
    """Mann-Whitney U检验。"""
    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)

    if X1.shape[1] == 1:
        stat, p_val = mannwhitneyu(X1.flatten(), X2.flatten())
        return stat, p_val
    else:
        stats_arr = []
        p_arr = []
        for i in range(X1.shape[1]):
            stat, p = mannwhitneyu(X1[:, i], X2[:, i])
            stats_arr.append(stat)
            p_arr.append(p)
        return np.array(stats_arr), np.array(p_arr)


def permutation_test(X1, X2, n_permutations=5000, tail=0, random_state=42):
    """置换检验。"""
    from sklearn.utils import shuffle
    np.random.seed(random_state)

    if X1.ndim == 3:
        X1_flat = X1.reshape(len(X1), -1)
        X2_flat = X2.reshape(len(X2), -1)
        observed_stat, _ = stats.ttest_ind(X1_flat, X2_flat, axis=0)
        combined = np.concatenate([X1_flat, X2_flat], axis=0)
    else:
        observed_stat, _ = stats.ttest_ind(X1, X2, axis=0)
        combined = np.concatenate([X1, X2], axis=0)

    n1 = len(X1)
    perm_dist = []

    for _ in range(n_permutations):
        shuffled = shuffle(combined, random_state=random_state)
        t, _ = stats.ttest_ind(shuffled[:n1], shuffled[n1:], axis=0)
        perm_dist.append(t)

    perm_dist = np.array(perm_dist)

    if np.isscalar(observed_stat):
        if tail == 0:
            p_val = np.mean(np.abs(perm_dist) >= np.abs(observed_stat))
        elif tail == 1:
            p_val = np.mean(perm_dist >= observed_stat)
        else:
            p_val = np.mean(perm_dist <= observed_stat)
    else:
        if tail == 0:
            p_val = np.mean(np.abs(perm_dist) >= np.abs(observed_stat), axis=0)
        elif tail == 1:
            p_val = np.mean(perm_dist >= observed_stat, axis=0)
        else:
            p_val = np.mean(perm_dist <= observed_stat, axis=0)

    return observed_stat, p_val, perm_dist


# ============================================================================
# 3. 时空聚类置换检验（纯统计）
# ============================================================================

def cluster_permutation_test(X1, X2,
                            srate=None,
                            adjacency=None,
                            n_permutations=5000,
                            threshold_p=0.05,
                            tail=0,
                            min_cluster_duration=0.015,
                            n_jobs=-1,
                            random_state=42):
    """
    时空聚类置换检验 —— 纯统计计算，不涉及任何数据提取。

    Parameters
    ----------
    X1, X2 : ndarray, shape (n_trials, n_channels, n_times)
        两组时空数据。【必须已分段，本函数不进行任何数据提取】
    srate : float, optional
        采样率（Hz），用于最小持续时间过滤
    adjacency : ndarray, optional
        自定义空间邻接矩阵，默认8邻域
    n_permutations : int, default=5000
        置换次数
    threshold_p : float, default=0.05
        簇形成阈值（逐点p值）
    tail : {0, 1, -1}, default=0
        检验尾型
    min_cluster_duration : float, default=0.015
        最小簇持续时间（秒）
    n_jobs : int, default=-1
        并行核心数
    random_state : int, default=42
        随机种子

    Returns
    -------
    dict
        聚类检验结果
    """
    np.random.seed(random_state)

    n1, n2 = len(X1), len(X2)
    n_channels, n_times = X1.shape[1], X1.shape[2]

    # 1. 逐点t检验
    t_map = np.zeros((n_channels, n_times))
    p_map = np.ones((n_channels, n_times))

    for ch in range(n_channels):
        for tp in range(n_times):
            t, p = stats.ttest_ind(X1[:, ch, tp], X2[:, ch, tp])

            if tail == 1:
                p = p / 2 if t > 0 else 1 - p / 2
            elif tail == -1:
                p = p / 2 if t < 0 else 1 - p / 2

            t_map[ch, tp] = t
            p_map[ch, tp] = p

    # 2. 时空聚类
    sig_mask = p_map < threshold_p
    structure = adjacency or ndimage.generate_binary_structure(2, 2)
    labeled, n_clusters = ndimage.label(sig_mask, structure=structure)

    # 3. 过滤最小持续时间
    if min_cluster_duration > 0 and srate is not None:
        min_samples = int(min_cluster_duration * srate)
        for i in range(1, n_clusters + 1):
            if np.sum(labeled == i) < min_samples:
                labeled[labeled == i] = 0
        labeled, n_clusters = ndimage.label(labeled > 0, structure=structure)

    # 4. 观测簇质量
    observed_masses = []
    for i in range(1, n_clusters + 1):
        mass = np.sum(np.abs(t_map[labeled == i]))
        observed_masses.append(mass)

    # 5. 置换检验
    def _permutation_iter(seed):
        np.random.seed(seed)
        combined = np.concatenate([X1, X2], axis=0)
        perm_idx = np.random.permutation(len(combined))
        perm1 = combined[perm_idx[:n1]]
        perm2 = combined[perm_idx[n1:]]

        perm_t = np.zeros((n_channels, n_times))
        for ch in range(n_channels):
            for tp in range(n_times):
                t, _ = stats.ttest_ind(perm1[:, ch, tp], perm2[:, ch, tp])
                perm_t[ch, tp] = t

        t_thresh = stats.t.ppf(1 - threshold_p/2, n1 + n2 - 2)
        perm_sig = np.abs(perm_t) > t_thresh
        perm_labeled, _ = ndimage.label(perm_sig, structure=structure)

        max_mass = 0
        for i in range(1, np.max(perm_labeled) + 1):
            mass = np.sum(np.abs(perm_t[perm_labeled == i]))
            max_mass = max(max_mass, mass)
        return max_mass

    seeds = np.random.randint(0, 2**32, n_permutations)

    if JOBLIB_AVAILABLE and n_jobs != 1:
        perm_max_masses = Parallel(n_jobs=n_jobs)(
            delayed(_permutation_iter)(seed) for seed in seeds
        )
    else:
        perm_max_masses = [_permutation_iter(seed) for seed in seeds]

    perm_max_masses = np.array(perm_max_masses)

    # 6. 簇p值
    cluster_pvals = [np.mean(perm_max_masses >= mass) for mass in observed_masses]

    # 7. 时间轴
    times = np.arange(n_times) / srate if srate else np.arange(n_times)

    return {
        'cluster_mask': labeled,
        'n_clusters': n_clusters,
        'cluster_masses': observed_masses,
        'cluster_p_values': cluster_pvals,
        'significant_clusters': [i for i, p in enumerate(cluster_pvals) if p < 0.05],
        't_map': t_map,
        'p_map': p_map,
        'times': times,
        'n_permutations': n_permutations,
        'threshold_p': threshold_p,
        'tail': tail
    }


# ============================================================================
# 4. 效应量计算
# ============================================================================

def cohens_d(X1, X2, pooled=True):
    """Cohen's d效应量。"""
    n1, n2 = len(X1), len(X2)
    mean1, mean2 = np.mean(X1), np.mean(X2)
    std1, std2 = np.std(X1, ddof=1), np.std(X2, ddof=1)

    if pooled:
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / (pooled_std + 1e-10)
    else:
        d = (mean1 - mean2) / (np.sqrt((std1**2 + std2**2) / 2) + 1e-10)
    return np.abs(d)


def eta_squared(ss_between, ss_total):
    """η²效应量。"""
    return ss_between / (ss_total + 1e-10)


def partial_eta_squared(ss_effect, ss_error):
    """偏η²效应量。"""
    return ss_effect / (ss_effect + ss_error + 1e-10)


def auc_effect(y_true, y_score):
    """AUC效应量。"""
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_score)
    except ImportError:
        return _fast_auc(y_true, y_score)


# ============================================================================
# 5. 多重比较校正
# ============================================================================

def multiple_comparison_correction(p_values, method='fdr'):
    """多重比较校正统一接口。"""
    if method == 'fdr':
        return _fdr_bh(p_values)
    elif method == 'bonferroni':
        n = np.sum(~np.isnan(p_values.flatten()))
        return np.clip(p_values * n, 0, 1)
    else:
        return p_values


# ============================================================================
# 6. 四层结构适配接口（纯数据IO，无特征提取）
# ============================================================================

def hypothesis_testing(data_dict, condition1, condition2,
                       feature_key,  # 必须显式指定，绝不自动提取
                       test_type='ttest',
                       correction='fdr',
                       tail=0,
                       n_permutations=5000,
                       **kwargs):
    """
    两组条件统计假设检验 —— 纯统计接口，绝不进行任何特征提取！

    Parameters
    ----------
    data_dict : dict
        四层嵌套结构，必须包含已提取的特征
    condition1, condition2 : list
        两组条件的标签
    feature_key : str
        【必须指定】特征矩阵的key，从 data_dict['processed']['features'] 获取
    test_type : {'ttest', 'wilcoxon', 'permutation'}, default='ttest'
        检验方法
    correction : {'fdr', 'bonferroni', None}, default='fdr'
        多重比较校正方法
    tail : {0, 1, -1}, default=0
        检验尾型
    n_permutations : int, default=5000
        置换检验迭代次数

    Returns
    -------
    data_dict : dict
        更新后的数据字典，结果存入 data_dict['processed']['statistics']['inference']

    Raises
    ------
    ValueError
        当找不到指定特征时抛出

    Notes
    -----
    本函数【绝不】进行任何特征提取！
    特征矩阵必须由外部模块提前提取并存入 data_dict['processed']['features'][feature_key]
    """
    # ============ 1. 强制要求特征已存在 ============
    if 'processed' not in data_dict:
        raise ValueError("data_dict中缺少processed字段")

    proc = data_dict['processed']

    if feature_key is None:
        raise ValueError("必须指定feature_key，本模块不进行任何特征提取！")

    # 只接受显式存储的特征矩阵
    if 'features' not in proc or feature_key not in proc['features']:
        raise ValueError(
            f"找不到特征: {feature_key}\n"
            "请先运行特征提取模块，将特征矩阵存入 data_dict['processed']['features']\n"
            "本模块【绝不】进行任何动态特征提取！"
        )

    # 2. 获取特征矩阵
    X = proc['features'][feature_key]['data']
    y = np.array(proc['features'][feature_key]['labels'])

    # 3. 按条件分割
    mask1 = np.isin(y, condition1)
    mask2 = np.isin(y, condition2)

    if mask1.sum() == 0 or mask2.sum() == 0:
        raise ValueError(f"条件标签不存在: {condition1} 或 {condition2}")

    X1, X2 = X[mask1], X[mask2]

    # 4. 统计检验
    if test_type == 'ttest':
        stat, p_val, df = ttest_independent(X1, X2, tail=tail)
        results = {'statistic': stat, 'p_value': p_val, 'df': df, 'test_type': 'ttest'}
    elif test_type == 'wilcoxon':
        stat, p_val = wilcoxon_test(X1, X2)
        results = {'statistic': stat, 'p_value': p_val, 'test_type': 'wilcoxon'}
    elif test_type == 'permutation':
        stat, p_val, perm_dist = permutation_test(
            X1, X2, n_permutations, tail,
            random_state=kwargs.get('random_state', 42)
        )
        results = {
            'statistic': stat,
            'p_value': p_val,
            'permutation_distribution': perm_dist,
            'n_permutations': n_permutations,
            'test_type': 'permutation'
        }
    else:
        raise ValueError(f"不支持的检验类型: {test_type}")

    # 5. 多重比较校正
    if correction == 'fdr':
        results['p_corrected'] = _fdr_bh(results['p_value'])
        results['correction_method'] = 'fdr_bh'
    elif correction == 'bonferroni':
        n = np.sum(~np.isnan(results['p_value'].flatten()))
        results['p_corrected'] = np.clip(results['p_value'] * n, 0, 1)
        results['correction_method'] = 'bonferroni'

    # 6. 效应量
    if X1.ndim <= 2 and X1.shape[1] == 1:
        results['effect_size'] = {
            'cohens_d': cohens_d(X1.flatten(), X2.flatten()),
            'auc': auc_effect(
                np.concatenate([np.zeros(len(X1)), np.ones(len(X2))]),
                np.concatenate([X1.flatten(), X2.flatten()])
            )
        }
    elif X1.ndim <= 2:
        d_list = [cohens_d(X1[:, i], X2[:, i]) for i in range(X1.shape[1])]
        results['effect_size'] = {
            'cohens_d': np.array(d_list),
            'mean_cohens_d': np.mean(d_list)
        }

    # 7. 元信息
    results.update({
        'conditions': [condition1, condition2],
        'n_trials_cond1': len(X1),
        'n_trials_cond2': len(X2),
        'feature_key': feature_key,
        'tail': tail
    })

    # 8. 写入data_dict
    if 'statistics' not in proc:
        proc['statistics'] = {}
    proc['statistics']['inference'] = results

    p_min = np.nanmin(p_val) if not np.isscalar(p_val) else p_val
    p_max = np.nanmax(p_val) if not np.isscalar(p_val) else p_val
    logger.info(f"统计检验完成: {test_type}, p值范围: {p_min:.4f} - {p_max:.4f}")

    return data_dict


def cluster_permutation(data_dict, condition1, condition2,
                        epoch_key,  # 必须显式指定，绝不自动提取
                        channels=None,
                        n_permutations=5000,
                        threshold_p=0.05,
                        tail=0,
                        min_cluster_duration=0.015,
                        n_jobs=-1):
    """
    时空聚类置换检验 —— 纯统计接口，绝不进行任何epoch提取！

    Parameters
    ----------
    data_dict : dict
        四层嵌套结构，必须包含已分段的epoch数据
    condition1, condition2 : list
        两组条件的标签
    epoch_key : str
        【必须指定】epoch数据的key，从 data_dict['processed']['epochs'] 获取
    channels : list, optional
        指定通道
    n_permutations : int, default=5000
        置换次数
    threshold_p : float, default=0.05
        簇形成阈值
    tail : {0, 1, -1}, default=0
        检验尾型
    min_cluster_duration : float, default=0.015
        最小簇持续时间（秒）
    n_jobs : int, default=-1
        并行核心数

    Returns
    -------
    data_dict : dict
        更新后的数据字典
    """
    # ============ 1. 强制要求epoch已存在 ============
    if 'processed' not in data_dict:
        raise ValueError("data_dict中缺少processed字段")

    proc = data_dict['processed']

    if epoch_key is None:
        raise ValueError("必须指定epoch_key，本模块不进行任何epoch提取！")

    if 'epochs' not in proc or epoch_key not in proc['epochs']:
        raise ValueError(
            f"找不到epoch数据: {epoch_key}\n"
            "请先运行数据分段模块，将时空数据存入 data_dict['processed']['epochs']\n"
            "本模块【绝不】进行任何动态epoch提取！"
        )

    # 2. 获取epoch数据
    epoch_data = proc['epochs'][epoch_key]
    X = epoch_data['data']
    y = np.array(epoch_data['labels'])
    srate = epoch_data.get('sampling_rate')
    ch_names = epoch_data.get('ch_names')

    # 3. 通道选择
    if channels is not None:
        if ch_names is None:
            raise ValueError("指定了channels但epoch数据中没有ch_names字段")
        if isinstance(channels[0], str):
            ch_idx = [ch_names.index(ch) for ch in channels if ch in ch_names]
        else:
            ch_idx = channels
        X = X[:, ch_idx, :]
        ch_names = [ch_names[i] for i in ch_idx]

    # 4. 按条件分割
    mask1 = np.isin(y, condition1)
    mask2 = np.isin(y, condition2)

    if mask1.sum() == 0 or mask2.sum() == 0:
        raise ValueError(f"条件标签不存在: {condition1} 或 {condition2}")

    # 5. 执行聚类检验
    cluster_results = cluster_permutation_test(
        X[mask1], X[mask2],
        srate=srate,
        n_permutations=n_permutations,
        threshold_p=threshold_p,
        tail=tail,
        min_cluster_duration=min_cluster_duration,
        n_jobs=n_jobs
    )

    # 6. 添加元信息
    cluster_results.update({
        'condition1': condition1,
        'condition2': condition2,
        'epoch_key': epoch_key,
        'channels': ch_names,
        'n_trials_cond1': mask1.sum(),
        'n_trials_cond2': mask2.sum()
    })

    # 7. 写入data_dict
    if 'statistics' not in proc:
        proc['statistics'] = {}
    if 'inference' not in proc['statistics']:
        proc['statistics']['inference'] = {}

    proc['statistics']['inference']['cluster_permutation'] = cluster_results

    n_sig = len(cluster_results['significant_clusters'])
    logger.info(f"时空聚类完成: 发现{cluster_results['n_clusters']}个簇，{n_sig}个显著")

    return data_dict


# ============================================================================
# 7. 便捷函数
# ============================================================================

def effect_size(data_dict, condition1, condition2, feature_key, **kwargs):
    """仅计算效应量，不进行完整假设检验。"""
    result_dict = hypothesis_testing(
        data_dict, condition1, condition2,
        feature_key=feature_key, **kwargs
    )
    return result_dict['processed']['statistics']['inference'].get('effect_size', {})