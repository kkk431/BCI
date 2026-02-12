"""
connectivity_stats.py
================================================================================
脑机接口连接性统计模块 —— 纯连接分析，无特征提取

核心功能：
    1. 功能连接：相干性、锁相值、相位滞后指数、加权PLI、互信息
    2. 跨模态连接：EEG-EMG/EOG相位耦合
    3. 图论指标：聚类系数、特征路径长度、全局/局部效率、模块化、小世界性
    4. 网络统计推断：NBS、基于置换的网络指标比较

设计哲学：
    - 纯连接：只接收时空数据，输出连接矩阵
    - 单模态/跨模态二象性：fusion_method=None单模态，fusion_method='cross'跨模态
    - 依赖可选：networkx缺失时图论指标不可用，但连接计算不受影响

输入格式：
    - 时空数据：ndarray, shape (n_trials, n_channels, n_times)

输出格式：
    - 统一存入 data_dict['processed']['statistics']['connectivity']

依赖：
    - 必需：numpy, scipy
    - 可选：networkx（图论指标、NBS）

版本: 2.0.0
最后更新: 2024
================================================================================
"""

import numpy as np
from scipy import signal, stats
from scipy.signal import butter, filtfilt, hilbert
import warnings
from joblib import Parallel, delayed
import logging

logger = logging.getLogger(__name__)

# networkx可选导入
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("networkx未安装，图论指标和NBS不可用。如需使用请: pip install networkx")

DEFAULT_MODALITY = 'EEG'
DEFAULT_FUSION = None


# ============================================================================
# 1. 单模态功能连接
# ============================================================================

def coherence(data, fs, freq_band=None, nperseg=None):
    """
    计算相干性（Coherence）连接矩阵。

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        单试次时空数据。
    fs : float
        采样率（Hz）。
    freq_band : list of two floats, optional
        频带范围，如[8, 13]。
    nperseg : int, optional
        Welch法每段长度，默认min(256, n_times//4)。

    Returns
    -------
    coh_mat : ndarray, shape (n_channels, n_channels)
        相干性矩阵，取值[0,1]。
    """
    n_channels, n_times = data.shape
    if nperseg is None:
        nperseg = min(256, n_times // 4)

    f, Cxy = signal.coherence(data, fs=fs, nperseg=nperseg)

    if freq_band is not None:
        freq_mask = (f >= freq_band[0]) & (f <= freq_band[1])
        coh_mat = Cxy[:, :, freq_mask].mean(axis=2)
    else:
        coh_mat = Cxy.mean(axis=2)

    return coh_mat


def phase_locking_value(data, fs=None, freq_band=None):
    """
    计算锁相值（Phase Locking Value）连接矩阵。

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        单试次时空数据。
    fs : float, optional
        采样率，仅当freq_band不为None时用于滤波。
    freq_band : list of two floats, optional
        频带范围。

    Returns
    -------
    plv_mat : ndarray, shape (n_channels, n_channels)
        锁相值矩阵，取值[0,1]。
    """
    n_channels = data.shape[0]

    # 频带滤波
    if freq_band is not None and fs is not None:
        b, a = butter(4, freq_band, btype='band', fs=fs)
        data_filt = np.array([filtfilt(b, a, ch_data) for ch_data in data])
    else:
        data_filt = data

    # 希尔伯特变换提取瞬时相位
    analytic = hilbert(data_filt)
    phase = np.angle(analytic)

    # 计算PLV
    plv_mat = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = phase[i] - phase[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_mat[i, j] = plv
            plv_mat[j, i] = plv

    return plv_mat


def phase_lag_index(data, fs=None, freq_band=None):
    """
    计算相位滞后指数（Phase Lag Index）连接矩阵。

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        单试次时空数据。
    fs : float, optional
        采样率。
    freq_band : list of two floats, optional
        频带范围。

    Returns
    -------
    pli_mat : ndarray, shape (n_channels, n_channels)
        PLI矩阵，取值[0,1]。

    Notes
    -----
    PLI对容积传导不敏感，通过忽略零相位差来抑制伪迹。
    """
    n_channels = data.shape[0]

    if freq_band is not None and fs is not None:
        b, a = butter(4, freq_band, btype='band', fs=fs)
        data_filt = np.array([filtfilt(b, a, ch_data) for ch_data in data])
    else:
        data_filt = data

    analytic = hilbert(data_filt)
    phase = np.angle(analytic)

    pli_mat = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = phase[i] - phase[j]
            pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
            pli_mat[i, j] = pli
            pli_mat[j, i] = pli

    return pli_mat


def weighted_phase_lag_index(data, fs=None, freq_band=None):
    """
    计算加权相位滞后指数（Weighted PLI）连接矩阵。

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        单试次时空数据。
    fs : float, optional
        采样率。
    freq_band : list of two floats, optional
        频带范围。

    Returns
    -------
    wpli_mat : ndarray, shape (n_channels, n_channels)
        wPLI矩阵，取值[0,1]。

    Notes
    -----
    wPLI对噪声更鲁棒，通过加权幅度来减少小相位差的影响。
    """
    n_channels = data.shape[0]

    if freq_band is not None and fs is not None:
        b, a = butter(4, freq_band, btype='band', fs=fs)
        data_filt = np.array([filtfilt(b, a, ch_data) for ch_data in data])
    else:
        data_filt = data

    analytic = hilbert(data_filt)
    phase = np.angle(analytic)

    wpli_mat = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = phase[i] - phase[j]
            imag_component = np.sin(phase_diff)
            wpli = np.abs(np.mean(np.abs(imag_component) * np.sign(imag_component)))
            wpli_mat[i, j] = wpli
            wpli_mat[j, i] = wpli

    return wpli_mat


def mutual_information(data, bins=20):
    """
    计算互信息（Mutual Information）连接矩阵。

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        单试次时空数据。
    bins : int, default=20
        直方图分箱数。

    Returns
    -------
    mi_mat : ndarray, shape (n_channels, n_channels)
        互信息矩阵。

    Notes
    -----
    依赖sklearn.metrics.mutual_info_score。
    """
    try:
        from sklearn.metrics import mutual_info_score
    except ImportError:
        raise ImportError("互信息需要scikit-learn: pip install scikit-learn")

    n_channels = data.shape[0]
    mi_mat = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            c_xy = np.histogram2d(data[i], data[j], bins)[0]
            mi = mutual_info_score(None, None, contingency=c_xy)
            mi_mat[i, j] = mi
            mi_mat[j, i] = mi

    return mi_mat


# ============================================================================
# 2. 跨模态功能连接
# ============================================================================

def cross_modal_plv(data1, data2, fs=None, freq_band=None):
    """
    跨模态锁相值计算（如EEG-EMG相位耦合）。

    Parameters
    ----------
    data1 : ndarray, shape (n_channels1, n_times)
        模态1数据。
    data2 : ndarray, shape (n_channels2, n_times)
        模态2数据。
    fs : float, optional
        采样率。
    freq_band : list of two floats, optional
        频带范围。

    Returns
    -------
    plv_mat : ndarray, shape (n_channels1, n_channels2)
        跨模态PLV矩阵。
    """
    n_channels1 = data1.shape[0]
    n_channels2 = data2.shape[0]

    # 频带滤波
    if freq_band is not None and fs is not None:
        b, a = butter(4, freq_band, btype='band', fs=fs)
        data1_filt = np.array([filtfilt(b, a, ch_data) for ch_data in data1])
        data2_filt = np.array([filtfilt(b, a, ch_data) for ch_data in data2])
    else:
        data1_filt = data1
        data2_filt = data2

    # 提取相位
    phase1 = np.angle(hilbert(data1_filt))
    phase2 = np.angle(hilbert(data2_filt))

    # 计算跨模态PLV
    plv_mat = np.zeros((n_channels1, n_channels2))
    for i in range(n_channels1):
        for j in range(n_channels2):
            phase_diff = phase1[i] - phase2[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_mat[i, j] = plv

    return plv_mat


def cross_modal_coherence(data1, data2, fs, freq_band=None, nperseg=None):
    """
    跨模态相干性计算。

    Parameters
    ----------
    data1 : ndarray, shape (n_channels1, n_times)
        模态1数据。
    data2 : ndarray, shape (n_channels2, n_times)
        模态2数据。
    fs : float
        采样率。
    freq_band : list of two floats, optional
        频带范围。
    nperseg : int, optional
        Welch法每段长度。

    Returns
    -------
    coh_mat : ndarray, shape (n_channels1, n_channels2)
        跨模态相干性矩阵。
    """
    n_channels1 = data1.shape[0]
    n_channels2 = data2.shape[0]

    if nperseg is None:
        nperseg = min(256, data1.shape[1] // 4)

    coh_mat = np.zeros((n_channels1, n_channels2))

    for i in range(n_channels1):
        for j in range(n_channels2):
            f, Cxy = signal.coherence(data1[i], data2[j], fs=fs, nperseg=nperseg)
            if freq_band is not None:
                freq_mask = (f >= freq_band[0]) & (f <= freq_band[1])
                coh_mat[i, j] = np.mean(Cxy[freq_mask])
            else:
                coh_mat[i, j] = np.mean(Cxy)

    return coh_mat


# ============================================================================
# 3. 高级连接计算接口
# ============================================================================

def functional_connectivity(data_dict,
                            modality=DEFAULT_MODALITY,
                            fusion_method=DEFAULT_FUSION,
                            modalities=None,
                            method='pli',
                            freq_band=None,
                            epoch_based=True,
                            use_existing_epoch=True,
                            save_to_dict=True):
    """
    功能连接计算统一接口（支持单模态/跨模态）。

    Parameters
    ----------
    data_dict : dict
        四层嵌套结构。
    modality : str, default='EEG'
        单模态分析时的模态。
    fusion_method : {None, 'cross'}, default=None
        - None: 单模态连接分析
        - 'cross': 跨模态连接分析
    modalities : list of two str, optional
        跨模态分析时指定两个模态，如['EEG', 'EMG']。
    method : {'pli', 'plv', 'wpli', 'coh', 'mi'}, default='pli'
        连接计算方法。
    freq_band : list of two floats, optional
        频带范围。
    epoch_based : bool, default=True
        是否返回所有试次的连接矩阵。
    use_existing_epoch : bool, default=True
        是否使用已存在的epoch数据。
    save_to_dict : bool, default=True
        是否将结果存入data_dict。

    Returns
    -------
    data_dict or (conn_mean, conn_epochs)
        当save_to_dict=True时返回更新后的data_dict，
        否则返回(平均连接矩阵, 试次级连接矩阵)。

    Examples
    --------
    >>> # 单模态PLI连接
    >>> data = functional_connectivity(data, modality='EEG', method='pli')
    >>>
    >>> # 跨模态EEG-EMG相位耦合
    >>> data = functional_connectivity(data,
    ...                                fusion_method='cross',
    ...                                modalities=['EEG', 'EMG'],
    ...                                method='plv')
    """
    if fusion_method is None:
        return _single_modal_connectivity(
            data_dict, modality, method, freq_band,
            epoch_based, use_existing_epoch, save_to_dict
        )
    elif fusion_method == 'cross':
        return _cross_modal_connectivity(
            data_dict, modalities, method, freq_band,
            epoch_based, use_existing_epoch, save_to_dict
        )
    else:
        raise ValueError(f"不支持的融合方法: {fusion_method}")


def _get_epoch_data(data_dict, modality, use_existing_epoch):
    """获取单模态epoch数据 —— 绝不动态提取！"""
    if not use_existing_epoch:
        raise ValueError("connectivity_stats 模块不支持动态 epoch 提取，请提前运行数据分段模块")

    if 'processed' not in data_dict:
        raise ValueError("data_dict 中缺少 processed 字段")

    proc = data_dict['processed']

    # 只接受显式存储的 epoch 数据
    if 'epochs' in proc and modality in proc['epochs']:
        epoch_data = proc['epochs'][modality]
        return epoch_data['data'], epoch_data.get('labels')
    elif 'epoch' in proc and proc['epoch'].get('modality') == modality:
        epoch_data = proc['epoch']
        return epoch_data['data'], epoch_data.get('labels')
    else:
        raise ValueError(
            f"找不到模态 {modality} 的 epoch 数据。\n"
            "connectivity_stats 模块绝不进行动态 epoch 提取！\n"
            "请先运行数据分段模块，将时空数据存入 data_dict['processed']['epochs']"
        )


def _single_modal_connectivity(data_dict, modality, method, freq_band,
                               epoch_based, use_existing_epoch, save_to_dict):
    """单模态连接计算"""
    # 1. 获取epoch数据
    epochs, labels = _get_epoch_data(data_dict, modality, use_existing_epoch)
    n_epochs, n_channels, n_times = epochs.shape
    srate = data_dict['signal'][modality]['sampling_rate']
    ch_names = data_dict['signal'][modality]['channel_names']

    # 2. 方法映射
    method_map = {
        'coh': coherence,
        'plv': phase_locking_value,
        'pli': phase_lag_index,
        'wpli': weighted_phase_lag_index,
        'mi': mutual_information
    }

    if method not in method_map:
        raise ValueError(f"不支持的方法: {method}, 可选: {list(method_map.keys())}")

    conn_func = method_map[method]

    # 3. 逐试次计算连接矩阵
    conn_epochs = []
    for ep_idx in range(n_epochs):
        data = epochs[ep_idx]

        # 频带滤波在具体函数内处理
        if method in ['coh', 'plv', 'pli', 'wpli']:
            conn = conn_func(data, srate, freq_band)
        else:
            conn = conn_func(data)

        conn_epochs.append(conn)

    conn_epochs = np.array(conn_epochs)
    conn_mean = np.mean(conn_epochs, axis=0)

    # 4. 结果封装
    result = {
        'method': method,
        'freq_band': freq_band,
        'matrix': conn_mean,
        'matrices_epoch': conn_epochs if not epoch_based else None,
        'channels': ch_names,
        'n_epochs': n_epochs,
        'modality': modality,
        'fusion_method': 'single'
    }

    if save_to_dict:
        # 生成唯一键
        if freq_band is not None:
            key = f"{method}_{freq_band[0]}-{freq_band[1]}Hz"
        else:
            key = method

        # 存入data_dict
        if 'processed' not in data_dict:
            data_dict['processed'] = {}
        if 'statistics' not in data_dict['processed']:
            data_dict['processed']['statistics'] = {}
        if 'connectivity' not in data_dict['processed']['statistics']:
            data_dict['processed']['statistics']['connectivity'] = {}

        data_dict['processed']['statistics']['connectivity'][key] = result
        return data_dict
    else:
        return conn_mean, result


def _cross_modal_connectivity(data_dict, modalities, method, freq_band,
                              epoch_based, use_existing_epoch, save_to_dict):
    """跨模态连接计算"""
    if modalities is None or len(modalities) != 2:
        raise ValueError("跨模态连接需要指定两个模态，如['EEG', 'EMG']")

    mod1, mod2 = modalities

    # 1. 获取两个模态的epoch数据
    epochs1, _ = _get_epoch_data(data_dict, mod1, use_existing_epoch)
    epochs2, _ = _get_epoch_data(data_dict, mod2, use_existing_epoch)

    # 2. 对齐试次数
    n_trials = min(len(epochs1), len(epochs2))
    epochs1 = epochs1[:n_trials]
    epochs2 = epochs2[:n_trials]

    # 3. 采样率对齐
    srate1 = data_dict['signal'][mod1]['sampling_rate']
    srate2 = data_dict['signal'][mod2]['sampling_rate']

    # 重采样到较低采样率
    target_srate = min(srate1, srate2)

    if srate1 != target_srate:
        from scipy import signal as resamp
        n_times1 = int(epochs1.shape[2] * target_srate / srate1)
        epochs1 = resamp.resample(epochs1, n_times1, axis=2)
    if srate2 != target_srate:
        from scipy import signal as resamp
        n_times2 = int(epochs2.shape[2] * target_srate / srate2)
        epochs2 = resamp.resample(epochs2, n_times2, axis=2)

    # 4. 统一时间长度
    n_times = min(epochs1.shape[2], epochs2.shape[2])
    epochs1 = epochs1[:, :, :n_times]
    epochs2 = epochs2[:, :, :n_times]

    # 5. 跨模态连接计算
    cross_conn_epochs = []

    for ep_idx in range(n_trials):
        data1 = epochs1[ep_idx]
        data2 = epochs2[ep_idx]

        if method == 'plv':
            conn = cross_modal_plv(data1, data2, target_srate, freq_band)
        elif method == 'coh':
            conn = cross_modal_coherence(data1, data2, target_srate, freq_band)
        else:
            raise ValueError(f"跨模态连接不支持方法: {method}")

        cross_conn_epochs.append(conn)

    cross_conn_mean = np.mean(cross_conn_epochs, axis=0)

    # 6. 结果封装
    result = {
        'method': f'cross_{method}',
        'freq_band': freq_band,
        'matrix': cross_conn_mean,
        'matrices_epoch': cross_conn_epochs if not epoch_based else None,
        'modality_from': mod1,
        'modality_to': mod2,
        'channels_from': data_dict['signal'][mod1]['channel_names'],
        'channels_to': data_dict['signal'][mod2]['channel_names'],
        'n_epochs': n_trials,
        'fusion_method': 'cross'
    }

    if save_to_dict:
        # 生成唯一键
        key = f"cross_{method}_{mod1}2{mod2}"
        if freq_band is not None:
            key += f"_{freq_band[0]}-{freq_band[1]}Hz"

        if 'processed' not in data_dict:
            data_dict['processed'] = {}
        if 'statistics' not in data_dict['processed']:
            data_dict['processed']['statistics'] = {}
        if 'connectivity' not in data_dict['processed']['statistics']:
            data_dict['processed']['statistics']['connectivity'] = {}

        data_dict['processed']['statistics']['connectivity'][key] = result
        return data_dict
    else:
        return cross_conn_mean, result


# ============================================================================
# 4. 图论指标
# ============================================================================

def graph_theory_metrics(connectivity_matrix, ch_names=None,
                         threshold=None, weighted=True):
    """
    计算图论指标。

    Parameters
    ----------
    connectivity_matrix : ndarray, shape (n_nodes, n_nodes)
        连接矩阵。
    ch_names : list of str, optional
        通道名称，用于节点中心性字典。
    threshold : float, optional
        阈值化参数：
        - 0-1之间：按比例保留最强边（稀疏度）
        - >=1：绝对值阈值
        - None：不阈值化
    weighted : bool, default=True
        是否使用加权图。

    Returns
    -------
    metrics : dict
        包含以下指标：
        - n_nodes, n_edges, density
        - global_clustering, global_efficiency, local_efficiency
        - characteristic_path_length
        - modularity, n_communities
        - small_world_sigma, small_world_omega
        - degree_centrality_mean, betweenness_centrality_mean
        - node_degree, node_betweenness（如果提供ch_names）

    Raises
    ------
    ImportError
        当networkx未安装时抛出。
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("图论指标需要networkx: pip install networkx")

    n_nodes = connectivity_matrix.shape[0]

    # 阈值化
    if threshold is not None:
        if isinstance(threshold, float) and threshold < 1:
            # 按比例保留最强边
            flat = connectivity_matrix[np.triu_indices_from(connectivity_matrix, k=1)]
            k = int(len(flat) * threshold)
            thresh_val = np.sort(flat)[-k] if k > 0 else np.max(flat)
            adj = connectivity_matrix * (connectivity_matrix >= thresh_val)
        else:
            # 绝对值阈值
            adj = connectivity_matrix * (connectivity_matrix >= threshold)
    else:
        adj = connectivity_matrix.copy()

    # 构建图
    if weighted:
        G = nx.from_numpy_array(adj)
    else:
        G = nx.from_numpy_array((adj > 0).astype(int))

    metrics = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'threshold': threshold,
        'weighted': weighted
    }

    if G.number_of_edges() > 0:
        # 聚类系数
        metrics['global_clustering'] = nx.average_clustering(G)

        # 效率
        metrics['global_efficiency'] = nx.global_efficiency(G)
        metrics['local_efficiency'] = nx.local_efficiency(G)

        # 特征路径长度
        try:
            if weighted:
                path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
                total = 0
                pairs = 0
                for s in path_lengths:
                    for t, l in path_lengths[s].items():
                        if s != t:
                            total += l
                            pairs += 1
                metrics['characteristic_path_length'] = total / pairs if pairs > 0 else np.inf
            else:
                metrics['characteristic_path_length'] = nx.average_shortest_path_length(G)
        except (nx.NetworkXError, ZeroDivisionError):
            metrics['characteristic_path_length'] = np.inf

        # 模块化
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = greedy_modularity_communities(G, weight='weight' if weighted else None)
            metrics['modularity'] = nx.community.modularity(G, communities)
            metrics['n_communities'] = len(communities)
        except:
            metrics['modularity'] = np.nan
            metrics['n_communities'] = np.nan

        # 小世界性
        try:
            # 生成随机图
            G_rand = nx.erdos_renyi_graph(n_nodes, metrics['density'])
            C_rand = nx.average_clustering(G_rand)
            try:
                L_rand = nx.average_shortest_path_length(G_rand)
            except:
                L_rand = np.inf

            C = metrics['global_clustering']
            L = metrics['characteristic_path_length']

            gamma = C / C_rand if C_rand > 0 else 1
            lambda_ = L / L_rand if L_rand > 0 and L_rand != np.inf else 1
            metrics['small_world_sigma'] = gamma / lambda_ if lambda_ != 0 else np.inf

            # omega指数（-1到1，越接近1越小世界）
            try:
                L_lattice = L * 2  # 近似
                omega = (L_rand / L) - (C / C_rand)
                metrics['small_world_omega'] = omega
            except:
                pass
        except:
            metrics['small_world_sigma'] = np.nan

        # 节点中心性
        deg_cen = nx.degree_centrality(G)
        metrics['degree_centrality_mean'] = np.mean(list(deg_cen.values()))

        try:
            bet_cen = nx.betweenness_centrality(G, weight='weight' if weighted else None)
            metrics['betweenness_centrality_mean'] = np.mean(list(bet_cen.values()))

            # 按通道名存储
            if ch_names is not None:
                if len(ch_names) == n_nodes:
                    metrics['node_degree'] = dict(zip(ch_names, deg_cen.values()))
                    metrics['node_betweenness'] = dict(zip(ch_names, bet_cen.values()))
        except:
            pass

    return metrics


# ============================================================================
# 5. 网络统计推断
# ============================================================================

def network_based_statistic(data_dict,
                            condition1, condition2,
                            modality=DEFAULT_MODALITY,
                            method='pli',
                            freq_band=None,
                            n_permutations=1000,
                            threshold_t=2.5,
                            alpha=0.05,
                            use_existing_epoch=True):
    """
    网络基于统计量（Network-Based Statistic, NBS）。

    Parameters
    ----------
    data_dict : dict
        四层嵌套结构。
    condition1, condition2 : list
        两组条件的标签。
    modality : str, default='EEG'
        分析模态。
    method : str, default='pli'
        连接计算方法。
    freq_band : list of two floats, optional
        频带范围。
    n_permutations : int, default=1000
        置换次数。
    threshold_t : float, default=2.5
        边水平t统计量阈值。
    alpha : float, default=0.05
        簇水平显著性阈值。
    use_existing_epoch : bool, default=True
        是否使用已存在的epoch数据。

    Returns
    -------
    data_dict : dict
        更新后的数据字典，结果存入：
        data_dict['processed']['statistics']['connectivity']['nbs']

    Notes
    -----
    算法原理：
        1. 为每个试次计算连接矩阵
        2. 逐边t检验，得到t统计量图
        3. 阈值化形成超阈值图
        4. 识别连通分量
        5. 置换检验构建零分布
        6. 计算各分量的显著性

    参考文献：
        Zalesky et al. (2010) NeuroImage
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("NBS需要networkx: pip install networkx")

    # 1. 获取epoch数据
    epochs, labels = _get_epoch_data(data_dict, modality, use_existing_epoch)

    # 2. 按条件分割
    labels = np.array(labels)
    mask1 = np.isin(labels, condition1)
    mask2 = np.isin(labels, condition2)

    epochs1 = epochs[mask1]
    epochs2 = epochs[mask2]

    n1, n2 = len(epochs1), len(epochs2)
    n_channels = epochs.shape[1]
    srate = data_dict['signal'][modality]['sampling_rate']
    ch_names = data_dict['signal'][modality]['channel_names']

    # 3. 为每个试次计算连接矩阵
    def _compute_conn_batch(epoch_set):
        conn_list = []
        for ep in epoch_set:
            # 临时数据字典
            temp_dict = {
                'signal': {
                    modality: {
                        'data': ep,
                        'sampling_rate': srate,
                        'channel_names': ch_names
                    }
                }
            }
            conn, _ = functional_connectivity(
                temp_dict,
                modality=modality,
                method=method,
                freq_band=freq_band,
                epoch_based=False,
                use_existing_epoch=False,
                save_to_dict=False
            )
            conn_list.append(conn)
        return np.array(conn_list)

    logger.info(f"   [NBS] 计算{len(epochs1)}个试次的条件1连接矩阵...")
    conn1 = _compute_conn_batch(epochs1)
    logger.info(f"   [NBS] 计算{len(epochs2)}个试次的条件2连接矩阵...")
    conn2 = _compute_conn_batch(epochs2)

    # 4. 逐边t检验
    t_matrix = np.zeros((n_channels, n_channels))
    p_matrix = np.ones((n_channels, n_channels))

    upper_tri = np.triu_indices(n_channels, k=1)
    for idx in range(len(upper_tri[0])):
        i, j = upper_tri[0][idx], upper_tri[1][idx]
        t, p = stats.ttest_ind(conn1[:, i, j], conn2[:, i, j])
        t_matrix[i, j] = t
        t_matrix[j, i] = t
        p_matrix[i, j] = p
        p_matrix[j, i] = p

    # 5. 阈值化构建超阈值图
    adj_thresh = (np.abs(t_matrix) > threshold_t).astype(int)
    np.fill_diagonal(adj_thresh, 0)

    # 6. 识别连通分量
    G = nx.from_numpy_array(adj_thresh)
    components = list(nx.connected_components(G))
    comp_sizes = [len(c) for c in components]

    # 7. 置换检验
    combined = np.concatenate([conn1, conn2], axis=0)
    perm_max_sizes = []

    logger.info(f"   [NBS] 进行{n_permutations}次置换检验...")
    for perm_idx in range(n_permutations):
        if perm_idx % 200 == 0 and perm_idx > 0:
            logger.info(f"     已完成{perm_idx}/{n_permutations}次置换")

        perm_idx_vec = np.random.permutation(len(combined))
        perm1_idx = perm_idx_vec[:n1]
        perm2_idx = perm_idx_vec[n1:]

        perm_t = np.zeros((n_channels, n_channels))
        for idx in range(len(upper_tri[0])):
            i, j = upper_tri[0][idx], upper_tri[1][idx]
            t, _ = stats.ttest_ind(
                combined[perm1_idx, i, j],
                combined[perm2_idx, i, j]
            )
            perm_t[i, j] = t
            perm_t[j, i] = t

        perm_adj = (np.abs(perm_t) > threshold_t).astype(int)
        np.fill_diagonal(perm_adj, 0)

        G_perm = nx.from_numpy_array(perm_adj)
        perm_comps = list(nx.connected_components(G_perm))
        if perm_comps:
            perm_max_sizes.append(max([len(c) for c in perm_comps]))
        else:
            perm_max_sizes.append(0)

    # 8. 计算分量显著性
    comp_pvals = []
    sig_comps = []

    for idx, size in enumerate(comp_sizes):
        p_val = np.mean(np.array(perm_max_sizes) >= size)
        comp_pvals.append(p_val)
        if p_val < alpha:
            sig_comps.append(idx)

    # 9. 结果封装
    result = {
        'condition1': condition1,
        'condition2': condition2,
        'method': method,
        'freq_band': freq_band,
        'threshold_t': threshold_t,
        'n_permutations': n_permutations,
        'n_components': len(components),
        'component_sizes': comp_sizes,
        'component_p_values': comp_pvals,
        'significant_components': sig_comps,
        't_matrix': t_matrix,
        'p_matrix': p_matrix,
        'channels': ch_names,
        'modality': modality
    }

    # 10. 写入data_dict
    if 'processed' not in data_dict:
        data_dict['processed'] = {}
    if 'statistics' not in data_dict['processed']:
        data_dict['processed']['statistics'] = {}
    if 'connectivity' not in data_dict['processed']['statistics']:
        data_dict['processed']['statistics']['connectivity'] = {}

    data_dict['processed']['statistics']['connectivity']['nbs'] = result

    n_sig = len(sig_comps)
    logger.info(f"   [NBS] 完成: 发现{len(components)}个分量，{n_sig}个显著")

    return data_dict


def network_permutation_test(data_dict,
                            condition1, condition2,
                            modality=DEFAULT_MODALITY,
                            metric='global_efficiency',
                            n_permutations=5000,
                            use_existing_epoch=True):
    """
    基于置换检验的网络指标比较。

    Parameters
    ----------
    data_dict : dict
        四层嵌套结构。
    condition1, condition2 : list
        两组条件的标签。
    modality : str, default='EEG'
        分析模态。
    metric : str, default='global_efficiency'
        图论指标名称。
    n_permutations : int, default=5000
        置换次数。
    use_existing_epoch : bool, default=True
        是否使用已存在的epoch数据。

    Returns
    -------
    result : dict
        包含观测差异、p值、各组均值标准差等。
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("网络指标比较需要networkx: pip install networkx")

    # 1. 获取epoch数据
    epochs, labels = _get_epoch_data(data_dict, modality, use_existing_epoch)

    # 2. 按条件分割
    labels = np.array(labels)
    mask1 = np.isin(labels, condition1)
    mask2 = np.isin(labels, condition2)

    epochs1 = epochs[mask1]
    epochs2 = epochs[mask2]

    srate = data_dict['signal'][modality]['sampling_rate']
    ch_names = data_dict['signal'][modality]['channel_names']

    # 3. 计算各组网络指标分布
    def _compute_metric_batch(epoch_set):
        metrics = []
        for ep in epoch_set:
            temp_dict = {
                'signal': {
                    modality: {
                        'data': ep,
                        'sampling_rate': srate,
                        'channel_names': ch_names
                    }
                }
            }
            conn, _ = functional_connectivity(
                temp_dict,
                modality=modality,
                method='pli',
                freq_band=None,
                epoch_based=False,
                use_existing_epoch=False,
                save_to_dict=False
            )
            g = graph_theory_metrics(conn, threshold=0.3)
            metrics.append(g.get(metric, 0))
        return np.array(metrics)

    metrics1 = _compute_metric_batch(epochs1)
    metrics2 = _compute_metric_batch(epochs2)

    # 4. 观测差异
    observed_diff = np.mean(metrics1) - np.mean(metrics2)

    # 5. 置换检验
    combined = np.concatenate([metrics1, metrics2])
    n1 = len(metrics1)
    perm_diffs = []

    for _ in range(n_permutations):
        perm = np.random.permutation(combined)
        diff = np.mean(perm[:n1]) - np.mean(perm[n1:])
        perm_diffs.append(diff)

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    # 6. 结果封装
    result = {
        'metric': metric,
        'observed_difference': observed_diff,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'mean_cond1': np.mean(metrics1),
        'mean_cond2': np.mean(metrics2),
        'std_cond1': np.std(metrics1),
        'std_cond2': np.std(metrics2),
        'n_trials_cond1': len(metrics1),
        'n_trials_cond2': len(metrics2)
    }

    return result