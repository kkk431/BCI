"""
connectivity_stats.py
四层结构 + 状态感知：
- 自动使用processed.epoch（优先）或动态提取
- 连接矩阵存入 processed.statistics.connectivity
- 图论指标存入 processed.statistics.connectivity.graph

优化项：
- NBS函数增加networkx可用性检查
- 增加动态连接预留接口
"""

import numpy as np
from scipy import signal, linalg, stats
from scipy.signal import welch, butter, filtfilt, hilbert
import warnings

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("networkx未安装，图论指标和NBS不可用")

DEFAULT_MODALITY = 'EEG'


def _prepare_epoch_data(data_dict, modality, use_existing_epoch=True):
    """准备epoch数据：优先使用已存epoch，否则动态提取"""
    # 优先使用已存在的epoch
    if use_existing_epoch and 'processed' in data_dict:
        proc = data_dict['processed']
        if 'epoch' in proc:
            epoch_data = proc['epoch']
            if 'data' in epoch_data:
                print(f"   [连接] 使用已分段epoch数据")
                return epoch_data['data'], epoch_data.get('labels', None)

    # 动态提取
    print(f"   [连接] 从连续信号提取epoch")
    from stat_inference import _get_epochs_from_signal
    epochs, labels = _get_epochs_from_signal(
        data_dict, modality, tmin=-1.0, tmax=3.0, baseline=None
    )
    return epochs, labels


def functional_connectivity(data_dict,
                            modality=DEFAULT_MODALITY,
                            method='pli',
                            freq_band=None,
                            epoch_based=True,
                            use_existing_epoch=True,
                            save_to_dict=True):
    """
    功能连接计算

    参数:
        method: 'coh', 'plv', 'pli', 'wpli', 'mi'
    """
    # 1. 获取epoch数据
    epochs, labels = _prepare_epoch_data(data_dict, modality, use_existing_epoch)
    n_epochs, n_channels, n_times = epochs.shape
    srate = data_dict['signal'][modality]['sampling_rate']
    ch_names = data_dict['signal'][modality]['channel_names']

    # 2. 逐trial计算连接矩阵
    conn_epochs = []

    for ep_idx in range(n_epochs):
        data = epochs[ep_idx]

        # 频带滤波
        if freq_band is not None:
            b, a = butter(4, freq_band, btype='band', fs=srate)
            data_filt = np.array([filtfilt(b, a, ch_data) for ch_data in data])
        else:
            data_filt = data

        # 初始化连接矩阵
        conn = np.zeros((n_channels, n_channels))

        if method == 'coh':
            # 相干性
            f, Cxy = signal.coherence(data_filt, fs=srate, nperseg=min(256, n_times // 4))
            if freq_band is not None:
                freq_mask = (f >= freq_band[0]) & (f <= freq_band[1])
                conn = Cxy[:, :, freq_mask].mean(axis=2)
            else:
                conn = Cxy.mean(axis=2)

        elif method == 'plv':
            # 锁相值
            analytic = hilbert(data_filt)
            phase = np.angle(analytic)
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    phase_diff = phase[i] - phase[j]
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    conn[i, j] = plv
                    conn[j, i] = plv

        elif method == 'pli':
            # 相位滞后指数
            analytic = hilbert(data_filt)
            phase = np.angle(analytic)
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    phase_diff = phase[i] - phase[j]
                    pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                    conn[i, j] = pli
                    conn[j, i] = pli

        elif method == 'wpli':
            # 加权相位滞后指数
            analytic = hilbert(data_filt)
            phase = np.angle(analytic)
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    phase_diff = phase[i] - phase[j]
                    imag_component = np.sin(phase_diff)
                    wpli = np.abs(np.mean(np.abs(imag_component) * np.sign(imag_component)))
                    conn[i, j] = wpli
                    conn[j, i] = wpli

        conn_epochs.append(conn)

    # 3. 聚合
    conn_epochs = np.array(conn_epochs)
    conn_mean = np.mean(conn_epochs, axis=0)

    # 4. 存储结果
    result = {
        'method': method,
        'freq_band': freq_band,
        'matrix': conn_mean,
        'matrices_epoch': conn_epochs if not epoch_based else None,
        'channels': ch_names,
        'n_epochs': n_epochs,
        'modality': modality
    }

    # 5. 写入data_dict
    if save_to_dict:
        if 'processed' not in data_dict:
            data_dict['processed'] = {}
        if 'statistics' not in data_dict['processed']:
            data_dict['processed']['statistics'] = {}
        if 'connectivity' not in data_dict['processed']['statistics']:
            data_dict['processed']['statistics']['connectivity'] = {}

        key = f"{method}_{freq_band[0]}-{freq_band[1]}Hz" if freq_band else method
        data_dict['processed']['statistics']['connectivity'][key] = result

        return data_dict
    else:
        return conn_mean, result


def graph_theory_metrics(connectivity_matrix, ch_names=None,
                         threshold=None, weighted=True):
    """
    图论指标计算

    参数:
        connectivity_matrix: (n_channels, n_channels) 连接矩阵
        ch_names: 通道名列表
        threshold: 阈值（绝对值）或稀疏度（比例）
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("请安装networkx: pip install networkx")

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
    G = nx.from_numpy_array(adj)

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
                # 加权图的最短路径
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
        except:
            metrics['characteristic_path_length'] = np.inf

        # 节点中心性（平均）
        deg_cen = nx.degree_centrality(G)
        metrics['degree_centrality_mean'] = np.mean(list(deg_cen.values()))

        try:
            bet_cen = nx.betweenness_centrality(G, weight='weight' if weighted else None)
            metrics['betweenness_centrality_mean'] = np.mean(list(bet_cen.values()))

            # 按通道名存储
            if ch_names is not None:
                metrics['node_degree'] = dict(zip(ch_names, deg_cen.values()))
                metrics['node_betweenness'] = dict(zip(ch_names, bet_cen.values()))
        except:
            pass

    return metrics


def network_based_statistic(data_dict,
                            condition1, condition2,
                            modality=DEFAULT_MODALITY,
                            method='pli',
                            freq_band=None,
                            n_permutations=1000,
                            threshold_t=2.5,
                            alpha=0.05):
    """
    网络基于统计量(NBS) - 识别差异连接子网

    注意: 依赖networkx
    """
    # 检查networkx可用性
    if not NETWORKX_AVAILABLE:
        raise ImportError("NBS需要networkx: pip install networkx")

    # 1. 获取epoch和标签
    epochs, labels = _prepare_epoch_data(data_dict, modality)

    mask1 = np.isin(labels, condition1)
    mask2 = np.isin(labels, condition2)

    epochs1 = epochs[mask1]
    epochs2 = epochs[mask2]

    n1, n2 = len(epochs1), len(epochs2)
    n_channels = epochs.shape[1]
    srate = data_dict['signal'][modality]['sampling_rate']

    # 2. 为每个trial计算连接矩阵
    def _compute_conn_for_epochs(epoch_set):
        conn_list = []
        for ep in epoch_set:
            temp_dict = {
                'signal': {modality: {
                    'data': ep,
                    'sampling_rate': srate,
                    'channel_names': data_dict['signal'][modality]['channel_names']
                }}
            }
            conn, _ = functional_connectivity(
                temp_dict, modality, method, freq_band,
                epoch_based=False, save_to_dict=False
            )
            conn_list.append(conn)
        return np.array(conn_list)

    conn1 = _compute_conn_for_epochs(epochs1)
    conn2 = _compute_conn_for_epochs(epochs2)

    # 3. 逐边t检验
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

    # 4. 阈值化构建超阈值图
    adj_thresh = (np.abs(t_matrix) > threshold_t).astype(int)
    np.fill_diagonal(adj_thresh, 0)

    # 5. 找连通分量
    G = nx.from_numpy_array(adj_thresh)
    components = list(nx.connected_components(G))
    comp_sizes = [len(c) for c in components]

    # 6. 置换检验
    combined = np.concatenate([conn1, conn2], axis=0)
    perm_max_sizes = []

    for _ in range(n_permutations):
        perm_idx = np.random.permutation(len(combined))
        perm1_idx = perm_idx[:n1]
        perm2_idx = perm_idx[n1:]

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

    # 7. 计算分量显著性
    comp_pvals = []
    sig_comps = []

    for idx, size in enumerate(comp_sizes):
        p_val = np.mean(np.array(perm_max_sizes) >= size)
        comp_pvals.append(p_val)
        if p_val < alpha:
            sig_comps.append(idx)

    # 8. 结果
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
        'p_matrix': p_matrix
    }

    # 9. 写入
    if 'processed' not in data_dict:
        data_dict['processed'] = {}
    if 'statistics' not in data_dict['processed']:
        data_dict['processed']['statistics'] = {}
    if 'connectivity' not in data_dict['processed']['statistics']:
        data_dict['processed']['statistics']['connectivity'] = {}

    data_dict['processed']['statistics']['connectivity']['nbs'] = result

    return data_dict