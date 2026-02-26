# -*- coding: utf-8 -*-
"""
EEG Feature Extraction Library (Optimized for NumPy 2.4.0)
===========================================================
文件名: eeg_features.py
描述: 
    该模块包含专门用于EEG信号的特征提取算法。
    继承自通用特征提取器，并添加EEG特有的节律波段功率、ERP、功能连接性和空间特征。
    适配 NumPy 2.4.0，修复 trapz 兼容性问题，增强连接性特征的鲁棒性。

依赖:
    继承自 common_features.py
    pip install numpy scipy pywt networkx

作者: Wan-Jingyu
版本: 1.1.0
"""

import numpy as np
import scipy.stats
import scipy.signal
import scipy.spatial
import warnings
from typing import List, Tuple, Dict, Optional
import networkx as nx
from core.processing.feature_extraction.common_features import CommonFeatureExtractor

class EEGFeatureExtractor(CommonFeatureExtractor):
    """
    EEG特征提取器类。
    继承通用特征提取器，并添加EEG特有的特征提取方法。
    """
    
    def __init__(self, fs: float, channel_names: List[str] = None, 
                 channel_locations: np.ndarray = None):
        """
        初始化EEG特征提取器。
        
        Args:
            fs (float): 信号的采样频率 (Sampling Frequency)，单位 Hz。
            channel_names (List[str]): 通道名称列表，用于连接性分析和空间特征。
            channel_locations (np.ndarray): 通道位置坐标，形状为 (n_channels, 2) 或 (n_channels, 3)。
        """
        super().__init__(fs)
        self.channel_names = channel_names
        self.channel_locations = channel_locations
        self.n_channels = len(channel_names) if channel_names else None
        
        # 定义EEG标准频带
        self.band_definitions = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45),
            'low_gamma': (30, 50),
            'high_gamma': (50, 100)
        }
    
    # =========================================================================
    # 1. 生理频带功率特征 (Band Power Features)
    # =========================================================================
    
    def extract_band_powers(self, data: np.ndarray, 
                           bands: List[str] = None,
                           relative: bool = True,
                           ratio_pairs: List[Tuple[str, str]] = None) -> Dict:
        """
        提取EEG生理频带功率特征。
        """
        if bands is None:
            bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
            
        # 确保数据是二维数组
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        n_channels, n_samples = data.shape
        features = {}
        
        for ch_idx in range(n_channels):
            channel_signal = data[ch_idx, :]
            channel_name = self.channel_names[ch_idx] if self.channel_names else f"ch{ch_idx}"
            
            # 计算功率谱密度
            nperseg = min(256, n_samples)
            freqs, psd = scipy.signal.welch(channel_signal, fs=self.fs, nperseg=nperseg)
            
            # 总功率
            total_power = np.sum(psd)
            
            for band in bands:
                if band in self.band_definitions:
                    low_freq, high_freq = self.band_definitions[band]
                    
                    # 找到频带内的频率索引
                    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    
                    if np.any(band_mask):
                        # 绝对功率
                        abs_power = np.sum(psd[band_mask])
                        features[f'{channel_name}_{band}_abs_power'] = abs_power
                        
                        # 相对功率
                        if relative:
                            rel_power = abs_power / (total_power + 1e-10)
                            features[f'{channel_name}_{band}_rel_power'] = rel_power
                
                # 计算频带功率比
                if ratio_pairs:
                    for band1, band2 in ratio_pairs:
                        if (band1 in self.band_definitions and band2 in self.band_definitions and
                            f'{channel_name}_{band1}_abs_power' in features and
                            f'{channel_name}_{band2}_abs_power' in features):
                            
                            power1 = features[f'{channel_name}_{band1}_abs_power']
                            power2 = features[f'{channel_name}_{band2}_abs_power']
                            
                            if power2 > 0:
                                ratio = power1 / power2
                                features[f'{channel_name}_{band1}_{band2}_ratio'] = ratio
        
        return features
    
    # =========================================================================
    # 2. ERP特征 (Event-Related Potential Features) - 修复 trapz 兼容性
    # =========================================================================
    
    def extract_erp_features(self, data: np.ndarray, 
                            event_times: List[int],
                            window_pre: float = 0.1,
                            window_post: float = 0.8,
                            baseline_correction: bool = True,
                            baseline_window: Tuple[float, float] = (-0.1, 0.0)) -> Dict:
        """
        提取事件相关电位(ERP)特征。
        使用 np.trapezoid 替代已移除的 np.trapz。
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        n_channels, n_samples = data.shape
        features = {}
        
        # 将时间窗口转换为样本点数
        pre_samples = int(window_pre * self.fs)
        post_samples = int(window_post * self.fs)
        baseline_pre = int(baseline_window[0] * self.fs)
        baseline_post = int(baseline_window[1] * self.fs)
        
        # 存储所有事件的ERP片段
        all_epochs = []
        
        for event_time in event_times:
            start_idx = event_time - pre_samples
            end_idx = event_time + post_samples
            
            if start_idx >= 0 and end_idx <= n_samples:
                epoch = data[:, start_idx:end_idx]
                all_epochs.append(epoch)
        
        if not all_epochs:
            warnings.warn("No valid epochs found for ERP extraction")
            return features
            
        # 平均所有epoch得到ERP
        erp = np.mean(all_epochs, axis=0)  # (n_channels, epoch_length)
        
        # 基线校正
        if baseline_correction:
            baseline_start = pre_samples + baseline_pre
            baseline_end = pre_samples + baseline_post
            
            if baseline_end > baseline_start:
                for ch_idx in range(n_channels):
                    baseline = np.mean(erp[ch_idx, baseline_start:baseline_end])
                    erp[ch_idx, :] = erp[ch_idx, :] - baseline
        
        # 提取ERP特征
        time_vector = np.arange(-pre_samples, post_samples) / self.fs * 1000  # 毫秒
        
        # 定义ERP成分的时间窗口(毫秒)
        erp_components = {
            'N1': (80, 120),
            'P2': (150, 250),
            'N2': (200, 350),
            'P3': (300, 500),
            'N4': (350, 450),
            'LPP': (500, 800),
        }
        
        for ch_idx in range(n_channels):
            channel_name = self.channel_names[ch_idx] if self.channel_names else f"ch{ch_idx}"
            channel_erp = erp[ch_idx, :]
            
            for component, (start_ms, end_ms) in erp_components.items():
                start_idx = np.argmin(np.abs(time_vector - start_ms))
                end_idx = np.argmin(np.abs(time_vector - end_ms))
                
                if start_idx < end_idx:
                    window_data = channel_erp[start_idx:end_idx]
                    
                    # 幅值特征
                    if 'N' in component:
                        amplitude = np.min(window_data)
                        latency_idx = np.argmin(window_data)
                    else:
                        amplitude = np.max(window_data)
                        latency_idx = np.argmax(window_data)
                    
                    latency = time_vector[start_idx + latency_idx]
                    
                    features[f'{channel_name}_{component}_amplitude'] = amplitude
                    features[f'{channel_name}_{component}_latency'] = latency
                    
                    # 面积特征 - 使用 numpy.trapezoid (NumPy 2.0+)
                    # 时间间隔（毫秒转换为秒后乘以采样率？但面积只需数值积分）
                    # 这里直接对幅值积分，默认x间隔为1（样本点），若需真实面积可乘以时间步长
                    dx = 1000 / self.fs  # 每个样本点的毫秒数
                    area = np.trapezoid(np.abs(window_data), dx=dx)
                    features[f'{channel_name}_{component}_area'] = area
        
        return features
    
    # =========================================================================
    # 3. 功能连接性特征 (Functional Connectivity Features) - 增强鲁棒性
    # =========================================================================
    
    def compute_connectivity_features(self, data: np.ndarray,
                                     method: str = 'coherence',
                                     freq_band: Tuple[float, float] = None,
                                     symmetric: bool = True,
                                     verbose: bool = False) -> Dict:
        """
        计算EEG通道间的功能连接性。
        
        Args:
            data (np.ndarray): EEG信号，形状为 (n_channels, n_samples)
            method (str): 连接性计算方法，可选 'coherence', 'plv', 'pli'
            freq_band (Tuple[float, float]): 计算连接性的频带范围
            symmetric (bool): 是否返回对称矩阵
            verbose (bool): 是否打印调试信息
            
        Returns:
            Dict: 包含连接性矩阵和统计特征的字典
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        n_channels, n_samples = data.shape
        features = {}
        
        if n_channels < 2:
            warnings.warn("At least 2 channels required for connectivity analysis")
            return features
        
        # 初始化连接矩阵
        conn_matrix = np.zeros((n_channels, n_channels))
        
        try:
            # 根据选择的方法计算连接性
            if method == 'coherence':
                conn_matrix = self._compute_coherence_matrix(data, freq_band, verbose)
            elif method == 'plv':
                conn_matrix = self._compute_plv_matrix(data, freq_band, verbose)
            elif method == 'pli':
                conn_matrix = self._compute_pli_matrix(data, freq_band, verbose)
            else:
                warnings.warn(f"Unsupported connectivity method: {method}")
                return features
        except Exception as e:
            warnings.warn(f"Error computing connectivity matrix ({method}): {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return features
        
        # 确保对称性
        if symmetric:
            conn_matrix = (conn_matrix + conn_matrix.T) / 2
            np.fill_diagonal(conn_matrix, 1.0)
        
        # 存储连接矩阵
        features[f'connectivity_matrix_{method}'] = conn_matrix
        
        # 计算连接矩阵的统计特征
        upper_tri = conn_matrix[np.triu_indices(n_channels, k=1)]
        
        if len(upper_tri) > 0:
            features[f'connectivity_{method}_mean'] = np.mean(upper_tri)
            features[f'connectivity_{method}_std'] = np.std(upper_tri)
            features[f'connectivity_{method}_max'] = np.max(upper_tri)
            features[f'connectivity_{method}_min'] = np.min(upper_tri)
            
            # 连接密度
            threshold = np.median(upper_tri)
            density = np.sum(upper_tri > threshold) / len(upper_tri)
            features[f'connectivity_{method}_density'] = density
        
        return features
    
    def _compute_coherence_matrix(self, data: np.ndarray, 
                                 freq_band: Tuple[float, float] = None,
                                 verbose: bool = False) -> np.ndarray:
        """计算相干性矩阵"""
        n_channels, n_samples = data.shape
        coh_matrix = np.zeros((n_channels, n_channels))
        
        # 自动选择窗口长度
        nperseg = min(256, n_samples // 4)
        if nperseg < 32:
            nperseg = n_samples // 2
        if nperseg < 8:
            warnings.warn(f"Signal too short for coherence (n_samples={n_samples})")
            return coh_matrix
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                try:
                    f, Cxy = scipy.signal.coherence(data[i], data[j], 
                                                   fs=self.fs, 
                                                   nperseg=nperseg)
                    
                    if freq_band:
                        low_freq, high_freq = freq_band
                        band_mask = (f >= low_freq) & (f <= high_freq)
                        if np.any(band_mask):
                            coh_value = np.mean(Cxy[band_mask])
                        else:
                            coh_value = 0
                    else:
                        coh_value = np.mean(Cxy)
                    
                    coh_matrix[i, j] = coh_value
                    coh_matrix[j, i] = coh_value
                except Exception as e:
                    if verbose:
                        print(f"Coherence error between ch{i} and ch{j}: {e}")
                    coh_matrix[i, j] = 0
                    coh_matrix[j, i] = 0
        
        return coh_matrix
    
    def _compute_plv_matrix(self, data: np.ndarray,
                           freq_band: Tuple[float, float] = None,
                           verbose: bool = False) -> np.ndarray:
        """计算锁相值矩阵"""
        n_channels, n_samples = data.shape
        plv_matrix = np.zeros((n_channels, n_channels))
        
        # 计算希尔伯特变换
        analytic_signals = []
        for i in range(n_channels):
            analytic = scipy.signal.hilbert(data[i])
            analytic_signals.append(analytic)
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                try:
                    phase_i = np.angle(analytic_signals[i])
                    phase_j = np.angle(analytic_signals[j])
                    phase_diff = phase_i - phase_j
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    plv_matrix[i, j] = plv
                    plv_matrix[j, i] = plv
                except Exception as e:
                    if verbose:
                        print(f"PLV error between ch{i} and ch{j}: {e}")
                    plv_matrix[i, j] = 0
                    plv_matrix[j, i] = 0
        
        return plv_matrix
    
    def _compute_pli_matrix(self, data: np.ndarray,
                           freq_band: Tuple[float, float] = None,
                           verbose: bool = False) -> np.ndarray:
        """计算相位滞后指数矩阵"""
        n_channels, n_samples = data.shape
        pli_matrix = np.zeros((n_channels, n_channels))
        
        analytic_signals = []
        for i in range(n_channels):
            analytic = scipy.signal.hilbert(data[i])
            analytic_signals.append(analytic)
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                try:
                    phase_i = np.angle(analytic_signals[i])
                    phase_j = np.angle(analytic_signals[j])
                    phase_diff = phase_i - phase_j
                    pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                    pli_matrix[i, j] = pli
                    pli_matrix[j, i] = pli
                except Exception as e:
                    if verbose:
                        print(f"PLI error between ch{i} and ch{j}: {e}")
                    pli_matrix[i, j] = 0
                    pli_matrix[j, i] = 0
        
        return pli_matrix
    
    # =========================================================================
    # 4. 图论特征 (Graph Theory Features)
    # =========================================================================
    
    def compute_graph_features(self, connectivity_matrix: np.ndarray,
                              threshold_type: str = 'density',
                              threshold_value: float = 0.2,
                              weighted: bool = True) -> Dict:
        """从连接性矩阵中提取图论特征"""
        n_channels = connectivity_matrix.shape[0]
        features = {}
        
        # 创建邻接矩阵
        if threshold_type == 'density':
            threshold = np.percentile(connectivity_matrix, 
                                     (1 - threshold_value) * 100)
            adj_matrix = (connectivity_matrix > threshold).astype(float)
        else:
            adj_matrix = (connectivity_matrix > threshold_value).astype(float)
        
        # 创建图对象
        if weighted:
            G = nx.from_numpy_array(connectivity_matrix)
        else:
            G = nx.from_numpy_array(adj_matrix)
        
        # 基本图属性
        features['graph_nodes'] = G.number_of_nodes()
        features['graph_edges'] = G.number_of_edges()
        
        # 度特征
        if n_channels > 0:
            degrees = [d for _, d in G.degree()]
            features['mean_degree'] = np.mean(degrees)
            features['max_degree'] = np.max(degrees)
            features['degree_std'] = np.std(degrees)
        
        # 聚类系数
        clustering_coeffs = nx.clustering(G, weight='weight' if weighted else None)
        features['mean_clustering'] = np.mean(list(clustering_coeffs.values()))
        
        # 特征路径长度（仅对连通图）
        if nx.is_connected(G):
            if weighted:
                path_lengths = dict(nx.all_pairs_dijkstra_path_length(G))
            else:
                path_lengths = dict(nx.all_pairs_shortest_path_length(G))
            
            all_paths = []
            for source in path_lengths:
                for target, length in path_lengths[source].items():
                    if source != target:
                        all_paths.append(length)
            
            if all_paths:
                features['characteristic_path_length'] = np.mean(all_paths)
                features['global_efficiency'] = 1 / features['characteristic_path_length']
        
        # 中心性度量
        if weighted:
            betweenness = nx.betweenness_centrality(G, weight='weight')
        else:
            betweenness = nx.betweenness_centrality(G)
        features['mean_betweenness'] = np.mean(list(betweenness.values()))
        
        # 小世界属性（仅当节点数 > 10）
        if n_channels > 10:
            try:
                random_G = nx.erdos_renyi_graph(n_channels, threshold_value)
                random_clustering = nx.average_clustering(random_G)
                if nx.is_connected(random_G):
                    random_path_length = nx.average_shortest_path_length(random_G)
                    sigma = (features['mean_clustering'] / random_clustering) / \
                            (features['characteristic_path_length'] / random_path_length)
                    features['small_world_sigma'] = sigma
            except:
                pass
        
        return features
    
    # =========================================================================
    # 5. 空间特征 (Spatial/Topographic Features)
    # =========================================================================
    
    def compute_topographic_features(self, data: np.ndarray,
                                    freq_band: Tuple[float, float] = (8, 13),
                                    method: str = 'power') -> Dict:
        """计算地形图空间特征"""
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        n_channels, n_samples = data.shape
        features = {}
        
        if self.channel_locations is None or len(self.channel_locations) != n_channels:
            warnings.warn("Channel locations not provided or mismatch with data")
            return features
        
        # 计算每个通道的特征值
        channel_features = np.zeros(n_channels)
        
        if method == 'power':
            for ch_idx in range(n_channels):
                nperseg = min(256, n_samples)
                freqs, psd = scipy.signal.welch(data[ch_idx], fs=self.fs, nperseg=nperseg)
                low_freq, high_freq = freq_band
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                if np.any(band_mask):
                    channel_features[ch_idx] = np.sum(psd[band_mask])
        elif method == 'coherence':
            coh_matrix = self._compute_coherence_matrix(data, freq_band)
            channel_features = np.mean(coh_matrix, axis=1)
        
        # 空间统计特征
        features['spatial_mean'] = np.mean(channel_features)
        features['spatial_std'] = np.std(channel_features)
        features['spatial_gradient'] = np.max(channel_features) - np.min(channel_features)
        
        # 空间熵
        if np.sum(channel_features) > 0:
            normalized = channel_features / np.sum(channel_features)
            spatial_entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
            features['spatial_entropy'] = spatial_entropy
        
        # 前-后梯度
        if self.channel_locations.shape[1] >= 2:
            x_coords = self.channel_locations[:, 0]
            median_x = np.median(x_coords)
            frontal_mask = x_coords < median_x
            posterior_mask = x_coords >= median_x
            if np.any(frontal_mask) and np.any(posterior_mask):
                frontal_mean = np.mean(channel_features[frontal_mask])
                posterior_mean = np.mean(channel_features[posterior_mask])
                features['frontal_posterior_ratio'] = frontal_mean / (posterior_mean + 1e-10)
                features['anterior_posterior_gradient'] = frontal_mean - posterior_mean
        
        # 左-右梯度
        if self.channel_locations.shape[1] >= 2:
            y_coords = self.channel_locations[:, 1]
            median_y = np.median(y_coords)
            left_mask = y_coords < median_y
            right_mask = y_coords >= median_y
            if np.any(left_mask) and np.any(right_mask):
                left_mean = np.mean(channel_features[left_mask])
                right_mean = np.mean(channel_features[right_mask])
                features['left_right_ratio'] = left_mean / (right_mean + 1e-10)
                features['hemispheric_asymmetry'] = np.abs(left_mean - right_mean)
        
        return features
    
    # =========================================================================
    # 6. 总接口 (Master Interface) - 增加 verbose 参数便于调试
    # =========================================================================
    
    def extract_all_eeg_features(self, data: np.ndarray,
                                extract_band_powers: bool = True,
                                extract_erp: bool = False,
                                event_times: List[int] = None,
                                extract_connectivity: bool = True,
                                extract_graph: bool = True,
                                extract_spatial: bool = True,
                                verbose: bool = False) -> Dict:
        """
        一键提取所有EEG特征。
        
        Args:
            data (np.ndarray): EEG信号，形状为 (n_channels, n_samples) 或 (n_samples,)
            extract_band_powers (bool): 是否提取频带功率特征
            extract_erp (bool): 是否提取ERP特征
            event_times (List[int]): ERP分析的事件时间点
            extract_connectivity (bool): 是否提取连接性特征
            extract_graph (bool): 是否提取图论特征
            extract_spatial (bool): 是否提取空间特征
            verbose (bool): 是否打印调试信息
            
        Returns:
            Dict: 包含所有EEG特征的字典
        """
        features = {}
        
        # 预处理检查
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # 1. 基础特征（从父类继承）
        try:
            for ch_idx in range(data.shape[0]):
                channel_signal = data[ch_idx, :]
                channel_name = self.channel_names[ch_idx] if self.channel_names else f"ch{ch_idx}"
                common_feats = super().extract_all_features(channel_signal)
                for key, value in common_feats.items():
                    features[f'{channel_name}_{key}'] = value
            if verbose:
                print(f"Common features extracted for {data.shape[0]} channels")
        except Exception as e:
            warnings.warn(f"Error computing common features: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
        
        # 2. EEG特有特征
        if extract_band_powers:
            try:
                band_feats = self.extract_band_powers(data)
                features.update(band_feats)
                if verbose:
                    print(f"Band power features extracted: {len(band_feats)}")
            except Exception as e:
                warnings.warn(f"Error computing band powers: {e}")
        
        if extract_erp and event_times:
            try:
                erp_feats = self.extract_erp_features(data, event_times)
                features.update(erp_feats)
                if verbose:
                    print(f"ERP features extracted: {len(erp_feats)}")
            except Exception as e:
                warnings.warn(f"Error computing ERP features: {e}")
        
        if extract_connectivity and data.shape[0] > 1:
            try:
                for band_name, (low_freq, high_freq) in self.band_definitions.items():
                    if high_freq <= 45:  # 常用频带
                        conn_feats = self.compute_connectivity_features(
                            data, method='coherence', freq_band=(low_freq, high_freq),
                            verbose=verbose
                        )
                        if conn_feats:
                            for key, value in conn_feats.items():
                                if key.startswith('connectivity_'):
                                    new_key = f'{key}_{band_name}'
                                    features[new_key] = value
                            if verbose:
                                print(f"Connectivity features ({band_name}): {len(conn_feats)}")
            except Exception as e:
                warnings.warn(f"Error computing connectivity features: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        if extract_graph and extract_connectivity:
            try:
                # 使用alpha频带的连接矩阵计算图论特征
                conn_matrix_key = 'connectivity_matrix_coherence_alpha'
                if conn_matrix_key in features:
                    graph_feats = self.compute_graph_features(features[conn_matrix_key])
                    features.update(graph_feats)
                    if verbose:
                        print(f"Graph features extracted: {len(graph_feats)}")
                else:
                    if verbose:
                        print(f"Warning: {conn_matrix_key} not found, skipping graph features")
            except Exception as e:
                warnings.warn(f"Error computing graph features: {e}")
        
        if extract_spatial and self.channel_locations is not None:
            try:
                for band_name, (low_freq, high_freq) in self.band_definitions.items():
                    if band_name in ['alpha', 'beta', 'theta']:
                        spatial_feats = self.compute_topographic_features(
                            data, freq_band=(low_freq, high_freq), method='power'
                        )
                        for key, value in spatial_feats.items():
                            features[f'{key}_{band_name}'] = value
                        if verbose:
                            print(f"Spatial features ({band_name}): {len(spatial_feats)}")
            except Exception as e:
                warnings.warn(f"Error computing spatial features: {e}")
        
        return features


# =============================================================================
# 使用示例 (Usage Example)
# =============================================================================

if __name__ == "__main__":
    # 模拟多通道EEG信号: 4通道，256 Hz，5秒长
    fs = 256
    n_channels = 4
    duration = 5
    n_samples = duration * fs
    
    t = np.linspace(0, duration, n_samples)
    
    # 生成模拟信号
    data = np.zeros((n_channels, n_samples))
    data[0] = 10 * np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 1, n_samples)
    data[1] = 5 * np.sin(2 * np.pi * 6 * t) + 3 * np.sin(2 * np.pi * 20 * t) + np.random.normal(0, 1, n_samples)
    data[2] = 2 * np.sin(2 * np.pi * 8 * t) + np.random.normal(0, 2, n_samples)
    data[3] = 3 * np.sin(2 * np.pi * 35 * t) + np.random.normal(0, 1, n_samples)
    
    # 通道名称和位置
    channel_names = ['Fz', 'Cz', 'Pz', 'Oz']
    channel_locations = np.array([
        [0, 0.5],
        [0, 0],
        [0, -0.5],
        [0, -1]
    ])
    
    # 实例化EEG特征提取器
    eeg_extractor = EEGFeatureExtractor(
        fs=fs,
        channel_names=channel_names,
        channel_locations=channel_locations
    )
    
    # 模拟事件时间点
    event_times = [int(fs * 1.0), int(fs * 2.5), int(fs * 4.0)]
    
    # 提取所有EEG特征（打开详细输出以便调试）
    print("开始提取EEG特征...")
    eeg_features = eeg_extractor.extract_all_eeg_features(
        data=data,
        extract_band_powers=True,
        extract_erp=True,
        event_times=event_times,
        extract_connectivity=True,
        extract_graph=True,
        extract_spatial=True,
        verbose=True  # 打开调试信息
    )
    
    # 打印部分结果
    print("\n--- EEG特征提取结果示例 ---")
    
    print("\n1. 频带功率特征:")
    for band in ['alpha', 'beta', 'theta']:
        for ch in ['Fz', 'Cz']:
            key = f"{ch}_{band}_rel_power"
            if key in eeg_features:
                print(f"  {key}: {eeg_features[key]:.4f}")
    
    print("\n2. 连接性特征:")
    conn_keys = [k for k in eeg_features.keys() if 'connectivity' in k and 'mean' in k]
    if conn_keys:
        for k in conn_keys[:5]:  # 只显示前5个
            print(f"  {k}: {eeg_features[k]:.4f}")
    else:
        print("  (无连接性特征)")
        # 显示所有以 connectivity 开头的键，帮助排查
        all_conn = [k for k in eeg_features.keys() if 'connectivity' in k]
        if all_conn:
            print(f"  相关键: {all_conn[:5]}...")
    
    print("\n3. 图论特征:")
    if 'mean_clustering' in eeg_features:
        print(f"  平均聚类系数: {eeg_features['mean_clustering']:.4f}")
    if 'characteristic_path_length' in eeg_features:
        print(f"  特征路径长度: {eeg_features['characteristic_path_length']:.4f}")
    
    print("\n4. 空间特征:")
    if 'spatial_entropy_alpha' in eeg_features:
        print(f"  Alpha频带空间熵: {eeg_features['spatial_entropy_alpha']:.4f}")
    if 'hemispheric_asymmetry_alpha' in eeg_features:
        print(f"  Alpha频带半球不对称性: {eeg_features['hemispheric_asymmetry_alpha']:.4f}")
    
    print(f"\n总计提取EEG特征数量: {len(eeg_features)}")