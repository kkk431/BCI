# -*- coding: utf-8 -*-
"""
Common Feature Extraction Library for Multimodal BCI
====================================================
文件名: common_features.py
描述: 
    该模块包含适用于 EEG, fNIRS, ECG, EMG 等生物信号的通用特征提取算法。
    包括时域、频域、时频域以及非线性动力学特征。
    所有算法均基于 numpy, scipy, pywt 实现，不依赖复杂的第三方黑盒库，
    便于后续移植和理解。

依赖:
    pip install numpy scipy PyWavelets

作者:XCT-byte
版本: 1.0.0
"""

import math
import numpy as np
import scipy.stats
import scipy.signal
import pywt
import warnings

class CommonFeatureExtractor:
    """
    通用特征提取器类。
    包含了时域、频域、时频域和非线性特征的提取方法。
    """

    def __init__(self, fs: float):
        """
        初始化特征提取器。

        Args:
            fs (float): 信号的采样频率 (Sampling Frequency)，单位 Hz。
                        这对频域特征和部分非线性特征至关重要。
        """
        self.fs = fs

    # =========================================================================
    # 1. 时域特征 (Time Domain Features)
    # =========================================================================

    def compute_time_domain_features(self, data: np.ndarray) -> dict:
        """
        提取基础时域统计特征。

        Args:
            data (np.ndarray): 1D 信号数组。

        Returns:
            dict: 包含时域特征的字典。
        """
        if len(data) == 0:
            return {}

        features = {}
        
        # --- 统计特征 ---
        features['mean'] = np.mean(data)
        features['std'] = np.std(data, ddof=1) # 样本标准差
        features['var'] = np.var(data, ddof=1)
        # 偏度 (Skewness): 衡量分布的非对称性
        features['skewness'] = scipy.stats.skew(data)
        # 峭度 (Kurtosis): 衡量分布的尾部厚度 (Fisher定义，正态分布为0)
        features['kurtosis'] = scipy.stats.kurtosis(data)
        
        # --- 幅值特征 ---
        features['max'] = np.max(data)
        features['min'] = np.min(data)
        features['peak_to_peak'] = features['max'] - features['min']
        # 均方根值 (RMS): 表征信号的有效能量
        features['rms'] = np.sqrt(np.mean(data**2))
        
        # --- 波形特征 ---
        # 形状因子 (Shape Factor): RMS / Mean Absolute Value
        mean_abs = np.mean(np.abs(data))
        features['shape_factor'] = features['rms'] / (mean_abs + 1e-10)
        # 脉冲因子 (Impulse Factor): Max / Mean Absolute Value
        features['impulse_factor'] = np.max(np.abs(data)) / (mean_abs + 1e-10)

        # --- Hjorth 参数 (常用于EEG) ---
        # Activity: 信号的方差
        activity = features['var']
        
        # Mobility: 一阶导数的标准差 / 信号的标准差
        # 一阶差分近似导数
        diff1 = np.diff(data)
        sigma_diff1 = np.std(diff1, ddof=1)
        mobility = sigma_diff1 / (features['std'] + 1e-10)
        
        # Complexity: Mobility of diff1 / Mobility of signal
        # 二阶差分
        diff2 = np.diff(diff1)
        sigma_diff2 = np.std(diff2, ddof=1)
        mobility_diff1 = sigma_diff2 / (sigma_diff1 + 1e-10)
        complexity = mobility_diff1 / (mobility + 1e-10)

        features['hjorth_activity'] = activity
        features['hjorth_mobility'] = mobility
        features['hjorth_complexity'] = complexity

        # --- 过零点率 (ZCR) ---
        # 计算信号穿过零轴的次数，通常需要先去直流(demean)或设定阈值
        centered_data = data - np.mean(data)
        # 符号变化的次数 / 信号长度
        zcr = ((centered_data[:-1] * centered_data[1:]) < 0).sum() / len(data)
        features['zero_crossing_rate'] = zcr

        return features

    # =========================================================================
    # 2. 频域特征 (Frequency Domain Features)
    # =========================================================================

    def compute_freq_domain_features(self, data: np.ndarray, nperseg=None) -> dict:
        """
        提取频域特征。主要基于功率谱密度 (PSD)。

        Args:
            data (np.ndarray): 1D 信号数组。
            nperseg (int): Welch 方法的窗口长度，默认为 None (自动调整)。

        Returns:
            dict: 包含频域特征的字典。
        """
        features = {}
        
        # 使用 Welch 方法估计功率谱密度 (PSD)，比直接 FFT 更平滑、方差更小
        if nperseg is None:
            nperseg = min(len(data), 256)
            
        freqs, psd = scipy.signal.welch(data, fs=self.fs, nperseg=nperseg)
        
        # 归一化 PSD 用于计算熵和矩
        psd_norm = psd / (np.sum(psd) + 1e-10)

        # --- 基础频域特征 ---
        # 总功率 (Total Power)
        features['total_power'] = np.sum(psd)
        
        # 峰值频率 (Peak Frequency): 能量最大的频率点
        features['peak_freq'] = freqs[np.argmax(psd)]

        # --- 频谱矩特征 ---
        # 频谱质心 (Spectral Centroid): 频谱能量分布的重心
        # Sum(f * P(f)) / Sum(P(f))
        spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
        features['spectral_centroid'] = spectral_centroid

        # 频谱方差/带宽 (Spectral Spread/Bandwidth)
        # Sum((f - centroid)^2 * P(f)) / Sum(P(f))
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * psd) / (np.sum(psd) + 1e-10))
        features['spectral_bandwidth'] = spectral_spread

        # --- 谱熵 (Spectral Entropy) ---
        # 衡量功率谱分布的平坦度/混乱度。白噪声熵最高，单一正弦波熵最低。
        # SE = -Sum(p * log2(p)) / log2(N_bins)
        se = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        # 归一化到 [0, 1]
        features['spectral_entropy'] = se / np.log2(len(psd_norm))

        # --- 频谱边缘频率 (Spectral Edge Frequency, SEF) ---
        # 低于该频率的能量占总能量的 x% (通常是 90% 或 95%)
        cutoff_percent = 0.95
        cumsum_psd = np.cumsum(psd)
        total_p = cumsum_psd[-1]
        # 找到累积能量首次超过阈值的索引
        idx = np.where(cumsum_psd >= total_p * cutoff_percent)[0]
        if len(idx) > 0:
            features[f'sef_{int(cutoff_percent*100)}'] = freqs[idx[0]]
        else:
            features[f'sef_{int(cutoff_percent*100)}'] = 0

        return features

    # =========================================================================
    # 3. 时频域特征 (Time-Frequency Domain Features - Wavelet)
    # =========================================================================

    def compute_wavelet_features(self, data: np.ndarray, wavelet='db4', level=4) -> dict:
        """
        提取小波变换特征。
        适用于非平稳信号 (EEG, EMG) 的瞬态分析。

        Args:
            data (np.ndarray): 1D 信号数组。
            wavelet (str): 小波基名称，如 'db4', 'sym8', 'coif1'。
            level (int): 分解层数。

        Returns:
            dict: 包含各层小波系数能量和熵的特征。
        """
        features = {}
        
        # 小波分解
        # coeffs 格式: [cA_n, cD_n, cD_n-1, ..., cD_1]
        # cA: 近似系数 (低频), cD: 细节系数 (高频)
        try:
            coeffs = pywt.wavedec(data, wavelet, level=level)
        except ValueError:
            # 如果信号太短无法进行指定层数的分解
            warnings.warn(f"Signal too short for wavelet level {level}. Reducing level.")
            max_level = pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len)
            coeffs = pywt.wavedec(data, wavelet, level=max_level)

        # 提取每一层的特征
        feature_names = ['cA' + str(level)] + ['cD' + str(i) for i in range(level, 0, -1)]
        
        for i, (coef, name) in enumerate(zip(coeffs, feature_names)):
            # 1. 能量 (Energy): 系数平方和
            energy = np.sum(coef ** 2)
            features[f'wavelet_{name}_energy'] = energy
            
            # 2. 能量占比 (Relative Energy)
            # 注：这里计算的是该子带能量占所有子带能量之和的比例
            # 也可以计算 log 能量
            features[f'wavelet_{name}_log_energy'] = np.log10(energy + 1e-10)

            # 3. 统计特征 (均值、标准差)
            features[f'wavelet_{name}_std'] = np.std(coef, ddof=1)
            
            # 4. 小波熵 (Wavelet Entropy) - 衡量该频带内能量分布的复杂性
            # 这里计算的是该子带系数分布的熵
            p = np.abs(coef) / (np.sum(np.abs(coef)) + 1e-10)
            entropy = -np.sum(p * np.log2(p + 1e-10))
            features[f'wavelet_{name}_entropy'] = entropy

        return features

    # =========================================================================
    # 4. 非线性动力学特征 (Nonlinear / Complexity Features)
    # =========================================================================

    def compute_nonlinear_features(self, data: np.ndarray) -> dict:
        """
        提取非线性/复杂性特征。
        注意：这些计算通常计算量较大，对于长信号建议分段处理。

        Args:
            data (np.ndarray): 1D 信号数组。

        Returns:
            dict: 包含熵和分形维数的字典。
        """
        features = {}
        
        # 1. 排列熵 (Permutation Entropy)
        # 衡量时间序列的顺序模式复杂性，计算速度较快，抗噪性好
        features['permutation_entropy'] = self._permutation_entropy(data, order=3, delay=1)
        
        # 2. 样本熵 (Sample Entropy)
        # 衡量信号产生的规则性。值越小，序列自我相似性越高。
        # 注意：为了效率，如果数据太长，建议降采样或切片
        if len(data) > 2000:
            # 简单的降采样以保证实时性，或者取中间段
            data_for_entropy = data[::2] 
        else:
            data_for_entropy = data
            
        features['sample_entropy'] = self._sample_entropy(data_for_entropy, m=2, r=0.2*np.std(data_for_entropy))

        # 3. Higuchi 分形维数 (HFD)
        # 描述信号曲线的粗糙度
        features['higuchi_fd'] = self._higuchi_fd(data, kmax=10)
        
        # 4. SVD 熵 (SVD Entropy) / 奇异值分解熵
        # 反映信号构建的特征矩阵的正交分量数量
        features['svd_entropy'] = self._svd_entropy(data, delay=1, embed_dim=10)

        return features

    # --- 非线性特征的内部实现方法 ---

    def _permutation_entropy(self, time_series, order=3, delay=1):
        """
        计算排列熵 (Permutation Entropy)。
        """
        x = np.array(time_series)
        hashmult = np.power(order, np.arange(order))
        
        # 创建嵌入矩阵
        # 使用 stride_tricks 高效生成滑动窗口
        n = len(x)
        if n < order * delay:
            return 0.0
            
        # 简单的嵌入实现
        sorted_idx = []
        for i in range(n - (order - 1) * delay):
            # 获取每个窗口的切片索引 [i, i+delay, i+2*delay...]
            window = [x[i + j * delay] for j in range(order)]
            # 获取排序后的索引模式 (argsort)
            sorted_idx.append(tuple(np.argsort(window)))
            
        # 统计每种模式出现的频率
        _, c = np.unique(sorted_idx, return_counts=True, axis=0)
        p = c / c.sum()
        pe = -np.multiply(p, np.log2(p)).sum()
        
        # 归一化
        return pe / np.log2(math.factorial(order))

    def _sample_entropy(self, L, m, r):
        """
        计算样本熵 (Sample Entropy)。
        算法复杂度 O(N^2)，已尽量使用 NumPy 向量化。
        
        Args:
            L: 信号序列
            m: 模板长度 (通常为 2)
            r: 容忍度 (通常为 0.2 * std)
        """
        N = len(L)
        B = 0.0
        A = 0.0
        
        # 将数据重构为矩阵以便向量化计算距离
        # xmi 是 m 维向量的集合
        xmi = np.array([L[i : i + m] for i in range(N - m)])
        # xmj 是 m+1 维向量的集合
        xmj = np.array([L[i : i + m + 1] for i in range(N - m - 1)])

        # 1. 计算 B: 匹配长度为 m 的模板数量
        # 自匹配不算 (i != j)，这里通过构建距离矩阵计算
        # 注意：为了性能，这里使用了简化版的双循环逻辑，也可用 cdist
        
        # 使用切比雪夫距离 (Chebyshev distance, max(|x_i - x_j|))
        # 简单实现：对于每个模板 i，计算它与其他模板 j 的距离
        for i in range(len(xmi)):
            # 排除自匹配，实际上 SampleEntropy 允许 i 和 j 的范围有重叠，
            # 但通常排除 i==j。此处计算所有 j != i
            # 向量化计算：xmi[i] 与 所有 xmi 的差的绝对值的最大值
            dist = np.max(np.abs(xmi - xmi[i]), axis=1)
            # 统计距离小于 r 的数量，减 1 是因为 dist 中包含自己与自己的距离(0)
            B += np.sum(dist <= r) - 1
            
        # 2. 计算 A: 匹配长度为 m+1 的模板数量
        for i in range(len(xmj)):
            dist = np.max(np.abs(xmj - xmj[i]), axis=1)
            A += np.sum(dist <= r) - 1
            
        if B == 0:
            return 0.0 # 避免除零
            
        return -np.log(A / B)

    def _higuchi_fd(self, x, kmax):
        """
        计算 Higuchi 分形维数 (HFD)。
        """
        x = np.asarray(x, dtype=np.float64)
        N = len(x)
        L = []
        x_indices = []
        
        for k in range(1, kmax + 1):
            Lk = []
            for m in range(0, k):
                # 构建子序列
                # indices: m, m+k, m+2k, ...
                idxs = np.arange(m, N, k)
                sub_series = x[idxs]
                n_elements = len(sub_series)
                
                # 计算长度
                # Sum |x(i) - x(i-1)|
                L_m_k = np.sum(np.abs(np.diff(sub_series)))
                
                # 归一化因子: (N - 1) / (floor((N - m - 1) / k) * k)
                norm_factor = (N - 1) / (np.floor((N - m - 1) / k) * k)
                
                L_m_k = (L_m_k * norm_factor) / k
                Lk.append(L_m_k)
                
            # 对当前 k 的所有 m 取平均
            L.append(np.mean(Lk))
            x_indices.append(k)
            
        # HFD 是 log(L(k)) 对 log(1/k) 回归直线的斜率
        # ln(L(k)) ~ -D * ln(k)
        # 所以我们对 log(L) 和 log(k) 做线性回归
        x_vals = np.log(1.0 / np.array(x_indices))
        y_vals = np.log(np.array(L))
        
        slope, _, _, _, _ = scipy.stats.linregress(x_vals, y_vals)
        return slope

    def _svd_entropy(self, x, delay, embed_dim):
        """
        奇异值分解熵 (SVD Entropy)。
        通过相空间重构和SVD分析信号复杂性。
        """
        x = np.array(x)
        N = len(x)
        
        # 1. 相空间重构 (Phase Space Reconstruction)
        # 构建轨迹矩阵
        if N < delay * embed_dim:
            return 0
            
        M = N - (embed_dim - 1) * delay
        # 构建矩阵 X (M x embed_dim)
        X = np.zeros((M, embed_dim))
        for i in range(embed_dim):
            X[:, i] = x[i*delay : i*delay + M]
            
        # 2. 奇异值分解
        # 只需要奇异值 S
        try:
            _, S, _ = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0
            
        # 3. 计算熵
        # 归一化奇异值
        S_norm = S / np.sum(S)
        # SVD Entropy
        entropy = -np.sum(S_norm * np.log2(S_norm + 1e-10))
        
        return entropy

    # =========================================================================
    # 5. 总接口 (Master Interface)
    # =========================================================================

    def extract_all_features(self, data: np.ndarray) -> dict:
        """
        一键提取所有类型的特征（时域、频域、非线性等）。
        
        Args:
            data (np.ndarray): 1D 信号数组。

        Returns:
            dict: 包含所有特征的扁平字典。
        """
        features = {}
        
        # 1. 预处理检查
        data = np.asarray(data)
        if np.isnan(data).any() or np.isinf(data).any():
            # 简单的清洗策略：用0或均值填充，这里选择将其设为0并警告
            warnings.warn("Input data contains NaN or Inf. Replacing with 0.")
            data = np.nan_to_num(data)

        # 2. 逐步提取
        try:
            features.update(self.compute_time_domain_features(data))
        except Exception as e:
            warnings.warn(f"Error computing time domain features: {e}")

        try:
            features.update(self.compute_freq_domain_features(data))
        except Exception as e:
            warnings.warn(f"Error computing frequency domain features: {e}")
            
        try:
            features.update(self.compute_wavelet_features(data))
        except Exception as e:
            warnings.warn(f"Error computing wavelet features: {e}")

        try:
            # 非线性特征计算较慢，可根据需求注释掉
            features.update(self.compute_nonlinear_features(data))
        except Exception as e:
            warnings.warn(f"Error computing nonlinear features: {e}")

        return features

# =============================================================================
# 使用示例 (Usage Example)
# =============================================================================

if __name__ == "__main__":
    # 模拟一段 EEG 信号: 256 Hz, 2秒长
    fs = 256
    t = np.linspace(0, 2, 2 * fs)
    # 构造信号：Alpha波(10Hz) + 少量Gamma波(40Hz) + 噪声
    signal = 5 * np.sin(2 * np.pi * 10 * t) + \
             1 * np.sin(2 * np.pi * 40 * t) + \
             np.random.normal(0, 1, len(t))

    # 1. 实例化提取器
    extractor = CommonFeatureExtractor(fs=fs)

    # 2. 提取特征
    print("开始提取特征...")
    feats = extractor.extract_all_features(signal)

    # 3. 打印部分结果
    print("\n--- 提取结果示例 ---")
    print(f"信号均值 (Mean): {feats.get('mean'):.4f}")
    print(f"均方根 (RMS): {feats.get('rms'):.4f}")
    print(f"Hjorth 复杂度: {feats.get('hjorth_complexity'):.4f}")
    print(f"频谱质心 (Hz): {feats.get('spectral_centroid'):.4f}")
    print(f"Alpha波段小波能量 (近似): {feats.get('wavelet_cD3_energy'):.4f}") # 256Hz下 cD3 约对应 16-32Hz, cD4 8-16Hz
    print(f"样本熵 (Sample Entropy): {feats.get('sample_entropy'):.4f}")
    print(f"Higuchi分形维数: {feats.get('higuchi_fd'):.4f}")
    
    print(f"\n总计提取特征数量: {len(feats)}")
