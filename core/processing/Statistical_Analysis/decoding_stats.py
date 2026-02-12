"""
decoding_stats.py
四层结构 + 状态感知：
- 特征存入 processed.feature
- 评估结果存入 processed.statistics.decoding
- 自动复用已存在特征

优化项：
- 增加sklearn缺失保护
- 统一导入风格
"""

import numpy as np
from scipy import linalg
import warnings

# 可选依赖：scikit-learn
try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn未安装，解码功能不可用")

DEFAULT_MODALITY = 'EEG'


def csp_features(data_dict,
                 condition1, condition2,
                 modality=DEFAULT_MODALITY,
                 n_components=4,
                 overwrite=False):
    """
    共同空间模式(CSP)特征提取
    结果存入 processed.feature
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("CSP需要scikit-learn: pip install scikit-learn")

    # 1. 检查是否已存在特征
    if not overwrite and 'processed' in data_dict:
        proc = data_dict['processed']
        if 'feature' in proc:
            feat = proc['feature']
            if feat.get('type') == 'CSP':
                print(f"   [CSP] 使用已存在特征")
                return data_dict

    # 2. 获取epoch数据
    from stat_inference import _get_epochs_from_signal
    epochs, labels = _get_epochs_from_signal(
        data_dict, modality, tmin=-0.5, tmax=2.0, baseline=None
    )

    # 3. 按条件分割
    mask1 = np.isin(labels, condition1)
    mask2 = np.isin(labels, condition2)

    X1 = epochs[mask1]  # (n1, n_channels, n_times)
    X2 = epochs[mask2]  # (n2, n_channels, n_times)

    n_channels = X1.shape[1]

    # 4. 计算协方差矩阵
    # 时间维度平均
    X1_mean = X1.mean(axis=2)
    X2_mean = X2.mean(axis=2)

    # 中心化
    X1_centered = X1_mean - X1_mean.mean(axis=0)
    X2_centered = X2_mean - X2_mean.mean(axis=0)

    # 协方差
    C1 = (X1_centered.T @ X1_centered) / (X1_centered.shape[0] - 1)
    C2 = (X2_centered.T @ X2_centered) / (X2_centered.shape[0] - 1)

    # 正则化
    C1 += 1e-6 * np.eye(n_channels)
    C2 += 1e-6 * np.eye(n_channels)

    # 5. 广义特征值分解
    try:
        eigvals, eigvecs = linalg.eig(C1, C1 + C2)
    except:
        eigvals, eigvecs = linalg.eig(C1, C1 + C2 + 1e-8 * np.eye(n_channels))

    # 6. 选择前n_components个滤波器
    idx = np.argsort(-np.real(eigvals))
    W = np.real(eigvecs[:, idx[:n_components]]).T

    # 7. 提取特征
    features = []
    for epoch in epochs:
        filtered = W @ epoch
        var = np.var(filtered, axis=1)
        log_var = np.log(var / np.sum(var))
        features.append(log_var)

    features = np.array(features)

    # 8. 存储
    if 'processed' not in data_dict:
        data_dict['processed'] = {}

    data_dict['processed']['feature'] = {
        'type': 'CSP',
        'data': features,
        'labels': labels,
        'channels': data_dict['signal'][modality]['channel_names'],
        'n_components': n_components,
        'filters': W,
        'conditions': [condition1, condition2],
        'modality': modality
    }

    return data_dict


def evaluate_decoder(data_dict,
                     feature_key=None,  # None表示使用processed.feature
                     classifier='lda',
                     cv_folds=5,
                     trial_time=4.0):
    """
    解码器评估
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("解码评估需要scikit-learn: pip install scikit-learn")

    # 1. 获取特征
    if 'processed' not in data_dict or 'feature' not in data_dict['processed']:
        raise ValueError("请先运行特征提取")

    if feature_key is None:
        # 使用当前feature
        feat = data_dict['processed']['feature']
        X = feat['data']
        y = np.array(feat['labels'])
    else:
        # 使用指定key（预留）
        raise NotImplementedError("多特征存储待实现")

    # 2. 分类器
    if classifier == 'lda':
        clf = LinearDiscriminantAnalysis()
    elif classifier == 'svm':
        clf = SVC(kernel='rbf', gamma='scale', probability=True)
    else:
        clf = LinearDiscriminantAnalysis()

    # 3. 交叉验证
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    accuracies = []
    all_y_true = []
    all_y_pred = []
    all_y_score = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        if hasattr(clf, 'predict_proba') and len(np.unique(y)) == 2:
            y_score = clf.predict_proba(X_test)[:, 1]
            all_y_score.extend(y_score)

    # 4. 综合指标
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    metrics = {
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'kappa': cohen_kappa_score(all_y_true, all_y_pred),
        'f1_macro': f1_score(all_y_true, all_y_pred, average='macro')
    }

    if all_y_score:
        metrics['auc'] = roc_auc_score(all_y_true, all_y_score)

    metrics['confusion_matrix'] = confusion_matrix(all_y_true, all_y_pred)

    # 5. ITR
    n_classes = len(np.unique(y))

    def _itr(acc, n, t):
        if acc <= 1 / n:
            return 0
        if acc >= 1:
            return np.log2(n) * (60 / t)
        B = np.log2(n) + acc * np.log2(acc) + (1 - acc) * np.log2((1 - acc) / (n - 1))
        return B * (60 / t)

    itr = _itr(mean_acc, n_classes, trial_time)

    # 6. 结果
    result = {
        'feature_type': data_dict['processed']['feature']['type'],
        'classifier': classifier,
        'cv_folds': cv_folds,
        'accuracy_mean': mean_acc,
        'accuracy_std': std_acc,
        'accuracy_per_fold': accuracies,
        'metrics': metrics,
        'itr': itr,
        'trial_time_sec': trial_time,
        'n_classes': n_classes
    }

    # 7. 写入
    if 'processed' not in data_dict:
        data_dict['processed'] = {}
    if 'statistics' not in data_dict['processed']:
        data_dict['processed']['statistics'] = {}
    if 'decoding' not in data_dict['processed']['statistics']:
        data_dict['processed']['statistics']['decoding'] = {}

    data_dict['processed']['statistics']['decoding']['evaluation'] = result

    return data_dict


def information_transfer_rate(accuracy, n_classes, trial_time):
    """ITR计算工具函数"""
    if accuracy <= 1 / n_classes:
        return 0
    if accuracy >= 1:
        return np.log2(n_classes) * (60 / trial_time)

    B = np.log2(n_classes) + \
        accuracy * np.log2(accuracy) + \
        (1 - accuracy) * np.log2((1 - accuracy) / (n_classes - 1))

    return B * (60 / trial_time)