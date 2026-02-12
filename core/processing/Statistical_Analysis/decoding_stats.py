"""
decoding_stats.py
================================================================================
脑机接口解码评估模块 —— 纯性能指标，无特征提取

核心功能：
    1. 分类指标：准确率、Kappa、F1-score、AUC、混淆矩阵
    2. 回归指标：MSE、RMSE、MAE、R²、调整R²
    3. 信息传输率：Wolpaw ITR、Nykopp ITR

设计哲学：
    - 纯评估：只接收预测标签/概率和真实标签，绝不包含分类器训练代码
    - 通用接口：与任何分类器/回归器解耦

输入格式：
    - y_true: 真实标签
    - y_pred: 预测标签
    - y_score: 预测概率（用于AUC）

输出格式：
    - 统一存入 data_dict['processed']['statistics']['decoding']

依赖：
    - 必需：numpy
    - 可选：scikit-learn（无时自动降级部分功能）

版本: 2.0.0
最后更新: 2024
================================================================================
"""

import numpy as np
import warnings
import logging
logger = logging.getLogger(__name__)

# scikit-learn可选导入
try:
    from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                               roc_auc_score, confusion_matrix,
                               mean_squared_error, mean_absolute_error, r2_score)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn未安装，部分指标将使用手动实现。推荐: pip install scikit-learn")


# ============================================================================
# 1. 分类指标（零依赖手动实现）
# ============================================================================

def _confusion_matrix_manual(y_true, y_pred):
    """手动实现混淆矩阵"""
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    for t, p in zip(y_true, y_pred):
        i = label_to_idx[t]
        j = label_to_idx[p]
        cm[i, j] += 1

    return cm, labels


def _accuracy_score_manual(y_true, y_pred):
    """手动实现准确率"""
    return np.mean(y_true == y_pred)


def _cohen_kappa_manual(y_true, y_pred):
    """手动实现Cohen's Kappa"""
    cm, labels = _confusion_matrix_manual(y_true, y_pred)
    n = len(y_true)

    # 观察一致性
    p_o = np.trace(cm) / n

    # 期望一致性
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    p_e = np.sum(row_sum * col_sum) / (n * n)

    return (p_o - p_e) / (1 - p_e + 1e-10)


def _f1_score_manual(y_true, y_pred, average='macro'):
    """手动实现F1分数"""
    cm, labels = _confusion_matrix_manual(y_true, y_pred)
    n_classes = len(labels)

    f1_scores = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        f1_scores.append(f1)

    if average == 'macro':
        return np.mean(f1_scores)
    elif average == 'weighted':
        weights = cm.sum(axis=1) / len(y_true)
        return np.sum(np.array(f1_scores) * weights)
    else:
        return f1_scores


def _auc_manual(y_true, y_score):
    """手动实现AUC（Mann-Whitney U统计量）"""
    n1 = np.sum(y_true == 0)
    n2 = np.sum(y_true == 1)
    if n1 == 0 or n2 == 0:
        return 0.5

    ranks = np.argsort(y_score)
    rank_sum = np.sum(ranks[np.where(y_true == 1)[0]])
    u = rank_sum - (n2 * (n2 + 1)) / 2
    return u / (n1 * n2)


# ============================================================================
# 2. 回归指标（零依赖手动实现）
# ============================================================================

def _mse_manual(y_true, y_pred):
    """手动实现均方误差"""
    return np.mean((y_true - y_pred) ** 2)


def _rmse_manual(y_true, y_pred):
    """手动实现均方根误差"""
    return np.sqrt(_mse_manual(y_true, y_pred))


def _mae_manual(y_true, y_pred):
    """手动实现平均绝对误差"""
    return np.mean(np.abs(y_true - y_pred))


def _r2_manual(y_true, y_pred):
    """手动实现R²决定系数"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-10))


# ============================================================================
# 3. 公开API - 分类指标
# ============================================================================

def classification_metrics(y_true, y_pred, y_score=None, average='macro'):
    """
    计算完整的分类性能指标。

    Parameters
    ----------
    y_true : array_like
        真实标签。
    y_pred : array_like
        预测标签。
    y_score : array_like, optional
        预测概率，用于AUC计算（仅二分类）。
    average : {'macro', 'weighted', None}, default='macro'
        F1分数的平均方法。

    Returns
    -------
    metrics : dict
        包含以下指标：
        - accuracy : 准确率
        - kappa : Cohen's Kappa
        - f1_macro : 宏平均F1
        - f1_weighted : 加权平均F1
        - auc : AUC（二分类且提供y_score时）
        - confusion_matrix : 混淆矩阵
        - per_class : 每类的精确率、召回率、F1

    Examples
    --------
    >>> y_true = [0, 1, 0, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 0]
    >>> metrics = classification_metrics(y_true, y_pred)
    >>> print(f"准确率: {metrics['accuracy']:.3f}")
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 选择实现
    if SKLEARN_AVAILABLE:
        # sklearn实现
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        # 每类指标
        labels = np.unique(np.concatenate([y_true, y_pred]))
        per_class = {}
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)

            per_class[str(label)] = {
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
            }
    else:
        # 手动实现
        cm, labels = _confusion_matrix_manual(y_true, y_pred)
        accuracy = _accuracy_score_manual(y_true, y_pred)
        kappa = _cohen_kappa_manual(y_true, y_pred)
        f1_macro = np.mean(_f1_score_manual(y_true, y_pred, average=None))
        f1_weighted = _f1_score_manual(y_true, y_pred, average='weighted')

        # 每类指标
        per_class = {}
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)

            per_class[str(label)] = {
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
            }

    metrics = {
        'accuracy': accuracy,
        'kappa': kappa,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'per_class': per_class
    }

    # AUC（仅二分类）
    if y_score is not None and len(np.unique(y_true)) == 2:
        if SKLEARN_AVAILABLE:
            metrics['auc'] = roc_auc_score(y_true, y_score)
        else:
            metrics['auc'] = _auc_manual(y_true, y_score)

    return metrics


# ============================================================================
# 4. 公开API - 回归指标
# ============================================================================

def regression_metrics(y_true, y_pred, n_features=None):
    """
    计算完整的回归性能指标。

    Parameters
    ----------
    y_true : array_like
        真实值。
    y_pred : array_like
        预测值。
    n_features : int, optional
        特征数量，用于调整R²计算。

    Returns
    -------
    metrics : dict
        包含以下指标：
        - mse : 均方误差
        - rmse : 均方根误差
        - mae : 平均绝对误差
        - r2 : R²决定系数
        - adjusted_r2 : 调整R²（需提供n_features）

    Examples
    --------
    >>> y_true = [1.0, 2.0, 3.0, 4.0]
    >>> y_pred = [1.1, 1.9, 3.2, 3.8]
    >>> metrics = regression_metrics(y_true, y_pred, n_features=5)
    >>> print(f"RMSE: {metrics['rmse']:.3f}")
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if SKLEARN_AVAILABLE:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
    else:
        mse = _mse_manual(y_true, y_pred)
        rmse = _rmse_manual(y_true, y_pred)
        mae = _mae_manual(y_true, y_pred)
        r2 = _r2_manual(y_true, y_pred)

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    # 调整R²
    if n_features is not None:
        n = len(y_true)
        metrics['adjusted_r2'] = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    return metrics


# ============================================================================
# 5. 信息传输率（ITR）
# ============================================================================

def information_transfer_rate(accuracy, n_classes, trial_time):
    """
    计算Wolpaw信息传输率（bits/min）。

    Parameters
    ----------
    accuracy : float
        分类准确率，范围[0, 1]。
    n_classes : int
        类别数。
    trial_time : float
        单试次持续时间（秒）。

    Returns
    -------
    itr : float
        信息传输率（bits/min）。

    Notes
    -----
    公式：ITR = (log2(N) + acc * log2(acc) + (1-acc) * log2((1-acc)/(N-1))) * (60/T)

    参考文献：
        Wolpaw et al. (1998) IEEE Trans Rehabil Eng
    """
    if accuracy <= 1 / n_classes:
        return 0
    if accuracy >= 1:
        return np.log2(n_classes) * (60 / trial_time)

    B = np.log2(n_classes) + \
        accuracy * np.log2(accuracy) + \
        (1 - accuracy) * np.log2((1 - accuracy) / (n_classes - 1))

    return B * (60 / trial_time)


def information_transfer_rate_nykopp(accuracy, n_classes, trial_time):
    """
    计算Nykopp信息传输率（bits/min）- 基于互信息的修正版本。

    Parameters
    ----------
    accuracy : float
        分类准确率，范围[0, 1]。
    n_classes : int
        类别数。
    trial_time : float
        单试次持续时间（秒）。

    Returns
    -------
    itr : float
        信息传输率（bits/min）。

    Notes
    -----
    相比Wolpaw ITR，Nykopp ITR在高准确率时更保守。

    参考文献：
        Nykopp (2006) TKK dissertation
    """
    if accuracy <= 1 / n_classes:
        return 0

    p_correct = accuracy
    p_error = (1 - accuracy) / (n_classes - 1)

    # 构建理想混淆矩阵
    conf_mat = np.ones((n_classes, n_classes)) * p_error
    np.fill_diagonal(conf_mat, p_correct)

    # 计算互信息
    p_y = np.ones(n_classes) / n_classes
    p_ypred = conf_mat.mean(axis=0)

    mi = 0
    for i in range(n_classes):
        for j in range(n_classes):
            if conf_mat[i, j] > 0:
                mi += p_y[i] * conf_mat[i, j] * np.log2(conf_mat[i, j] / p_ypred[j])

    return mi * (60 / trial_time)


# ============================================================================
# 6. 四层结构适配接口
# ============================================================================

def evaluate_decoder(data_dict,
                     feature_key=None,
                     y_true=None, y_pred=None, y_score=None,
                     task_type='classification',
                     trial_time=4.0,
                     n_classes=None,
                     n_features=None):
    """
    解码器评估接口 —— 纯指标计算，无分类器训练。

    Parameters
    ----------
    data_dict : dict
        四层嵌套结构。
    feature_key : str, optional
        指定特征来源（仅用于从data_dict读取真实标签）。
    y_true, y_pred, y_score : array_like, optional
        直接传入预测结果，优先级高于从data_dict读取。
    task_type : {'classification', 'regression'}, default='classification'
        任务类型。
    trial_time : float, default=4.0
        单试次时间（秒），用于ITR计算。
    n_classes : int, optional
        类别数，自动推断时可不提供。
    n_features : int, optional
        特征数，用于调整R²。

    Returns
    -------
    data_dict : dict
        更新后的数据字典，结果存入：
        data_dict['processed']['statistics']['decoding']['evaluation']

    Examples
    --------
    >>> # 从data_dict读取标签，直接传入预测结果
    >>> y_pred = model.predict(X_test)
    >>> data = evaluate_decoder(data, y_true=y_test, y_pred=y_pred)
    >>>
    >>> # 使用已存储的特征和标签
    >>> data = evaluate_decoder(data, feature_key='csp')
    """
    # 1. 获取真实标签
    if y_true is None:
        if feature_key is not None and 'processed' in data_dict:
            if 'features' in data_dict['processed'] and feature_key in data_dict['processed']['features']:
                y_true = data_dict['processed']['features'][feature_key]['labels']
            elif 'feature' in data_dict['processed']:
                y_true = data_dict['processed']['feature']['labels']
            else:
                raise ValueError("无法从data_dict读取标签")
        else:
            raise ValueError("请提供y_true或指定feature_key")

    # 2. 计算指标
    if task_type == 'classification':
        if y_pred is None:
            raise ValueError("分类任务需要提供y_pred")

        metrics = classification_metrics(y_true, y_pred, y_score)

        # 自动推断类别数
        if n_classes is None:
            n_classes = len(np.unique(y_true))

        # 计算ITR
        itr = information_transfer_rate(metrics['accuracy'], n_classes, trial_time)
        itr_nykopp = information_transfer_rate_nykopp(metrics['accuracy'], n_classes, trial_time)

        result = {
            'task_type': 'classification',
            'metrics': metrics,
            'itr': itr,
            'itr_nykopp': itr_nykopp,
            'trial_time_sec': trial_time,
            'n_classes': n_classes
        }

    elif task_type == 'regression':
        if y_pred is None:
            raise ValueError("回归任务需要提供y_pred")

        metrics = regression_metrics(y_true, y_pred, n_features)

        result = {
            'task_type': 'regression',
            'metrics': metrics
        }

    else:
        raise ValueError(f"不支持的任务类型: {task_type}")

    # 3. 元信息
    result.update({
        'n_samples': len(y_true),
        'feature_key': feature_key
    })

    # 4. 写入data_dict
    if 'processed' not in data_dict:
        data_dict['processed'] = {}
    if 'statistics' not in data_dict['processed']:
        data_dict['processed']['statistics'] = {}
    if 'decoding' not in data_dict['processed']['statistics']:
        data_dict['processed']['statistics']['decoding'] = {}

    data_dict['processed']['statistics']['decoding']['evaluation'] = result

    logger.info(f"解码评估完成: {task_type}, 准确率/R²: {metrics.get('accuracy', metrics.get('r2', 0)):.4f}")

    return data_dict


# ============================================================================
# 7. 便捷函数
# ============================================================================

def classification_report(y_true, y_pred, y_score=None, return_dict=False):
    """
    生成分类报告（类似sklearn的classification_report）。

    Parameters
    ----------
    y_true, y_pred : array_like
        真实标签和预测标签。
    y_score : array_like, optional
        预测概率。
    return_dict : bool, default=False
        是否返回字典格式。

    Returns
    -------
    report : str or dict
        分类报告。
    """
    metrics = classification_metrics(y_true, y_pred, y_score)

    if return_dict:
        return metrics

    # 生成文本报告
    labels = list(metrics['per_class'].keys())
    report = "\n分类报告:\n"
    report += "=" * 50 + "\n"
    report += f"{'':10s} {'precision':10s} {'recall':10s} {'f1-score':10s}\n"
    report += "-" * 50 + "\n"

    for label in labels:
        p = metrics['per_class'][label]['precision']
        r = metrics['per_class'][label]['recall']
        f = metrics['per_class'][label]['f1']
        report += f"{str(label):10s} {p:10.4f} {r:10.4f} {f:10.4f}\n"

    report += "-" * 50 + "\n"
    report += f"{'accuracy':10s} {metrics['accuracy']:10.4f}\n"
    report += f"{'macro avg':10s} {metrics['f1_macro']:10.4f}\n"
    report += f"{'weighted avg':10s} {metrics['f1_weighted']:10.4f}\n"

    if 'auc' in metrics:
        report += f"{'auc':10s} {metrics['auc']:10.4f}\n"

    report += "=" * 50 + "\n"

    return report