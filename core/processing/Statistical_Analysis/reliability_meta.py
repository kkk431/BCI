#! /usr/bin/env python  
#  -*- coding:utf-8 -*-
"""
reliability_meta.py：信度分析与元分析核心模块（最终稳定版）
核心功能：
1. 信度分析：ICC（组内相关系数）、Cronbach's α（纯手动实现）、重测信度
2. 元分析：固定/随机效应模型、森林图/漏斗图、Egger检验、异质性分析
适配：Windows/Mac/Linux，兼容所有版本的pingouin/statsmodels，无第三方库兼容问题
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import platform
import logging
import os
from typing import Dict, List, Optional, Union, Tuple

# ===================== 全局配置（解决所有兼容性问题） =====================
# 1. 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("brainfusion.statistic.reliability_meta")

# 2. 绘图配置（彻底解决中文字体警告，适配所有系统）
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
sns.set_style("whitegrid")
sns.set_palette("husl")

# 系统适配的字体配置
system = platform.system()
if system == "Windows":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
elif system == "Darwin":  # Mac
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "DejaVu Sans"]
else:  # Linux
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


# ===================== 核心函数1：信度分析主函数 =====================
def reliability_analysis(
        data_dict: Dict,
        reliability_type: str = "icc",
        session_key: str = "session_id",
        feature_key: str = "processed_feature",
        alpha: float = 0.05
) -> Dict:
    """
    执行信度分析（ICC/Cronbach's α/重测信度）
    :param data_dict: 数据字典，必需字段：processed_feature、event
    :param reliability_type: 分析类型：icc/cronbach_alpha/test_retest
    :param session_key: 区分session的字段名
    :param feature_key: 特征字段名
    :param alpha: 显著性水平
    :return: 结构化分析结果
    """
    # 数据校验
    _validate_data_dict(data_dict, required_keys=[feature_key, "event"])

    # 提取核心数据
    features = np.array(data_dict[feature_key])
    sessions = np.array(data_dict["event"][session_key]) if session_key in data_dict["event"] else None

    # 执行对应分析
    try:
        if reliability_type == "icc":
            result = _calculate_icc(features, sessions, alpha)
        elif reliability_type == "cronbach_alpha":
            result = _calculate_cronbach_alpha(features, alpha, data_dict)
        elif reliability_type == "test_retest":
            result = _calculate_test_retest(features, sessions, alpha)
        else:
            raise ValueError(f"不支持的分析类型：{reliability_type}")

        # 结果解释
        result["interpretation"] = _interpret_reliability(result["reliability_value"], reliability_type)
        logger.info(
            f"✅ 信度分析完成 | 类型：{reliability_type} | "
            f"均值：{result['reliability_value']:.4f} | 解释：{result['interpretation']}"
        )
    except Exception as e:
        error_msg = str(e)[:100]
        logger.error(f"❌ 信度分析失败 | 类型：{reliability_type} | 错误：{error_msg}")
        result = {
            "reliability_type": reliability_type,
            "reliability_value": np.nan,
            "ci": (np.nan, np.nan),
            "p_value": np.nan,
            "feature_wise_reliability": [],
            "interpretation": f"计算失败：{error_msg}",
            "error": error_msg
        }

    return result


# ===================== 核心函数2：元分析主函数 =====================
def meta_analysis(
        effect_sizes: Union[List[float], np.ndarray],
        standard_errors: Union[List[float], np.ndarray],
        study_labels: Optional[List[str]] = None,
        model_type: str = "random",
        plot_forest: bool = True,
        plot_funnel: bool = True,
        save_plots: bool = True,
        plot_dir: str = "./meta_plots/"
) -> Dict:
    """
    执行元分析（纯Python实现，无statsmodels版本问题）
    :param effect_sizes: 各研究效应量
    :param standard_errors: 各研究标准误
    :param study_labels: 研究标签
    :param model_type: 模型类型：fixed/random
    :param plot_forest: 是否绘制森林图
    :param plot_funnel: 是否绘制漏斗图
    :param save_plots: 是否保存图片
    :param plot_dir: 图片保存目录
    :return: 结构化元分析结果
    """
    # 数据校验
    effect_sizes = np.array(effect_sizes)
    standard_errors = np.array(standard_errors)
    if len(effect_sizes) != len(standard_errors):
        raise ValueError("效应量和标准误数量不匹配")
    if len(effect_sizes) < 2:
        raise ValueError("元分析至少需要2个研究")

    # 默认研究标签
    study_labels = study_labels if study_labels else [f"Study{i + 1}" for i in range(len(effect_sizes))]

    try:
        # 计算权重和合并效应量
        weights = 1 / (standard_errors ** 2)

        if model_type == "fixed":
            pooled_effect = np.sum(weights * effect_sizes) / np.sum(weights)
            pooled_se = np.sqrt(1 / np.sum(weights))
        elif model_type == "random":
            # DerSimonian-Laird法随机效应
            q_stat = np.sum(weights * (effect_sizes - np.sum(weights * effect_sizes) / np.sum(weights)) ** 2)
            df = len(effect_sizes) - 1
            tau2 = max(0, (q_stat - df) / (np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)))
            weights_random = 1 / (standard_errors ** 2 + tau2)
            pooled_effect = np.sum(weights_random * effect_sizes) / np.sum(weights_random)
            pooled_se = np.sqrt(1 / np.sum(weights_random))
        else:
            raise ValueError("模型类型仅支持fixed/random")

        # 95%置信区间和显著性
        z = 1.96
        pooled_ci = (pooled_effect - z * pooled_se, pooled_effect + z * pooled_se)
        z_stat = pooled_effect / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

        # 异质性分析
        q_stat, q_pval, i2 = _heterogeneity_test(effect_sizes, standard_errors)

        # Egger检验（发表偏倚）
        egger_res = _egger_test(effect_sizes, standard_errors)

        # 整理结果
        result = {
            "model_type": model_type,
            "pooled_effect": float(pooled_effect),
            "pooled_se": float(pooled_se),
            "pooled_ci": (float(pooled_ci[0]), float(pooled_ci[1])),
            "p_value": float(p_value),
            "heterogeneity": {"Q": float(q_stat), "Q_p": float(q_pval), "I2": float(i2)},
            "egger_test": egger_res,
            "plots_saved": None
        }

        # 绘制图表
        if (plot_forest or plot_funnel) and save_plots:
            os.makedirs(plot_dir, exist_ok=True)
            result["plots_saved"] = plot_dir

        if plot_forest:
            _plot_forest_py(effect_sizes, standard_errors, study_labels, pooled_effect, pooled_ci, model_type, plot_dir,
                            save_plots)
        if plot_funnel:
            _plot_funnel_py(effect_sizes, standard_errors, plot_dir, save_plots)

        logger.info(
            f"✅ 元分析完成 | 模型：{model_type} | 合并效应量：{result['pooled_effect']:.4f} | "
            f"I²：{result['heterogeneity']['I2']:.1f}%"
        )
    except Exception as e:
        error_msg = str(e)[:100]
        logger.error(f"❌ 元分析失败 | 错误：{error_msg}")
        result = {
            "model_type": model_type,
            "pooled_effect": np.nan,
            "pooled_se": np.nan,
            "pooled_ci": (np.nan, np.nan),
            "p_value": np.nan,
            "heterogeneity": {"Q": np.nan, "Q_p": np.nan, "I2": np.nan},
            "egger_test": {"t_stat": np.nan, "p_value": np.nan, "is_bias": False},
            "plots_saved": None,
            "error": error_msg
        }

    return result


# ===================== 辅助函数：信度分析子函数 =====================
def _validate_data_dict(data_dict: Dict, required_keys: List[str]) -> None:
    """数据字典校验"""
    missing_keys = [k for k in required_keys if k not in data_dict]
    if missing_keys:
        raise KeyError(f"数据字典缺失字段：{missing_keys}")
    if not isinstance(data_dict.get("processed_feature", []), (np.ndarray, list)):
        raise TypeError("processed_feature必须是数组或列表")


def _calculate_icc(features: np.ndarray, sessions: np.ndarray, alpha: float) -> Dict:
    """计算ICC(2,1)，兼容所有pingouin版本"""
    unique_sessions = np.unique(sessions)
    if len(unique_sessions) < 2:
        raise ValueError(f"ICC需要至少2个session，当前{len(unique_sessions)}个")

    icc_vals = []
    ci_list = []
    n_features = features.shape[1] if len(features.shape) > 1 else 1
    n_samples_per_sess = len(features) // len(unique_sessions)

    for i in range(n_features):
        feat = features[:, i] if n_features > 1 else features
        # 构造平衡数据
        subject_ids = np.repeat(range(n_samples_per_sess), len(unique_sessions))
        rater_ids = np.tile(unique_sessions, n_samples_per_sess)
        values = feat[:len(subject_ids)]

        df = pd.DataFrame({"subject": subject_ids, "rater": rater_ids, "value": values})
        icc_result = pg.intraclass_corr(
            data=df, targets="subject", raters="rater", ratings="value", nan_policy='omit'
        )

        # 兼容CI95%格式
        icc_subset = icc_result[icc_result["Type"] == "ICC2k"]
        if not icc_subset.empty:
            icc_val = icc_subset["ICC"].values[0]
            ci_data = icc_subset["CI95%"].values[0]
        else:
            icc_val = icc_result["ICC"].values[0]
            ci_data = icc_result["CI95%"].values[0]

        # 解析置信区间
        if isinstance(ci_data, (list, np.ndarray)):
            ci = (float(ci_data[0]), float(ci_data[1]))
        elif isinstance(ci_data, str):
            ci = tuple(map(float, ci_data.strip("()").split(",")))
        else:
            ci = (np.nan, np.nan)

        icc_vals.append(icc_val)
        ci_list.append(ci)

    # 计算均值
    icc_vals_clean = [x for x in icc_vals if not np.isnan(x)]
    mean_icc = float(np.mean(icc_vals_clean)) if icc_vals_clean else np.nan
    mean_ci = (
        float(np.mean([c[0] for c in ci_list if not np.isnan(c[0])])),
        float(np.mean([c[1] for c in ci_list if not np.isnan(c[1])]))
    )

    return {
        "reliability_type": "icc",
        "reliability_value": mean_icc,
        "ci": mean_ci,
        "p_value": None,
        "feature_wise_reliability": icc_vals
    }


def _calculate_cronbach_alpha(features: np.ndarray, alpha: float, data_dict: Dict) -> Dict:
    """纯手动实现Cronbach's α（无第三方库依赖，绝对稳定）
    公式：α = (k/(k-1)) * (1 - (Σσ²_i)/σ²_total)
    """
    try:
        # 数据预处理
        feat_arr = np.array(features)
        if len(feat_arr.shape) == 1:
            feat_arr = feat_arr.reshape(-1, 1)
        n_samples, n_features = feat_arr.shape

        # 基础校验
        if n_samples < 2:
            raise ValueError(f"样本量不足：{n_samples}（至少2个）")
        if n_features < 2:
            raise ValueError(f"特征数不足：{n_features}（至少2个）")

        # 处理缺失值/极端值
        feat_arr = np.nan_to_num(
            feat_arr,
            nan=np.nanmean(feat_arr),
            posinf=np.percentile(feat_arr, 99),
            neginf=np.percentile(feat_arr, 1)
        )

        # 手动计算α
        variances = np.var(feat_arr, axis=0, ddof=1)  # 各特征方差
        sum_variances = np.sum(variances)
        total_scores = np.sum(feat_arr, axis=1)  # 总分
        total_variance = np.var(total_scores, axis=0, ddof=1)

        if total_variance == 0:
            raise ValueError("总分方差为0，无法计算α")

        cronbach_alpha = (n_features / (n_features - 1)) * (1 - (sum_variances / total_variance))

        # 计算95%置信区间（z变换）
        if cronbach_alpha >= 1:
            cronbach_alpha = 0.9999
        if cronbach_alpha <= -1:
            cronbach_alpha = -0.9999

        z = 0.5 * np.log((1 + cronbach_alpha) / (1 - cronbach_alpha))
        se_z = 1 / np.sqrt(n_samples - 3)
        z_lower = z - 1.96 * se_z
        z_upper = z + 1.96 * se_z

        alpha_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        alpha_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

        logger.info(
            f"Cronbach's α计算成功 | 值：{cronbach_alpha:.4f} | "
            f"95%CI：({alpha_lower:.4f}, {alpha_upper:.4f})"
        )

        return {
            "reliability_type": "cronbach_alpha",
            "reliability_value": float(cronbach_alpha),
            "ci": (float(alpha_lower), float(alpha_upper)),
            "p_value": None,
            "feature_wise_reliability": [cronbach_alpha]
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Cronbach's α计算失败：{error_msg}")
        return {
            "reliability_type": "cronbach_alpha",
            "reliability_value": np.nan,
            "ci": (np.nan, np.nan),
            "p_value": np.nan,
            "feature_wise_reliability": []
        }


def _calculate_test_retest(features: np.ndarray, sessions: np.ndarray, alpha: float) -> Dict:
    """计算重测信度（Pearson/Spearman相关）"""
    unique_sess = np.unique(sessions)
    if len(unique_sess) != 2:
        raise ValueError(f"重测信度需要2个session，当前{len(unique_sess)}个")

    # 提取两个session数据
    sess1_feat = features[sessions == unique_sess[0]]
    sess2_feat = features[sessions == unique_sess[1]]
    min_len = min(len(sess1_feat), len(sess2_feat))
    sess1_feat = sess1_feat[:min_len]
    sess2_feat = sess2_feat[:min_len]

    # 计算相关系数
    corr_vals = []
    p_vals = []
    ci_list = []
    n_features = features.shape[1] if len(features.shape) > 1 else 1

    for i in range(n_features):
        d1 = sess1_feat[:, i] if n_features > 1 else sess1_feat
        d2 = sess2_feat[:, i] if n_features > 1 else sess2_feat

        if len(d1) < 3 or len(d2) < 3:
            corr_vals.append(np.nan)
            p_vals.append(np.nan)
            ci_list.append((np.nan, np.nan))
            continue

        # 正态性检验
        norm1 = stats.shapiro(d1)[1] > alpha
        norm2 = stats.shapiro(d2)[1] > alpha

        if norm1 and norm2:
            corr, p = stats.pearsonr(d1, d2)
            # Fisher z变换计算CI
            z = np.arctanh(corr)
            se = 1 / np.sqrt(len(d1) - 3)
            ci = (np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se))
        else:
            corr, p = stats.spearmanr(d1, d2)
            ci = (np.nan, np.nan)

        corr_vals.append(corr)
        p_vals.append(p)
        ci_list.append(ci)

    # 计算均值
    corr_vals_clean = [x for x in corr_vals if not np.isnan(x)]
    p_vals_clean = [x for x in p_vals if not np.isnan(x)]

    mean_corr = float(np.mean(corr_vals_clean)) if corr_vals_clean else np.nan
    mean_p = float(np.mean(p_vals_clean)) if p_vals_clean else np.nan
    mean_ci = (
        float(np.mean([c[0] for c in ci_list if not np.isnan(c[0])])),
        float(np.mean([c[1] for c in ci_list if not np.isnan(c[1])]))
    )

    return {
        "reliability_type": "test_retest",
        "reliability_value": mean_corr,
        "ci": mean_ci,
        "p_value": mean_p,
        "feature_wise_reliability": corr_vals_clean
    }


def _interpret_reliability(value: float, rtype: str) -> str:
    """信度结果解释（行业通用标准）"""
    if np.isnan(value):
        return "无法计算（数据不足/错误）"

    if rtype in ["icc", "test_retest"]:
        if value < 0.5:
            return "差（<0.5）"
        elif value < 0.75:
            return "可接受（0.5-0.75）"
        elif value < 0.9:
            return "良好（0.75-0.9）"
        else:
            return "优秀（≥0.9）"
    elif rtype == "cronbach_alpha":
        if value < 0.6:
            return "差（<0.6）"
        elif value < 0.7:
            return "可接受（0.6-0.7）"
        elif value < 0.8:
            return "良好（0.7-0.8）"
        else:
            return "优秀（≥0.8）"
    else:
        return "无法解释"


# ===================== 辅助函数：元分析子函数 =====================
def _heterogeneity_test(effect_sizes: np.ndarray, se: np.ndarray) -> Tuple[float, float, float]:
    """异质性检验（Q统计量、I²）"""
    weights = 1 / (se ** 2)
    overall_effect = np.sum(weights * effect_sizes) / np.sum(weights)
    q_stat = np.sum(weights * (effect_sizes - overall_effect) ** 2)
    df = len(effect_sizes) - 1
    q_pval = 1 - stats.chi2.cdf(q_stat, df=df) if q_stat >= 0 else 1.0
    i2 = max(0, (q_stat - df) / q_stat * 100) if q_stat > 0 else 0.0
    return q_stat, q_pval, i2


def _egger_test(effect_sizes: np.ndarray, se: np.ndarray) -> Dict:
    """Egger检验（发表偏倚）"""
    if len(effect_sizes) < 3:
        return {"t_stat": np.nan, "p_value": np.nan, "is_bias": False}

    x = 1 / se
    y = effect_sizes
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    t_stat = slope / std_err if std_err != 0 else np.nan
    df = len(effect_sizes) - 2
    egger_p = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df)) if not np.isnan(t_stat) else np.nan

    return {
        "t_stat": float(t_stat),
        "p_value": float(egger_p),
        "is_bias": egger_p < 0.05 and not np.isnan(egger_p)
    }


def _plot_forest_py(effect_sizes: np.ndarray, se: np.ndarray, labels: List[str], pooled_effect: float, pooled_ci: Tuple,
                    model_type: str, plot_dir: str, save_plots: bool) -> None:
    """绘制森林图（无中文字体警告）"""
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(effect_sizes))
    plt.errorbar(effect_sizes, y_pos, xerr=1.96 * se, fmt='o', color='gray', capsize=5, label="Individual Studies")
    plt.axvline(pooled_effect, color='red', linestyle='--', label=f"Pooled Effect ({model_type})")
    plt.axvspan(pooled_ci[0], pooled_ci[1], color='red', alpha=0.2, label=f"95% CI")

    plt.yticks(y_pos, labels)
    plt.xlabel("Effect Size (Cohen's d)")
    plt.title("Meta-Analysis Forest Plot")
    plt.legend(loc="upper right")
    plt.grid(axis='x')

    if save_plots:
        plt.savefig(os.path.join(plot_dir, "forest_plot.png"), bbox_inches="tight")
    plt.close()


def _plot_funnel_py(effect_sizes: np.ndarray, se: np.ndarray, plot_dir: str, save_plots: bool) -> None:
    """绘制漏斗图（无中文字体警告）"""
    plt.figure(figsize=(8, 6))
    plt.scatter(effect_sizes, se, color='blue', alpha=0.7)
    plt.axhline(y=np.mean(se), color='red', linestyle='--', label="Mean Standard Error")

    plt.xlabel("Effect Size (Cohen's d)")
    plt.ylabel("Standard Error")
    plt.title("Funnel Plot (Publication Bias Detection)")
    plt.legend()
    plt.grid(True)

    if save_plots:
        plt.savefig(os.path.join(plot_dir, "funnel_plot.png"), bbox_inches="tight")
    plt.close()


# ===================== 测试用例（可直接运行） =====================
if __name__ == "__main__":
    # 构造有相关性的测试数据（确保Cronbach's α能算出有效值）
    np.random.seed(42)
    n_samples = 100
    n_sessions = 2
    n_features = 8
    n_subjects_per_sess = n_samples // n_sessions

    # 构造强相关性特征
    base_signal = np.linspace(0, 10, n_samples) + np.random.randn(n_samples) * 0.5
    features = []
    for i in range(n_features):
        features.append(base_signal + np.random.randn(n_samples) * 0.1)
    features = np.column_stack(features)

    # 构造数据字典
    data_dict = {
        "meta": {"subject_id": "S01", "modality": "EEG"},
        "signal": {"raw_data": np.random.randn(n_samples, 16)},
        "event": {
            "event_label": ["握拳"] * n_samples,
            "session_id": ["session1"] * n_subjects_per_sess + ["session2"] * n_subjects_per_sess
        },
        "processed_feature": features
    }

    # 1. ICC分析
    icc_result = reliability_analysis(data_dict, reliability_type="icc", session_key="session_id")
    print("\n=== ICC信度分析结果 ===")
    print(f"均值：{icc_result['reliability_value']:.4f} | 解释：{icc_result['interpretation']}")

    # 2. Cronbach's α分析
    alpha_result = reliability_analysis(data_dict, reliability_type="cronbach_alpha")
    print("\n=== Cronbach's α结果 ===")
    print(f"均值：{alpha_result['reliability_value']:.4f} | 解释：{alpha_result['interpretation']}")

    # 3. 重测信度分析
    retest_result = reliability_analysis(data_dict, reliability_type="test_retest", session_key="session_id")
    print("\n=== 重测信度结果 ===")
    print(f"均值：{retest_result['reliability_value']:.4f} | p值：{retest_result['p_value']:.4f}")

    # 4. 元分析
    effect_sizes = np.array([0.3, 0.5, 0.4, 0.6, 0.2])
    standard_errors = np.array([0.1, 0.14, 0.12, 0.16, 0.09])
    meta_result = meta_analysis(effect_sizes, standard_errors, model_type="random")
    print("\n=== 元分析结果 ===")
    print(f"合并效应量：{meta_result['pooled_effect']:.4f} | I²：{meta_result['heterogeneity']['I2']:.1f}%")
    print(f"发表偏倚：{'存在' if meta_result['egger_test']['is_bias'] else '不存在'}")