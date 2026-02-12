#! /usr/bin/env python
#  -*- coding:utf-8 -*-
"""
advanced_stats.py：BCI数据进阶统计分析模块
核心功能：
1. 效应量计算：Cohen's d/Hedges' g/Glass's Δ（组间）、Pearson r/R²（相关）、OR/RR（分类）
2. 统计检验：t检验/方差分析/卡方检验（与效应量联动）
3. 功效分析：样本量估算、检验力计算
适配：与reliability_meta.py无缝兼容，为元分析提供标准化效应量输入
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import logging
from typing import Dict, List, Optional, Union, Tuple

# ===================== 全局配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("brainfusion.statistic.advanced_stats")


# ===================== 核心函数1：效应量计算主入口 =====================
def calculate_effect_size(
        data1: Union[np.ndarray, List[float]],
        data2: Optional[Union[np.ndarray, List[float]]] = None,
        effect_type: str = "cohens_d",
        test_type: str = "independent",  # independent/paired（独立/配对样本）
        alpha: float = 0.05
) -> Dict:
    """
    统一效应量计算入口（适配BCI数据场景）
    :param data1: 样本1数据（实验组/处理前/特征列）
    :param data2: 样本2数据（对照组/处理后/对照列），部分效应量无需此参数
    :param effect_type: 效应量类型：
                        - 组间比较：cohens_d/hedges_g/glass_delta
                        - 相关分析：pearson_r/r_squared/cohens_f2
                        - 分类数据：odds_ratio/risk_ratio/cramers_v
    :param test_type: 检验类型（仅组间比较用）：independent/paired
    :param alpha: 显著性水平（用于置信区间计算）
    :return: 结构化结果（值、标准误、95%CI、解释）
    """
    # 数据预处理：移除缺失值，转为数组
    data1 = np.array(data1)
    data1 = data1[~np.isnan(data1)]
    if data2 is not None:
        data2 = np.array(data2)
        data2 = data2[~np.isnan(data2)]

    # 基础校验
    if len(data1) < 2:
        raise ValueError(f"样本1有效数据量不足（{len(data1)}），至少需要2个")
    if data2 is not None and len(data2) < 2 and effect_type not in ["pearson_r", "r_squared", "cohens_f2"]:
        raise ValueError(f"样本2有效数据量不足（{len(data2)}），至少需要2个")

    # 分类型计算效应量
    try:
        if effect_type in ["cohens_d", "hedges_g", "glass_delta"]:
            result = _calculate_group_effect_size(data1, data2, effect_type, test_type, alpha)
        elif effect_type in ["pearson_r", "r_squared", "cohens_f2"]:
            result = _calculate_correlation_effect_size(data1, data2, effect_type, alpha)
        elif effect_type in ["odds_ratio", "risk_ratio", "cramers_v"]:
            result = _calculate_categorical_effect_size(data1, data2, effect_type, alpha)
        else:
            raise ValueError(f"不支持的效应量类型：{effect_type}")

        # 自动解释效应量大小（行业通用标准）
        result["interpretation"] = _interpret_effect_size(result["effect_size"], effect_type)
        logger.info(
            f"✅ 效应量计算完成 | 类型：{effect_type} | "
            f"值：{result['effect_size']:.4f} | 解释：{result['interpretation']}"
        )
    except Exception as e:
        error_msg = str(e)[:100]
        logger.error(f"❌ 效应量计算失败 | 类型：{effect_type} | 错误：{error_msg}")
        result = {
            "effect_type": effect_type,
            "effect_size": np.nan,
            "se": np.nan,
            "ci": (np.nan, np.nan),
            "interpretation": f"计算失败：{error_msg}",
            "error": error_msg
        }

    return result


# ===================== 核心函数2：统计检验（与效应量联动） =====================
def statistical_test(
        data1: Union[np.ndarray, List[float]],
        data2: Optional[Union[np.ndarray, List[float]]] = None,
        test_type: str = "t_test",  # t_test/anova/chi2/correlation
        paired: bool = False,
        alpha: float = 0.05
) -> Dict:
    """
    统计检验（自动联动效应量计算）
    :param data1: 样本1数据
    :param data2: 样本2数据（ANOVA/卡方检验可为多组数据列表）
    :param test_type: 检验类型：t_test（t检验）/anova（方差分析）/chi2（卡方）/correlation（相关）
    :param paired: 是否配对样本（仅t检验用）
    :param alpha: 显著性水平
    :return: 检验结果 + 对应效应量
    """
    # 数据预处理
    data1 = np.array(data1)
    data1 = data1[~np.isnan(data1)]
    if data2 is not None:
        if isinstance(data2, list) and test_type == "anova":
            # ANOVA支持多组数据
            data2 = [np.array(d)[~np.isnan(np.array(d))] for d in data2]
        else:
            data2 = np.array(data2)
            data2 = data2[~np.isnan(data2)]

    try:
        if test_type == "t_test":
            # t检验 + Cohen's d
            if paired:
                t_stat, p_val = stats.ttest_rel(data1, data2[:len(data1)])
                effect_res = calculate_effect_size(data1, data2, "cohens_d", "paired", alpha)
            else:
                t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)  # 不等方差t检验
                effect_res = calculate_effect_size(data1, data2, "cohens_d", "independent", alpha)

            result = {
                "test_type": "t_test",
                "statistic": t_stat,
                "p_value": p_val,
                "significant": p_val < alpha,
                "effect_size": effect_res
            }

        elif test_type == "anova":
            # 单因素ANOVA + Eta²（效应量）
            if not isinstance(data2, list):
                data2 = [data2]  # 转为多组格式
            all_data = [data1] + data2
            f_stat, p_val = stats.f_oneway(*all_data)

            # 计算Eta²（ANOVA效应量）
            ss_between = sum([len(d) * (np.mean(d) - np.mean(np.concatenate(all_data))) ** 2 for d in all_data])
            ss_total = sum([np.sum((d - np.mean(d)) ** 2) for d in all_data])
            eta_squared = ss_between / ss_total

            result = {
                "test_type": "anova",
                "statistic": f_stat,
                "p_value": p_val,
                "significant": p_val < alpha,
                "effect_size": {
                    "effect_type": "eta_squared",
                    "effect_size": eta_squared,
                    "interpretation": _interpret_effect_size(eta_squared, "eta_squared")
                }
            }

        elif test_type == "chi2":
            # 卡方检验 + Cramer's V
            # 构造列联表
            obs = pd.crosstab(pd.Series(data1), pd.Series(data2))
            chi2_stat, p_val, dof, expected = stats.chi2_contingency(obs)
            effect_res = calculate_effect_size(data1, data2, "cramers_v", alpha=alpha)

            result = {
                "test_type": "chi2",
                "statistic": chi2_stat,
                "p_value": p_val,
                "dof": dof,
                "significant": p_val < alpha,
                "effect_size": effect_res
            }

        elif test_type == "correlation":
            # 相关分析 + Pearson r
            corr_stat, p_val = stats.pearsonr(data1, data2)
            effect_res = calculate_effect_size(data1, data2, "pearson_r", alpha=alpha)

            result = {
                "test_type": "correlation",
                "statistic": corr_stat,
                "p_value": p_val,
                "significant": p_val < alpha,
                "effect_size": effect_res
            }

        else:
            raise ValueError(f"不支持的检验类型：{test_type}")

        logger.info(
            f"✅ 统计检验完成 | 类型：{test_type} | "
            f"统计量：{result['statistic']:.4f} | p值：{result['p_value']:.4f} | "
            f"显著：{'是' if result['significant'] else '否'}"
        )
    except Exception as e:
        error_msg = str(e)[:100]
        logger.error(f"❌ 统计检验失败 | 类型：{test_type} | 错误：{error_msg}")
        result = {
            "test_type": test_type,
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "effect_size": {"effect_size": np.nan, "interpretation": "计算失败"},
            "error": error_msg
        }

    return result


# ===================== 核心函数3：功效分析（样本量估算） =====================
def power_analysis(
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.8,
        test_type: str = "t_test",
        groups: int = 2
) -> Dict:
    """
    统计功效分析（估算所需最小样本量）
    :param effect_size: 预期效应量（如Cohen's d=0.5为中等效应）
    :param alpha: 显著性水平
    :param power: 检验力（通常取0.8）
    :param test_type: 检验类型：t_test/anova/correlation
    :param groups: 组数（仅ANOVA用，默认2）
    :return: 最小样本量 + 功效分析结果
    """
    try:
        if effect_size <= 0:
            raise ValueError("效应量必须大于0")

        if test_type == "t_test":
            # t检验样本量估算（Cohen, 1988）
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(power)
            n_per_group = ((z_alpha + z_beta) / effect_size) ** 2 * 2
            total_n = int(np.ceil(n_per_group * 2))

        elif test_type == "anova":
            # ANOVA样本量估算（基于F检验）
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(power)
            f2 = effect_size ** 2 / (1 - effect_size ** 2)  # 转换为f²
            n_per_group = ((z_alpha + z_beta) ** 2 * (groups - 1)) / (groups * f2)
            total_n = int(np.ceil(n_per_group * groups))

        elif test_type == "correlation":
            # 相关分析样本量估算
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(power)
            # Fisher z变换
            z_r = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
            n = ((z_alpha + z_beta) / z_r) ** 2 + 3
            total_n = int(np.ceil(n))

        else:
            raise ValueError(f"不支持的检验类型：{test_type}")

        result = {
            "test_type": test_type,
            "effect_size": effect_size,
            "alpha": alpha,
            "power": power,
            "n_per_group": int(np.ceil(total_n / groups)),
            "total_sample_size": total_n,
            "interpretation": f"在α={alpha}、检验力={power}下，每组至少需要{int(np.ceil(total_n / groups))}个样本"
        }

        logger.info(
            f"✅ 功效分析完成 | 检验类型：{test_type} | "
            f"总样本量：{total_n} | 每组样本量：{int(np.ceil(total_n / groups))}"
        )
    except Exception as e:
        error_msg = str(e)[:100]
        logger.error(f"❌ 功效分析失败 | 错误：{error_msg}")
        result = {
            "test_type": test_type,
            "total_sample_size": np.nan,
            "n_per_group": np.nan,
            "interpretation": f"计算失败：{error_msg}",
            "error": error_msg
        }

    return result


# ===================== 辅助函数：效应量计算子函数 =====================
def _calculate_group_effect_size(
        data1: np.ndarray,
        data2: np.ndarray,
        effect_type: str,
        test_type: str,
        alpha: float
) -> Dict:
    """组间比较效应量（Cohen's d/Hedges' g/Glass's Δ）"""
    n1, n2 = len(data1), len(data2)
    m1, m2 = np.mean(data1), np.mean(data2)
    s1, s2 = np.std(data1, ddof=1), np.std(data2, ddof=1)  # 样本标准差

    if effect_type == "cohens_d":
        # Cohen's d（标准化均值差）
        if test_type == "independent":
            pooled_std = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
            d = (m1 - m2) / pooled_std
        else:
            diff = data1 - data2[:n1]
            diff_std = np.std(diff, ddof=1)
            d = np.mean(diff) / diff_std

        se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
        z = stats.norm.ppf(1 - alpha / 2)
        ci = (d - z * se, d + z * se)
        effect_size = d

    elif effect_type == "hedges_g":
        # Hedges' g（Cohen's d小样本校正）
        if test_type == "independent":
            pooled_std = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
            d = (m1 - m2) / pooled_std
        else:
            diff = data1 - data2[:n1]
            diff_std = np.std(diff, ddof=1)
            d = np.mean(diff) / diff_std

        df = n1 + n2 - 2 if test_type == "independent" else n1 - 1
        j = 1 - 3 / (4 * df - 1)  # 校正因子
        g = d * j

        se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
        z = stats.norm.ppf(1 - alpha / 2)
        ci = (g - z * se, g + z * se)
        effect_size = g

    elif effect_type == "glass_delta":
        # Glass's Δ（用对照组标准差标准化，适用于方差不齐）
        delta = (m1 - m2) / s2  # 用data2（对照组）的标准差
        se = np.sqrt((n1 + n2) / (n1 * n2) + delta ** 2 / (2 * (n1 + n2)))
        z = stats.norm.ppf(1 - alpha / 2)
        ci = (delta - z * se, delta + z * se)
        effect_size = delta

    return {
        "effect_type": effect_type,
        "effect_size": float(effect_size),
        "se": float(se),
        "ci": (float(ci[0]), float(ci[1])),
        "group_stats": {
            "group1_mean": float(m1),
            "group2_mean": float(m2),
            "group1_std": float(s1),
            "group2_std": float(s2)
        }
    }


def _calculate_correlation_effect_size(
        data1: np.ndarray,
        data2: np.ndarray,
        effect_type: str,
        alpha: float
) -> Dict:
    """相关分析效应量（Pearson r/R²/Cohen's f²）"""
    # Pearson r
    r, p_val = stats.pearsonr(data1, data2)
    n = len(data1)

    # Fisher z变换计算置信区间
    z_r = 0.5 * np.log((1 + r) / (1 - r))
    se_z = 1 / np.sqrt(n - 3)
    z = stats.norm.ppf(1 - alpha / 2)
    ci_z = (z_r - z * se_z, z_r + z * se_z)
    ci_r = (np.tanh(ci_z[0]), np.tanh(ci_z[1]))

    if effect_type == "pearson_r":
        effect_size = r
        se = se_z * (1 - r ** 2)  # r的标准误
        ci = ci_r

    elif effect_type == "r_squared":
        effect_size = r ** 2  # 决定系数（解释变异比例）
        se = 2 * r * se  # R²的标准误
        ci = (ci_r[0] ** 2, ci_r[1] ** 2)

    elif effect_type == "cohens_f2":
        # Cohen's f² = R² / (1 - R²)（用于回归分析）
        effect_size = r ** 2 / (1 - r ** 2) if r ** 2 < 1 else 10.0  # 避免除0
        se = (2 * r * se) / (1 - r ** 2) ** 2
        ci = (ci_r[0] ** 2 / (1 - ci_r[0] ** 2), ci_r[1] ** 2 / (1 - ci_r[1] ** 2))

    return {
        "effect_type": effect_type,
        "effect_size": float(effect_size),
        "se": float(se),
        "ci": (float(ci[0]), float(ci[1])),
        "correlation_p": float(p_val)
    }


def _calculate_categorical_effect_size(
        data1: np.ndarray,
        data2: np.ndarray,
        effect_type: str,
        alpha: float
) -> Dict:
    """分类数据效应量（OR/RR/Cramer's V）"""
    # 构造2x2列联表
    cross_tab = pd.crosstab(pd.Series(data1), pd.Series(data2))
    if cross_tab.shape != (2, 2) and effect_type in ["odds_ratio", "risk_ratio"]:
        raise ValueError("OR/RR仅支持2x2列联表（二分类数据）")

    # 提取列联表数值
    a = cross_tab.iloc[0, 0] if cross_tab.shape[0] >= 2 and cross_tab.shape[1] >= 2 else 0
    b = cross_tab.iloc[0, 1] if cross_tab.shape[0] >= 2 and cross_tab.shape[1] >= 2 else 0
    c = cross_tab.iloc[1, 0] if cross_tab.shape[0] >= 2 and cross_tab.shape[1] >= 2 else 0
    d = cross_tab.iloc[1, 1] if cross_tab.shape[0] >= 2 and cross_tab.shape[1] >= 2 else 0

    if effect_type == "odds_ratio":
        # 优势比OR = (a/b) / (c/d)
        if b == 0 or c == 0 or d == 0:
            # 避免除0，加0.5校正
            a += 0.5
            b += 0.5
            c += 0.5
            d += 0.5
        or_val = (a * d) / (b * c)
        # 对数转换计算CI
        log_or = np.log(or_val)
        se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        z = stats.norm.ppf(1 - alpha / 2)
        ci_log = (log_or - z * se_log_or, log_or + z * se_log_or)
        ci = (np.exp(ci_log[0]), np.exp(ci_log[1]))
        effect_size = or_val
        se = se_log_or

    elif effect_type == "risk_ratio":
        # 风险比RR = (a/(a+b)) / (c/(c+d))
        if a + b == 0 or c + d == 0:
            raise ValueError("风险比计算失败：组内样本量为0")
        rr_val = (a / (a + b)) / (c / (c + d))
        log_rr = np.log(rr_val)
        se_log_rr = np.sqrt(b / (a * (a + b)) + d / (c * (c + d)))
        z = stats.norm.ppf(1 - alpha / 2)
        ci_log = (log_rr - z * se_log_rr, log_rr + z * se_log_rr)
        ci = (np.exp(ci_log[0]), np.exp(ci_log[1]))
        effect_size = rr_val
        se = se_log_rr

    elif effect_type == "cramers_v":
        # Cramer's V（卡方检验效应量）
        chi2, p_val, dof, _ = stats.chi2_contingency(cross_tab)
        n = len(data1)
        cramers_v = np.sqrt(chi2 / (n * (min(cross_tab.shape) - 1)))
        # 简化CI计算（近似）
        se = cramers_v / np.sqrt(2 * n)
        z = stats.norm.ppf(1 - alpha / 2)
        ci = (max(0, cramers_v - z * se), min(1, cramers_v + z * se))
        effect_size = cramers_v
        se = se

    return {
        "effect_type": effect_type,
        "effect_size": float(effect_size),
        "se": float(se),
        "ci": (float(ci[0]), float(ci[1])),
        "contingency_table": cross_tab.to_dict()
    }


def _interpret_effect_size(effect_size: float, effect_type: str) -> str:
    """效应量大小解释（遵循Cohen, 1988行业标准）"""
    if np.isnan(effect_size):
        return "无法解释"

    effect_size_abs = abs(effect_size)

    # 组间比较（Cohen's d/Hedges' g/Glass's Δ）
    if effect_type in ["cohens_d", "hedges_g", "glass_delta"]:
        if effect_size_abs < 0.2:
            return "小效应（<0.2）"
        elif effect_size_abs < 0.5:
            return "较小效应（0.2-0.5）"
        elif effect_size_abs < 0.8:
            return "中等效应（0.5-0.8）"
        else:
            return "大效应（≥0.8）"

    # 相关分析
    elif effect_type == "pearson_r":
        if effect_size_abs < 0.1:
            return "弱相关（<0.1）"
        elif effect_size_abs < 0.3:
            return "低相关（0.1-0.3）"
        elif effect_size_abs < 0.5:
            return "中等相关（0.3-0.5）"
        else:
            return "强相关（≥0.5）"

    elif effect_type == "r_squared":
        if effect_size < 0.01:
            return "弱解释力（<1%）"
        elif effect_size < 0.09:
            return "低解释力（1%-9%）"
        elif effect_size < 0.25:
            return "中等解释力（9%-25%）"
        else:
            return "强解释力（≥25%）"

    # 分类数据
    elif effect_type == "cramers_v":
        if effect_size < 0.1:
            return "弱关联（<0.1）"
        elif effect_size < 0.3:
            return "低关联（0.1-0.3）"
        elif effect_size < 0.5:
            return "中等关联（0.3-0.5）"
        else:
            return "强关联（≥0.5）"

    # OR/RR（分类数据）
    elif effect_type in ["odds_ratio", "risk_ratio"]:
        if effect_size_abs < 1.5:
            return "弱效应（<1.5）"
        elif effect_size_abs < 3.0:
            return "中等效应（1.5-3.0）"
        else:
            return "强效应（≥3.0）"

    # ANOVA效应量
    elif effect_type == "eta_squared":
        if effect_size < 0.01:
            return "小效应（<1%）"
        elif effect_size < 0.06:
            return "中等效应（1%-6%）"
        else:
            return "大效应（≥6%）"

    else:
        return "无通用解释标准"


# ===================== 测试用例（与reliability_meta.py联动） =====================
if __name__ == "__main__":
    # 模拟BCI数据（EEG特征，实验组vs对照组）
    np.random.seed(42)
    # 实验组（想象运动）
    group1 = np.random.normal(5.2, 1.2, 50)
    # 对照组（静息态）
    group2 = np.random.normal(4.0, 1.0, 50)
    # 相关分析数据（特征1 vs 行为指标）
    feature_data = np.random.normal(0, 1, 100)
    behavior_data = 0.6 * feature_data + np.random.normal(0, 0.8, 100)
    # 分类数据（二分类：有效/无效）
    categorical1 = np.random.choice([0, 1], size=100, p=[0.3, 0.7])
    categorical2 = np.random.choice([0, 1], size=100, p=[0.5, 0.5])

    # 1. 效应量计算示例
    print("=== 1. 效应量计算 ===")
    d_result = calculate_effect_size(group1, group2, "cohens_d", "independent")
    print(f"Cohen's d：{d_result['effect_size']:.4f} | 解释：{d_result['interpretation']}")

    r_result = calculate_effect_size(feature_data, behavior_data, "pearson_r")
    print(f"Pearson r：{r_result['effect_size']:.4f} | 解释：{r_result['interpretation']}")

    # 2. 统计检验示例（联动效应量）
    print("\n=== 2. 统计检验 ===")
    t_test_result = statistical_test(group1, group2, "t_test", paired=False)
    print(f"t检验 p值：{t_test_result['p_value']:.4f} | 显著：{t_test_result['significant']}")
    print(f"对应效应量：{t_test_result['effect_size']['effect_size']:.4f}")

    # 3. 功效分析示例（估算样本量）
    print("\n=== 3. 功效分析 ===")
    power_result = power_analysis(effect_size=0.5, alpha=0.05, power=0.8, test_type="t_test")
    print(f"所需总样本量：{power_result['total_sample_size']} | 解释：{power_result['interpretation']}")

    # 4. 与元分析联动（为reliability_meta.py提供效应量）
    print("\n=== 4. 与元分析联动示例 ===")
    # 模拟5个研究的效应量和标准误
    study_effects = [calculate_effect_size(
        np.random.normal(5 + i * 0.1, 1.2, 40),
        np.random.normal(4, 1.0, 40),
        "cohens_d"
    ) for i in range(5)]
    effect_sizes = [res["effect_size"] for res in study_effects]
    standard_errors = [res["se"] for res in study_effects]
    print(f"5个研究的效应量：{[f'{e:.4f}' for e in effect_sizes]}")
    print("可直接传入reliability_meta.py的meta_analysis函数使用")