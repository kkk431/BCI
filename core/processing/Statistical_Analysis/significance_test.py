import scipy.stats as stats
import numpy as np
from itertools import combinations
from statsmodels.stats.multitest import multipletests


def calculate_significance(data_dict, method="t-test", paired=False):
    """
    Perform statistical tests between group means or distributions.
    :param data_dict: Dictionary containing data groups with group names as keys
    :type data_dict: dict
    :param method: Statistical method to use (default: "t-test")
    :type method: str
    :param paired: Flag for paired/dependent samples (default: False)
    :type paired: bool
    :return: List of results containing test statistics and p-values
    :rtype: list
    :raises ValueError: For insufficient groups or invalid method names
    """
    groups = list(data_dict.keys())
    results = []

    if len(groups) < 2:
        raise ValueError("At least two groups required for comparison")

    # ---------- 方法名标准化处理 ----------
    # 去除首尾空白、替换连字符为空格、去除所有空格、转为小写
    method_clean = method.strip().replace('-', ' ').replace(' ', '').lower()

    # 定义标准方法名映射（键为标准化后的字符串，值为(实际方法名, 是否需要配对标记)）
    method_map = {
        'ttest': ('t-test', False),
        'ttestpaired': ('t-test', True),
        'anova': ('anova', False),
        'mannwhitneyu': ('mann-whitney U', False),
        'wilcoxon': ('wilcoxon', False),
        'wilcoxonpaired': ('wilcoxon', True),
        'kruskalwallis': ('kruskal-wallis', False),
    }

    if method_clean not in method_map:
        raise ValueError(f"Unsupported statistical method: {method}")

    actual_method, requires_paired = method_map[method_clean]
    # 如果方法要求配对但用户未设置，抛出错误
    if requires_paired and not paired:
        raise ValueError(f"{actual_method} test requires paired data")

    # ---------- 两两比较 ----------
    for group1, group2 in combinations(groups, 2):
        result = {}
        data1 = data_dict[group1]
        data2 = data_dict[group2]

        if actual_method == "t-test":
            if paired:
                stat, p_value = stats.ttest_rel(data1, data2)
                result["method"] = "paired t-test"
            else:
                stat, p_value = stats.ttest_ind(data1, data2)
                result["method"] = "independent t-test"

        elif actual_method == "anova":
            stat, p_value = stats.f_oneway(data1, data2)
            result["method"] = "ANOVA"

        elif actual_method == "mann-whitney U":
            stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            result["method"] = "Mann-Whitney U test"

        elif actual_method == "wilcoxon":
            if not paired:
                raise ValueError("Wilcoxon test requires paired data")
            if len(data1) != len(data2):
                raise ValueError("Equal sample size required for Wilcoxon test")
            differences = np.array(data1) - np.array(data2)
            try:
                stat, p_value = stats.wilcoxon(differences)
                result["method"] = "Wilcoxon signed-rank test"
            except ValueError as e:
                raise ValueError(f"Wilcoxon test error: {str(e)}")

        elif actual_method == "kruskal-wallis":
            stat, p_value = stats.kruskal(data1, data2)
            result["method"] = "Kruskal-Wallis test"

        result["group_comparison"] = f"{group1} vs {group2}"
        result["stat"] = stat
        result["p_value"] = p_value
        results.append(result)

    return results

def multiple_comparison_correction(results, correction_method="bonferroni"):
    """
    Apply multiple comparison correction to raw p-values.
    :param results: Output from calculate_significance function
    :type results: list
    :param correction_method: Correction method (default: "bonferroni")
    :type correction_method: str
    :return: Results with corrected p-values and significance flags
    :rtype: list
    """
    p_values = [result["p_value"] for result in results]

    # Skip correction if ≤2 comparisons
    if len(p_values) <= 1:
        for result in results:
            result["corrected_p_value"] = result["p_value"]
            result["significant_after_correction"] = result["p_value"] < 0.05
        return results

    # Apply chosen correction method
    corrected = multipletests(p_values, method=correction_method)
    corrected_p_vals = corrected[1]  # Adjusted p-values
    significance_flags = corrected[0]  # Significance indicators

    # Update results with corrected values
    for i, result in enumerate(results):
        result["corrected_p_value"] = corrected_p_vals[i]
        result["significant_after_correction"] = significance_flags[i]

    return results


if __name__ == '__main__':
    # Demonstration of statistical functions (对齐原始代码测试逻辑)
    np.random.seed(42)  # Set random seed for reproducibility
    n = 50  # Sample size per group

    # Generate synthetic group data
    group_data = {
        "Group A": np.random.normal(loc=50, scale=10, size=n),
        "Group B": np.random.normal(loc=52, scale=10, size=n),
        "Group C": np.random.normal(loc=52, scale=10, size=n)
    }

    # Compute significance tests (适配修改后的方法名mann-whitney U)
    test_results = calculate_significance(group_data, method="mann-whitney U")
    # Apply multiple comparison correction
    corrected_results = multiple_comparison_correction(
        test_results,
        correction_method="fdr_bh"
    )

    print(corrected_results)