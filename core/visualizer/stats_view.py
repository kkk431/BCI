#!/usr/bin/env python3
"""
stats_view.py
统计分析视图 - 箱线图/ROC曲线/混淆矩阵
基于统计分析模块的输出结果
"""

import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QGroupBox,
                             QGridLayout, QTabWidget, QFileDialog, QMessageBox,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QSplitter, QCheckBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any


class StatsView(QMainWindow):
    """
    统计分析视图类
    显示统计模块的输出结果
    """

    def __init__(self, data_dict: Dict[str, Any] = None, stats_file: str = None, parent=None):
        """
        初始化统计视图

        Args:
            data_dict: 包含统计结果的四层数据字典
            stats_file: 直接加载统计结果文件
        """
        super().__init__(parent)

        self.data_dict = data_dict
        self.stats_file = stats_file
        self.stats_results = {}

        # 加载数据
        self._load_data()

        # 设置窗口
        self.setWindowTitle("统计分析视图")
        self.resize(1100, 800)

        # 设置UI
        self.setup_ui()

        # 更新显示
        self.update_plots()

    def _load_data(self):
        """加载统计数据"""
        if self.stats_file:
            # 从文件加载
            try:
                if self.stats_file.endswith('.json'):
                    with open(self.stats_file, 'r', encoding='utf-8') as f:
                        self.stats_results = json.load(f)
                elif self.stats_file.endswith(('.pkl', '.pickle')):
                    import pickle
                    with open(self.stats_file, 'rb') as f:
                        self.stats_results = pickle.load(f)
                elif self.stats_file.endswith('.csv'):
                    df = pd.read_csv(self.stats_file)
                    self.stats_results = df.to_dict(orient='list')
            except Exception as e:
                QMessageBox.warning(self, "加载失败", f"无法加载统计文件: {str(e)}")
                self.stats_results = self._create_demo_data()

        elif self.data_dict:
            # 从数据字典加载
            processed = self.data_dict.get("processed", {})
            self.stats_results = processed.get("stat_results", {})

        # 如果没有数据，创建演示数据
        if not self.stats_results:
            self.stats_results = self._create_demo_data()

    def _create_demo_data(self) -> Dict:
        """创建演示数据"""
        np.random.seed(42)

        # 箱线图数据 - 两组对比
        group1 = np.random.normal(10, 2, 30)
        group2 = np.random.normal(12, 2.5, 30)
        group3 = np.random.normal(9, 1.8, 30)

        # ROC数据
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = np.concatenate([np.random.normal(0.3, 0.2, 50),
                                  np.random.normal(0.7, 0.2, 50)])

        # 混淆矩阵
        y_pred = (y_score > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        return {
            "boxplot": {
                "data": [group1.tolist(), group2.tolist(), group3.tolist()],
                "labels": ["左手", "右手", "脚"],
                "colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
                "title": "不同运动想象任务的EEG功率"
            },
            "roc": {
                "y_true": y_true.tolist(),
                "y_score": y_score.tolist(),
                "title": "分类器性能 (左手 vs 右手)",
                "auc": 0.85
            },
            "confusion_matrix": {
                "matrix": cm.tolist(),
                "labels": ["左手", "右手"],
                "title": "混淆矩阵"
            },
            "statistics": {
                "t_test": {
                    "t_statistic": 3.24,
                    "p_value": 0.002,
                    "significant": True
                },
                "anova": {
                    "f_statistic": 5.67,
                    "p_value": 0.004,
                    "significant": True
                }
            }
        }

    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # ========== 顶部控制栏 ==========
        control_layout = QHBoxLayout()

        control_layout.addWidget(QLabel("分析类型:"))

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["箱线图", "ROC曲线", "混淆矩阵", "所有"])
        self.plot_type_combo.currentTextChanged.connect(self.update_plots)
        control_layout.addWidget(self.plot_type_combo)

        control_layout.addStretch()

        # 显著性水平
        control_layout.addWidget(QLabel("显著性水平 α:"))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.001, 0.1)
        self.alpha_spin.setValue(0.05)
        self.alpha_spin.setSingleStep(0.005)
        self.alpha_spin.valueChanged.connect(self.update_plots)
        control_layout.addWidget(self.alpha_spin)

        # 保存按钮
        save_btn = QPushButton("保存图表")
        save_btn.clicked.connect(self.save_plots)
        control_layout.addWidget(save_btn)

        main_layout.addLayout(control_layout)

        # ========== 中间：绘图区域 ==========
        # 创建选项卡
        self.tab_widget = QTabWidget()

        # 箱线图选项卡
        self.boxplot_widget = QWidget()
        self.setup_boxplot_tab()
        self.tab_widget.addTab(self.boxplot_widget, "箱线图")

        # ROC曲线选项卡
        self.roc_widget = QWidget()
        self.setup_roc_tab()
        self.tab_widget.addTab(self.roc_widget, "ROC曲线")

        # 混淆矩阵选项卡
        self.cm_widget = QWidget()
        self.setup_cm_tab()
        self.tab_widget.addTab(self.cm_widget, "混淆矩阵")

        # 统计表格选项卡
        self.stats_table_widget = QWidget()
        self.setup_stats_table_tab()
        self.tab_widget.addTab(self.stats_table_widget, "统计结果")

        main_layout.addWidget(self.tab_widget, 1)

    def setup_boxplot_tab(self):
        """设置箱线图选项卡"""
        layout = QHBoxLayout(self.boxplot_widget)

        # 左侧控制面板
        control_panel = QWidget()
        control_panel.setMaximumWidth(250)
        control_layout = QVBoxLayout(control_panel)

        control_layout.addWidget(QLabel("数据组选择:"))

        self.boxplot_list = QComboBox()
        self.boxplot_list.addItems(["所有组", "组1 vs 组2", "组1 vs 组3", "组2 vs 组3"])
        control_layout.addWidget(self.boxplot_list)

        control_layout.addWidget(QLabel("显示选项:"))

        self.show_points_check = QCheckBox("显示数据点")
        self.show_points_check.setChecked(True)
        self.show_points_check.toggled.connect(self.update_plots)
        control_layout.addWidget(self.show_points_check)

        self.show_stats_check = QCheckBox("显示统计标记")
        self.show_stats_check.setChecked(True)
        self.show_stats_check.toggled.connect(self.update_plots)
        control_layout.addWidget(self.show_stats_check)

        control_layout.addStretch()

        layout.addWidget(control_panel)

        # 右侧绘图区域
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)

        self.boxplot_figure = Figure(figsize=(8, 6), dpi=100)
        self.boxplot_canvas = FigureCanvas(self.boxplot_figure)
        self.boxplot_toolbar = NavigationToolbar(self.boxplot_canvas, self)

        plot_layout.addWidget(self.boxplot_toolbar)
        plot_layout.addWidget(self.boxplot_canvas)

        layout.addWidget(plot_panel, 1)

    def setup_roc_tab(self):
        """设置ROC曲线选项卡"""
        layout = QHBoxLayout(self.roc_widget)

        # 左侧控制面板
        control_panel = QWidget()
        control_panel.setMaximumWidth(250)
        control_layout = QVBoxLayout(control_panel)

        control_layout.addWidget(QLabel("ROC曲线选项:"))

        self.show_auc_check = QCheckBox("显示AUC值")
        self.show_auc_check.setChecked(True)
        self.show_auc_check.toggled.connect(self.update_plots)
        control_layout.addWidget(self.show_auc_check)

        self.show_diag_check = QCheckBox("显示对角线")
        self.show_diag_check.setChecked(True)
        self.show_diag_check.toggled.connect(self.update_plots)
        control_layout.addWidget(self.show_diag_check)

        control_layout.addStretch()

        layout.addWidget(control_panel)

        # 右侧绘图区域
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)

        self.roc_figure = Figure(figsize=(8, 6), dpi=100)
        self.roc_canvas = FigureCanvas(self.roc_figure)
        self.roc_toolbar = NavigationToolbar(self.roc_canvas, self)

        plot_layout.addWidget(self.roc_toolbar)
        plot_layout.addWidget(self.roc_canvas)

        layout.addWidget(plot_panel, 1)

    def setup_cm_tab(self):
        """设置混淆矩阵选项卡"""
        layout = QHBoxLayout(self.cm_widget)

        # 左侧控制面板
        control_panel = QWidget()
        control_panel.setMaximumWidth(250)
        control_layout = QVBoxLayout(control_panel)

        control_layout.addWidget(QLabel("混淆矩阵选项:"))

        self.show_numbers_check = QCheckBox("显示数值")
        self.show_numbers_check.setChecked(True)
        self.show_numbers_check.toggled.connect(self.update_plots)
        control_layout.addWidget(self.show_numbers_check)

        self.show_percent_check = QCheckBox("显示百分比")
        self.show_percent_check.setChecked(False)
        self.show_percent_check.toggled.connect(self.update_plots)
        control_layout.addWidget(self.show_percent_check)

        control_layout.addWidget(QLabel("颜色映射:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'viridis'])
        self.cmap_combo.currentTextChanged.connect(self.update_plots)
        control_layout.addWidget(self.cmap_combo)

        control_layout.addStretch()

        layout.addWidget(control_panel)

        # 右侧绘图区域
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)

        self.cm_figure = Figure(figsize=(6, 6), dpi=100)
        self.cm_canvas = FigureCanvas(self.cm_figure)
        self.cm_toolbar = NavigationToolbar(self.cm_canvas, self)

        plot_layout.addWidget(self.cm_toolbar)
        plot_layout.addWidget(self.cm_canvas)

        layout.addWidget(plot_panel, 1)

    def setup_stats_table_tab(self):
        """设置统计结果表格选项卡"""
        layout = QVBoxLayout(self.stats_table_widget)

        # 统计表格
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(5)
        self.stats_table.setHorizontalHeaderLabels(["统计量", "数值", "自由度", "p值", "显著性"])

        # 设置列宽
        header = self.stats_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 5):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        layout.addWidget(self.stats_table)

        # 填充表格
        self.populate_stats_table()

    def populate_stats_table(self):
        """填充统计结果表格"""
        stats_data = self.stats_results.get("statistics", {})
        self.stats_table.setRowCount(0)

        row = 0
        for test_name, test_result in stats_data.items():
            if isinstance(test_result, dict):
                self.stats_table.insertRow(row)
                self.stats_table.setItem(row, 0, QTableWidgetItem(test_name))

                # 根据不同测试类型显示
                if "t_statistic" in test_result:
                    self.stats_table.setItem(row, 1, QTableWidgetItem(f"{test_result['t_statistic']:.4f}"))
                    self.stats_table.setItem(row, 2, QTableWidgetItem(str(test_result.get('df', 'N/A'))))
                    self.stats_table.setItem(row, 3, QTableWidgetItem(f"{test_result['p_value']:.4f}"))

                    # 显著性标记
                    sig_item = QTableWidgetItem("★" if test_result.get('significant', False) else "")
                    if test_result.get('significant', False):
                        sig_item.setForeground(QColor(255, 0, 0))
                    self.stats_table.setItem(row, 4, sig_item)

                elif "f_statistic" in test_result:
                    self.stats_table.setItem(row, 1, QTableWidgetItem(f"{test_result['f_statistic']:.4f}"))
                    self.stats_table.setItem(row, 2, QTableWidgetItem(
                        f"{test_result.get('df1', 'N/A')}, {test_result.get('df2', 'N/A')}"))
                    self.stats_table.setItem(row, 3, QTableWidgetItem(f"{test_result['p_value']:.4f}"))

                    sig_item = QTableWidgetItem("★" if test_result.get('significant', False) else "")
                    if test_result.get('significant', False):
                        sig_item.setForeground(QColor(255, 0, 0))
                    self.stats_table.setItem(row, 4, sig_item)

                row += 1

    def update_plots(self):
        """更新所有图表"""
        plot_type = self.plot_type_combo.currentText()

        if plot_type in ["箱线图", "所有"]:
            self.update_boxplot()
        if plot_type in ["ROC曲线", "所有"]:
            self.update_roc()
        if plot_type in ["混淆矩阵", "所有"]:
            self.update_cm()

        # 更新选项卡
        if plot_type == "箱线图":
            self.tab_widget.setCurrentIndex(0)
        elif plot_type == "ROC曲线":
            self.tab_widget.setCurrentIndex(1)
        elif plot_type == "混淆矩阵":
            self.tab_widget.setCurrentIndex(2)

    def update_boxplot(self):
        """更新箱线图"""
        self.boxplot_figure.clear()
        ax = self.boxplot_figure.add_subplot(111)

        # 获取数据
        boxplot_data = self.stats_results.get("boxplot", {})
        data = boxplot_data.get("data", [])
        labels = boxplot_data.get("labels", [])
        colors = boxplot_data.get("colors", ["#1f77b4"] * len(data))
        title = boxplot_data.get("title", "箱线图")

        if not data:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', transform=ax.transAxes)
            self.boxplot_canvas.draw()
            return

        # 绘制箱线图
        bp = ax.boxplot(data, patch_artist=True, labels=labels, showfliers=False)

        # 设置颜色
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 显示数据点
        if self.show_points_check.isChecked():
            for i, d in enumerate(data):
                x = np.random.normal(i + 1, 0.04, size=len(d))
                ax.plot(x, d, 'o', color='black', alpha=0.5, markersize=3)

        # 显示统计标记
        if self.show_stats_check.isChecked() and len(data) >= 2:
            alpha = self.alpha_spin.value()

            # 进行t检验
            for i in range(len(data)):
                for j in range(i + 1, len(data)):
                    t_stat, p_val = stats.ttest_ind(data[i], data[j])

                    if p_val < alpha:
                        # 在两组之间添加显著性标记
                        y_max = max(np.max(data[i]), np.max(data[j]))
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        y_pos = y_max + y_range * 0.1

                        x1, x2 = i + 1, j + 1
                        ax.plot([x1, x1, x2, x2], [y_pos, y_pos + y_range * 0.03,
                                                   y_pos + y_range * 0.03, y_pos],
                                'k-', linewidth=1)

                        # 添加星号
                        stars = '*' * sum([p_val < lvl for lvl in [0.001, 0.01, 0.05]])
                        ax.text((x1 + x2) / 2, y_pos + y_range * 0.04, stars,
                                ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_title(title, fontsize=14)
        ax.set_ylabel("值")
        ax.grid(True, alpha=0.3, axis='y')

        self.boxplot_figure.tight_layout()
        self.boxplot_canvas.draw()

    def update_roc(self):
        """更新ROC曲线"""
        self.roc_figure.clear()
        ax = self.roc_figure.add_subplot(111)

        # 获取数据
        roc_data = self.stats_results.get("roc", {})
        y_true = np.array(roc_data.get("y_true", []))
        y_score = np.array(roc_data.get("y_score", []))
        title = roc_data.get("title", "ROC曲线")

        if len(y_true) == 0 or len(y_score) == 0:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', transform=ax.transAxes)
            self.roc_canvas.draw()
            return

        # 计算ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # 绘制ROC曲线
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')

        # 显示对角线
        if self.show_diag_check.isChecked():
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='随机分类器')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('假阳性率 (FPR)', fontsize=12)
        ax.set_ylabel('真阳性率 (TPR)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        self.roc_figure.tight_layout()
        self.roc_canvas.draw()

    def update_cm(self):
        """更新混淆矩阵"""
        self.cm_figure.clear()
        ax = self.cm_figure.add_subplot(111)

        # 获取数据
        cm_data = self.stats_results.get("confusion_matrix", {})
        cm = np.array(cm_data.get("matrix", []))
        labels = cm_data.get("labels", ["类别1", "类别2"])
        title = cm_data.get("title", "混淆矩阵")

        if cm.size == 0:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', transform=ax.transAxes)
            self.cm_canvas.draw()
            return

        # 绘制混淆矩阵
        cmap = plt.cm.get_cmap(self.cmap_combo.currentText())
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

        # 添加颜色条
        self.cm_figure.colorbar(im, ax=ax)

        # 设置刻度
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # 添加数值
        if self.show_numbers_check.isChecked() or self.show_percent_check.isChecked():
            total = np.sum(cm)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if self.show_percent_check.isChecked():
                        text = f"{cm[i, j]}\n({cm[i, j] / total * 100:.1f}%)"
                    else:
                        text = str(cm[i, j])

                    color = "white" if cm[i, j] > cm.max() / 2 else "black"
                    ax.text(j, i, text, ha="center", va="center", color=color)

        ax.set_xlabel('预测标签', fontsize=12)
        ax.set_ylabel('真实标签', fontsize=12)
        ax.set_title(title, fontsize=14)

        self.cm_figure.tight_layout()
        self.cm_canvas.draw()

    def save_plots(self):
        """保存图表"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图表", "", "PNG图像 (*.png);;PDF文件 (*.pdf);;SVG图像 (*.svg)")

        if not file_path:
            return

        try:
            # 保存当前显示的图表
            current_tab = self.tab_widget.currentIndex()
            if current_tab == 0:
                self.boxplot_figure.savefig(file_path, dpi=300, bbox_inches='tight')
            elif current_tab == 1:
                self.roc_figure.savefig(file_path, dpi=300, bbox_inches='tight')
            elif current_tab == 2:
                self.cm_figure.savefig(file_path, dpi=300, bbox_inches='tight')

            QMessageBox.information(self, "保存成功", f"图表已保存到:\n{file_path}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"保存图表时出错:\n{str(e)}")


# 测试代码
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    view = StatsView()
    view.show()
    sys.exit(app.exec_())