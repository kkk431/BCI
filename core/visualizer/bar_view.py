#!/usr/bin/env python3
"""
bar_view.py
柱状图视图 - 特征对比/Excel数据可视化
支持多子图柱状图和数值标签
"""

import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QGroupBox,
                             QGridLayout, QFileDialog, QMessageBox, QSpinBox,
                             QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QSplitter, QTabWidget, QDoubleSpinBox, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

import pandas as pd
import os
from typing import Dict, List, Optional, Tuple, Any


class BarView(QMainWindow):
    """
    柱状图视图类
    支持特征对比和Excel数据可视化
    """

    def __init__(self, data_dict: Dict[str, Any] = None, excel_file: str = None, parent=None):
        """
        初始化柱状图视图

        Args:
            data_dict: 包含特征数据的数据字典
            excel_file: Excel文件路径
        """
        super().__init__(parent)

        self.data_dict = data_dict
        self.excel_file = excel_file
        self.data = None
        self.column_names = []

        # 加载数据
        self._load_data()

        # 设置窗口
        self.setWindowTitle("柱状图视图")
        self.resize(1200, 800)

        # 设置UI
        self.setup_ui()

        # 更新图表
        self.update_plot()

    def _load_data(self):
        """加载数据"""
        if self.excel_file and os.path.exists(self.excel_file):
            try:
                # 从Excel文件加载
                self.df = pd.read_excel(self.excel_file)
                self.column_names = self.df.columns.tolist()
                self.data_source = "excel"
                self.file_name = os.path.basename(self.excel_file)

                QMessageBox.information(self, "加载成功",
                                        f"已加载Excel文件:\n{self.excel_file}\n"
                                        f"共{len(self.df)}行，{len(self.column_names)}列")

            except Exception as e:
                QMessageBox.warning(self, "加载失败", f"无法加载Excel文件:\n{str(e)}")
                self._create_demo_data()

        elif self.data_dict:
            # 从数据字典的特征中加载
            processed = self.data_dict.get("processed", {})
            features = processed.get("features", {})

            if features:
                # 将特征字典转换为DataFrame
                data_rows = []

                # 提取通道和特征
                feature_keys = list(features.keys())
                channels = set()
                feature_names = set()

                for key in feature_keys:
                    parts = key.split('_', 1)
                    if len(parts) == 2:
                        channels.add(parts[0])
                        feature_names.add(parts[1])

                channels = sorted(list(channels))
                feature_names = sorted(list(feature_names))

                # 构建DataFrame
                data = []
                for ch in channels:
                    row = [ch]
                    for feat in feature_names:
                        key = f"{ch}_{feat}"
                        row.append(features.get(key, np.nan))
                    data.append(row)

                self.df = pd.DataFrame(data, columns=['channel'] + feature_names)
                self.column_names = self.df.columns.tolist()
                self.data_source = "features"

                QMessageBox.information(self, "加载成功",
                                        f"已从数据字典加载特征\n"
                                        f"共{len(channels)}个通道，{len(feature_names)}个特征")
            else:
                self._create_demo_data()
        else:
            self._create_demo_data()

    def _create_demo_data(self):
        """创建演示数据"""
        # 创建示例特征数据
        channels = ['Fz', 'Cz', 'Pz', 'Oz', 'F3', 'F4', 'C3', 'C4']
        features = ['mean', 'std', 'alpha_power', 'beta_power', 'theta_power']

        np.random.seed(42)
        data = []
        for ch in channels:
            row = [ch]
            for _ in features:
                row.append(np.random.rand() * 100)
            data.append(row)

        self.df = pd.DataFrame(data, columns=['channel'] + features)
        self.column_names = self.df.columns.tolist()
        self.data_source = "demo"

        QMessageBox.information(self, "演示模式", "使用演示数据进行展示")

    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # ========== 顶部控制栏 ==========
        control_layout = QHBoxLayout()

        # 文件信息
        if hasattr(self, 'file_name'):
            info_label = QLabel(f"文件: {self.file_name}")
        else:
            info_label = QLabel(f"数据源: {getattr(self, 'data_source', '未知')}")
        info_label.setFont(QFont("Microsoft YaHei", 10))
        control_layout.addWidget(info_label)

        control_layout.addStretch()

        # 打开文件按钮
        open_btn = QPushButton("打开Excel文件")
        open_btn.clicked.connect(self.open_excel_file)
        control_layout.addWidget(open_btn)

        # 刷新按钮
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.update_plot)
        control_layout.addWidget(refresh_btn)

        main_layout.addLayout(control_layout)

        # ========== 中间：数据选择 + 绘图 ==========
        splitter = QSplitter(Qt.Horizontal)

        # 左侧控制面板
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_layout = QVBoxLayout(control_panel)

        # X轴选择
        x_group = QGroupBox("X轴 (类别)")
        x_layout = QVBoxLayout(x_group)

        self.x_combo = QComboBox()
        self.x_combo.addItems(self.column_names)
        self.x_combo.currentTextChanged.connect(self.on_x_changed)
        x_layout.addWidget(self.x_combo)

        control_layout.addWidget(x_group)

        # Y轴选择（多选）
        y_group = QGroupBox("Y轴 (数值列)")
        y_layout = QVBoxLayout(y_group)

        self.y_list = QComboBox()  # 改为下拉框便于测试
        self.y_list.addItems([col for col in self.column_names if col != self.x_combo.currentText()])
        self.y_list.currentTextChanged.connect(self.update_plot)
        y_layout.addWidget(self.y_list)

        control_layout.addWidget(y_group)

        # 图表设置
        plot_group = QGroupBox("图表设置")
        plot_layout = QGridLayout(plot_group)

        plot_layout.addWidget(QLabel("标题:"), 0, 0)
        self.title_edit = QLineEdit("特征对比柱状图")
        plot_layout.addWidget(self.title_edit, 0, 1)

        plot_layout.addWidget(QLabel("X轴标签:"), 1, 0)
        self.xlabel_edit = QLineEdit("通道")
        plot_layout.addWidget(self.xlabel_edit, 1, 1)

        plot_layout.addWidget(QLabel("Y轴标签:"), 2, 0)
        self.ylabel_edit = QLineEdit("值")
        plot_layout.addWidget(self.ylabel_edit, 2, 1)

        plot_layout.addWidget(QLabel("柱子宽度:"), 3, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.1, 1.0)
        self.width_spin.setValue(0.6)
        self.width_spin.setSingleStep(0.1)
        self.width_spin.valueChanged.connect(self.update_plot)
        plot_layout.addWidget(self.width_spin, 3, 1)

        control_layout.addWidget(plot_group)

        # 显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout(display_group)

        self.show_values_check = QCheckBox("显示数值标签")
        self.show_values_check.setChecked(True)
        self.show_values_check.toggled.connect(self.update_plot)
        display_layout.addWidget(self.show_values_check)

        self.show_grid_check = QCheckBox("显示网格")
        self.show_grid_check.setChecked(True)
        self.show_grid_check.toggled.connect(self.update_plot)
        display_layout.addWidget(self.show_grid_check)

        self.rotate_x_check = QCheckBox("旋转X轴标签")
        self.rotate_x_check.setChecked(False)
        self.rotate_x_check.toggled.connect(self.update_plot)
        display_layout.addWidget(self.rotate_x_check)

        control_layout.addWidget(display_group)

        control_layout.addStretch()

        splitter.addWidget(control_panel)

        # 右侧绘图区域 + 数据表格
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 创建选项卡
        tab_widget = QTabWidget()

        # 绘图选项卡
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        tab_widget.addTab(plot_widget, "柱状图")

        # 数据表格选项卡
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)

        self.data_table = QTableWidget()
        self.data_table.setColumnCount(len(self.column_names))
        self.data_table.setHorizontalHeaderLabels(self.column_names)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 填充表格
        self.populate_table()

        table_layout.addWidget(self.data_table)

        tab_widget.addTab(table_widget, "数据表格")

        right_layout.addWidget(tab_widget)

        splitter.addWidget(right_panel)
        splitter.setSizes([300, 800])

        main_layout.addWidget(splitter, 1)

    def populate_table(self):
        """填充数据表格"""
        self.data_table.setRowCount(len(self.df))

        for i, row in self.df.iterrows():
            for j, col in enumerate(self.column_names):
                value = row[col]
                if isinstance(value, (int, float)):
                    item = QTableWidgetItem(f"{value:.4f}" if abs(value) < 1000 else f"{value:.2e}")
                else:
                    item = QTableWidgetItem(str(value))
                self.data_table.setItem(i, j, item)

    def on_x_changed(self, x_col):
        """X轴列改变"""
        # 更新Y轴列表，排除X轴列
        self.y_list.clear()
        self.y_list.addItems([col for col in self.column_names if col != x_col])
        self.update_plot()

    def open_excel_file(self):
        """打开Excel文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择Excel文件", "", "Excel文件 (*.xlsx *.xls);;所有文件 (*)")

        if file_path:
            self.excel_file = file_path
            self._load_data()

            # 更新UI
            self.x_combo.clear()
            self.x_combo.addItems(self.column_names)
            self.on_x_changed(self.x_combo.currentText())
            self.populate_table()
            self.update_plot()

    def update_plot(self):
        """更新柱状图"""
        self.figure.clear()

        x_col = self.x_combo.currentText()
        y_col = self.y_list.currentText()

        if not x_col or not y_col or x_col == y_col:
            return

        # 获取数据
        x_categories = self.df[x_col].tolist()
        y_values = self.df[y_col].tolist()

        # 转换为数值（如果是字符串）
        if not all(isinstance(v, (int, float)) for v in y_values):
            try:
                y_values = [float(v) if v else 0 for v in y_values]
            except:
                QMessageBox.warning(self, "错误", f"列 '{y_col}' 包含非数值数据")
                return

        ax = self.figure.add_subplot(111)

        # 创建x轴位置
        x_pos = np.arange(len(x_categories))

        # 绘制柱状图
        bars = ax.bar(x_pos, y_values, width=self.width_spin.value(),
                      color='steelblue', edgecolor='black', alpha=0.7)

        # 设置x轴标签
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_categories, rotation=45 if self.rotate_x_check.isChecked() else 0)

        # 设置标签和标题
        ax.set_xlabel(self.xlabel_edit.text(), fontsize=12)
        ax.set_ylabel(self.ylabel_edit.text(), fontsize=12)
        ax.set_title(self.title_edit.text(), fontsize=14)

        # 显示网格
        if self.show_grid_check.isChecked():
            ax.grid(True, alpha=0.3, axis='y')

        # 显示数值标签
        if self.show_values_check.isChecked():
            for i, (bar, val) in enumerate(zip(bars, y_values)):
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                offset = 3 if height >= 0 else -3
                ax.text(bar.get_x() + bar.get_width() / 2., height + offset,
                        f'{val:.2f}', ha='center', va=va, fontsize=8, rotation=90 if len(x_categories) > 8 else 0)

        # 如果有多个Y列（这里简化，只显示一列）
        # 实际应用中可能需要多组柱状图

        self.figure.tight_layout()
        self.canvas.draw()

    def save_plot(self):
        """保存图表"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图表", "", "PNG图像 (*.png);;PDF文件 (*.pdf);;SVG图像 (*.svg)")

        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "保存成功", f"图表已保存到:\n{file_path}")
            except Exception as e:
                QMessageBox.warning(self, "保存失败", f"保存图表时出错:\n{str(e)}")


# 测试代码
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    view = BarView()
    view.show()
    sys.exit(app.exec_())