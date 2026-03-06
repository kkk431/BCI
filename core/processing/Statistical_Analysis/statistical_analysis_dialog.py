import json
import os
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
start_path = Path(__file__).resolve().parent
for parent in [start_path] + list(start_path.parents):
    if parent.name == 'core':
        project_root = parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            print(f"子进程: 已将项目根目录 {project_root} 添加到 sys.path")
        break
else:
    raise RuntimeError("子进程: 未找到名为 'core' 的目录")
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# PyQt5 组件导入
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QPushButton, QListWidget, QFileDialog, QDialog,
                             QLineEdit, QLabel, QMessageBox, QCheckBox, QComboBox,
                             QScrollArea, QFrame, QFormLayout, QColorDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

# 项目内模块导入（使用自定义的 read_xlsx 替代 DataLoader）
from core.processing.Statistical_Analysis.file_io_fallback import read_xlsx
from core.processing.Statistical_Analysis.significance_test import calculate_significance, multiple_comparison_correction
from core.processing.Statistical_Analysis.statistical_plot import (ScatterPlotWindow, DensityHistogramWindow,
                                    SignificanceBoxPlotWindow, SignificanceViolinPlotWindow)
# 临时使用 QPushButton 代替 BFPushButton，直到找到真实路径
BFPushButton = QPushButton


def get_feature(feature_dict, channel_name=None, feature_name=None):
    """
    Extract specific feature data from a feature dictionary.
    Ensures returned data is always a 1D list of sample values.
    """
    if feature_name not in feature_dict["feature"]:
        raise ValueError(f"Feature '{feature_name}' not found")

    feature_data = feature_dict["feature"][feature_name]

    if channel_name is not None:
        # 如果指定了通道，必须存在通道信息
        if "ch_names" not in feature_dict:
            raise ValueError("Data has no channel information, cannot select channel")
        if channel_name not in feature_dict["ch_names"]:
            raise ValueError(f"Channel '{channel_name}' not found")
        channel_index = feature_dict["ch_names"].index(channel_name)
        data = feature_data[channel_index]          # 取该通道的所有样本
    else:
        data = feature_data                         # 无通道时，直接取整个特征数据

    # 统一转换为 1D 列表：处理可能出现的单个数值、numpy数组或嵌套列表
    if isinstance(data, (np.ndarray, list)):
        # 使用 ravel 展平（如果已经是1D则不变），再转列表
        data = np.asarray(data).ravel().tolist()
    else:
        # 单个数值（如 int/float）包装成列表
        data = [data]

    return data


def convert_to_significance_dict(result):
    """
    Convert significance test results to a comparison dictionary.
    If corrected_p_value is not present, use original p_value.
    """
    significance_dict = {}
    for entry in result:
        # 优先使用 corrected_p_value，如果没有则用 p_value
        p_val = entry.get('corrected_p_value', entry['p_value'])
        if not pd.isna(p_val):
            group1, group2 = entry['group_comparison'].split(' vs ')
            significance_dict[(f"{group1}", f"{group2}")] = p_val
    return significance_dict


class GroupDialog(QDialog):
    """
    Dialog window for creating or editing statistical groups.
    :param parent: Parent widget
    :type parent: QWidget
    :param group_name: Initial group name
    :type group_name: str
    :param folder_path: Initial folder path
    :type folder_path: str
    """
    def __init__(self, parent=None, group_name="", folder_path=""):
        super().__init__(parent)
        self.setWindowTitle("Group Configuration")
        self._init_ui(group_name, folder_path)

    def _init_ui(self, group_name, folder_path):
        """Initialize UI elements."""
        layout = QVBoxLayout()
        # 分组名称输入
        name_label = QLabel("Group Name:")
        self.name_edit = QLineEdit(group_name)
        layout.addWidget(name_label)
        layout.addWidget(self.name_edit)
        # 数据文件夹选择
        folder_label = QLabel("Data Folder:")
        self.folder_edit = QLineEdit(folder_path)
        folder_button = BFPushButton("Browse")
        folder_button.clicked.connect(self._select_folder)
        layout.addWidget(folder_label)
        layout.addWidget(self.folder_edit)
        layout.addWidget(folder_button)
        # 确认按钮
        confirm_button = BFPushButton("Confirm")
        confirm_button.clicked.connect(self.accept)
        layout.addWidget(confirm_button)
        self.setLayout(layout)

    def _select_folder(self):
        """Open folder selection dialog."""
        path = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if path:
            self.folder_edit.setText(path)

    def get_group_data(self):
        """Retrieve input group configuration data."""
        return self.name_edit.text(), self.folder_edit.text()


class ChannelSelectionDialog(QDialog):
    """
    Dialog for selecting data channels.
    :param available_channels: List of available channel names
    :type available_channels: list
    :param selected_channels: Pre-selected channel names
    :type selected_channels: list
    :param parent: Parent widget
    :type parent: QWidget
    """
    def __init__(self, available_channels, selected_channels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Channel Selection")
        self.setMinimumSize(300, 300)
        self._init_ui(available_channels, selected_channels)

    def _init_ui(self, channels, selected):
        """Initialize UI elements."""
        layout = QVBoxLayout()
        # 滚动复选框区域
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        # 创建通道复选框
        self.checkboxes = {}
        for channel in channels:
            cb = QCheckBox(channel)
            cb.setChecked(channel in selected)
            scroll_layout.addWidget(cb)
            self.checkboxes[channel] = cb
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        # 确认按钮
        confirm_button = BFPushButton("Confirm")
        confirm_button.clicked.connect(self.accept)
        layout.addWidget(confirm_button)
        self.setLayout(layout)

    def get_selected_channels(self):
        """Retrieve selected channel names."""
        return [ch for ch, cb in self.checkboxes.items() if cb.isChecked()]


class VisualisationSettingsDialog(QDialog):
    """
    Dialog for customizing plot visualization parameters.
    :param title: Current plot title
    :type title: str
    :param y_label: Current Y-axis label
    :type y_label: str
    :param x_label: Current X-axis label
    :type x_label: str
    :param legend: Current legend items
    :type legend: list
    :param x_ticks: Current X-axis tick labels
    :type x_ticks: list
    :param y_range: Current Y-axis range
    :type y_range: tuple
    :param color: Current primary color
    :type color: str
    :param parent: Parent widget
    :type parent: QWidget
    """
    def __init__(self, title="Default Title", y_label="Y Axis", x_label="X Axis",
                 legend=None, x_ticks=None, y_range=None, color="#ff0000", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Customization")
        self.setGeometry(100, 100, 300, 300)
        self._init_ui(title, y_label, x_label, legend, x_ticks, y_range, color)

    def _init_ui(self, title, y_label, x_label, legend, x_ticks, y_range, color):
        """Initialize UI elements."""
        layout = QFormLayout(self)
        # 带默认值的输入框
        self.title_input = QLineEdit(title)
        self.y_label_input = QLineEdit(y_label)
        self.x_label_input = QLineEdit(x_label)
        self.legend_input = QLineEdit(", ".join(legend) if legend else "")
        self.x_ticks_input = QLineEdit(", ".join(x_ticks) if x_ticks else "")
        self.y_range_input = QLineEdit(f"{y_range[0]}, {y_range[1]}" if y_range else "0, 10")
        self.color_input = QLineEdit(color)
        # 添加输入框到表单
        layout.addRow("Title:", self.title_input)
        layout.addRow("Y Label:", self.y_label_input)
        layout.addRow("X Label:", self.x_label_input)
        layout.addRow("Legend (comma-separated):", self.legend_input)
        layout.addRow("X Ticks (comma-separated):", self.x_ticks_input)
        layout.addRow("Y Range (min, max):", self.y_range_input)
        layout.addRow("Color (hex):", self.color_input)
        # 确认按钮
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.accept)
        layout.addRow(self.confirm_button)

    def get_settings(self):
        """Retrieve visualization settings from dialog inputs."""
        title = self.title_input.text()
        y_label = self.y_label_input.text()
        x_label = self.x_label_input.text()
        legend = self.legend_input.text().split(", ")
        x_ticks = self.x_ticks_input.text().split(", ")
        y_range = tuple(map(float, self.y_range_input.text().split(",")))
        color = self.color_input.text()
        return title, y_label, x_label, legend, x_ticks, y_range, color


class MatplotlibWidget(QWidget):
    """Embedded widget for displaying Matplotlib visualizations."""
    def __init__(self, parent=None):
        """
        Initialize Matplotlib figure container.
        :param parent: Parent widget
        :type parent: QWidget
        """
        super().__init__(parent)
        self._initialize_plot_components()
        self.plot_data()

    def _initialize_plot_components(self):
        """Set up figure, axes, and canvas components."""
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_default(self):
        """Generate default boxplot visualization with random data."""
        data = [np.random.randn(100) for _ in range(5)]
        self.plot(
            data=data,
            title="Sample Boxplot",
            y_label="Values",
            x_label="Groups",
            x_ticks=["A", "B", "C", "D", "E"]
        )

    def plot(self, data, title="", y_label="", x_label="", legend=None,
             x_ticks=None, y_range=None, color="blue"):
        """
        Render customized boxplot visualization.
        :param data: Input dataset for visualization
        :type data: list
        :param title: Plot title text
        :type title: str
        :param y_label: Y-axis label text
        :type y_label: str
        :param x_label: X-axis label text
        :type x_label: str
        :param legend: Legend items list
        :type legend: list
        :param x_ticks: X-axis tick labels
        :type x_ticks: list
        :param y_range: Y-axis range limits
        :type y_range: tuple
        :param color: Primary boxplot color
        :type color: str
        """
        self.ax.clear()
        self.ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor=color))
        self.ax.set_title(title)
        self.ax.set_ylabel(y_label)
        self.ax.set_xlabel(x_label)
        if x_ticks:
            self.ax.set_xticklabels(x_ticks)
        if y_range:
            self.ax.set_ylim(y_range)
        if legend:
            self.ax.legend(legend)
        self.canvas.draw()

    def plot_data(self):
        """Generate NVC boxplot visualization for left vs right hand MI (原始示例逻辑保留)."""
        # EEG和fNIRS通道配置
        eeg_channels = [['FCC5h'], ['FCC3h'], ['FCC4h'], ['FCC6h'],
                        ['CCP5h'], ['CCP3h'], ['CCP4h'], ['CCP6h']]
        fnirs_channels = [['S8_D9', 'S8_D10', 'S7_D10', 'S7_D9'],
                          ['S8_D11', 'S10_D11', 'S10_D10', 'S8_D10'],
                          ['S12_D13', 'S12_D15', 'S11_D15', 'S11_D13'],
                          ['S12_D16', 'S14_D16', 'S14_D15', 'S12_D15'],
                          ['S7_D10', 'S9_D10', 'S9_D5', 'S7_D5'],
                          ['S10_D10', 'S10_D12', 'S9_D12', 'S9_D10'],
                          ['S11_D15', 'S13_D15', 'S13_D14', 'S11_D14'],
                          ['S14_D15', 'S14_D8', 'S13_D8', 'S13_D15']]

        # 加载NVC结果（路径保留原始示例，实际使用可替换）
        results_path = os.path.join('E:\\DATA\\public_datasets\\EEG-fNIRS\\TUBerlinBCI\\Analysis Folder\\NVC\\02',
                                    'nvc_results.json')
        try:
            with open(results_path, 'r') as file:
                nvc_data = json.load(file)
        except FileNotFoundError:
            self.plot_default()
            return

        # 处理受试者数据
        subject_id = 'subject 24'
        subject_values = nvc_data['data'].get(subject_id, [])
        left_values = {eeg[0]: [] for eeg in eeg_channels}
        right_values = {eeg[0]: [] for eeg in eeg_channels}

        # 提取NVC值
        for epoch in subject_values:
            label = nvc_data['Labels'][subject_id][subject_values.index(epoch)]
            for idx, eeg_group in enumerate(eeg_channels):
                current_eeg = eeg_group[0]
                fnirs_group = [ch + ' hbo' for ch in fnirs_channels[idx]]
                nvc_vals = []
                for result in epoch:
                    if (result['EEG_Channel'] == current_eeg and
                            result['fNIRS_Channel'] in fnirs_group):
                        nvc_vals.append(abs(result['NVC_Value']))
                if nvc_vals:
                    avg = np.mean(nvc_vals)
                    if label == 'left':
                        left_values[current_eeg].append(avg)
                    elif label == 'right':
                        right_values[current_eeg].append(avg)

        # 绘制箱线图
        self.ax.clear()
        eeg_labels = list(left_values.keys())
        left_data = [left_values[ch] for ch in eeg_labels]
        right_data = [right_values[ch] for ch in eeg_labels]
        left_positions = np.arange(len(eeg_labels)) * 2.0
        right_positions = left_positions + 0.8

        # 绘制左右手箱线图
        self.ax.boxplot(left_data, positions=left_positions, widths=0.6,
                        patch_artist=True, boxprops=dict(facecolor="skyblue"),
                        labels=eeg_labels)
        self.ax.boxplot(right_data, positions=right_positions, widths=0.6,
                        patch_artist=True, boxprops=dict(facecolor="salmon"))

        # 添加显著性标记
        significance_level = 0.05
        bracket_height = 0.05
        asterisk_offset = 0.01
        line_width = 1.5
        for i, channel in enumerate(eeg_labels):
            t_stat, p_val = ttest_ind(left_values[channel], right_values[channel], nan_policy='omit')
            if p_val < significance_level:
                max_val = max(max(left_values[channel]), max(right_values[channel]))
                y_bracket = max_val + bracket_height
                y_asterisk = y_bracket + asterisk_offset
                self.ax.text(left_positions[i] + 0.4, y_asterisk, '*',
                             ha='center', va='bottom', fontsize=14, color='red')
                self.ax.plot([left_positions[i], right_positions[i]], [y_bracket, y_bracket],
                             color='black', lw=line_width)
                self.ax.plot([left_positions[i], left_positions[i]],
                             [y_bracket, y_bracket - 0.02], color='black', lw=line_width)
                self.ax.plot([right_positions[i], right_positions[i]],
                             [y_bracket, y_bracket - 0.02], color='black', lw=line_width)

        # 图表样式配置
        self.ax.set_xlabel('EEG Channels', fontsize=12)
        self.ax.set_ylabel('Absolute NVC Value', fontsize=12)
        self.ax.set_title('Neurovascular Coupling for Hand Motor Imagery', fontsize=12)
        self.ax.set_xticks(left_positions + 0.4)
        self.ax.set_xticklabels(eeg_labels)
        # 图例
        legend_items = [
            plt.Line2D([0], [0], color="skyblue", lw=4, label='Left Hand'),
            plt.Line2D([0], [0], color="salmon", lw=4, label='Right Hand'),
            plt.Line2D([0], [0], marker='*', color='w', label='p<0.05',
                       markerfacecolor='red', markersize=10)
        ]
        self.ax.legend(handles=legend_items, loc='upper right')
        self.canvas.draw()


class StatisticalAnalysisDialog(QMainWindow):
    """Dialog window for performing statistical analyses and visualizations (核心主窗口)."""
    def __init__(self):
        """Initialize statistical analysis interface components."""
        super().__init__()
        self._initialize_state_variables()
        self._configure_window_properties()
        self._create_main_layout()
        self._connect_signal_handlers()

    def _initialize_state_variables(self):
        """Initialize application state variables (初始化全局状态)."""
        self.result = []
        self.group_select_features = {}
        self.data = None
        self.group_folder = {}
        self.group_files = {}
        self.is_valid = False

    def _configure_window_properties(self):
        """Set window title and dimensions (窗口基础配置)."""
        self.setWindowTitle("Statistical Analysis")
        self.setGeometry(100, 100, 1200, 800)

    def _create_main_layout(self):
        """Construct primary interface layout (主布局：分组管理+分析可视化)."""
        main_layout = QHBoxLayout()
        central_widget = QWidget()
        # 分组管理区域
        self.group_widget = self._create_group_management_section()
        # 分析和可视化区域
        analysis_visual_widget = self._create_analysis_visualization_section()
        # 组合区域并设置拉伸比
        main_layout.addWidget(self.group_widget)
        main_layout.addWidget(analysis_visual_widget)
        main_layout.setStretch(0, 25)
        main_layout.setStretch(1, 75)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _create_group_management_section(self):
        """Create group management interface components (左侧分组管理区域)."""
        widget = QFrame()
        widget.setFrameShape(QFrame.Box)
        widget.setStyleSheet("background-color: white;")
        layout = QVBoxLayout()
        # 标题
        title_label = QLabel("Group Management")
        title_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
        layout.addWidget(title_label)
        # 分组操作按钮
        button_layout = QHBoxLayout()
        self.new_group_button = BFPushButton("Add Group")
        self.import_groups_button = BFPushButton("Import Groups")
        button_layout.addWidget(self.new_group_button)
        button_layout.addWidget(self.import_groups_button)
        layout.addLayout(button_layout)
        # 分组列表
        self.group_list = QListWidget()
        layout.addWidget(self.group_list)
        # 验证按钮
        self.validate_button = BFPushButton("Validate Groups")
        layout.addWidget(self.validate_button)
        widget.setLayout(layout)
        return widget

    def _create_analysis_visualization_section(self):
        """Create analysis and visualization interface components (右侧分析+可视化区域)."""
        widget = QWidget()
        layout = QVBoxLayout()
        # 统计分析区域
        self.analysis_widget = self._create_statistical_analysis_section()
        # 可视化区域
        self.visual_widget = self._create_visualization_section()
        layout.addWidget(self.analysis_widget)
        layout.addWidget(self.visual_widget)
        layout.setStretch(0, 3)
        layout.setStretch(1, 7)
        widget.setLayout(layout)
        return widget

    def _create_statistical_analysis_section(self):
        """Create statistical analysis configuration components (统计分析配置区域)."""
        widget = QFrame()
        widget.setFrameShape(QFrame.Box)
        widget.setStyleSheet("background-color: white;")
        outer_layout = QVBoxLayout()
        layout = QGridLayout()

        # 标题
        title_label = QLabel("Statistical Analysis")
        title_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
        outer_layout.addWidget(title_label)

        # 描述文字
        description = QLabel('Performs pairwise group comparisons using selected method. '
                             'Multiple comparison correction applied when >2 groups exist. '
                             'MANOVA automatically used when features >2.')
        description.setStyleSheet("font-family: 'Times New Roman'; font-size: 10pt; color: blue;")
        description.setWordWrap(True)
        outer_layout.addWidget(description)
        outer_layout.addLayout(layout)
        outer_layout.addStretch(1)

        # 通道选择
        channel_label = QLabel("Channel:")
        self.channel_combo = QComboBox()
        layout.addWidget(channel_label, 0, 0)
        layout.addWidget(self.channel_combo, 0, 1)

        # 特征选择
        feature_label = QLabel("Feature:")
        self.feature_combo = QComboBox()
        layout.addWidget(feature_label, 1, 0)
        layout.addWidget(self.feature_combo, 1, 1)

        # ========== 统计方法选择（关键修改：逐项添加，确保字符串绝对干净）==========
        method_label = QLabel("Method:")
        self.method_combo = QComboBox()
        self.method_combo.clear()  # 先清空，避免设计器残留
        self.method_combo.addItem("t-test")
        self.method_combo.addItem("t-test(paired)")
        self.method_combo.addItem("anova")
        self.method_combo.addItem("mann-whitney U")
        self.method_combo.addItem("wilcoxon(paired)")
        self.method_combo.addItem("kruskal-wallis")
        layout.addWidget(method_label, 2, 0)
        layout.addWidget(self.method_combo, 2, 1)

        # 多重校正配置
        correction_label = QLabel("Correction:")
        self.correction_combo = QComboBox()
        self.correction_combo.addItems([
            "bonferroni", "fdr_bh", "fdr_by", "holm-sidak", "sidak"
        ])
        self.enable_correction = QCheckBox("Enable")
        self.enable_correction.setChecked(True)
        layout.addWidget(correction_label, 3, 0)
        layout.addWidget(self.correction_combo, 3, 1)
        layout.addWidget(self.enable_correction, 3, 2)

        # 操作按钮
        self.run_button = BFPushButton("Run Analysis")
        self.run_button.setFixedWidth(120)
        self.status_label = QLabel("Status: Waiting")
        self.export_button = BFPushButton("Export Results")
        self.export_button.setFixedWidth(120)
        layout.addWidget(self.run_button, 4, 0)
        layout.addWidget(self.status_label, 4, 1)
        layout.addWidget(self.export_button, 4, 2)

        layout.setVerticalSpacing(10)
        widget.setLayout(outer_layout)
        return widget

    def _create_visualization_section(self):
        """Create visualization control components (数据可视化区域)."""
        widget = QFrame()
        widget.setFrameShape(QFrame.Box)
        widget.setStyleSheet("background-color: white;")
        layout = QVBoxLayout()
        self.scroll_area = QScrollArea()
        # 标题
        title_label = QLabel("Data Visualization")
        title_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
        layout.addWidget(title_label)
        # 可视化控制按钮
        control_layout = QHBoxLayout()
        plot_type_label = QLabel('Plot Type: ')
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            'scatter plot', 'bar plot', 'box plot', 'violin plot'
        ])
        self.plot_type_combo.setFixedHeight(25)
        self.plot_button = BFPushButton("Generate Plot")
        self.save_button = BFPushButton("Save Image")
        self.save_button.setFixedWidth(100)
        self.settings_button = BFPushButton("Plot Settings")
        self.settings_button.setFixedWidth(100)
        # 组装控制布局
        control_layout.addWidget(plot_type_label)
        control_layout.addWidget(self.plot_type_combo)
        control_layout.addWidget(self.plot_button)
        control_layout.addStretch(1)
        control_layout.addWidget(self.save_button)
        control_layout.addWidget(self.settings_button)
        # 可视化容器（初始为散点图）
        self.plot_container = ScatterPlotWindow()
        self.scroll_area.setWidget(self.plot_container)
        # 组装整体布局
        layout.addLayout(control_layout)
        layout.addWidget(self.scroll_area)
        widget.setLayout(layout)
        return widget

    def _connect_signal_handlers(self):
        """Connect UI components to action handlers (信号与槽绑定)."""
        self.new_group_button.clicked.connect(self.add_group)
        self.validate_button.clicked.connect(self.validate_groups)
        self.group_list.itemDoubleClicked.connect(self.modify_group)
        self.run_button.clicked.connect(self.run_analysis)
        self.export_button.clicked.connect(self.export_results)
        self.plot_button.clicked.connect(self.generate_visualization)
        # 绘图设置按钮预留槽位，可后续绑定
        # self.settings_button.clicked.connect(self.configure_visualization)

    def add_group(self):
        """Open dialog to create new analysis group (添加新分组)."""
        dialog = GroupDialog(self)
        if dialog.exec_():
            name, folder = dialog.get_group_data()
            self.group_list.addItem(f"{name} - {folder}")
        self.is_valid = False

    def modify_group(self, item):
        """Open dialog to edit existing analysis group (编辑已有分组)."""
        try:
            name, folder = item.text().split(" - ")
            dialog = GroupDialog(self, name, folder)
            if dialog.exec_():
                new_name, new_folder = dialog.get_group_data()
                item.setText(f"{new_name} - {new_folder}")
            self.is_valid = False
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid group format!")

    def validate_groups(self):
        """Validate and load group datasets (验证分组并加载数据，核心数据加载逻辑)."""
        self.group_files.clear()
        self.group_select_features.clear()
        self.group_folder.clear()

        # 无分组时提示
        if self.group_list.count() < 2:
            QMessageBox.warning(self, "Warning", "At least two groups required for analysis!")
            return

        # 遍历加载每个分组的文件
        for i in range(self.group_list.count()):
            try:
                group_data = self.group_list.item(i).text()
                group_name, folder_path = group_data.split(" - ")
                self.group_files[group_name] = []
                # 校验文件夹是否存在
                if not os.path.exists(folder_path):
                    QMessageBox.warning(self, "Error", f"Folder {folder_path} does not exist!")
                    return
                # 加载xlsx文件（使用自定义的 read_xlsx）
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.xlsx'):
                        file_path = os.path.join(folder_path, file_name)
                        print(f"读取: {file_name} (xlsx)")
                        file_data = read_xlsx(file_path)
                        self.group_files[group_name].append(file_data)
                # 无有效文件时提示
                if not self.group_files[group_name]:
                    QMessageBox.warning(self, "Error", f"No XLSX files found in {folder_path}!")
                    return
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Load group data failed: {str(e)}")
                return

        # 初始化UI的通道和特征下拉框
        try:
            self.data = next(iter(self.group_files.values()))[0]
            # 根据 read_xlsx 返回的数据结构填充下拉框
            # feature_dict 包含 'feature' 字典（特征名: 数据列表）和 'ch_names' 列表
            self.feature_combo.clear()
            self.feature_combo.addItems(list(self.data['feature'].keys()))
            # 通道下拉框显隐控制
            if 'ch_names' not in self.data or not self.data['ch_names']:
                self.channel_combo.setVisible(False)
            else:
                self.channel_combo.clear()
                self.channel_combo.addItems(self.data['ch_names'])
            self.status_label.setText("Status: Waiting")
            self.is_valid = True
            QMessageBox.information(self, "Success", "Data validation complete!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Initialize UI failed: {str(e)}")
            self.is_valid = False

    def run_analysis(self):
        """Execute statistical analysis on selected features (执行统计分析，核心业务逻辑)."""
        if not self.is_valid:
            QMessageBox.warning(self, "Action Required", "Validate groups before analysis!")
            return

        self.status_label.setText("Status: Processing...")
        ##QApplication.processEvents()  # 刷新UI，显示处理状态

        try:
            # 提取选中的通道和特征，加载数据
            self.group_select_features.clear()
            for group_name in self.group_files:
                self.group_select_features[group_name] = []
                for dataset in self.group_files[group_name]:
                    channel = self.channel_combo.currentText() if self.channel_combo.isVisible() else None
                    feature = self.feature_combo.currentText()
                    # get_feature 需返回一维列表（之前已修改）
                    self.group_select_features[group_name].extend(get_feature(dataset, channel, feature))

            # ========== 方法名处理：去除所有可能干扰的字符 ==========
            raw_method = self.method_combo.currentText()
            print("=" * 50)
            print(f"[DEBUG] Raw method repr: {repr(raw_method)}")  # 显示原始字符串的精确表示

            # 去除首尾空白
            method = raw_method.strip()
            print(f"[DEBUG] After strip: {repr(method)}")

            # 处理配对标记
            is_paired = False
            if method == 't-test(paired)':
                method = 't-test'
                is_paired = True
            elif method == 'wilcoxon(paired)':
                method = 'wilcoxon'
                is_paired = True

            # 再次去除首尾空白（防止转换后引入）
            method = method.strip()
            print(f"[DEBUG] Final method: {repr(method)}")
            print("=" * 50)

            # ========== 调用统计检验函数 ==========
            self.result = calculate_significance(
                self.group_select_features,
                method=method,
                paired=is_paired
            )

            # 应用多重校正（仅当有多个比较且启用了校正）
            if self.enable_correction.isChecked() and len(self.result) > 1:
                correct_method = self.correction_combo.currentText().strip()
                self.result = multiple_comparison_correction(self.result, correct_method)

            self.status_label.setText("Status: Completed!")
            QMessageBox.information(self, "Success", "Statistical analysis completed!")

        except Exception as e:
            self.status_label.setText("Status: Failed!")
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")

    def export_results(self):
        """Export analysis results to external format (导出结果，预留扩展接口)."""
        if not self.result:
            QMessageBox.warning(self, "Warning", "No analysis results to export!")
            return
        # 弹出保存对话框，默认保存为xlsx格式
        save_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "Excel Files (*.xlsx);;JSON Files (*.json)")
        if not save_path:
            return
        try:
            # 转换结果为DataFrame，方便导出
            result_df = pd.DataFrame(self.result)
            if save_path.endswith('.xlsx'):
                result_df.to_excel(save_path, index=False)
            else:
                result_df.to_json(save_path, orient='records', indent=4)
            QMessageBox.information(self, "Success", f"Results exported to {save_path} successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def generate_visualization(self):
        """Generate selected visualization type (生成可视化图表，联动statistical_plot.py)."""
        if not self.group_select_features:
            QMessageBox.warning(self, "Data Required", "Run analysis before generating plot!")
            return

        try:
            plot_type = self.plot_type_combo.currentText()
            # 根据选择的图表类型，创建对应窗口并绘图
            if plot_type == 'scatter plot':
                self.plot_container = ScatterPlotWindow()
                self.scroll_area.setWidget(self.plot_container)
                self.plot_container.plot_scatter(self.group_select_features)
            elif plot_type == 'bar plot':
                self.plot_container = DensityHistogramWindow()
                self.scroll_area.setWidget(self.plot_container)
                self.plot_container.plot_density_histogram(self.group_select_features)
            elif plot_type == 'box plot':
                self.plot_container = SignificanceBoxPlotWindow()
                self.scroll_area.setWidget(self.plot_container)
                significance_dict = convert_to_significance_dict(self.result)
                self.plot_container.plot_boxplot_with_significance(self.group_select_features, significance_dict)
            elif plot_type == 'violin plot':
                self.plot_container = SignificanceViolinPlotWindow()
                self.scroll_area.setWidget(self.plot_container)
                significance_dict = convert_to_significance_dict(self.result)
                self.plot_container.plot_violin_with_significance(self.group_select_features, significance_dict)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Generate plot failed: {str(e)}")


if __name__ == "__main__":
    # 主函数，单独运行测试
    app = QApplication(sys.argv)
    window = StatisticalAnalysisDialog()
    window.show()
    sys.exit(app.exec_())