#
# import json
# import os
# import sys
# from pathlib import Path
#
# # 添加项目根目录到 sys.path
# start_path = Path(__file__).resolve().parent
# for parent in [start_path] + list(start_path.parents):
#     if parent.name == 'core':
#         project_root = parent.parent
#         if str(project_root) not in sys.path:
#             sys.path.insert(0, str(project_root))
#             print(f"子进程: 已将项目根目录 {project_root} 添加到 sys.path")
#         break
# else:
#     raise RuntimeError("子进程: 未找到名为 'core' 的目录")
#
# import pandas as pd
# import numpy as np
# from scipy.stats import ttest_ind
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
#
# # PyQt5 组件导入
# from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
#                              QGridLayout, QPushButton, QListWidget, QFileDialog, QDialog,
#                              QLineEdit, QLabel, QMessageBox, QCheckBox, QComboBox,
#                              QScrollArea, QFrame, QFormLayout)
# from PyQt5.QtCore import Qt
#
# # 项目内模块导入
# from core.processing.Statistical_Analysis.significance_test import calculate_significance, multiple_comparison_correction
# from core.processing.Statistical_Analysis.statistical_plot import (ScatterPlotWindow, DensityHistogramWindow,
#                                     SignificanceBoxPlotWindow, SignificanceViolinPlotWindow)
#
# BFPushButton = QPushButton
#
#
# # [MODIFIED] 读取 Excel 文件，正确解析通道和特征
# def read_xlsx(file_path):
#     """
#     读取 xlsx 文件，返回符合要求的字典格式：
#     {
#         'feature': {
#             'mean': [[val1], [val2], ...],   # 每个元素是包含一个样本值的列表
#             'std': [[val1], [val2], ...],
#             ...
#         },
#         'ch_names': ['通道1', '通道2', ...]
#     }
#     假设第一个 sheet 的第一列是通道名称，其余列为特征。
#     """
#     try:
#         # 读取第一个 sheet
#         df = pd.read_excel(file_path, engine='openpyxl', sheet_name=0)
#         # 第一列作为通道名称
#         channel_col = df.columns[0]
#         ch_names = df[channel_col].astype(str).tolist()
#         # 处理每一特征列
#         feature_dict = {}
#         for col in df.columns[1:]:
#             # 将列转换为数值，无法转换的变为 NaN
#             numeric_vals = pd.to_numeric(df[col], errors='coerce')
#             # 转换为列表，每个元素可能为 NaN
#             vals = numeric_vals.tolist()
#             # 将每个值包装成列表，形成二维结构：[[val1], [val2], ...]
#             # 这样后续 get_feature 通过 channel_index 取出的就是 [val_i]（列表形式）
#             feature_dict[col] = [[v] for v in vals]
#         if not feature_dict:
#             raise ValueError(f"文件 {file_path} 中未找到任何数值列。")
#         return {
#             'feature': feature_dict,
#             'ch_names': ch_names
#         }
#     except Exception as e:
#         raise ValueError(f"无法读取文件 {file_path}: {str(e)}")
#
#
# def get_feature(feature_dict, channel_name=None, feature_name=None):
#     """
#     Extract specific feature data from a feature dictionary.
#     Ensures returned data is a 1D list of float values.
#     If data has channel information, channel_name must be specified.
#     """
#     if feature_name not in feature_dict["feature"]:
#         raise ValueError(f"Feature '{feature_name}' not found")
#
#     has_channels = "ch_names" in feature_dict and feature_dict["ch_names"]
#     if has_channels and channel_name is None:
#         raise ValueError("Data has channel information. Please select a specific channel.")
#     if not has_channels and channel_name is not None:
#         # 如果数据无通道，但传入了通道名（可能是界面占位项），自动忽略
#         channel_name = None
#
#     feature_data = feature_dict["feature"][feature_name]
#
#     if channel_name is not None and has_channels:
#         if channel_name not in feature_dict["ch_names"]:
#             raise ValueError(f"Channel '{channel_name}' not found")
#         channel_index = feature_dict["ch_names"].index(channel_name)
#         data = feature_data[channel_index]
#     else:
#         data = feature_data
#
#     # 统一转换为 1D 列表
#     if isinstance(data, (np.ndarray, list)):
#         data = np.asarray(data).ravel().tolist()
#     else:
#         data = [data]
#
#     # 将所有元素转换为 float，无法转换的替换为 NaN
#     numeric_data = []
#     for val in data:
#         try:
#             numeric_data.append(float(val))
#         except (ValueError, TypeError):
#             numeric_data.append(np.nan)
#     return numeric_data
#
#
# def convert_to_significance_dict(result):
#     significance_dict = {}
#     for entry in result:
#         p_val = entry.get('corrected_p_value', entry['p_value'])
#         if not pd.isna(p_val):
#             group1, group2 = entry['group_comparison'].split(' vs ')
#             significance_dict[(f"{group1}", f"{group2}")] = p_val
#     return significance_dict
#
#
# class GroupDialog(QDialog):
#     def __init__(self, parent=None, group_name="", folder_path=""):
#         super().__init__(parent)
#         self.setWindowTitle("Group Configuration")
#         self._init_ui(group_name, folder_path)
#
#     def _init_ui(self, group_name, folder_path):
#         layout = QVBoxLayout()
#         name_label = QLabel("Group Name:")
#         self.name_edit = QLineEdit(group_name)
#         layout.addWidget(name_label)
#         layout.addWidget(self.name_edit)
#         folder_label = QLabel("Data Folder:")
#         self.folder_edit = QLineEdit(folder_path)
#         folder_button = BFPushButton("Browse")
#         folder_button.clicked.connect(self._select_folder)
#         layout.addWidget(folder_label)
#         layout.addWidget(self.folder_edit)
#         layout.addWidget(folder_button)
#         confirm_button = BFPushButton("Confirm")
#         confirm_button.clicked.connect(self.accept)
#         layout.addWidget(confirm_button)
#         self.setLayout(layout)
#
#     def _select_folder(self):
#         path = QFileDialog.getExistingDirectory(self, "Select Data Folder")
#         if path:
#             self.folder_edit.setText(path)
#
#     def get_group_data(self):
#         return self.name_edit.text(), self.folder_edit.text()
#
#
# class ChannelSelectionDialog(QDialog):
#     def __init__(self, available_channels, selected_channels, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Channel Selection")
#         self.setMinimumSize(300, 300)
#         self._init_ui(available_channels, selected_channels)
#
#     def _init_ui(self, channels, selected):
#         layout = QVBoxLayout()
#         scroll = QScrollArea()
#         scroll_widget = QWidget()
#         scroll_layout = QVBoxLayout(scroll_widget)
#         self.checkboxes = {}
#         for channel in channels:
#             cb = QCheckBox(channel)
#             cb.setChecked(channel in selected)
#             scroll_layout.addWidget(cb)
#             self.checkboxes[channel] = cb
#         scroll.setWidget(scroll_widget)
#         scroll.setWidgetResizable(True)
#         layout.addWidget(scroll)
#         confirm_button = BFPushButton("Confirm")
#         confirm_button.clicked.connect(self.accept)
#         layout.addWidget(confirm_button)
#         self.setLayout(layout)
#
#     def get_selected_channels(self):
#         return [ch for ch, cb in self.checkboxes.items() if cb.isChecked()]
#
#
# class VisualisationSettingsDialog(QDialog):
#     def __init__(self, title="Default Title", y_label="Y Axis", x_label="X Axis",
#                  legend=None, x_ticks=None, y_range=None, color="#ff0000", parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Plot Customization")
#         self.setGeometry(100, 100, 300, 300)
#         self._init_ui(title, y_label, x_label, legend, x_ticks, y_range, color)
#
#     def _init_ui(self, title, y_label, x_label, legend, x_ticks, y_range, color):
#         layout = QFormLayout(self)
#         self.title_input = QLineEdit(title)
#         self.y_label_input = QLineEdit(y_label)
#         self.x_label_input = QLineEdit(x_label)
#         self.legend_input = QLineEdit(", ".join(legend) if legend else "")
#         self.x_ticks_input = QLineEdit(", ".join(x_ticks) if x_ticks else "")
#         self.y_range_input = QLineEdit(f"{y_range[0]}, {y_range[1]}" if y_range else "0, 10")
#         self.color_input = QLineEdit(color)
#         layout.addRow("Title:", self.title_input)
#         layout.addRow("Y Label:", self.y_label_input)
#         layout.addRow("X Label:", self.x_label_input)
#         layout.addRow("Legend (comma-separated):", self.legend_input)
#         layout.addRow("X Ticks (comma-separated):", self.x_ticks_input)
#         layout.addRow("Y Range (min, max):", self.y_range_input)
#         layout.addRow("Color (hex):", self.color_input)
#         self.confirm_button = QPushButton("Confirm")
#         self.confirm_button.clicked.connect(self.accept)
#         layout.addRow(self.confirm_button)
#
#     def get_settings(self):
#         title = self.title_input.text()
#         y_label = self.y_label_input.text()
#         x_label = self.x_label_input.text()
#         legend = self.legend_input.text().split(", ")
#         x_ticks = self.x_ticks_input.text().split(", ")
#         y_range = tuple(map(float, self.y_range_input.text().split(",")))
#         color = self.color_input.text()
#         return title, y_label, x_label, legend, x_ticks, y_range, color
#
#
# class MatplotlibWidget(QWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self._initialize_plot_components()
#         self.plot_data()
#
#     def _initialize_plot_components(self):
#         self.figure, self.ax = plt.subplots()
#         self.canvas = FigureCanvas(self.figure)
#         layout = QVBoxLayout()
#         layout.addWidget(self.canvas)
#         self.setLayout(layout)
#
#     def plot_default(self):
#         data = [np.random.randn(100) for _ in range(5)]
#         self.plot(data=data, title="Sample Boxplot", y_label="Values", x_label="Groups",
#                   x_ticks=["A", "B", "C", "D", "E"])
#
#     def plot(self, data, title="", y_label="", x_label="", legend=None,
#              x_ticks=None, y_range=None, color="blue"):
#         self.ax.clear()
#         self.ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor=color))
#         self.ax.set_title(title)
#         self.ax.set_ylabel(y_label)
#         self.ax.set_xlabel(x_label)
#         if x_ticks:
#             self.ax.set_xticklabels(x_ticks)
#         if y_range:
#             self.ax.set_ylim(y_range)
#         if legend:
#             self.ax.legend(legend)
#         self.canvas.draw()
#
#     def plot_data(self):
#         # 原始示例绘图代码，保持不变
#         eeg_channels = [['FCC5h'], ['FCC3h'], ['FCC4h'], ['FCC6h'],
#                         ['CCP5h'], ['CCP3h'], ['CCP4h'], ['CCP6h']]
#         fnirs_channels = [['S8_D9', 'S8_D10', 'S7_D10', 'S7_D9'],
#                           ['S8_D11', 'S10_D11', 'S10_D10', 'S8_D10'],
#                           ['S12_D13', 'S12_D15', 'S11_D15', 'S11_D13'],
#                           ['S12_D16', 'S14_D16', 'S14_D15', 'S12_D15'],
#                           ['S7_D10', 'S9_D10', 'S9_D5', 'S7_D5'],
#                           ['S10_D10', 'S10_D12', 'S9_D12', 'S9_D10'],
#                           ['S11_D15', 'S13_D15', 'S13_D14', 'S11_D14'],
#                           ['S14_D15', 'S14_D8', 'S13_D8', 'S13_D15']]
#
#         results_path = os.path.join('E:\\DATA\\public_datasets\\EEG-fNIRS\\TUBerlinBCI\\Analysis Folder\\NVC\\02',
#                                     'nvc_results.json')
#         try:
#             with open(results_path, 'r') as file:
#                 nvc_data = json.load(file)
#         except FileNotFoundError:
#             self.plot_default()
#             return
#
#         subject_id = 'subject 24'
#         subject_values = nvc_data['data'].get(subject_id, [])
#         left_values = {eeg[0]: [] for eeg in eeg_channels}
#         right_values = {eeg[0]: [] for eeg in eeg_channels}
#
#         for epoch in subject_values:
#             label = nvc_data['Labels'][subject_id][subject_values.index(epoch)]
#             for idx, eeg_group in enumerate(eeg_channels):
#                 current_eeg = eeg_group[0]
#                 fnirs_group = [ch + ' hbo' for ch in fnirs_channels[idx]]
#                 nvc_vals = []
#                 for result in epoch:
#                     if (result['EEG_Channel'] == current_eeg and
#                             result['fNIRS_Channel'] in fnirs_group):
#                         nvc_vals.append(abs(result['NVC_Value']))
#                 if nvc_vals:
#                     avg = np.mean(nvc_vals)
#                     if label == 'left':
#                         left_values[current_eeg].append(avg)
#                     elif label == 'right':
#                         right_values[current_eeg].append(avg)
#
#         self.ax.clear()
#         eeg_labels = list(left_values.keys())
#         left_data = [left_values[ch] for ch in eeg_labels]
#         right_data = [right_values[ch] for ch in eeg_labels]
#         left_positions = np.arange(len(eeg_labels)) * 2.0
#         right_positions = left_positions + 0.8
#
#         self.ax.boxplot(left_data, positions=left_positions, widths=0.6,
#                         patch_artist=True, boxprops=dict(facecolor="skyblue"),
#                         labels=eeg_labels)
#         self.ax.boxplot(right_data, positions=right_positions, widths=0.6,
#                         patch_artist=True, boxprops=dict(facecolor="salmon"))
#
#         significance_level = 0.05
#         bracket_height = 0.05
#         asterisk_offset = 0.01
#         line_width = 1.5
#         for i, channel in enumerate(eeg_labels):
#             t_stat, p_val = ttest_ind(left_values[channel], right_values[channel], nan_policy='omit')
#             if p_val < significance_level:
#                 max_val = max(max(left_values[channel]), max(right_values[channel]))
#                 y_bracket = max_val + bracket_height
#                 y_asterisk = y_bracket + asterisk_offset
#                 self.ax.text(left_positions[i] + 0.4, y_asterisk, '*',
#                              ha='center', va='bottom', fontsize=14, color='red')
#                 self.ax.plot([left_positions[i], right_positions[i]], [y_bracket, y_bracket],
#                              color='black', lw=line_width)
#                 self.ax.plot([left_positions[i], left_positions[i]],
#                              [y_bracket, y_bracket - 0.02], color='black', lw=line_width)
#                 self.ax.plot([right_positions[i], right_positions[i]],
#                              [y_bracket, y_bracket - 0.02], color='black', lw=line_width)
#
#         self.ax.set_xlabel('EEG Channels', fontsize=12)
#         self.ax.set_ylabel('Absolute NVC Value', fontsize=12)
#         self.ax.set_title('Neurovascular Coupling for Hand Motor Imagery', fontsize=12)
#         self.ax.set_xticks(left_positions + 0.4)
#         self.ax.set_xticklabels(eeg_labels)
#         legend_items = [
#             plt.Line2D([0], [0], color="skyblue", lw=4, label='Left Hand'),
#             plt.Line2D([0], [0], color="salmon", lw=4, label='Right Hand'),
#             plt.Line2D([0], [0], marker='*', color='w', label='p<0.05',
#                        markerfacecolor='red', markersize=10)
#         ]
#         self.ax.legend(handles=legend_items, loc='upper right')
#         self.canvas.draw()
#
#
# class StatisticalAnalysisDialog(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self._initialize_state_variables()
#         self._configure_window_properties()
#         self._create_main_layout()
#         self._connect_signal_handlers()
#
#     def _initialize_state_variables(self):
#         self.result = []
#         self.group_select_features = {}
#         self.data = None
#         self.group_folder = {}
#         self.group_files = {}
#         self.is_valid = False
#
#     def _configure_window_properties(self):
#         self.setWindowTitle("Statistical Analysis")
#         self.setGeometry(100, 100, 1200, 800)
#
#     def _create_main_layout(self):
#         main_layout = QHBoxLayout()
#         central_widget = QWidget()
#         self.group_widget = self._create_group_management_section()
#         analysis_visual_widget = self._create_analysis_visualization_section()
#         main_layout.addWidget(self.group_widget)
#         main_layout.addWidget(analysis_visual_widget)
#         main_layout.setStretch(0, 25)
#         main_layout.setStretch(1, 75)
#         central_widget.setLayout(main_layout)
#         self.setCentralWidget(central_widget)
#
#     def _create_group_management_section(self):
#         widget = QFrame()
#         widget.setFrameShape(QFrame.Box)
#         widget.setStyleSheet("background-color: white;")
#         layout = QVBoxLayout()
#         title_label = QLabel("Group Management")
#         title_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
#         layout.addWidget(title_label)
#         button_layout = QHBoxLayout()
#         self.new_group_button = BFPushButton("Add Group")
#         self.import_groups_button = BFPushButton("Import Groups")
#         button_layout.addWidget(self.new_group_button)
#         button_layout.addWidget(self.import_groups_button)
#         layout.addLayout(button_layout)
#         self.group_list = QListWidget()
#         layout.addWidget(self.group_list)
#         self.validate_button = BFPushButton("Validate Groups")
#         layout.addWidget(self.validate_button)
#         widget.setLayout(layout)
#         return widget
#
#     def _create_analysis_visualization_section(self):
#         widget = QWidget()
#         layout = QVBoxLayout()
#         self.analysis_widget = self._create_statistical_analysis_section()
#         self.visual_widget = self._create_visualization_section()
#         layout.addWidget(self.analysis_widget)
#         layout.addWidget(self.visual_widget)
#         layout.setStretch(0, 3)
#         layout.setStretch(1, 7)
#         widget.setLayout(layout)
#         return widget
#
#     def _create_statistical_analysis_section(self):
#         widget = QFrame()
#         widget.setFrameShape(QFrame.Box)
#         widget.setStyleSheet("background-color: white;")
#         outer_layout = QVBoxLayout()
#         layout = QGridLayout()
#
#         title_label = QLabel("Statistical Analysis")
#         title_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
#         outer_layout.addWidget(title_label)
#
#         description = QLabel('Performs pairwise group comparisons using selected method. '
#                              'Multiple comparison correction applied when >2 groups exist. '
#                              'MANOVA automatically used when features >2.')
#         description.setStyleSheet("font-family: 'Times New Roman'; font-size: 10pt; color: blue;")
#         description.setWordWrap(True)
#         outer_layout.addWidget(description)
#         outer_layout.addLayout(layout)
#         outer_layout.addStretch(1)
#
#         channel_label = QLabel("Channel:")
#         self.channel_combo = QComboBox()
#         layout.addWidget(channel_label, 0, 0)
#         layout.addWidget(self.channel_combo, 0, 1)
#
#         feature_label = QLabel("Feature:")
#         self.feature_combo = QComboBox()
#         layout.addWidget(feature_label, 1, 0)
#         layout.addWidget(self.feature_combo, 1, 1)
#
#         method_label = QLabel("Method:")
#         self.method_combo = QComboBox()
#         self.method_combo.clear()
#         self.method_combo.addItem("t-test")
#         self.method_combo.addItem("t-test(paired)")
#         self.method_combo.addItem("anova")
#         self.method_combo.addItem("mann-whitney U")
#         self.method_combo.addItem("wilcoxon(paired)")
#         self.method_combo.addItem("kruskal-wallis")
#         layout.addWidget(method_label, 2, 0)
#         layout.addWidget(self.method_combo, 2, 1)
#
#         correction_label = QLabel("Correction:")
#         self.correction_combo = QComboBox()
#         self.correction_combo.addItems([
#             "bonferroni", "fdr_bh", "fdr_by", "holm-sidak", "sidak"
#         ])
#         self.enable_correction = QCheckBox("Enable")
#         self.enable_correction.setChecked(True)
#         layout.addWidget(correction_label, 3, 0)
#         layout.addWidget(self.correction_combo, 3, 1)
#         layout.addWidget(self.enable_correction, 3, 2)
#
#         self.run_button = BFPushButton("Run Analysis")
#         self.run_button.setFixedWidth(120)
#         self.status_label = QLabel("Status: Waiting")
#         self.export_button = BFPushButton("Export Results")
#         self.export_button.setFixedWidth(120)
#         layout.addWidget(self.run_button, 4, 0)
#         layout.addWidget(self.status_label, 4, 1)
#         layout.addWidget(self.export_button, 4, 2)
#
#         layout.setVerticalSpacing(10)
#         widget.setLayout(outer_layout)
#         return widget
#
#     def _create_visualization_section(self):
#         widget = QFrame()
#         widget.setFrameShape(QFrame.Box)
#         widget.setStyleSheet("background-color: white;")
#         layout = QVBoxLayout()
#         self.scroll_area = QScrollArea()
#         title_label = QLabel("Data Visualization")
#         title_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
#         layout.addWidget(title_label)
#         control_layout = QHBoxLayout()
#         plot_type_label = QLabel('Plot Type: ')
#         self.plot_type_combo = QComboBox()
#         self.plot_type_combo.addItems([
#             'scatter plot', 'density histogram', 'box plot', 'violin plot'
#         ])
#         self.plot_type_combo.setFixedHeight(25)
#         self.plot_button = BFPushButton("Generate Plot")
#         self.save_button = BFPushButton("Save Image")
#         self.save_button.setFixedWidth(100)
#         self.settings_button = BFPushButton("Plot Settings")
#         self.settings_button.setFixedWidth(100)
#         control_layout.addWidget(plot_type_label)
#         control_layout.addWidget(self.plot_type_combo)
#         control_layout.addWidget(self.plot_button)
#         control_layout.addStretch(1)
#         control_layout.addWidget(self.save_button)
#         control_layout.addWidget(self.settings_button)
#         self.plot_container = ScatterPlotWindow()
#         self.scroll_area.setWidget(self.plot_container)
#         layout.addLayout(control_layout)
#         layout.addWidget(self.scroll_area)
#         widget.setLayout(layout)
#         return widget
#
#     def _connect_signal_handlers(self):
#         self.new_group_button.clicked.connect(self.add_group)
#         self.validate_button.clicked.connect(self.validate_groups)
#         self.group_list.itemDoubleClicked.connect(self.modify_group)
#         self.run_button.clicked.connect(self.run_analysis)
#         self.export_button.clicked.connect(self.export_results)
#         self.plot_button.clicked.connect(self.generate_visualization)
#
#     def add_group(self):
#         dialog = GroupDialog(self)
#         if dialog.exec_():
#             name, folder = dialog.get_group_data()
#             self.group_list.addItem(f"{name} - {folder}")
#         self.is_valid = False
#
#     def modify_group(self, item):
#         try:
#             name, folder = item.text().split(" - ")
#             dialog = GroupDialog(self, name, folder)
#             if dialog.exec_():
#                 new_name, new_folder = dialog.get_group_data()
#                 item.setText(f"{new_name} - {new_folder}")
#             self.is_valid = False
#         except ValueError:
#             QMessageBox.warning(self, "Error", "Invalid group format!")
#
#     def validate_groups(self):
#         self.group_files.clear()
#         self.group_select_features.clear()
#         self.group_folder.clear()
#
#         if self.group_list.count() < 2:
#             QMessageBox.warning(self, "Warning", "At least two groups required for analysis!")
#             return
#
#         for i in range(self.group_list.count()):
#             try:
#                 group_data = self.group_list.item(i).text()
#                 group_name, folder_path = group_data.split(" - ")
#                 self.group_files[group_name] = []
#                 if not os.path.exists(folder_path):
#                     QMessageBox.warning(self, "Error", f"Folder {folder_path} does not exist!")
#                     return
#                 for file_name in os.listdir(folder_path):
#                     if file_name.endswith('.xlsx'):
#                         file_path = os.path.join(folder_path, file_name)
#                         print(f"读取: {file_name} (xlsx)")
#                         file_data = read_xlsx(file_path)
#                         self.group_files[group_name].append(file_data)
#                 if not self.group_files[group_name]:
#                     QMessageBox.warning(self, "Error", f"No XLSX files found in {folder_path}!")
#                     return
#             except Exception as e:
#                 QMessageBox.warning(self, "Error", f"Load group data failed: {str(e)}")
#                 return
#
#         try:
#             self.data = next(iter(self.group_files.values()))[0]
#             self.feature_combo.clear()
#             self.feature_combo.addItems(list(self.data['feature'].keys()))
#
#             # 通道下拉框：有通道则填充，否则显示占位并禁用
#             if 'ch_names' in self.data and self.data['ch_names']:
#                 self.channel_combo.clear()
#                 self.channel_combo.addItems(self.data['ch_names'])
#                 self.channel_combo.setEnabled(True)
#             else:
#                 self.channel_combo.clear()
#                 self.channel_combo.addItem("<无通道>")
#                 self.channel_combo.setEnabled(False)
#
#             self.status_label.setText("Status: Waiting")
#             self.is_valid = True
#             QMessageBox.information(self, "Success", "Data validation complete!")
#         except Exception as e:
#             QMessageBox.warning(self, "Error", f"Initialize UI failed: {str(e)}")
#             self.is_valid = False
#
#     def run_analysis(self):
#         if not self.is_valid:
#             QMessageBox.warning(self, "Action Required", "Validate groups before analysis!")
#             return
#
#         self.status_label.setText("Status: Processing...")
#
#         try:
#             self.group_select_features.clear()
#             for group_name in self.group_files:
#                 self.group_select_features[group_name] = []
#                 for dataset in self.group_files[group_name]:
#                     if self.channel_combo.isEnabled():
#                         channel = self.channel_combo.currentText()
#                     else:
#                         channel = None
#                     feature = self.feature_combo.currentText()
#                     self.group_select_features[group_name].extend(get_feature(dataset, channel, feature))
#
#             # 过滤无效数值
#             for group_name in list(self.group_select_features.keys()):
#                 cleaned = []
#                 for val in self.group_select_features[group_name]:
#                     try:
#                         cleaned.append(float(val))
#                     except (ValueError, TypeError):
#                         cleaned.append(np.nan)
#                 data = np.array(cleaned)
#                 data = data[np.isfinite(data)]
#                 if len(data) == 0:
#                     raise ValueError(f"Group '{group_name}' has no valid finite data after filtering.")
#                 self.group_select_features[group_name] = data.tolist()
#
#             raw_method = self.method_combo.currentText()
#             method = raw_method.strip()
#             is_paired = False
#             if method == 't-test(paired)':
#                 method = 't-test'
#                 is_paired = True
#             elif method == 'wilcoxon(paired)':
#                 method = 'wilcoxon'
#                 is_paired = True
#             method = method.strip()
#
#             self.result = calculate_significance(
#                 self.group_select_features,
#                 method=method,
#                 paired=is_paired
#             )
#
#             if self.enable_correction.isChecked() and len(self.result) > 1:
#                 correct_method = self.correction_combo.currentText().strip()
#                 self.result = multiple_comparison_correction(self.result, correct_method)
#
#             self.status_label.setText("Status: Completed!")
#             QMessageBox.information(self, "Success", "Statistical analysis completed!")
#
#         except Exception as e:
#             self.status_label.setText("Status: Failed!")
#             QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
#
#     def export_results(self):
#         if not self.result:
#             QMessageBox.warning(self, "Warning", "No analysis results to export!")
#             return
#         save_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "Excel Files (*.xlsx);;JSON Files (*.json)")
#         if not save_path:
#             return
#         try:
#             result_df = pd.DataFrame(self.result)
#             if save_path.endswith('.xlsx'):
#                 result_df.to_excel(save_path, index=False)
#             else:
#                 result_df.to_json(save_path, orient='records', indent=4)
#             QMessageBox.information(self, "Success", f"Results exported to {save_path} successfully!")
#         except Exception as e:
#             QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
#
#     def generate_visualization(self):
#         if not self.group_select_features:
#             QMessageBox.warning(self, "Data Required", "Run analysis before generating plot!")
#             return
#
#         try:
#             plot_type = self.plot_type_combo.currentText()
#
#             old = self.scroll_area.takeWidget()
#             if old:
#                 old.deleteLater()
#
#             if plot_type == 'scatter plot':
#                 self.plot_container = ScatterPlotWindow()
#                 self.plot_container.plot_scatter(self.group_select_features)
#             elif plot_type == 'density histogram':
#                 self.plot_container = DensityHistogramWindow()
#                 self.plot_container.plot_density_histogram(self.group_select_features)
#             elif plot_type == 'box plot':
#                 self.plot_container = SignificanceBoxPlotWindow()
#                 significance_dict = convert_to_significance_dict(self.result)
#                 self.plot_container.plot_boxplot_with_significance(self.group_select_features, significance_dict)
#             elif plot_type == 'violin plot':
#                 self.plot_container = SignificanceViolinPlotWindow()
#                 significance_dict = convert_to_significance_dict(self.result)
#                 self.plot_container.plot_violin_with_significance(self.group_select_features, significance_dict)
#             else:
#                 QMessageBox.warning(self, "Error", f"Unknown plot type: {plot_type}")
#                 return
#
#             self.scroll_area.setWidget(self.plot_container)
#
#         except Exception as e:
#             QMessageBox.critical(self, "Error", f"Generate plot failed: {str(e)}")
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = StatisticalAnalysisDialog()
#     window.show()
#     sys.exit(app.exec_())









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
                             QScrollArea, QFrame, QFormLayout, QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView)
from PyQt5.QtCore import Qt

# 项目内模块导入
from core.processing.Statistical_Analysis.significance_test import calculate_significance, multiple_comparison_correction
from core.processing.Statistical_Analysis.statistical_plot import (ScatterPlotWindow, DensityHistogramWindow,
                                    SignificanceBoxPlotWindow, SignificanceViolinPlotWindow)

BFPushButton = QPushButton


# ========== 新增：结果展示对话框 ==========
class ResultDialog(QDialog):
    """用于展示统计分析结果的对话框，包含表格和导出功能。"""
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Statistical Analysis Results")
        self.resize(900, 500)
        self.results = results
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 标题
        title = QLabel("Pairwise Group Comparison Results")
        title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        layout.addWidget(title)

        # 表格
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 只读
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)

        # 设置列头
        columns = ["Group Comparison", "Method", "Statistic", "p-value", "Corrected p-value", "Significant"]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)

        # 填充数据
        self.table.setRowCount(len(self.results))
        for i, res in enumerate(self.results):
            self.table.setItem(i, 0, QTableWidgetItem(res.get("group_comparison", "")))
            self.table.setItem(i, 1, QTableWidgetItem(res.get("method", "")))
            self.table.setItem(i, 2, QTableWidgetItem(f"{res.get('stat', 0):.4f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{res.get('p_value', 1):.4f}"))
            corr_p = res.get("corrected_p_value", None)
            if corr_p is not None:
                self.table.setItem(i, 4, QTableWidgetItem(f"{corr_p:.4f}"))
            else:
                self.table.setItem(i, 4, QTableWidgetItem("N/A"))
            sig = res.get("significant_after_correction", False)
            self.table.setItem(i, 5, QTableWidgetItem("Yes" if sig else "No"))

        # 调整列宽
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        layout.addWidget(self.table)

        # 底部按钮
        btn_layout = QHBoxLayout()
        export_btn = QPushButton("Export to Excel")
        export_btn.clicked.connect(self.export_to_excel)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(export_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def export_to_excel(self):
        """将结果导出为 Excel 文件"""
        save_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "Excel Files (*.xlsx)")
        if not save_path:
            return
        try:
            df = pd.DataFrame(self.results)
            df.to_excel(save_path, index=False)
            QMessageBox.information(self, "Success", f"Results exported to {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")


# ========== Excel 读取函数（支持通道） ==========
def read_xlsx(file_path):
    """
    读取 xlsx 文件，返回符合要求的字典格式：
    {
        'feature': { 'mean': [[val1], [val2], ...], 'std': [[val1], [val2], ...], ... },
        'ch_names': ['通道1', '通道2', ...]
    }
    假设第一个 sheet 的第一列是通道名称，其余列为特征。
    """
    try:
        df = pd.read_excel(file_path, engine='openpyxl', sheet_name=0)
        channel_col = df.columns[0]
        ch_names = df[channel_col].astype(str).tolist()
        feature_dict = {}
        for col in df.columns[1:]:
            numeric_vals = pd.to_numeric(df[col], errors='coerce')
            vals = numeric_vals.tolist()
            # 每个值包装成列表，便于后续索引
            feature_dict[col] = [[v] for v in vals]
        if not feature_dict:
            raise ValueError(f"文件 {file_path} 中未找到任何数值列。")
        return {
            'feature': feature_dict,
            'ch_names': ch_names
        }
    except Exception as e:
        raise ValueError(f"无法读取文件 {file_path}: {str(e)}")


def get_feature(feature_dict, channel_name=None, feature_name=None):
    """
    从特征字典中提取指定通道和特征的数据，返回一维浮点数列表。
    """
    if feature_name not in feature_dict["feature"]:
        raise ValueError(f"Feature '{feature_name}' not found")

    has_channels = "ch_names" in feature_dict and feature_dict["ch_names"]
    if has_channels and channel_name is None:
        raise ValueError("Data has channel information. Please select a specific channel.")
    if not has_channels and channel_name is not None:
        channel_name = None  # 忽略传入的通道名

    feature_data = feature_dict["feature"][feature_name]

    if channel_name is not None and has_channels:
        if channel_name not in feature_dict["ch_names"]:
            raise ValueError(f"Channel '{channel_name}' not found")
        channel_index = feature_dict["ch_names"].index(channel_name)
        data = feature_data[channel_index]
    else:
        data = feature_data

    # 展平为一维列表
    if isinstance(data, (np.ndarray, list)):
        data = np.asarray(data).ravel().tolist()
    else:
        data = [data]

    # 转换为 float，无法转换的置为 NaN
    numeric_data = []
    for val in data:
        try:
            numeric_data.append(float(val))
        except (ValueError, TypeError):
            numeric_data.append(np.nan)
    return numeric_data


def convert_to_significance_dict(result):
    """将统计结果转换为显著性字典，用于绘图"""
    significance_dict = {}
    for entry in result:
        p_val = entry.get('corrected_p_value', entry['p_value'])
        if not pd.isna(p_val):
            group1, group2 = entry['group_comparison'].split(' vs ')
            significance_dict[(group1, group2)] = p_val
    return significance_dict


# ========== 原有的各个对话框类 ==========
class GroupDialog(QDialog):
    def __init__(self, parent=None, group_name="", folder_path=""):
        super().__init__(parent)
        self.setWindowTitle("Group Configuration")
        self._init_ui(group_name, folder_path)

    def _init_ui(self, group_name, folder_path):
        layout = QVBoxLayout()
        name_label = QLabel("Group Name:")
        self.name_edit = QLineEdit(group_name)
        layout.addWidget(name_label)
        layout.addWidget(self.name_edit)
        folder_label = QLabel("Data Folder:")
        self.folder_edit = QLineEdit(folder_path)
        folder_button = BFPushButton("Browse")
        folder_button.clicked.connect(self._select_folder)
        layout.addWidget(folder_label)
        layout.addWidget(self.folder_edit)
        layout.addWidget(folder_button)
        confirm_button = BFPushButton("Confirm")
        confirm_button.clicked.connect(self.accept)
        layout.addWidget(confirm_button)
        self.setLayout(layout)

    def _select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if path:
            self.folder_edit.setText(path)

    def get_group_data(self):
        return self.name_edit.text(), self.folder_edit.text()


class ChannelSelectionDialog(QDialog):
    def __init__(self, available_channels, selected_channels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Channel Selection")
        self.setMinimumSize(300, 300)
        self._init_ui(available_channels, selected_channels)

    def _init_ui(self, channels, selected):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        self.checkboxes = {}
        for channel in channels:
            cb = QCheckBox(channel)
            cb.setChecked(channel in selected)
            scroll_layout.addWidget(cb)
            self.checkboxes[channel] = cb
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        confirm_button = BFPushButton("Confirm")
        confirm_button.clicked.connect(self.accept)
        layout.addWidget(confirm_button)
        self.setLayout(layout)

    def get_selected_channels(self):
        return [ch for ch, cb in self.checkboxes.items() if cb.isChecked()]


class VisualisationSettingsDialog(QDialog):
    def __init__(self, title="Default Title", y_label="Y Axis", x_label="X Axis",
                 legend=None, x_ticks=None, y_range=None, color="#ff0000", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Customization")
        self.setGeometry(100, 100, 300, 300)
        self._init_ui(title, y_label, x_label, legend, x_ticks, y_range, color)

    def _init_ui(self, title, y_label, x_label, legend, x_ticks, y_range, color):
        layout = QFormLayout(self)
        self.title_input = QLineEdit(title)
        self.y_label_input = QLineEdit(y_label)
        self.x_label_input = QLineEdit(x_label)
        self.legend_input = QLineEdit(", ".join(legend) if legend else "")
        self.x_ticks_input = QLineEdit(", ".join(x_ticks) if x_ticks else "")
        self.y_range_input = QLineEdit(f"{y_range[0]}, {y_range[1]}" if y_range else "0, 10")
        self.color_input = QLineEdit(color)
        layout.addRow("Title:", self.title_input)
        layout.addRow("Y Label:", self.y_label_input)
        layout.addRow("X Label:", self.x_label_input)
        layout.addRow("Legend (comma-separated):", self.legend_input)
        layout.addRow("X Ticks (comma-separated):", self.x_ticks_input)
        layout.addRow("Y Range (min, max):", self.y_range_input)
        layout.addRow("Color (hex):", self.color_input)
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.accept)
        layout.addRow(self.confirm_button)

    def get_settings(self):
        title = self.title_input.text()
        y_label = self.y_label_input.text()
        x_label = self.x_label_input.text()
        legend = self.legend_input.text().split(", ")
        x_ticks = self.x_ticks_input.text().split(", ")
        y_range = tuple(map(float, self.y_range_input.text().split(",")))
        color = self.color_input.text()
        return title, y_label, x_label, legend, x_ticks, y_range, color


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._initialize_plot_components()
        self.plot_data()

    def _initialize_plot_components(self):
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_default(self):
        data = [np.random.randn(100) for _ in range(5)]
        self.plot(data=data, title="Sample Boxplot", y_label="Values", x_label="Groups",
                  x_ticks=["A", "B", "C", "D", "E"])

    def plot(self, data, title="", y_label="", x_label="", legend=None,
             x_ticks=None, y_range=None, color="blue"):
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
        # 示例绘图（可根据需要修改）
        pass


# ========== 主窗口 ==========
class StatisticalAnalysisDialog(QMainWindow):
    def __init__(self):
        super().__init__()
        self._initialize_state_variables()
        self._configure_window_properties()
        self._create_main_layout()
        self._connect_signal_handlers()

    def _initialize_state_variables(self):
        self.result = []
        self.group_select_features = {}
        self.data = None
        self.group_folder = {}
        self.group_files = {}
        self.is_valid = False

    def _configure_window_properties(self):
        self.setWindowTitle("Statistical Analysis")
        self.setGeometry(100, 100, 1200, 800)

    def _create_main_layout(self):
        main_layout = QHBoxLayout()
        central_widget = QWidget()
        self.group_widget = self._create_group_management_section()
        analysis_visual_widget = self._create_analysis_visualization_section()
        main_layout.addWidget(self.group_widget)
        main_layout.addWidget(analysis_visual_widget)
        main_layout.setStretch(0, 25)
        main_layout.setStretch(1, 75)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _create_group_management_section(self):
        widget = QFrame()
        widget.setFrameShape(QFrame.Box)
        widget.setStyleSheet("background-color: white;")
        layout = QVBoxLayout()
        title_label = QLabel("Group Management")
        title_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
        layout.addWidget(title_label)
        button_layout = QHBoxLayout()
        self.new_group_button = BFPushButton("Add Group")
        self.import_groups_button = BFPushButton("Import Groups")
        button_layout.addWidget(self.new_group_button)
        button_layout.addWidget(self.import_groups_button)
        layout.addLayout(button_layout)
        self.group_list = QListWidget()
        layout.addWidget(self.group_list)
        self.validate_button = BFPushButton("Validate Groups")
        layout.addWidget(self.validate_button)
        widget.setLayout(layout)
        return widget

    def _create_analysis_visualization_section(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.analysis_widget = self._create_statistical_analysis_section()
        self.visual_widget = self._create_visualization_section()
        layout.addWidget(self.analysis_widget)
        layout.addWidget(self.visual_widget)
        layout.setStretch(0, 3)
        layout.setStretch(1, 7)
        widget.setLayout(layout)
        return widget

    def _create_statistical_analysis_section(self):
        widget = QFrame()
        widget.setFrameShape(QFrame.Box)
        widget.setStyleSheet("background-color: white;")
        outer_layout = QVBoxLayout()
        layout = QGridLayout()

        title_label = QLabel("Statistical Analysis")
        title_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
        outer_layout.addWidget(title_label)

        description = QLabel('Performs pairwise group comparisons using selected method. '
                             'Multiple comparison correction applied when >2 groups exist.')
        description.setStyleSheet("font-family: 'Times New Roman'; font-size: 10pt; color: blue;")
        description.setWordWrap(True)
        outer_layout.addWidget(description)
        outer_layout.addLayout(layout)
        outer_layout.addStretch(1)

        channel_label = QLabel("Channel:")
        self.channel_combo = QComboBox()
        layout.addWidget(channel_label, 0, 0)
        layout.addWidget(self.channel_combo, 0, 1)

        feature_label = QLabel("Feature:")
        self.feature_combo = QComboBox()
        layout.addWidget(feature_label, 1, 0)
        layout.addWidget(self.feature_combo, 1, 1)

        method_label = QLabel("Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItem("t-test")
        self.method_combo.addItem("t-test(paired)")
        self.method_combo.addItem("anova")
        self.method_combo.addItem("mann-whitney U")
        self.method_combo.addItem("wilcoxon(paired)")
        self.method_combo.addItem("kruskal-wallis")
        layout.addWidget(method_label, 2, 0)
        layout.addWidget(self.method_combo, 2, 1)

        correction_label = QLabel("Correction:")
        self.correction_combo = QComboBox()
        self.correction_combo.addItems(["bonferroni", "fdr_bh", "fdr_by", "holm-sidak", "sidak"])
        self.enable_correction = QCheckBox("Enable")
        self.enable_correction.setChecked(True)
        layout.addWidget(correction_label, 3, 0)
        layout.addWidget(self.correction_combo, 3, 1)
        layout.addWidget(self.enable_correction, 3, 2)

        self.run_button = BFPushButton("Run Analysis")
        self.run_button.setFixedWidth(120)
        self.status_label = QLabel("Status: Waiting")
        self.export_button = BFPushButton("Export Results")
        self.export_button.setFixedWidth(120)
        layout.addWidget(self.run_button, 4, 0)
        layout.addWidget(self.status_label, 4, 1)
        layout.addWidget(self.export_button, 4, 2)

        widget.setLayout(outer_layout)
        return widget

    def _create_visualization_section(self):
        widget = QFrame()
        widget.setFrameShape(QFrame.Box)
        widget.setStyleSheet("background-color: white;")
        layout = QVBoxLayout()
        self.scroll_area = QScrollArea()
        title_label = QLabel("Data Visualization")
        title_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
        layout.addWidget(title_label)
        control_layout = QHBoxLayout()
        plot_type_label = QLabel('Plot Type: ')
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(['scatter plot', 'density histogram', 'box plot', 'violin plot'])
        self.plot_type_combo.setFixedHeight(25)
        self.plot_button = BFPushButton("Generate Plot")
        self.save_button = BFPushButton("Save Image")
        self.save_button.setFixedWidth(100)
        self.settings_button = BFPushButton("Plot Settings")
        self.settings_button.setFixedWidth(100)
        control_layout.addWidget(plot_type_label)
        control_layout.addWidget(self.plot_type_combo)
        control_layout.addWidget(self.plot_button)
        control_layout.addStretch(1)
        control_layout.addWidget(self.save_button)
        control_layout.addWidget(self.settings_button)
        self.plot_container = ScatterPlotWindow()
        self.scroll_area.setWidget(self.plot_container)
        layout.addLayout(control_layout)
        layout.addWidget(self.scroll_area)
        widget.setLayout(layout)
        return widget

    def _connect_signal_handlers(self):
        self.new_group_button.clicked.connect(self.add_group)
        self.validate_button.clicked.connect(self.validate_groups)
        self.group_list.itemDoubleClicked.connect(self.modify_group)
        self.run_button.clicked.connect(self.run_analysis)
        self.export_button.clicked.connect(self.export_results)
        self.plot_button.clicked.connect(self.generate_visualization)

    def add_group(self):
        dialog = GroupDialog(self)
        if dialog.exec_():
            name, folder = dialog.get_group_data()
            self.group_list.addItem(f"{name} - {folder}")
        self.is_valid = False

    def modify_group(self, item):
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
        self.group_files.clear()
        self.group_select_features.clear()
        self.group_folder.clear()

        if self.group_list.count() < 2:
            QMessageBox.warning(self, "Warning", "At least two groups required for analysis!")
            return

        for i in range(self.group_list.count()):
            try:
                group_data = self.group_list.item(i).text()
                group_name, folder_path = group_data.split(" - ")
                self.group_files[group_name] = []
                if not os.path.exists(folder_path):
                    QMessageBox.warning(self, "Error", f"Folder {folder_path} does not exist!")
                    return
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.xlsx'):
                        file_path = os.path.join(folder_path, file_name)
                        print(f"读取: {file_name} (xlsx)")
                        file_data = read_xlsx(file_path)
                        self.group_files[group_name].append(file_data)
                if not self.group_files[group_name]:
                    QMessageBox.warning(self, "Error", f"No XLSX files found in {folder_path}!")
                    return
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Load group data failed: {str(e)}")
                return

        try:
            self.data = next(iter(self.group_files.values()))[0]
            self.feature_combo.clear()
            self.feature_combo.addItems(list(self.data['feature'].keys()))
            if 'ch_names' in self.data and self.data['ch_names']:
                self.channel_combo.clear()
                self.channel_combo.addItems(self.data['ch_names'])
                self.channel_combo.setEnabled(True)
            else:
                self.channel_combo.clear()
                self.channel_combo.addItem("<无通道>")
                self.channel_combo.setEnabled(False)
            self.status_label.setText("Status: Waiting")
            self.is_valid = True
            QMessageBox.information(self, "Success", "Data validation complete!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Initialize UI failed: {str(e)}")
            self.is_valid = False

    def run_analysis(self):
        if not self.is_valid:
            QMessageBox.warning(self, "Action Required", "Validate groups before analysis!")
            return

        self.status_label.setText("Status: Processing...")
        QApplication.processEvents()

        try:
            self.group_select_features.clear()
            for group_name in self.group_files:
                self.group_select_features[group_name] = []
                for dataset in self.group_files[group_name]:
                    if self.channel_combo.isEnabled():
                        channel = self.channel_combo.currentText()
                    else:
                        channel = None
                    feature = self.feature_combo.currentText()
                    self.group_select_features[group_name].extend(get_feature(dataset, channel, feature))

            # 过滤无效数值
            for group_name in list(self.group_select_features.keys()):
                cleaned = []
                for val in self.group_select_features[group_name]:
                    try:
                        cleaned.append(float(val))
                    except (ValueError, TypeError):
                        cleaned.append(np.nan)
                data = np.array(cleaned)
                data = data[np.isfinite(data)]
                if len(data) == 0:
                    raise ValueError(f"Group '{group_name}' has no valid finite data after filtering.")
                self.group_select_features[group_name] = data.tolist()

            raw_method = self.method_combo.currentText()
            method = raw_method.strip()
            is_paired = False
            if method == 't-test(paired)':
                method = 't-test'
                is_paired = True
            elif method == 'wilcoxon(paired)':
                method = 'wilcoxon'
                is_paired = True
            method = method.strip()

            self.result = calculate_significance(
                self.group_select_features,
                method=method,
                paired=is_paired
            )

            if self.enable_correction.isChecked() and len(self.result) > 1:
                correct_method = self.correction_combo.currentText().strip()
                self.result = multiple_comparison_correction(self.result, correct_method)

            self.status_label.setText("Status: Completed!")

            # ====== 弹出结果对话框 ======
            result_dialog = ResultDialog(self.result, self)
            result_dialog.exec_()

            # 可选的消息提示（可省略）
            QMessageBox.information(self, "Success", "Statistical analysis completed!")

        except Exception as e:
            self.status_label.setText("Status: Failed!")
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")

    def export_results(self):
        if not self.result:
            QMessageBox.warning(self, "Warning", "No analysis results to export!")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "Excel Files (*.xlsx);;JSON Files (*.json)")
        if not save_path:
            return
        try:
            result_df = pd.DataFrame(self.result)
            if save_path.endswith('.xlsx'):
                result_df.to_excel(save_path, index=False)
            else:
                result_df.to_json(save_path, orient='records', indent=4)
            QMessageBox.information(self, "Success", f"Results exported to {save_path} successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def generate_visualization(self):
        if not self.group_select_features:
            QMessageBox.warning(self, "Data Required", "Run analysis before generating plot!")
            return

        try:
            plot_type = self.plot_type_combo.currentText()
            old = self.scroll_area.takeWidget()
            if old:
                old.deleteLater()

            if plot_type == 'scatter plot':
                self.plot_container = ScatterPlotWindow()
                self.plot_container.plot_scatter(self.group_select_features)
            elif plot_type == 'density histogram':
                self.plot_container = DensityHistogramWindow()
                self.plot_container.plot_density_histogram(self.group_select_features)
            elif plot_type == 'box plot':
                self.plot_container = SignificanceBoxPlotWindow()
                significance_dict = convert_to_significance_dict(self.result)
                self.plot_container.plot_boxplot_with_significance(self.group_select_features, significance_dict)
            elif plot_type == 'violin plot':
                self.plot_container = SignificanceViolinPlotWindow()
                significance_dict = convert_to_significance_dict(self.result)
                self.plot_container.plot_violin_with_significance(self.group_select_features, significance_dict)
            else:
                QMessageBox.warning(self, "Error", f"Unknown plot type: {plot_type}")
                return

            self.scroll_area.setWidget(self.plot_container)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Generate plot failed: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StatisticalAnalysisDialog()
    window.show()
    sys.exit(app.exec_())