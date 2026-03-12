import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到 sys.path
start_path = Path(__file__).resolve().parent
for parent in [start_path] + list(start_path.parents):
    if parent.name == 'core':
        project_root = parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        break
else:
    raise RuntimeError("未找到名为 'core' 的目录")

# PyQt5 完整导入
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QPushButton, QListWidget, QFileDialog,
                             QDialog, QLineEdit, QLabel, QMessageBox, QCheckBox,
                             QComboBox, QScrollArea, QFrame, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView)
from PyQt5.QtCore import Qt, QSize, QTimer, QEvent
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon

# 导入统计核心代码
from core.processing.Statistical_Analysis.significance_test import calculate_significance, \
    multiple_comparison_correction

BFPushButton = QPushButton


# ========== 辅助类：下拉框箭头按钮（最终版） ==========
class ComboArrowButton(QPushButton):
    def __init__(self, parent, combo_box, resource_path):
        super().__init__(combo_box)
        self.combo = combo_box
        self.resource_path = Path(resource_path)
        self._view = None
        self._filter_installed = False

        self.setFixedSize(30, 30)
        self.setStyleSheet("border: none; background: transparent; cursor: pointer;")
        self.move(combo_box.width() - 35, 0)

        # 加载图片（11.png = 收起状态，10.png = 展开状态）
        self.normal_pix = QPixmap(str(self.resource_path / "11.png"))
        self.open_pix = QPixmap(str(self.resource_path / "10.png"))
        self._update_icon(False)

        self.raise_()

    def _update_icon(self, is_open):
        target_pix = self.open_pix if is_open else self.normal_pix
        if not target_pix.isNull():
            scaled = target_pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setIcon(QIcon(scaled))
            self.setIconSize(self.size())

    def _install_view_filter(self):
        if not self._filter_installed:
            view = self.combo.view()
            if view is not None:
                view.installEventFilter(self)
                self._view = view
                self._filter_installed = True

    def eventFilter(self, obj, event):
        if obj is self._view:
            if event.type() == QEvent.Show:
                self._update_icon(True)
            elif event.type() == QEvent.Hide:
                self._update_icon(False)
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        view = self.combo.view()
        is_open = view and view.isVisible()
        if not is_open:
            self.combo.showPopup()
            self._install_view_filter()
        else:
            self.combo.hidePopup()
        super().mousePressEvent(event)


# ========== 结果展示对话框（背景设为白色） ==========
class ResultDialog(QDialog):
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Statistical Analysis Results")
        self.resize(1000, 600)
        self.results = results
        self.init_ui()

    def init_ui(self):
        # 设置对话框背景为白色
        self.setStyleSheet("QDialog { background-color: white; }")

        layout = QVBoxLayout(self)
        title = QLabel("Pairwise Group Comparison Results")
        title.setStyleSheet("font-size: 18pt; font-weight: bold; font-family: 'Times New Roman';")
        layout.addWidget(title)

        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)

        columns = ["Group Comparison", "Method", "Statistic", "p-value", "Corrected p-value", "Significant"]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)

        if not self.results or len(self.results) == 0:
            self.table.setRowCount(1)
            empty_tip = QTableWidgetItem("⚠️ 无有效统计结果，请检查数据")
            empty_tip.setTextAlignment(Qt.AlignCenter)
            empty_tip.setForeground(Qt.red)
            self.table.setItem(0, 0, empty_tip)
            self.table.setSpan(0, 0, 1, len(columns))
        else:
            self.table.setRowCount(len(self.results))
            for i, res in enumerate(self.results):
                self.table.setItem(i, 0, QTableWidgetItem(res.get("group_comparison", "")))
                self.table.setItem(i, 1, QTableWidgetItem(res.get("method", "")))
                self.table.setItem(i, 2, QTableWidgetItem(f"{res.get('stat', 0):.4f}"))
                self.table.setItem(i, 3, QTableWidgetItem(f"{res.get('p_value', 1):.4f}"))
                corr_p = res.get("corrected_p_value", None)
                self.table.setItem(i, 4, QTableWidgetItem(f"{corr_p:.4f}" if corr_p is not None else "N/A"))
                sig = res.get("significant_after_correction", res.get("p_value", 1) < 0.05)
                sig_item = QTableWidgetItem("Yes" if sig else "No")
                sig_item.setForeground(Qt.darkGreen if sig else Qt.gray)
                self.table.setItem(i, 5, sig_item)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        for col in range(2, len(columns)):
            header.setSectionResizeMode(col, QHeaderView.Stretch)
        header.setMinimumSectionSize(100)

        layout.addWidget(self.table)

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
        if not self.results or len(self.results) == 0:
            QMessageBox.warning(self, "Warning", "无有效结果可导出！")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "Excel Files (*.xlsx)")
        if not save_path:
            return
        try:
            df = pd.DataFrame(self.results)
            df.to_excel(save_path, index=False)
            QMessageBox.information(self, "Success", f"Results exported to {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")


# ========== 组配置对话框（背景设为白色） ==========
class GroupDialog(QDialog):
    def __init__(self, parent=None, group_name="", folder_path=""):
        super().__init__(parent)
        self.setWindowTitle("Group Configuration")
        self._init_ui(group_name, folder_path)

    def _init_ui(self, group_name, folder_path):
        # 设置对话框背景为白色
        self.setStyleSheet("QDialog { background-color: white; }")

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


# ========== Excel 读取函数 ==========
def read_xlsx(file_path):
    try:
        df = pd.read_excel(file_path, engine='openpyxl', sheet_name=0)
        channel_col = df.columns[0]
        ch_names = df[channel_col].astype(str).tolist()
        feature_dict = {}
        for col in df.columns[1:]:
            numeric_vals = pd.to_numeric(df[col], errors='coerce')
            vals = numeric_vals.tolist()
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
    if feature_name not in feature_dict["feature"]:
        raise ValueError(f"Feature '{feature_name}' not found")
    has_channels = "ch_names" in feature_dict and feature_dict["ch_names"]
    if has_channels and channel_name is None:
        raise ValueError("Data has channel information. Please select a specific channel.")
    if not has_channels and channel_name is not None:
        channel_name = None
    feature_data = feature_dict["feature"][feature_name]
    if channel_name is not None and has_channels:
        if channel_name not in feature_dict["ch_names"]:
            raise ValueError(f"Channel '{channel_name}' not found")
        channel_index = feature_dict["ch_names"].index(channel_name)
        data = feature_data[channel_index]
    else:
        data = feature_data
    if isinstance(data, (np.ndarray, list)):
        data = np.asarray(data).ravel().tolist()
    else:
        data = [data]
    numeric_data = []
    for val in data:
        try:
            numeric_data.append(float(val))
        except (ValueError, TypeError):
            numeric_data.append(np.nan)
    return numeric_data


# ========== 辅助类：图片导航按钮（无压缩版） ==========
class ImageNavButton(QPushButton):
    def __init__(self, parent, nav_name, base_path, selected_callback=None):
        super().__init__(parent)
        self.nav_name = nav_name
        self.base_path = Path(base_path)
        self.selected_callback = selected_callback
        self._is_selected = False

        self.selected_img_path = self.base_path / "Buttons" / "Selected" / f"{nav_name}_Button.png"
        self.unselected_img_path = self.base_path / "Buttons" / "Unselected" / f"{nav_name}_Button.png"

        self.original_size = QSize(240, 70)
        if self.unselected_img_path.exists():
            pix = QPixmap(str(self.unselected_img_path))
            if not pix.isNull():
                self.original_size = pix.size()

        self.setFixedSize(self.original_size)
        self.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                background-repeat: no-repeat;
                background-position: center;
            }
        """)
        self.setCursor(Qt.PointingHandCursor)
        self.set_selected(False)
        self.clicked.connect(self._on_click)

    def set_selected(self, selected):
        self._is_selected = selected
        img_path = self.selected_img_path if selected else self.unselected_img_path

        if img_path.exists():
            path_str = str(img_path).replace("\\", "/")
            self.setStyleSheet(f"""
                QPushButton {{
                    border: none;
                    background-color: transparent;
                    background-image: url({path_str});
                    background-repeat: no-repeat;
                    background-position: center;
                }}
            """)
        else:
            self.setText(self.nav_name.replace("_", " "))

    def _on_click(self):
        if self.selected_callback:
            self.selected_callback(self)


# ========== 核心：统计分析面板 ==========
class AnalysisPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._initialize_state_variables()
        self._create_main_layout()
        self._connect_signal_handlers()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: transparent;")

    def _initialize_state_variables(self):
        self.result = []
        self.group_select_features = {}
        self.data = None
        self.group_folder = {}
        self.group_files = {}
        self.is_valid = False

    def _create_main_layout(self):
        # ========== 1. 组管理区域 ==========
        self.group_widget = QFrame(self)
        self.group_widget.setGeometry(0, 9, 258, 1006)
        self.group_widget.setStyleSheet("""
            QFrame {
                border-radius: 40px;
                background-color: transparent;
            }
        """)

        # --- 组管理背景图 (2.png) ---
        self.group_bg_label = QLabel(self.group_widget)
        self.group_bg_label.setGeometry(0, 0, 258, 1006)
        group_bg_path = Path(__file__).parent.parent / "UI_resource" / "Analysis_Panel" / "2.png"
        if group_bg_path.exists():
            pixmap = QPixmap(str(group_bg_path))
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(258, 1006, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                self.group_bg_label.setPixmap(scaled_pixmap)
                self.group_bg_label.setStyleSheet("border-radius: 40px;")
        self.group_bg_label.lower()

        self._init_group_widget_content()

        # --- 组管理装饰图 (3, 4, 5, 6.png) ---
        # 3.png
        self.group_deco_label = QLabel(self.group_widget)
        self.group_deco_label.setGeometry(29, 32, 21, 16)
        deco1_path = Path(__file__).parent.parent / "UI_resource" / "Analysis_Panel" / "3.png"
        if deco1_path.exists():
            pixmap = QPixmap(str(deco1_path))
            if not pixmap.isNull():
                self.group_deco_label.setPixmap(pixmap.scaled(21, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.group_deco_label.raise_()

        # 4.png
        self.group_deco_label_2 = QLabel(self.group_widget)
        self.group_deco_label_2.setGeometry(60, 20, 72, 38)
        deco2_path = Path(__file__).parent.parent / "UI_resource" / "Analysis_Panel" / "4.png"
        if deco2_path.exists():
            pixmap = QPixmap(str(deco2_path))
            if not pixmap.isNull():
                self.group_deco_label_2.setPixmap(pixmap.scaled(72, 38, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.group_deco_label_2.raise_()

        # 5.png
        self.group_deco_label_3 = QLabel(self.group_widget)
        self.group_deco_label_3.setGeometry(28, 99, 23, 23)
        deco3_path = Path(__file__).parent.parent / "UI_resource" / "Analysis_Panel" / "5.png"
        if deco3_path.exists():
            pixmap = QPixmap(str(deco3_path))
            if not pixmap.isNull():
                self.group_deco_label_3.setPixmap(pixmap.scaled(23, 23, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.group_deco_label_3.raise_()

        # 6.png
        self.group_deco_label_4 = QLabel(self.group_widget)
        self.group_deco_label_4.setGeometry(149, 101, 16, 18)
        deco4_path = Path(__file__).parent.parent / "UI_resource" / "Analysis_Panel" / "6.png"
        if deco4_path.exists():
            pixmap = QPixmap(str(deco4_path))
            if not pixmap.isNull():
                self.group_deco_label_4.setPixmap(pixmap.scaled(16, 18, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.group_deco_label_4.raise_()

        # ========== 2. 数据分析区域 ==========
        self.analysis_widget = QFrame(self)
        self.analysis_widget.setGeometry(275, 12, 855, 375)
        self.analysis_widget.setStyleSheet("""
            QFrame {
                border-radius: 40px;
                background-color: transparent;
            }
        """)

        # --- 数据分析背景图 (8.png) ---
        self.analysis_bg_label = QLabel(self.analysis_widget)
        self.analysis_bg_label.setGeometry(0, 0, 855, 375)
        analysis_bg_path = Path(__file__).parent.parent / "UI_resource" / "Analysis_Panel" / "8.png"
        if analysis_bg_path.exists():
            pixmap = QPixmap(str(analysis_bg_path))
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(855, 375, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                self.analysis_bg_label.setPixmap(scaled_pixmap)
                self.analysis_bg_label.setStyleSheet("border-radius: 40px;")
        self.analysis_bg_label.lower()

        # --- 初始化内容 ---
        self._init_analysis_widget_content()

        # --- 数据分析装饰图 (12.png, 13.png) ---
        # 12.png
        self.analysis_deco_label_12 = QLabel(self.analysis_widget)
        self.analysis_deco_label_12.setGeometry(45, 27, 23, 22)
        deco12_path = Path(__file__).parent.parent / "UI_resource" / "Analysis_Panel" / "12.png"
        if deco12_path.exists():
            pixmap = QPixmap(str(deco12_path))
            if not pixmap.isNull():
                self.analysis_deco_label_12.setPixmap(
                    pixmap.scaled(23, 22, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.analysis_deco_label_12.raise_()

        # 13.png
        self.analysis_deco_label_13 = QLabel(self.analysis_widget)
        self.analysis_deco_label_13.setGeometry(50, 315, 18, 21)
        deco13_path = Path(__file__).parent.parent / "UI_resource" / "Analysis_Panel" / "13.png"
        if deco13_path.exists():
            pixmap = QPixmap(str(deco13_path))
            if not pixmap.isNull():
                self.analysis_deco_label_13.setPixmap(
                    pixmap.scaled(18, 21, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.analysis_deco_label_13.raise_()

        # ========== 3. 数据可视化区域 ==========
        self.visual_widget = QFrame(self)
        self.visual_widget.setGeometry(275, 395, 855, 614)
        self.visual_widget.setStyleSheet("""
            QFrame {
                border-radius: 40px;
                background: rgba(255, 255, 255, 0.6);
                box-shadow: inset 4px 2px 4px rgba(0, 0, 0, 0.25), 4px 2px 4px rgba(0, 0, 0, 0.25);
            }
        """)

        # --- 15.png 背景图 (位于内容之下) ---
        self.visual_bg_label = QLabel(self.visual_widget)
        self.visual_bg_label.setGeometry(0, 0, 855, 614)
        visual_bg_path = Path(__file__).parent.parent / "UI_resource" / "Analysis_Panel" / "15.png"
        if visual_bg_path.exists():
            pixmap = QPixmap(str(visual_bg_path))
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(855, 614, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                self.visual_bg_label.setPixmap(scaled_pixmap)
                self.visual_bg_label.setStyleSheet("border-radius: 40px;")
        self.visual_bg_label.lower()  # 置于最底层，让内容显示在上方

        # --- 初始化内容 ---
        self._init_visual_widget_content()

        # ========== 4. 最顶层：Project_Name.png ==========
        self.project_name_label = QLabel(self)
        self.project_name_label.setGeometry(720, 324, 520, 139)  # 按比例缩放
        project_img_path = Path(
            __file__).parent.parent / "UI_resource" / "Feature_Extraction_Panel" / "Project_Name.png"
        if project_img_path.exists():
            pixmap = QPixmap(str(project_img_path))
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(self.project_name_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.project_name_label.setPixmap(scaled_pixmap)
        self.project_name_label.raise_()  # 保持在最上层

    def _init_group_widget_content(self):
        layout = QVBoxLayout(self.group_widget)
        layout.setContentsMargins(20, 30, 20, 30)
        layout.setSpacing(15)

        title_label = QLabel("")
        title_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 18pt; font-weight: bold; color: #333;")
        layout.addWidget(title_label)

        button_layout = QHBoxLayout()
        self.new_group_button = BFPushButton("     添加组")
        self.import_groups_button = BFPushButton("   导入组")
        for btn in [self.new_group_button, self.import_groups_button]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255,255,255,0.8);
                    color: #333;
                    border: 2px solid rgba(0,0,0,0.1);
                    padding: 10px;
                    border-radius: 15px;
                    font-weight: bold;
                    font-family: 'Times New Roman';
                    font-size: 9pt;
                }
                QPushButton:hover {
                    background-color: white;
                    border-color: #4a90e2;
                }
            """)
        button_layout.addWidget(self.new_group_button)
        button_layout.addWidget(self.import_groups_button)
        layout.addLayout(button_layout)

        self.group_list = QListWidget()
        self.group_list.setStyleSheet("""
            QListWidget {
                background-color: rgba(255,255,255,0.7);
                border-radius: 20px;
                border: none;
                padding: 10px;
                font-family: 'Times New Roman';
                font-size: 11pt;
            }
        """)
        layout.addWidget(self.group_list)

        self.validate_button = BFPushButton("√     确认")
        self.validate_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: #333;
                border: 1px solid #e0e0e0;
                padding: 12px;
                border-radius: 20px;
                font-weight: bold;
                font-family: 'Times New Roman';
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #f8f9fa;
            }
        """)
        layout.addWidget(self.validate_button)

    def _init_analysis_widget_content(self):
        layout = QVBoxLayout(self.analysis_widget)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(10)

        title_label = QLabel("       数据分析")
        title_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 15pt; font-weight: bold; color: #333;")
        layout.addWidget(title_label)

        description = QLabel('使用选定方法进行成组两两比较。当组别数量大于2时，将执行多重比较校正。')
        description.setStyleSheet("font-family: 'Times New Roman'; font-size: 11pt; color: #2980b9;")
        description.setWordWrap(True)
        layout.addWidget(description)

        # 下拉框样式
        combo_bg_path = Path(__file__).parent.parent / "UI_resource" / "Analysis_Panel" / "9.png"
        combo_bg_str = str(combo_bg_path).replace("\\", "/")
        combo_style = f"""
            QComboBox {{
                padding: 8px;
                padding-left: 15px;
                padding-right: 30px;
                border-radius: 10px;
                background-color: transparent;
                background-image: url({combo_bg_str});
                background-repeat: no-repeat;
                background-position: center;
                font-family: 'Times New Roman';
                font-size: 11pt;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 0px;
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0px;
                height: 0px;
            }}
            QComboBox QAbstractItemView {{
                border: 1px solid #ddd;
                border-radius: 10px;
                background-color: white;
                padding: 5px;
                selection-background-color: #e0e0e0;
            }}
        """
        resource_path = Path(__file__).parent.parent / "UI_resource" / "Analysis_Panel"

        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        grid_layout.setContentsMargins(0, 10, 0, 10)
        label_style = "font-family: 'Times New Roman'; font-size: 12pt; font-weight: bold; color: #444;"

        # --- 1. Channel ---
        channel_label = QLabel("Channel:")
        channel_label.setStyleSheet(label_style)
        self.channel_combo = QComboBox()
        self.channel_combo.setFixedWidth(500)
        self.channel_combo.setStyleSheet(combo_style)
        self.channel_arrow_btn = ComboArrowButton(self.analysis_widget, self.channel_combo, resource_path)
        grid_layout.addWidget(channel_label, 0, 0)
        grid_layout.addWidget(self.channel_combo, 0, 1)

        # --- 2. Feature ---
        feature_label = QLabel("Feature:")
        feature_label.setStyleSheet(label_style)
        self.feature_combo = QComboBox()
        self.feature_combo.setFixedWidth(500)
        self.feature_combo.setStyleSheet(combo_style)
        self.feature_arrow_btn = ComboArrowButton(self.analysis_widget, self.feature_combo, resource_path)
        grid_layout.addWidget(feature_label, 1, 0)
        grid_layout.addWidget(self.feature_combo, 1, 1)

        # --- 3. Method ---
        method_label = QLabel("Method:")
        method_label.setStyleSheet(label_style)
        self.method_combo = QComboBox()
        self.method_combo.setFixedWidth(500)
        self.method_combo.setStyleSheet(combo_style)
        self.method_combo.addItems([
            "t-test", "t-test(paired)", "anova",
            "mann-whitney U", "wilcoxon(paired)", "kruskal-wallis"
        ])
        self.method_arrow_btn = ComboArrowButton(self.analysis_widget, self.method_combo, resource_path)
        grid_layout.addWidget(method_label, 2, 0)
        grid_layout.addWidget(self.method_combo, 2, 1)

        # --- 4. Correction ---
        correction_label = QLabel("Correction:")
        correction_label.setStyleSheet(label_style)
        self.correction_combo = QComboBox()
        self.correction_combo.setFixedWidth(500)
        self.correction_combo.setStyleSheet(combo_style)
        self.correction_combo.addItems(["bonferroni", "fdr_bh", "fdr_by", "holm-sidak", "sidak"])
        self.correction_arrow_btn = ComboArrowButton(self.analysis_widget, self.correction_combo, resource_path)
        grid_layout.addWidget(correction_label, 3, 0)
        grid_layout.addWidget(self.correction_combo, 3, 1)

        # Enable
        self.enable_correction = QCheckBox("使用")
        self.enable_correction.setChecked(True)
        self.enable_correction.setStyleSheet("font-weight: bold; font-family: 'Times New Roman';")
        grid_layout.addWidget(self.enable_correction, 3, 3)
        grid_layout.setColumnStretch(4, 1)

        # --- 底部按钮（使用水平布局紧凑排列） ---
        self.run_button = BFPushButton("    开始")
        self.run_button.setFixedSize(140, 40)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #d4e6f1;
                color: #2c3e50;
                border: 1px solid #b8d4e3;
                border-radius: 15px;
                font-weight: bold;
                font-family: 'Times New Roman';
                font-size: 11pt;
            }
            QPushButton:hover { 
                background-color: #b8d4e3;
            }
        """)

        self.status_label = QLabel("状态：等待")
        self.status_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 11pt; font-style: italic;")

        self.export_button = BFPushButton("导出结果")
        self.export_button.setFixedSize(120, 40)  # 宽度减小为120
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #d4e6f1;
                color: #2c3e50;
                border: 1px solid #b8d4e3;
                border-radius: 15px;
                font-weight: bold;
                font-family: 'Times New Roman';
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #b8d4e3;
            }
        """)

        # 水平布局容纳三个控件，右侧加弹簧防止超出
        button_row_layout = QHBoxLayout()
        button_row_layout.setSpacing(10)
        button_row_layout.addWidget(self.run_button)
        button_row_layout.addWidget(self.status_label)
        button_row_layout.addWidget(self.export_button)
        button_row_layout.addStretch()  # 右侧弹簧，避免溢出

        # 将水平布局添加到网格的第4行第0列，占1行4列（确保空间足够）
        grid_layout.addLayout(button_row_layout, 4, 0, 1, 4)

        layout.addLayout(grid_layout)
        layout.addStretch()

    def _init_visual_widget_content(self):
        layout = QVBoxLayout(self.visual_widget)
        layout.setContentsMargins(30, 20, 30, 30)
        layout.setSpacing(15)

        title_label = QLabel("   数据可视化")
        title_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 18pt; font-weight: bold; color: #333;")
        layout.addWidget(title_label)

        control_layout = QHBoxLayout()
        plot_type_label = QLabel('类型: ')
        plot_type_label.setStyleSheet(
            "font-family: 'Times New Roman'; font-size: 12pt; font-weight: bold; color: #444;")
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 2px solid rgba(0,0,0,0.1);
                border-radius: 10px;
                background-color: white;
                font-family: 'Times New Roman';
                font-size: 11pt;
                min-width: 150px;
            }
        """)
        self.plot_type_combo.addItems(['scatter plot', 'density histogram', 'box plot', 'violin plot'])

        # 生成按钮 - 统一样式
        self.plot_button = BFPushButton("生成")
        self.plot_button.setFixedSize(100, 35)  # 高度35
        self.plot_button.setStyleSheet("""
            QPushButton {
                background-color: #d4e6f1;
                color: #2c3e50;
                border: 1px solid #b8d4e3;
                padding: 8px 15px;
                border-radius: 15px;
                font-weight: bold;
                font-family: 'Times New Roman';
                font-size: 9pt;
            }
            QPushButton:hover { background-color: #b8d4e3; }
        """)

        # 保存按钮
        self.save_button = BFPushButton("保存")
        self.save_button.setFixedSize(100, 35)  # 高度35
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #d4e6f1;
                color: #2c3e50;
                border: 1px solid #b8d4e3;
                padding: 8px 12px;
                border-radius: 13px;
                font-weight: bold;
                font-family: 'Times New Roman';
                font-size: 9pt;
            }
            QPushButton:hover { background-color: #b8d4e3; }
        """)

        # 设置按钮
        self.settings_button = BFPushButton("设置")
        self.settings_button.setFixedSize(100, 35)  # 高度35
        self.settings_button.setStyleSheet("""
            QPushButton {
                background-color: #d4e6f1;
                color: #2c3e50;
                border: 1px solid #b8d4e3;
                padding: 8px 12px;
                border-radius: 13px;
                font-weight: bold;
                font-family: 'Times New Roman';
                font-size: 9pt;
            }
            QPushButton:hover { background-color: #b8d4e3; }
        """)

        control_layout.addWidget(plot_type_label)
        control_layout.addWidget(self.plot_type_combo)
        control_layout.addWidget(self.plot_button)
        control_layout.addStretch(1)
        control_layout.addWidget(self.save_button)
        control_layout.addWidget(self.settings_button)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: rgba(255,255,255,0.8);
                border-radius: 25px;
                border: none;
            }
        """)
        self.plot_container = QLabel("📊 Run analysis first, then generate a plot here")
        self.plot_container.setAlignment(Qt.AlignCenter)
        self.plot_container.setStyleSheet("font-size: 16pt; color: #999; font-family: 'Times New Roman';")
        self.scroll_area.setWidget(self.plot_container)

        layout.addLayout(control_layout)
        layout.addWidget(self.scroll_area)

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
                        print(f"读取: {file_path} (xlsx)")
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
            QMessageBox.information(self, "Success", "✅ Data validation complete!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Initialize UI failed: {str(e)}")
            self.is_valid = False

    def run_analysis(self):
        if not self.is_valid:
            QMessageBox.warning(self, "Action Required", "Validate groups before analysis!")
            return

        self.status_label.setText("Status: Processing...")

        try:
            self.group_select_features.clear()
            for group_name in self.group_files:
                self.group_select_features[group_name] = []
                for dataset in self.group_files[group_name]:
                    channel = self.channel_combo.currentText() if self.channel_combo.isEnabled() else None
                    feature = self.feature_combo.currentText()
                    self.group_select_features[group_name].extend(get_feature(dataset, channel, feature))

            for group_name in list(self.group_select_features.keys()):
                cleaned = [float(v) for v in self.group_select_features[group_name] if isinstance(v, (int, float))]
                data = np.array(cleaned)
                data = data[np.isfinite(data)]
                if len(data) == 0:
                    raise ValueError(f"Group '{group_name}' has no valid finite data.")
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

            self.status_label.setText("状态：完成！")

            if not self.result or len(self.result) == 0:
                QMessageBox.warning(self, "Warning", "⚠️ 未生成有效统计结果")
                return

            result_dialog = ResultDialog(self.result, self)
            result_dialog.exec_()
            QMessageBox.information(self, "Success", "✅ Statistical analysis completed!")

        except Exception as e:
            self.status_label.setText("Status: Failed!")
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")

    def export_results(self):
        if not self.result:
            QMessageBox.warning(self, "Warning", "No analysis results to export!")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "",
                                                   "Excel Files (*.xlsx);;JSON Files (*.json)")
        if not save_path:
            return
        try:
            result_df = pd.DataFrame(self.result)
            if save_path.endswith('.xlsx'):
                result_df.to_excel(save_path, index=False)
            else:
                result_df.to_json(save_path, orient='records', indent=4)
            QMessageBox.information(self, "Success", f"✅ Results exported to {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def generate_visualization(self):
        if not self.group_select_features:
            QMessageBox.warning(self, "Data Required", "⚠️ 请先运行分析（Run Analysis）！")
            return

        try:
            plot_type = self.plot_type_combo.currentText()
            old = self.scroll_area.takeWidget()
            if old:
                old.deleteLater()

            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(10, 10, 10, 10)

            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            import matplotlib.pyplot as plt
            plt.rcParams['font.family'] = 'Times New Roman'

            fig = Figure(figsize=(10, 6), dpi=100)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            groups = list(self.group_select_features.keys())
            data_list = [self.group_select_features[g] for g in groups]
            colors = plt.cm.tab10.colors

            if plot_type == 'scatter plot':
                for i, (name, vals) in enumerate(self.group_select_features.items()):
                    ax.scatter([i] * len(vals), vals, color=colors[i % len(colors)], label=name, alpha=0.7, s=50)
                ax.set_xticks(range(len(groups)))
                ax.set_xticklabels(groups)
                ax.set_title("Scatter Plot", fontsize=14, fontweight='bold')
                ax.legend()
            elif plot_type == 'box plot':
                bp = ax.boxplot(data_list, patch_artist=True, labels=groups)
                for patch, color in zip(bp['boxes'], colors[:len(groups)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.set_title("Box Plot", fontsize=14, fontweight='bold')
            elif plot_type == 'violin plot':
                vp = ax.violinplot(data_list, showmedians=True)
                for i, pc in enumerate(vp['bodies']):
                    pc.set_facecolor(colors[i % len(colors)])
                    pc.set_alpha(0.7)
                ax.set_xticks(range(1, len(groups) + 1))
                ax.set_xticklabels(groups)
                ax.set_title("Violin Plot", fontsize=14, fontweight='bold')
            elif plot_type == 'density histogram':
                for i, (name, vals) in enumerate(self.group_select_features.items()):
                    ax.hist(vals, bins=15, alpha=0.5, density=True,
                            color=colors[i % len(colors)], label=name, edgecolor='black')
                ax.set_title("Density Histogram", fontsize=14, fontweight='bold')
                ax.legend()

            ax.set_xlabel("Groups", fontsize=12)
            ax.set_ylabel("Values", fontsize=12)
            ax.grid(True, alpha=0.3)
            canvas.draw()

            layout.addWidget(canvas)
            self.scroll_area.setWidget(container)

        except Exception as e:
            QMessageBox.critical(self, "绘图错误", f"报错信息:\n{str(e)}")
            import traceback
            print(traceback.format_exc())


# ========== 完整主窗口 ==========
class CompleteAnalysisWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(1440, 1024)
        self.setWindowTitle("BCI Analysis - Statistical Module")
        self.setStyleSheet("background-color: white;")

        self.nav_resource_path = Path(__file__).parent.parent / "UI_resource" / "Navigation"
        self.nav_buttons = []
        self.init_ui()

    def init_ui(self):
        self.left_nav = QWidget(self)
        self.left_nav.setGeometry(9, 9, 270, 1006)
        self.left_nav.setStyleSheet("""
            QWidget {
                border-radius: 40px;
                background-color: transparent;
            }
        """)

        self.bg_label = QLabel(self.left_nav)
        self.bg_label.setGeometry(0, 0, 270, 1006)
        bg_img_path = self.nav_resource_path / "Background.png"
        if bg_img_path.exists():
            pixmap = QPixmap(str(bg_img_path))
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(270, 1006, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                self.bg_label.setPixmap(scaled_pixmap)
                self.bg_label.setStyleSheet("border-radius: 40px;")
        self.bg_label.lower()

        self.nav_content = QWidget(self.left_nav)
        self.nav_content.setGeometry(0, 0, 270, 1006)
        self.nav_content.setAttribute(Qt.WA_TranslucentBackground)

        nav_layout = QVBoxLayout(self.nav_content)
        nav_layout.setContentsMargins(15, 100, 15, 40)
        nav_layout.setSpacing(10)

        logo_label = QLabel("")
        logo_label.setFixedHeight(80)
        nav_layout.addWidget(logo_label)
        nav_layout.addSpacing(10)

        button_names = [
            "Home",
            "Preprocessing",
            "Feature_Extraction",
            "Statistical_Analysis",
            "Virtualization"
        ]

        for name in button_names:
            btn = ImageNavButton(
                self.nav_content,
                name,
                self.nav_resource_path,
                selected_callback=self.on_nav_button_clicked
            )
            self.nav_buttons.append(btn)
            nav_layout.addWidget(btn, 0, Qt.AlignCenter)

            if name == "Statistical_Analysis":
                self.on_nav_button_clicked(btn)

        nav_layout.addStretch()

        self.analysis_panel = AnalysisPanel(self)
        self.analysis_panel.setGeometry(296, 0, 1144, 1024)

    def on_nav_button_clicked(self, clicked_button):
        for btn in self.nav_buttons:
            btn.set_selected(btn == clicked_button)


# ========== 程序入口 ==========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CompleteAnalysisWindow()
    window.show()
    sys.exit(app.exec_())