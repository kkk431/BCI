#!/usr/bin/env python3
"""
topography_view.py
地形图视图 - 支持多模态生物信号和纯元数据文件
修复版 - 修复 time_slider 属性不存在的问题
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
                             QPushButton, QLabel, QGroupBox, QGridLayout,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
                             QMessageBox, QApplication, QMainWindow, QTabWidget,
                             QSplitter, QSlider, QTextEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


class ModalityType(Enum):
    """模态类型枚举（与 data_io.py 保持一致）"""
    EEG = "EEG"
    EMG = "EMG"
    ECG = "ECG"
    GSR = "GSR"
    FNIRS = "FNIRS"
    ET = "ET"
    RESP = "RESP"
    OTHER = "OTHER"
    METADATA = "METADATA"  # 新增：纯元数据模态


class TopographyView(QMainWindow):
    """
    多模态地形图视图类
    支持 EEG、fNIRS、EMG、ECG、GSR、ET、RESP 等多种模态
    也支持纯元数据文件（如光极位置文件）
    """

    def __init__(self, data_dict: Dict[str, Any], modality: Optional[str] = None):
        super().__init__()
        self.data_dict = data_dict
        self.modality = modality
        self.current_data = None
        self.channels_positions = {}
        self.channel_values = {}
        self.current_time_point = 0
        self.is_3d = False
        self.is_metadata_only = False  # 标记是否为纯元数据文件

        print("=" * 60)
        print("TopographyView 初始化")
        print("=" * 60)

        # 从数据字典中提取信息
        self._extract_data_info()

        self.init_ui()

        # 强制更新显示
        print("强制更新显示...")
        self.update_display()

    def _extract_data_info(self):
        """从数据字典中提取信息"""
        self.meta = self.data_dict.get('meta', {})
        self.signal = self.data_dict.get('signal', {})
        self.metadata = self.data_dict.get('metadata', {})

        print(f"meta: {list(self.meta.keys())}")
        print(f"signal: {list(self.signal.keys())}")
        print(f"metadata: {list(self.metadata.keys())}")

        # 检查是否为纯元数据文件
        has_signal = bool(self.signal)
        has_metadata = bool(self.metadata)

        if not has_signal and has_metadata:
            self.is_metadata_only = True
            print("📋 检测到纯元数据文件")
            # 直接获取元数据位置
            self._get_metadata_positions()
            self.current_modality = "METADATA"
        else:
            # 获取所有可用的模态
            self.available_modalities = list(self.signal.keys())

            # 如果指定了模态，检查是否可用
            if self.modality and self.modality.upper() in self.available_modalities:
                self.current_modality = self.modality.upper()
            elif self.available_modalities:
                self.current_modality = self.available_modalities[0]
                self._extract_modality_data()
            else:
                self.current_modality = None
                QMessageBox.warning(self, "警告", "数据中没有找到可用的信息")

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("多模态地形图")
        self.setGeometry(100, 100, 1400, 900)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧控制面板
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel, 1)

        # 右侧图形显示区域
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        plot_widget = QWidget()
        plot_widget.setLayout(plot_layout)
        main_layout.addWidget(plot_widget, 3)

        # 设置分割器比例
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 3)

    def _create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 标题
        title = QLabel("地形图控制")
        title.setFont(QFont("微软雅黑", 12, QFont.Bold))
        layout.addWidget(title)

        # 如果是纯元数据文件，显示提示
        if self.is_metadata_only:
            info_label = QLabel("📋 纯元数据文件模式\n显示光极/传感器位置")
            info_label.setStyleSheet("color: blue; background-color: #e6f3ff; padding: 5px;")
            info_label.setWordWrap(True)
            layout.addWidget(info_label)

        # 模态选择（如果不是纯元数据）
        if not self.is_metadata_only and hasattr(self, 'available_modalities') and self.available_modalities:
            modality_group = QGroupBox("模态选择")
            modality_layout = QVBoxLayout()

            self.modality_combo = QComboBox()
            self.modality_combo.addItems(self.available_modalities)
            self.modality_combo.currentTextChanged.connect(self.on_modality_changed)
            modality_layout.addWidget(self.modality_combo)

            modality_group.setLayout(modality_layout)
            layout.addWidget(modality_group)

        # 时间点控制（仅在有信号数据时创建）
        if not self.is_metadata_only:
            self.time_group = QGroupBox("时间点")
            time_layout = QVBoxLayout()

            self.time_slider = QSlider(Qt.Horizontal)
            self.time_slider.setMinimum(0)
            self.time_slider.setMaximum(100)
            self.time_slider.valueChanged.connect(self.on_time_changed)
            time_layout.addWidget(self.time_slider)

            self.time_label = QLabel("时间点: 0")
            time_layout.addWidget(self.time_label)

            self.time_group.setLayout(time_layout)
            layout.addWidget(self.time_group)

        # 显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QGridLayout()

        self.show_labels = QCheckBox("显示标签")
        self.show_labels.setChecked(True)
        self.show_labels.stateChanged.connect(self.update_display)
        display_layout.addWidget(self.show_labels, 0, 0)

        self.show_grid = QCheckBox("显示网格")
        self.show_grid.setChecked(True)
        self.show_grid.stateChanged.connect(self.update_display)
        display_layout.addWidget(self.show_grid, 0, 1)

        # 插值选项（仅在有信号数据时启用）
        self.interpolate = QCheckBox("插值显示")
        self.interpolate.setChecked(True)
        self.interpolate.stateChanged.connect(self.update_display)
        if self.is_metadata_only:
            self.interpolate.setEnabled(False)
        display_layout.addWidget(self.interpolate, 1, 0)

        self.colorbar = QCheckBox("显示颜色条")
        self.colorbar.setChecked(True)
        self.colorbar.stateChanged.connect(self.update_display)
        if self.is_metadata_only:
            self.colorbar.setEnabled(False)
        display_layout.addWidget(self.colorbar, 1, 1)

        # 显示连接线（对光极位置特别有用）
        self.show_connections = QCheckBox("显示连接线")
        self.show_connections.setChecked(False)
        self.show_connections.stateChanged.connect(self.update_display)
        display_layout.addWidget(self.show_connections, 2, 0)

        # 区分光源和探测器颜色
        self.color_by_type = QCheckBox("按类型着色")
        self.color_by_type.setChecked(True)
        self.color_by_type.stateChanged.connect(self.update_display)
        display_layout.addWidget(self.color_by_type, 2, 1)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # 3D/2D切换
        view_group = QGroupBox("视图模式")
        view_layout = QVBoxLayout()

        self.view_2d = QCheckBox("2D视图")
        self.view_2d.setChecked(True)
        self.view_2d.toggled.connect(self.on_view_changed)
        view_layout.addWidget(self.view_2d)

        self.view_3d = QCheckBox("3D视图")
        self.view_3d.toggled.connect(self.on_view_changed)
        view_layout.addWidget(self.view_3d)

        view_group.setLayout(view_layout)
        layout.addWidget(view_group)

        # 颜色映射（仅在有信号数据时显示）
        if not self.is_metadata_only:
            cmap_group = QGroupBox("颜色映射")
            cmap_layout = QVBoxLayout()

            self.cmap_combo = QComboBox()
            self.cmap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma',
                                      'coolwarm', 'RdBu', 'jet', 'hot'])
            self.cmap_combo.currentTextChanged.connect(self.update_display)
            cmap_layout.addWidget(self.cmap_combo)

            cmap_group.setLayout(cmap_layout)
            layout.addWidget(cmap_group)

        # 信息显示区域（对元数据特别有用）
        if self.is_metadata_only:
            info_group = QGroupBox("元数据信息")
            info_layout = QVBoxLayout()

            self.info_text = QTextEdit()
            self.info_text.setReadOnly(True)
            self.info_text.setMaximumHeight(150)
            info_layout.addWidget(self.info_text)

            info_group.setLayout(info_layout)
            layout.addWidget(info_group)

            # 填充元数据信息
            self._update_metadata_info()

        # 测试按钮 - 强制刷新
        test_btn = QPushButton("强制刷新")
        test_btn.clicked.connect(self.force_redraw)
        layout.addWidget(test_btn)

        # 导出按钮
        export_btn = QPushButton("导出图像")
        export_btn.clicked.connect(self.export_figure)
        layout.addWidget(export_btn)

        # 添加弹簧
        layout.addStretch()

        return panel

    def force_redraw(self):
        """强制重绘"""
        print("强制重绘...")
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 绘制一些测试点
        x = [0, 1, 2, 3, 4]
        y = [0, 1, 4, 9, 16]
        ax.plot(x, y, 'ro-', linewidth=2, markersize=8)
        ax.set_title("测试图 - 如果看到这个，说明绘图正常")
        ax.grid(True)

        self.canvas.draw()

    def _update_metadata_info(self):
        """更新元数据信息显示"""
        if not hasattr(self, 'info_text'):
            return

        info_lines = []

        # 基本信息
        content_type = self.meta.get('content_type', 'unknown')
        info_lines.append(f"内容类型: {content_type}")

        # 统计光极信息
        if 'info' in self.metadata:
            items = self.metadata['info']
            sources = sum(1 for item in items if item.get('type') == 'source')
            detectors = sum(1 for item in items if item.get('type') == 'detector')
            others = len(items) - sources - detectors

            info_lines.append(f"光源数量: {sources}")
            info_lines.append(f"探测器数量: {detectors}")
            if others > 0:
                info_lines.append(f"其他: {others}")
            info_lines.append(f"总计: {len(items)} 个光极")

            # 坐标范围
            if items:
                x_vals = [float(item.get('x', 0)) for item in items if 'x' in item]
                y_vals = [float(item.get('y', 0)) for item in items if 'y' in item]
                z_vals = [float(item.get('z', 0)) for item in items if 'z' in item]

                if x_vals:
                    info_lines.append(f"X范围: [{min(x_vals):.4f}, {max(x_vals):.4f}]")
                if y_vals:
                    info_lines.append(f"Y范围: [{min(y_vals):.4f}, {max(y_vals):.4f}]")
                if z_vals and any(z != 0 for z in z_vals):
                    info_lines.append(f"Z范围: [{min(z_vals):.4f}, {max(z_vals):.4f}]")

        self.info_text.setText("\n".join(info_lines))

    def on_modality_changed(self, modality: str):
        """模态改变时的处理"""
        self.current_modality = modality

        if modality.startswith("METADATA_"):
            # 元数据模态
            self.is_metadata_only = True
            self._get_metadata_positions()
        else:
            # 信号模态
            self.is_metadata_only = False
            self._extract_modality_data()

        self.update_display()

    def _extract_modality_data(self):
        """提取当前模态的数据"""
        if not self.current_modality:
            return

        modality_data = self.signal.get(self.current_modality, {})
        self.current_data = modality_data.get('data', np.array([]))
        self.sampling_rate = modality_data.get('sampling_rate', 1000)
        self.channel_names = modality_data.get('channel_names', [])
        self.unit = modality_data.get('unit', 'unknown')

        # 更新滑块范围（确保 time_slider 存在）
        if hasattr(self, 'time_slider'):
            if self.current_data.ndim == 2 and self.current_data.size > 0:
                n_timepoints = self.current_data.shape[1]
                self.time_slider.setMaximum(n_timepoints - 1)
                self.time_group.setEnabled(True)
            else:
                self.time_group.setEnabled(False)

        # 获取通道位置
        self._get_channel_positions()

    def _get_metadata_positions(self):
        """从元数据获取位置信息"""
        print("获取元数据位置信息...")
        self.channels_positions = {}
        self.channel_names = []
        self.unit = 'position'
        self.item_types = {}

        if 'info' in self.metadata:
            items = self.metadata['info']
            print(f"找到 {len(items)} 个元数据项")

            for item in items:
                name = item.get('name', 'unknown')
                try:
                    x = float(item.get('x', 0))
                    y = float(item.get('y', 0))
                    z = float(item.get('z', 0))
                except (ValueError, TypeError) as e:
                    print(f"坐标转换错误: {e}")
                    continue

                self.channels_positions[name] = (x, y, z)
                self.channel_names.append(name)
                self.item_types[name] = item.get('type', 'unknown')

                print(f"  添加 {name}: ({x}, {y}, {z}) 类型: {self.item_types[name]}")

            print(f"总共有 {len(self.channels_positions)} 个位置点")
        else:
            print("警告: metadata 中没有 'info' 字段")

    def _get_channel_positions(self):
        """根据模态获取通道位置"""
        if self.current_modality == ModalityType.EEG.value:
            self._get_eeg_positions()
        elif self.current_modality == ModalityType.FNIRS.value:
            self._get_fnirs_positions()
        elif self.current_modality == ModalityType.EMG.value:
            self._get_emg_positions()
        elif self.current_modality == ModalityType.ECG.value:
            self._get_ecg_positions()
        elif self.current_modality == ModalityType.GSR.value:
            self._get_gsr_positions()
        elif self.current_modality == ModalityType.ET.value:
            self._get_et_positions()
        elif self.current_modality == ModalityType.RESP.value:
            self._get_resp_positions()
        else:
            self._get_default_positions()

    def _get_fnirs_positions(self):
        """获取 fNIRS 通道位置（从光极坐标计算）"""
        # 检查是否有光极位置信息
        if 'metadata' in self.data_dict and 'info' in self.data_dict['metadata']:
            optodes = self.data_dict['metadata']['info']

            # 分离光源和探测器
            sources = {}
            detectors = {}
            for opt in optodes:
                name = opt.get('name', '')
                opt_type = opt.get('type', '')
                x = float(opt.get('x', 0))
                y = float(opt.get('y', 0))
                z = float(opt.get('z', 0))

                if opt_type == 'source':
                    sources[name] = (x, y, z)
                elif opt_type == 'detector':
                    detectors[name] = (x, y, z)

            # 计算通道位置（光源和探测器的中点）
            # 这里需要根据实际的源-探测器配对规则
            # 暂时使用简单的配对规则：每个光源与最近的探测器配对
            for i, ch_name in enumerate(self.channel_names):
                if i < len(sources) and i < len(detectors):
                    s_name = list(sources.keys())[i % len(sources)]
                    d_name = list(detectors.keys())[i % len(detectors)]
                    s_pos = sources[s_name]
                    d_pos = detectors[d_name]

                    # 计算中点
                    mid_x = (s_pos[0] + d_pos[0]) / 2
                    mid_y = (s_pos[1] + d_pos[1]) / 2
                    self.channels_positions[ch_name] = (mid_x, mid_y)
                else:
                    # 如果没有足够的光极，使用圆形布局
                    angle = 2 * np.pi * i / len(self.channel_names)
                    radius = 0.8
                    self.channels_positions[ch_name] = (radius * np.cos(angle),
                                                        radius * np.sin(angle))
        else:
            # 没有光极位置信息，使用默认布局
            self._get_default_positions()

    def _get_eeg_positions(self):
        """获取 EEG 电极位置（标准 10-20 系统）"""
        # 标准 10-20 系统的 2D 投影坐标
        standard_positions = {
            'Fp1': (-0.5, 0.9), 'Fp2': (0.5, 0.9),
            'F7': (-0.9, 0.5), 'F3': (-0.4, 0.5), 'Fz': (0, 0.5), 'F4': (0.4, 0.5), 'F8': (0.9, 0.5),
            'T3': (-0.9, 0), 'C3': (-0.4, 0), 'Cz': (0, 0), 'C4': (0.4, 0), 'T4': (0.9, 0),
            'T5': (-0.9, -0.5), 'P3': (-0.4, -0.5), 'Pz': (0, -0.5), 'P4': (0.4, -0.5), 'T6': (0.9, -0.5),
            'O1': (-0.5, -0.9), 'O2': (0.5, -0.9)
        }

        self.channels_positions = {}
        for i, ch_name in enumerate(self.channel_names):
            # 尝试匹配标准名称
            matched = False
            for std_name, pos in standard_positions.items():
                if std_name in ch_name or ch_name in std_name:
                    self.channels_positions[ch_name] = pos
                    matched = True
                    break
            if not matched:
                # 如果没有匹配，使用圆形布局
                angle = 2 * np.pi * i / len(self.channel_names)
                radius = 0.8
                self.channels_positions[ch_name] = (radius * np.cos(angle),
                                                    radius * np.sin(angle))

    def _get_emg_positions(self):
        """获取 EMG 电极位置（肌肉分布）"""
        # EMG 电极通常放置在特定肌肉上
        muscle_positions = {
            'masseter': (-0.3, 0.8), 'temporalis': (0, 0.9),
            'sternocleidomastoid': (-0.2, 0.5), 'trapezius': (0, 0.3),
            'deltoid': (-0.5, 0.2), 'biceps': (-0.4, 0), 'triceps': (0.4, 0),
            'forearm': (-0.3, -0.2), 'thenar': (-0.2, -0.4),
            'quadriceps': (-0.1, -0.5), 'hamstring': (0.1, -0.6),
            'gastrocnemius': (-0.2, -0.8), 'soleus': (0.2, -0.9)
        }

        self.channels_positions = {}
        for i, ch_name in enumerate(self.channel_names):
            ch_lower = ch_name.lower()
            matched = False
            for muscle, pos in muscle_positions.items():
                if muscle in ch_lower:
                    self.channels_positions[ch_name] = pos
                    matched = True
                    break
            if not matched:
                angle = 2 * np.pi * i / len(self.channel_names)
                radius = 0.8
                self.channels_positions[ch_name] = (radius * np.cos(angle),
                                                    radius * np.sin(angle))

    def _get_ecg_positions(self):
        """获取 ECG 电极位置（心电导联）"""
        # 标准 ECG 导联位置
        ecg_positions = {
            'RA': (-0.3, 0.5), 'LA': (0.3, 0.5),
            'RL': (-0.2, -0.5), 'LL': (0.2, -0.5),
            'V1': (-0.1, 0.2), 'V2': (0, 0.2), 'V3': (0.1, 0.1),
            'V4': (0.2, 0), 'V5': (0.2, -0.1), 'V6': (0.2, -0.2)
        }

        self.channels_positions = {}
        for i, ch_name in enumerate(self.channel_names):
            ch_upper = ch_name.upper()
            matched = False
            for std_name, pos in ecg_positions.items():
                if std_name in ch_upper:
                    self.channels_positions[ch_name] = pos
                    matched = True
                    break
            if not matched:
                angle = 2 * np.pi * i / len(self.channel_names)
                radius = 0.6
                self.channels_positions[ch_name] = (radius * np.cos(angle),
                                                    radius * np.sin(angle))

    def _get_gsr_positions(self):
        """获取 GSR 电极位置（通常在手部）"""
        # GSR 电极通常放置在手指或手掌
        hand_positions = {
            'index': (-0.2, 0.1), 'middle': (0, 0.15), 'ring': (0.2, 0.1),
            'palm': (0, 0), 'wrist': (0, -0.2)
        }

        self.channels_positions = {}
        for i, ch_name in enumerate(self.channel_names):
            ch_lower = ch_name.lower()
            matched = False
            for part, pos in hand_positions.items():
                if part in ch_lower:
                    self.channels_positions[ch_name] = pos
                    matched = True
                    break
            if not matched:
                x = (i - len(self.channel_names) / 2) * 0.3
                self.channels_positions[ch_name] = (x, 0)

    def _get_et_positions(self):
        """获取眼动追踪位置（视野分布）"""
        n_channels = len(self.channel_names)
        grid_size = int(np.ceil(np.sqrt(n_channels)))

        self.channels_positions = {}
        for i, ch_name in enumerate(self.channel_names):
            row = i // grid_size
            col = i % grid_size
            x = (col - grid_size / 2) * (2.0 / grid_size)
            y = (grid_size / 2 - row) * (2.0 / grid_size)
            self.channels_positions[ch_name] = (x, y)

    def _get_resp_positions(self):
        """获取呼吸传感器位置"""
        resp_positions = {
            'chest': (0, 0.3), 'abdomen': (0, -0.1),
            'nasal': (0, 0.6), 'oral': (0, 0.5)
        }

        self.channels_positions = {}
        for i, ch_name in enumerate(self.channel_names):
            ch_lower = ch_name.lower()
            matched = False
            for pos_name, pos in resp_positions.items():
                if pos_name in ch_lower:
                    self.channels_positions[ch_name] = pos
                    matched = True
                    break
            if not matched:
                y = 0.5 - i * (1.0 / len(self.channel_names))
                self.channels_positions[ch_name] = (0, y)

    def _get_default_positions(self):
        """获取默认的通道位置（圆形布局）"""
        self.channels_positions = {}
        n_channels = len(self.channel_names)
        for i, ch_name in enumerate(self.channel_names):
            angle = 2 * np.pi * i / n_channels
            radius = 0.8
            self.channels_positions[ch_name] = (radius * np.cos(angle),
                                                radius * np.sin(angle))

    def _get_current_values(self) -> Dict[str, float]:
        """获取当前要显示的通道值"""
        values = {}

        if self.is_metadata_only:
            # 元数据模式：没有数值，只显示位置
            return values

        if self.current_data is None or self.current_data.size == 0:
            return values

        # 显示所有通道的值
        if self.current_data.ndim == 2:
            # 时间序列数据
            for i, ch_name in enumerate(self.channel_names):
                if i < self.current_data.shape[0]:
                    values[ch_name] = self.current_data[i, self.current_time_point]
        elif self.current_data.ndim == 1:
            # 单值数据
            for i, ch_name in enumerate(self.channel_names):
                if i < len(self.current_data):
                    values[ch_name] = self.current_data[i]

        return values

    def on_time_changed(self, value: int):
        """时间点改变时的处理"""
        self.current_time_point = value
        if hasattr(self, 'time_label'):
            self.time_label.setText(f"时间点: {value}")
        self.update_display()

    def on_view_changed(self):
        """视图模式改变时的处理"""
        if self.view_2d.isChecked():
            self.is_3d = False
        elif self.view_3d.isChecked():
            self.is_3d = True
        self.update_display()

    def update_display(self):
        """更新显示"""
        print("\n" + "=" * 40)
        print("update_display 被调用")
        print("=" * 40)

        # 调试信息
        print(f"current_modality: {self.current_modality}")
        print(f"is_metadata_only: {self.is_metadata_only}")
        print(f"channels_positions 数量: {len(self.channels_positions)}")

        if not self.current_modality and not self.is_metadata_only:
            print("警告: current_modality 为空且不是元数据模式")
            self._show_no_data_message()
            return

        if not self.channels_positions:
            print("警告: channels_positions 为空")
            self._show_no_data_message()
            return

        # 获取当前值（如果是元数据模式，值为空）
        values = self._get_current_values()
        print(f"values 数量: {len(values)}")

        # 清除图形
        self.figure.clear()

        if self.is_3d:
            ax = self.figure.add_subplot(111, projection='3d')
            self._draw_3d_topography(ax, values)
        else:
            ax = self.figure.add_subplot(111)
            self._draw_2d_topography(ax, values)

        # 设置标题
        if self.is_metadata_only:
            content_type = self.meta.get('content_type', 'metadata')
            title = f"{content_type} 位置分布图"
        else:
            title = f"{self.current_modality} 地形图"
            if hasattr(self, 'current_data') and self.current_data is not None and self.current_data.ndim == 2:
                if hasattr(self, 'sampling_rate') and self.sampling_rate:
                    time_sec = self.current_time_point / self.sampling_rate
                else:
                    time_sec = self.current_time_point
                title += f" (t={time_sec:.2f}s)"

        ax.set_title(title, fontsize=14, fontweight='bold')

        self.figure.tight_layout()
        self.canvas.draw()
        print("绘图完成")

    def _show_no_data_message(self):
        """显示无数据消息"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "无位置数据\n请检查数据文件",
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("地形图", fontsize=14, fontweight='bold')
        self.canvas.draw()

    def _draw_2d_topography(self, ax, values: Dict[str, float]):
        """绘制真正的2D地形图（连续色块）"""
        print("绘制真正的地形图...")

        if not self.channels_positions:
            ax.text(0.5, 0.5, "无位置数据", ha='center', va='center', transform=ax.transAxes)
            return

        # 提取有数值的电极点
        points = []
        point_values = []
        labels = []

        for ch_name, pos in self.channels_positions.items():
            if ch_name in values:
                if len(pos) == 3:
                    points.append([pos[0], pos[1]])  # 只取x,y
                else:
                    points.append([pos[0], pos[1]])
                point_values.append(values[ch_name])
                labels.append(ch_name)

        print(f"有效电极点数量: {len(points)}")

        if len(points) < 3:
            # 点太少，无法插值，只能画散点
            ax.scatter([p[0] for p in points], [p[1] for p in points],
                       c=point_values, s=100, cmap='jet', edgecolors='black')
            ax.set_title("电极点太少，无法生成地形图")
            return

        points = np.array(points)
        point_values = np.array(point_values)

        # 创建网格（覆盖所有电极点范围）
        margin = 0.1
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        # 增加边距
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= margin * x_range
        x_max += margin * x_range
        y_min -= margin * y_range
        y_max += margin * y_range

        # 创建高分辨率网格
        grid_x, grid_y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

        # 插值
        from scipy.interpolate import griddata
        grid_z = griddata(points, point_values, (grid_x, grid_y), method='cubic')

        # 如果cubic失败，尝试linear
        if np.isnan(grid_z).all():
            grid_z = griddata(points, point_values, (grid_x, grid_y), method='linear')

        # 绘制地形图（连续色块）
        cmap = plt.get_cmap(self.cmap_combo.currentText() if hasattr(self, 'cmap_combo') else 'jet')
        im = ax.imshow(grid_z.T, extent=[x_min, x_max, y_min, y_max],
                       origin='lower', cmap=cmap, alpha=0.8, aspect='auto')

        # 叠加电极点位置（用小圆点标记）
        ax.scatter(points[:, 0], points[:, 1], c='black', s=20, zorder=5)

        # 显示电极标签
        if hasattr(self, 'show_labels') and self.show_labels.isChecked():
            for i, label in enumerate(labels):
                ax.annotate(label, (points[i, 0], points[i, 1]),
                            xytext=(3, 3), textcoords='offset points',
                            fontsize=8, zorder=6)

        # 显示颜色条
        if hasattr(self, 'colorbar') and self.colorbar.isChecked():
            unit = getattr(self, 'unit', 'value')
            plt.colorbar(im, ax=ax, label=f'值 ({unit})')

        # 显示网格
        if hasattr(self, 'show_grid') and self.show_grid.isChecked():
            ax.grid(True, alpha=0.3, linestyle='--', zorder=1)

        ax.set_aspect('equal')
        ax.set_xlabel('X 位置')
        ax.set_ylabel('Y 位置')

    def _draw_3d_topography(self, ax, values: Dict[str, float]):
        """绘制真正的3D地形图（连续曲面）"""
        print("绘制真正的3D地形图...")

        if not self.channels_positions:
            ax.text(0.5, 0.5, 0.5, "无位置数据", ha='center', va='center')
            return

        # 提取有数值的电极点
        points = []
        point_values = []
        labels = []

        for ch_name, pos in self.channels_positions.items():
            if ch_name in values:
                if len(pos) == 3:
                    points.append([pos[0], pos[1], pos[2]])  # 取x,y,z
                else:
                    points.append([pos[0], pos[1], 0])  # 2D坐标的z设为0
                point_values.append(values[ch_name])
                labels.append(ch_name)

        print(f"有效电极点数量: {len(points)}")

        if len(points) < 4:
            # 点太少，无法插值，只能画散点
            for i in range(len(points)):
                ax.scatter([points[i][0]], [points[i][1]], [points[i][2]],
                           c=[point_values[i]], cmap='jet', s=100, edgecolors='black', vmin=min(point_values),
                           vmax=max(point_values))
            ax.set_title("电极点太少，无法生成3D地形图")
            return

        points = np.array(points)
        point_values = np.array(point_values)

        # 创建网格（覆盖所有电极点范围）
        margin = 0.1
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        # 增加边距
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range == 0:
            x_range = 1
        if y_range == 0:
            y_range = 1

        x_min -= margin * x_range
        x_max += margin * x_range
        y_min -= margin * y_range
        y_max += margin * y_range

        # 创建高分辨率网格
        grid_x, grid_y = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]

        # 插值（只使用x,y坐标，z值由插值决定）
        from scipy.interpolate import griddata

        # 插值得到每个网格点的值（高度）
        grid_z = griddata(points[:, :2], point_values, (grid_x, grid_y), method='cubic')

        # 如果cubic失败，尝试linear
        if grid_z is None or np.isnan(grid_z).all():
            grid_z = griddata(points[:, :2], point_values, (grid_x, grid_y), method='linear')

        # 方法1：简单方法 - 只用一个颜色映射，不单独设置facecolors
        cmap = plt.get_cmap(self.cmap_combo.currentText() if hasattr(self, 'cmap_combo') else 'jet')

        # 创建曲面图 - 简化版本，避免facecolors问题
        surf = ax.plot_surface(grid_x, grid_y, grid_z,
                               cmap=cmap,
                               alpha=0.9,
                               linewidth=0,
                               antialiased=True,
                               vmin=point_values.min() if len(point_values) > 0 else None,
                               vmax=point_values.max() if len(point_values) > 0 else None)

        # 叠加电极点位置（用小球标记）
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2] + 0.001,
                             c=point_values,
                             cmap=cmap,
                             s=50,
                             edgecolors='black',
                             linewidth=1,
                             zorder=10,
                             vmin=point_values.min() if len(point_values) > 0 else None,
                             vmax=point_values.max() if len(point_values) > 0 else None)

        # 显示电极标签
        if hasattr(self, 'show_labels') and self.show_labels.isChecked():
            for i, label in enumerate(labels):
                ax.text(points[i, 0], points[i, 1], points[i, 2] + 0.002,
                        f' {label}', fontsize=8, zorder=11)

        # 设置坐标轴
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel(f'值 ({getattr(self, "unit", "value")})')

        # 添加颜色条
        if hasattr(self, 'colorbar') and self.colorbar.isChecked():
            plt.colorbar(surf, ax=ax, label=f'值 ({getattr(self, "unit", "value")})', shrink=0.5)

    def export_figure(self):
        """导出图形"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )

        if file_path:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "成功", f"图像已保存到:\n{file_path}")


def show_topography(data_dict: Dict[str, Any], modality: Optional[str] = None):
    """显示地形图视图"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = TopographyView(data_dict, modality)
    window.show()
    return window


if __name__ == "__main__":
    # 测试代码
    app = QApplication(sys.argv)

    # 创建测试数据（包含光极位置）
    test_data = {
        "meta": {
            "subject_id": "test",
            "content_type": "optode_positions",
            "modality": []
        },
        "signal": {},
        "metadata": {
            "info": [
                {"name": "S1", "type": "source", "x": -0.00247, "y": 0.00247, "z": 0},
                {"name": "S2", "type": "source", "x": -0.00247, "y": -0.00247, "z": 0},
                {"name": "S3", "type": "source", "x": 0.00247, "y": 0.00247, "z": 0},
                {"name": "S4", "type": "source", "x": 0.00247, "y": -0.00247, "z": 0},
                {"name": "S5", "type": "source", "x": -0.00247, "y": -0.00597, "z": 0},
                {"name": "S6", "type": "source", "x": -0.00247, "y": -0.01092, "z": 0},
                {"name": "S7", "type": "source", "x": 0.00247, "y": -0.00597, "z": 0},
                {"name": "S8", "type": "source", "x": 0.00247, "y": -0.01092, "z": 0},
                {"name": "D1", "type": "detector", "x": 0, "y": 0, "z": 0},
                {"name": "D2", "type": "detector", "x": 0, "y": -0.00845, "z": 0}
            ]
        }
    }

    window = TopographyView(test_data)
    window.show()

    sys.exit(app.exec_())