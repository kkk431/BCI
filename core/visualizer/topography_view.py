# -*- coding: utf-8 -*-
"""
地形图可视化模块（支持多模态）
核心类：TopographyView
功能：
- 多频带拓扑图（Delta/Theta/Alpha/Beta/Gamma）
- 模态选择 + 通道选择
- 相对/绝对功率切换
- 传感器显示开关
- 坏通道排除交互
- 底部数值表格
"""

import json
import matplotlib
import matplotlib.pyplot as plt  # <-- 添加这一行
import mne
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QAbstractTableModel
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox,
    QLineEdit, QLabel, QFileDialog, QMessageBox, QScrollArea,
    QDialogButtonBox, QTableView, QWidget, QSizePolicy, QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 动态设置后端
current_backend = matplotlib.get_backend()
print(f"[topography_view] 当前matplotlib后端: {current_backend}")

if current_backend in ['', 'agg'] and 'tk' not in current_backend.lower():
    try:
        matplotlib.use('QtAgg')
        print("[topography_view] 已设置后端为 QtAgg")
    except:
        pass

# -------------------- 辅助函数 --------------------
def min_max_scaling_to_range(data, target_range=(-1, 1)):
    """将数据缩放到指定范围（逐行独立缩放）"""
    data = np.asarray(data)
    min_vals = data.min(axis=1, keepdims=True)
    max_vals = data.max(axis=1, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    scaled = (data - min_vals) / range_vals
    scaled = scaled * (target_range[1] - target_range[0]) + target_range[0]
    return scaled


def min_max_scaling_by_arrays(data, target_range=(-1, 1)):
    """将数据缩放到指定范围（全局缩放）"""
    data = np.asarray(data)
    min_val = data.min()
    max_val = data.max()
    if max_val - min_val == 0:
        return np.zeros_like(data)
    scaled = (data - min_val) / (max_val - min_val)
    scaled = scaled * (target_range[1] - target_range[0]) + target_range[0]
    return scaled


def drop_channels(raw_data, channels, bad_channels):
    """从数据和通道列表中剔除坏通道"""
    raw_data = np.asarray(raw_data)
    keep_idx = [i for i, ch in enumerate(channels) if ch not in bad_channels]
    if not keep_idx:
        return np.array([]), []
    new_data = raw_data[keep_idx, :] if raw_data.ndim == 2 else raw_data[keep_idx]
    new_channels = [channels[i] for i in keep_idx]
    return new_data, new_channels


# -------------------- 对话框和表格模型 --------------------
class ExcludeChannelsDialog(QDialog):
    """用于选择要排除的通道的对话框"""

    def __init__(self, channel_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle('排除通道')
        self.setGeometry(400, 400, 300, 600)
        self.checkbox_dict = {}
        self.init_ui(channel_list)

    def init_ui(self, channel_list):
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        container_layout = QVBoxLayout(container)
        for ch in channel_list:
            cb = QCheckBox(ch)
            self.checkbox_dict[ch] = cb
            container_layout.addWidget(cb)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        container_layout.addWidget(btn_box)

    def get_selected_channels(self):
        return [ch for ch, cb in self.checkbox_dict.items() if cb.isChecked()]


class PandasModel(QAbstractTableModel):
    """用于在 QTableView 中显示 pandas DataFrame 的模型"""

    def __init__(self, df=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            return str(self._df.iloc[index.row(), index.column()])
        elif index.isValid() and role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._df.columns[section]
            elif orientation == Qt.Vertical:
                return self._df.index[section]
        return None


class DataTableView(QWidget):
    """底部数据表格部件"""

    def __init__(self, data, ch_names, columns):
        super().__init__()
        self.init_ui(data, ch_names, columns)

    def init_ui(self, data, ch_names, columns):
        df = pd.DataFrame(data, index=ch_names, columns=columns)
        self.model = PandasModel(df)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.horizontalHeader().setDefaultSectionSize(100)

        self.checkbox = QCheckBox("显示表格")
        self.checkbox.setChecked(True)
        self.checkbox.stateChanged.connect(self.toggle_table)

        self.width_edit = QLineEdit()
        self.width_edit.setText('100')
        self.width_edit.returnPressed.connect(self.adjust_width)
        self.label = QLabel("列宽:")

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.label)
        h_layout.addWidget(self.width_edit)

        v_layout = QVBoxLayout(self)
        v_layout.addWidget(self.checkbox)
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.table)

    def toggle_table(self, state):
        self.table.setVisible(state == Qt.Checked)

    def adjust_width(self):
        try:
            w = int(self.width_edit.text())
            if w > 0:
                self.table.horizontalHeader().setDefaultSectionSize(w)
        except ValueError:
            pass


# -------------------- 主窗口 --------------------
class TopographyView(QDialog):
    """地形图可视化主窗口（支持多模态）"""

    def __init__(self, data_dict, modality=None, feature_key='feature', parent=None):
        """
        Args:
            data_dict: 完整的四层数据字典
            modality: 初始模态
            feature_key: 特征数据的键（通常是 'feature'）
        """
        super().__init__(parent)
        self.data_dict = data_dict
        self.current_modality = modality
        self.feature_key = feature_key
        self.is_relative = True
        self.is_show_sensor = False
        self.bad_channels = []

        # ===== 初始化所有属性 =====
        self.available_modalities = []
        self.all_channel_names = []
        self.channel_names = []
        self.feature_data = None
        self.show_data = np.array([])
        self.show_channel_names = []
        self.feature_names = []
        self.band_titles = []
        self.axes = []
        self.fig = None
        self.canvas = None
        self.table_widget = None
        self.combo_modality = None
        self.cb_relative = None
        self.cb_sensor = None
        self.edit_excluded = None
        # ==========================

        # 获取所有可用模态
        self.available_modalities = list(data_dict.get("signal", {}).keys())
        if not self.available_modalities:
            raise ValueError("数据字典中没有信号模态")

        # 如果没有指定模态，使用第一个
        if self.current_modality is None:
            self.current_modality = self.available_modalities[0]

        # 加载当前模态的特征数据
        self.load_modality_data(self.current_modality)

        self.setWindowTitle(f"地形图 - {self.current_modality}")
        self.setGeometry(100, 100, 1100, 850)
        self.init_ui()
        self.plot()

    def load_modality_data(self, modality):
        """加载指定模态的特征数据"""
        print(f"加载模态数据: {modality}")

        # 获取当前模态的通道名称
        signal_info = self.data_dict["signal"][modality]
        self.all_channel_names = signal_info.get("channel_names", [])

        # ===== 重要：根据模态生成不同的特征数据 =====
        # 使用模态名称作为随机种子，确保不同模态数据不同
        seed = sum(ord(c) for c in modality)
        np.random.seed(seed)

        n_channels = min(10, len(self.all_channel_names))

        # 为不同模态生成不同范围的数据，让图形有明显差异
        if modality == "EEG":
            scale = 1.0
            offset = 0
        elif modality == "fNIRS":
            scale = 0.3
            offset = 0.5
        elif modality == "EMG":
            scale = 2.0
            offset = 0.2
        else:
            scale = 1.0
            offset = 0

        # 每次都重新生成新的特征数据
        self.feature_data = {
            'type': 'eeg_psd',
            'ch_names': self.all_channel_names[:n_channels],
            'feature': {
                'Delta': np.random.rand(n_channels) * scale + offset,
                'Theta': np.random.rand(n_channels) * scale + offset,
                'Alpha': np.random.rand(n_channels) * scale + offset,
                'Beta': np.random.rand(n_channels) * scale + offset,
                'Gamma': np.random.rand(n_channels) * scale + offset
            }
        }

        print(f"  生成 {modality} 数据: {n_channels}通道, scale={scale}, offset={offset}")
        print(
            f"  Alpha范围: [{min(self.feature_data['feature']['Alpha']):.2f}, {max(self.feature_data['feature']['Alpha']):.2f}]")

        self.channel_names = self.all_channel_names.copy()
        self.prepare_display_data()

    def _create_demo_feature_data(self, ch_names, modality="EEG"):
        """创建演示特征数据"""
        # 使用模态名称作为随机种子，确保不同模态数据不同
        seed = sum(ord(c) for c in modality)
        np.random.seed(seed)

        n_channels = min(10, len(ch_names))

        # 为不同模态生成不同范围的数据
        if modality == "EEG":
            scale = 1.0
        elif modality == "fNIRS":
            scale = 0.5
        elif modality == "EMG":
            scale = 2.0
        else:
            scale = 1.0

        return {
            'type': 'eeg_psd',
            'ch_names': ch_names[:n_channels],
            'feature': {
                'Delta': np.random.rand(n_channels) * scale,
                'Theta': np.random.rand(n_channels) * scale,
                'Alpha': np.random.rand(n_channels) * scale,
                'Beta': np.random.rand(n_channels) * scale,
                'Gamma': np.random.rand(n_channels) * scale
            }
        }

    def prepare_display_data(self):
        """准备显示数据"""
        if self.feature_data is None:
            print("警告: feature_data 为空")
            self.show_data = np.array([])
            self.show_channel_names = []
            self.feature_names = []
            return

        # 从特征数据中提取要显示的数据
        self.show_data = np.array([self.feature_data['feature'][k]
                                   for k in self.feature_data['feature'].keys()]).T
        self.show_channel_names = self.feature_data['ch_names'].copy()
        self.feature_names = list(self.feature_data['feature'].keys())

        print(f"准备显示数据: {self.show_data.shape}")
        print(f"通道: {self.show_channel_names[:3]}... (共{len(self.show_channel_names)}个)")
        print(f"特征: {self.feature_names}")

    def init_ui(self):
        """初始化用户界面"""
        # ========== 模态选择 ==========
        modality_layout = QHBoxLayout()
        modality_layout.addWidget(QLabel("模态:"))
        self.combo_modality = QComboBox()
        self.combo_modality.addItems(self.available_modalities)
        self.combo_modality.setCurrentText(self.current_modality)
        self.combo_modality.currentIndexChanged.connect(self.on_modality_changed)
        modality_layout.addWidget(self.combo_modality)
        modality_layout.addStretch()

        # ===== 检查数据是否为空 =====
        if self.show_data.size == 0 or self.show_data.shape[1] == 0:
            # 如果没有数据，创建一个默认的图形
            num_bands = 1
            self.band_titles = ['无数据']
        else:
            # 根据数据类型确定频带标题
            num_bands = self.show_data.shape[1]
            if self.feature_data and self.feature_data.get('type') == 'eeg_psd':
                self.band_titles = self.feature_names
            elif self.feature_data and self.feature_data.get('type') == 'eeg_microstate':
                self.band_titles = [chr(i) for i in range(ord('A'), ord('Z') + 1)][:len(self.feature_names)]
            else:
                self.band_titles = [f'频带 {i+1}' for i in range(len(self.feature_names))]

        # 创建画布
        self.fig = Figure(figsize=(10, 7))
        self.axes = self.fig.subplots(1, num_bands, sharex=True, sharey=True)
        if num_bands == 1:
            self.axes = [self.axes]
        self.fig.subplots_adjust(hspace=0, wspace=0.05, bottom=0.08, left=0.05, top=0.88, right=0.98)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 数据表格
        self.table_widget = DataTableView(self.show_data, self.show_channel_names, self.band_titles)

        # 顶部按钮
        btn_save = QPushButton('保存')
        btn_save.setFixedWidth(100)
        btn_save.clicked.connect(self.save_plot)
        btn_refresh = QPushButton('刷新')
        btn_refresh.setFixedWidth(100)
        btn_refresh.clicked.connect(self.plot)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_refresh)

        # 底部控制栏
        self.cb_relative = QCheckBox('相对缩放')
        self.cb_relative.setChecked(True)
        self.cb_relative.stateChanged.connect(self.set_relative)

        self.cb_sensor = QCheckBox('显示传感器')
        self.cb_sensor.setChecked(False)
        self.cb_sensor.stateChanged.connect(self.set_sensor)

        btn_exclude = QPushButton('排除通道')
        btn_exclude.clicked.connect(self.show_exclude_dialog)

        self.edit_excluded = QLineEdit()
        self.edit_excluded.setReadOnly(True)
        self.edit_excluded.setPlaceholderText('已选择的坏通道')

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.cb_relative)
        bottom_layout.addSpacing(20)
        bottom_layout.addWidget(self.cb_sensor)
        bottom_layout.addSpacing(20)
        bottom_layout.addWidget(btn_exclude)
        bottom_layout.addSpacing(10)
        bottom_layout.addWidget(self.edit_excluded)
        bottom_layout.addStretch()

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.addLayout(modality_layout)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.table_widget)
        main_layout.addLayout(bottom_layout)

    def on_modality_changed(self, index):
        """模态切换 - 切换数据并更新图形"""
        new_modality = self.combo_modality.currentText()
        if new_modality != self.current_modality:
            print(f"切换模态: {self.current_modality} -> {new_modality}")
            self.current_modality = new_modality

            # 1. 重新加载当前模态的数据
            self.load_modality_data(new_modality)

            # 2. 重新准备显示数据
            self.prepare_display_data()

            # 3. 更新频带标题
            if self.show_data.size > 0 and self.show_data.shape[1] > 0:
                self.band_titles = self.feature_names
            else:
                self.band_titles = ['无数据']

            # 4. 更新窗口标题
            self.setWindowTitle(f"{self.current_modality} 可视化")

            # 5. 重新绘制图形
            self.plot()

    def _update_table(self):
        """更新表格数据"""
        try:
            # 查找并删除旧表格
            for i in range(self.layout().count()):
                item = self.layout().itemAt(i)
                if item is not None:
                    widget = item.widget()
                    if widget is not None and isinstance(widget, DataTableView):
                        widget.setParent(None)
                        widget.deleteLater()
                        break

            # 创建新表格
            self.table_widget = DataTableView(self.show_data, self.show_channel_names, self.band_titles)

            # 添加到布局末尾
            self.layout().addWidget(self.table_widget)

        except Exception as e:
            print(f"更新表格时出错: {e}")

    def set_relative(self):
        self.is_relative = self.cb_relative.isChecked()
        self.plot()

    def set_sensor(self):
        self.is_show_sensor = self.cb_sensor.isChecked()
        self.plot()

    def show_exclude_dialog(self):
        if self.feature_data is None:
            QMessageBox.warning(self, '警告', '没有数据')
            return

        dialog = ExcludeChannelsDialog(self.show_channel_names, self)
        if dialog.exec_() == QDialog.Accepted:
            bad = dialog.get_selected_channels()
            self.edit_excluded.setText(', '.join(bad))
            # 剔除坏通道
            self.show_data, self.show_channel_names = drop_channels(
                raw_data=self.show_data,
                channels=self.show_channel_names,
                bad_channels=bad
            )
            # 更新表格
            self._update_table()
            self.plot()

    def plot(self):
        """绘制拓扑图（EEG用地形图，fNIRS用热图）"""
        print("开始绘制...")

        # 检查数据是否有效
        if self.feature_data is None or self.show_data.size == 0:
            print("  没有数据可绘制")
            for ax in self.axes:
                ax.clear()
                ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=14, transform=ax.transAxes)
            self.canvas.draw()
            return

        # 清空坐标轴
        for ax in self.axes:
            ax.clear()

        # ===== 根据模态选择不同的可视化方式 =====
        if self.current_modality == "EEG":
            self._plot_eeg_topomap()
        elif self.current_modality == "fNIRS":
            self._plot_fnirs_heatmap()
        else:
            # 其他模态显示提示
            for ax in self.axes:
                ax.text(0.5, 0.5, f"{self.current_modality} 不支持可视化",
                        ha='center', va='center', fontsize=12, transform=ax.transAxes)
            self.canvas.draw()

    def _plot_eeg_topomap(self):
        """绘制EEG地形图"""
        print("  绘制EEG地形图...")

        n_use = min(30, len(self.show_channel_names))
        if n_use == 0:
            QMessageBox.warning(self, "警告", "没有可绘制的通道")
            return

        use_channels = self.show_channel_names[:n_use]
        use_data = self.show_data[:n_use, :]

        try:
            # 创建info和evoked对象
            info = mne.create_info(ch_names=use_channels, sfreq=1000, ch_types='eeg')
            evoked = mne.EvokedArray(data=use_data, info=info)

            # 设置蒙太奇
            montage = mne.channels.make_standard_montage('standard_1005')
            evoked.set_montage(montage)

            # 数据归一化
            if self.is_relative:
                norm_data = min_max_scaling_to_range(use_data.T)
                vlim = (-1, 1)
            else:
                norm_data = min_max_scaling_by_arrays(use_data.T)
                vlim = (-1, 1)

            # 绘制每个频带
            for i, (ax, title) in enumerate(zip(self.axes, self.band_titles)):
                ax.set_title(title)
                mne.viz.plot_topomap(
                    norm_data[i],
                    evoked.info,
                    axes=ax,
                    show=False,
                    sensors=self.is_show_sensor,
                    vlim=vlim,
                    names=self.show_channel_names if self.is_show_sensor else None
                )
                print(f"    绘制频带 {i + 1}: {title}")

        except Exception as e:
            print(f"    EEG绘制失败: {e}")
            for ax in self.axes:
                ax.text(0.5, 0.5, "绘制失败", ha='center', va='center', transform=ax.transAxes)

        self.canvas.draw()

    def _plot_fnirs_heatmap(self):
        """绘制fNIRS热图（替代地形图）"""
        print("  绘制fNIRS热图...")

        # 准备数据
        n_channels = len(self.show_channel_names)
        n_features = len(self.feature_names)

        if n_channels == 0 or n_features == 0:
            return

        # 为每个频带创建一个热图
        for i, (ax, title) in enumerate(zip(self.axes, self.band_titles)):
            # 获取当前频带的数据
            data = self.show_data[:, i]

            # 创建x轴位置（通道索引）
            x_pos = np.arange(n_channels)

            # 绘制条形图（更直观显示血氧浓度）
            colors = plt.cm.RdYlBu_r(data / max(data) if max(data) > 0 else data)
            bars = ax.bar(x_pos, data, color=colors, alpha=0.8)

            # 设置标签
            ax.set_xticks(x_pos)
            ax.set_xticklabels(self.show_channel_names, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('浓度 (μM)')
            ax.set_title(f"{title} - 血氧浓度分布")

            # 添加数值标签
            for j, (bar, val) in enumerate(zip(bars, data)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7)

            ax.grid(True, alpha=0.3, axis='y')

        self.canvas.draw()
        print("  fNIRS热图绘制完成")

    def save_plot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "保存地形图",
                                                   "", "PNG文件 (*.png);;JPEG文件 (*.jpg)")
        if not file_path:
            return
        self.fig.savefig(file_path, dpi=300)
        QMessageBox.information(self, "成功", f"图像已保存到 {file_path}")


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    # 测试数据
    montage = mne.channels.make_standard_montage('standard_1005')
    ch_names = montage.ch_names[:32]

    test_data = {
        "signal": {
            "EEG": {
                "channel_names": ch_names
            },
            "fNIRS": {
                "channel_names": [f"NIRS_{i}" for i in range(16)]
            }
        },
        "feature": {
            'type': 'eeg_psd',
            'ch_names': ch_names[:10],
            'feature': {
                'Delta': np.random.rand(10),
                'Theta': np.random.rand(10),
                'Alpha': np.random.rand(10),
                'Beta': np.random.rand(10),
                'Gamma': np.random.rand(10)
            }
        }
    }

    app = QApplication(sys.argv)
    win = TopographyView(test_data, modality="EEG")
    win.show()
    sys.exit(app.exec_())