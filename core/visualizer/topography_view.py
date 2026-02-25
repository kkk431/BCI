# -*- coding: utf-8 -*-
"""
地形图可视化模块（对应 viewer_topomap.py）
核心类：TopographyView
功能：
- 多频带拓扑图（Delta/Theta/Alpha/Beta/Gamma）
- 相对/绝对功率切换（通过归一化实现[-1,1]范围）
- 传感器显示开关
- 坏通道排除交互
- 底部数值表格（支持列宽调节）
- 保存拓扑图为图片
"""

import json
import matplotlib
import mne
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QAbstractTableModel
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox,
    QLineEdit, QLabel, QFileDialog, QMessageBox, QScrollArea,
    QDialogButtonBox, QTableView, QWidget, QSizePolicy
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

matplotlib.use('QtAgg')


# -------------------- 辅助函数（替代 BrainFusion.utils.normalize）--------------------
def min_max_scaling_to_range(data, target_range=(-1, 1)):
    """将数据缩放到指定范围（逐行独立缩放）"""
    data = np.asarray(data)
    min_vals = data.min(axis=1, keepdims=True)
    max_vals = data.max(axis=1, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # 避免除零
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


# -------------------- 辅助函数（替代 BrainFusion.utils.channels.drop_channels）--------------------
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
        self.setWindowTitle('Exclude Channels')
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

        self.checkbox = QCheckBox("Show Table")
        self.checkbox.setChecked(True)
        self.checkbox.stateChanged.connect(self.toggle_table)

        self.width_edit = QLineEdit()
        self.width_edit.setText('100')
        self.width_edit.returnPressed.connect(self.adjust_width)
        self.label = QLabel("Column Width:")

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
    """地形图可视化主窗口"""

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.is_relative = True
        self.is_show_sensor = False
        self.setWindowTitle("Topography Viewer")
        self.setGeometry(100, 100, 1000, 800)

        # 准备显示数据
        self.show_data = np.array([self.data['feature'][k] for k in self.data['feature'].keys()]).T
        self.show_channel_names = self.data['ch_names'].copy()
        self.init_ui()
        self.plot()

    def init_ui(self):
        # 根据数据类型确定频带标题
        if self.data['type'] == 'eeg_psd':
            self.band_titles = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        elif self.data['type'] == 'eeg_microstate':
            self.band_titles = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        else:
            self.band_titles = [f'Band {i+1}' for i in range(self.show_data.shape[1])]

        # 创建画布
        num_bands = self.show_data.shape[1]
        self.fig = Figure(figsize=(8, 6))
        self.axes = self.fig.subplots(1, num_bands, sharex=True, sharey=True)
        if num_bands == 1:
            self.axes = [self.axes]
        self.fig.subplots_adjust(hspace=0, wspace=0.05, bottom=0.08, left=0.05, top=0.88, right=0.98)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 数据表格
        self.table_widget = DataTableView(self.show_data, self.show_channel_names, self.band_titles)

        # 顶部按钮
        btn_save = QPushButton('Save')
        btn_save.setFixedWidth(150)
        btn_save.clicked.connect(self.save_plot)
        btn_settings = QPushButton('Settings')
        btn_settings.setFixedWidth(150)
        # settings 功能可扩展，此处仅占位
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_settings)

        # 底部控制栏
        self.cb_relative = QCheckBox('Relative')
        self.cb_relative.setChecked(True)
        self.cb_relative.stateChanged.connect(self.set_relative)

        self.cb_sensor = QCheckBox('Sensors')
        self.cb_sensor.setChecked(False)
        self.cb_sensor.stateChanged.connect(self.set_sensor)

        btn_exclude = QPushButton('Excluded Channels')
        btn_exclude.clicked.connect(self.show_exclude_dialog)

        self.edit_excluded = QLineEdit()
        self.edit_excluded.setReadOnly(True)
        self.edit_excluded.setPlaceholderText('Selected bad channels')

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
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.table_widget)
        main_layout.addLayout(bottom_layout)

    def set_relative(self):
        self.is_relative = self.cb_relative.isChecked()
        self.plot()

    def set_sensor(self):
        self.is_show_sensor = self.cb_sensor.isChecked()
        self.plot()

    def show_exclude_dialog(self):
        if self.data is None:
            QMessageBox.warning(self, 'Warning', 'No data loaded.')
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
            # 更新表格部件（简单替换）
            self.table_widget.deleteLater()
            self.table_widget = DataTableView(self.show_data, self.show_channel_names, self.band_titles)
            # 需重新插入布局（简化处理，此处仅重新绘图）
            self.plot()

    def plot(self):
        """绘制拓扑图"""
        if self.data is None or self.show_data.size == 0:
            return

        # 清空坐标轴
        for ax in self.axes:
            ax.clear()

        # 准备蒙太奇和信息
        montage = mne.channels.make_standard_montage('standard_1005')
        # 限制通道数量（实际可用全部，但 MNE 绘图可能对过多通道有性能问题）
        n_use = min(30, len(self.show_channel_names))
        if n_use == 0:
            QMessageBox.warning(self, "Warning", "No channels to plot")
            return
        use_channels = self.show_channel_names[:n_use]
        use_data = self.show_data[:n_use, :]
        info = mne.create_info(ch_names=use_channels, sfreq=1000, ch_types='eeg')
        evoked = mne.EvokedArray(data=use_data, info=info)
        try:
            evoked.set_montage(montage)
        except ValueError as e:
            # 如果通道名与蒙太奇不匹配，提示用户并选择忽略
            reply = QMessageBox.question(
                self, "Montage Mismatch",
                f"Channel names do not match the standard montage.\n\nError: {str(e)}\n\n"
                "Ignore and continue? (Electrode positions will be missing)",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                evoked.set_montage(montage, on_missing='ignore')
            else:
                return

        # 数据归一化
        if self.is_relative:
            norm_data = min_max_scaling_to_range(use_data.T)  # shape (bands, channels)
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
        self.canvas.draw()

    def save_plot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Topography",
                                                   "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if not file_path:
            return
        self.fig.savefig(file_path, dpi=300)
        QMessageBox.information(self, "Success", f"Plot saved to {file_path}")


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    # 使用标准通道名生成示例数据
    montage = mne.channels.make_standard_montage('standard_1005')
    ch_names = montage.ch_names[:32]  # 取前32个标准通道
    data = {
        'type': 'eeg_psd',
        'ch_names': ch_names,
        'feature': {
            'Delta': np.random.rand(32),
            'Theta': np.random.rand(32),
            'Alpha': np.random.rand(32),
            'Beta': np.random.rand(32),
            'Gamma': np.random.rand(32)
        }
    }
    app = QApplication(sys.argv)
    win = TopographyView(data)
    win.show()
    sys.exit(app.exec_())