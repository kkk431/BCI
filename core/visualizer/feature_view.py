# -*- coding: utf-8 -*-
"""
特征可视化模块
核心类：FeatureView (QWidget) 及其子类
功能：
- 曲线图、柱状图、表格、拓扑图
- 通道/特征选择对话框
- 绘图设置（标题、轴标签、尺寸）
- 保存图像
"""

import sys
import json
import mne
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLineEdit, QLabel, QDialog, QFormLayout,
    QSpinBox, QMessageBox, QInputDialog, QScrollArea, QCheckBox,
    QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# -------------------- 辅助函数--------------------
def min_max_scaling_to_range(data, target_range=(-1, 1)):
    data = np.asarray(data)
    min_vals = data.min(axis=1, keepdims=True)
    max_vals = data.max(axis=1, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    scaled = (data - min_vals) / range_vals
    scaled = scaled * (target_range[1] - target_range[0]) + target_range[0]
    return scaled


def min_max_scaling_by_arrays(data, target_range=(-1, 1)):
    data = np.asarray(data)
    min_val = data.min()
    max_val = data.max()
    if max_val - min_val == 0:
        return np.zeros_like(data)
    scaled = (data - min_val) / (max_val - min_val)
    scaled = scaled * (target_range[1] - target_range[0]) + target_range[0]
    return scaled


# -------------------- 辅助函数（替代 BrainFusion.utils.transform.read_info）--------------------
def read_info(file_path):
    """从 JSON 文件读取传感器信息"""
    with open(file_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    return info


# ---------- 辅助对话框 ----------
class SelectItemsDialog(QDialog):
    """带复选框的项目选择对话框"""

    def __init__(self, items, title, selected_items=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.items = items
        self.selected_items = selected_items if selected_items else []
        self.checkboxes = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        for item in self.items:
            cb = QCheckBox(item)
            cb.setChecked(item in self.selected_items)
            self.checkboxes.append(cb)
            layout.addWidget(cb)

        hlayout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.toggle_select_all)
        hlayout.addWidget(self.select_all_btn)
        confirm_btn = QPushButton("Confirm")
        confirm_btn.clicked.connect(self.accept)
        hlayout.addWidget(confirm_btn)
        layout.addLayout(hlayout)
        self.setLayout(layout)

    def toggle_select_all(self):
        all_checked = all(cb.isChecked() for cb in self.checkboxes)
        for cb in self.checkboxes:
            cb.setChecked(not all_checked)

    def get_selected_items(self):
        self.selected_items = [cb.text() for cb in self.checkboxes if cb.isChecked()]
        return self.selected_items


class PlotSettingsDialog(QWidget):
    """绘图设置对话框"""
    closed_signal = pyqtSignal()

    def __init__(self, parent=None, initial_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Settings")
        self.resize(300, 200)
        self.layout = QVBoxLayout()
        self.init_ui(initial_settings)

    def init_ui(self, initial_settings):
        form = QFormLayout()
        self.title_edit = QLineEdit()
        self.xlabel_edit = QLineEdit()
        self.ylabel_edit = QLineEdit()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 50)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 50)

        if initial_settings:
            self.title_edit.setText(initial_settings.get("title", ""))
            self.xlabel_edit.setText(initial_settings.get("xlabel", ""))
            self.ylabel_edit.setText(initial_settings.get("ylabel", ""))
            self.width_spin.setValue(initial_settings.get("width", 8))
            self.height_spin.setValue(initial_settings.get("height", 6))
        else:
            self.width_spin.setValue(8)
            self.height_spin.setValue(6)

        form.addRow("Title:", self.title_edit)
        form.addRow("X Label:", self.xlabel_edit)
        form.addRow("Y Label:", self.ylabel_edit)
        form.addRow("Width:", self.width_spin)
        form.addRow("Height:", self.height_spin)

        self.layout.addLayout(form)
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self.close)
        self.layout.addWidget(self.save_btn)
        self.setLayout(self.layout)

    def get_settings(self):
        return {
            "title": self.title_edit.text(),
            "xlabel": self.xlabel_edit.text(),
            "ylabel": self.ylabel_edit.text(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value()
        }

    def closeEvent(self, event):
        self.closed_signal.emit()
        event.accept()


# ---------- 画布基类及实现 ----------
class BaseCanvas(FigureCanvas):
    """绘图画布基类"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)


class CurveCanvas(BaseCanvas):
    """曲线图画布"""

    def plot(self, data, selected_channels, selected_features, settings):
        self.ax.clear()
        select_data = self._get_selected_data(data, selected_channels, selected_features)
        x_labels = selected_features
        chan_values = np.array([select_data[f] for f in selected_features]).T
        for i, ch in enumerate(selected_channels):
            self.ax.plot(x_labels, chan_values[i], marker='o', label=ch)
        self.ax.set_xticks(range(len(x_labels)))
        self.ax.set_xticklabels(x_labels, rotation=45, ha="right")
        self.ax.set_title(settings.get("title", ""))
        self.ax.set_xlabel(settings.get("xlabel", ""))
        self.ax.set_ylabel(settings.get("ylabel", ""))
        self.ax.legend()
        self.draw()

    def _get_selected_data(self, feature_dict, channels, features):
        idx = [i for i, ch in enumerate(feature_dict['ch_names']) if ch in channels]
        out = {}
        for f in features:
            if f in feature_dict['feature']:
                out[f] = [feature_dict['feature'][f][i] for i in idx]
        return out


class BarCanvas(BaseCanvas):
    """柱状图画布"""

    def plot(self, data, selected_channels, selected_features, settings):
        self.ax.clear()
        select_data = self._get_selected_data(data, selected_channels, selected_features)
        x = range(len(selected_channels))
        bar_width = 0.8 / len(selected_features)
        for i, f in enumerate(selected_features):
            values = select_data[f]
            pos = [xi + i * bar_width for xi in x]
            self.ax.bar(pos, values, bar_width, label=f)
        self.ax.set_xticks([xi + (len(selected_features)-1)*bar_width/2 for xi in x])
        self.ax.set_xticklabels(selected_channels)
        self.ax.set_title(settings.get("title", ""))
        self.ax.set_xlabel(settings.get("xlabel", ""))
        self.ax.set_ylabel(settings.get("ylabel", ""))
        self.ax.legend()
        self.draw()

    def _get_selected_data(self, feature_dict, channels, features):
        idx = [i for i, ch in enumerate(feature_dict['ch_names']) if ch in channels]
        out = {}
        for f in features:
            if f in feature_dict['feature']:
                out[f] = [feature_dict['feature'][f][i] for i in idx]
        return out


class TableCanvas(QTableWidget):
    """表格画布"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.horizontalHeader().setDefaultSectionSize(200)

    def plot(self, df):
        self.clear()
        self.setRowCount(len(df))
        self.setColumnCount(len(df.columns))
        self.setHorizontalHeaderLabels(df.columns)
        for i in range(len(df)):
            for j in range(len(df.columns)):
                self.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))


class TopomapCanvas(FigureCanvas):
    """地形图画布"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot(self, data, info_dict, selected_channels, selected_features,
             is_relative=False, is_show_sensor=False):
        if not data:
            return
        num = len(selected_features)
        self.axes = self.fig.subplots(1, num, sharex=True, sharey=True)
        if num == 1:
            self.axes = [self.axes]

        select_data = self._get_selected_data(data, selected_channels, selected_features)
        chan_data = np.array([select_data[f] for f in selected_features])

        if is_relative:
            norm_data = min_max_scaling_to_range(chan_data)
        else:
            norm_data = min_max_scaling_by_arrays(chan_data)
        vlim = (-1, 1)

        if 'montage' in info_dict:
            montage = mne.channels.make_standard_montage(info_dict['montage'])
            info = mne.create_info(ch_names=selected_channels, sfreq=info_dict.get('srate', 1000), ch_types='eeg')
            evoked = mne.EvokedArray(data=chan_data.T, info=info)
            evoked.set_montage(montage)

            for i, (ax, psd) in enumerate(zip(self.axes, norm_data)):
                ax.clear()
                ax.set_title(selected_features[i])
                mne.viz.plot_topomap(
                    psd, evoked.info, axes=ax, show=False,
                    sensors=is_show_sensor, vlim=vlim,
                    names=selected_channels if is_show_sensor else None
                )
            self.draw()
        elif 'loc' in info_dict:
            # 可扩展自定义位置
            pass

    def _get_selected_data(self, feature_dict, channels, features):
        idx = [i for i, ch in enumerate(feature_dict['ch_names']) if ch in channels]
        out = {}
        for f in features:
            if f in feature_dict['feature']:
                out[f] = [feature_dict['feature'][f][i] for i in idx]
        return out


# ---------- 特征视图基类 ----------
class FeatureView(QWidget):
    """特征可视化基类"""

    def __init__(self, data, channels, features, parent=None):
        super().__init__(parent)
        self.data = data
        self.channels = channels
        self.features = features
        self.settings = {"width": 8, "height": 6, "title": "", "xlabel": "", "ylabel": ""}
        self.init_ui()

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)

        # 通道选择行
        ch_layout = QHBoxLayout()
        self.ch_btn = QPushButton("Select Channels")
        self.ch_btn.clicked.connect(self.select_channels)
        self.ch_edit = QLineEdit()
        ch_layout.addWidget(self.ch_btn)
        ch_layout.addWidget(self.ch_edit)
        self.main_layout.addLayout(ch_layout)

        # 特征选择行
        feat_layout = QHBoxLayout()
        self.feat_btn = QPushButton("Select Features")
        self.feat_btn.clicked.connect(self.select_features)
        self.feat_edit = QLineEdit()
        feat_layout.addWidget(self.feat_btn)
        feat_layout.addWidget(self.feat_edit)
        self.main_layout.addLayout(feat_layout)

        # 中间布局（供子类扩展）
        self.mid_layout = QHBoxLayout()
        self.main_layout.addLayout(self.mid_layout)

        # 按钮行
        btn_layout = QHBoxLayout()
        self.plot_btn = QPushButton("Generate Plot")
        self.plot_btn.setFixedWidth(120)
        self.plot_btn.clicked.connect(self.plot)
        btn_layout.addWidget(self.plot_btn)

        self.save_btn = QPushButton("Save Plot")
        self.save_btn.setFixedWidth(120)
        self.save_btn.clicked.connect(self.save_plot)
        btn_layout.addWidget(self.save_btn)

        btn_layout.addStretch(1)

        self.settings_btn = QPushButton("Configure Settings")
        self.settings_btn.setFixedWidth(120)
        self.settings_dialog = PlotSettingsDialog()
        self.settings_dialog.save_btn.clicked.connect(self.save_settings)
        self.settings_btn.clicked.connect(self.settings_dialog.show)
        btn_layout.addWidget(self.settings_btn)

        self.main_layout.addLayout(btn_layout)

        # 滚动区域（用于放置画布）
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.main_layout.addWidget(self.scroll_area)

    def select_channels(self):
        dialog = SelectItemsDialog(self.channels, "Select Channels")
        if dialog.exec_() == QDialog.Accepted:
            sel = dialog.get_selected_items()
            self.ch_edit.setText(", ".join(sel))

    def select_features(self):
        dialog = SelectItemsDialog(self.features, "Select Features")
        if dialog.exec_() == QDialog.Accepted:
            sel = dialog.get_selected_items()
            self.feat_edit.setText(", ".join(sel))

    def save_settings(self):
        self.settings = self.settings_dialog.get_settings()

    def plot(self):
        raise NotImplementedError

    def save_plot(self):
        raise NotImplementedError


# ---------- 具体子类 ----------
class CurveFeatureView(FeatureView):
    def __init__(self, data, channels, features):
        super().__init__(data, channels, features)
        self.canvas = CurveCanvas()
        self.scroll_area.setWidget(self.canvas)

    def plot(self):
        if not self.data:
            QMessageBox.warning(self, "Warning", "No data")
            return
        ch = self.ch_edit.text().split(", ") if self.ch_edit.text() else []
        feat = self.feat_edit.text().split(", ") if self.feat_edit.text() else []
        if not ch or not feat:
            QMessageBox.warning(self, "Warning", "Select channels and features")
            return
        w = self.settings.get("width", 8)
        h = self.settings.get("height", 6)
        self.canvas = CurveCanvas(self, width=w, height=h)
        self.scroll_area.setWidget(self.canvas)
        self.canvas.fig.set_size_inches(w, h, forward=True)
        self.canvas.plot(self.data, ch, feat, self.settings)

    def save_plot(self):
        if not self.canvas.fig.axes:
            QMessageBox.warning(self, "Warning", "Generate plot first")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Figure", "",
                                              "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        dpi, ok = QInputDialog.getInt(self, "DPI", "DPI:", 300, 50, 600)
        if ok:
            self.canvas.fig.savefig(path, dpi=dpi)
            QMessageBox.information(self, "Success", f"Saved to {path}")


class BarFeatureView(FeatureView):
    def __init__(self, data, channels, features):
        super().__init__(data, channels, features)
        self.canvas = BarCanvas()
        self.scroll_area.setWidget(self.canvas)

    def plot(self):
        if not self.data:
            QMessageBox.warning(self, "Warning", "No data")
            return
        ch = self.ch_edit.text().split(", ") if self.ch_edit.text() else []
        feat = self.feat_edit.text().split(", ") if self.feat_edit.text() else []
        if not ch or not feat:
            QMessageBox.warning(self, "Warning", "Select channels and features")
            return
        w = self.settings.get("width", 8)
        h = self.settings.get("height", 6)
        self.canvas = BarCanvas(self, width=w, height=h)
        self.scroll_area.setWidget(self.canvas)
        self.canvas.fig.set_size_inches(w, h, forward=True)
        self.canvas.plot(self.data, ch, feat, self.settings)

    def save_plot(self):
        if not self.canvas.fig.axes:
            QMessageBox.warning(self, "Warning", "Generate plot first")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Figure", "",
                                              "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        dpi, ok = QInputDialog.getInt(self, "DPI", "DPI:", 300, 50, 600)
        if ok:
            self.canvas.fig.savefig(path, dpi=dpi)
            QMessageBox.information(self, "Success", f"Saved to {path}")


class TableFeatureView(FeatureView):
    def __init__(self, data, channels, features):
        super().__init__(data, channels, features)
        self.canvas = TableCanvas()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.canvas)
        # 隐藏通道/特征选择（表格显示全部）
        self.ch_btn.setVisible(False)
        self.ch_edit.setVisible(False)
        self.feat_btn.setVisible(False)
        self.feat_edit.setVisible(False)
        self.settings_btn.setVisible(False)

    def plot(self):
        df = self._feature_dict_to_df(self.data)
        self.canvas.plot(df)

    def save_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Table", "", "Excel (*.xlsx)")
        if not path:
            return
        df = self._feature_dict_to_df(self.data)
        df.to_excel(path, index=False)
        QMessageBox.information(self, "Success", f"Saved to {path}")

    def _feature_dict_to_df(self, fd):
        data = {"Channel": fd["ch_names"]}
        for k, v in fd["feature"].items():
            data[k] = v
        df = pd.DataFrame(data)
        df["Type"] = fd["type"]
        return df


class TopomapFeatureView(FeatureView):
    def __init__(self, data, channels, features):
        super().__init__(data, channels, features)
        self.canvas = TopomapCanvas()
        self.scroll_area.setWidget(self.canvas)
        self.info = None

        # 添加信息文件选择
        self.info_btn = QPushButton('Select Info File')
        self.info_btn.setFixedWidth(120)
        self.info_edit = QLineEdit()
        self.info_btn.clicked.connect(self.select_info)
        self.mid_layout.addWidget(self.info_btn)
        self.mid_layout.addWidget(self.info_edit)

        # 底部复选框
        self.bottom_layout = QHBoxLayout()
        self.cb_relative = QCheckBox('Relative Scaling')
        self.cb_sensor = QCheckBox('Show Sensors')
        self.bottom_layout.addWidget(self.cb_relative)
        self.bottom_layout.addWidget(self.cb_sensor)
        self.bottom_layout.addStretch()
        self.main_layout.addLayout(self.bottom_layout)

    def select_info(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Info File", "", "JSON (*.json)")
        if path:
            self.info = read_info(path)
            self.info_edit.setText(path)

    def plot(self):
        if not self.data:
            QMessageBox.warning(self, "Warning", "No data")
            return
        ch = self.ch_edit.text().split(", ") if self.ch_edit.text() else []
        feat = self.feat_edit.text().split(", ") if self.feat_edit.text() else []
        if not ch or not feat:
            QMessageBox.warning(self, "Warning", "Select channels and features")
            return
        if not self.info:
            QMessageBox.warning(self, "Warning", "Select info file first")
            return
        w = self.settings.get("width", 8)
        h = self.settings.get("height", 6)
        self.canvas = TopomapCanvas(self, width=w, height=h)
        self.scroll_area.setWidget(self.canvas)
        self.canvas.plot(self.data, self.info, ch, feat,
                         self.cb_relative.isChecked(), self.cb_sensor.isChecked())

    def save_plot(self):
        if not self.canvas.fig.axes:
            QMessageBox.warning(self, "Warning", "Generate plot first")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Figure", "",
                                              "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        dpi, ok = QInputDialog.getInt(self, "DPI", "DPI:", 300, 50, 600)
        if ok:
            self.canvas.fig.savefig(path, dpi=dpi)
            QMessageBox.information(self, "Success", f"Saved to {path}")


# 使用示例
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 模拟数据
    test_data = {
        'ch_names': ['Fz', 'Cz', 'Pz'],
        'feature': {
            'Delta': [0.1, 0.2, 0.3],
            'Theta': [0.4, 0.5, 0.6],
            'Alpha': [0.7, 0.8, 0.9]
        },
        'type': 'test'
    }
    view = CurveFeatureView(test_data, test_data['ch_names'], list(test_data['feature'].keys()))
    view.show()
    sys.exit(app.exec_())