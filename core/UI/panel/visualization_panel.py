#!/usr/bin/env python3
"""
visualization_panel.py
可视化集成面板 - 纯图片按钮 + 透明文本版本
"""
import sys
from pathlib import Path

# 将项目根目录动态添加到 sys.path
start_path = Path(__file__).resolve().parent
for parent in [start_path] + list(start_path.parents):
    if parent.name == 'core':
        project_root = parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        break
else:
    raise RuntimeError("未找到名为 'core' 的目录")

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

# 导入功能模块
from core.visualizer.signal_view import SignalView, fNIRSView
from core.visualizer.stats_view import StatsView
from core.visualizer.bar_view import BarView
from core.visualizer.time_frequency_view import TimeFrequencyView
from core.visualizer.topography_view import TopographyView
from core.visualizer.feature_view import FeatureView
from core.io.data_io import DataLoader

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("警告: PIL 未安装，无法加载 PNG 图片，请安装 Pillow 库。")


class ModernVisualizationPanel(ttk.Frame):
    """
    可视化集成面板 - 纯图片按钮 + 透明文本
    """

    def __init__(self, parent, data_dict=None, file_path=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.data_dict = data_dict
        self.file_path = file_path
        self.data_loader = DataLoader()
        self.current_views = {}
        self.qt_app = None

        # 图片资源根目录
        self.resource_dir = project_root / "core" / "UI" / "UI_resource" / "virtualization_panel_resource"

        # 存储图片引用
        self.images = {}

        # 文件路径变量（用于更新显示）
        self.file_path_var = tk.StringVar()

        # 设置UI
        self.setup_ui()

        if file_path and not data_dict:
            self.load_file(file_path)

    def setup_ui(self):
        """创建纯 Canvas 界面，所有元素均为图像或文本"""
        # 主 Canvas
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 加载背景图并获取尺寸
        self._load_and_place_background()

        # 放置文件地址显示框（背景图）
        self._place_file_address_box()

        # 放置四个控制按钮（纯图片，无灰色背景）
        self._place_control_buttons()

        # 放置数据信息显示框（背景图）及内部文本
        self._place_info_box()

        # 放置六个功能模块按钮（纯图片）
        self._place_function_buttons()

        # 初始化显示
        self.update_file_info()

    def _load_image(self, filename):
        """加载图片并返回 PhotoImage 对象，若失败则返回 None"""
        if not PIL_AVAILABLE:
            return None
        img_path = self.resource_dir / filename
        if not img_path.exists():
            print(f"图片不存在: {img_path}")
            return None
        try:
            pil_img = Image.open(img_path)
            return ImageTk.PhotoImage(pil_img)
        except Exception as e:
            print(f"加载图片失败 {filename}: {e}")
            return None

    def _load_and_place_background(self):
        """加载并放置背景图，记录尺寸"""
        bg_img = self._load_image("virtualization_panel_background.png")
        if bg_img:
            self.images["background"] = bg_img
            self.bg_width = bg_img.width()
            self.bg_height = bg_img.height()
            self.canvas.create_image(0, 0, image=bg_img, anchor="nw", tags="background")
            # 设置 Canvas 滚动区域为背景图大小（如果需要滚动，但这里我们让窗口匹配背景）
            self.canvas.config(scrollregion=(0, 0, self.bg_width, self.bg_height))
        else:
            # 降级：使用空白画布
            self.bg_width, self.bg_height = 1200, 1000

    def _place_file_address_box(self):
        """放置文件地址显示框图片及透明路径文本"""
        box_img = self._load_image("file_address_display_box.png")
        if box_img:
            self.images["file_address_box"] = box_img
            self.canvas.create_image(200, 190, image=box_img, anchor="nw", tags="file_address_box")

            # 计算文本显示位置（图片内适当偏移，使文字居中）
            # 假设图片内部有效区域从 (10,10) 到 (width-10, height-10)
            # 我们根据实际图片微调：这里以图片宽度 400，高度 40 为例，您可以手动调整
            # 更好的方法是读取图片后获取尺寸并计算
            box_width = box_img.width()
            box_height = box_img.height()
            text_x = 200 + box_width // 2   # 水平居中
            text_y = 190 + box_height // 2  # 垂直居中

            # 创建透明文本显示文件路径
            self.file_path_text_id = self.canvas.create_text(
                text_x, text_y,
                text="",
                font=('微软雅黑', 16,"bold"),
                fill="#000000",      # 文字颜色
                anchor='center',
                tags="file_path_text"
            )
            # 点击文本同样打开文件选择（方便操作）
            self.canvas.tag_bind("file_path_text", "<Button-1>", lambda e: self.browse_file())
        else:
            # 备用：直接显示普通文本
            self.file_path_text_id = self.canvas.create_text(
                400, 210, text="", font=('微软雅黑', 16,'bold'), fill='black', anchor='center'
            )

    def _place_control_buttons(self):
        """放置四个控制按钮（纯图片，绑定点击）"""
        # 按钮坐标（新坐标）
        btn_specs = [
            ("browse_button.png",       70, 250, self.browse_file),
            ("load_button.png",        274, 250, self.load_current_file),
            ("demonstration_button.png",478, 250, self.load_demo_data),
            ("reset_button.png",        682, 250, self.reset_data),
        ]
        for fname, x, y, cmd in btn_specs:
            img = self._load_image(fname)
            if img:
                self.images[f"ctrl_{fname}"] = img
                # 创建图片项
                item_id = self.canvas.create_image(x, y, image=img, anchor="nw", tags=("ctrl_btn", fname))
                # 绑定点击事件
                self.canvas.tag_bind(item_id, "<Button-1>", lambda e, c=cmd: c())

    def _place_info_box(self):
        """放置数据信息显示框图片及内部文本"""
        info_img = self._load_image("data_information_display_box.png")
        if info_img:
            self.images["info_box"] = info_img
            self.canvas.create_image(50, 320, image=info_img, anchor="nw", tags="info_box")

            # 计算文本居中位置
            img_width = info_img.width()
            img_height = info_img.height()
            center_x = 50 + img_width // 2
            center_y = 320 + img_height // 2

            # 创建文本项（多行文本）
            self.info_text_id = self.canvas.create_text(
                center_x, center_y,
                text="",
                font=('微软雅黑',16,'bold'),
                fill="#000000",
                width=img_width - 40,      # 文本换行宽度，留出边距
                anchor='center',
                justify='left',
                tags="info_text"
            )
        else:
            # 备用
            self.info_text_id = self.canvas.create_text(
                300, 400, text="", font=('微软雅黑',16,'bold'), fill='black', anchor='center'
            )

    def _place_function_buttons(self):
        """放置六个功能模块按钮（纯图片）"""
        btn_specs = [
            ("waveform_button.png",       80, 720, self.open_signal_view),
            ("analysis_button.png",      595, 720, self.open_stats_view),
            ("barchart_button.png",     1110, 720, self.open_bar_view),
            ("time_frequency_button.png", 80, 900, self.open_time_frequency),
            ("topographic_map_button.png",595, 900, self.open_topography),
            ("feature_map_button.png",   1110, 900, self.open_feature_view),
        ]
        for fname, x, y, cmd in btn_specs:
            img = self._load_image(fname)
            if img:
                self.images[f"func_{fname}"] = img
                item_id = self.canvas.create_image(x, y, image=img, anchor="nw", tags=("func_btn", fname))
                self.canvas.tag_bind(item_id, "<Button-1>", lambda e, c=cmd: c())

    # ========== 文件操作方法 ==========
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[
                ("所有支持文件", "*.edf *.bdf *.gdf *.mat *.csv *.xlsx *.xls *.json *.npy *.set *.vhdr"),
                ("EDF文件", "*.edf"), ("BDF文件", "*.bdf"), ("GDF文件", "*.gdf"),
                ("MAT文件", "*.mat"), ("NumPy文件", "*.npy"), ("JSON文件", "*.json"),
                ("CSV文件", "*.csv"), ("Excel文件", "*.xlsx *.xls"),
                ("EEGLAB文件", "*.set *.vhdr")
            ]
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.update_file_info()  # 更新显示

    def load_current_file(self):
        file_path = self.file_path_var.get().strip()
        if not file_path:
            messagebox.showwarning("警告", "请先选择文件")
            return
        self.load_file(file_path)

    def load_file(self, file_path):
        try:
            if file_path.endswith('.npy'):
                self.data_dict = np.load(file_path, allow_pickle=True).item()
            elif file_path.endswith('.json'):
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.data_dict = json.load(f)
            else:
                self.data_dict = self.data_loader.load(file_path)
            self.file_path = file_path
            self.file_path_var.set(file_path)
            self.update_file_info()
            messagebox.showinfo("成功", "文件加载成功！")
        except Exception as e:
            messagebox.showerror("错误", f"文件加载失败:\n{str(e)}")

    def load_demo_data(self):
        fs = 1000
        t = np.arange(0, 30, 1 / fs)
        self.data_dict = {
            "meta": {
                "subject_id": "demo", "session_id": "session1", "task": "rest",
                "modality": ["EEG", "fNIRS"]
            },
            "signal": {
                "EEG": {
                    "data": np.array([np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t)) for _ in range(8)]),
                    "sampling_rate": fs, "channel_names": [f"EEG_{i}" for i in range(8)], "unit": "uV"
                },
                "fNIRS": {
                    "data": np.array([np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(len(t)) for _ in range(4)]),
                    "sampling_rate": fs, "channel_names": [f"NIRS_{i}" for i in range(4)], "unit": "uM"
                }
            },
            "feature": {
                "type": "eeg_psd", "ch_names": [f"EEG_{i}" for i in range(8)],
                "feature": {
                    "Delta": np.random.rand(8), "Theta": np.random.rand(8),
                    "Alpha": np.random.rand(8), "Beta": np.random.rand(8), "Gamma": np.random.rand(8)
                }
            }
        }
        self.file_path = None
        self.file_path_var.set("")
        self.update_file_info()
        messagebox.showinfo("成功", "演示数据加载成功！")

    def reset_data(self):
        self.data_dict = None
        self.file_path = None
        self.file_path_var.set("")
        self.update_file_info()

    def update_file_info(self):
        """更新文件地址文本和数据信息文本"""
        # 更新文件地址显示
        self.canvas.itemconfig(self.file_path_text_id, text=self.file_path_var.get())

        # 更新数据信息显示
        if not hasattr(self, 'info_text_id'):
            return

        if not self.data_dict:
            info = "请加载数据文件或使用演示数据"
        else:
            meta = self.data_dict.get("meta", {})
            subject = meta.get("subject_id", "unknown")
            task = meta.get("task", "unknown")
            modalities = meta.get("modality", [])
            info = f"🧑 被试: {subject}\n📋 任务: {task}\n🔬 模态: {', '.join(modalities)}\n"

            signal_dict = self.data_dict.get("signal", {})
            for mod, sig in signal_dict.items():
                if isinstance(sig, dict) and 'data' in sig:
                    data = sig['data']
                    n_ch = data.shape[0] if hasattr(data, 'shape') else 1
                    fs = sig.get('sampling_rate', '?')
                    info += f"📊 {mod}: {n_ch}通道, {fs}Hz\n"

        self.canvas.itemconfig(self.info_text_id, text=info)

    # ========== 功能模块打开方法（保持不变） ==========
    def open_signal_view(self):
        if not self.data_dict:
            messagebox.showwarning("警告", "请先加载数据")
            return
        try:
            window = tk.Toplevel(self)
            window.title("信号波形视图")
            window.geometry("1200x800")
            meta = self.data_dict.get("meta", {})
            modalities = meta.get("modality", [])
            if "fNIRS" in modalities or "NIRS" in modalities:
                view = fNIRSView(window, self.data_dict)
            else:
                view = SignalView(window, self.data_dict)
            view.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("错误", f"打开信号视图失败:\n{str(e)}")

    def open_stats_view(self):
        if not self.data_dict:
            messagebox.showwarning("警告", "请先加载数据")
            return
        try:
            window = tk.Toplevel(self)
            window.title("统计分析视图")
            window.geometry("1100x800")

            # 直接传递整个数据字典
            view = StatsView(window, data_dict=self.data_dict)
            view.pack(fill=tk.BOTH, expand=True)

            # 添加调试信息
            print("打开统计视图:")
            print(f"data_dict keys: {list(self.data_dict.keys())}")
            if "processed" in self.data_dict:
                print(f"processed keys: {list(self.data_dict['processed'].keys())}")
        except Exception as e:
            messagebox.showerror("错误", f"打开统计视图失败:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def open_bar_view(self):
        if not self.data_dict:
            messagebox.showwarning("警告", "请先加载数据")
            return
        try:
            window = tk.Toplevel(self)
            window.title("柱状图视图")
            window.geometry("1200x800")
            view = BarView(window, self.data_dict)
            view.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("错误", f"打开柱状图视图失败:\n{str(e)}")

    def open_time_frequency(self):
        if not self.data_dict:
            messagebox.showwarning("警告", "请先加载数据")
            return
        try:
            from core.visualizer.time_frequency_view import short_time_Fourier_transform
            signal_dict = self.data_dict.get("signal", {})
            if not signal_dict:
                messagebox.showwarning("警告", "数据中没有信号")
                return
            modality = list(signal_dict.keys())[0]
            signal_info = signal_dict[modality]
            raw_data = signal_info.get('data')
            fs = signal_info.get('sampling_rate', 1000)
            ch_names = signal_info.get('channel_names', [])
            stft_input = {'data': raw_data, 'srate': fs, 'ch_names': ch_names}
            stft_result = short_time_Fourier_transform(stft_input)
            self.ensure_pyqt_app()
            self.tf_window = TimeFrequencyView(stft_result)
            self.tf_window.show()
        except Exception as e:
            messagebox.showerror("错误", f"打开时频分析视图失败:\n{str(e)}")

    def open_topography(self):
        if not self.data_dict:
            messagebox.showwarning("警告", "请先加载数据")
            return
        try:
            feature_data = self.data_dict.get("feature")
            if not feature_data:
                self.prepare_topography_data()
                feature_data = self.data_dict.get("feature")
            if not feature_data:
                messagebox.showwarning("警告", "无法准备地形图数据")
                return
            self.ensure_pyqt_app()
            self.topo_window = TopographyView(feature_data)
            self.topo_window.show()
        except Exception as e:
            messagebox.showerror("错误", f"打开地形图视图失败:\n{str(e)}")

    def prepare_topography_data(self):
        standard_channels = ['Fz', 'Cz', 'Pz', 'Oz', 'F3', 'F4', 'C3', 'C4',
                             'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8']
        self.data_dict["feature"] = {
            'type': 'eeg_psd', 'ch_names': standard_channels[:10],
            'feature': {
                'Delta': np.random.rand(10), 'Theta': np.random.rand(10),
                'Alpha': np.random.rand(10), 'Beta': np.random.rand(10), 'Gamma': np.random.rand(10)
            }
        }

    def open_feature_view(self):
        if not self.data_dict:
            messagebox.showwarning("警告", "请先加载数据")
            return
        try:
            from core.visualizer.feature_view import show_feature_view
            self.feature_window = show_feature_view(self, self.data_dict)

        except Exception as e:
            messagebox.showerror("错误", f"打开特征视图失败:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def ensure_pyqt_app(self):
        try:
            from PyQt5.QtWidgets import QApplication
            if not QApplication.instance():
                self.qt_app = QApplication(sys.argv)
        except ImportError:
            pass


# 测试代码
if __name__ == "__main__":
    root = tk.Tk()
    root.title("可视化面板测试 - 纯图片按钮 + 透明文本")

    # 创建面板，但不立即 pack，先获取背景尺寸
    panel = ModernVisualizationPanel(root)
    # 如果有背景图片，设置窗口大小为背景尺寸
    if hasattr(panel, 'bg_width') and hasattr(panel, 'bg_height'):
        root.geometry(f"{panel.bg_width}x{panel.bg_height}")
    else:
        root.geometry("1300x1000")  # 降级尺寸

    panel.pack(fill=tk.BOTH, expand=True)

    # 确保关闭窗口时退出程序
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()