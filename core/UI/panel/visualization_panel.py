#!/usr/bin/env python3
"""
visualization_panel.py
可视化集成面板 - 固定尺寸1440×1024绝对坐标布局（与预处理模块一致）
包含导航按钮选中/未选中状态切换
"""
import sys
from pathlib import Path

# 将项目根目录动态添加到 sys.path
start_path = Path(__file__).resolve().parent
current_path = start_path
for path in [current_path] + list(current_path.parents):
    if path.name == 'core':
        project_root = path.parent
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


class NeuroPioneerPanel(ttk.Frame):
    def __init__(self, parent, data_dict=None, file_path=None, show_navigation=True, **kwargs):
        """
        show_navigation: 是否显示左侧导航栏，默认为True
                        在主界面作为子面板时设为False
        """
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.data_dict = data_dict
        self.file_path = file_path
        self.data_loader = DataLoader()
        self.images = {}  # 保持图片引用
        self.file_path_var = tk.StringVar()
        self.show_navigation = show_navigation  # 新增参数

        # 资源目录
        self.resource_dir = project_root / "core" / "UI" / "UI_resource" / "Virtualization_Panel"
        self.navigation_dir = project_root / "core" / "UI" / "UI_resource" / "Navigation"
        self.buttons_function_dir = self.resource_dir / "Buttons" / "Function"
        self.buttons_loading_dir = self.resource_dir / "Buttons" / "Data_Loading"

        # 导航按钮目录
        self.nav_buttons_unselected_dir = self.navigation_dir / "Buttons" / "Unselected"
        self.nav_buttons_selected_dir = self.navigation_dir / "Buttons" / "Selected"

        # 初始化 UI
        self.setup_ui()

        if file_path and not data_dict:
            self.load_file(file_path)

    def setup_ui(self):
        """初始化UI - 使用设计稿绝对坐标 (1440×1024) - 固定尺寸"""
        # 设置固定尺寸
        self.configure(width=1440, height=1024)
        self.pack_propagate(False)  # 禁止子控件改变Frame大小

        # 创建Canvas，固定尺寸
        self.canvas = tk.Canvas(
            self,
            width=1440,
            height=1024,
            highlightthickness=0,
            bg="#FFFFFF"
        )
        self.canvas.pack(fill=tk.BOTH, expand=False)  # 不扩展

        # 计算内容区域的起始X坐标
        if self.show_navigation:
            # 显示导航栏时，内容从298开始（导航栏宽度270 + 左边距9 + 右边距19）
            self.content_start_x = 298
        else:
            # 不显示导航栏时，内容从9开始（紧贴左边）
            self.content_start_x = 9

        # ============ 1. 左侧导航区（可选） ============
        if self.show_navigation:
            # 导航背景 - left:9, top:9, width:270, height:1006
            nav_bg = self._load_image(self.navigation_dir, "Background.png", (270, 1006))
            if nav_bg:
                self.images["nav_bg"] = nav_bg
                self.canvas.create_image(9, 9, image=nav_bg, anchor="nw")

            # 当前选中的导航按钮（默认为"可视化"）
            self.current_nav_button = "virtualization"

            # 导航按钮定义
            nav_buttons = [
                ("Home_Button.png", 39, 231, self.home, "home"),
                ("Preprocessing_Button.png", 39, 323, self.preprocessing, "preprocessing"),
                ("Feature_Extraction_Button.png", 39, 415, self.feature_extraction, "feature_extraction"),
                ("Statistical_Analysis_Button.png", 39, 507, self.statistical_analysis, "statistical_analysis"),
                ("Virtualization_Button.png", 39, 599, self.virtualization, "virtualization"),
            ]

            # 存储按钮ID和对应信息的字典
            self.nav_button_ids = {}

            for filename, x, y, command, button_id in nav_buttons:
                # 默认使用未选中状态的图片
                btn = self._load_image(self.nav_buttons_unselected_dir, filename, (213, 62))
                if btn:
                    key = f"nav_{filename}"
                    self.images[key] = btn
                    img_id = self.canvas.create_image(x, y, image=btn, anchor="nw")

                    # 存储按钮信息
                    self.nav_button_ids[img_id] = {
                        "id": button_id,
                        "command": command,
                        "x": x,
                        "y": y,
                        "filename": filename
                    }

                    # 绑定点击事件
                    self.canvas.tag_bind(img_id, "<Button-1>", lambda e, iid=img_id: self.on_nav_button_click(iid))

                    # 添加鼠标悬停效果
                    self.canvas.tag_bind(img_id, "<Enter>", lambda e: self.canvas.config(cursor="hand2"))
                    self.canvas.tag_bind(img_id, "<Leave>", lambda e: self.canvas.config(cursor=""))

            # 设置默认选中"可视化"按钮
            self.highlight_nav_button("virtualization")

        # ============ 2. 顶部文件选择区 ============
        # 文件选择背景 - left:content_start_x, top:9, width:1134, height:201
        file_bg = self._load_image(self.resource_dir, "File_Selection_Background.png", (1134, 201))
        if file_bg:
            self.images["file_bg"] = file_bg
            self.canvas.create_image(self.content_start_x, 9, image=file_bg, anchor="nw")

        # 地址栏 - left:510, top:33, width:889, height:55
        # 注意：地址栏相对于文件选择背景的位置是固定的
        addr_bar = self._load_image(self.resource_dir, "File_Address_Bar.png", (889, 55))
        if addr_bar:
            # 地址栏在文件选择背景内的偏移是 (212, 24)
            addr_x = self.content_start_x + 212
            addr_y = 33
            self.images["addr_bar"] = addr_bar
            self.canvas.create_image(addr_x, addr_y, image=addr_bar, anchor="nw")

        # 直接在 Canvas 上创建文字（完全透明，不会覆盖任何东西）
        self.file_path_text_id = self.canvas.create_text(
            self.content_start_x + 222,  # x坐标 (212 + 10)
            33 + 55 // 2,  # y坐标（垂直居中）
            text="",
            font=("微软雅黑", 11),
            fill="#333333",
            anchor="w",  # 左对齐
            width=840  # 限制宽度，防止溢出
        )

        # ============ 3. 顶部四个操作按钮 ============
        top_buttons = [
            ("browse_button.png", 329, 120, self.browse_file),
            ("load_button.png", 576, 120, self.load_current_file),
            ("demonstration_button.png", 823, 120, self.load_demo_data),
            ("reset_button.png", 1070, 120, self.reset_data),
        ]

        for filename, design_x, design_y, command in top_buttons:
            # 设计稿中的x坐标是相对于文件选择背景的
            actual_x = self.content_start_x + (design_x - 298)  # 298是设计稿中文件选择背景的起始x
            btn_img = self._load_image(self.buttons_loading_dir, filename, (211, 55))
            if btn_img:
                # 直接使用canvas.create_image，不创建Button控件
                img_id = self.canvas.create_image(actual_x, design_y, image=btn_img, anchor="nw")
                self.images[f"btn_{filename}"] = btn_img
                # 绑定点击事件
                self.canvas.tag_bind(img_id, "<Button-1>", lambda e, c=command: c())
                # 添加鼠标悬停效果
                self.canvas.tag_bind(img_id, "<Enter>", lambda e: self.canvas.config(cursor="hand2"))
                self.canvas.tag_bind(img_id, "<Leave>", lambda e: self.canvas.config(cursor=""))

        # ============ 4. 功能区背景 ============
        func_bg = self._load_image(self.resource_dir, "Function_Selection.png", (516, 775))
        if func_bg:
            self.images["func_bg"] = func_bg
            self.canvas.create_image(self.content_start_x, 240, image=func_bg, anchor="nw")

        # ============ 5. 六个功能卡片 ============
        function_cards = [
            ("waveform_button.png", self.open_signal_view, 312, 331),
            ("analysis_button.png", self.open_stats_view, 559, 331),
            ("barchart_button.png", self.open_bar_view, 312, 557),
            ("time_frequency_button.png", self.open_time_frequency, 559, 557),
            ("topographic_map_button.png", self.open_topography, 312, 783),
            ("feature_map_button.png", self.open_feature_view, 559, 783),
        ]

        for filename, command, design_x, design_y in function_cards:
            # 设计稿中的x坐标是相对于文件选择背景的
            actual_x = self.content_start_x + (design_x - 298)
            card_img = self._load_image(self.buttons_function_dir, filename, (222, 205))
            if card_img:
                key = f"card_{filename}"
                self.images[key] = card_img
                img_id = self.canvas.create_image(actual_x, design_y, image=card_img, anchor="nw")
                # 绑定点击事件
                self.canvas.tag_bind(img_id, "<Button-1>", lambda e, c=command: c())
                # 添加鼠标悬停效果
                self.canvas.tag_bind(img_id, "<Enter>", lambda e: self.canvas.config(cursor="hand2"))
                self.canvas.tag_bind(img_id, "<Leave>", lambda e: self.canvas.config(cursor=""))

        # ============ 6. 右侧数据信息面板 ============
        info_img = self._load_image(self.resource_dir, "Data_Information.png", (583, 451))
        if info_img:
            # 信息面板在文件选择背景内的偏移是 (539, 231)
            info_x = self.content_start_x + 539
            info_y = 240
            self.images["info_bg"] = info_img
            self.canvas.create_image(info_x, info_y, image=info_img, anchor="nw")

        # 数据信息内容（动态更新）- 放在面板中央
        self.info_text_id = self.canvas.create_text(
            info_x + 583 // 2,  # 水平居中
            info_y + 451 // 2,  # 垂直居中
            text="请加载数据文件或使用演示数据",
            width=583 - 80,  # 留些边距
            font=("微软雅黑", 12),
            fill="#333333"
        )

        # ============ 7. 右下品牌区 ============
        # 智融脑机 - left:896, top:754, width:502, height:133
        znrj = self._load_image(self.resource_dir, "znrj.png", (502, 133))
        if znrj:
            # 品牌区在文件选择背景内的偏移是 (598, 745)
            znrj_x = self.content_start_x + 598
            znrj_y = 754
            self.images["znrj"] = znrj
            self.canvas.create_image(znrj_x, znrj_y, image=znrj, anchor="nw")

        # dmxfnd - left:996, top:870, width:235, height:56
        dmxfnd = self._load_image(self.resource_dir, "dmxfnd.png", (235, 56))
        if dmxfnd:
            dmxfnd_x = self.content_start_x + 698
            dmxfnd_y = 870
            self.images["dmxfnd"] = dmxfnd
            self.canvas.create_image(dmxfnd_x, dmxfnd_y, image=dmxfnd, anchor="nw")

        # mtbci - left:1203, top:884, width:209, height:46
        mtbci = self._load_image(self.resource_dir, "mtbci.png", (209, 46))
        if mtbci:
            mtbci_x = self.content_start_x + 905
            mtbci_y = 884
            self.images["mtbci"] = mtbci
            self.canvas.create_image(mtbci_x, mtbci_y, image=mtbci, anchor="nw")

        # 初次刷新信息
        self.update_file_info()

    def on_nav_button_click(self, img_id):
        """导航按钮点击事件 - 仅在独立运行时有效"""
        if not self.show_navigation:
            return

        if img_id in self.nav_button_ids:
            button_info = self.nav_button_ids[img_id]
            button_id = button_info["id"]

            # 高亮当前按钮
            self.highlight_nav_button(button_id)

            # 执行对应的命令
            button_info["command"]()

    def highlight_nav_button(self, button_id):
        """高亮指定的导航按钮 - 仅在独立运行时有效"""
        if not self.show_navigation:
            return

        # 按钮ID和文件名的对应关系
        button_files = {
            "home": "Home_Button.png",
            "preprocessing": "Preprocessing_Button.png",
            "feature_extraction": "Feature_Extraction_Button.png",
            "statistical_analysis": "Statistical_Analysis_Button.png",
            "virtualization": "Virtualization_Button.png",
        }

        # 遍历所有按钮
        for img_id, info in self.nav_button_ids.items():
            if info["id"] == button_id:
                # 当前选中的按钮：使用选中状态的图片
                selected_img = self._load_image(
                    self.nav_buttons_selected_dir,
                    button_files[button_id],
                    (213, 62)
                )
                if selected_img:
                    key = f"nav_selected_{button_id}"
                    self.images[key] = selected_img
                    self.canvas.itemconfig(img_id, image=selected_img)
            else:
                # 未选中的按钮：使用未选中状态的图片
                unselected_img = self._load_image(
                    self.nav_buttons_unselected_dir,
                    button_files[info["id"]],
                    (213, 62)
                )
                if unselected_img:
                    key = f"nav_unselected_{info['id']}"
                    self.images[key] = unselected_img
                    self.canvas.itemconfig(img_id, image=unselected_img)

    def _load_image(self, directory, filename, size=None):
        """加载图片并缩放到指定尺寸"""
        if not PIL_AVAILABLE:
            return None

        img_path = Path(directory) / filename
        if not img_path.exists():
            print(f"⚠️ 图片不存在: {img_path}")
            return None

        try:
            img = Image.open(img_path)
            if size:
                img = img.resize(size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"❌ 加载图片失败 {filename}: {e}")
            return None

    # ============ 导航按钮功能（仅在独立运行时有效） ============
    def home(self):
        if not self.show_navigation:
            return
        print("首页")
        messagebox.showinfo("提示", "首页功能待实现")

    def preprocessing(self):
        if not self.show_navigation:
            return
        print("预处理")
        messagebox.showinfo("提示", "预处理功能待实现")

    def feature_extraction(self):
        if not self.show_navigation:
            return
        print("特征提取")
        messagebox.showinfo("提示", "特征提取功能待实现")

    def statistical_analysis(self):
        if not self.show_navigation:
            return
        print("数据分析")
        messagebox.showinfo("提示", "数据分析功能待实现")

    def virtualization(self):
        if not self.show_navigation:
            return
        print("可视化")
        messagebox.showinfo("提示", "可视化功能待实现")

    # ============ 文件/数据操作 ============
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
            self.update_file_info()

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
            "meta": {"subject_id": "demo", "session_id": "session1", "task": "rest", "modality": ["EEG", "fNIRS"]},
            "signal": {
                "EEG": {
                    "data": np.array([np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t)) for _ in range(8)]),
                    "sampling_rate": fs,
                    "channel_names": [f"EEG_{i}" for i in range(8)],
                    "unit": "uV"
                },
                "fNIRS": {
                    "data": np.array([np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(len(t)) for _ in range(4)]),
                    "sampling_rate": fs,
                    "channel_names": [f"NIRS_{i}" for i in range(4)],
                    "unit": "uM"
                }
            },
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
        """更新文件信息显示"""
        # 更新文件路径显示
        if self.file_path_var.get():
            self.canvas.itemconfig(self.file_path_text_id, text=self.file_path_var.get())
        else:
            self.canvas.itemconfig(self.file_path_text_id, text="")

        # 更新右侧信息面板（原来的代码保持不变）
        if not self.data_dict:
            info = "请加载数据文件或使用演示数据"
        else:
            meta = self.data_dict.get("meta", {})
            subject = meta.get("subject_id", "unknown")
            task = meta.get("task", "unknown")
            modalities = meta.get("modality", [])
            lines = [f"被试: {subject}", f"任务: {task}", f"模态: {', '.join(modalities)}", ""]

            for mod, sig in self.data_dict.get("signal", {}).items():
                if isinstance(sig, dict) and 'data' in sig:
                    data = sig['data']
                    n_ch = data.shape[0] if hasattr(data, 'shape') and len(data.shape) > 1 else 1
                    fs = sig.get('sampling_rate', '?')
                    lines.append(f"{mod}: {n_ch}通道, {fs}Hz")

            info = "\n".join(lines)

        try:
            self.canvas.itemconfig(self.info_text_id, text=info)
        except Exception:
            pass

    # ============ 打开各视图 ============
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
            view = StatsView(window, data_dict=self.data_dict)
            view.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("错误", f"打开统计视图失败:\n{str(e)}")

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
            self.ensure_pyqt_app()
            self.tf_window = TimeFrequencyView(self.data_dict, modality=None)
            self.tf_window.show()
        except Exception as e:
            messagebox.showerror("错误", f"打开时频分析视图失败:\n{str(e)}")

    def open_topography(self):
        if not self.data_dict:
            messagebox.showwarning("警告", "请先加载数据")
            return
        try:
            self.ensure_pyqt_app()
            self.topo_window = TopographyView(self.data_dict, modality=None)
            self.topo_window.show()
        except Exception as e:
            messagebox.showerror("错误", f"打开地形图视图失败:\n{str(e)}")

    def open_feature_view(self):
        if not self.data_dict:
            messagebox.showwarning("警告", "请先加载数据")
            return
        try:
            from core.visualizer.feature_view import show_feature_view
            self.feature_window = show_feature_view(self, self.data_dict)
        except Exception as e:
            messagebox.showerror("错误", f"打开特征视图失败:\n{str(e)}")

    def ensure_pyqt_app(self):
        try:
            from PyQt5.QtWidgets import QApplication
            if not QApplication.instance():
                self.qt_app = QApplication(sys.argv)
        except Exception:
            pass


if __name__ == "__main__":
    # 独立运行时显示导航栏
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    root = tk.Tk()
    root.title("NeuroPioneer 可视化面板")

    # 固定尺寸，禁止调整窗口大小
    root.geometry("1440x1024")
    root.resizable(False, False)

    # 设置窗口背景色
    root.configure(bg="#FFFFFF")

    # 创建应用实例 - 独立运行时显示导航栏
    app = NeuroPioneerPanel(root, show_navigation=True)
    app.pack(fill=tk.BOTH, expand=True)

    # 进入主循环
    root.mainloop()