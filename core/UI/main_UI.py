#!/usr/bin/env python3
"""
main_ui.py
智融脑机 - 主界面 (tkinter版本)
完全保持原主界面的布局和功能
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
from tkinter import ttk, messagebox

# 导入功能面板
from core.UI.panel.visualization_panel import NeuroPioneerPanel
from core.UI.panel.preprocessing_panel import PreprocessingApp

try:
    from PIL import Image, ImageTk

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("警告: PIL 未安装，无法加载 PNG 图片，请安装 Pillow 库。")


class MainUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("智融脑机 - 主界面")
        self.geometry("1440x1024")
        self.resizable(False, False)

        # 动态获取项目根目录
        self.project_root = project_root
        print(f"项目根目录: {self.project_root}")

        # 资源目录
        self.navigation_dir = self.project_root / "core" / "UI" / "UI_resource" / "Navigation"
        self.homepage_dir = self.project_root / "core" / "UI" / "UI_resource" / "Homepage"

        # 导航按钮目录
        self.nav_buttons_unselected_dir = self.navigation_dir / "Buttons" / "Unselected"
        self.nav_buttons_selected_dir = self.navigation_dir / "Buttons" / "Selected"

        # 存储图片引用
        self.images = {}

        # 当前选中的按钮（默认为"首页"）
        self.current_button = "首页"

        # 初始化UI
        self.setup_ui()

    def setup_ui(self):
        """初始化UI - 完全保持原主界面布局"""
        # 创建Canvas作为主画布
        self.canvas = tk.Canvas(
            self,
            width=1440,
            height=1024,
            highlightthickness=0,
            bg="#FFFFFF"
        )
        self.canvas.pack()

        # ============ 1. 左侧导航背景 ============
        # 导航背景 - left:9, top:9, width:270, height:1006
        nav_bg = self._load_image(self.navigation_dir, "Background.png", (270, 1006))
        if nav_bg:
            self.images["nav_bg"] = nav_bg
            self.canvas.create_image(9, 9, image=nav_bg, anchor="nw")

        # ============ 2. 首页所有静态图片 ============
        # 严格按照原坐标放置
        homepage_images = [
            ("Files.png", 899, 267),  # 610+289=899
            ("Group.png", 899, 813),  # 610+289=899
            ("Project_Name.png", 296, 142),  # 7+289=296
            ("Search.png", 899, 182),  # 610+289=899
            ("User.png", 296, 665),  # 7+289=296
            ("Welcome.png", 296, 7)  # 7+289=296
        ]

        for filename, x, y in homepage_images:
            img = self._load_image(self.homepage_dir, filename)
            if img:
                key = f"home_{filename}"
                self.images[key] = img
                self.canvas.create_image(x, y, image=img, anchor="nw")

        # ============ 3. 创建导航按钮 ============
        self.create_nav_buttons()

        # ============ 4. 右侧面板容器 ============
        # 创建一个容器Frame放在右侧区域，但不覆盖首页图片
        self.panel_container = tk.Frame(
            self,
            bg="#FFFFFF",
            highlightthickness=0
        )
        # 放在右侧区域，但初始隐藏
        self.panel_container.place(x=289, y=0, width=1151, height=1024)
        self.panel_container.place_forget()

        # ============ 5. 初始化各个功能面板 ============
        self.init_panels()

    def create_nav_buttons(self):
        """创建导航按钮 - 保持原位置和样式"""
        nav_buttons = [
            ("首页", "Home_Button.png", 39, 231),
            ("预处理", "Preprocessing_Button.png", 39, 323),
            ("特征提取", "Feature_Extraction_Button.png", 39, 415),
            ("统计分析", "Statistical_Analysis_Button.png", 39, 507),
            ("可视化", "Virtualization_Button.png", 39, 599),
        ]

        self.nav_button_ids = {}

        for text, filename, x, y in nav_buttons:
            # 根据当前选中状态决定使用哪张图片
            if text == self.current_button:
                btn = self._load_image(self.nav_buttons_selected_dir, filename, (213, 62))
            else:
                btn = self._load_image(self.nav_buttons_unselected_dir, filename, (213, 62))

            if btn:
                key = f"nav_{text}"
                self.images[key] = btn
                img_id = self.canvas.create_image(x, y, image=btn, anchor="nw")

                # 存储按钮信息
                self.nav_button_ids[img_id] = {
                    "text": text,
                    "filename": filename
                }

                # 绑定点击事件
                self.canvas.tag_bind(img_id, "<Button-1>", lambda e, iid=img_id: self.on_nav_click(iid))
                self.canvas.tag_bind(img_id, "<Enter>", lambda e: self.canvas.config(cursor="hand2"))
                self.canvas.tag_bind(img_id, "<Leave>", lambda e: self.canvas.config(cursor=""))

    def init_panels(self):
        """初始化所有功能面板"""
        self.panels = {}

        # 预处理面板
        try:
            self.panels["预处理"] = PreprocessingApp(self.panel_container)
        except Exception as e:
            print(f"加载预处理面板失败: {e}")
            self.panels["预处理"] = None

        # 可视化面板 - 关键修改：加上 show_navigation=False
        try:
            self.panels["可视化"] = NeuroPioneerPanel(
                self.panel_container,
                show_navigation=False  # 👈👈👈 加上这个参数
            )
        except Exception as e:
            print(f"加载可视化面板失败: {e}")
            self.panels["可视化"] = None

        # 其他面板暂未实现
        self.panels["特征提取"] = None
        self.panels["统计分析"] = None

        # 预加载面板但不显示
        for panel in self.panels.values():
            if panel:
                panel.place_forget()

    def on_nav_click(self, img_id):
        """导航按钮点击事件"""
        if img_id not in self.nav_button_ids:
            return

        button_info = self.nav_button_ids[img_id]
        button_text = button_info["text"]

        if button_text == self.current_button:
            return

        print(f"切换到: {button_text}")

        # 更新按钮状态
        self.update_button_states(button_text)

        # 处理面板切换
        if button_text == "首页":
            # 显示首页图片，隐藏面板容器
            self.show_home()
        else:
            # 隐藏首页图片，显示对应的功能面板
            self.show_panel(button_text)

        # 更新当前按钮
        self.current_button = button_text

    def update_button_states(self, selected_text):
        """更新所有按钮的状态"""
        for img_id, info in self.nav_button_ids.items():
            if info["text"] == selected_text:
                # 选中状态
                btn = self._load_image(
                    self.nav_buttons_selected_dir,
                    info["filename"],
                    (213, 62)
                )
                if btn:
                    key = f"nav_sel_{info['text']}"
                    self.images[key] = btn
                    self.canvas.itemconfig(img_id, image=btn)
            else:
                # 未选中状态
                btn = self._load_image(
                    self.nav_buttons_unselected_dir,
                    info["filename"],
                    (213, 62)
                )
                if btn:
                    key = f"nav_unsel_{info['text']}"
                    self.images[key] = btn
                    self.canvas.itemconfig(img_id, image=btn)

    def show_home(self):
        """显示首页"""
        # 隐藏面板容器
        self.panel_container.place_forget()

        # 显示所有首页图片
        for key in self.images:
            if key.startswith("home_"):
                # 图片已经在Canvas上，不需要额外操作
                pass

    def show_panel(self, panel_name):
        """显示指定的功能面板"""
        # 隐藏所有首页图片
        for key in self.images:
            if key.startswith("home_"):
                # 这些图片需要隐藏，但Canvas没有直接的hide方法
                # 我们可以用panel_container覆盖它们
                pass

        # 显示面板容器
        self.panel_container.place(x=289, y=0, width=1151, height=1024)

        # 隐藏容器内所有面板
        for panel in self.panels.values():
            if panel:
                panel.place_forget()

        # 显示选中的面板
        if panel_name in self.panels and self.panels[panel_name]:
            self.panels[panel_name].place(x=0, y=0, width=1151, height=1024)
        else:
            # 显示未实现提示
            self.show_not_implemented(panel_name)

    def show_not_implemented(self, panel_name):
        """显示未实现提示"""
        temp_frame = tk.Frame(self.panel_container, bg="#FFFFFF")
        temp_frame.place(x=0, y=0, width=1151, height=1024)

        tk.Label(
            temp_frame,
            text=f"{panel_name}功能开发中",
            font=("微软雅黑", 24),
            bg="#FFFFFF",
            fg="#999999"
        ).place(relx=0.5, rely=0.5, anchor="center")

        # 3秒后自动销毁
        self.after(3000, temp_frame.destroy)

    def _load_image(self, directory, filename, size=None):
        """加载图片"""
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


if __name__ == "__main__":
    # 尝试设置DPI感知
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    app = MainUI()
    app.mainloop()