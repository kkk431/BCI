import sys
from pathlib import Path

# 将项目根目录（即core的上一级目录）动态添加到 sys.path，确保无论从哪个位置运行都能正确导入核心模块
start_path = Path(__file__).resolve().parent
for parent in [start_path] + list(start_path.parents):
    if parent.name == 'core':
        project_root = parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            print(f"已将项目根目录 {project_root} 添加到 sys.path")
        break
else:
    raise RuntimeError("未找到名为 'core' 的目录")
    
import tkinter as tk
import os
import re
import time

from core.UI.panel.extraction_panel import ExtractionApp

# --- 视觉配置 ---
COLOR_TAB_BAR_BG = "#c0c0c0"
COLOR_TAB_INACTIVE = "#d0d0d0"
COLOR_TAB_ACTIVE = "white"
COLOR_CONTENT_BG = "white"

# 图片资源路径（使用 os.path.join 自动处理分隔符）
LOGO_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "logo.png")
MENU_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "menu.png")
CONTENT_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "Content.png")
BUTTON1_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "button1.png")
BUTTON2_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "button2.png")
BUTTON3_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "button3.png")
BUTTON4_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "button4.png")
SIGNIN_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "SignIn.png")
BACKGROUND_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "background.png")

# ========== 启动时预加载所有可视化模块 ==========
print("正在预加载可视化模块...")
preload_start = time.time()

# 预导入所有需要的库
import matplotlib
matplotlib.use('TkAgg')  # 设置后端

# 预导入可视化模块

preload_time = time.time() - preload_start
print(f"预加载完成，耗时: {preload_time:.2f}秒")
# ===========================================

class NeuroPioneerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智融脑机 - 大模型赋能多模态BCI平台")
        self.root.geometry("1100x700")  # 临时大小，稍后会被主页背景图覆盖
        self.root.configure(bg=COLOR_CONTENT_BG)

        # 标签管理
        self.tabs = {}               # {tab_id: {"frame":..., "tab_btn":..., "base_name":..., "display_name":...}}
        self.active_tab_id = None
        self.tab_history = []         # 记录标签激活顺序，用于关闭时返回上一个

        # 用于保持主页图片的引用
        self.home_images = []

        self.setup_ui()
        self.show_homepage()

    def setup_ui(self):
        """构建顶栏和内容容器"""
        self.header_frame = tk.Frame(self.root, bg=COLOR_TAB_BAR_BG, height=45)
        self.header_frame.pack(side="top", fill="x")
        self.header_frame.pack_propagate(False)

        self.tab_container = tk.Frame(self.header_frame, bg=COLOR_TAB_BAR_BG)
        self.tab_container.pack(side="left", fill="y")

        self.add_btn = tk.Label(self.header_frame, text="+", bg=COLOR_TAB_BAR_BG,
                                 fg="black", font=("Arial", 16), width=4, cursor="hand2")
        self.add_btn.pack(side="left", fill="y")
        self.add_btn.bind("<Button-1>", lambda e: self.show_launcher_popup())
        self.add_btn.bind("<Enter>", lambda e: e.widget.config(bg="#e0e0e0"))
        self.add_btn.bind("<Leave>", lambda e: e.widget.config(bg=COLOR_TAB_BAR_BG))

        self.main_container = tk.Frame(self.root, bg=COLOR_CONTENT_BG)
        self.main_container.pack(side="top", expand=True, fill="both")

    def create_tab_widget(self, display_name, tab_id, can_close=True):
        """创建单个标签页 UI，并绑定对应 tab_id 的事件"""
        tab_frame = tk.Frame(self.tab_container, bg=COLOR_TAB_INACTIVE, bd=0)
        tab_frame.pack(side="left", fill="y", padx=(0, 1))

        line = tk.Frame(tab_frame, bg="black", width=1)
        line.pack(side="left", fill="y")

        lbl = tk.Label(tab_frame, text=display_name, bg=COLOR_TAB_INACTIVE, padx=15,
                       font=("微软雅黑", 9), cursor="hand2")
        lbl.pack(side="left", fill="y")
        lbl.bind("<Button-1>", lambda e, tid=tab_id: self.switch_to_tab(tid))
        lbl.bind("<Button-3>", lambda e, tid=tab_id: self.show_context_menu(e, tid))

        if can_close:
            close_lbl = tk.Label(tab_frame, text="✕", bg=COLOR_TAB_INACTIVE,
                                 fg="#666", font=("Arial", 8), cursor="hand2")
            close_lbl.pack(side="left", fill="y", padx=(0, 10))
            close_lbl.bind("<Button-1>", lambda e, tid=tab_id: self.close_tab(tid))
            close_lbl.bind("<Enter>", lambda e: e.widget.config(fg="red"))
            close_lbl.bind("<Leave>", lambda e: e.widget.config(fg="#666"))

        return tab_frame

    def show_homepage(self):
        if "Homepage" in self.tabs:
            old_data = self.tabs["Homepage"]
            old_data["frame"].destroy()
            old_data["tab_btn"].destroy()
            del self.tabs["Homepage"]
            while "Homepage" in self.tab_history:
                self.tab_history.remove("Homepage")

        # 清空之前的图片引用
        self.home_images.clear()

        # 加载背景图以获取尺寸
        try:
            from PIL import Image, ImageTk
            bg_img = Image.open(BACKGROUND_PATH)
            bg_width, bg_height = bg_img.size
            # 调整窗口大小：背景高度 + 标签栏高度45
            self.root.geometry(f"{bg_width}x{bg_height + 45}")
            self.root.resizable(False, False)  # 禁止缩放，保持绝对坐标
        except Exception as e:
            print(f"背景图片加载失败: {e}")
            # 降级处理：使用默认大小
            bg_width, bg_height = 1100, 700
            self.root.geometry(f"{bg_width}x{bg_height + 45}")
            bg_img = None

        # 创建 Canvas 作为主页容器
        home_canvas = tk.Canvas(self.main_container, bg=COLOR_CONTENT_BG, highlightthickness=0,
                                width=bg_width, height=bg_height)
        home_canvas.pack_propagate(False)  # 禁止自动调整大小
        home_canvas.config(width=bg_width, height=bg_height)

        # 放置背景图片
        if bg_img:
            try:
                self.bg_photo = ImageTk.PhotoImage(bg_img)
                # 背景图置于底层
                home_canvas.create_image(0, 0, anchor="nw", image=self.bg_photo)
                self.home_images.append(self.bg_photo)
            except Exception as e:
                print(f"背景图片显示失败: {e}")

        # 辅助函数：加载图片并在 canvas 上创建 image 项，返回图片对象和 canvas 项 ID
        def add_image(path, x, y, tag=None):
            try:
                img = Image.open(path)
                photo = ImageTk.PhotoImage(img)
                item_id = home_canvas.create_image(x, y, anchor="nw", image=photo, tag=tag)
                self.home_images.append(photo)  # 保持引用
                return photo, item_id
            except Exception as e:
                print(f"图片加载失败 {path}: {e}")
                return None, None

        # ① logo
        add_image(LOGO_PATH, 56, 24)

        # ② 菜单栏（暂不绑定事件）
        add_image(MENU_PATH, 185, 31)

        # ③ 文字介绍区域
        add_image(CONTENT_PATH, 91, 250)

        # ④ 四个功能按钮（绑定打开对应标签页）
        button_info = [
            (BUTTON1_PATH, 89, 557, "Preprocessing"),
            (BUTTON2_PATH, 249, 557, "Extraction"),
            (BUTTON3_PATH, 436, 557, "Analysis"),
            (BUTTON4_PATH, 625, 557, "Visualization")
        ]
        for path, x, y, name in button_info:
            _, item_id = add_image(path, x, y, tag=f"btn_{name}")
            if item_id:
                # 绑定点击事件
                home_canvas.tag_bind(item_id, "<Button-1>", lambda e, n=name: self.open_functional_tab(n))
                # 鼠标悬停时改变光标样式
                home_canvas.tag_bind(item_id, "<Enter>", lambda e: home_canvas.config(cursor="hand2"))
                home_canvas.tag_bind(item_id, "<Leave>", lambda e: home_canvas.config(cursor=""))

        # ⑤ 登录键（暂不绑定事件）
        add_image(SIGNIN_PATH, 1274, 24)

        # 创建 Homepage 标签按钮（不可关闭）
        tab_btn = self.create_tab_widget("Homepage", tab_id="Homepage", can_close=False)
        self.tabs["Homepage"] = {
            "frame": home_canvas,  # 注意这里存储的是 Canvas，但它是 Frame 的子类，可以正常 place
            "tab_btn": tab_btn,
            "base_name": "Homepage",
            "display_name": "Homepage"
        }

        self.switch_to_tab("Homepage")

    def _get_next_number_for_base(self, base_name):
        """获取当前 base_name 类型标签的下一个编号"""
        max_num = 0
        for data in self.tabs.values():
            if data.get("base_name") == base_name:
                display = data["display_name"]
                match = re.search(r'\((\d+)\)', display)
                if match:
                    num = int(match.group(1))
                    max_num = max(max_num, num)
        return max_num + 1

    def open_functional_tab(self, base_name):
        """打开一个新的功能标签页（总是新建）"""
        if base_name == "Visualization":
            # 打开可视化集成面板
            self.open_visualization_tab()
        elif base_name == "Extraction":
            # 打开特征提取面板
            self.open_extraction_tab()
        else:
            # 其他功能保持原样
            next_num = self._get_next_number_for_base(base_name)
            display_name = f"{base_name} ({next_num})"
            tab_id = f"{base_name}_{next_num}"

            new_frame = tk.Frame(self.main_container, bg=COLOR_CONTENT_BG)
            tk.Label(new_frame, text=f"{base_name.lower()} 界面",
                     font=("微软雅黑", 24), fg="#333", bg=COLOR_CONTENT_BG).place(relx=0.5, rely=0.5, anchor="center")

            tab_btn = self.create_tab_widget(display_name, tab_id=tab_id)
            self.tabs[tab_id] = {"frame": new_frame, "tab_btn": tab_btn,
                                 "base_name": base_name, "display_name": display_name}

            self.switch_to_tab(tab_id)

    def open_extraction_tab(self):
        """打开特征提取标签页"""
        next_num = self._get_next_number_for_base("Extraction")
        display_name = f"Extraction ({next_num})"
        tab_id = f"Extraction_{next_num}"

        # 创建框架
        new_frame = tk.Frame(self.main_container, bg=COLOR_CONTENT_BG)

        # 显示加载提示
        loading_label = tk.Label(new_frame, text="正在加载特征提取模块...",
                                font=("微软雅黑", 14), fg="#666", bg=COLOR_CONTENT_BG)
        loading_label.place(relx=0.5, rely=0.5, anchor="center")
        new_frame.update()

        # 创建标签按钮
        tab_btn = self.create_tab_widget(display_name, tab_id=tab_id)

        # 存储标签信息
        self.tabs[tab_id] = {
            "frame": new_frame,
            "tab_btn": tab_btn,
            "base_name": "Extraction",
            "display_name": display_name
        }

        # 切换到新标签页
        self.switch_to_tab(tab_id)

        # 延迟加载，不阻塞UI
        def load_extraction():
            try:
                # 移除加载提示
                loading_label.destroy()

                # 创建特征提取面板
                extraction_app = ExtractionApp(new_frame)
                extraction_app.pack(fill=tk.BOTH, expand=True)

                print("特征提取面板加载完成")
            except Exception as e:
                loading_label.config(text=f"加载失败: {str(e)}", fg="red")
                import traceback
                traceback.print_exc()

        new_frame.after(10, load_extraction)

    def open_visualization_tab(self):
        """打开可视化标签页（带加载动画）"""
        next_num = self._get_next_number_for_base("Visualization")
        display_name = f"Visualization ({next_num})"
        tab_id = f"Visualization_{next_num}"

        # 创建框架
        new_frame = tk.Frame(self.main_container, bg=COLOR_CONTENT_BG)

        # 显示加载提示
        loading_label = tk.Label(new_frame, text="正在加载可视化模块...",
                                 font=("微软雅黑", 14), fg="#666", bg=COLOR_CONTENT_BG)
        loading_label.place(relx=0.5, rely=0.5, anchor="center")

        # 更新UI显示加载提示
        new_frame.update()

        # 创建标签按钮
        tab_btn = self.create_tab_widget(display_name, tab_id=tab_id)

        # 存储标签信息
        self.tabs[tab_id] = {
            "frame": new_frame,
            "tab_btn": tab_btn,
            "base_name": "Visualization",
            "display_name": display_name
        }

        # 切换到新标签页
        self.switch_to_tab(tab_id)

        # 使用after延迟加载，不阻塞UI
        def load_visualization():
            try:
                import time
                start = time.time()

                # 导入模块
                from core.UI.panel.visualization_panel import ModernVisualizationPanel as VisualizationPanel

                # 移除加载提示
                loading_label.destroy()

                # 创建可视化面板
                vis_panel = VisualizationPanel(new_frame)
                vis_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                elapsed = time.time() - start
                print(f"可视化面板加载完成，耗时: {elapsed:.2f}秒")

            except Exception as e:
                loading_label.config(text=f"加载失败: {str(e)}", fg="red")

        # 延迟10ms后开始加载，让UI先显示加载提示
        new_frame.after(10, load_visualization)

    def switch_to_tab(self, tab_id):
        """切换到指定 ID 的标签页，并更新历史记录"""
        if tab_id not in self.tabs:
            return

        self.active_tab_id = tab_id
        if tab_id in self.tab_history:
            self.tab_history.remove(tab_id)
        self.tab_history.append(tab_id)

        for tid, data in self.tabs.items():
            if tid == tab_id:
                data["frame"].place(relwidth=1, relheight=1)
                data["tab_btn"].config(bg=COLOR_TAB_ACTIVE)
                for child in data["tab_btn"].winfo_children():
                    if isinstance(child, tk.Label):
                        child.config(bg=COLOR_TAB_ACTIVE)
            else:
                data["frame"].place_forget()
                data["tab_btn"].config(bg=COLOR_TAB_INACTIVE)
                for child in data["tab_btn"].winfo_children():
                    if isinstance(child, tk.Label):
                        child.config(bg=COLOR_TAB_INACTIVE)

    def close_tab(self, tab_id):
        """关闭指定 ID 的标签页，并自动切换到上一个激活的标签（如果是当前标签）"""
        if tab_id == "Homepage":
            return
        data = self.tabs.get(tab_id)
        if not data:
            return

        while tab_id in self.tab_history:
            self.tab_history.remove(tab_id)

        data["frame"].destroy()
        data["tab_btn"].destroy()
        del self.tabs[tab_id]

        if self.active_tab_id == tab_id:
            if self.tab_history:
                target_id = self.tab_history[-1]
            else:
                target_id = "Homepage"
            self.switch_to_tab(target_id)

    def show_launcher_popup(self):
        """加号点击弹出模块列表，点击后新建标签"""
        popup = tk.Toplevel(self.root)
        popup.overrideredirect(True)
        x = self.add_btn.winfo_rootx()
        y = self.add_btn.winfo_rooty() + 40
        popup.geometry(f"180x200+{x}+{y}")
        popup.config(bg="white", highlightthickness=1, highlightbackground="#ccc")

        tk.Label(popup, text="快速跳转", bg="#eee", font=("微软雅黑", 9, "bold")).pack(fill="x")

        others = ["Preprocessing", "Extraction", "Analysis", "Visualization", "Settings"]
        for o in others:
            lbl = tk.Label(popup, text=o, bg="white", pady=5, cursor="hand2")
            lbl.pack(fill="x")
            lbl.bind("<Button-1>", lambda e, name=o: [self.open_functional_tab(name), popup.destroy()])
            lbl.bind("<Enter>", lambda e: e.widget.config(bg="#f0f0f0"))
            lbl.bind("<Leave>", lambda e: e.widget.config(bg="white"))

        popup.bind("<FocusOut>", lambda e: popup.destroy())
        popup.focus_set()

    def show_context_menu(self, event, tab_id):
        """右键菜单：关闭其他 / 全部关闭"""
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="关闭其他标签", command=lambda: self.close_others(tab_id))
        menu.add_command(label="全部关闭", command=self.close_all_functional)
        menu.post(event.x_root, event.y_root)

    def close_others(self, keep_tab_id):
        """关闭除 keep_tab_id 和 Homepage 外的所有标签"""
        ids = list(self.tabs.keys())
        for tid in ids:
            if tid != keep_tab_id and tid != "Homepage":
                self.close_tab(tid)
        self.switch_to_tab(keep_tab_id)

    def close_all_functional(self):
        """关闭所有功能标签（保留 Homepage）"""
        ids = list(self.tabs.keys())
        for tid in ids:
            if tid != "Homepage":
                self.close_tab(tid)
        self.switch_to_tab("Homepage")

if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    app = NeuroPioneerApp(root)
    root.mainloop()