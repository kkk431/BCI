import sys
from pathlib import Path

# 将项目根目录（即core的上一级目录）动态添加到 sys.path
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
import math
from PIL import Image, ImageDraw, ImageFilter, ImageTk

# 导入 AI 相关模块
try:
    from core.ai_core.llm_client import SimpleAIChat
    from core.UI.panel.ai_chat_panel import AIChatWindow
except ImportError:
    print("AI 模块路径配置错误或未安装 openai 库")
    SimpleAIChat = None
    AIChatWindow = None

from core.UI.panel.extraction_panel import ExtractionApp

# 视觉配置
COLOR_TAB_BAR_BG = "#c0c0c0"
COLOR_TAB_INACTIVE = "#d0d0d0"
COLOR_TAB_ACTIVE = "white"
COLOR_CONTENT_BG = "white"

# 图片资源路径
LOGO_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "logo.png")
MENU_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "menu.png")
CONTENT_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "Content.png")
BUTTON1_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "button1.png")
BUTTON2_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "button2.png")
BUTTON3_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "button3.png")
BUTTON4_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "button4.png")
SIGNIN_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "SignIn.png")
BACKGROUND_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "background.png")
A_LOGO_PATH = os.path.join(project_root, "core", "UI", "UI_resource", "a_logo.png")

# 预加载可视化模块
print("正在预加载可视化模块...")
preload_start = time.time()
import matplotlib

matplotlib.use('TkAgg')
preload_time = time.time() - preload_start
print(f"预加载完成，耗时: {preload_time:.2f}秒")


class NeuroPioneerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智融脑机 - 大模型赋能多模态BCI平台")
        self.root.geometry("1100x700")
        self.root.configure(bg=COLOR_CONTENT_BG)

        # AI 相关
        self.ai_logic = SimpleAIChat() if SimpleAIChat else None
        self.ai_window = None

        # 标签管理
        self.tabs = {}
        self.active_tab_id = None
        self.tab_history = []
        self.home_images = []

        # 气泡相关
        self.bubble_size = 60
        self.bubble_x = 0
        self.bubble_y = 0
        self.bubble_dragging = False
        self.bubble_hover = False

        self.setup_ui()
        self.show_homepage()

        # 创建灵动悬浮气泡
        self.create_ai_bubble()

    def setup_ui(self):
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
        tab_frame = tk.Frame(self.tab_container, bg=COLOR_TAB_INACTIVE, bd=0)
        tab_frame.pack(side="left", fill="y", padx=(0, 1))

        line = tk.Frame(tab_frame, bg="black", width=1)
        line.pack(side="left", fill="y")

        lbl = tk.Label(tab_frame, text=display_name, bg=COLOR_TAB_INACTIVE, padx=15,
                       font=("微软雅黑", 16), cursor="hand2")
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

        self.home_images.clear()

        try:
            from PIL import Image, ImageTk
            bg_img = Image.open(BACKGROUND_PATH)
            bg_width, bg_height = bg_img.size
            self.root.geometry(f"{bg_width}x{bg_height + 45}")
            self.root.resizable(False, False)
        except Exception as e:
            print(f"背景图片加载失败: {e}")
            bg_width, bg_height = 1100, 700
            self.root.geometry(f"{bg_width}x{bg_height + 45}")
            bg_img = None

        home_canvas = tk.Canvas(self.main_container, bg=COLOR_CONTENT_BG, highlightthickness=0,
                                width=bg_width, height=bg_height)
        home_canvas.pack_propagate(False)
        home_canvas.config(width=bg_width, height=bg_height)

        if bg_img:
            try:
                self.bg_photo = ImageTk.PhotoImage(bg_img)
                home_canvas.create_image(0, 0, anchor="nw", image=self.bg_photo)
                self.home_images.append(self.bg_photo)
            except Exception as e:
                print(f"背景图片显示失败: {e}")

        def add_image(path, x, y, tag=None):
            try:
                img = Image.open(path)
                photo = ImageTk.PhotoImage(img)
                item_id = home_canvas.create_image(x, y, anchor="nw", image=photo, tag=tag)
                self.home_images.append(photo)
                return photo, item_id
            except Exception as e:
                print(f"图片加载失败 {path}: {e}")
                return None, None

        add_image(MENU_PATH, 280, 100)
        add_image(CONTENT_PATH, 60, 330)

        button_info = [
            (BUTTON1_PATH, 60, 700, "Preprocessing"),
            (BUTTON2_PATH, 400, 700, "Extraction"),
            (BUTTON3_PATH, 60, 800, "Analysis"),
            (BUTTON4_PATH, 400, 800, "Visualization")
        ]
        for path, x, y, name in button_info:
            _, item_id = add_image(path, x, y, tag=f"btn_{name}")
            if item_id:
                home_canvas.tag_bind(item_id, "<Button-1>", lambda e, n=name: self.open_functional_tab(n))
                home_canvas.tag_bind(item_id, "<Enter>", lambda e: home_canvas.config(cursor="hand2"))
                home_canvas.tag_bind(item_id, "<Leave>", lambda e: home_canvas.config(cursor=""))

        tab_btn = self.create_tab_widget("Homepage", tab_id="Homepage", can_close=False)
        self.tabs["Homepage"] = {
            "frame": home_canvas,
            "tab_btn": tab_btn,
            "base_name": "Homepage",
            "display_name": "Homepage"
        }

        self.switch_to_tab("Homepage")

    def _get_next_number_for_base(self, base_name):
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
        if base_name == "Visualization":
            self.open_visualization_tab()
        elif base_name == "Extraction":
            self.open_extraction_tab()
        else:
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
        next_num = self._get_next_number_for_base("Extraction")
        display_name = f"Extraction ({next_num})"
        tab_id = f"Extraction_{next_num}"

        new_frame = tk.Frame(self.main_container, bg=COLOR_CONTENT_BG)

        loading_label = tk.Label(new_frame, text="正在加载特征提取模块...",
                                 font=("微软雅黑", 14), fg="#666", bg=COLOR_CONTENT_BG)
        loading_label.place(relx=0.5, rely=0.5, anchor="center")
        new_frame.update()

        tab_btn = self.create_tab_widget(display_name, tab_id=tab_id)

        self.tabs[tab_id] = {
            "frame": new_frame,
            "tab_btn": tab_btn,
            "base_name": "Extraction",
            "display_name": display_name
        }

        self.switch_to_tab(tab_id)

        def load_extraction():
            try:
                loading_label.destroy()
                extraction_app = ExtractionApp(new_frame)
                extraction_app.pack(fill=tk.BOTH, expand=True)
                print("特征提取面板加载完成")
            except Exception as e:
                loading_label.config(text=f"加载失败: {str(e)}", fg="red")
                import traceback
                traceback.print_exc()

        new_frame.after(10, load_extraction)

    def open_visualization_tab(self):
        next_num = self._get_next_number_for_base("Visualization")
        display_name = f"Visualization ({next_num})"
        tab_id = f"Visualization_{next_num}"

        new_frame = tk.Frame(self.main_container, bg=COLOR_CONTENT_BG)

        loading_label = tk.Label(new_frame, text="正在加载可视化模块...",
                                 font=("微软雅黑", 14), fg="#666", bg=COLOR_CONTENT_BG)
        loading_label.place(relx=0.5, rely=0.5, anchor="center")
        new_frame.update()

        tab_btn = self.create_tab_widget(display_name, tab_id=tab_id)

        self.tabs[tab_id] = {
            "frame": new_frame,
            "tab_btn": tab_btn,
            "base_name": "Visualization",
            "display_name": display_name
        }

        self.switch_to_tab(tab_id)

        def load_visualization():
            try:
                import time
                start = time.time()
                from core.UI.panel.visualization_panel import ModernVisualizationPanel as VisualizationPanel
                loading_label.destroy()
                vis_panel = VisualizationPanel(new_frame)
                vis_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                elapsed = time.time() - start
                print(f"可视化面板加载完成，耗时: {elapsed:.2f}秒")
            except Exception as e:
                loading_label.config(text=f"加载失败: {str(e)}", fg="red")

        new_frame.after(10, load_visualization)

    def switch_to_tab(self, tab_id):
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
        popup = tk.Toplevel(self.root)
        popup.overrideredirect(True)
        x = self.add_btn.winfo_rootx()
        y = self.add_btn.winfo_rooty() + 40
        popup.geometry(f"180x200+{x}+{y}")
        popup.config(bg="white", highlightthickness=1, highlightbackground="#ccc")

        tk.Label(popup, text="快速跳转", bg="#eee", font=("微软雅黑", 12, "bold")).pack(fill="x")

        others = ["Preprocessing", "Extraction", "Analysis", "Visualization", "Settings"]
        for o in others:
            lbl = tk.Label(popup, text=o, bg="white", font=("微软雅黑", 12, "bold"), pady=5, cursor="hand2")
            lbl.pack(fill="x")
            lbl.bind("<Button-1>", lambda e, name=o: [self.open_functional_tab(name), popup.destroy()])
            lbl.bind("<Enter>", lambda e: e.widget.config(bg="#f0f0f0"))
            lbl.bind("<Leave>", lambda e: e.widget.config(bg="white"))

        popup.bind("<FocusOut>", lambda e: popup.destroy())
        popup.focus_set()

    def show_context_menu(self, event, tab_id):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="关闭其他标签", command=lambda: self.close_others(tab_id))
        menu.add_command(label="全部关闭", command=self.close_all_functional)
        menu.post(event.x_root, event.y_root)

    def close_others(self, keep_tab_id):
        ids = list(self.tabs.keys())
        for tid in ids:
            if tid != keep_tab_id and tid != "Homepage":
                self.close_tab(tid)
        self.switch_to_tab(keep_tab_id)

    def close_all_functional(self):
        ids = list(self.tabs.keys())
        for tid in ids:
            if tid != "Homepage":
                self.close_tab(tid)
        self.switch_to_tab("Homepage")

    # ========== 最终防误触版：AI悬浮气泡 ==========
    def create_ai_bubble(self):
        # 基础固定配置（全局统一，杜绝错位）
        self.bubble_window_size = 64  # 气泡窗口固定总尺寸
        self.bubble_core_radius = 26  # 气泡主体半径（固定值，所有圆形以此为基准）
        # 全兼容6位十六进制颜色（Python3.8无报错）
        self.theme_main = "#2D7DDB"  # 主色，100%匹配主界面
        self.theme_hover = "#1E6BC6"  # hover加深色
        self.theme_press = "#1557A0"  # 点击按压色
        self.shadow_color = "#CCCCCC"  # 阴影底色
        self.shadow_sub_color = "#E0E0E0"  # 阴影过渡色
        self.theme_breath_light = "#3A88E2"  # 呼吸动画浅色
        self.theme_breath_dark = "#2572C9"  # 呼吸动画深色

        # 状态管理
        self.is_hovering = False
        self.is_dragging = False
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        # ========== 新增：防误触核心状态配置 ==========
        self.is_valid_click = False  # 标记是否为有效点击动作
        self.press_root_x = 0  # 鼠标按下时的屏幕x坐标
        self.press_root_y = 0  # 鼠标按下时的屏幕y坐标
        self.drag_threshold = 3  # 拖拽判定阈值（像素）：移动超过3px判定为拖拽，不触发点击
        # ==================================================

        # 创建气泡窗口（固定无边框、置顶、透明背景）
        self.bubble_window = tk.Toplevel(self.root)
        self.bubble_window.overrideredirect(True)
        self.bubble_window.attributes("-topmost", True)
        self.bubble_window.configure(bg="white")
        # 透明通道兼容，消除白边
        try:
            self.bubble_window.wm_attributes("-transparentcolor", "white")
        except:
            pass

        # 气泡画布（固定尺寸，全局唯一中心点，杜绝错位根源）
        self.canvas_center_x = self.bubble_window_size / 2
        self.canvas_center_y = self.bubble_window_size / 2
        self.bubble_canvas = tk.Canvas(
            self.bubble_window,
            width=self.bubble_window_size,
            height=self.bubble_window_size,
            bg="white",
            highlightthickness=0,
            cursor="hand2"
        )
        self.bubble_canvas.pack()

        # 初始化绘制气泡
        self._draw_perfect_bubble(self.theme_main)

        # ========== 事件绑定（核心修改：移除直接点击触发，改为松开判定触发） ==========
        # 移除原来的直接点击触发，避免拖拽误触
        # self.bubble_canvas.bind("<Button-1>", lambda e: self.toggle_ai())
        # 拖拽功能（仅在主界面内拖动）
        self.bubble_canvas.bind("<ButtonPress-1>", self._bubble_drag_start)
        self.bubble_canvas.bind("<B1-Motion>", self._bubble_drag_move)
        self.bubble_canvas.bind("<ButtonRelease-1>", self._bubble_drag_end)
        # 微交互（仅变色，不改变位置大小）
        self.bubble_canvas.bind("<Enter>", self._bubble_on_hover)
        self.bubble_canvas.bind("<Leave>", self._bubble_on_leave)
        self.bubble_canvas.bind("<ButtonPress-1>", self._bubble_on_press, add="+")
        self.bubble_canvas.bind("<ButtonRelease-1>", self._bubble_on_release, add="+")

        # 初始位置：主窗口右下角（固定不变）
        self._update_bubble_position()
        # 主窗口移动时，气泡跟随主窗口
        self.root.bind("<Configure>", lambda e: self._update_bubble_position())
        # 呼吸动画（仅同色系深浅变化，无位移缩放）
        self._bubble_breath_animation()

    def _draw_perfect_bubble(self, fill_color):
        """完美同心气泡绘制：彻底解决圈错位问题，所有元素严格同心"""
        self.bubble_canvas.delete("all")
        r = self.bubble_core_radius
        cx = self.canvas_center_x
        cy = self.canvas_center_y

        # ========== 1. 双层阴影（严格同心，仅固定偏移，无错位） ==========
        shadow_offset_x = 2
        shadow_offset_y = 3
        # 底层主阴影
        self.bubble_canvas.create_oval(
            cx - r + shadow_offset_x, cy - r + shadow_offset_y,
            cx + r + shadow_offset_x, cy + r + shadow_offset_y,
            fill=self.shadow_color, outline=""
        )
        # 内层过渡阴影
        self.bubble_canvas.create_oval(
            cx - r + shadow_offset_x - 1, cy - r + shadow_offset_y - 1,
            cx + r + shadow_offset_x - 1, cy + r + shadow_offset_y - 1,
            fill=self.shadow_sub_color, outline=""
        )

        # ========== 2. 气泡主体（完美正圆，严格居中） ==========
        self.bubble_canvas.create_oval(
            cx - r, cy - r,
            cx + r, cy + r,
            fill=fill_color, outline="#4A90E2", width=1
        )

        # ========== 3. 内高光（和主体严格同心，无错位） ==========
        highlight_r = r - 4
        self.bubble_canvas.create_arc(
            cx - highlight_r, cy - highlight_r,
            cx + highlight_r, cy + highlight_r,
            start=45, extent=90, style="arc",
            outline="#FFFFFF", width=2
        )

        # ========== 4. 中心图标/文字（严格和主体同心，无偏移） ==========
        try:
            from PIL import Image, ImageTk, ImageDraw
            # 加载并处理logo，固定尺寸，正圆形蒙版
            logo_size = int(r * 1.4)
            logo_img = Image.open(A_LOGO_PATH).convert("RGBA")
            logo_img = logo_img.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
            # 强制正圆形蒙版
            mask = Image.new('L', (logo_size, logo_size), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.ellipse((0, 0, logo_size, logo_size), fill=255)
            logo_img.putalpha(mask)
            # 严格居中渲染
            self.bubble_logo = ImageTk.PhotoImage(logo_img)
            self.bubble_canvas.create_image(cx, cy, anchor="center", image=self.bubble_logo)
        except:
            # 降级方案：AI文字严格居中
            self.bubble_canvas.create_text(
                cx, cy, text="AI",
                fill="white", font=("Arial", 18, "bold"),
                anchor="center"
            )

    # ========== 呼吸动画（仅颜色变化，无位移缩放） ==========
    def _bubble_breath_animation(self):
        if not self.is_hovering and not self.is_dragging:
            # 正弦函数自然呼吸，仅同色系深浅切换
            breath_progress = abs(math.sin(time.time() * 1.5))
            current_color = self.theme_breath_light if breath_progress > 0.5 else self.theme_breath_dark
            self._draw_perfect_bubble(current_color)
        self.bubble_window.after(30, self._bubble_breath_animation)

    # ========== 微交互（仅变色，无位移） ==========
    def _bubble_on_hover(self, e):
        self.is_hovering = True
        self._draw_perfect_bubble(self.theme_hover)

    def _bubble_on_leave(self, e):
        self.is_hovering = False
        self._draw_perfect_bubble(self.theme_main)

    def _bubble_on_press(self, e):
        self._draw_perfect_bubble(self.theme_press)

    def _bubble_on_release(self, e):
        if self.is_hovering:
            self._draw_perfect_bubble(self.theme_hover)
        else:
            self._draw_perfect_bubble(self.theme_main)

    # ========== 核心修改：防误触拖拽逻辑 ==========
    def _bubble_drag_start(self, e):
        """鼠标按下：初始化状态，记录按下坐标"""
        self.is_dragging = False
        self.is_valid_click = True  # 初始标记为有效点击
        # 记录鼠标按下时的屏幕绝对坐标
        self.press_root_x = e.x_root
        self.press_root_y = e.y_root
        # 计算拖拽偏移量
        self.drag_offset_x = e.x_root - self.bubble_window.winfo_x()
        self.drag_offset_y = e.y_root - self.bubble_window.winfo_y()

    def _bubble_drag_move(self, e):
        """鼠标拖拽移动：超过阈值判定为拖拽，取消有效点击标记"""
        # 计算当前鼠标与按下位置的移动距离
        move_distance = math.sqrt(
            (e.x_root - self.press_root_x) ** 2 +
            (e.y_root - self.press_root_y) ** 2
        )

        # 移动距离超过阈值，判定为拖拽动作
        if move_distance > self.drag_threshold:
            self.is_valid_click = False  # 取消点击标记，不会触发窗口弹出
            self.is_dragging = True

        # 执行拖拽逻辑
        if self.is_dragging:
            # 计算气泡新位置（屏幕绝对坐标）
            new_x = e.x_root - self.drag_offset_x
            new_y = e.y_root - self.drag_offset_y

            # 限制：仅允许在主界面内拖动
            root_x = self.root.winfo_rootx()
            root_y = self.root.winfo_rooty()
            root_w = self.root.winfo_width()
            root_h = self.root.winfo_height()

            # 边界限制：气泡完整区域必须在主窗口内
            min_x = root_x
            max_x = root_x + root_w - self.bubble_window_size
            min_y = root_y
            max_y = root_y + root_h - self.bubble_window_size

            # 强制限制坐标在范围内
            final_x = max(min_x, min(new_x, max_x))
            final_y = max(min_y, min(new_y, max_y))

            # 更新气泡位置
            self.bubble_window.geometry(f"+{final_x}+{final_y}")

    def _bubble_drag_end(self, e):
        """鼠标松开：仅有效点击才触发AI窗口，拖拽动作不触发"""
        # 只有有效点击（无拖拽移动）才打开AI窗口
        if self.is_valid_click and not self.is_dragging:
            self.toggle_ai()

        # 重置状态
        self.is_dragging = False
        self.is_valid_click = False

        # 松开后恢复对应状态
        if self.is_hovering:
            self._draw_perfect_bubble(self.theme_hover)
        else:
            self._draw_perfect_bubble(self.theme_main)

    # ========== 位置跟随：初始位置固定主窗口右下角 ==========
    def _update_bubble_position(self):
        # 拖拽中不更新位置，避免冲突
        if self.is_dragging:
            return
        if not self.root.winfo_viewable():
            return
        # 获取主窗口实时坐标与尺寸
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_w = self.root.winfo_width()
        root_h = self.root.winfo_height()
        # 固定初始位置：主窗口右下角，距离边缘20px
        target_x = root_x + root_w - self.bubble_window_size - 20
        target_y = root_y + root_h - self.bubble_window_size - 20
        self.bubble_window.geometry(f"+{target_x}+{target_y}")

    # ========== 防误触版代码结束 ==========

    def toggle_ai(self):
        """切换AI窗口显示/隐藏"""
        if not self.ai_logic:
            print("AI模块未正确加载")
            return

        # 计算AI窗口位置（气泡左侧）
        bubble_x = self.bubble_window.winfo_x()
        bubble_y = self.bubble_window.winfo_y()
        bubble_w = self.bubble_size
        bubble_h = self.bubble_size

        # AI窗口尺寸
        chat_w = 420
        chat_h = 600

        # 位置：气泡左侧，垂直居中
        x = bubble_x - chat_w - 10
        y = bubble_y + (bubble_h - chat_h) // 2

        # 屏幕边界检查
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        if x < 0:
            x = bubble_x + bubble_w + 10  # 右侧放不下放左侧
        if y < 0:
            y = 50
        if y + chat_h > screen_h:
            y = screen_h - chat_h - 50

        # 处理AI窗口
        if self.ai_window is None or not self.ai_window.window.winfo_exists():
            self.ai_window = AIChatWindow(self.root, self.ai_logic, position=(x, y))
        else:
            self.ai_window.window.geometry(f"{chat_w}x{chat_h}+{x}+{y}")
            if not self.ai_window.window.winfo_viewable():
                self.ai_window.show()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    app = NeuroPioneerApp(root)
    root.mainloop()
