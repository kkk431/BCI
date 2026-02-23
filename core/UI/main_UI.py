import tkinter as tk
from tkinter import ttk
import os
import re

# --- 视觉配置 ---
COLOR_TAB_BAR_BG = "#c0c0c0"
COLOR_TAB_INACTIVE = "#d0d0d0"
COLOR_TAB_ACTIVE = "white"
COLOR_CONTENT_BG = "white"

# 路径配置
BG_IMAGE_PATH = os.path.join("core", "UI", "UI_resource", "homepage_background.png")

class NeuroPioneerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智融脑机 - 大模型赋能多模态BCI平台")
        self.root.geometry("1100x700")
        self.root.configure(bg=COLOR_CONTENT_BG)

        # 标签管理
        self.tabs = {}               # {tab_id: {"frame":..., "tab_btn":..., "base_name":..., "display_name":...}}
        self.active_tab_id = None
        self.tab_history = []         # 记录标签激活顺序，用于关闭时返回上一个

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
        """初始化并显示 Homepage（唯一且不可关闭）"""
        if "Homepage" not in self.tabs:
            home_frame = tk.Frame(self.main_container, bg=COLOR_CONTENT_BG)

            # --- 背景图片自动缩放 ---
            try:
                from PIL import Image, ImageTk
                self.original_bg_image = Image.open(BG_IMAGE_PATH)

                bg_canvas = tk.Canvas(home_frame, bg=COLOR_CONTENT_BG, highlightthickness=0)
                bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

                def resize_bg(event):
                    cw, ch = event.width, event.height
                    if cw <= 0 or ch <= 0:
                        return
                    img_w, img_h = self.original_bg_image.size
                    ratio = min(cw / img_w, ch / img_h)
                    new_w, new_h = int(img_w * ratio), int(img_h * ratio)
                    resized = self.original_bg_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    self.bg_photo = ImageTk.PhotoImage(resized)
                    bg_canvas.delete("all")
                    bg_canvas.create_image(cw // 2, ch // 2, image=self.bg_photo, anchor="center")

                bg_canvas.bind("<Configure>", resize_bg)
            except Exception as e:
                print(f"背景图片加载失败: {e}")
                tk.Label(home_frame, text="智融脑机", font=("微软雅黑", 36, "bold"), bg=COLOR_CONTENT_BG).pack(pady=(150, 0))
                tk.Label(home_frame, text="大模型赋能多模态BCI平台", font=("微软雅黑", 18), bg=COLOR_CONTENT_BG).pack()
            # -------------------------

            btn_box = tk.Frame(home_frame, bg=COLOR_CONTENT_BG)
            btn_box.place(relx=0.5, rely=0.7, anchor="center")
            funcs = ["Preprocessing", "Extraction", "Analysis", "Visualization"]
            for f in funcs:
                btn = tk.Button(btn_box, text=f, font=("Arial", 11, "bold"), bg="#3d85a1", fg="white",
                                relief="flat", padx=20, pady=8, cursor="hand2",
                                command=lambda name=f: self.open_functional_tab(name))
                btn.pack(side="left", padx=15)

            tab_btn = self.create_tab_widget("Homepage", tab_id="Homepage", can_close=False)
            self.tabs["Homepage"] = {"frame": home_frame, "tab_btn": tab_btn,
                                      "base_name": "Homepage", "display_name": "Homepage"}

        self.switch_to_tab("Homepage")

    def _get_next_number_for_base(self, base_name):
        """获取当前 base_name 类型标签的下一个编号"""
        max_num = 0
        for data in self.tabs.values():
            if data.get("base_name") == base_name:
                display = data["display_name"]
                # 格式为 "BaseName (num)"
                match = re.search(r'\((\d+)\)', display)
                if match:
                    num = int(match.group(1))
                    max_num = max(max_num, num)
        return max_num + 1

    def open_functional_tab(self, base_name):
        """打开一个新的功能标签页（总是新建）"""
        next_num = self._get_next_number_for_base(base_name)
        display_name = f"{base_name} ({next_num})"
        tab_id = f"{base_name}_{next_num}"  # 唯一ID

        new_frame = tk.Frame(self.main_container, bg=COLOR_CONTENT_BG)
        # 临时占位内容，可替换为实际功能界面
        tk.Label(new_frame, text=f"{base_name.lower()} 界面",
                 font=("微软雅黑", 24), fg="#333", bg=COLOR_CONTENT_BG).place(relx=0.5, rely=0.5, anchor="center")

        tab_btn = self.create_tab_widget(display_name, tab_id=tab_id)
        self.tabs[tab_id] = {"frame": new_frame, "tab_btn": tab_btn,
                             "base_name": base_name, "display_name": display_name}

        self.switch_to_tab(tab_id)

    def switch_to_tab(self, tab_id):
        """切换到指定 ID 的标签页，并更新历史记录"""
        if tab_id not in self.tabs:
            return

        self.active_tab_id = tab_id
        # 更新历史：移除旧位置，追加到末尾
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

        # 从历史记录中移除该标签的所有出现
        while tab_id in self.tab_history:
            self.tab_history.remove(tab_id)

        # 销毁界面组件
        data["frame"].destroy()
        data["tab_btn"].destroy()
        del self.tabs[tab_id]

        # 如果关闭的是当前激活的标签，则切换到历史中的上一个（或 Homepage）
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
        # 确保最后激活的是 keep_tab_id（可能已经被前面的 close_tab 切换走了）
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