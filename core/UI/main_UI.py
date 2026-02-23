import tkinter as tk
from tkinter import font
import os
from PIL import Image, ImageTk

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("智融脑机 · 大模型赋能多模态BCI平台")
        self.root.geometry("1000x600")  # 5:3 比例
        self.root.resizable(False, False)

        # 设置列权重：左侧1份，右侧4份
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=4)
        self.root.grid_rowconfigure(0, weight=1)

        # 左侧面板（白色背景）
        self.left_frame = tk.Frame(self.root, bg="white")
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        # 右侧容器（所有界面放置于此）
        self.right_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        self.right_frame.grid_propagate(False)

        # 存储按钮对象
        self.buttons = {}
        # 存储功能界面的Frame（key: 功能名, value: Frame）
        self.frames = {}
        # 记录当前打开的功能界面（按z-order，最后一个为最上层）
        self.open_frames = []

        # 创建左侧内容
        self.create_left_content()

        # 创建底层界面（永远存在）
        self.create_default_right()

    def create_left_content(self):
        """创建左侧的Logo和导航按钮"""
        # ---------- Logo ----------
        logo_path = os.path.join("core", "UI", "UI_resource", "logo.png")
        if os.path.exists(logo_path):
            try:
                img = Image.open(logo_path)
                # 缩放至高度80像素
                base_height = 80
                w_percent = base_height / float(img.size[1])
                new_width = int(float(img.size[0]) * w_percent)
                img = img.resize((new_width, base_height), Image.Resampling.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(img)
                logo_label = tk.Label(self.left_frame, image=self.logo_img, bg="white")
            except Exception as e:
                logo_label = tk.Label(self.left_frame, text="LOGO加载失败", bg="white", fg="red")
        else:
            logo_label = tk.Label(self.left_frame, text="LOGO (文件不存在)", bg="white", fg="red")
        logo_label.pack(pady=(30, 40))

        # ---------- 四个导航按钮 ----------
        btn_texts = ["Preprocessing", "Extraction", "Analysis", "Visualization"]
        for text in btn_texts:
            btn = tk.Button(self.left_frame, text=text,
                            font=("Arial", 12),
                            bg="white", fg="gray",          # 未选中样式
                            activebackground="purple",
                            activeforeground="white",
                            relief="flat", bd=0,
                            padx=10, pady=8,
                            command=lambda t=text: self.on_button_click(t))
            btn.pack(fill="x", padx=10, pady=5)
            self.buttons[text] = btn

    def create_default_right(self):
        """创建底层界面：两行居中文字"""
        self.default_frame = tk.Frame(self.right_frame, bg="#f0f0f0")
        # 第一行大字
        label1 = tk.Label(self.default_frame, text="智融脑机",
                          font=("Microsoft YaHei", 36, "bold"),
                          bg="#f0f0f0", fg="#333333")
        label1.place(relx=0.5, rely=0.4, anchor="center")
        # 第二行小字
        label2 = tk.Label(self.default_frame, text="大模型赋能多模态BCI平台",
                          font=("Microsoft YaHei", 16),
                          bg="#f0f0f0", fg="#666666")
        label2.place(relx=0.5, rely=0.55, anchor="center")
        # 放置到底层
        self.default_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.default_frame.lower()

    def create_function_frame(self, name):
        """根据功能名称创建对应的界面Frame（示例内容）"""
        frame = tk.Frame(self.right_frame, bg="#e0e0e0")  # 稍深一点的灰色以区分
        # 标题
        title = tk.Label(frame, text=f"{name} 功能界面", font=("Arial", 20), bg="#e0e0e0")
        title.pack(pady=20)

        # 根据功能添加不同的示例控件
        if name == "Preprocessing":
            tk.Button(frame, text="加载数据").pack(pady=5)
            tk.Checkbutton(frame, text="滤波").pack(pady=5)
            tk.Checkbutton(frame, text="归一化").pack(pady=5)
        elif name == "Extraction":
            tk.Label(frame, text="特征提取参数：").pack(pady=5)
            tk.Entry(frame).pack(pady=5)
            tk.Button(frame, text="开始提取").pack(pady=5)
        elif name == "Analysis":
            tk.Label(frame, text="分析结果：").pack(pady=5)
            tk.Text(frame, height=5, width=40).pack(pady=5)
        elif name == "Visualization":
            tk.Label(frame, text="绘图区域：").pack(pady=5)
            tk.Canvas(frame, bg="white", height=150, width=300).pack(pady=5)

        return frame

    def update_button_styles(self):
        """根据界面打开状态更新所有按钮的样式"""
        for name, btn in self.buttons.items():
            if name in self.frames and self.frames[name] in self.open_frames:
                # 界面存在且显示中 -> 选中样式
                btn.config(bg="purple", fg="white")
            else:
                # 界面不存在或隐藏 -> 未选中样式
                btn.config(bg="white", fg="gray")

    def on_button_click(self, name):
        """处理按钮点击事件"""
        frame = self.frames.get(name)

        # 情况1：界面不存在 -> 创建并显示
        if frame is None:
            frame = self.create_function_frame(name)
            self.frames[name] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.open_frames.append(frame)
            # 新界面自动在最上层
            frame.lift()
        else:
            # 界面已存在
            if frame in self.open_frames:
                # 当前显示中
                if frame == self.open_frames[-1]:
                    # 是最上层 -> 隐藏它
                    frame.place_forget()
                    self.open_frames.remove(frame)
                else:
                    # 不是最上层 -> 提升到最上层
                    frame.lift()
                    # 更新open_frames顺序
                    self.open_frames.remove(frame)
                    self.open_frames.append(frame)
            else:
                # 隐藏状态 -> 重新显示并提升到最上层
                frame.place(relx=0, rely=0, relwidth=1, relheight=1)
                self.open_frames.append(frame)
                frame.lift()

        # 更新所有按钮样式
        self.update_button_styles()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()