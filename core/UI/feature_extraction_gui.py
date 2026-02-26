# -*- coding: utf-8 -*-
# isort: skip_file
# flake8: noqa
"""
智融脑机 - 独立特征提取模块 GUI
(彻底解决路径导入问题，严格遵循 main_UI 风格，隐式执行全管线)
"""

import os
import sys

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import traceback
import numpy as np

# --- 导入底层业务模块 (现在绝对能找到了) ---
try:
    from core.io.data_io import DataLoader
    from core.processing.preprocessing.multimodal_preprocessing import MultiModalPreprocessor, MultiModalConfigFactory
    from core.processing.preprocessing.multimodal_preprocessing import example_usage
    from core.processing.feature_extraction.multimodal_pipeline import MultimodalFeaturePipeline

    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    print(f"模块导入依旧失败，详细信息: {e}")
    traceback.print_exc()

# --- 视觉配置 (严格复刻 main_UI.py 风格) ---
COLOR_CONTENT_BG = "white"
COLOR_BTN_BG = "#3d85a1"
COLOR_BTN_FG = "white"
COLOR_TEXT_MAIN = "#333333"
COLOR_TEXT_SUB = "#666666"
FONT_TITLE = ("微软雅黑", 14, "bold")
FONT_NORMAL = ("微软雅黑", 10)
FONT_BTN = ("Arial", 10, "bold")

# 特征映射表（用于动态生成复选框）
FEATURE_MAP = {
    "EEG": {
        "time_domain": "时域特征 (均值, 方差, Hjorth参数等)",
        "freq_domain": "频域特征 (Welch PSD, 谱熵等)",
        "wavelet": "小波时频特征 (各频带能量)",
        "nonlinear": "非线性特征 (排列熵, 样本熵等)",
        "band_power": "生理频带功率 (Alpha, Beta功率)",
        "erp": "ERP事件相关电位 (需包含事件)",
        "connectivity": "功能连接性 (通道间相干性)",
        "spatial": "空间地形图特征"
    },
    "fNIRS": {
        "time_domain": "时域统计特征",
        "freq_domain": "频域功率特征",
        "wavelet": "小波分析特征",
        "nonlinear": "非线性动力学特征",
        "hbo_hbr": "双信号源特征 (HbO, HbR, HbT)",
        "channel_correlation": "通道间相关性分析"
    },
    "EMG": {
        "time_domain": "肌电时域特征 (MAV, RMS, 过零率等)",
        "freq_domain": "肌电频域特征 (中频, 均频等)",
        "wavelet": "肌电时频特征 (小波能量分布)",
        "nonlinear": "肌电非线性特征 (分形维数等)"
    },
    "ECG": {
        "morphological": "心电形态学特征 (心率, QRS波群)",
        "hrv_time": "HRV 心率变异性 - 时域 (SDNN, RMSSD)",
        "hrv_frequency": "HRV 心率变异性 - 频域 (LF, HF)",
        "wavelet": "时频特征 (STFT短时能量)",
        "nonlinear": "非线性特征 (Poincare, DFA)"
    }
}


class ExtractionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智融脑机 - 特征提取模块")
        self.root.geometry("1000x700")
        self.root.configure(bg=COLOR_CONTENT_BG)

        # 核心数据容器
        self.clean_data_dict = None  # 存放 IO 加载 + 预处理后的纯净数据
        self.extracted_features = {}  # 存放提取出的特征字典
        self.checkbox_vars = {}  # 存放用户勾选的特征变量

        if not MODULES_LOADED:
            messagebox.showerror("初始化失败", "底层算法模块导入失败，请检查终端输出的路径信息。")

        self.setup_ui()

    def setup_ui(self):
        """构建整体 UI 布局"""
        main_container = tk.Frame(self.root, bg=COLOR_CONTENT_BG, padx=30, pady=20)
        main_container.pack(fill="both", expand=True)

        # 标题
        title_lbl = tk.Label(main_container, text="多模态信号特征提取", font=("微软雅黑", 20, "bold"),
                             bg=COLOR_CONTENT_BG, fg=COLOR_TEXT_MAIN)
        title_lbl.pack(anchor="w", pady=(0, 20))

        # ================= 区域 1：数据加载 =================
        frame_step1 = tk.Frame(main_container, bg=COLOR_CONTENT_BG)
        frame_step1.pack(fill="x", pady=10)

        lbl_step1 = tk.Label(frame_step1, text="1. 加载数据源", font=FONT_TITLE, bg=COLOR_CONTENT_BG,
                             fg=COLOR_TEXT_MAIN)
        lbl_step1.pack(anchor="w", pady=(0, 10))

        btn_box1 = tk.Frame(frame_step1, bg=COLOR_CONTENT_BG)
        btn_box1.pack(fill="x")

        self.btn_load = tk.Button(btn_box1, text="上传本地数据文件", font=FONT_BTN, bg=COLOR_BTN_BG, fg=COLOR_BTN_FG,
                                  relief="flat", padx=15, pady=6, cursor="hand2", command=self.action_load_data)
        self.btn_load.pack(side="left")

        self.lbl_status = tk.Label(btn_box1, text="请上传文件 (后台将自动完成解析与降噪)...", font=FONT_NORMAL,
                                   bg=COLOR_CONTENT_BG, fg=COLOR_TEXT_SUB)
        self.lbl_status.pack(side="left", padx=15)

        tk.Frame(main_container, bg="#e0e0e0", height=1).pack(fill="x", pady=15)

        # ================= 区域 2：特征勾选与提取 =================
        frame_step2 = tk.Frame(main_container, bg=COLOR_CONTENT_BG)
        frame_step2.pack(fill="x", pady=10)

        lbl_step2 = tk.Label(frame_step2, text="2. 精细化特征勾选", font=FONT_TITLE, bg=COLOR_CONTENT_BG,
                             fg=COLOR_TEXT_MAIN)
        lbl_step2.pack(anchor="w", pady=(0, 10))

        mod_box = tk.Frame(frame_step2, bg=COLOR_CONTENT_BG)
        mod_box.pack(fill="x")

        tk.Label(mod_box, text="目标模态：", font=FONT_NORMAL, bg=COLOR_CONTENT_BG, fg=COLOR_TEXT_MAIN).pack(side="left")

        style = ttk.Style()
        style.theme_use('clam')
        self.modality_var = tk.StringVar()
        self.combo_modality = ttk.Combobox(mod_box, textvariable=self.modality_var, state="readonly", width=15,
                                           font=FONT_NORMAL)
        self.combo_modality.pack(side="left", padx=10)
        self.combo_modality.bind("<<ComboboxSelected>>", self.refresh_checkboxes)

        self.check_container = tk.Frame(frame_step2, bg="#f8f9fa", highlightbackground="#e0e0e0", highlightthickness=1,
                                        padx=15, pady=10)
        self.check_container.pack(fill="x", pady=15)
        tk.Label(self.check_container, text="请先上传数据并选择模态...", bg="#f8f9fa", font=FONT_NORMAL,
                 fg=COLOR_TEXT_SUB).pack(anchor="w")

        self.btn_extract = tk.Button(frame_step2, text="⚡ 提取已勾选特征", font=FONT_BTN, bg=COLOR_BTN_BG,
                                     fg=COLOR_BTN_FG,
                                     relief="flat", padx=15, pady=6, cursor="hand2", command=self.action_extract,
                                     state="disabled")
        self.btn_extract.pack(anchor="w")

        tk.Frame(main_container, bg="#e0e0e0", height=1).pack(fill="x", pady=15)

        # ================= 区域 3：结果展示 =================
        frame_step3 = tk.Frame(main_container, bg=COLOR_CONTENT_BG)
        frame_step3.pack(fill="both", expand=True)

        lbl_step3 = tk.Label(frame_step3, text="3. 提取结果查看", font=FONT_TITLE, bg=COLOR_CONTENT_BG,
                             fg=COLOR_TEXT_MAIN)
        lbl_step3.pack(anchor="w", pady=(0, 10))

        table_frame = tk.Frame(frame_step3, bg=COLOR_CONTENT_BG)
        table_frame.pack(fill="both", expand=True)

        columns = ("Name", "Value")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("Name", text="特征名称", anchor="w")
        self.tree.heading("Value", text="特征数值", anchor="w")
        self.tree.column("Name", width=400, anchor="w")
        self.tree.column("Value", width=300, anchor="w")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # ================= 业务逻辑 =================

    def action_load_data(self):
        """隐式执行 data_io 读取 -> preprocessor 降噪"""
        if not MODULES_LOADED:
            messagebox.showerror("错误", "底层模块未加载，请检查文件路径是否正确。")
            return
        #
        # filepath = filedialog.askopenfilename(
        #     title="选择数据文件",
        #     filetypes=[("支持的格式", "*.csv *.mat *.edf *.bdf *.npy *.npz *.snirf *.set *.pkl *.json"),
        #                ("All Files", "*.*")]
        # )
        # if not filepath:
        #     return
        #
        # self.btn_load.config(state="disabled", bg="#a0a0a0", cursor="arrow")
        # self.btn_extract.config(state="disabled", bg="#a0a0a0", cursor="arrow")
        # self.lbl_status.config(text="正在读取文件并进行后台降噪处理，这可能需要几秒钟，请稍候...", fg="#0056b3")
        #
        # for item in self.tree.get_children():
        #     self.tree.delete(item)

        def background_task():
            try:
                # [隐式步骤1]：调用 data_io.py 解析格式
                # loader = DataLoader()
                # raw_dict = loader.load(filepath)

                # [隐式步骤2]：调用预处理模块进行净化
                # prep_config = MultiModalConfigFactory.create_resting_state_config()
                # preprocessor = MultiModalPreprocessor(prep_config)
                # result = preprocessor.process(raw_dict)
                result = example_usage()
                if not result.success:
                    raise Exception(f"后台预处理去噪失败: {result.error_message}")

                self.clean_data_dict = result.processed_data

                mods = list(self.clean_data_dict.get('signal', {}).keys())
                valid_mods = [m for m in mods if m in FEATURE_MAP]

                # 防止跨线程刷新UI导致卡死
                if valid_mods:
                    self.root.after(0, self._ui_update_on_load_success, valid_mods)
                else:
                    self.root.after(0, self._ui_update_on_load_warning)

            except Exception as e:
                traceback.print_exc()
                self.root.after(0, self._ui_update_on_error, f"处理失败: {str(e)}")

        threading.Thread(target=background_task, daemon=True).start()

    def _ui_update_on_load_success(self, valid_mods):
        self.combo_modality['values'] = valid_mods
        self.combo_modality.current(0)
        self.refresh_checkboxes(None)

        self.lbl_status.config(text=f"数据就绪！已净化并识别出模态: {', '.join(valid_mods)}", fg="green")
        self.btn_extract.config(state="normal", bg=COLOR_BTN_BG, cursor="hand2")
        self.btn_load.config(state="normal", bg=COLOR_BTN_BG, cursor="hand2")

    def _ui_update_on_load_warning(self):
        self.lbl_status.config(text="数据已加载，但未检测到支持提取特征的有效模态。", fg="orange")
        self.btn_load.config(state="normal", bg=COLOR_BTN_BG, cursor="hand2")

    def _ui_update_on_error(self, error_msg):
        self.lbl_status.config(text=error_msg, fg="red")
        self.btn_load.config(state="normal", bg=COLOR_BTN_BG, cursor="hand2")
        self.btn_extract.config(state="disabled", bg="#a0a0a0", cursor="arrow")

    def refresh_checkboxes(self, event):
        """根据下拉框选中的模态，动态生成复选框"""
        mod = self.modality_var.get()
        if not mod or mod not in FEATURE_MAP:
            return

        for widget in self.check_container.winfo_children():
            widget.destroy()

        self.checkbox_vars = {}
        row, col = 0, 0

        for feat_key, feat_desc in FEATURE_MAP[mod].items():
            var = tk.BooleanVar(value=True)
            self.checkbox_vars[feat_key] = var

            chk = tk.Checkbutton(self.check_container, text=feat_desc, variable=var,
                                 bg="#f8f9fa", font=FONT_NORMAL, fg=COLOR_TEXT_MAIN,
                                 activebackground="#f8f9fa", selectcolor="white")
            chk.grid(row=row, column=col, sticky="w", padx=20, pady=5)

            col += 1
            if col > 1:
                col = 0
                row += 1

    def action_extract(self):
        """执行精确的特征提取 - 使用 MultimodalFeaturePipeline"""
        mod = self.modality_var.get()
        if not mod or not self.clean_data_dict:
            return

        selected_cats = [key for key, var in self.checkbox_vars.items() if var.get()]
        if not selected_cats:
            messagebox.showwarning("提示", "请至少勾选一种特征集！")
            return

        self.btn_extract.config(state="disabled", bg="#a0a0a0", cursor="arrow")
        self.btn_load.config(state="disabled", bg="#a0a0a0", cursor="arrow")
        self.lbl_status.config(text=f"正在通过 MultimodalFeaturePipeline 提取 {mod} 的特征，请稍候...", fg="#0056b3")

        def background_extraction():
            try:
                # 构建 selected_features 字典，格式为 {"EMG": ["time_domain", "nonlinear"]}
                selected_features = {mod: selected_cats}

                # 实例化 MultimodalFeaturePipeline 并运行
                pipeline = MultimodalFeaturePipeline(
                    data_dict=self.clean_data_dict,
                    selected_features=selected_features
                )

                # 运行完整的特征提取流程
                final_dict = pipeline.run_pipeline()

                # 从 processed['feature'] 中提取当前模态的特征
                if 'processed' in final_dict and 'feature' in final_dict['processed']:
                    all_features = final_dict['processed']['feature']
                    self.extracted_features = all_features.get(mod, {})
                else:
                    self.extracted_features = {}

                feat_count = len(self.extracted_features)
                print(f"特征提取完成，共提取 {feat_count} 个特征")

                self.root.after(0, self._ui_update_on_extract_success, mod, feat_count)

            except Exception as e:
                traceback.print_exc()
                self.root.after(0, self._ui_update_on_error, f"特征提取异常: {str(e)}")

        threading.Thread(target=background_extraction, daemon=True).start()

    def _ui_update_on_extract_success(self, mod, feat_count):
        """成功提取后刷新表格和状态"""
        self.lbl_status.config(text=f"【{mod}】{feat_count}个特征提取完毕！", fg="green")
        self.btn_extract.config(state="normal", bg=COLOR_BTN_BG, cursor="hand2")
        self.btn_load.config(state="normal", bg=COLOR_BTN_BG, cursor="hand2")
        self.render_table(mod)

    def render_table(self, mod):
        """将提取完毕的特征铺入表格"""
        for item in self.tree.get_children():
            self.tree.delete(item)

        mod_feats = self.extracted_features

        # 展平嵌套的特征字典
        flat_features = self._flatten_features(mod_feats)

        for name, val in flat_features.items():
            if isinstance(val, float):
                if abs(val) < 0.001:
                    val_str = f"{val:.6f}"
                elif abs(val) < 1:
                    val_str = f"{val:.4f}"
                else:
                    val_str = f"{val:.2f}"
            elif isinstance(val, (list, np.ndarray)):
                val_str = f"Array {np.shape(val)}: {str(val)[:50]}..."
            elif isinstance(val, dict):
                # 如果是字典，跳过（已经在扁平化时处理了）
                continue
            else:
                val_str = str(val)

            self.tree.insert('', "end", values=(name, val_str))

    def _flatten_features(self, features, parent_key='', sep='_'):
        """递归展平嵌套的特征字典"""
        items = []
        if not isinstance(features, dict):
            return {parent_key: features} if parent_key else {}

        for k, v in features.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_features(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


if __name__ == "__main__":
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    root = tk.Tk()
    app = ExtractionApp(root)
    root.mainloop()