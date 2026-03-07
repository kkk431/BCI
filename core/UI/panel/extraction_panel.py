# -*- coding: utf-8 -*-
# isort: skip_file
# flake8: noqa
"""
智融脑机 - 独立特征提取模块 GUI
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import traceback
import numpy as np
from pathlib import Path
import sys

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

# --- 导入底层业务模块---
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

# --- 视觉配置 ---
COLOR_CONTENT_BG = "white"
COLOR_BTN_BG = "#4f8080"
COLOR_BTN_FG = "white"
COLOR_TEXT_MAIN = "#333333"
COLOR_TEXT_SUB = "#666666"
FONT_TITLE = ("微软雅黑", 16, "bold")
FONT_NORMAL = ("微软雅黑", 14)
FONT_BTN = ("Arial", 14, "bold")

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


class ExtractionApp(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        # 调用父类初始化，将当前 Frame 放入 parent 中
        super().__init__(parent, *args, **kwargs)
        self.parent = parent  # 保存父容器，但后续主要使用 self.master 或直接 self

        # 核心数据容器
        self.clean_data_dict = None
        self.extracted_features = None
        self.checkbox_vars = {}
        self.current_filepath = None  # 保存当前文件路径，用于重新处理

        if not MODULES_LOADED:
            messagebox.showerror("初始化失败", "底层算法模块导入失败，请检查终端输出的路径信息。")

        self.setup_ui()

    def setup_ui(self):
        """构建整体 UI 布局（所有控件都放在 self 上）"""
        # 主容器直接使用 self（因为 self 就是 Frame）
        main_container = tk.Frame(self, bg=COLOR_CONTENT_BG, padx=30, pady=20)
        main_container.pack(fill="both", expand=True)

        # 标题
        title_lbl = tk.Label(main_container, text="多模态信号特征提取", font=("微软雅黑", 20, "bold"),
                             bg=COLOR_CONTENT_BG, fg=COLOR_TEXT_MAIN)
        title_lbl.pack(anchor="w", pady=(0, 20))

        # ================= 区域1：数据加载 =================
        frame_step1 = tk.Frame(main_container, bg=COLOR_CONTENT_BG)
        frame_step1.pack(fill="x", pady=10)

        lbl_step1 = tk.Label(frame_step1, text="1. 加载数据源", font=FONT_TITLE, bg=COLOR_CONTENT_BG,
                             fg=COLOR_TEXT_MAIN)
        lbl_step1.pack(anchor="w", pady=(0, 10))

        btn_box1 = tk.Frame(frame_step1, bg=COLOR_CONTENT_BG)
        btn_box1.pack(fill="x")

        self.btn_load = tk.Button(btn_box1, text="📁 上传本地数据文件", font=FONT_BTN, bg=COLOR_BTN_BG, fg=COLOR_BTN_FG,
                                  relief="flat", padx=15, pady=6, cursor="hand2", command=self.action_load_data)
        self.btn_load.pack(side="left")

        self.lbl_status = tk.Label(btn_box1, text="请上传文件 (后台将自动完成解析与降噪)...", font=FONT_NORMAL,
                                   bg=COLOR_CONTENT_BG, fg=COLOR_TEXT_SUB)
        self.lbl_status.pack(side="left", padx=15)

        # ================= 手动选择模态区域（初始隐藏） =================
        self.manual_mod_frame = tk.Frame(frame_step1, bg=COLOR_CONTENT_BG)

        tk.Label(self.manual_mod_frame, text="手动指定模态类型:", font=FONT_NORMAL,
                 bg=COLOR_CONTENT_BG, fg=COLOR_TEXT_MAIN).pack(side="left", padx=(0, 10))

        self.manual_mod_var = tk.StringVar()
        self.manual_mod_combo = ttk.Combobox(self.manual_mod_frame, textvariable=self.manual_mod_var,
                                             state="readonly", width=10, font=FONT_NORMAL,
                                             values=["EEG", "EMG", "ECG", "fNIRS"])
        self.manual_mod_combo.pack(side="left", padx=5)
        self.manual_mod_combo.current(0)

        self.btn_confirm_mod = tk.Button(self.manual_mod_frame, text="确认选择", font=FONT_BTN,
                                         bg="#28a745", fg=COLOR_BTN_FG, relief="flat", padx=10, pady=2,
                                         cursor="hand2", command=self.action_confirm_manual_mod)
        self.btn_confirm_mod.pack(side="left", padx=10)

        # 初始隐藏手动选择区域
        self.manual_mod_frame.pack_forget()

        tk.Frame(main_container, bg="#e0e0e0", height=1).pack(fill="x", pady=15)

        # ================= 区域2：特征勾选与提取 =================
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

        self.btn_extract = tk.Button(frame_step2, text="⚡ 提取已勾选特征", font=FONT_BTN, bg="#b9dcff",
                                     fg=COLOR_BTN_FG,
                                     relief="flat", padx=15, pady=6, cursor="hand2", command=self.action_extract,
                                     state="disabled")
        self.btn_extract.pack(anchor="w")

        tk.Frame(main_container, bg="#e0e0e0", height=1).pack(fill="x", pady=15)

        # ================= 区域3：结果展示 =================
        frame_step3 = tk.Frame(main_container, bg=COLOR_CONTENT_BG)
        frame_step3.pack(fill="both", expand=True)

        lbl_step3 = tk.Label(frame_step3, text="3. 提取结果查看", font=FONT_TITLE, bg=COLOR_CONTENT_BG,
                             fg=COLOR_TEXT_MAIN)
        lbl_step3.pack(anchor="w", pady=(0, 10))

        table_frame = tk.Frame(frame_step3, bg=COLOR_CONTENT_BG)
        table_frame.pack(fill="both", expand=True)

        columns = ("Name", "Value")

        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("Name", text="特征名称", anchor="w", )
        self.tree.heading("Value", text="特征数值", anchor="w")
        self.tree.column("Name", width=400, anchor="w")
        self.tree.column("Value", width=300, anchor="w")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # ================= 业务逻辑 =================

    def action_load_data(self):
        if not MODULES_LOADED:
            messagebox.showerror("错误", "底层模块未加载，请检查文件路径是否正确。")
            return

        filepath = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("支持的格式", "*.csv *.mat *.edf *.bdf *.npy *.npz *.snirf *.set *.pkl *.json"),
                       ("All Files", "*.*")]
        )
        if not filepath:
            return

        self.current_filepath = filepath
        self._start_data_processing(filepath)

    def _start_data_processing(self, filepath, manual_mod=None):
        """启动数据处理流程，可指定手动模态"""
        self.btn_load.config(state="disabled", bg="#a0a0a0", cursor="arrow")
        self.btn_extract.config(state="disabled", bg="#a0a0a0", cursor="arrow")

        if manual_mod:
            self.lbl_status.config(text=f"正在以【{manual_mod}】模式重新处理数据，请稍候...", fg="#0056b3")
        else:
            self.lbl_status.config(text="正在读取文件并进行后台降噪处理，这可能需要几秒钟，请稍候...", fg="#0056b3")

        # 隐藏手动选择区域
        self.manual_mod_frame.pack_forget()

        for item in self.tree.get_children():
            self.tree.delete(item)

        def background_task():
            try:
                loader = DataLoader()
                raw_dict = loader.load(filepath)

                # 如果指定了手动模态，替换 UNKNOWN 模态
                if manual_mod:
                    # 检查是否有 UNKNOWN 模态需要替换
                    if 'signal' in raw_dict:
                        # 创建一个新的信号字典
                        new_signal_dict = {}

                        # 遍历所有模态
                        for mod_name, signal_data in raw_dict['signal'].items():
                            if mod_name == 'UNKNOWN':
                                # 将 UNKNOWN 替换为手动选择的模态
                                print(f"替换 UNKNOWN 模态为 {manual_mod}")

                                # 如果信号数据是字典格式
                                if isinstance(signal_data, dict):
                                    signal_data['signal_type'] = manual_mod.lower()
                                    new_signal_dict[manual_mod] = signal_data
                                else:
                                    # 如果是直接的数据数组，包装成标准格式
                                    fs = 250  # 默认采样率
                                    if 'fs' in raw_dict:
                                        fs = raw_dict['fs']

                                    # 获取数据形状
                                    if hasattr(signal_data, 'shape'):
                                        if len(signal_data.shape) == 1:
                                            n_channels = 1
                                            n_samples = signal_data.shape[0]
                                        else:
                                            n_channels, n_samples = signal_data.shape

                                        signal_entry = {
                                            'data': signal_data,
                                            'sampling_rate': fs,
                                            'channel_names': [f'Ch{i + 1}' for i in range(n_channels)],
                                            'signal_type': manual_mod.lower(),
                                            'unit': 'unknown',
                                            'n_channels': n_channels,
                                            'n_samples': n_samples,
                                            'duration': n_samples / fs
                                        }
                                        new_signal_dict[manual_mod] = signal_entry
                            else:
                                # 保留其他模态
                                new_signal_dict[mod_name] = signal_data

                        # 更新信号字典
                        raw_dict['signal'] = new_signal_dict

                        # 更新meta信息
                        if 'meta' not in raw_dict:
                            raw_dict['meta'] = {}

                        # 更新modality列表
                        raw_dict['meta']['modality'] = list(new_signal_dict.keys())

                        print(f"已将 UNKNOWN 模态替换为 {manual_mod}")

                prep_config = MultiModalConfigFactory.create_resting_state_config()
                preprocessor = MultiModalPreprocessor(prep_config)
                result = preprocessor.process(raw_dict)

                if not result.success:
                    raise Exception(f"后台预处理去噪失败: {result.error_message}")

                self.clean_data_dict = result.processed_data

                mods = list(self.clean_data_dict.get('signal', {}).keys())
                valid_mods = [m for m in mods if m in FEATURE_MAP]

                if valid_mods:
                    self.after(0, self._ui_update_on_load_success, valid_mods)
                else:
                    self.after(0, self._ui_update_on_load_warning)

            except Exception as e:
                traceback.print_exc()
                self.after(0, self._ui_update_on_error, f"处理失败: {str(e)}")

        threading.Thread(target=background_task, daemon=True).start()
    def action_confirm_manual_mod(self):
        """确认手动选择的模态"""
        selected_mod = self.manual_mod_var.get()
        if not selected_mod:
            messagebox.showwarning("警告", "请选择模态类型")
            return

        if not self.current_filepath:
            messagebox.showwarning("警告", "请先上传数据文件")
            return

        # 使用手动选择的模态重新处理数据
        self._start_data_processing(self.current_filepath, selected_mod)

    def _ui_update_on_load_success(self, valid_mods):
        self.combo_modality['values'] = valid_mods
        self.combo_modality.current(0)
        self.refresh_checkboxes(None)

        self.lbl_status.config(text=f"数据就绪！已净化并识别出模态: {', '.join(valid_mods)}", fg="green")
        self.btn_extract.config(state="normal", bg=COLOR_BTN_BG, cursor="hand2")
        self.btn_load.config(state="normal", bg=COLOR_BTN_BG, cursor="hand2")

        # 确保手动选择区域隐藏
        self.manual_mod_frame.pack_forget()

    def _ui_update_on_load_warning(self):
        self.lbl_status.config(text="数据已加载，但未检测到支持提取特征的有效模态。请手动选择模态类型并点击确认。",
                               fg="orange")
        self.btn_load.config(state="normal", bg=COLOR_BTN_BG, cursor="hand2")

        # 显示手动选择模态的区域
        self.manual_mod_frame.pack(after=self.lbl_status.master, fill="x", pady=(5, 0))

    def _ui_update_on_error(self, error_msg):
        self.lbl_status.config(text=error_msg, fg="red")
        self.btn_load.config(state="normal", bg=COLOR_BTN_BG, cursor="hand2")
        self.btn_extract.config(state="disabled", bg="#a0a0a0", cursor="arrow")

        # 错误时隐藏手动选择区域
        self.manual_mod_frame.pack_forget()

    def refresh_checkboxes(self, event):
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
        mod = self.modality_var.get()
        if not mod or not self.clean_data_dict:
            return

        selected_cats = [key for key, var in self.checkbox_vars.items() if var.get()]
        if not selected_cats:
            messagebox.showwarning("提示", "请至少勾选一种特征集！")
            return

        self.btn_extract.config(state="disabled", bg="#a0a0a0", cursor="arrow")
        self.btn_load.config(state="disabled", bg="#a0a0a0", cursor="arrow")
        self.lbl_status.config(text=f"正在精确提取 {mod} 的特征，请稍候...", fg="#0056b3")

        def background_extraction():
            try:
                request = {mod: selected_cats}
                pipeline = MultimodalFeaturePipeline(self.clean_data_dict, selected_features=request)
                final_dict = pipeline.run_pipeline()

                all_feats = final_dict.get('processed', {}).get('feature', {})
                self.extracted_features = {mod: all_feats.get(mod, {})}

                self.after(0, self._ui_update_on_extract_success, mod)

            except Exception as e:
                traceback.print_exc()
                self.after(0, self._ui_update_on_error, f"特征提取异常: {str(e)}")

        threading.Thread(target=background_extraction, daemon=True).start()

    def _ui_update_on_extract_success(self, mod):
        self.lbl_status.config(text=f"【{mod}】所选特征提取完毕！", fg="green")
        self.btn_extract.config(state="normal", bg=COLOR_BTN_BG, cursor="hand2")
        self.btn_load.config(state="normal", bg=COLOR_BTN_BG, cursor="hand2")
        self.render_table(mod)

    def render_table(self, mod):
        for item in self.tree.get_children():
            self.tree.delete(item)

        mod_feats = self.extracted_features.get(mod, {})

        flat = {}
        for k, v in mod_feats.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    flat[f"[{k}] {sk}"] = sv
            else:
                flat[k] = v

        for name, val in flat.items():
            if isinstance(val, float):
                val_str = f"{val:.6f}"
            elif isinstance(val, (list, np.ndarray)):
                val_str = f"Array {np.shape(val)}: {str(val)[:30]}..."
            else:
                val_str = str(val)

            self.tree.insert('', "end", values=(name, val_str))


if __name__ == "__main__":
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    root = tk.Tk()
    root.title("智融脑机 - 特征提取模块")
    root.geometry("1000x700")
    root.configure(bg=COLOR_CONTENT_BG)

    app = ExtractionApp(root)
    app.pack(fill=tk.BOTH, expand=True)

    root.mainloop()