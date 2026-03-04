import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import re
import os
import sys
import threading

# 确保中文显示正常
import matplotlib
matplotlib.use('Agg')  # 避免tkinter和matplotlib冲突

# 兼容 PyInstaller 打包后的路径
def _get_base_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(_get_base_dir(), "models")

class MultiModelAIDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("多模型AI文本检测工具 - 本地版")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # 初始化变量
        self.detector = None
        self.model_list = {
            "中文优先（RoBERTa）": "Hello-SimpleAI/chatgpt-detector-roberta-chinese",
            "英文优先（OpenAI Detector）": "roberta-base-openai-detector",
            "通用轻量版（ChatGPT Detector）": "Hello-SimpleAI/chatgpt-detector-roberta",
        }
        # 将 HuggingFace model_id 映射到本地目录名（与 download_models.py 保持一致）
        self._local_model_path = lambda model_id: os.path.join(
            MODELS_DIR, model_id.replace("/", "__")
        )
        self.current_model = tk.StringVar(value=list(self.model_list.keys())[0])
        self.is_detecting = False
        
        # 创建UI布局
        self._create_widgets()
        
        # 预加载模型（后台线程，不卡界面）
        self._load_model_in_background()

    def _create_widgets(self):
        """创建GUI界面组件"""
        # 1. 顶部设置区
        setting_frame = ttk.LabelFrame(self.root, text="检测设置")
        setting_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 模型选择
        ttk.Label(setting_frame, text="选择检测模型：").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        model_combobox = ttk.Combobox(
            setting_frame, 
            textvariable=self.current_model,
            values=list(self.model_list.keys()),
            state="readonly",
            width=30
        )
        model_combobox.grid(row=0, column=1, padx=5, pady=5)
        model_combobox.bind("<<ComboboxSelected>>", self._on_model_change)
        
        # 2. 文本输入区
        input_frame = ttk.LabelFrame(self.root, text="待检测文本")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.text_input = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, font=("SimHei", 10))
        self.text_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 3. 操作按钮区
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.detect_btn = ttk.Button(
            btn_frame, 
            text="开始检测", 
            command=self._start_detection,
            state="normal"
        )
        self.detect_btn.pack(side=tk.LEFT, padx=5)
        
        self.export_btn = ttk.Button(
            btn_frame, 
            text="导出结果", 
            command=self._export_results,
            state="disabled"
        )
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(
            btn_frame, 
            text="清空文本", 
            command=lambda: self.text_input.delete(1.0, tk.END)
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # 4. 结果展示区
        result_frame = ttk.LabelFrame(self.root, text="检测结果（红色=高概率AI，黄色=疑似，绿色=人类）")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.result_text = scrolledtext.ScrolledText(
            result_frame, 
            wrap=tk.WORD, 
            font=("SimHei", 10),
            state=tk.DISABLED
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 5. 状态提示
        self.status_var = tk.StringVar(value="状态：就绪 - 模型加载中...")
        status_label = ttk.Label(self.root, textvariable=self.status_var)
        status_label.pack(anchor=tk.W, padx=10, pady=2)

        # 预注册颜色标签（必须在主线程执行）
        self._init_color_tags()

    def _load_model_in_background(self):
        """后台加载模型，避免界面卡顿"""
        def load_model():
            try:
                model_id = self.model_list[self.current_model.get()]
                local_path = self._local_model_path(model_id)

                # 优先使用本地模型，本地不存在时从网络下载
                if os.path.exists(local_path):
                    source = local_path
                    self.root.after(0, lambda: self.status_var.set("状态：加载中 - 读取本地模型..."))
                else:
                    source = model_id
                    self.root.after(0, lambda: self.status_var.set("状态：加载中 - 本地模型不存在，从网络下载..."))

                self.tokenizer = AutoTokenizer.from_pretrained(source)
                self.model = AutoModelForSequenceClassification.from_pretrained(source)

                # 设置设备
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                self.model.eval()

                # 自动检测哪个标签对应 AI（不同模型标签顺序不同）
                ai_keywords = {"fake", "chatgpt", "ai", "machine", "generated"}
                self.ai_label_idx = 1  # 默认
                for idx, label in self.model.config.id2label.items():
                    if label.lower() in ai_keywords:
                        self.ai_label_idx = idx
                        break

                self.root.after(0, lambda: self.status_var.set(
                    f"状态：就绪 - 模型加载完成（{self.device}）"
                ))
            except Exception as e:
                msg = str(e)
                self.root.after(0, lambda: self.status_var.set(f"状态：错误 - 模型加载失败：{msg[:50]}"))
                self.root.after(0, lambda: messagebox.showerror("模型加载失败", f"请检查本地模型目录或网络：{msg}"))

        threading.Thread(target=load_model, daemon=True).start()

    def _on_model_change(self, event):
        """切换模型时重新加载"""
        self.status_var.set("状态：加载中 - 切换模型，请稍候...")
        self._load_model_in_background()

    def _split_text(self, text):
        """分句处理，支持中英文"""
        # 中文分句
        cn_sentences = re.split(r'[。！？；]', text)
        cn_separators = re.findall(r'[。！？；]', text)
        sentences = [s.strip() + sep for s, sep in zip(cn_sentences, cn_separators) if s.strip()]
        
        # 如果没有中文分句，用英文分句
        if not sentences:
            sentences = re.split(r'[.!?;]', text)
            en_separators = re.findall(r'[.!?;]', text)
            sentences = [s.strip() + sep for s, sep in zip(sentences, en_separators) if s.strip()]
        
        # 兜底：按换行/空格分割
        if not sentences:
            sentences = [line.strip() for line in text.split('\n') if line.strip()]
        
        return sentences

    def _detect_sentence(self, sentence):
        """检测单句AI概率"""
        try:
            inputs = self.tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            ai_idx = getattr(self, 'ai_label_idx', 1)
            human_idx = 1 - ai_idx
            ai_prob = probabilities[0][ai_idx].item() * 100
            human_prob = probabilities[0][human_idx].item() * 100
            
            return {
                "sentence": sentence,
                "ai_prob": round(ai_prob, 2),
                "human_prob": round(human_prob, 2),
                "is_ai": ai_prob > 50
            }
        except Exception as e:
            return {
                "sentence": sentence,
                "ai_prob": 0.0,
                "human_prob": 0.0,
                "is_ai": False,
                "error": str(e)
            }

    def _generate_explanation(self, ai_prob):
        """生成检测原因解释"""
        if ai_prob < 30:
            return "文本符合人类写作特征：语言表达自然，有正常的逻辑波动，无明显AI生成痕迹。"
        elif 30 <= ai_prob < 70:
            return "文本疑似混合生成：部分语句结构规整度较高，存在少量AI生成特征，但仍有人类表达痕迹。"
        else:
            return "文本高度疑似AI生成：语言表达过于规范，语句模板化明显，缺少人类写作的情感和逻辑瑕疵。"

    def _start_detection(self):
        """开始检测（后台线程执行，避免界面卡死）"""
        if self.is_detecting:
            return

        # 检查模型是否已加载完成
        if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
            messagebox.showwarning("提示", "模型尚未加载完成，请稍候再试！")
            return

        text = self.text_input.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "请输入待检测的文本！")
            return
        
        self.is_detecting = True
        self.detect_btn.config(state="disabled")
        self.status_var.set("状态：检测中 - 正在分析文本，请稍候...")
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        # 线程安全的UI更新辅助函数
        def ui_insert(content, tag=None):
            def _do():
                if tag:
                    self.result_text.insert(tk.END, content, tag)
                else:
                    self.result_text.insert(tk.END, content)
            self.root.after(0, _do)

        def ui_set_status(msg):
            self.root.after(0, lambda: self.status_var.set(msg))

        def ui_finish(overall_ai, results):
            def _do():
                self.detection_results = {
                    "overall_ai_rate": overall_ai,
                    "sentence_results": results
                }
                self.status_var.set(f"状态：完成 - 检测结束，整体AI概率：{overall_ai}%")
                self.export_btn.config(state="normal")
                self.is_detecting = False
                self.detect_btn.config(state="normal")
                self.result_text.config(state=tk.DISABLED)
            self.root.after(0, _do)

        def ui_error(msg):
            def _do():
                self.result_text.insert(tk.END, f"\n❌ 检测出错：{msg}\n")
                self.status_var.set(f"状态：错误 - {msg[:50]}")
                self.is_detecting = False
                self.detect_btn.config(state="normal")
                self.result_text.config(state=tk.DISABLED)
            self.root.after(0, _do)

        # 后台执行检测
        def detect_task():
            try:
                # 1. 分句
                sentences = self._split_text(text)
                if not sentences:
                    raise ValueError("文本无法分割，请检查格式")

                # 2. 逐句检测
                results = []
                total_ai_prob = 0.0

                for idx, sentence in enumerate(sentences, 1):
                    res = self._detect_sentence(sentence)
                    res["explanation"] = self._generate_explanation(res["ai_prob"])
                    results.append(res)
                    total_ai_prob += res["ai_prob"]

                    # 通过 root.after 调度 UI 更新到主线程
                    color_tag = self._get_color_tag(res["ai_prob"])
                    ui_insert(f"\n【第{idx}句】{res['sentence']}\n", color_tag)
                    ui_insert(
                        f"AI概率：{res['ai_prob']}% | 人类概率：{res['human_prob']}%\n"
                        f"原因：{res['explanation']}\n{'-'*80}\n"
                    )
                    ui_set_status(f"状态：检测中 - 已处理 {idx}/{len(sentences)} 句")

                # 3. 计算整体AI率
                overall_ai = round(total_ai_prob / len(results), 2)
                conclusion = (
                    '高度疑似AI生成' if overall_ai > 70
                    else '疑似混合生成' if 30 <= overall_ai <= 70
                    else '基本判定为人类生成'
                )
                ui_insert(
                    f"\n整体检测结果：\n"
                    f"文本整体AI生成概率：{overall_ai}%\n"
                    f"结论：{conclusion}\n"
                )
                ui_finish(overall_ai, results)

            except Exception as e:
                ui_error(str(e))

        threading.Thread(target=detect_task, daemon=True).start()

    def _init_color_tags(self):
        """在主线程中预先注册所有颜色标签"""
        self.result_text.tag_configure("red", foreground="red", font=("SimHei", 10, "bold"))
        self.result_text.tag_configure("yellow", foreground="orange", font=("SimHei", 10))
        self.result_text.tag_configure("green", foreground="green", font=("SimHei", 10))

    def _get_color_tag(self, ai_prob):
        """根据AI概率返回颜色标签"""
        if ai_prob > 70:
            return "red"
        elif 30 <= ai_prob <= 70:
            return "yellow"
        else:
            return "green"

    def _export_results(self):
        """导出检测结果到CSV文件"""
        if not hasattr(self, 'detection_results'):
            messagebox.showwarning("提示", "暂无检测结果可导出！")
            return
        
        # 选择保存路径
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
            title="导出检测结果"
        )
        if not file_path:
            return
        
        try:
            # 整理结果为DataFrame
            df = pd.DataFrame(self.detection_results["sentence_results"])
            # 添加整体结果行
            overall_row = pd.DataFrame({
                "sentence": ["【整体结果】"],
                "ai_prob": [self.detection_results["overall_ai_rate"]],
                "human_prob": [100 - self.detection_results["overall_ai_rate"]],
                "is_ai": [self.detection_results["overall_ai_rate"] > 50],
                "explanation": ["整体AI生成概率计算结果"]
            })
            df = pd.concat([overall_row, df], ignore_index=True)
            
            # 保存文件
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
            messagebox.showinfo("成功", f"结果已导出到：{file_path}")
        except Exception as e:
            messagebox.showerror("导出失败", f"保存文件出错：{e}")

if __name__ == "__main__":
    # 创建主窗口
    root = tk.Tk()
    app = MultiModelAIDetectorGUI(root)
    
    # 运行GUI
    root.mainloop()
