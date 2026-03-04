import sys
import os

# 修复 Windows --windowed 模式下 stdout/stderr 为 None 导致的报错
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import torch
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import pandas as pd
import re
import threading

# 确保中文显示正常
import matplotlib
matplotlib.use('Agg')  # 避免tkinter和matplotlib冲突

# 模型目录：打包版查找可执行文件旁边，开发版查找脚本目录
def _get_models_dir():
    if getattr(sys, 'frozen', False):
        exe_path = sys.executable
        # macOS .app bundle：executable 在 MyApp.app/Contents/MacOS/ 内
        # 模型放在 .app 同级目录下的 models/
        parts = exe_path.replace('\\', '/').split('/')
        app_idx = next((i for i, p in enumerate(parts) if p.endswith('.app')), None)
        if app_idx is not None:
            base = '/'.join(parts[:app_idx])
        else:
            # Windows .exe：模型放在 .exe 同级目录下的 models/
            base = os.path.dirname(exe_path)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "models")

MODELS_DIR = _get_models_dir()

class MultiModelAIDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("多模型AI文本检测工具 - 本地版")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # 初始化变量
        self.detector = None
        self.model_list = {
            "中文优先（RoBERTa）":         "Hello-SimpleAI/chatgpt-detector-roberta-chinese",
            "中文新版（AIGC v2）":          "yuchuantian/AIGC_detector_zhv2",
            "英文通用（OpenAI Detector）":  "roberta-base-openai-detector",
            "英文新版（TMR Detector）":     "Oxidane/tmr-ai-text-detector",
            "多语言（ChatGPT Detector）":   "Hello-SimpleAI/chatgpt-detector-roberta",
        }
        # 将 HuggingFace model_id 映射到本地目录名（与 download_models.py 保持一致）
        self._local_model_path = lambda model_id: os.path.join(
            MODELS_DIR, model_id.replace("/", "__")
        )
        self.current_model = tk.StringVar(value=list(self.model_list.keys())[0])
        self.is_detecting = False
        # 灵敏度阈值：高于此值判定为AI（默认50%）
        self.threshold = tk.IntVar(value=50)
        # 困惑度辅助检测开关
        self.use_perplexity = tk.BooleanVar(value=False)
        # 困惑度模型（GPT-2中文，懒加载）
        self.ppl_tokenizer = None
        self.ppl_model = None
        
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

        # 灵敏度滑块
        ttk.Label(setting_frame, text="判定灵敏度（AI阈值）：").grid(row=0, column=2, padx=(20, 5), pady=5, sticky=tk.W)
        threshold_slider = ttk.Scale(
            setting_frame, from_=10, to=90,
            orient=tk.HORIZONTAL, length=160,
            variable=self.threshold
        )
        threshold_slider.grid(row=0, column=3, padx=5, pady=5)
        self.threshold_label = ttk.Label(setting_frame, text="50%  ← 越低越严格")
        self.threshold_label.grid(row=0, column=4, padx=5, pady=5)
        self.threshold.trace_add("write", self._on_threshold_change)

        # 困惑度辅助开关
        ppl_check = ttk.Checkbutton(
            setting_frame,
            text="启用困惑度辅助检测（更准确，首次需下载~400MB）",
            variable=self.use_perplexity,
            command=self._on_perplexity_toggle
        )
        ppl_check.grid(row=1, column=0, columnspan=5, padx=5, pady=2, sticky=tk.W)

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
        result_frame = ttk.LabelFrame(self.root, text="检测结果（按段落展示：红色=高概率AI，黄色=疑似，绿色=人类）")
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

    def _on_model_change(self, _event):
        """切换模型时重新加载"""
        self.status_var.set("状态：加载中 - 切换模型，请稍候...")
        self._load_model_in_background()

    def _on_threshold_change(self, *_args):
        """阈值变化时更新标签显示"""
        val = self.threshold.get()
        self.threshold_label.config(text=f"{val}%  ← 越低越严格")

    def _on_perplexity_toggle(self):
        """开启困惑度时懒加载 GPT-2 中文模型"""
        if self.use_perplexity.get() and self.ppl_model is None:
            self.status_var.set("状态：加载中 - 正在下载困惑度模型（GPT-2中文，约400MB）...")
            def load_ppl():
                try:
                    ppl_model_id = "uer/gpt2-chinese-cluecorpussmall"
                    self.ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_id)
                    self.ppl_model = AutoModelForCausalLM.from_pretrained(ppl_model_id)
                    self.ppl_model.to(self.device if hasattr(self, 'device') else 'cpu')
                    self.ppl_model.eval()
                    self.root.after(0, lambda: self.status_var.set("状态：就绪 - 困惑度模型加载完成"))
                except Exception as e:
                    msg = str(e)
                    self.root.after(0, lambda: self.use_perplexity.set(False))
                    self.root.after(0, lambda: messagebox.showerror("加载失败", f"困惑度模型下载失败：{msg}"))
            threading.Thread(target=load_ppl, daemon=True).start()

    def _calculate_perplexity_score(self, text):
        """用 GPT-2 计算困惑度并转换为AI概率（困惑度低=AI概率高）"""
        try:
            inputs = self.ppl_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.ppl_model.device)
            input_ids = inputs["input_ids"]
            with torch.no_grad():
                loss = self.ppl_model(input_ids, labels=input_ids).loss
            perplexity = torch.exp(loss).item()
            # sigmoid 转换：困惑度中心点约 40，越低越像AI
            ai_prob = 1 / (1 + math.exp((perplexity - 40) / 12)) * 100
            return round(ai_prob, 2), round(perplexity, 2)
        except Exception:
            return None, None

    def _split_text(self, text):
        """按段落分割文本，保留完整语义单元"""
        # 优先按空行分段（双换行）
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

        # 没有空行时按单换行分段
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        # 兜底：整段文字无换行，按中文句号/英文句号分句
        if len(paragraphs) <= 1:
            cn_sentences = re.split(r'[。！？；]', text)
            cn_separators = re.findall(r'[。！？；]', text)
            paragraphs = [s.strip() + sep for s, sep in zip(cn_sentences, cn_separators) if s.strip()]

        if not paragraphs:
            paragraphs = [text.strip()]

        return paragraphs

    def _detect_sentence(self, sentence):
        """检测文本片段AI概率（sentence 可以是带上下文的窗口文本）"""
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

                # 2. 整体检测：全文一次性送入模型，得到最准确的整体得分
                ui_set_status("状态：检测中 - 分析整体文本...")
                overall_res = self._detect_sentence(text)
                overall_ai = overall_res["ai_prob"]
                overall_ppl_info = ""
                use_ppl_overall = self.use_perplexity.get() and self.ppl_model is not None
                if use_ppl_overall:
                    ppl_ai_prob, ppl_value = self._calculate_perplexity_score(text)
                    if ppl_ai_prob is not None:
                        overall_ai = round(0.6 * overall_ai + 0.4 * ppl_ai_prob, 2)
                        overall_ppl_info = f"困惑度：{ppl_value} | "

                # 3. 逐段检测：每段作为完整语义单元送入模型
                use_ppl = use_ppl_overall
                results = []
                for idx, paragraph in enumerate(sentences, 1):
                    res = self._detect_sentence(paragraph)
                    res["sentence"] = paragraph

                    # 如果开启困惑度辅助，融合两个得分
                    if use_ppl:
                        ppl_ai_prob, ppl_value = self._calculate_perplexity_score(paragraph)
                        if ppl_ai_prob is not None:
                            res["ai_prob"] = round(0.6 * res["ai_prob"] + 0.4 * ppl_ai_prob, 2)
                            res["human_prob"] = round(100 - res["ai_prob"], 2)
                            res["perplexity"] = ppl_value

                    res["explanation"] = self._generate_explanation(res["ai_prob"])
                    results.append(res)

                    color_tag = self._get_color_tag(res["ai_prob"])
                    preview = paragraph[:60] + "..." if len(paragraph) > 60 else paragraph
                    ui_insert(f"\n【第{idx}段】{preview}\n", color_tag)
                    ppl_info = f" | 困惑度：{res.get('perplexity', '-')}" if use_ppl else ""
                    ui_insert(
                        f"AI概率：{res['ai_prob']}%{ppl_info} | 人类概率：{res['human_prob']}%\n"
                        f"原因：{res['explanation']}\n{'-'*80}\n"
                    )
                    ui_set_status(f"状态：检测中 - 已处理 {idx}/{len(sentences)} 段")

                # 4. 整体结论（用全文检测得分，而非逐句平均）
                t = self.threshold.get()
                conclusion = (
                    '高度疑似AI生成' if overall_ai >= t
                    else '疑似混合生成' if overall_ai >= t // 2
                    else '基本判定为人类生成'
                )
                ui_insert(
                    f"\n整体检测结果（全文分析）：\n"
                    f"{overall_ppl_info}综合AI生成概率：{overall_ai}%\n"
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
        """根据AI概率和当前阈值返回颜色标签"""
        t = self.threshold.get()
        mid = t // 2  # 黄色区下界 = 阈值的一半
        if ai_prob >= t:
            return "red"
        elif ai_prob >= mid:
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
    # 防止 PyInstaller 在 macOS 上 multiprocessing fork 导致无限开窗
    import multiprocessing
    multiprocessing.freeze_support()

    root = tk.Tk()
    app = MultiModelAIDetectorGUI(root)
    root.mainloop()
