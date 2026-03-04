"""
运行此脚本一次，将所有模型下载到本地 models/ 目录。
打包前必须先运行此脚本。

用法：
    python download_models.py
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

MODEL_LIST = {
    "中文优先（RoBERTa）": "Hello-SimpleAI/chatgpt-detector-roberta-chinese",
    "英文优先（OpenAI Detector）": "roberta-base-openai-detector",
    "通用轻量版（ChatGPT Detector）": "Hello-SimpleAI/chatgpt-detector-roberta",
}

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def download_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for display_name, model_id in MODEL_LIST.items():
        local_path = os.path.join(MODELS_DIR, model_id.replace("/", "__"))
        if os.path.exists(local_path):
            print(f"[跳过] {display_name} 已存在：{local_path}")
            continue
        print(f"\n[下载] {display_name}  ({model_id})")
        print("  下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(local_path)
        print("  下载 model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.save_pretrained(local_path)
        print(f"  完成 → {local_path}")

    print("\n所有模型下载完成，可以运行 build.sh 打包了。")


if __name__ == "__main__":
    download_all()
