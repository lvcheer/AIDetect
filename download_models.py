"""
运行此脚本一次，将所有模型下载到本地 models/ 目录。
打包前必须先运行此脚本。

用法：
    python download_models.py
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

from aidetect.models import MODEL_REGISTRY, local_model_path

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def download_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for display_name, model_id in MODEL_REGISTRY.items():
        local_path = local_model_path(MODELS_DIR, model_id)
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
