<a id="english"></a>

# AI Text Detector

[English](#english) | [中文](#中文)

A fully **local AI-text detection tool** supporting Chinese and English. No internet connection is required, and your text is never uploaded to a server.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **Paragraph-level analysis**: evaluates individual paragraphs while preserving their full context for more reliable results
- **Whole-document scoring**: scores the full document first, then provides a more detailed paragraph-level analysis
- **Colour highlighting**: 🔴 highly likely AI / 🟡 possibly mixed / 🟢 likely human-written
- **Adjustable sensitivity**: set the decision threshold from 10% to 90%; lower values are more stringent
- **Perplexity-assisted detection**: combines GPT-2 perplexity with classifier scores for a second analytical signal
- **Five detector models**: covers Chinese, English, and multilingual use cases
- **CSV export**: exports detection results as a spreadsheet-compatible file
- **Fully offline**: all inference runs locally and your data stays on your computer

---

## Download

Download the following files from the [latest release](https://github.com/lvcheer/AIDetect/releases/latest):

| File | Description |
|------|-------------|
| `AI检测工具-Windows.exe` | Windows application |
| `AI检测工具-mac.zip` | macOS application |
| `models.zip` | Model files shared by Windows and macOS; download once |

> For detailed instructions, see the [User Guide (Chinese)](用户使用指南.md).

### Windows

1. Create a folder, for example `AI检测工具` on your desktop.
2. Put `AI检测工具-Windows.exe` in that folder.
3. Extract `models.zip` and put the resulting `models` folder alongside the executable.
4. Double-click the `.exe` file.

```text
AI检测工具/
├── AI检测工具-Windows.exe
└── models/
```

### macOS

1. Create a folder and extract `AI检测工具-mac.zip` to obtain the `.app`.
2. Extract `models.zip` and put the `models` folder alongside the `.app`.
3. The first time you launch it, **right-click the `.app` → Open → Open** to pass Gatekeeper.

```text
AI检测工具/
├── AI检测工具-mac.app
└── models/
```

---

## Models

| Display name | Hugging Face model | Recommended language |
|--------------|--------------------|----------------------|
| Chinese First (RoBERTa) | `Hello-SimpleAI/chatgpt-detector-roberta-chinese` | Chinese |
| Chinese AIGC v2 | `yuchuantian/AIGC_detector_zhv2` | Chinese |
| English General (OpenAI Detector) | `roberta-base-openai-detector` | English |
| English TMR Detector | `Oxidane/tmr-ai-text-detector` | English |
| Multilingual ChatGPT Detector | `Hello-SimpleAI/chatgpt-detector-roberta` | Chinese and English |

Optional perplexity model: `uer/gpt2-chinese-cluecorpussmall` (downloaded automatically when enabled; approximately 400 MB).

---

## Local Development

### Requirements

- Python 3.11+
- macOS, Windows, or Linux

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/lvcheer/AIDetect.git
cd AIDetect

# 2. Create a virtual environment (Python 3.11 is required)
python3.11 -m venv .venv311
source .venv311/bin/activate   # Windows: .venv311\Scripts\activate

# 3. Install the project and its dependencies
pip install -e .

# 4. Download the classifier models (first run only; approximately 1.5 GB)
python download_models.py

# 5. Run the application
python MainCode.py
```

Run the tests:

```bash
python3.11 -m unittest discover -s tests -v
```

To use pytest, install the test dependencies with `pip install -e ".[test]"`.

### Batch CLI

Input may be JSONL or CSV. Each record must contain a unique `document_id` and a non-empty `text` value. Each run uses one classifier:

```bash
python -m aidetect \
  --input benchmark/input.jsonl \
  --output benchmark/output.jsonl \
  --model roberta-base-openai-detector
```

After installation, you can use the `aidetect` command directly with the same arguments as `python -m aidetect`.

Add `--perplexity` to enable the existing GPT-2 perplexity feature. Output includes the model source, device, raw classifier score, heuristic feature scores, and fused score. These scores are uncalibrated and must not be interpreted as probabilities.

### Benchmark Manifest Splitting

Candidate manifests use JSONL, with each line conforming to `benchmark/dataset_manifest_schema.json`. The input `split` and `evaluation_partition` values are valid placeholders. The script validates records, text SHA-256 values, exact duplicates, and parent-child lineages before replacing those fields. The random seed, split proportions, and held-out generator must be specified explicitly. Proportions are calculated over indivisible source and near-duplicate components. Formal non-held-out data requires at least three components so that the train, calibration, and in-distribution test sets are all non-empty:

```bash
python -m aidetect.manifest \
  --input benchmark/candidate_manifest.jsonl \
  --output benchmark/frozen_manifest.jsonl \
  --metadata-output benchmark/split_metadata.json \
  --schema benchmark/dataset_manifest_schema.json \
  --seed 2026 \
  --train-fraction 0.6 \
  --calibration-fraction 0.2 \
  --held-out-generator generator-id
```

Relative `text_path` values are resolved from the directory containing the input manifest. For a metadata-only manifest whose text cannot be published, use `--skip-text-file-checks` explicitly. This option does not skip schema, ID, lineage, unique-hash, or leakage-safe split validation.

A small manifest used only for end-to-end pipeline validation does not need fabricated formal split parameters. In this mode, every input record must already be labelled `dry_run/pipeline_dry_run`:

```bash
python -m aidetect.manifest \
  --input benchmark/dry_run/candidate_manifest.jsonl \
  --output benchmark/dry_run/frozen_manifest.jsonl \
  --metadata-output benchmark/dry_run/split_metadata.json \
  --schema benchmark/dataset_manifest_schema.json \
  --dry-run-only
```

`--dry-run-only` cannot be combined with a seed, split proportions, or held-out generator. Its output is strictly for pipeline validation and must not be used to support formal performance claims.

### Benchmark Runner

The runner accepts only a frozen manifest whose hash matches the split metadata. Model revisions and the code commit must be full 40-character commit hashes. The AI label index and maximum token length must also be confirmed explicitly:

```bash
python -m aidetect.benchmark_runner \
  --manifest benchmark/frozen_manifest.jsonl \
  --split-metadata benchmark/split_metadata.json \
  --schema benchmark/dataset_manifest_schema.json \
  --output benchmark/results_raw.jsonl \
  --run-metadata-output benchmark/run_metadata.json \
  --run-id baseline-model-1 \
  --code-commit <40-character-git-commit> \
  --model roberta-base-openai-detector \
  --model-revision <40-character-model-commit> \
  --ai-label-index 1 \
  --max-length 512 \
  --include-split dry_run
```

After installation, `python -m aidetect.benchmark_runner` can be replaced with `aidetect-benchmark`. Per-sample results contain the complete class-score vector, raw AI score, character and token lengths, truncation direction, elapsed time, device, and error status. They do not contain the source text or a manually fused score. When classification fails, the raw score is `null`, never zero. `--include-split` is required and repeatable, preventing dry-run and formal partitions from being mixed accidentally. Add `--perplexity --perplexity-revision <commit>` to record perplexity as a separate feature.

### Project Structure

```text
AIDetect/
├── MainCode.py              # GUI entry point
├── pyproject.toml           # Project metadata, dependencies, and CLI entry points
├── aidetect/                # Reusable inference, feature, fusion, schema, and CLI code
├── benchmark/               # Benchmark protocol, manifest schemas, and metrics
├── tests/                   # Unit tests
├── download_models.py       # Downloads all classifier models locally
├── 用户使用指南.md           # End-user guide in Chinese
├── setup_and_run.bat        # One-click Windows launcher
├── models/                  # Local model files (not committed to Git)
├── local-build/
│   └── build.sh             # Local macOS packaging script
└── .github/
    └── workflows/
        └── build-windows.yml  # Automated GitHub Actions builds
```

### Local Packaging on macOS

```bash
python download_models.py   # If the models have not been downloaded
./local-build/build.sh
# Output: local-build/dist/AI检测工具-mac分享包.zip
```

### Automated Builds with GitHub Actions

Pushing to `main` triggers GitHub Actions to build a Windows `.exe` and macOS `.zip`, then publish them to Releases.

The workflow contains four jobs:

- `cleanup-release`: removes the previous release
- `build-models`: packages the cross-platform `models.zip`
- `build-windows`: packages the Windows `.exe`
- `build-macos`: packages the macOS `.app.zip`

---

## Contributing

Contributions are welcome. Here are some possible improvements for different experience levels:

### Beginner Friendly

- [ ] Improve the UI, including layout and dark mode
- [ ] Add support for more languages
- [ ] Improve error messages

### Intermediate

- [ ] Add batch detection for uploaded TXT or DOCX files
- [ ] Visualise detection-score distributions
- [ ] Add detection history

### Advanced

- [ ] Integrate additional open detector models
- [ ] Add GPU acceleration with CUDA or MPS
- [ ] Improve Chinese sentence segmentation
- [ ] Improve language coverage for the perplexity model, which currently targets Chinese

### How to Contribute

1. Fork this repository.
2. Create a feature branch: `git checkout -b feature/your-feature-name`.
3. Commit your changes: `git commit -m 'feat: describe your feature'`.
4. Push the branch: `git push origin feature/your-feature-name`.
5. Open a Pull Request.

Ideas and questions are welcome in [Issues](https://github.com/lvcheer/AIDetect/issues).

---

## License

This project is released under the [MIT License](LICENSE).

---

<a id="中文"></a>

# AI 文本检测工具

[English](#english) | [中文](#中文)

一款完全**本地运行**的 AI 文本检测工具，支持中英文，无需联网，文字不会上传到任何服务器。

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 功能特点

- **逐段分析**：按段落检测，保留完整上下文语义，结果更准确
- **整体评分**：先对全文整体评分，再逐段细化分析
- **颜色高亮**：🔴 高度疑似AI / 🟡 疑似混合 / 🟢 人类写作
- **灵敏度调节**：可自由调整判定阈值（10%–90%），越低越严格
- **困惑度辅助检测**：融合 GPT-2 困惑度与分类器得分，双维度提升准确率
- **5 种检测模型**：覆盖中文、英文、多语言场景
- **导出 CSV**：支持将检测结果导出为表格文件
- **完全离线**：所有推理在本地完成，数据不出本地

---

## 下载使用

前往 [Releases 页面](https://github.com/lvcheer/AIDetect/releases/latest) 下载：

| 文件 | 说明 |
|------|------|
| `AI检测工具-Windows.exe` | Windows 程序本体 |
| `AI检测工具-mac.zip` | macOS 程序本体 |
| `models.zip` | AI 模型文件（Windows / macOS 通用，只需下载一次） |

> 详细使用步骤请查看 [用户使用指南](用户使用指南.md)

### Windows 使用步骤

1. 新建一个文件夹（例如桌面上的 `AI检测工具`）
2. 将 `AI检测工具-Windows.exe` 放入该文件夹
3. 解压 `models.zip`，将 `models` 文件夹也放入该文件夹
4. 双击 `.exe` 运行

```
AI检测工具/
├── AI检测工具-Windows.exe
└── models/
```

### macOS 使用步骤

1. 新建一个文件夹，解压 `AI检测工具-mac.zip` 得到 `.app`
2. 解压 `models.zip`，将 `models` 文件夹与 `.app` 放在同一目录
3. **右键点击 `.app` → 打开 → 打开**（首次需要这样操作以绕过 Gatekeeper）

```
AI检测工具/
├── AI检测工具-mac.app
└── models/
```

---

## 使用的模型

| 显示名称 | HuggingFace 模型 | 适用语言 |
|----------|-----------------|----------|
| 中文优先（RoBERTa） | `Hello-SimpleAI/chatgpt-detector-roberta-chinese` | 中文 |
| 中文新版（AIGC v2） | `yuchuantian/AIGC_detector_zhv2` | 中文 |
| 英文通用（OpenAI Detector） | `roberta-base-openai-detector` | 英文 |
| 英文新版（TMR Detector） | `Oxidane/tmr-ai-text-detector` | 英文 |
| 多语言（ChatGPT Detector） | `Hello-SimpleAI/chatgpt-detector-roberta` | 中英文通用 |

困惑度辅助模型：`uer/gpt2-chinese-cluecorpussmall`（启用时自动下载，约 400MB）

---

## 本地开发

### 环境要求

- Python 3.11+
- macOS / Windows / Linux

### 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/lvcheer/AIDetect.git
cd AIDetect

# 2. 创建虚拟环境（务必使用 Python 3.11）
python3.11 -m venv .venv311
source .venv311/bin/activate   # Windows: .venv311\Scripts\activate

# 3. 安装项目及依赖
pip install -e .

# 4. 下载分类器模型（首次需要，约 1.5GB）
python download_models.py

# 5. 运行
python MainCode.py
```

运行测试：

```bash
python3.11 -m unittest discover -s tests -v
```

如需使用 pytest，可安装测试依赖：`pip install -e ".[test]"`。

### 批量 CLI

输入文件可以是 JSONL 或 CSV，每条记录必须包含唯一的 `document_id` 和
非空 `text`。一次运行使用一个分类器：

```bash
python -m aidetect \
  --input benchmark/input.jsonl \
  --output benchmark/output.jsonl \
  --model roberta-base-openai-detector
```

安装项目后也可以直接使用 `aidetect` 命令，参数与 `python -m aidetect` 相同。

添加 `--perplexity` 可启用现有 GPT-2 困惑度特征。输出包含模型来源、
设备、raw classifier score、启发式特征分数和 fused score。这些分数尚未
校准，不是概率。

### Benchmark manifest 划分

候选 manifest 使用 JSONL，每行须符合
`benchmark/dataset_manifest_schema.json`。输入中的 `split` 和
`evaluation_partition` 是合法占位值；脚本校验记录、文本 SHA-256、精确重复、
父子 lineage 后覆盖这两个字段。随机种子、划分比例及 held-out generator 必须
显式指定。比例按不可拆分的 source/近重复组件计算；正式非 held-out 数据至少
需要三个组件，以保证 train、calibration 和 in-distribution test 均非空：

```bash
python -m aidetect.manifest \
  --input benchmark/candidate_manifest.jsonl \
  --output benchmark/frozen_manifest.jsonl \
  --metadata-output benchmark/split_metadata.json \
  --schema benchmark/dataset_manifest_schema.json \
  --seed 2026 \
  --train-fraction 0.6 \
  --calibration-fraction 0.2 \
  --held-out-generator generator-id
```

相对 `text_path` 默认基于输入 manifest 所在目录解析。仅验证不便公开文本的
metadata-only manifest 时可显式使用 `--skip-text-file-checks`；此选项不会跳过
schema、ID、lineage、哈希唯一性或无泄漏划分检查。

仅验证端到端管线的小型 manifest 不需要伪造正式划分参数。此时所有输入记录必须
已标为 `dry_run/pipeline_dry_run`，使用：

```bash
python -m aidetect.manifest \
  --input benchmark/dry_run/candidate_manifest.jsonl \
  --output benchmark/dry_run/frozen_manifest.jsonl \
  --metadata-output benchmark/dry_run/split_metadata.json \
  --schema benchmark/dataset_manifest_schema.json \
  --dry-run-only
```

`--dry-run-only` 不能与 seed、划分比例或 held-out generator 参数混用，其输出只
用于管线验证，不得用于正式性能结论。

### Benchmark runner

runner 只接受与 split metadata 哈希一致的 frozen manifest。模型 revision 和代码
commit 必须使用完整的 40 位 commit，AI 标签索引及最大 token 长度也必须显式
确认：

```bash
python -m aidetect.benchmark_runner \
  --manifest benchmark/frozen_manifest.jsonl \
  --split-metadata benchmark/split_metadata.json \
  --schema benchmark/dataset_manifest_schema.json \
  --output benchmark/results_raw.jsonl \
  --run-metadata-output benchmark/run_metadata.json \
  --run-id baseline-model-1 \
  --code-commit <40-character-git-commit> \
  --model roberta-base-openai-detector \
  --model-revision <40-character-model-commit> \
  --ai-label-index 1 \
  --max-length 512 \
  --include-split dry_run
```

安装项目后可将 `python -m aidetect.benchmark_runner` 替换为
`aidetect-benchmark`。逐样本结果包含完整类别分数、raw AI score、字符和 token
长度、截断方向、耗时、设备及错误状态，但不包含原文或手工融合分数。分类器失败
时 raw score 为 `null`，不会替换为零。`--include-split` 为必填参数，可重复使用，
从而避免意外混合 dry-run 与正式分区；添加
`--perplexity --perplexity-revision <commit>` 可单独记录困惑度特征。

### 项目结构

```
AIDetect/
├── MainCode.py              # GUI 入口
├── pyproject.toml           # 项目元数据、依赖和 CLI 入口
├── aidetect/                # 可复用推理、特征、融合、schema 和 CLI
├── benchmark/               # benchmark 协议、manifest schema 和指标
├── tests/                   # 单元测试
├── download_models.py       # 下载所有分类器模型到本地
├── 用户使用指南.md           # 面向普通用户的操作说明
├── setup_and_run.bat        # Windows 一键启动脚本
├── models/                  # 本地模型文件（不纳入 git）
├── local-build/
│   └── build.sh             # 本地 macOS 打包脚本
└── .github/
    └── workflows/
        └── build-windows.yml  # GitHub Actions 自动构建
```

### 本地打包（macOS）

```bash
python download_models.py   # 如果还没下载模型
./local-build/build.sh
# 输出：local-build/dist/AI检测工具-mac分享包.zip
```

### 自动构建（GitHub Actions）

推送到 `main` 分支后，Actions 自动构建 Windows `.exe` 和 macOS `.zip` 并发布到 Releases。

包含四个 Job：
- `cleanup-release`：清理旧 Release
- `build-models`：打包 `models.zip`（两平台通用）
- `build-windows`：打包 Windows `.exe`
- `build-macos`：打包 macOS `.app.zip`

---

## 参与开发

欢迎一起完善这个项目！以下是一些可以改进的方向，适合不同水平的贡献者：

### 适合新手
- [ ] 改进 UI 界面（更好的布局、深色模式）
- [ ] 添加更多语言的支持
- [ ] 完善错误提示信息

### 中级难度
- [ ] 支持批量检测（上传 txt / docx 文件）
- [ ] 检测结果可视化（图表展示 AI 检测分数分布）
- [ ] 添加检测历史记录功能

### 进阶方向
- [ ] 接入更多开源检测模型
- [ ] 支持 GPU 加速（CUDA / MPS）
- [ ] 提升中文分句准确率
- [ ] 优化困惑度模型的语言适配（目前仅中文）

### 如何贡献

1. Fork 本项目
2. 创建你的功能分支：`git checkout -b feature/你的功能名`
3. 提交修改：`git commit -m 'feat: 添加某功能'`
4. 推送分支：`git push origin feature/你的功能名`
5. 提交 Pull Request

有任何想法或问题，欢迎直接开 [Issue](https://github.com/lvcheer/AIDetect/issues) 讨论！

---

## License

本项目采用 [MIT License](LICENSE)。
