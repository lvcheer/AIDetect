# AI 文本检测工具

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

# 3. 安装依赖
pip install torch transformers pandas matplotlib

# 4. 下载分类器模型（首次需要，约 1.5GB）
python download_models.py

# 5. 运行
python MainCode.py
```

### 项目结构

```
AIDetect/
├── MainCode.py              # 主程序（GUI + 检测逻辑）
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
- [ ] 检测结果可视化（图表展示 AI 概率分布）
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
KIYA
