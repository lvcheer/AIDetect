# AI 文本检测工具

一款完全**本地运行**的 AI 文本检测工具，支持中英文，无需联网，文字不会上传到任何服务器。

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 功能特点

- 逐句分析，实时显示检测结果
- 颜色高亮：🔴 高度疑似AI / 🟡 疑似混合 / 🟢 人类写作
- 支持三种检测模型（中文 / 英文 / 通用）
- 支持导出检测结果为 CSV 文件
- 完全离线运行，数据不出本地

---

## 下载使用

前往 [Releases 页面](https://github.com/lvcheer/AIDetect/releases/latest) 下载：

| 文件 | 适用系统 |
|------|----------|
| `AI检测工具-Windows.exe` | Windows |
| `AI检测工具-mac.zip` | macOS |
| `models.zip` | 两平台通用模型文件 |

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

## 本地开发

### 环境要求

- Python 3.11+
- macOS / Windows / Linux

### 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/lvcheer/AIDetect.git
cd AIDetect

# 2. 创建虚拟环境
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install torch transformers pandas matplotlib

# 4. 下载模型（首次需要，约 1.5GB）
python download_models.py

# 5. 运行
python MainCode.py
```

### 项目结构

```
AIDetect/
├── MainCode.py              # 主程序（GUI + 检测逻辑）
├── download_models.py       # 下载所有模型到本地
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

包含三个并行 Job：
- `cleanup-release`：清理旧 Release
- `build-windows`：打包 Windows 版
- `build-macos`：打包 macOS 版
- `build-models`：打包 `models.zip`（两平台通用）

---

## 使用的模型

| 模型名称 | 来源 | 适用语言 |
|----------|------|----------|
| `Hello-SimpleAI/chatgpt-detector-roberta-chinese` | HuggingFace | 中文 |
| `roberta-base-openai-detector` | OpenAI | 英文 |
| `Hello-SimpleAI/chatgpt-detector-roberta` | HuggingFace | 通用 |

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

### 如何贡献

1. Fork 本项目
2. 创建你的功能分支：`git checkout -b feature/你的功能名`
3. 提交修改：`git commit -m 'feat: 添加某功能'`
4. 推送分支：`git push origin feature/你的功能名`
5. 提交 Pull Request

有任何想法或问题，欢迎直接开 [Issue](https://github.com/lvcheer/AIDetect/issues) 讨论！

---

## License

MIT License
