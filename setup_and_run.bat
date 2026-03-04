@echo off
chcp 65001 >nul
title AI检测工具 - 安装并运行

echo ============================================
echo   AI检测工具 - 首次安装（约需5-10分钟）
echo ============================================
echo.

:: 检查 Python 是否已安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10 或以上版本：
    echo   下载地址：https://www.python.org/downloads/
    echo   安装时请勾选 "Add Python to PATH"
    pause
    exit /b 1
)

echo [1/3] Python 已检测到，安装依赖库...
pip install torch transformers pandas matplotlib --quiet
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络连接后重试。
    pause
    exit /b 1
)

echo [2/3] 依赖安装完成！
echo [3/3] 启动程序...
echo.
echo 注意：首次运行会自动下载AI模型（约500MB），请保持网络连接。
echo.

python MainCode.py
pause
