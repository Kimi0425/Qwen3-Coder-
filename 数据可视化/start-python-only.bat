@echo off
chcp 65001 >nul

echo ==========================================
echo     启动Python分析服务
echo ==========================================

REM 设置Python路径
set PYTHON_PATH=D:\Python311\python.exe

REM 检查Python是否存在
if not exist "%PYTHON_PATH%" (
    echo 错误: 未找到Python，请检查路径: %PYTHON_PATH%
    pause
    exit /b 1
)

echo 使用Python: %PYTHON_PATH%

REM 进入Python分析目录
cd python-analysis

REM 启动Python分析服务
echo 启动Python分析服务 (端口5000)...
echo 按Ctrl+C停止服务
echo.

"%PYTHON_PATH%" app.py

pause
