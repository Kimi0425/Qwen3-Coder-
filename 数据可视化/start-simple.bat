@echo off
chcp 65001 >nul

echo ==========================================
echo     数据可视化系统 - 简化启动
echo ==========================================

REM 设置Python路径
set PYTHON_PATH=D:\Python311\python.exe
set PIP_PATH=D:\Python311\Scripts\pip.exe

REM 检查Python是否存在
if not exist "%PYTHON_PATH%" (
    echo 错误: 未找到Python，请检查路径: %PYTHON_PATH%
    pause
    exit /b 1
)

echo 使用Python: %PYTHON_PATH%

REM 创建必要目录
if not exist "uploads" mkdir uploads
if not exist "logs" mkdir logs

REM 启动Python分析服务
echo 启动Python分析服务...
cd python-analysis
if exist "requirements.txt" (
    echo 安装依赖...
    "%PIP_PATH%" install -r requirements.txt
    echo 启动服务 (端口5000)...
    start "Python Analysis" cmd /k "%PYTHON_PATH% app.py"
) else (
    echo 启动服务 (端口5000)...
    start "Python Analysis" cmd /k "%PYTHON_PATH% app.py"
)
cd ..

REM 等待服务启动
echo 等待服务启动...
timeout /t 5 /nobreak >nul

REM 启动前端服务
echo 启动前端服务...
cd frontend
if exist "index.html" (
    echo 启动前端 (端口3000)...
    start "Frontend" cmd /k "%PYTHON_PATH% -m http.server 3000"
)
cd ..

REM 等待前端启动
echo 等待前端启动...
timeout /t 3 /nobreak >nul

REM 显示访问信息
echo.
echo ==========================================
echo     服务启动完成！
echo ==========================================
echo 前端界面: http://localhost:3000
echo Python分析: http://localhost:5000
echo.
echo 正在打开浏览器...
start http://localhost:3000
echo ==========================================

pause
