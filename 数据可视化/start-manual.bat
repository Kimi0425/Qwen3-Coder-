@echo off
chcp 65001 >nul

echo ==========================================
echo     数据可视化系统 - 手动启动模式
echo ==========================================

REM 检查Java环境
echo 检查Java环境...
java -version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: Java未安装，请先安装Java 11+
    echo 下载地址: https://adoptium.net/
    pause
    exit /b 1
)

REM 检查Python环境
echo 检查Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 尝试使用D:\Python311路径...
    if exist "D:\Python311\python.exe" (
        echo 找到Python: D:\Python311\python.exe
        set PYTHON_PATH=D:\Python311\python.exe
        set PIP_PATH=D:\Python311\Scripts\pip.exe
    ) else (
        echo 错误: Python未安装，请先安装Python 3.8+
        echo 下载地址: https://www.python.org/downloads/
        pause
        exit /b 1
    )
) else (
    echo Python环境正常
    set PYTHON_PATH=python
    set PIP_PATH=pip
)

REM 检查Node.js环境
echo 检查Node.js环境...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 警告: Node.js未安装，前端服务将使用Python内置服务器
)

REM 创建必要的目录
echo 创建必要的目录...
if not exist "uploads" mkdir uploads
if not exist "logs" mkdir logs
if not exist "java-backend\uploads" mkdir java-backend\uploads
if not exist "python-analysis\logs" mkdir python-analysis\logs

REM 启动Java后端
echo 启动Java后端服务...
cd java-backend
if exist "pom.xml" (
    echo 构建Java后端...
    mvn clean package -DskipTests
    if %errorlevel% neq 0 (
        echo Java后端构建失败
        pause
        exit /b 1
    )
    echo 启动Java后端 (端口8080)...
    start "Java Backend" cmd /k "java -jar target\data-visualization-backend-1.0.0.jar"
) else (
    echo 警告: 未找到pom.xml文件，跳过Java后端
)
cd ..

REM 等待Java后端启动
echo 等待Java后端启动...
timeout /t 10 /nobreak >nul

REM 启动Python分析服务
echo 启动Python分析服务...
cd python-analysis
if exist "requirements.txt" (
    echo 安装Python依赖...
    %PIP_PATH% install -r requirements.txt
    echo 启动Python分析服务 (端口5000)...
    start "Python Analysis" cmd /k "%PYTHON_PATH% app.py"
) else (
    echo 警告: 未找到requirements.txt文件，跳过Python分析服务
)
cd ..

REM 等待Python服务启动
echo 等待Python服务启动...
timeout /t 10 /nobreak >nul

REM 启动API网关
echo 启动API网关...
cd api-gateway
if exist "requirements.txt" (
    echo 安装API网关依赖...
    %PIP_PATH% install -r requirements.txt
    echo 启动API网关 (端口8000)...
    start "API Gateway" cmd /k "%PYTHON_PATH% gateway.py"
) else (
    echo 警告: 未找到requirements.txt文件，跳过API网关
)
cd ..

REM 等待API网关启动
echo 等待API网关启动...
timeout /t 10 /nobreak >nul

REM 启动前端服务
echo 启动前端服务...
cd frontend
if exist "index.html" (
    echo 启动前端服务 (端口3000)...
    start "Frontend" cmd /k "%PYTHON_PATH% -m http.server 3000"
) else (
    echo 警告: 未找到index.html文件，跳过前端服务
)
cd ..

REM 显示访问信息
echo.
echo ==========================================
echo     服务启动完成！
echo ==========================================
echo 前端界面: http://localhost:3000
echo API网关: http://localhost:8000
echo Java后端: http://localhost:8080
echo Python分析: http://localhost:5000
echo.
echo 注意: 请确保所有服务都正常启动
echo 如果某个服务启动失败，请检查对应的命令行窗口
echo.
echo 停止服务: 关闭所有命令行窗口
echo ==========================================

REM 打开浏览器
echo 正在打开浏览器...
start http://localhost:3000

pause
