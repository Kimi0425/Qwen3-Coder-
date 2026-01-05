@echo off
chcp 65001 >nul

echo ==========================================
echo     数据可视化系统启动脚本
echo ==========================================

REM 检查Docker是否安装
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: Docker未安装，请先安装Docker Desktop
    pause
    exit /b 1
)

REM 检查Docker Compose是否安装
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: Docker Compose未安装，请先安装Docker Compose
    pause
    exit /b 1
)

REM 创建必要的目录
echo 创建必要的目录...
if not exist "uploads" mkdir uploads
if not exist "logs" mkdir logs
if not exist "docker\mysql" mkdir docker\mysql
if not exist "docker\nginx" mkdir docker\nginx
if not exist "docker\prometheus" mkdir docker\prometheus
if not exist "docker\grafana" mkdir docker\grafana

REM 创建MySQL初始化脚本
echo 创建MySQL初始化脚本...
(
echo CREATE DATABASE IF NOT EXISTS data_visualization;
echo USE data_visualization;
echo.
echo CREATE TABLE IF NOT EXISTS data_upload ^(
echo     id VARCHAR^(36^) PRIMARY KEY,
echo     name VARCHAR^(255^) NOT NULL,
echo     description TEXT,
echo     category VARCHAR^(100^),
echo     file_name VARCHAR^(255^),
echo     file_size BIGINT,
echo     upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
echo     status VARCHAR^(50^) DEFAULT 'active'
echo ^);
echo.
echo CREATE TABLE IF NOT EXISTS analysis_result ^(
echo     id VARCHAR^(36^) PRIMARY KEY,
echo     data_id VARCHAR^(36^),
echo     analysis_type VARCHAR^(100^),
echo     result JSON,
echo     processing_time INT,
echo     created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
echo     FOREIGN KEY ^(data_id^) REFERENCES data_upload^(id^)
echo ^);
echo.
echo CREATE INDEX idx_data_upload_time ON data_upload^(upload_time^);
echo CREATE INDEX idx_analysis_data_id ON analysis_result^(data_id^);
) > docker\mysql\init.sql

REM 创建Nginx配置
echo 创建Nginx配置...
(
echo events {
echo     worker_connections 1024;
echo }
echo.
echo http {
echo     upstream api_gateway {
echo         server api-gateway:8000;
echo     }
echo.
echo     server {
echo         listen 80;
echo         server_name localhost;
echo.
echo         location / {
echo             proxy_pass http://frontend:80;
echo             proxy_set_header Host $host;
echo             proxy_set_header X-Real-IP $remote_addr;
echo         }
echo.
echo         location /api/ {
echo             proxy_pass http://api_gateway;
echo             proxy_set_header Host $host;
echo             proxy_set_header X-Real-IP $remote_addr;
echo             proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
echo         }
echo     }
echo }
) > docker\nginx\nginx.conf

REM 创建Prometheus配置
echo 创建Prometheus配置...
(
echo global:
echo   scrape_interval: 15s
echo.
echo scrape_configs:
echo   - job_name: 'api-gateway'
echo     static_configs:
echo       - targets: ['api-gateway:8000']
echo     metrics_path: '/metrics'
echo     scrape_interval: 5s
echo.
echo   - job_name: 'java-backend'
echo     static_configs:
echo       - targets: ['java-backend:8080']
echo     metrics_path: '/actuator/prometheus'
echo     scrape_interval: 5s
echo.
echo   - job_name: 'python-analysis'
echo     static_configs:
echo       - targets: ['python-analysis:5000']
echo     metrics_path: '/metrics'
echo     scrape_interval: 5s
) > docker\prometheus\prometheus.yml

REM 创建Grafana数据源配置
if not exist "docker\grafana\datasources" mkdir docker\grafana\datasources
(
echo apiVersion: 1
echo.
echo datasources:
echo   - name: Prometheus
echo     type: prometheus
echo     access: proxy
echo     url: http://prometheus:9090
echo     isDefault: true
) > docker\grafana\datasources\prometheus.yml

REM 构建Java后端
echo 构建Java后端...
cd java-backend
if exist "pom.xml" (
    mvn clean package -DskipTests
    if %errorlevel% neq 0 (
        echo Java后端构建失败
        pause
        exit /b 1
    )
) else (
    echo 警告: 未找到pom.xml文件，跳过Java后端构建
)
cd ..

REM 启动服务
echo 启动Docker服务...
docker-compose up -d

REM 等待服务启动
echo 等待服务启动...
timeout /t 30 /nobreak >nul

REM 检查服务状态
echo 检查服务状态...
docker-compose ps

REM 显示访问信息
echo.
echo ==========================================
echo     服务启动完成！
echo ==========================================
echo 前端界面: http://localhost:3000
echo API网关: http://localhost:8000
echo Java后端: http://localhost:8080
echo Python分析: http://localhost:5000
echo C优化模块: http://localhost:8081
echo Grafana监控: http://localhost:3001 ^(admin/admin123^)
echo Prometheus: http://localhost:9090
echo.
echo 查看日志: docker-compose logs -f
echo 停止服务: docker-compose down
echo ==========================================

pause
