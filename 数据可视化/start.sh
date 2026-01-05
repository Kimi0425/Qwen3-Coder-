#!/bin/bash

# 数据可视化系统启动脚本

echo "=========================================="
echo "    数据可视化系统启动脚本"
echo "=========================================="

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装，请先安装Docker"
    exit 1
fi

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "错误: Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p uploads
mkdir -p logs
mkdir -p docker/mysql
mkdir -p docker/nginx
mkdir -p docker/prometheus
mkdir -p docker/grafana

# 创建MySQL初始化脚本
cat > docker/mysql/init.sql << 'EOF'
CREATE DATABASE IF NOT EXISTS data_visualization;
USE data_visualization;

CREATE TABLE IF NOT EXISTS data_upload (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    file_name VARCHAR(255),
    file_size BIGINT,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS analysis_result (
    id VARCHAR(36) PRIMARY KEY,
    data_id VARCHAR(36),
    analysis_type VARCHAR(100),
    result JSON,
    processing_time INT,
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (data_id) REFERENCES data_upload(id)
);

CREATE INDEX idx_data_upload_time ON data_upload(upload_time);
CREATE INDEX idx_analysis_data_id ON analysis_result(data_id);
EOF

# 创建Nginx配置
cat > docker/nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream api_gateway {
        server api-gateway:8000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://frontend:80;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        location /api/ {
            proxy_pass http://api_gateway;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
}
EOF

# 创建Prometheus配置
cat > docker/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'api-gateway'
    static_configs:
      - targets: ['api-gateway:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'java-backend'
    static_configs:
      - targets: ['java-backend:8080']
    metrics_path: '/actuator/prometheus'
    scrape_interval: 5s

  - job_name: 'python-analysis'
    static_configs:
      - targets: ['python-analysis:5000']
    metrics_path: '/metrics'
    scrape_interval: 5s
EOF

# 创建Grafana数据源配置
mkdir -p docker/grafana/datasources
cat > docker/grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

# 构建Java后端
echo "构建Java后端..."
cd java-backend
if [ -f "pom.xml" ]; then
    mvn clean package -DskipTests
    if [ $? -ne 0 ]; then
        echo "Java后端构建失败"
        exit 1
    fi
else
    echo "警告: 未找到pom.xml文件，跳过Java后端构建"
fi
cd ..

# 启动服务
echo "启动Docker服务..."
docker-compose up -d

# 等待服务启动
echo "等待服务启动..."
sleep 30

# 检查服务状态
echo "检查服务状态..."
docker-compose ps

# 显示访问信息
echo ""
echo "=========================================="
echo "    服务启动完成！"
echo "=========================================="
echo "前端界面: http://localhost:3000"
echo "API网关: http://localhost:8000"
echo "Java后端: http://localhost:8080"
echo "Python分析: http://localhost:5000"
echo "C优化模块: http://localhost:8081"
echo "Grafana监控: http://localhost:3001 (admin/admin123)"
echo "Prometheus: http://localhost:9090"
echo ""
echo "查看日志: docker-compose logs -f"
echo "停止服务: docker-compose down"
echo "=========================================="
