# 数据可视化系统部署指南

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   前端界面      │    │   API网关       │    │   Java后端      │
│   (Port 3000)   │◄──►│   (Port 8000)   │◄──►│   (Port 8080)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Python分析服务 │    │   C优化模块     │
                       │   (Port 5000)   │    │   (Port 8081)   │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   数据库        │
                       │   (MySQL)       │
                       └─────────────────┘
```

## 环境要求

### 系统要求
- **操作系统**: Linux (Ubuntu 20.04+) / Windows 10+ / macOS 10.15+
- **内存**: 最低 8GB，推荐 16GB+
- **存储**: 最低 50GB 可用空间
- **网络**: 稳定的互联网连接

### 软件依赖
- **Java**: 11+
- **Python**: 3.8+
- **Node.js**: 16+
- **MySQL**: 8.0+
- **Redis**: 6.0+
- **Docker**: 20.0+ (可选)

## 快速部署

### 1. 克隆项目
```bash
git clone <repository-url>
cd 数据可视化
```

### 2. 环境配置
```bash
# 创建环境变量文件
cp .env.example .env

# 编辑环境变量
nano .env
```

### 3. 数据库设置
```bash
# 启动MySQL
sudo systemctl start mysql

# 创建数据库
mysql -u root -p
CREATE DATABASE data_visualization;
CREATE USER 'dv_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON data_visualization.* TO 'dv_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

### 4. 启动Redis
```bash
# 启动Redis
sudo systemctl start redis

# 验证Redis运行状态
redis-cli ping
```

### 5. 启动服务

#### 方式一：手动启动
```bash
# 启动Java后端
cd java-backend
mvn spring-boot:run

# 启动Python分析服务
cd ../python-analysis
pip install -r requirements.txt
python app.py

# 启动API网关
cd ../api-gateway
pip install -r requirements.txt
python gateway.py

# 启动前端
cd ../frontend
# 使用任意HTTP服务器，如Python内置服务器
python -m http.server 3000
```

#### 方式二：使用Docker Compose
```bash
# 构建并启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

## 详细部署步骤

### Java后端部署

#### 1. 环境准备
```bash
# 安装Java 11
sudo apt update
sudo apt install openjdk-11-jdk

# 安装Maven
sudo apt install maven

# 验证安装
java -version
mvn -version
```

#### 2. 配置数据库
```yaml
# java-backend/src/main/resources/application.yml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/data_visualization
    username: dv_user
    password: your_password
    driver-class-name: com.mysql.cj.jdbc.Driver
```

#### 3. 启动服务
```bash
cd java-backend
mvn clean package
java -jar target/data-visualization-backend-1.0.0.jar
```

### Python分析服务部署

#### 1. 环境准备
```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 2. 配置环境变量
```bash
# 设置API Key
export DASHSCOPE_API_KEY="sk-d1f0eab96c7d456cbe8311c82f10dffa"

# 或创建.env文件
echo "DASHSCOPE_API_KEY=sk-d1f0eab96c7d456cbe8311c82f10dffa" > .env
```

#### 3. 启动服务
```bash
cd python-analysis
python app.py
```

### C优化模块部署

#### 1. 编译模块
```bash
cd c-optimization
make clean
make all

# 运行测试
make test

# 安装到系统
sudo make install
```

#### 2. 创建HTTP服务
```bash
# 使用Python创建简单的HTTP服务
cd c-optimization
python -m http.server 8081
```

### API网关部署

#### 1. 安装依赖
```bash
cd api-gateway
pip install -r requirements.txt
```

#### 2. 配置服务
```python
# api-gateway/gateway.py
SERVICES = {
    "java_backend": {
        "url": "http://localhost:8080",
        "health_endpoint": "/actuator/health"
    },
    "python_analysis": {
        "url": "http://localhost:5000",
        "health_endpoint": "/api/health"
    },
    "c_optimization": {
        "url": "http://localhost:8081",
        "health_endpoint": "/health"
    }
}
```

#### 3. 启动网关
```bash
cd api-gateway
python gateway.py
```

### 前端部署

#### 1. 使用静态文件服务器
```bash
cd frontend
python -m http.server 3000
```

#### 2. 使用Nginx (生产环境)
```nginx
# /etc/nginx/sites-available/data-visualization
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        root /path/to/frontend;
        index index.html;
        try_files $uri $uri/ /index.html;
    }
    
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Docker部署

### 1. 创建Dockerfile

#### Java后端Dockerfile
```dockerfile
# java-backend/Dockerfile
FROM openjdk:11-jre-slim

WORKDIR /app
COPY target/data-visualization-backend-1.0.0.jar app.jar

EXPOSE 8080
CMD ["java", "-jar", "app.jar"]
```

#### Python服务Dockerfile
```dockerfile
# python-analysis/Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

### 2. Docker Compose配置
```yaml
# docker-compose.yml
version: '3.8'

services:
  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: data_visualization
      MYSQL_USER: dv_user
      MYSQL_PASSWORD: dv_password
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql

  redis:
    image: redis:6.2-alpine
    ports:
      - "6379:6379"

  java-backend:
    build: ./java-backend
    ports:
      - "8080:8080"
    depends_on:
      - mysql
    environment:
      SPRING_DATASOURCE_URL: jdbc:mysql://mysql:3306/data_visualization
      SPRING_DATASOURCE_USERNAME: dv_user
      SPRING_DATASOURCE_PASSWORD: dv_password

  python-analysis:
    build: ./python-analysis
    ports:
      - "5000:5000"
    environment:
      DASHSCOPE_API_KEY: sk-d1f0eab96c7d456cbe8311c82f10dffa

  api-gateway:
    build: ./api-gateway
    ports:
      - "8000:8000"
    depends_on:
      - java-backend
      - python-analysis

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - api-gateway

volumes:
  mysql_data:
```

### 3. 启动Docker服务
```bash
# 构建并启动
docker-compose up -d

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## 生产环境部署

### 1. 使用Nginx反向代理
```nginx
# /etc/nginx/sites-available/data-visualization
upstream api_gateway {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # 前端静态文件
    location / {
        root /var/www/data-visualization/frontend;
        index index.html;
        try_files $uri $uri/ /index.html;
    }
    
    # API代理
    location /api/ {
        proxy_pass http://api_gateway;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # WebSocket支持
    location /ws/ {
        proxy_pass http://api_gateway;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 2. 使用Systemd管理服务
```ini
# /etc/systemd/system/data-visualization.service
[Unit]
Description=Data Visualization System
After=network.target mysql.service redis.service

[Service]
Type=simple
User=dv_user
WorkingDirectory=/opt/data-visualization
ExecStart=/usr/bin/python3 gateway.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 3. 配置SSL证书
```bash
# 使用Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 监控和日志

### 1. 日志配置
```yaml
# 日志轮转配置
logging:
  level:
    com.datavisualization: INFO
    org.springframework.web: WARN
  file:
    name: /var/log/data-visualization/app.log
    max-size: 100MB
    max-history: 30
```

### 2. 监控指标
```bash
# 访问Prometheus指标
curl http://localhost:8000/metrics

# 使用Grafana可视化
# 导入Prometheus数据源
# 创建监控面板
```

### 3. 健康检查
```bash
# 检查服务健康状态
curl http://localhost:8000/health

# 检查各个服务
curl http://localhost:8080/actuator/health
curl http://localhost:5000/api/health
```

## 性能优化

### 1. 数据库优化
```sql
-- 创建索引
CREATE INDEX idx_data_upload_time ON data_upload(upload_time);
CREATE INDEX idx_analysis_data_id ON analysis_result(data_id);

-- 优化查询
EXPLAIN SELECT * FROM data_upload WHERE upload_time > '2024-01-01';
```

### 2. 缓存配置
```yaml
# Redis缓存配置
spring:
  cache:
    type: redis
    redis:
      time-to-live: 3600000  # 1小时
      cache-null-values: false
```

### 3. 连接池配置
```yaml
# 数据库连接池
spring:
  datasource:
    hikari:
      maximum-pool-size: 20
      minimum-idle: 5
      connection-timeout: 30000
      idle-timeout: 600000
```

## 故障排除

### 常见问题

#### 1. 服务启动失败
```bash
# 检查端口占用
netstat -tlnp | grep :8080

# 检查日志
tail -f /var/log/data-visualization/app.log

# 检查依赖
mvn dependency:tree
pip list
```

#### 2. 数据库连接失败
```bash
# 检查MySQL状态
sudo systemctl status mysql

# 测试连接
mysql -u dv_user -p -h localhost data_visualization

# 检查防火墙
sudo ufw status
```

#### 3. API调用失败
```bash
# 检查服务健康状态
curl -v http://localhost:8000/health

# 检查网络连接
telnet localhost 8000

# 查看API日志
docker-compose logs api-gateway
```

### 性能问题

#### 1. 内存不足
```bash
# 检查内存使用
free -h
top -p $(pgrep java)

# 调整JVM参数
export JAVA_OPTS="-Xmx4g -Xms2g"
```

#### 2. 磁盘空间不足
```bash
# 检查磁盘使用
df -h

# 清理日志文件
find /var/log -name "*.log" -mtime +30 -delete

# 清理Docker镜像
docker system prune -a
```

## 备份和恢复

### 1. 数据库备份
```bash
# 创建备份
mysqldump -u root -p data_visualization > backup_$(date +%Y%m%d).sql

# 恢复备份
mysql -u root -p data_visualization < backup_20240101.sql
```

### 2. 文件备份
```bash
# 备份上传文件
tar -czf uploads_backup_$(date +%Y%m%d).tar.gz uploads/

# 恢复文件
tar -xzf uploads_backup_20240101.tar.gz
```

### 3. 配置备份
```bash
# 备份配置文件
cp -r /etc/data-visualization/ backup/config/
cp docker-compose.yml backup/
```

## 安全配置

### 1. 防火墙设置
```bash
# 配置UFW
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### 2. 用户权限
```bash
# 创建专用用户
sudo useradd -r -s /bin/false dv_user
sudo chown -R dv_user:dv_user /opt/data-visualization
```

### 3. API安全
```yaml
# 启用API Key认证
security:
  api:
    key: "your-secret-api-key"
    rate-limit: 100  # 每分钟请求数
```

## 更新和维护

### 1. 系统更新
```bash
# 更新代码
git pull origin main

# 重新构建
docker-compose build
docker-compose up -d

# 数据库迁移
mvn flyway:migrate
```

### 2. 定期维护
```bash
# 清理日志
find /var/log -name "*.log" -mtime +30 -delete

# 优化数据库
mysql -u root -p -e "OPTIMIZE TABLE data_upload, analysis_result;"

# 更新依赖
mvn versions:use-latest-versions
pip list --outdated
```

## 联系支持

如遇到部署问题，请：

1. 查看日志文件
2. 检查系统资源使用情况
3. 验证网络连接
4. 联系技术支持团队

**技术支持邮箱**: support@datavisualization.com  
**文档更新**: 2024-01-01
