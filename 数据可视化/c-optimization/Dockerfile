# C优化模块Dockerfile
FROM ubuntu:20.04

# 设置工作目录
WORKDIR /app

# 安装编译工具和依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    libopenmpi-dev \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制源代码
COPY . .

# 编译C模块
RUN make clean && make all

# 安装Python依赖
RUN pip3 install flask flask-cors requests

# 创建简单的HTTP服务
RUN echo '#!/usr/bin/env python3\n\
from flask import Flask, jsonify\n\
from flask_cors import CORS\n\
import subprocess\n\
import os\n\
\n\
app = Flask(__name__)\n\
CORS(app)\n\
\n\
@app.route("/health")\n\
def health():\n\
    return jsonify({"status": "healthy", "service": "c-optimization"})\n\
\n\
@app.route("/api/optimize", methods=["POST"])\n\
def optimize():\n\
    try:\n\
        # 调用C优化模块\n\
        result = subprocess.run(["./bin/benchmark"], capture_output=True, text=True)\n\
        return jsonify({"success": True, "result": result.stdout})\n\
    except Exception as e:\n\
        return jsonify({"success": False, "error": str(e)})\n\
\n\
if __name__ == "__main__":\n\
    app.run(host="0.0.0.0", port=8081)\n\
' > server.py

# 创建日志目录
RUN mkdir -p /app/logs

# 暴露端口
EXPOSE 8081

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8081/health || exit 1

# 启动服务
CMD ["python3", "server.py"]
