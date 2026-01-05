# Java后端Dockerfile
FROM openjdk:11-jre-slim

# 设置工作目录
WORKDIR /app

# 复制Maven构建的JAR文件
COPY target/data-visualization-backend-1.0.0.jar app.jar

# 创建上传目录
RUN mkdir -p /app/uploads

# 设置环境变量
ENV JAVA_OPTS="-Xmx2g -Xms1g -Dspring.profiles.active=docker"

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/actuator/health || exit 1

# 启动应用
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
