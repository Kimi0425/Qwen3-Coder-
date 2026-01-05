# 数据探索与可视化项目

## 项目概述

基于Qwen3-Coder模型的数据分析与趋势预测可视化系统，采用Java Web后端 + Python数据分析 + C语言优化的多语言架构。

## 技术架构

- **后端服务**: Java Spring Boot
- **数据分析**: Python (Qwen3-Coder模型集成)
- **性能优化**: C语言模块
- **前端界面**: HTML5 + JavaScript + Chart.js
- **数据库**: MySQL/PostgreSQL
- **API通信**: RESTful API + HTTP接口

## 项目结构

```
数据可视化/
├── java-backend/          # Java Web后端服务
├── python-analysis/       # Python数据分析模块
├── c-optimization/        # C语言性能优化
├── frontend/              # 前端可视化界面
├── api-gateway/           # API网关和接口管理
├── docs/                  # 项目文档
└── docker/                # Docker部署配置
```

## 主要功能

1. **数据加载与预处理**
   - 支持CSV、Excel、JSON等格式
   - 数据清洗和预处理
   - 缺失值处理

2. **智能数据分析**
   - Qwen3-Coder模型集成
   - 趋势识别和预测
   - 模式识别和分类

3. **多维度可视化**
   - 折线图、柱状图、散点图
   - 热力图、雷达图、3D图表
   - 实时数据更新

4. **交互式分析界面**
   - 参数配置和调整
   - 实时分析结果展示
   - 多语言API支持

## 快速开始

### 环境要求
- Java 11+
- Python 3.8+
- Node.js 16+
- MySQL 8.0+
- Docker (可选)

### 安装步骤
1. 克隆项目
2. 配置数据库连接
3. 启动后端服务
4. 运行Python分析模块
5. 启动前端界面

## API文档

详细的API接口文档请参考 `docs/api.md`

## 许可证

MIT License
