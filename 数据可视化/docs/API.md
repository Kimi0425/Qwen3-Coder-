# 数据可视化系统 API 文档

## 概述

数据可视化系统提供统一的RESTful API接口，支持数据上传、分析、可视化和预测功能。

## 基础信息

- **基础URL**: `http://localhost:8000`
- **API版本**: v1.0.0
- **认证方式**: API Key (可选)
- **数据格式**: JSON

## 通用响应格式

### 成功响应
```json
{
    "success": true,
    "data": {},
    "message": "操作成功",
    "timestamp": "2024-01-01T00:00:00Z"
}
```

### 错误响应
```json
{
    "success": false,
    "error": "错误信息",
    "code": "ERROR_CODE",
    "timestamp": "2024-01-01T00:00:00Z"
}
```

## 接口列表

### 1. 系统健康检查

#### GET /health
检查系统健康状态

**响应示例:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-01T00:00:00Z",
    "services": {
        "java_backend": true,
        "python_analysis": true,
        "c_optimization": true
    }
}
```

### 2. 数据管理

#### POST /api/data/upload
上传数据文件

**请求参数:**
- `file` (multipart/form-data): 数据文件
- `name` (string): 数据名称
- `description` (string): 数据描述
- `category` (string): 数据类别

**响应示例:**
```json
{
    "success": true,
    "data": {
        "dataId": "uuid-string",
        "fileName": "data.csv",
        "rowCount": 1000,
        "columnCount": 10
    },
    "message": "数据上传成功"
}
```

#### GET /api/data/list
获取数据列表

**响应示例:**
```json
{
    "success": true,
    "data": [
        {
            "dataId": "uuid-string",
            "name": "销售数据",
            "description": "2024年销售数据",
            "category": "sales",
            "uploadTime": "2024-01-01T00:00:00Z",
            "rowCount": 1000,
            "columnCount": 10
        }
    ]
}
```

#### GET /api/data/{dataId}
获取数据详情

**路径参数:**
- `dataId` (string): 数据ID

**响应示例:**
```json
{
    "success": true,
    "data": {
        "dataId": "uuid-string",
        "name": "销售数据",
        "description": "2024年销售数据",
        "headers": ["日期", "销售额", "客户数"],
        "rowCount": 1000,
        "columnCount": 3,
        "preview": [
            {"日期": "2024-01-01", "销售额": 10000, "客户数": 50},
            {"日期": "2024-01-02", "销售额": 12000, "客户数": 60}
        ]
    }
}
```

#### DELETE /api/data/{dataId}
删除数据

**路径参数:**
- `dataId` (string): 数据ID

**响应示例:**
```json
{
    "success": true,
    "message": "数据删除成功"
}
```

### 3. 数据分析

#### POST /api/data/analyze
执行数据分析

**请求体:**
```json
{
    "dataId": "uuid-string",
    "analysisType": "trend_analysis",
    "modelType": "qwen3-coder-plus",
    "enablePrediction": true,
    "predictionSteps": 10,
    "parameters": {
        "confidence_level": 0.95,
        "seasonality": true
    }
}
```

**分析类型:**
- `trend_analysis`: 趋势分析
- `correlation_analysis`: 相关性分析
- `pattern_recognition`: 模式识别
- `clustering`: 聚类分析
- `anomaly_detection`: 异常检测
- `comprehensive`: 综合分析

**响应示例:**
```json
{
    "success": true,
    "data": {
        "analysisId": "uuid-string",
        "analysisType": "trend_analysis",
        "status": "success",
        "results": {
            "trends": [
                {
                    "column": "销售额",
                    "direction": "上升",
                    "slope": 0.15,
                    "r_squared": 0.85,
                    "strength": "强"
                }
            ],
            "correlations": [
                {
                    "column1": "销售额",
                    "column2": "客户数",
                    "correlation": 0.78,
                    "strength": "强正相关"
                }
            ],
            "predictions": {
                "forecasts": [20000, 21000, 22000],
                "confidence_intervals": [18000, 19000, 20000],
                "model_performance": {
                    "r2_score": 0.85,
                    "mse": 1000
                }
            },
            "insights": [
                "数据呈现明显的上升趋势",
                "销售额与客户数存在强正相关关系",
                "建议关注季节性波动"
            ]
        },
        "processingTime": 2500,
        "timestamp": "2024-01-01T00:00:00Z"
    }
}
```

### 4. 数据可视化

#### GET /api/visualize/{dataId}
生成可视化图表

**路径参数:**
- `dataId` (string): 数据ID

**查询参数:**
- `chartType` (string): 图表类型
- `xColumn` (string): X轴列名
- `yColumn` (string): Y轴列名
- `colorColumn` (string): 颜色映射列名

**图表类型:**
- `line`: 折线图
- `bar`: 柱状图
- `scatter`: 散点图
- `pie`: 饼图
- `heatmap`: 热力图
- `histogram`: 直方图
- `box`: 箱线图
- `radar`: 雷达图
- `3d`: 3D图表

**响应示例:**
```json
{
    "success": true,
    "data": {
        "chart_type": "line",
        "data": [
            {
                "x": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "y": [10000, 12000, 15000],
                "type": "scatter",
                "mode": "lines+markers",
                "name": "销售额"
            }
        ],
        "layout": {
            "title": "销售趋势图",
            "xaxis": {"title": "日期"},
            "yaxis": {"title": "销售额"}
        }
    }
}
```

### 5. 数据预处理

#### POST /api/data/preprocess
数据预处理

**请求体:**
```json
{
    "dataId": "uuid-string",
    "options": {
        "handle_missing": true,
        "missing_strategy": "fill_mean",
        "remove_duplicates": true,
        "convert_types": true,
        "handle_outliers": true,
        "outlier_method": "iqr",
        "scale_features": true,
        "scaling_method": "standard",
        "encode_features": true,
        "encoding_method": "label"
    }
}
```

**响应示例:**
```json
{
    "success": true,
    "data": {
        "dataId": "uuid-string",
        "processedData": [...],
        "originalInfo": {
            "shape": [1000, 10],
            "missing_values": {"列1": 5, "列2": 0},
            "duplicate_rows": 2
        },
        "processedInfo": {
            "shape": [998, 10],
            "missing_values": {},
            "duplicate_rows": 0
        },
        "preprocessingSteps": [
            {
                "step": "缺失值处理",
                "strategy": "fill_mean",
                "removed_missing": 5
            }
        ]
    }
}
```

### 6. 统计信息

#### GET /api/data/stats/{dataId}
获取数据统计信息

**路径参数:**
- `dataId` (string): 数据ID

**响应示例:**
```json
{
    "success": true,
    "data": {
        "overview": {
            "total_rows": 1000,
            "total_columns": 10,
            "numeric_columns": 8,
            "categorical_columns": 2
        },
        "descriptive_stats": {
            "销售额": {
                "mean": 15000,
                "median": 14500,
                "std": 3000,
                "min": 8000,
                "max": 25000
            }
        },
        "correlation_analysis": {
            "correlation_matrix": {...},
            "strong_correlations": [...]
        },
        "quality_metrics": {
            "completeness": {...},
            "consistency": {...},
            "uniqueness": {...}
        }
    }
}
```

### 7. 系统监控

#### GET /api/system/info
获取系统信息

**响应示例:**
```json
{
    "success": true,
    "data": {
        "version": "1.0.0",
        "uptime": "2 days, 5 hours",
        "memory_usage": 45,
        "cpu_usage": 23,
        "disk_usage": 67,
        "active_connections": 15
    }
}
```

#### GET /metrics
Prometheus监控指标

**响应格式:** Prometheus格式的监控指标

## 错误码

| 错误码 | 描述 | HTTP状态码 |
|--------|------|------------|
| INVALID_REQUEST | 请求参数无效 | 400 |
| DATA_NOT_FOUND | 数据不存在 | 404 |
| ANALYSIS_FAILED | 分析失败 | 500 |
| SERVICE_UNAVAILABLE | 服务不可用 | 503 |
| RATE_LIMIT_EXCEEDED | 请求频率超限 | 429 |

## 使用示例

### Python示例

```python
import requests
import json

# 上传数据
files = {'file': open('data.csv', 'rb')}
data = {
    'name': '销售数据',
    'description': '2024年销售数据',
    'category': 'sales'
}
response = requests.post('http://localhost:8000/api/data/upload', files=files, data=data)
result = response.json()

# 执行分析
analysis_data = {
    'dataId': result['data']['dataId'],
    'analysisType': 'trend_analysis',
    'enablePrediction': True,
    'predictionSteps': 10
}
response = requests.post('http://localhost:8000/api/data/analyze', json=analysis_data)
analysis_result = response.json()

# 生成图表
response = requests.get(f"http://localhost:8000/api/visualize/{result['data']['dataId']}?chartType=line")
chart_data = response.json()
```

### JavaScript示例

```javascript
// 上传数据
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('name', '销售数据');
formData.append('description', '2024年销售数据');

fetch('http://localhost:8000/api/data/upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('上传成功:', data);
    
    // 执行分析
    return fetch('http://localhost:8000/api/data/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            dataId: data.data.dataId,
            analysisType: 'trend_analysis',
            enablePrediction: true
        })
    });
})
.then(response => response.json())
.then(analysis => {
    console.log('分析结果:', analysis);
});
```

## 注意事项

1. **文件大小限制**: 单个文件最大100MB
2. **请求频率限制**: 每分钟最多100次请求
3. **超时设置**: 分析请求超时时间为60秒
4. **数据格式**: 支持CSV、Excel、JSON格式
5. **字符编码**: 建议使用UTF-8编码

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 支持基础数据分析和可视化功能
- 集成Qwen3-Coder模型
- 提供多语言API支持
