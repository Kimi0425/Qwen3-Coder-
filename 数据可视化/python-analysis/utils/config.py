#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理模块
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """配置管理类"""
    
    def __init__(self):
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        return {
            # 应用配置
            "app": {
                "name": os.getenv("APP_NAME", "数据可视化分析系统"),
                "version": os.getenv("APP_VERSION", "1.0.0"),
                "debug": os.getenv("DEBUG", "False").lower() == "true",
                "host": os.getenv("HOST", "0.0.0.0"),
                "port": int(os.getenv("PORT", 5000))
            },
            
            # 模型配置
            "model": {
                "primary_model": "qwen3-coder-plus",
                "secondary_model": "Moonshot-Kimi-K2-Instruct", 
                "api_key": "sk-d1f0eab96c7d456cbe8311c82f10dffa",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "max_length": int(os.getenv("MAX_LENGTH", 2048)),
                "temperature": float(os.getenv("TEMPERATURE", 0.7)),
                "top_p": float(os.getenv("TOP_P", 0.9)),
                "timeout": int(os.getenv("MODEL_TIMEOUT", 30))
            },
            
            # 数据处理配置
            "data": {
                "max_file_size": int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024)),  # 100MB
                "supported_formats": ["csv", "xlsx", "xls", "json"],
                "chunk_size": int(os.getenv("CHUNK_SIZE", 10000)),
                "encoding": os.getenv("ENCODING", "utf-8")
            },
            
            # 可视化配置
            "visualization": {
                "default_chart_type": os.getenv("DEFAULT_CHART_TYPE", "line"),
                "max_data_points": int(os.getenv("MAX_DATA_POINTS", 10000)),
                "chart_colors": [
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
                ],
                "figure_size": (12, 8),
                "dpi": int(os.getenv("DPI", 300))
            },
            
            # 缓存配置
            "cache": {
                "enabled": os.getenv("CACHE_ENABLED", "True").lower() == "true",
                "ttl": int(os.getenv("CACHE_TTL", 3600)),  # 1小时
                "max_size": int(os.getenv("CACHE_MAX_SIZE", 1000))
            },
            
            # 安全配置
            "security": {
                "api_key": os.getenv("API_KEY", "your-secret-api-key"),
                "rate_limit": int(os.getenv("RATE_LIMIT", 100)),  # 每分钟请求数
                "cors_origins": os.getenv("CORS_ORIGINS", "*").split(",")
            },
            
            # 日志配置
            "logging": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "format": os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                "file": os.getenv("LOG_FILE", "logs/app.log"),
                "max_size": int(os.getenv("LOG_MAX_SIZE", 10 * 1024 * 1024)),  # 10MB
                "backup_count": int(os.getenv("LOG_BACKUP_COUNT", 5))
            },
            
            # 数据库配置
            "database": {
                "url": os.getenv("DATABASE_URL", "sqlite:///data_analysis.db"),
                "pool_size": int(os.getenv("DB_POOL_SIZE", 10)),
                "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", 20)),
                "echo": os.getenv("DB_ECHO", "False").lower() == "true"
            },
            
            # 外部服务配置
            "external": {
                "java_backend_url": os.getenv("JAVA_BACKEND_URL", "http://localhost:8080"),
                "c_service_url": os.getenv("C_SERVICE_URL", "http://localhost:8081"),
                "timeout": int(os.getenv("EXTERNAL_TIMEOUT", 30))
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split(".")
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config
    
    def update(self, key: str, value: Any) -> None:
        """更新配置值"""
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def reload(self) -> None:
        """重新加载配置"""
        self._config = self._load_config()
