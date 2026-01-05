#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API网关服务
统一管理多语言API接口，提供负载均衡和路由功能
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import redis
from prometheus_client import Counter, Histogram, generate_latest
import time

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="数据可视化系统 API网关",
    description="统一管理Java、Python、C语言服务的API接口",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis连接池
redis_pool = redis.ConnectionPool(host='localhost', port=6379, db=0, decode_responses=True)
redis_client = redis.Redis(connection_pool=redis_pool)

# 服务配置
SERVICES = {
    "java_backend": {
        "url": "http://localhost:8080",
        "health_endpoint": "/actuator/health",
        "weight": 3,
        "timeout": 30
    },
    "python_analysis": {
        "url": "http://localhost:5000",
        "health_endpoint": "/api/health",
        "weight": 2,
        "timeout": 60
    },
    "c_optimization": {
        "url": "http://localhost:8081",
        "health_endpoint": "/health",
        "weight": 1,
        "timeout": 10
    }
}

# 监控指标
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['service', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration', ['service', 'endpoint'])

# 健康检查缓存
health_cache = {}
CACHE_TTL = 30  # 30秒缓存


class APIGateway:
    """API网关核心类"""
    
    def __init__(self):
        self.session = None
        self.service_weights = {}
        self.current_weights = {}
        self._init_weights()
    
    def _init_weights(self):
        """初始化服务权重"""
        for service_name, config in SERVICES.items():
            self.service_weights[service_name] = config["weight"]
            self.current_weights[service_name] = config["weight"]
    
    async def get_session(self):
        """获取HTTP会话"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def health_check(self, service_name: str) -> bool:
        """健康检查"""
        try:
            config = SERVICES.get(service_name)
            if not config:
                return False
            
            # 检查缓存
            cache_key = f"health_{service_name}"
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            session = await self.get_session()
            url = f"{config['url']}{config['health_endpoint']}"
            
            async with session.get(url) as response:
                is_healthy = response.status == 200
                
                # 缓存结果
                redis_client.setex(cache_key, CACHE_TTL, json.dumps(is_healthy))
                
                return is_healthy
                
        except Exception as e:
            logger.error(f"健康检查失败 {service_name}: {e}")
            return False
    
    async def get_healthy_services(self) -> List[str]:
        """获取健康服务列表"""
        healthy_services = []
        
        for service_name in SERVICES.keys():
            if await self.health_check(service_name):
                healthy_services.append(service_name)
        
        return healthy_services
    
    async def route_request(self, service_name: str, endpoint: str, method: str = "GET", 
                          data: Dict = None, params: Dict = None) -> Dict:
        """路由请求到指定服务"""
        try:
            config = SERVICES.get(service_name)
            if not config:
                raise HTTPException(status_code=404, detail=f"服务不存在: {service_name}")
            
            # 健康检查
            if not await self.health_check(service_name):
                raise HTTPException(status_code=503, detail=f"服务不可用: {service_name}")
            
            session = await self.get_session()
            url = f"{config['url']}{endpoint}"
            
            start_time = time.time()
            
            # 发送请求
            if method.upper() == "GET":
                async with session.get(url, params=params) as response:
                    result = await response.json()
            elif method.upper() == "POST":
                async with session.post(url, json=data, params=params) as response:
                    result = await response.json()
            elif method.upper() == "PUT":
                async with session.put(url, json=data, params=params) as response:
                    result = await response.json()
            elif method.upper() == "DELETE":
                async with session.delete(url, params=params) as response:
                    result = await response.json()
            else:
                raise HTTPException(status_code=405, detail=f"不支持的HTTP方法: {method}")
            
            # 记录指标
            duration = time.time() - start_time
            REQUEST_DURATION.labels(service=service_name, endpoint=endpoint).observe(duration)
            REQUEST_COUNT.labels(service=service_name, endpoint=endpoint, status=response.status).inc()
            
            if response.status >= 400:
                raise HTTPException(status_code=response.status, detail=result)
            
            return result
            
        except aiohttp.ClientError as e:
            logger.error(f"请求失败 {service_name}{endpoint}: {e}")
            raise HTTPException(status_code=503, detail=f"服务请求失败: {e}")
        except Exception as e:
            logger.error(f"路由请求异常 {service_name}{endpoint}: {e}")
            raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")
    
    async def load_balance_request(self, endpoint: str, method: str = "GET", 
                                 data: Dict = None, params: Dict = None) -> Dict:
        """负载均衡请求"""
        healthy_services = await self.get_healthy_services()
        
        if not healthy_services:
            raise HTTPException(status_code=503, detail="所有服务都不可用")
        
        # 简单的轮询负载均衡
        # 在实际生产环境中，可以使用更复杂的算法
        service_name = healthy_services[0]  # 简化实现
        
        return await self.route_request(service_name, endpoint, method, data, params)
    
    async def aggregate_data_analysis(self, request_data: Dict) -> Dict:
        """聚合数据分析请求"""
        try:
            results = {}
            
            # 1. 数据预处理 (Python服务)
            preprocess_result = await self.route_request(
                "python_analysis", 
                "/api/preprocess", 
                "POST", 
                request_data
            )
            results["preprocessing"] = preprocess_result
            
            # 2. 数据分析 (Python服务)
            analysis_result = await self.route_request(
                "python_analysis", 
                "/api/analyze", 
                "POST", 
                request_data
            )
            results["analysis"] = analysis_result
            
            # 3. 性能优化计算 (C服务)
            if "optimization" in request_data.get("options", {}):
                optimization_result = await self.route_request(
                    "c_optimization", 
                    "/api/optimize", 
                    "POST", 
                    request_data
                )
                results["optimization"] = optimization_result
            
            # 4. 数据存储 (Java服务)
            storage_result = await self.route_request(
                "java_backend", 
                "/api/data/upload", 
                "POST", 
                request_data
            )
            results["storage"] = storage_result
            
            return {
                "success": True,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"聚合数据分析失败: {e}")
            raise HTTPException(status_code=500, detail=f"聚合分析失败: {e}")
    
    async def close(self):
        """关闭连接"""
        if self.session:
            await self.session.close()


# 创建网关实例
gateway = APIGateway()


@app.on_event("startup")
async def startup_event():
    """启动事件"""
    logger.info("API网关启动中...")
    await gateway.get_session()


@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    logger.info("API网关关闭中...")
    await gateway.close()


@app.get("/health")
async def gateway_health():
    """网关健康检查"""
    healthy_services = await gateway.get_healthy_services()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            service: service in healthy_services 
            for service in SERVICES.keys()
        }
    }


@app.get("/services")
async def list_services():
    """列出所有服务"""
    healthy_services = await gateway.get_healthy_services()
    
    return {
        "services": [
            {
                "name": name,
                "url": config["url"],
                "healthy": name in healthy_services,
                "weight": config["weight"]
            }
            for name, config in SERVICES.items()
        ]
    }


@app.post("/api/data/analyze")
async def analyze_data(request: Request, background_tasks: BackgroundTasks):
    """数据分析接口"""
    try:
        request_data = await request.json()
        
        # 异步执行聚合分析
        result = await gateway.aggregate_data_analysis(request_data)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"数据分析接口错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/upload")
async def upload_data(request: Request):
    """数据上传接口"""
    try:
        request_data = await request.json()
        
        # 路由到Java后端
        result = await gateway.route_request(
            "java_backend", 
            "/api/data/upload", 
            "POST", 
            request_data
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"数据上传接口错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/list")
async def list_data():
    """数据列表接口"""
    try:
        result = await gateway.route_request("java_backend", "/api/data/list", "GET")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"数据列表接口错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/visualize/{data_id}")
async def get_visualization(data_id: str, chart_type: str = "line"):
    """可视化数据接口"""
    try:
        params = {"chartType": chart_type}
        
        # 路由到Python分析服务
        result = await gateway.route_request(
            "python_analysis", 
            f"/api/visualize/{data_id}", 
            "GET", 
            params=params
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"可视化接口错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Prometheus指标接口"""
    return generate_latest()


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "数据可视化系统 API网关",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run(
        "gateway:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
