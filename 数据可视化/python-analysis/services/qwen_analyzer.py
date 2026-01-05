#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen3-Coder数据分析器
集成Qwen3-Coder模型进行智能数据分析和趋势预测
"""

import json
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from utils.config import Config

# 导入机器学习库
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 导入OpenAI客户端
from openai import OpenAI

logger = logging.getLogger(__name__)


class QwenAnalyzer:
    """Qwen3-Coder数据分析器"""
    
    def __init__(self):
        self.config = Config()
        self.client = None
        self.scaler = StandardScaler()
        self._init_client()
    
    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            logger.info("正在初始化Qwen3-Coder API客户端...")
            self.client = OpenAI(
                api_key=self.config.get("model.api_key"),
                base_url=self.config.get("model.base_url"),
                timeout=self.config.get("model.timeout")
            )
            logger.info("Qwen3-Coder API客户端初始化完成")
        except Exception as e:
            logger.warning(f"Qwen3-Coder API客户端初始化失败: {e}")
            self.client = None
    
    def analyze(self, data_id: str, analysis_type: str, data: List[Dict], 
                headers: List[str], model_type: str = "qwen3-coder",
                enable_prediction: bool = False, prediction_steps: int = 10,
                parameters: Dict = None) -> Dict[str, Any]:
        """
        执行数据分析
        
        Args:
            data_id: 数据ID
            analysis_type: 分析类型
            data: 数据列表
            headers: 列名列表
            model_type: 模型类型
            enable_prediction: 是否启用预测
            prediction_steps: 预测步数
            parameters: 分析参数
            
        Returns:
            分析结果
        """
        try:
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 根据分析类型执行不同的分析
            if analysis_type == "trend_analysis":
                result = self._trend_analysis(df, headers, parameters)
            elif analysis_type == "pattern_recognition":
                result = self._pattern_recognition(df, headers, parameters)
            elif analysis_type == "correlation_analysis":
                result = self._correlation_analysis(df, headers, parameters)
            elif analysis_type == "clustering":
                result = self._clustering_analysis(df, headers, parameters)
            elif analysis_type == "anomaly_detection":
                result = self._anomaly_detection(df, headers, parameters)
            else:
                result = self._comprehensive_analysis(df, headers, parameters)
            
            # 如果启用预测，添加预测结果
            if enable_prediction:
                predictions = self._generate_predictions(df, headers, prediction_steps, parameters)
                result["predictions"] = predictions
            
            # 使用Qwen3-Coder生成智能洞察
            try:
                if self.client:
                    insights = self._generate_insights(df, analysis_type, result, parameters)
                    result["insights"] = insights
                else:
                    result["insights"] = ["数据分析完成，建议进一步探索数据特征"]
            except Exception as e:
                logger.warning(f"生成洞察失败: {e}")
                result["insights"] = ["数据分析完成，建议进一步探索数据特征"]
            
            result["analysis_id"] = data_id
            result["analysis_type"] = analysis_type
            result["timestamp"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"数据分析失败: {e}")
            raise
    
    def _trend_analysis(self, df: pd.DataFrame, headers: List[str], 
                       parameters: Dict) -> Dict[str, Any]:
        """趋势分析"""
        result = {
            "trends": [],
            "statistics": {},
            "recommendations": []
        }
        
        # 识别数值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_columns:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(series) > 1:
                    # 计算趋势
                    x = np.arange(len(series)).reshape(-1, 1)
                    y = series.values
                    
                    # 线性回归
                    reg = LinearRegression().fit(x, y)
                    slope = reg.coef_[0]
                    r2 = reg.score(x, y)
                    
                    trend_direction = "上升" if slope > 0 else "下降" if slope < 0 else "平稳"
                    
                    result["trends"].append({
                        "column": col,
                        "direction": trend_direction,
                        "slope": float(slope),
                        "r_squared": float(r2),
                        "strength": "强" if abs(r2) > 0.7 else "中等" if abs(r2) > 0.3 else "弱"
                    })
                    
                    # 统计信息
                    result["statistics"][col] = {
                        "mean": float(series.mean()),
                        "std": float(series.std()),
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "trend": trend_direction
                    }
        
        return result
    
    def _pattern_recognition(self, df: pd.DataFrame, headers: List[str], 
                           parameters: Dict) -> Dict[str, Any]:
        """模式识别"""
        result = {
            "patterns": [],
            "seasonality": {},
            "cyclical_patterns": []
        }
        
        # 识别数值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_columns:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(series) > 10:
                    # 检测周期性模式
                    autocorr = series.autocorr(lag=1)
                    if abs(autocorr) > 0.3:
                        result["cyclical_patterns"].append({
                            "column": col,
                            "autocorrelation": float(autocorr),
                            "pattern_type": "周期性" if autocorr > 0 else "反周期性"
                        })
                    
                    # 检测异常值
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = series[(series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)]
                    
                    if len(outliers) > 0:
                        result["patterns"].append({
                            "column": col,
                            "pattern_type": "异常值",
                            "count": len(outliers),
                            "outliers": outliers.tolist()[:10]  # 限制输出数量
                        })
        
        return result
    
    def _correlation_analysis(self, df: pd.DataFrame, headers: List[str], 
                            parameters: Dict) -> Dict[str, Any]:
        """相关性分析"""
        result = {
            "correlations": [],
            "strong_correlations": [],
            "correlation_matrix": {}
        }
        
        # 计算数值列之间的相关性
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            result["correlation_matrix"] = corr_matrix.to_dict()
            
            # 找出强相关性
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    result["correlations"].append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": float(corr_value)
                    })
                    
                    if abs(corr_value) > 0.7:
                        result["strong_correlations"].append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": float(corr_value),
                            "strength": "强正相关" if corr_value > 0.7 else "强负相关"
                        })
        
        return result
    
    def _clustering_analysis(self, df: pd.DataFrame, headers: List[str], 
                           parameters: Dict) -> Dict[str, Any]:
        """聚类分析"""
        result = {
            "clusters": [],
            "cluster_centers": [],
            "silhouette_score": 0
        }
        
        # 选择数值列进行聚类
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_df) > 10 and len(numeric_df.columns) > 1:
            # 标准化数据
            scaled_data = self.scaler.fit_transform(numeric_df)
            
            # K-means聚类
            n_clusters = min(5, len(numeric_df) // 3)  # 动态确定聚类数
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # 添加聚类标签到原始数据
            df_with_clusters = numeric_df.copy()
            df_with_clusters['cluster'] = cluster_labels
            
            # 分析每个聚类
            for i in range(n_clusters):
                cluster_data = df_with_clusters[df_with_clusters['cluster'] == i]
                result["clusters"].append({
                    "cluster_id": i,
                    "size": len(cluster_data),
                    "percentage": len(cluster_data) / len(numeric_df) * 100,
                    "characteristics": cluster_data.describe().to_dict()
                })
            
            # 聚类中心
            result["cluster_centers"] = kmeans.cluster_centers_.tolist()
        
        return result
    
    def _anomaly_detection(self, df: pd.DataFrame, headers: List[str], 
                         parameters: Dict) -> Dict[str, Any]:
        """异常检测"""
        result = {
            "anomalies": [],
            "anomaly_summary": {}
        }
        
        # 识别数值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_columns:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(series) > 5:
                    # 使用IQR方法检测异常值
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    anomalies = series[(series < lower_bound) | (series > upper_bound)]
                    
                    if len(anomalies) > 0:
                        result["anomalies"].append({
                            "column": col,
                            "count": len(anomalies),
                            "percentage": len(anomalies) / len(series) * 100,
                            "values": anomalies.tolist()[:20],  # 限制输出数量
                            "bounds": {
                                "lower": float(lower_bound),
                                "upper": float(upper_bound)
                            }
                        })
                        
                        result["anomaly_summary"][col] = {
                            "count": len(anomalies),
                            "percentage": len(anomalies) / len(series) * 100
                        }
        
        return result
    
    def _comprehensive_analysis(self, df: pd.DataFrame, headers: List[str], 
                              parameters: Dict) -> Dict[str, Any]:
        """综合分析"""
        result = {
            "data_overview": {},
            "quality_assessment": {},
            "insights": []
        }
        
        # 数据概览
        result["data_overview"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "missing_values": df.isnull().sum().to_dict()
        }
        
        # 数据质量评估
        missing_percentage = (df.isnull().sum() / len(df) * 100).to_dict()
        result["quality_assessment"] = {
            "missing_data_percentage": missing_percentage,
            "duplicate_rows": df.duplicated().sum(),
            "data_types": df.dtypes.to_dict()
        }
        
        # 生成洞察
        if len(df.select_dtypes(include=[np.number]).columns) > 0:
            result["insights"].append("数据包含数值列，可以进行统计分析")
        
        if df.isnull().sum().sum() > 0:
            result["insights"].append("数据存在缺失值，建议进行数据清洗")
        
        if df.duplicated().sum() > 0:
            result["insights"].append("数据存在重复行，建议去重处理")
        
        return result
    
    def _generate_predictions(self, df: pd.DataFrame, headers: List[str], 
                            prediction_steps: int, parameters: Dict) -> Dict[str, Any]:
        """生成预测"""
        predictions = {
            "forecasts": [],
            "confidence_intervals": [],
            "model_performance": {}
        }
        
        # 选择数值列进行预测
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_columns:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(series) > 10:
                    # 准备数据
                    X = np.arange(len(series)).reshape(-1, 1)
                    y = series.values
                    
                    # 训练模型
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    
                    # 生成预测
                    future_X = np.arange(len(series), len(series) + prediction_steps).reshape(-1, 1)
                    future_predictions = model.predict(future_X)
                    
                    # 计算置信区间（简化版本）
                    residuals = y - model.predict(X)
                    std_error = np.std(residuals)
                    confidence_interval = 1.96 * std_error
                    
                    predictions["forecasts"].append({
                        "column": col,
                        "predictions": future_predictions.tolist(),
                        "steps": prediction_steps
                    })
                    
                    predictions["confidence_intervals"].append({
                        "column": col,
                        "lower_bound": (future_predictions - confidence_interval).tolist(),
                        "upper_bound": (future_predictions + confidence_interval).tolist()
                    })
                    
                    # 模型性能
                    y_pred = model.predict(X)
                    mse = mean_squared_error(y, y_pred)
                    r2 = r2_score(y, y_pred)
                    
                    predictions["model_performance"][col] = {
                        "mse": float(mse),
                        "r2_score": float(r2),
                        "rmse": float(np.sqrt(mse))
                    }
        
        return predictions
    
    def _generate_insights(self, df: pd.DataFrame, analysis_type: str, 
                          result: Dict, parameters: Dict) -> List[str]:
        """使用Qwen3-Coder生成智能洞察"""
        insights = []
        
        try:
            # 构建提示词
            prompt = f"""
            基于以下数据分析结果，生成专业的业务洞察和建议：
            
            分析类型: {analysis_type}
            数据形状: {df.shape}
            分析结果: {json.dumps(result, ensure_ascii=False, indent=2)}
            
            请提供：
            1. 关键发现
            2. 业务建议
            3. 潜在风险
            4. 改进方向
            
            请用中文回答，语言要专业且易懂。
            """
            
            # 使用API生成洞察
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.config.get("model.primary_model"),
                    messages=[
                        {"role": "system", "content": "你是一个专业的数据分析师，擅长从数据中提取业务洞察。"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config.get("model.max_length"),
                    temperature=self.config.get("model.temperature"),
                    top_p=self.config.get("model.top_p")
                )
                
                generated_text = response.choices[0].message.content
                insights.append(generated_text)
            else:
                # 备用洞察生成
                insights.append("基于数据分析，建议关注数据质量和趋势变化")
                insights.append("考虑进行更深入的数据挖掘和模式识别")
                
        except Exception as e:
            logger.warning(f"智能洞察生成失败: {e}")
            insights.append("数据分析完成，建议进一步探索数据特征")
        
        return insights
