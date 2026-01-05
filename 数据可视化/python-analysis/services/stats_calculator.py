#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计计算器
计算各种统计指标和描述性统计
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis

logger = logging.getLogger(__name__)


class StatsCalculator:
    """统计计算器"""
    
    def __init__(self):
        pass
    
    def calculate(self, data_id: str, data: List[Dict]) -> Dict[str, Any]:
        """
        计算统计信息
        
        Args:
            data_id: 数据ID
            data: 数据列表
            
        Returns:
            统计结果
        """
        try:
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 计算各种统计信息
            result = {
                "dataId": data_id,
                "overview": self._calculate_overview(df),
                "descriptive_stats": self._calculate_descriptive_stats(df),
                "correlation_analysis": self._calculate_correlation_analysis(df),
                "distribution_analysis": self._calculate_distribution_analysis(df),
                "quality_metrics": self._calculate_quality_metrics(df),
                "summary": self._generate_summary(df)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"统计计算失败: {e}")
            raise
    
    def _calculate_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算数据概览"""
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "datetime_columns": len(df.select_dtypes(include=['datetime64']).columns),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "data_types": df.dtypes.to_dict()
        }
    
    def _calculate_descriptive_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算描述性统计"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return {"message": "没有数值列可供统计"}
        
        stats_dict = {}
        
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) > 0:
                stats_dict[col] = {
                    "count": len(series),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "mode": float(series.mode().iloc[0]) if len(series.mode()) > 0 else None,
                    "std": float(series.std()),
                    "var": float(series.var()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "range": float(series.max() - series.min()),
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                    "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
                    "skewness": float(skew(series)),
                    "kurtosis": float(kurtosis(series)),
                    "coefficient_of_variation": float(series.std() / series.mean()) if series.mean() != 0 else None
                }
        
        return stats_dict
    
    def _calculate_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算相关性分析"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {"message": "需要至少2个数值列进行相关性分析"}
        
        # 计算相关系数矩阵
        corr_matrix = numeric_df.corr()
        
        # 找出强相关性
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": float(corr_value),
                        "strength": "强正相关" if corr_value > 0.7 else "强负相关"
                    })
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_correlations,
            "average_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean())
        }
    
    def _calculate_distribution_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算分布分析"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return {"message": "没有数值列可供分布分析"}
        
        distribution_info = {}
        
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) > 0:
                # 正态性检验
                shapiro_stat, shapiro_p = stats.shapiro(series) if len(series) <= 5000 else (None, None)
                ks_stat, ks_p = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
                
                # 分布形状
                skewness = skew(series)
                kurtosis_val = kurtosis(series)
                
                # 分布类型判断
                if abs(skewness) < 0.5 and abs(kurtosis_val) < 0.5:
                    distribution_type = "近似正态分布"
                elif skewness > 1:
                    distribution_type = "右偏分布"
                elif skewness < -1:
                    distribution_type = "左偏分布"
                elif kurtosis_val > 3:
                    distribution_type = "尖峰分布"
                elif kurtosis_val < -1:
                    distribution_type = "平峰分布"
                else:
                    distribution_type = "其他分布"
                
                distribution_info[col] = {
                    "distribution_type": distribution_type,
                    "skewness": float(skewness),
                    "kurtosis": float(kurtosis_val),
                    "shapiro_test": {
                        "statistic": float(shapiro_stat) if shapiro_stat else None,
                        "p_value": float(shapiro_p) if shapiro_p else None,
                        "is_normal": shapiro_p > 0.05 if shapiro_p else None
                    },
                    "ks_test": {
                        "statistic": float(ks_stat),
                        "p_value": float(ks_p),
                        "is_normal": ks_p > 0.05
                    }
                }
        
        return distribution_info
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算数据质量指标"""
        quality_metrics = {
            "completeness": {},
            "consistency": {},
            "accuracy": {},
            "uniqueness": {}
        }
        
        # 完整性检查
        for col in df.columns:
            total_count = len(df)
            non_null_count = df[col].count()
            null_count = total_count - non_null_count
            completeness_ratio = non_null_count / total_count if total_count > 0 else 0
            
            quality_metrics["completeness"][col] = {
                "total_count": total_count,
                "non_null_count": non_null_count,
                "null_count": null_count,
                "completeness_ratio": float(completeness_ratio),
                "missing_percentage": float((null_count / total_count) * 100) if total_count > 0 else 0
            }
        
        # 一致性检查
        for col in df.columns:
            if df[col].dtype == 'object':
                # 检查字符串格式一致性
                unique_values = df[col].dropna().unique()
                format_consistency = len(unique_values) / len(df[col].dropna()) if len(df[col].dropna()) > 0 else 0
                
                quality_metrics["consistency"][col] = {
                    "unique_values": len(unique_values),
                    "format_consistency": float(format_consistency),
                    "most_common": df[col].value_counts().head(5).to_dict()
                }
        
        # 唯一性检查
        for col in df.columns:
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())
            uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
            
            quality_metrics["uniqueness"][col] = {
                "unique_count": unique_count,
                "total_count": total_count,
                "uniqueness_ratio": float(uniqueness_ratio),
                "is_key": uniqueness_ratio == 1.0
            }
        
        # 重复行检查
        duplicate_rows = df.duplicated().sum()
        quality_metrics["duplicate_analysis"] = {
            "duplicate_rows": int(duplicate_rows),
            "duplicate_percentage": float((duplicate_rows / len(df)) * 100) if len(df) > 0 else 0
        }
        
        return quality_metrics
    
    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """生成数据摘要"""
        numeric_df = df.select_dtypes(include=[np.number])
        categorical_df = df.select_dtypes(include=['object'])
        
        summary = {
            "data_characteristics": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": len(numeric_df.columns),
                "categorical_columns": len(categorical_df.columns)
            },
            "data_quality": {
                "missing_values": df.isnull().sum().sum(),
                "duplicate_rows": df.duplicated().sum(),
                "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
            },
            "insights": []
        }
        
        # 生成洞察
        if len(numeric_df.columns) > 0:
            summary["insights"].append(f"数据包含{len(numeric_df.columns)}个数值列，适合进行统计分析")
        
        if df.isnull().sum().sum() > 0:
            summary["insights"].append("数据存在缺失值，建议进行数据清洗")
        
        if df.duplicated().sum() > 0:
            summary["insights"].append("数据存在重复行，建议去重处理")
        
        if len(categorical_df.columns) > 0:
            summary["insights"].append(f"数据包含{len(categorical_df.columns)}个分类列，可进行类别分析")
        
        # 数据分布洞察
        if len(numeric_df.columns) > 0:
            for col in numeric_df.columns:
                series = numeric_df[col].dropna()
                if len(series) > 0:
                    cv = series.std() / series.mean() if series.mean() != 0 else 0
                    if cv > 1:
                        summary["insights"].append(f"列'{col}'变异系数较大({cv:.2f})，数据离散程度高")
                    elif cv < 0.1:
                        summary["insights"].append(f"列'{col}'变异系数较小({cv:.2f})，数据相对稳定")
        
        return summary
