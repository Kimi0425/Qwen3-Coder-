#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据可视化系统 - Python数据分析模块
集成Qwen3-Coder模型进行数据分析和趋势预测
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource
import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import lagrange, interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from services.data_processor import DataProcessor
from services.qwen_analyzer import QwenAnalyzer
from services.visualization_generator import VisualizationGenerator
from services.stats_calculator import StatsCalculator
from utils.logger import setup_logger
from utils.config import Config

# 设置日志
logger = setup_logger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)
api = Api(app)

# 全局配置
config = Config()

# 初始化服务
data_processor = DataProcessor()
qwen_analyzer = QwenAnalyzer()
visualization_generator = VisualizationGenerator()
stats_calculator = StatsCalculator()


class HealthCheck(Resource):
    """健康检查接口"""
    
    def get(self):
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "services": {
                "data_processor": "active",
                "qwen_analyzer": "active",
                "visualization_generator": "active",
                "stats_calculator": "active"
            }
        }


class AnalyzeData(Resource):
    """数据分析接口"""
    
    def post(self):
        try:
            data = request.get_json()
            logger.info(f"收到数据分析请求: {data.get('dataId')}")
            
            # 验证请求数据
            required_fields = ['dataId', 'analysisType']
            for field in required_fields:
                if field not in data:
                    return {"error": f"缺少必需字段: {field}"}, 400
            
            # 根据dataId读取数据
            upload_dir = os.path.join(os.getcwd(), 'uploads')
            info_file = os.path.join(upload_dir, f"{data['dataId']}_info.json")
            
            if not os.path.exists(info_file):
                return {"error": "数据不存在"}, 404
            
            with open(info_file, 'r', encoding='utf-8') as f:
                data_info = json.load(f)
            
            # 获取数据
            data_content = data_info.get('data', [])
            if not data_content:
                return {"error": "数据为空"}, 400
            
            # 转换为DataFrame
            df = pd.DataFrame(data_content)
            
            # 根据分析类型执行不同的分析
            analysis_type = data['analysisType']
            result = {
                "data_overview": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
                    "categorical_columns": len(df.select_dtypes(include=['object']).columns),
                    "missing_values": df.isnull().sum().to_dict()
                },
                "quality_assessment": {
                    "missing_data_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
                    "duplicate_rows": int(df.duplicated().sum()),
                    "data_types": {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
                },
                "analysis_results": {},
                "insights": [],
                "analysis_id": data['dataId'],
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat()
            }
            
            # 根据分析类型生成特定结果
            if analysis_type == "descriptive":
                result["analysis_results"] = self._descriptive_analysis(df)
            elif analysis_type == "trend_analysis":
                result["analysis_results"] = self._trend_analysis(df)
            elif analysis_type == "correlation_analysis":
                result["analysis_results"] = self._correlation_analysis(df)
            elif analysis_type == "clustering":
                result["analysis_results"] = self._clustering_analysis(df)
            else:
                result["analysis_results"] = self._descriptive_analysis(df)
            
            # 使用Qwen3-Coder生成智能洞察
            try:
                qwen_analyzer = QwenAnalyzer()
                ai_insights = qwen_analyzer.analyze(df.to_dict('records'), analysis_type)
                result["insights"] = ai_insights.get("insights", [])
            except Exception as e:
                logger.warning(f"Qwen分析失败，使用基础分析: {str(e)}")
                result["insights"] = self._generate_insights(df, analysis_type, result["analysis_results"])
            
            logger.info(f"数据分析完成: {data['dataId']}")
            return result, 200
            
        except Exception as e:
            logger.error(f"数据分析失败: {str(e)}")
            return {"error": f"数据分析失败: {str(e)}"}, 500
    
    def _descriptive_analysis(self, df):
        """描述性分析"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        result = {
            "statistical_summary": {},
            "distribution_analysis": {},
            "data_characteristics": {}
        }
        
        if len(numeric_cols) > 0:
            result["statistical_summary"] = {
                "mean": {str(k): float(v) for k, v in df[numeric_cols].mean().to_dict().items()},
                "std": {str(k): float(v) for k, v in df[numeric_cols].std().to_dict().items()},
                "min": {str(k): float(v) for k, v in df[numeric_cols].min().to_dict().items()},
                "max": {str(k): float(v) for k, v in df[numeric_cols].max().to_dict().items()},
                "median": {str(k): float(v) for k, v in df[numeric_cols].median().to_dict().items()}
            }
            
            # 分布分析
            result["distribution_analysis"] = {
                "skewness": {str(k): float(v) for k, v in df[numeric_cols].skew().to_dict().items()},
                "kurtosis": {str(k): float(v) for k, v in df[numeric_cols].kurtosis().to_dict().items()}
            }
        
        return result
    
    def _trend_analysis(self, df):
        """趋势分析"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        result = {
            "trends": [],
            "seasonality": {},
            "growth_rates": {}
        }
        
        # 找到时间列
        time_col = None
        for col in df.columns:
            if 'day' in col.lower() or 'time' in col.lower() or 'date' in col.lower():
                time_col = col
                break
        
        if time_col and len(numeric_cols) > 0:
            for col in numeric_cols[:10]:  # 分析前10个数值列
                if col != time_col:
                    series = df[col].dropna()
                    if len(series) > 1:
                        # 计算趋势
                        x = np.arange(len(series))
                        slope, intercept = np.polyfit(x, series, 1)
                        trend_direction = "上升" if slope > 0 else "下降" if slope < 0 else "平稳"
                        
                        result["trends"].append({
                            "column": col,
                            "direction": trend_direction,
                            "slope": float(slope),
                            "strength": "强" if abs(slope) > series.std() else "弱"
                        })
        
        return result
    
    def _correlation_analysis(self, df):
        """相关性分析"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        result = {
            "correlations": [],
            "strong_correlations": [],
            "correlation_matrix": {}
        }
        
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            result["correlation_matrix"] = corr_matrix.to_dict()
            
            # 找出强相关性
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    corr_value = corr_matrix.loc[col1, col2]
                    
                    if not np.isnan(corr_value):
                        result["correlations"].append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": float(corr_value),
                            "strength": "强正相关" if corr_value > 0.7 else "强负相关" if corr_value < -0.7 else "弱相关"
                        })
                        
                        if abs(corr_value) > 0.7:
                            result["strong_correlations"].append({
                                "column1": col1,
                                "column2": col2,
                                "correlation": float(corr_value)
                            })
        
        return result
    
    def _clustering_analysis(self, df):
        """聚类分析"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        result = {
            "clusters": [],
            "cluster_centers": {},
            "silhouette_score": 0
        }
        
        if len(numeric_cols) >= 2:
            try:
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                
                # 标准化数据
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[numeric_cols].dropna())
                
                # K-means聚类
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(scaled_data)
                
                result["clusters"] = clusters.tolist()
                result["cluster_centers"] = kmeans.cluster_centers_.tolist()
                
            except ImportError:
                result["error"] = "需要安装scikit-learn进行聚类分析"
        
        return result
    
    def _generate_insights(self, df, analysis_type, analysis_results):
        """生成基础洞察（Qwen失败时的备用方案）"""
        insights = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 基础洞察
        if len(numeric_cols) > 0:
            insights.append(f"数据包含{len(numeric_cols)}个数值列，适合进行统计分析")
        
        if df.isnull().sum().sum() > 0:
            insights.append(f"数据存在缺失值，总计{df.isnull().sum().sum()}个，建议进行数据清洗")
        
        # 根据分析类型生成特定洞察
        if analysis_type == "trend" and "trends" in analysis_results:
            trends = analysis_results["trends"]
            if trends:
                rising_trends = [t for t in trends if t["direction"] == "上升"]
                if rising_trends:
                    insights.append(f"发现{len(rising_trends)}个变量呈现上升趋势")
        
        elif analysis_type == "correlation" and "strong_correlations" in analysis_results:
            strong_corr = analysis_results["strong_correlations"]
            if strong_corr:
                insights.append(f"发现{len(strong_corr)}对变量存在强相关性")
        
        elif analysis_type == "clustering" and "clusters" in analysis_results:
            if "clusters" in analysis_results and analysis_results["clusters"]:
                unique_clusters = len(set(analysis_results["clusters"]))
                insights.append(f"数据可分为{unique_clusters}个不同的群组")
        
        if not insights:
            insights.append("数据质量良好，可以进行进一步分析")
        
        return insights


class DataPreprocessor:
    """数据预处理类 - 异常值检测、缺失值处理、插值等"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_outliers_iqr(self, df, columns=None):
        """使用IQR方法检测异常值"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outliers_info = {}
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_indices': outliers.index.tolist()
            }
        
        return outliers_info
    
    def detect_outliers_isolation_forest(self, df, columns=None, contamination=0.1):
        """使用Isolation Forest检测异常值"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        # 准备数据
        data = df[columns].dropna()
        if len(data) == 0:
            return {}
        
        # 标准化数据
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # 使用Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(data_scaled)
        
        # 获取异常值索引
        outlier_indices = data.index[outlier_labels == -1].tolist()
        
        return {
            'method': 'Isolation Forest',
            'contamination': contamination,
            'outlier_count': len(outlier_indices),
            'outlier_percentage': len(outlier_indices) / len(data) * 100,
            'outlier_indices': outlier_indices
        }
    
    def handle_missing_values(self, df, method='mean'):
        """处理缺失值"""
        df_processed = df.copy()
        missing_info = {}
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = {
                    'count': missing_count,
                    'percentage': missing_count / len(df) * 100
                }
                
                if method == 'mean' and df[col].dtype in ['int64', 'float64']:
                    df_processed[col].fillna(df[col].mean(), inplace=True)
                elif method == 'median' and df[col].dtype in ['int64', 'float64']:
                    df_processed[col].fillna(df[col].median(), inplace=True)
                elif method == 'mode':
                    df_processed[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
                elif method == 'forward':
                    df_processed[col].fillna(method='ffill', inplace=True)
                elif method == 'backward':
                    df_processed[col].fillna(method='bfill', inplace=True)
                elif method == 'drop':
                    df_processed = df_processed.dropna(subset=[col])
        
        return df_processed, missing_info
    
    def interpolate_lagrange(self, df, column, x_values=None):
        """使用拉格朗日插值法"""
        if x_values is None:
            x_values = df.index
        
        # 获取非空数据点
        valid_data = df[column].dropna()
        if len(valid_data) < 2:
            return df[column]
        
        # 创建拉格朗日插值函数
        try:
            lagrange_func = lagrange(valid_data.index, valid_data.values)
            interpolated_values = lagrange_func(x_values)
            return pd.Series(interpolated_values, index=x_values)
        except:
            # 如果拉格朗日插值失败，使用线性插值
            return df[column].interpolate(method='linear')
    
    def interpolate_kriging(self, df, column):
        """使用克里金插值法（简化版本）"""
        # 这里使用scipy的插值方法作为简化版本
        # 真正的克里金插值需要更复杂的实现
        return df[column].interpolate(method='cubic')
    
    def preprocess_data(self, df, options=None):
        """综合数据预处理"""
        if options is None:
            options = {
                'outlier_detection': 'iqr',
                'outlier_handling': 'keep',
                'missing_value_method': 'mean',
                'interpolation_method': 'linear'
            }
        
        # 确保所有必需的选项都存在
        default_options = {
            'outlier_detection': 'iqr',
            'outlier_handling': 'keep',
            'missing_value_method': 'mean',
            'interpolation_method': 'linear'
        }
        
        for key, default_value in default_options.items():
            if key not in options:
                options[key] = default_value
        
        df_processed = df.copy()
        preprocessing_report = {
            'original_shape': df.shape,
            'outliers_detected': {},
            'missing_values_handled': {},
            'final_shape': None
        }
        
        # 1. 异常值检测
        if options['outlier_detection'] == 'iqr':
            outliers_info = self.detect_outliers_iqr(df_processed)
        elif options['outlier_detection'] == 'isolation_forest':
            outliers_info = self.detect_outliers_isolation_forest(df_processed)
        else:
            outliers_info = {}
        
        preprocessing_report['outliers_detected'] = outliers_info
        
        # 2. 异常值处理
        if options['outlier_handling'] == 'remove':
            for col, info in outliers_info.items():
                if isinstance(info, dict) and 'outlier_indices' in info:
                    df_processed = df_processed.drop(info['outlier_indices'])
        
        # 3. 缺失值处理
        df_processed, missing_info = self.handle_missing_values(
            df_processed, 
            method=options['missing_value_method']
        )
        preprocessing_report['missing_values_handled'] = missing_info
        
        # 4. 插值处理
        if options['interpolation_method'] == 'lagrange':
            for col in df_processed.select_dtypes(include=[np.number]).columns:
                if df_processed[col].isnull().any():
                    df_processed[col] = self.interpolate_lagrange(df_processed, col)
        elif options['interpolation_method'] == 'kriging':
            for col in df_processed.select_dtypes(include=[np.number]).columns:
                if df_processed[col].isnull().any():
                    df_processed[col] = self.interpolate_kriging(df_processed, col)
        
        preprocessing_report['final_shape'] = df_processed.shape
        
        return df_processed, preprocessing_report


class AIModelPredictor:
    """AI模型预测类 - 支持多种模型（VGG、CNN、ANN、1D、2D等）"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def suggest_model_type(self, df):
        """根据数据特征建议模型类型"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        suggestions = []
        
        # 基于数据维度
        if len(numeric_cols) >= 10:
            suggestions.append("CNN (适合高维特征)")
            suggestions.append("ANN (多层感知机)")
        
        if len(numeric_cols) >= 5:
            suggestions.append("1D CNN (时间序列)")
            suggestions.append("LSTM (循环神经网络)")
        
        # 基于数据量
        if len(df) >= 1000:
            suggestions.append("VGG (深度卷积网络)")
            suggestions.append("ResNet (残差网络)")
        
        # 基于数据类型
        if len(categorical_cols) > 0:
            suggestions.append("Random Forest (集成学习)")
            suggestions.append("XGBoost (梯度提升)")
        
        # 基础模型
        suggestions.extend(["Linear Regression", "SVM", "KNN"])
        
        return suggestions[:5]  # 返回前5个建议
    
    def create_ann_model(self, input_dim, hidden_layers=[64, 32], output_dim=1):
        """创建人工神经网络模型"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            
            model = Sequential()
            
            # 输入层
            model.add(Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
            model.add(Dropout(0.2))
            
            # 隐藏层
            for units in hidden_layers[1:]:
                model.add(Dense(units, activation='relu'))
                model.add(Dropout(0.2))
            
            # 输出层
            model.add(Dense(output_dim, activation='linear'))
            
            # 编译模型
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            return model
        except ImportError:
            self.logger.warning("TensorFlow未安装，无法创建ANN模型")
            return None
    
    def create_1d_cnn_model(self, input_shape, filters=64, kernel_size=3):
        """创建1D CNN模型"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
            
            model = Sequential()
            
            # 1D卷积层
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))
            
            # 更多卷积层
            model.add(Conv1D(filters=filters//2, kernel_size=kernel_size, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))
            
            # 全连接层
            model.add(Flatten())
            model.add(Dense(50, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='linear'))
            
            # 编译模型
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            return model
        except ImportError:
            self.logger.warning("TensorFlow未安装，无法创建1D CNN模型")
            return None
    
    def predict_with_model(self, df, model_type='ann', target_column=None):
        """使用指定模型进行预测"""
        if target_column is None:
            # 自动选择目标列（最后一个数值列）
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {"error": "没有找到数值列"}
            target_column = numeric_cols[-1]
        
        # 准备数据
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_column]
        if len(feature_cols) == 0:
            return {"error": "没有找到特征列"}
        
        X = df[feature_cols].values
        y = df[target_column].values
        
        # 数据标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # 分割数据
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        
        predictions = {}
        
        if model_type == 'ann':
            model = self.create_ann_model(input_dim=len(feature_cols))
            if model:
                # 训练模型
                model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                
                # 预测
                y_pred = model.predict(X_test, verbose=0)
                y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
                
                # 计算指标
                mse = np.mean((y_test_original - y_pred_original) ** 2)
                mae = np.mean(np.abs(y_test_original - y_pred_original))
                
                predictions = {
                    'model_type': 'ANN',
                    'mse': float(mse),
                    'mae': float(mae),
                    'predictions': y_pred_original.tolist(),
                    'actual': y_test_original.tolist()
                }
        
        elif model_type == '1d_cnn':
            # 重塑数据为1D CNN格式
            X_train_1d = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test_1d = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            model = self.create_1d_cnn_model(input_shape=(len(feature_cols), 1))
            if model:
                # 训练模型
                model.fit(X_train_1d, y_train, epochs=50, batch_size=32, verbose=0)
                
                # 预测
                y_pred = model.predict(X_test_1d, verbose=0)
                y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
                
                # 计算指标
                mse = np.mean((y_test_original - y_pred_original) ** 2)
                mae = np.mean(np.abs(y_test_original - y_pred_original))
                
                predictions = {
                    'model_type': '1D CNN',
                    'mse': float(mse),
                    'mae': float(mae),
                    'predictions': y_pred_original.tolist(),
                    'actual': y_test_original.tolist()
                }
        
        else:
            # 使用传统机器学习模型
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.svm import SVR
            
            if model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == 'linear_regression':
                model = LinearRegression()
            elif model_type == 'svm':
                model = SVR(kernel='rbf')
            else:
                model = LinearRegression()
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            # 计算指标
            mse = np.mean((y_test_original - y_pred_original) ** 2)
            mae = np.mean(np.abs(y_test_original - y_pred_original))
            
            predictions = {
                'model_type': model_type,
                'mse': float(mse),
                'mae': float(mae),
                'predictions': y_pred_original.tolist(),
                'actual': y_test_original.tolist()
            }
        
        return predictions


class DataPreprocessingAPI(Resource):
    """数据预处理API接口"""
    
    def load_data(self, data_id):
        """加载数据"""
        try:
            upload_dir = os.path.join(os.getcwd(), 'uploads')
            storage_file = os.path.join(upload_dir, f"{data_id}_info.json")
            
            if not os.path.exists(storage_file):
                return None
            
            with open(storage_file, 'r', encoding='utf-8') as f:
                data_info = json.load(f)
            
            # 从原始文件重新读取完整数据
            file_path = os.path.join(upload_dir, f"{data_id}_{data_info['fileName']}")
            if not os.path.exists(file_path):
                return None
            
            if data_info['fileName'].endswith('.csv'):
                df = pd.read_csv(file_path)
            elif data_info['fileName'].endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return None
            
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            return None
    
    def save_data(self, df, data_id):
        """保存数据"""
        try:
            upload_dir = os.path.join(os.getcwd(), 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            
            # 保存为CSV
            file_path = os.path.join(upload_dir, f"{data_id}.csv")
            df.to_csv(file_path, index=False)
            
            # 保存数据信息
            data_info = {
                "dataId": data_id,
                "fileName": f"{data_id}.csv",
                "fileSize": os.path.getsize(file_path),
                "uploadTime": datetime.now().isoformat(),
                "shape": df.shape,
                "columns": df.columns.tolist()
            }
            
            storage_file = os.path.join(upload_dir, f"{data_id}_info.json")
            with open(storage_file, 'w', encoding='utf-8') as f:
                json.dump(data_info, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"保存数据失败: {str(e)}")
            return False
    
    def post(self):
        try:
            data = request.get_json()
            data_id = data.get('dataId')
            options = data.get('options', {})
            
            if not data_id:
                return {"error": "缺少数据ID"}, 400
            
            # 加载数据
            df = self.load_data(data_id)
            if df is None:
                return {"error": "数据不存在"}, 404
            
            # 执行数据预处理
            preprocessor = DataPreprocessor()
            df_processed, report = preprocessor.preprocess_data(df, options)
            
            # 保存处理后的数据
            processed_id = f"processed_{data_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.save_data(df_processed, processed_id)
            
            return {
                "success": True,
                "originalDataId": data_id,
                "processedDataId": processed_id,
                "preprocessingReport": report,
                "message": "数据预处理完成"
            }, 200
            
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            return {"error": f"数据预处理失败: {str(e)}"}, 500


class AIModelPredictionAPI(Resource):
    """AI模型预测API接口"""
    
    def load_data(self, data_id):
        """加载数据"""
        try:
            upload_dir = os.path.join(os.getcwd(), 'uploads')
            storage_file = os.path.join(upload_dir, f"{data_id}_info.json")
            
            if not os.path.exists(storage_file):
                return None
            
            with open(storage_file, 'r', encoding='utf-8') as f:
                data_info = json.load(f)
            
            # 从原始文件重新读取完整数据
            file_path = os.path.join(upload_dir, f"{data_id}_{data_info['fileName']}")
            if not os.path.exists(file_path):
                return None
            
            if data_info['fileName'].endswith('.csv'):
                df = pd.read_csv(file_path)
            elif data_info['fileName'].endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return None
            
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            return None
    
    def post(self):
        try:
            data = request.get_json()
            data_id = data.get('dataId')
            model_type = data.get('modelType', 'ann')
            target_column = data.get('targetColumn')
            
            if not data_id:
                return {"error": "缺少数据ID"}, 400
            
            # 加载数据
            df = self.load_data(data_id)
            if df is None:
                return {"error": "数据不存在"}, 404
            
            # 执行AI预测
            predictor = AIModelPredictor()
            
            # 获取模型建议
            model_suggestions = predictor.suggest_model_type(df)
            
            # 执行预测
            predictions = predictor.predict_with_model(df, model_type, target_column)
            
            return {
                "success": True,
                "dataId": data_id,
                "modelType": model_type,
                "modelSuggestions": model_suggestions,
                "predictions": predictions,
                "message": "AI预测完成"
            }, 200
            
        except Exception as e:
            logger.error(f"AI预测失败: {str(e)}")
            return {"error": f"AI预测失败: {str(e)}"}, 500


class GenerateVisualization(Resource):
    """生成可视化数据接口"""
    
    def post(self):
        try:
            data = request.get_json()
            data_id = data.get('dataId')
            chart_type = data.get('chartType', 'line')
            selected_columns = data.get('selectedColumns', '')
            
            if not data_id:
                return {"error": "缺少数据ID"}, 400
            
            # 加载数据
            df = self.load_data(data_id)
            if df is None:
                return {"error": "数据不存在"}, 404
            
            # 根据用户选择的列过滤数据
            if selected_columns and selected_columns.strip():
                selected_cols = selected_columns.split(',')
                available_cols = [col for col in selected_cols if col in df.columns]
                if available_cols:
                    df = df[available_cols]
            
            # 使用Qwen3-Coder生成图表建议和数据
            try:
                qwen_analyzer = QwenAnalyzer()
                chart_suggestion = qwen_analyzer.suggest_chart_type(df.to_dict('records'), chart_type)
                logger.info(f"Qwen图表建议: {chart_suggestion}")
            except Exception as e:
                logger.warning(f"Qwen图表建议失败: {str(e)}")
            
            # 根据图表类型生成数据
            if chart_type == 'line':
                chart_data = self._generate_line_chart(df, selected_columns)
            elif chart_type == 'bar':
                chart_data = self._generate_bar_chart(df, selected_columns)
            elif chart_type == 'heatmap':
                chart_data = self._generate_heatmap(df)
            elif chart_type == 'pie':
                chart_data = self._generate_pie_chart(df, selected_columns)
            elif chart_type == 'pca':
                chart_data = self._generate_pca_chart(df)
            else:
                chart_data = self._generate_line_chart(df, selected_columns)  # 默认折线图
            
            # 找到人员列用于前端选择
            person_cols = [col for col in df.columns if col.startswith('Person_')]
            
            return {
                "success": True,
                "chartType": chart_type,
                "data": chart_data,
                "columns": df.columns.tolist(),
                "personColumns": person_cols,
                "selectedColumns": selected_columns
            }, 200
            
        except Exception as e:
            logger.error(f"生成图表失败: {str(e)}")
            return {"error": f"生成图表失败: {str(e)}"}, 500
    
    def _generate_pca_chart(self, df):
        """生成PCA降维图表"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return {"error": "需要安装scikit-learn进行PCA分析"}
        
        # 找到数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {"error": "PCA需要至少2个数值列"}
        
        # 准备数据
        data = df[numeric_cols].dropna()
        if len(data) < 2:
            return {"error": "有效数据不足"}
        
        # 标准化数据
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # 执行PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data_scaled)
        
        # 计算解释方差比
        explained_variance = pca.explained_variance_ratio_
        
        # 返回Plotly格式的数据
        return {
            "data": [{
                "x": pca_result[:, 0].tolist(),
                "y": pca_result[:, 1].tolist(),
                "mode": "markers",
                "type": "scatter",
                "name": "PCA降维",
                "marker": {"color": "#1f77b4", "size": 8}
            }],
            "layout": {
                "title": f"PCA降维图 (解释方差: {explained_variance[0]:.2%}, {explained_variance[1]:.2%})",
                "xaxis": {"title": f"PC1 ({explained_variance[0]:.2%})"},
                "yaxis": {"title": f"PC2 ({explained_variance[1]:.2%})"},
                "showlegend": True
            }
        }


class GenerateVisualization(Resource):
    """生成可视化数据接口"""
    
    def post(self):
        try:
            data = request.get_json()
            logger.info(f"收到可视化请求: {data.get('dataId')}")
            
            # 验证请求数据
            required_fields = ['dataId', 'chartType', 'data']
            for field in required_fields:
                if field not in data:
                    return {"error": f"缺少必需字段: {field}"}, 400
            
            # 生成可视化数据
            result = visualization_generator.generate(
                data_id=data['dataId'],
                chart_type=data['chartType'],
                data=data['data'],
                parameters=data.get('parameters', {})
            )
            
            logger.info(f"可视化数据生成完成: {data['dataId']}")
            return result, 200
            
        except Exception as e:
            logger.error(f"可视化数据生成失败: {str(e)}")
            return {"error": f"可视化数据生成失败: {str(e)}"}, 500


class CalculateStats(Resource):
    """计算统计信息接口"""
    
    def post(self):
        try:
            data = request.get_json()
            logger.info(f"收到统计计算请求: {data.get('dataId')}")
            
            # 验证请求数据
            required_fields = ['dataId', 'data']
            for field in required_fields:
                if field not in data:
                    return {"error": f"缺少必需字段: {field}"}, 400
            
            # 计算统计信息
            result = stats_calculator.calculate(
                data_id=data['dataId'],
                data=data['data']
            )
            
            logger.info(f"统计信息计算完成: {data['dataId']}")
            return result, 200
            
        except Exception as e:
            logger.error(f"统计信息计算失败: {str(e)}")
            return {"error": f"统计信息计算失败: {str(e)}"}, 500


class PreprocessData(Resource):
    """数据预处理接口"""
    
    def post(self):
        try:
            data = request.get_json()
            logger.info(f"收到数据预处理请求: {data.get('dataId')}")
            
            # 验证请求数据
            required_fields = ['dataId', 'data', 'options']
            for field in required_fields:
                if field not in data:
                    return {"error": f"缺少必需字段: {field}"}, 400
            
            # 执行数据预处理
            result = data_processor.preprocess(
                data_id=data['dataId'],
                data=data['data'],
                options=data['options']
            )
            
            logger.info(f"数据预处理完成: {data['dataId']}")
            return result, 200
            
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            return {"error": f"数据预处理失败: {str(e)}"}, 500


class UploadData(Resource):
    """数据上传接口"""
    
    def post(self):
        try:
            # 检查是否有文件上传
            if 'file' not in request.files:
                return {"error": "没有上传文件"}, 400
            
            file = request.files['file']
            if file.filename == '':
                return {"error": "没有选择文件"}, 400
            
            # 获取其他表单数据
            name = request.form.get('name', '')
            category = request.form.get('category', '')
            description = request.form.get('description', '')
            
            if not name:
                return {"error": "数据名称不能为空"}, 400
            
            # 生成数据ID
            data_id = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(file.filename) % 10000}"
            
            # 保存文件
            upload_dir = os.path.join(os.getcwd(), 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, f"{data_id}_{file.filename}")
            file.save(file_path)
            
            # 读取数据
            try:
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file.filename.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    return {"error": "不支持的文件格式，请上传CSV或Excel文件"}, 400
                
                # 转换为JSON格式
                data_json = df.to_dict('records')
                
                # 保存数据信息
                data_info = {
                    "dataId": data_id,
                    "name": name,
                    "category": category,
                    "description": description,
                    "fileName": file.filename,
                    "fileSize": os.path.getsize(file_path),
                    "uploadTime": datetime.now().isoformat(),
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "data": data_json[:100]  # 只保存前100行用于预览
                }
                
                # 保存到本地存储
                storage_file = os.path.join(upload_dir, f"{data_id}_info.json")
                with open(storage_file, 'w', encoding='utf-8') as f:
                    json.dump(data_info, f, ensure_ascii=False, indent=2)
                
                logger.info(f"数据上传成功: {data_id}")
                
                return {
                    "success": True,
                    "dataId": data_id,
                    "message": "数据上传成功",
                    "info": data_info
                }
                
            except Exception as e:
                # 删除上传的文件
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise e
            
        except Exception as e:
            logger.error(f"数据上传失败: {str(e)}")
            return {"error": f"数据上传失败: {str(e)}"}, 500
    
class DataList(Resource):
    """数据列表接口"""
    
    def get(self):
        try:
            upload_dir = os.path.join(os.getcwd(), 'uploads')
            if not os.path.exists(upload_dir):
                return {"data": []}, 200
            
            data_list = []
            for filename in os.listdir(upload_dir):
                if filename.endswith('_info.json'):
                    info_file = os.path.join(upload_dir, filename)
                    try:
                        with open(info_file, 'r', encoding='utf-8') as f:
                            data_info = json.load(f)
                        data_list.append(data_info)
                    except Exception as e:
                        logger.warning(f"读取数据信息失败 {filename}: {e}")
                        continue
            
            # 按上传时间排序
            data_list.sort(key=lambda x: x.get('uploadTime', ''), reverse=True)
            
            return {"data": data_list}, 200
            
        except Exception as e:
            logger.error(f"获取数据列表失败: {str(e)}")
            return {"error": f"获取数据列表失败: {str(e)}"}, 500


class DashboardStats(Resource):
    """仪表板统计信息接口"""
    
    def get(self):
        try:
            # 返回模拟的仪表板统计数据
            stats = {
                "totalData": 5,
                "totalAnalysis": 12,
                "totalCharts": 8,
                "activeUsers": 3,
                "recentActivity": [
                    {"type": "upload", "message": "新数据上传", "time": "2分钟前"},
                    {"type": "analysis", "message": "数据分析完成", "time": "5分钟前"},
                    {"type": "chart", "message": "图表生成", "time": "10分钟前"}
                ]
            }
            return stats, 200
        except Exception as e:
            logger.error(f"获取仪表板统计失败: {str(e)}")
            return {"error": f"获取仪表板统计失败: {str(e)}"}, 500


class SystemInfo(Resource):
    """系统信息接口"""
    
    def get(self):
        try:
            import psutil
            import platform
            
            # 获取系统信息
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            
            info = {
                "memoryUsage": round(memory.percent, 1),
                "cpuUsage": round(cpu, 1),
                "diskUsage": round(psutil.disk_usage('/').percent, 1) if platform.system() != 'Windows' else round(psutil.disk_usage('C:').percent, 1),
                "platform": platform.system(),
                "pythonVersion": platform.python_version(),
                "uptime": "2小时15分钟"
            }
            return info, 200
        except Exception as e:
            logger.error(f"获取系统信息失败: {str(e)}")
            # 返回模拟数据
            return {
                "memoryUsage": 45.2,
                "cpuUsage": 12.8,
                "diskUsage": 67.3,
                "platform": "Windows",
                "pythonVersion": "3.11.3",
                "uptime": "2小时15分钟"
            }, 200


class VisualizeData(Resource):
    """数据可视化接口"""
    
    def get(self, data_id):
        try:
            chart_type = request.args.get('chartType', 'line')
            selected_columns = request.args.get('columns', '')  # 支持选择特定列
            
            # 读取数据文件
            upload_dir = os.path.join(os.getcwd(), 'uploads')
            info_file = os.path.join(upload_dir, f"{data_id}_info.json")
            
            if not os.path.exists(info_file):
                return {"error": "数据不存在"}, 404
            
            with open(info_file, 'r', encoding='utf-8') as f:
                data_info = json.load(f)
            
            # 生成图表数据
            data = data_info.get('data', [])
            if not data:
                return {"error": "数据为空"}, 400
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 如果指定了列，只使用选定的列
            if selected_columns:
                selected_cols = [col.strip() for col in selected_columns.split(',')]
                available_cols = [col for col in selected_cols if col in df.columns]
                if available_cols:
                    # 保留时间列和选定的列
                    time_cols = [col for col in df.columns if 'day' in col.lower() or 'time' in col.lower() or 'date' in col.lower()]
                    all_cols = time_cols + available_cols
                    # 确保列存在
                    existing_cols = [col for col in all_cols if col in df.columns]
                    if existing_cols:
                        df = df[existing_cols]
            
            # 使用Qwen3-Coder生成图表建议和数据
            try:
                qwen_analyzer = QwenAnalyzer()
                chart_suggestion = qwen_analyzer.suggest_chart_type(df.to_dict('records'), chart_type)
                logger.info(f"Qwen图表建议: {chart_suggestion}")
            except Exception as e:
                logger.warning(f"Qwen图表建议失败: {str(e)}")
            
            # 根据图表类型生成数据
            if chart_type == 'line':
                chart_data = self._generate_line_chart(df, selected_columns)
            elif chart_type == 'bar':
                chart_data = self._generate_bar_chart(df, selected_columns)
            elif chart_type == 'heatmap':
                chart_data = self._generate_heatmap(df)
            elif chart_type == 'pie':
                chart_data = self._generate_pie_chart(df, selected_columns)
            elif chart_type == 'pca':
                chart_data = self._generate_pca_chart(df)
            else:
                chart_data = self._generate_line_chart(df, selected_columns)  # 默认折线图
            
            # 找到人员列用于前端选择
            person_cols = [col for col in df.columns if col.startswith('Person_')]
            
            return {
                "success": True,
                "chartType": chart_type,
                "data": chart_data,
                "columns": df.columns.tolist(),
                "personColumns": person_cols,
                "selectedColumns": selected_columns
            }, 200
            
        except Exception as e:
            logger.error(f"生成图表失败: {str(e)}")
            return {"error": f"生成图表失败: {str(e)}"}, 500
    
    def _generate_line_chart(self, df, selected_columns=None):
        """生成折线图数据 - 支持人员选择"""
        # 找到时间列
        time_col = None
        for col in df.columns:
            if 'day' in col.lower() or 'time' in col.lower() or 'date' in col.lower():
                time_col = col
                break
        
        if time_col is None:
            time_col = df.columns[0]  # 使用第一列作为x轴
        
        # 找到人员列（Person_开头的列）
        person_cols = [col for col in df.columns if col.startswith('Person_')]
        
        if selected_columns and selected_columns.strip():
            # 用户选择了特定列
            selected_cols = [col.strip() for col in selected_columns.split(',')]
            available_cols = [col for col in selected_cols if col in df.columns]
            if available_cols:
                person_cols = available_cols
        
        if not person_cols:
            return {"error": "没有找到人员数据列"}
        
        # 生成多条线，每条线代表一个人
        data = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, person_col in enumerate(person_cols[:10]):  # 最多显示10个人
            data.append({
                "x": df[time_col].tolist(),
                "y": df[person_col].tolist(),
                "type": "scatter",
                "mode": "lines+markers",
                "name": person_col,
                "line": {"color": colors[i % len(colors)]}
            })
        
        return {
            "data": data,
            "layout": {
                "title": f"人员数据时间序列图 ({len(person_cols)}人)",
                "xaxis": {"title": time_col},
                "yaxis": {"title": "数值"},
                "showlegend": True
            }
        }
    
    def _generate_bar_chart(self, df, selected_columns=None):
        """生成柱状图数据 - 支持人员选择"""
        # 找到人员列（Person_开头的列）
        person_cols = [col for col in df.columns if col.startswith('Person_')]
        
        if selected_columns and selected_columns.strip():
            # 用户选择了特定列
            selected_cols = [col.strip() for col in selected_columns.split(',')]
            available_cols = [col for col in selected_cols if col in df.columns]
            if available_cols:
                person_cols = available_cols
        
        if not person_cols:
            return {"error": "没有找到人员数据列"}
        
        # 计算每个人的平均值
        person_means = []
        person_names = []
        for person_col in person_cols:
            mean_val = df[person_col].mean()
            person_means.append(mean_val)
            person_names.append(person_col)
        
        # 返回Plotly格式的数据
        return {
            "data": [{
                "x": person_names,
                "y": person_means,
                "type": "bar",
                "name": "平均数值",
                "marker": {"color": "#2ca02c"}
            }],
            "layout": {
                "title": f"人员数据柱状图 ({len(person_cols)}人)",
                "xaxis": {"title": "人员"},
                "yaxis": {"title": "平均数值"},
                "showlegend": True
            }
        }
    
    def _generate_scatter_chart(self, df):
        """生成散点图数据"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {"error": "数据不足，需要至少2个数值列"}
        
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        
        # 返回Plotly格式的数据
        return {
            "data": [{
                "x": df[x_col].tolist(),
                "y": df[y_col].tolist(),
                "type": "scatter",
                "mode": "markers",
                "name": f"{x_col} vs {y_col}",
                "marker": {"color": "#ff7f0e", "size": 8}
            }],
            "layout": {
                "title": f"{x_col} vs {y_col} 散点图",
                "xaxis": {"title": x_col},
                "yaxis": {"title": y_col},
                "showlegend": True
            }
        }
    
    def _generate_heatmap(self, df):
        """生成热力图数据"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {"error": "数据不足，需要至少2个数值列"}
        
        # 计算相关性矩阵
        corr_matrix = df[numeric_cols].corr()
        
        # 返回Plotly格式的数据
        return {
            "data": [{
                "z": corr_matrix.values.tolist(),
                "x": corr_matrix.columns.tolist(),
                "y": corr_matrix.index.tolist(),
                "type": "heatmap",
                "colorscale": "RdBu",
                "zmid": 0
            }],
            "layout": {
                "title": "数据相关性热力图",
                "xaxis": {"title": "变量"},
                "yaxis": {"title": "变量"}
            }
        }
    
    def _generate_multi_line_chart(self, df):
        """生成多线图数据"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {"error": "数据不足，需要至少2个数值列"}
        
        # 找到时间列
        time_col = None
        for col in df.columns:
            if 'day' in col.lower() or 'time' in col.lower() or 'date' in col.lower():
                time_col = col
                break
        
        if time_col is None:
            time_col = df.columns[0]  # 使用第一列作为x轴
        
        # 生成多条线
        traces = []
        for col in numeric_cols[:10]:  # 限制显示前10个数值列
            if col != time_col:
                traces.append({
                    "x": df[time_col].tolist(),
                    "y": df[col].tolist(),
                    "type": "scatter",
                    "mode": "lines",
                    "name": col
                })
        
        return {
            "data": traces,
            "layout": {
                "title": f"多变量时间序列图",
                "xaxis": {"title": time_col},
                "yaxis": {"title": "数值"},
                "showlegend": True
            }
        }
    
    def _generate_auto_chart(self, df):
        """根据数据特征自动选择图表类型"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # 如果有3个或更多数值列，生成3D散点图
        if len(numeric_cols) >= 3:
            return self._generate_3d_chart(df)
        # 如果有2个数值列，生成散点图
        elif len(numeric_cols) >= 2:
            return self._generate_scatter_chart(df)
        # 如果有1个数值列，生成柱状图
        elif len(numeric_cols) >= 1:
            return self._generate_bar_chart(df)
        # 如果只有分类数据，生成饼图
        elif len(categorical_cols) >= 1:
            return self._generate_pie_chart(df)
        else:
            return {"error": "无法识别数据类型"}
    
    def _generate_pie_chart(self, df, selected_columns=None):
        """生成饼图数据 - 支持人员选择"""
        try:
            # 找到人员列（Person_开头的列）
            person_cols = [col for col in df.columns if col.startswith('Person_')]
            
            if selected_columns and selected_columns.strip():
                # 用户选择了特定列
                selected_cols = [col.strip() for col in selected_columns.split(',')]
                available_cols = [col for col in selected_cols if col in df.columns]
                if available_cols:
                    person_cols = available_cols
            
            if not person_cols:
                return {"error": "没有找到人员数据列"}
            
            # 限制显示的人员数量，避免饼图过于复杂
            if len(person_cols) > 10:
                # 只显示前10个人员
                person_cols = person_cols[:10]
            
            # 计算每个人的总数值
            person_totals = []
            person_names = []
            for person_col in person_cols:
                if person_col in df.columns:
                    try:
                        total_val = float(df[person_col].sum())
                        if not np.isnan(total_val) and total_val > 0:
                            person_totals.append(total_val)
                            person_names.append(person_col)
                    except (ValueError, TypeError):
                        continue
            
            if not person_totals:
                return {"error": "没有有效的人员数据"}
            
            # 返回Plotly格式的数据
            return {
                "data": [{
                    "labels": person_names,
                    "values": person_totals,
                    "type": "pie",
                    "name": "人员数据分布"
                }],
                "layout": {
                    "title": f"人员数据分布饼图 ({len(person_cols)}人)",
                    "showlegend": True
                }
            }
        except Exception as e:
            logger.error(f"生成饼图失败: {str(e)}")
            return {"error": f"生成饼图失败: {str(e)}"}
    
    def _generate_pca_chart(self, df):
        """生成PCA降维图表"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return {"error": "需要安装scikit-learn进行PCA分析"}
        
        # 找到数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {"error": "PCA需要至少2个数值列"}
        
        # 准备数据
        data = df[numeric_cols].dropna()
        if len(data) < 2:
            return {"error": "有效数据不足"}
        
        # 标准化数据
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # 执行PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data_scaled)
        
        # 计算解释方差比
        explained_variance = pca.explained_variance_ratio_
        
        # 返回Plotly格式的数据
        return {
            "data": [{
                "x": pca_result[:, 0].tolist(),
                "y": pca_result[:, 1].tolist(),
                "mode": "markers",
                "type": "scatter",
                "name": "PCA降维",
                "marker": {"color": "#1f77b4", "size": 8}
            }],
            "layout": {
                "title": f"PCA降维图 (解释方差: {explained_variance[0]:.2%}, {explained_variance[1]:.2%})",
                "xaxis": {"title": f"PC1 ({explained_variance[0]:.2%})"},
                "yaxis": {"title": f"PC2 ({explained_variance[1]:.2%})"},
                "showlegend": True
            }
        }
    
    def _generate_3d_chart(self, df):
        """生成3D图表数据"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 3:
            return {"error": "数据不足，需要至少3个数值列"}
        
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        z_col = numeric_cols[2]
        
        # 返回Plotly格式的数据
        return {
            "data": [{
                "x": df[x_col].tolist(),
                "y": df[y_col].tolist(),
                "z": df[z_col].tolist(),
                "type": "scatter3d",
                "mode": "markers",
                "name": f"{x_col} vs {y_col} vs {z_col}",
                "marker": {"color": "#d62728", "size": 5}
            }],
            "layout": {
                "title": f"{x_col} vs {y_col} vs {z_col} 3D散点图",
                "scene": {
                    "xaxis": {"title": x_col},
                    "yaxis": {"title": y_col},
                    "zaxis": {"title": z_col}
                },
                "showlegend": True
            }
        }


# 注册API路由
api.add_resource(HealthCheck, '/api/health')
api.add_resource(UploadData, '/api/data/upload')
api.add_resource(DataList, '/api/data/list')
api.add_resource(AnalyzeData, '/api/data/analyze')
api.add_resource(GenerateVisualization, '/api/visualize')
api.add_resource(CalculateStats, '/api/stats')
api.add_resource(PreprocessData, '/api/preprocess')
api.add_resource(DashboardStats, '/api/dashboard/stats')
api.add_resource(SystemInfo, '/api/system/info')
api.add_resource(VisualizeData, '/api/visualize/<string:data_id>')
api.add_resource(DataPreprocessingAPI, '/api/data/preprocess')
api.add_resource(AIModelPredictionAPI, '/api/ai/predict')


@app.errorhandler(404)
def not_found(error):
    return {"error": "接口不存在"}, 404


@app.errorhandler(500)
def internal_error(error):
    return {"error": "服务器内部错误"}, 500


if __name__ == '__main__':
    # 启动应用
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"启动Python数据分析服务，端口: {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
