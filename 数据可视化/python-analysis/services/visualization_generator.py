#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化数据生成器
生成各种类型的图表数据
"""

import json
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


class VisualizationGenerator:
    """可视化数据生成器"""
    
    def __init__(self):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置样式
        sns.set_style("whitegrid")
        plt.style.use('seaborn-v0_8')
    
    def generate(self, data_id: str, chart_type: str, data: List[Dict], 
                parameters: Dict = None) -> Dict[str, Any]:
        """
        生成可视化数据
        
        Args:
            data_id: 数据ID
            chart_type: 图表类型
            data: 数据列表
            parameters: 图表参数
            
        Returns:
            可视化数据
        """
        try:
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 根据图表类型生成不同的可视化
            if chart_type == "line":
                return self._generate_line_chart(df, parameters)
            elif chart_type == "bar":
                return self._generate_bar_chart(df, parameters)
            elif chart_type == "scatter":
                return self._generate_scatter_chart(df, parameters)
            elif chart_type == "pie":
                return self._generate_pie_chart(df, parameters)
            elif chart_type == "heatmap":
                return self._generate_heatmap(df, parameters)
            elif chart_type == "histogram":
                return self._generate_histogram(df, parameters)
            elif chart_type == "box":
                return self._generate_box_plot(df, parameters)
            elif chart_type == "radar":
                return self._generate_radar_chart(df, parameters)
            elif chart_type == "3d":
                return self._generate_3d_chart(df, parameters)
            else:
                return self._generate_dashboard(df, parameters)
                
        except Exception as e:
            logger.error(f"可视化数据生成失败: {e}")
            raise
    
    def _generate_line_chart(self, df: pd.DataFrame, parameters: Dict) -> Dict[str, Any]:
        """生成折线图"""
        result = {
            "chart_type": "line",
            "data": [],
            "layout": {},
            "config": {}
        }
        
        # 获取参数
        x_column = parameters.get('x_column', df.columns[0])
        y_columns = parameters.get('y_columns', [col for col in df.columns if col != x_column])
        title = parameters.get('title', '折线图')
        
        # 确保y_columns是列表
        if isinstance(y_columns, str):
            y_columns = [y_columns]
        
        # 生成图表数据
        for y_col in y_columns:
            if y_col in df.columns:
                # 转换为数值类型
                df_clean = df[[x_column, y_col]].copy()
                df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors='coerce')
                df_clean = df_clean.dropna()
                
                if len(df_clean) > 0:
                    result["data"].append({
                        "x": df_clean[x_column].tolist(),
                        "y": df_clean[y_col].tolist(),
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": y_col
                    })
        
        # 设置布局
        result["layout"] = {
            "title": title,
            "xaxis": {"title": x_column},
            "yaxis": {"title": "数值"},
            "hovermode": "closest"
        }
        
        return result
    
    def _generate_bar_chart(self, df: pd.DataFrame, parameters: Dict) -> Dict[str, Any]:
        """生成柱状图"""
        result = {
            "chart_type": "bar",
            "data": [],
            "layout": {},
            "config": {}
        }
        
        # 获取参数
        x_column = parameters.get('x_column', df.columns[0])
        y_column = parameters.get('y_column', df.columns[1] if len(df.columns) > 1 else df.columns[0])
        title = parameters.get('title', '柱状图')
        orientation = parameters.get('orientation', 'v')
        
        # 数据聚合
        if y_column in df.columns:
            df_clean = df[[x_column, y_column]].copy()
            df_clean[y_column] = pd.to_numeric(df_clean[y_column], errors='coerce')
            df_clean = df_clean.dropna()
            
            if len(df_clean) > 0:
                # 按x_column分组求和
                grouped = df_clean.groupby(x_column)[y_column].sum().reset_index()
                
                result["data"].append({
                    "x": grouped[x_column].tolist(),
                    "y": grouped[y_column].tolist(),
                    "type": "bar",
                    "name": y_column,
                    "orientation": orientation
                })
        
        # 设置布局
        result["layout"] = {
            "title": title,
            "xaxis": {"title": x_column},
            "yaxis": {"title": y_column},
            "barmode": "group"
        }
        
        return result
    
    def _generate_scatter_chart(self, df: pd.DataFrame, parameters: Dict) -> Dict[str, Any]:
        """生成散点图"""
        result = {
            "chart_type": "scatter",
            "data": [],
            "layout": {},
            "config": {}
        }
        
        # 获取参数
        x_column = parameters.get('x_column', df.columns[0])
        y_column = parameters.get('y_column', df.columns[1] if len(df.columns) > 1 else df.columns[0])
        color_column = parameters.get('color_column', None)
        size_column = parameters.get('size_column', None)
        title = parameters.get('title', '散点图')
        
        # 准备数据
        columns = [x_column, y_column]
        if color_column and color_column in df.columns:
            columns.append(color_column)
        if size_column and size_column in df.columns:
            columns.append(size_column)
        
        df_clean = df[columns].copy()
        df_clean[x_column] = pd.to_numeric(df_clean[x_column], errors='coerce')
        df_clean[y_column] = pd.to_numeric(df_clean[y_column], errors='coerce')
        df_clean = df_clean.dropna()
        
        if len(df_clean) > 0:
            scatter_data = {
                "x": df_clean[x_column].tolist(),
                "y": df_clean[y_column].tolist(),
                "type": "scatter",
                "mode": "markers",
                "name": "数据点"
            }
            
            # 添加颜色映射
            if color_column and color_column in df_clean.columns:
                scatter_data["marker"] = {
                    "color": df_clean[color_column].tolist(),
                    "colorscale": "Viridis",
                    "showscale": True
                }
            
            # 添加大小映射
            if size_column and size_column in df_clean.columns:
                size_values = pd.to_numeric(df_clean[size_column], errors='coerce')
                scatter_data["marker"]["size"] = size_values.tolist()
            
            result["data"].append(scatter_data)
        
        # 设置布局
        result["layout"] = {
            "title": title,
            "xaxis": {"title": x_column},
            "yaxis": {"title": y_column}
        }
        
        return result
    
    def _generate_pie_chart(self, df: pd.DataFrame, parameters: Dict) -> Dict[str, Any]:
        """生成饼图"""
        result = {
            "chart_type": "pie",
            "data": [],
            "layout": {},
            "config": {}
        }
        
        # 获取参数
        label_column = parameters.get('label_column', df.columns[0])
        value_column = parameters.get('value_column', df.columns[1] if len(df.columns) > 1 else df.columns[0])
        title = parameters.get('title', '饼图')
        
        # 数据聚合
        if value_column in df.columns:
            df_clean = df[[label_column, value_column]].copy()
            df_clean[value_column] = pd.to_numeric(df_clean[value_column], errors='coerce')
            df_clean = df_clean.dropna()
            
            if len(df_clean) > 0:
                # 按label_column分组求和
                grouped = df_clean.groupby(label_column)[value_column].sum().reset_index()
                
                result["data"].append({
                    "labels": grouped[label_column].tolist(),
                    "values": grouped[value_column].tolist(),
                    "type": "pie",
                    "hole": 0.3
                })
        
        # 设置布局
        result["layout"] = {
            "title": title,
            "showlegend": True
        }
        
        return result
    
    def _generate_heatmap(self, df: pd.DataFrame, parameters: Dict) -> Dict[str, Any]:
        """生成热力图"""
        result = {
            "chart_type": "heatmap",
            "data": [],
            "layout": {},
            "config": {}
        }
        
        # 获取参数
        title = parameters.get('title', '热力图')
        
        # 选择数值列
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            # 计算相关性矩阵
            corr_matrix = numeric_df.corr()
            
            result["data"].append({
                "z": corr_matrix.values.tolist(),
                "x": corr_matrix.columns.tolist(),
                "y": corr_matrix.columns.tolist(),
                "type": "heatmap",
                "colorscale": "RdBu",
                "zmid": 0
            })
        
        # 设置布局
        result["layout"] = {
            "title": title,
            "xaxis": {"side": "bottom"},
            "yaxis": {"autorange": "reversed"}
        }
        
        return result
    
    def _generate_histogram(self, df: pd.DataFrame, parameters: Dict) -> Dict[str, Any]:
        """生成直方图"""
        result = {
            "chart_type": "histogram",
            "data": [],
            "layout": {},
            "config": {}
        }
        
        # 获取参数
        column = parameters.get('column', df.select_dtypes(include=[np.number]).columns[0])
        bins = parameters.get('bins', 30)
        title = parameters.get('title', f'{column} 直方图')
        
        if column in df.columns:
            # 转换为数值类型
            series = pd.to_numeric(df[column], errors='coerce').dropna()
            
            if len(series) > 0:
                result["data"].append({
                    "x": series.tolist(),
                    "type": "histogram",
                    "nbinsx": bins,
                    "name": column
                })
        
        # 设置布局
        result["layout"] = {
            "title": title,
            "xaxis": {"title": column},
            "yaxis": {"title": "频次"},
            "bargap": 0.1
        }
        
        return result
    
    def _generate_box_plot(self, df: pd.DataFrame, parameters: Dict) -> Dict[str, Any]:
        """生成箱线图"""
        result = {
            "chart_type": "box",
            "data": [],
            "layout": {},
            "config": {}
        }
        
        # 获取参数
        columns = parameters.get('columns', df.select_dtypes(include=[np.number]).columns.tolist())
        title = parameters.get('title', '箱线图')
        
        if isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            if col in df.columns:
                # 转换为数值类型
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                
                if len(series) > 0:
                    result["data"].append({
                        "y": series.tolist(),
                        "type": "box",
                        "name": col
                    })
        
        # 设置布局
        result["layout"] = {
            "title": title,
            "yaxis": {"title": "数值"}
        }
        
        return result
    
    def _generate_radar_chart(self, df: pd.DataFrame, parameters: Dict) -> Dict[str, Any]:
        """生成雷达图"""
        result = {
            "chart_type": "radar",
            "data": [],
            "layout": {},
            "config": {}
        }
        
        # 获取参数
        columns = parameters.get('columns', df.select_dtypes(include=[np.number]).columns.tolist()[:6])
        title = parameters.get('title', '雷达图')
        
        if isinstance(columns, str):
            columns = [columns]
        
        # 选择数值列
        numeric_df = df[columns].select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 2:
            # 计算平均值
            means = numeric_df.mean()
            
            result["data"].append({
                "r": means.tolist(),
                "theta": means.index.tolist(),
                "type": "scatterpolar",
                "fill": "toself",
                "name": "平均值"
            })
        
        # 设置布局
        result["layout"] = {
            "title": title,
            "polar": {
                "radialaxis": {
                    "visible": True,
                    "range": [0, 1]
                }
            }
        }
        
        return result
    
    def _generate_3d_chart(self, df: pd.DataFrame, parameters: Dict) -> Dict[str, Any]:
        """生成3D图表"""
        result = {
            "chart_type": "3d",
            "data": [],
            "layout": {},
            "config": {}
        }
        
        # 获取参数
        x_column = parameters.get('x_column', df.columns[0])
        y_column = parameters.get('y_column', df.columns[1] if len(df.columns) > 1 else df.columns[0])
        z_column = parameters.get('z_column', df.columns[2] if len(df.columns) > 2 else df.columns[0])
        title = parameters.get('title', '3D散点图')
        
        # 准备数据
        df_clean = df[[x_column, y_column, z_column]].copy()
        df_clean[x_column] = pd.to_numeric(df_clean[x_column], errors='coerce')
        df_clean[y_column] = pd.to_numeric(df_clean[y_column], errors='coerce')
        df_clean[z_column] = pd.to_numeric(df_clean[z_column], errors='coerce')
        df_clean = df_clean.dropna()
        
        if len(df_clean) > 0:
            result["data"].append({
                "x": df_clean[x_column].tolist(),
                "y": df_clean[y_column].tolist(),
                "z": df_clean[z_column].tolist(),
                "type": "scatter3d",
                "mode": "markers",
                "marker": {
                    "size": 3,
                    "color": df_clean[z_column].tolist(),
                    "colorscale": "Viridis"
                }
            })
        
        # 设置布局
        result["layout"] = {
            "title": title,
            "scene": {
                "xaxis": {"title": x_column},
                "yaxis": {"title": y_column},
                "zaxis": {"title": z_column}
            }
        }
        
        return result
    
    def _generate_dashboard(self, df: pd.DataFrame, parameters: Dict) -> Dict[str, Any]:
        """生成综合仪表板"""
        result = {
            "chart_type": "dashboard",
            "charts": [],
            "layout": {},
            "config": {}
        }
        
        # 获取数值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) >= 2:
            # 生成多个图表
            charts = []
            
            # 1. 趋势图
            if len(numeric_columns) >= 2:
                trend_chart = self._generate_line_chart(df, {
                    'x_column': df.columns[0],
                    'y_columns': numeric_columns[:3],
                    'title': '趋势分析'
                })
                charts.append(trend_chart)
            
            # 2. 相关性热力图
            if len(numeric_columns) > 2:
                heatmap_chart = self._generate_heatmap(df, {'title': '相关性分析'})
                charts.append(heatmap_chart)
            
            # 3. 分布直方图
            if len(numeric_columns) > 0:
                hist_chart = self._generate_histogram(df, {
                    'column': numeric_columns[0],
                    'title': f'{numeric_columns[0]} 分布'
                })
                charts.append(hist_chart)
            
            # 4. 箱线图
            if len(numeric_columns) > 1:
                box_chart = self._generate_box_plot(df, {
                    'columns': numeric_columns[:5],
                    'title': '数据分布对比'
                })
                charts.append(box_chart)
            
            result["charts"] = charts
        
        # 设置布局
        result["layout"] = {
            "title": "数据可视化仪表板",
            "grid": "2x2"
        }
        
        return result
    
    def _save_chart_as_base64(self, fig) -> str:
        """将图表保存为base64字符串"""
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            return image_base64
        except Exception as e:
            logger.error(f"图表保存失败: {e}")
            return ""
