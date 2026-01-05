#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据处理器
处理数据清洗、预处理和转换
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        self.knn_imputer = KNNImputer(n_neighbors=5)
    
    def preprocess(self, data_id: str, data: List[Dict], options: Dict) -> Dict[str, Any]:
        """
        数据预处理
        
        Args:
            data_id: 数据ID
            data: 原始数据
            options: 预处理选项
            
        Returns:
            预处理结果
        """
        try:
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 记录原始数据信息
            original_info = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "duplicate_rows": df.duplicated().sum()
            }
            
            # 执行预处理步骤
            processed_df = df.copy()
            preprocessing_steps = []
            
            # 1. 处理缺失值
            if options.get('handle_missing', False):
                processed_df, missing_info = self._handle_missing_values(processed_df, options)
                preprocessing_steps.append(missing_info)
            
            # 2. 处理重复值
            if options.get('remove_duplicates', False):
                processed_df, duplicate_info = self._remove_duplicates(processed_df)
                preprocessing_steps.append(duplicate_info)
            
            # 3. 数据类型转换
            if options.get('convert_types', False):
                processed_df, type_info = self._convert_data_types(processed_df, options)
                preprocessing_steps.append(type_info)
            
            # 4. 异常值处理
            if options.get('handle_outliers', False):
                processed_df, outlier_info = self._handle_outliers(processed_df, options)
                preprocessing_steps.append(outlier_info)
            
            # 5. 特征缩放
            if options.get('scale_features', False):
                processed_df, scale_info = self._scale_features(processed_df, options)
                preprocessing_steps.append(scale_info)
            
            # 6. 特征编码
            if options.get('encode_features', False):
                processed_df, encode_info = self._encode_features(processed_df, options)
                preprocessing_steps.append(encode_info)
            
            # 7. 特征选择
            if options.get('select_features', False):
                processed_df, select_info = self._select_features(processed_df, options)
                preprocessing_steps.append(select_info)
            
            # 转换为字典格式
            processed_data = processed_df.to_dict('records')
            
            # 记录处理后数据信息
            processed_info = {
                "shape": processed_df.shape,
                "columns": processed_df.columns.tolist(),
                "dtypes": processed_df.dtypes.to_dict(),
                "missing_values": processed_df.isnull().sum().to_dict(),
                "duplicate_rows": processed_df.duplicated().sum()
            }
            
            return {
                "dataId": data_id,
                "processedData": processed_data,
                "originalInfo": original_info,
                "processedInfo": processed_info,
                "preprocessingSteps": preprocessing_steps,
                "summary": {
                    "original_rows": original_info["shape"][0],
                    "processed_rows": processed_info["shape"][0],
                    "original_columns": original_info["shape"][1],
                    "processed_columns": processed_info["shape"][1],
                    "steps_applied": len(preprocessing_steps)
                }
            }
            
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame, options: Dict) -> tuple:
        """处理缺失值"""
        original_missing = df.isnull().sum().sum()
        
        strategy = options.get('missing_strategy', 'mean')
        
        if strategy == 'drop':
            # 删除包含缺失值的行
            df_cleaned = df.dropna()
        elif strategy == 'fill_mean':
            # 用均值填充数值列
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df_cleaned = df.copy()
            for col in numeric_columns:
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
        elif strategy == 'fill_median':
            # 用中位数填充数值列
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df_cleaned = df.copy()
            for col in numeric_columns:
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        elif strategy == 'fill_mode':
            # 用众数填充
            df_cleaned = df.copy()
            for col in df.columns:
                if df[col].dtype == 'object':
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df_cleaned[col].fillna(mode_value[0], inplace=True)
        elif strategy == 'knn':
            # 使用KNN填充
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                df_cleaned = df.copy()
                df_cleaned[numeric_columns] = self.knn_imputer.fit_transform(df_cleaned[numeric_columns])
            else:
                df_cleaned = df.copy()
        else:
            df_cleaned = df.copy()
        
        final_missing = df_cleaned.isnull().sum().sum()
        
        return df_cleaned, {
            "step": "缺失值处理",
            "strategy": strategy,
            "original_missing": int(original_missing),
            "final_missing": int(final_missing),
            "removed_missing": int(original_missing - final_missing)
        }
    
    def _remove_duplicates(self, df: pd.DataFrame) -> tuple:
        """移除重复值"""
        original_rows = len(df)
        df_cleaned = df.drop_duplicates()
        final_rows = len(df_cleaned)
        
        return df_cleaned, {
            "step": "重复值处理",
            "original_rows": original_rows,
            "final_rows": final_rows,
            "removed_duplicates": original_rows - final_rows
        }
    
    def _convert_data_types(self, df: pd.DataFrame, options: Dict) -> tuple:
        """转换数据类型"""
        df_cleaned = df.copy()
        conversions = []
        
        # 自动检测并转换数值列
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                # 尝试转换为数值类型
                try:
                    numeric_series = pd.to_numeric(df_cleaned[col], errors='coerce')
                    if not numeric_series.isna().all():
                        df_cleaned[col] = numeric_series
                        conversions.append(f"{col}: object -> numeric")
                except:
                    pass
        
        # 转换日期列
        date_columns = options.get('date_columns', [])
        for col in date_columns:
            if col in df_cleaned.columns:
                try:
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                    conversions.append(f"{col}: string -> datetime")
                except:
                    pass
        
        return df_cleaned, {
            "step": "数据类型转换",
            "conversions": conversions,
            "total_conversions": len(conversions)
        }
    
    def _handle_outliers(self, df: pd.DataFrame, options: Dict) -> tuple:
        """处理异常值"""
        df_cleaned = df.copy()
        outlier_info = []
        
        method = options.get('outlier_method', 'iqr')
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if method == 'iqr':
                # IQR方法
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
                outlier_count = len(outliers)
                
                if options.get('outlier_action', 'remove') == 'remove':
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                elif options.get('outlier_action', 'remove') == 'cap':
                    df_cleaned[col] = df_cleaned[col].clip(lower_bound, upper_bound)
                
                outlier_info.append({
                    "column": col,
                    "method": "IQR",
                    "outliers_found": outlier_count,
                    "bounds": [float(lower_bound), float(upper_bound)]
                })
            
            elif method == 'zscore':
                # Z-score方法
                z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
                threshold = options.get('zscore_threshold', 3)
                outliers = df_cleaned[z_scores > threshold]
                outlier_count = len(outliers)
                
                if options.get('outlier_action', 'remove') == 'remove':
                    df_cleaned = df_cleaned[z_scores <= threshold]
                elif options.get('outlier_action', 'remove') == 'cap':
                    mean_val = df_cleaned[col].mean()
                    std_val = df_cleaned[col].std()
                    df_cleaned[col] = df_cleaned[col].clip(
                        mean_val - threshold * std_val,
                        mean_val + threshold * std_val
                    )
                
                outlier_info.append({
                    "column": col,
                    "method": "Z-score",
                    "outliers_found": outlier_count,
                    "threshold": threshold
                })
        
        return df_cleaned, {
            "step": "异常值处理",
            "method": method,
            "outlier_details": outlier_info,
            "total_outliers": sum(info["outliers_found"] for info in outlier_info)
        }
    
    def _scale_features(self, df: pd.DataFrame, options: Dict) -> tuple:
        """特征缩放"""
        df_cleaned = df.copy()
        scaled_columns = []
        
        method = options.get('scaling_method', 'standard')
        columns_to_scale = options.get('scale_columns', df_cleaned.select_dtypes(include=[np.number]).columns.tolist())
        
        if isinstance(columns_to_scale, str):
            columns_to_scale = [columns_to_scale]
        
        for col in columns_to_scale:
            if col in df_cleaned.columns and df_cleaned[col].dtype in ['int64', 'float64']:
                if method == 'standard':
                    df_cleaned[col] = self.scaler.fit_transform(df_cleaned[[col]])
                elif method == 'minmax':
                    df_cleaned[col] = self.min_max_scaler.fit_transform(df_cleaned[[col]])
                elif method == 'robust':
                    median = df_cleaned[col].median()
                    mad = np.median(np.abs(df_cleaned[col] - median))
                    df_cleaned[col] = (df_cleaned[col] - median) / mad
                
                scaled_columns.append(col)
        
        return df_cleaned, {
            "step": "特征缩放",
            "method": method,
            "scaled_columns": scaled_columns,
            "total_scaled": len(scaled_columns)
        }
    
    def _encode_features(self, df: pd.DataFrame, options: Dict) -> tuple:
        """特征编码"""
        df_cleaned = df.copy()
        encoded_columns = []
        
        method = options.get('encoding_method', 'label')
        columns_to_encode = options.get('encode_columns', df_cleaned.select_dtypes(include=['object']).columns.tolist())
        
        if isinstance(columns_to_encode, str):
            columns_to_encode = [columns_to_encode]
        
        for col in columns_to_encode:
            if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
                if method == 'label':
                    df_cleaned[col] = self.label_encoder.fit_transform(df_cleaned[col].astype(str))
                elif method == 'onehot':
                    # 对于类别较少的列使用独热编码
                    unique_values = df_cleaned[col].nunique()
                    if unique_values <= 10:  # 限制独热编码的类别数
                        dummies = pd.get_dummies(df_cleaned[col], prefix=col)
                        df_cleaned = pd.concat([df_cleaned.drop(col, axis=1), dummies], axis=1)
                        encoded_columns.extend(dummies.columns.tolist())
                        continue
                
                encoded_columns.append(col)
        
        return df_cleaned, {
            "step": "特征编码",
            "method": method,
            "encoded_columns": encoded_columns,
            "total_encoded": len(encoded_columns)
        }
    
    def _select_features(self, df: pd.DataFrame, options: Dict) -> tuple:
        """特征选择"""
        df_cleaned = df.copy()
        
        method = options.get('selection_method', 'variance')
        max_features = options.get('max_features', 10)
        
        if method == 'variance':
            # 基于方差的特征选择
            numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
            variances = df_cleaned[numeric_columns].var()
            selected_features = variances.nlargest(max_features).index.tolist()
            
            # 保留非数值列
            non_numeric_columns = df_cleaned.select_dtypes(exclude=[np.number]).columns.tolist()
            selected_features.extend(non_numeric_columns)
            
            df_cleaned = df_cleaned[selected_features]
        
        elif method == 'correlation':
            # 基于相关性的特征选择
            numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 1:
                corr_matrix = df_cleaned[numeric_columns].corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
                selected_features = [col for col in numeric_columns if col not in to_drop]
                
                # 保留非数值列
                non_numeric_columns = df_cleaned.select_dtypes(exclude=[np.number]).columns.tolist()
                selected_features.extend(non_numeric_columns)
                
                df_cleaned = df_cleaned[selected_features]
        
        return df_cleaned, {
            "step": "特征选择",
            "method": method,
            "selected_features": df_cleaned.columns.tolist(),
            "total_selected": len(df_cleaned.columns)
        }
