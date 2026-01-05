package com.datavisualization.service;

import java.util.Map;

/**
 * Python API服务接口
 * 与Python数据分析模块通信
 */
public interface PythonApiService {

    /**
     * 分析数据
     * 
     * @param request 分析请求
     * @return 分析结果
     * @throws Exception 分析异常
     */
    Map<String, Object> analyzeData(Map<String, Object> request) throws Exception;

    /**
     * 生成可视化数据
     * 
     * @param request 可视化请求
     * @return 可视化数据
     * @throws Exception 生成异常
     */
    Map<String, Object> generateVisualization(Map<String, Object> request) throws Exception;

    /**
     * 计算统计信息
     * 
     * @param request 统计请求
     * @return 统计结果
     * @throws Exception 计算异常
     */
    Map<String, Object> calculateStats(Map<String, Object> request) throws Exception;

    /**
     * 预处理数据
     * 
     * @param request 预处理请求
     * @return 预处理结果
     * @throws Exception 预处理异常
     */
    Map<String, Object> preprocessData(Map<String, Object> request) throws Exception;

    /**
     * 健康检查
     * 
     * @return 服务状态
     * @throws Exception 检查异常
     */
    Map<String, Object> healthCheck() throws Exception;
}
