package com.datavisualization.service;

import com.datavisualization.model.AnalysisRequest;
import com.datavisualization.model.AnalysisResponse;

/**
 * 数据分析服务接口
 * 处理数据分析和趋势预测
 */
public interface AnalysisService {

    /**
     * 执行数据分析
     * 
     * @param request 分析请求
     * @return 分析结果
     * @throws Exception 分析异常
     */
    AnalysisResponse analyzeData(AnalysisRequest request) throws Exception;

    /**
     * 获取分析历史
     * 
     * @param dataId 数据ID
     * @return 分析历史列表
     * @throws Exception 查询异常
     */
    java.util.List<AnalysisResponse> getAnalysisHistory(String dataId) throws Exception;

    /**
     * 获取分析结果详情
     * 
     * @param analysisId 分析ID
     * @return 分析结果详情
     * @throws Exception 查询异常
     */
    AnalysisResponse getAnalysisDetail(String analysisId) throws Exception;
}
