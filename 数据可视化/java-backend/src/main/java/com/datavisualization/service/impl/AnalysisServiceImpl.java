package com.datavisualization.service.impl;

import com.datavisualization.model.AnalysisRequest;
import com.datavisualization.model.AnalysisResponse;
import com.datavisualization.service.AnalysisService;
import com.datavisualization.service.DataService;
import com.datavisualization.service.PythonApiService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 数据分析服务实现类
 */
@Service
public class AnalysisServiceImpl implements AnalysisService {

    @Autowired
    private DataService dataService;

    @Autowired
    private PythonApiService pythonApiService;

    private final Map<String, AnalysisResponse> analysisCache = new ConcurrentHashMap<>();

    @Override
    public AnalysisResponse analyzeData(AnalysisRequest request) throws Exception {
        long startTime = System.currentTimeMillis();
        
        // 生成分析ID
        String analysisId = UUID.randomUUID().toString();
        
        // 创建分析响应对象
        AnalysisResponse response = new AnalysisResponse(analysisId, request.getDataId(), request.getAnalysisType());
        
        try {
            // 获取数据详情
            Map<String, Object> dataInfo = dataService.getDataDetail(request.getDataId());
            
            // 构建Python API请求
            Map<String, Object> apiRequest = new HashMap<>();
            apiRequest.put("analysisId", analysisId);
            apiRequest.put("dataId", request.getDataId());
            apiRequest.put("analysisType", request.getAnalysisType());
            apiRequest.put("modelType", request.getModelType());
            apiRequest.put("enablePrediction", request.isEnablePrediction());
            apiRequest.put("predictionSteps", request.getPredictionSteps());
            apiRequest.put("parameters", request.getParameters());
            apiRequest.put("data", dataInfo.get("data"));
            apiRequest.put("headers", dataInfo.get("headers"));
            
            // 调用Python API进行分析
            Map<String, Object> analysisResult = pythonApiService.analyzeData(apiRequest);
            
            // 设置分析结果
            response.setResults((Map<String, Object>) analysisResult.get("results"));
            response.setVisualizations((List<Map<String, Object>>) analysisResult.get("visualizations"));
            
            if (request.isEnablePrediction()) {
                response.setPredictions((Map<String, Object>) analysisResult.get("predictions"));
            }
            
            response.setMessage("数据分析完成");
            response.setStatus("success");
            
        } catch (Exception e) {
            response.setStatus("error");
            response.setMessage("数据分析失败: " + e.getMessage());
            throw e;
        } finally {
            // 计算处理时间
            long processingTime = System.currentTimeMillis() - startTime;
            response.setProcessingTime(processingTime);
            
            // 缓存分析结果
            analysisCache.put(analysisId, response);
        }
        
        return response;
    }

    @Override
    public List<AnalysisResponse> getAnalysisHistory(String dataId) throws Exception {
        return analysisCache.values().stream()
                .filter(response -> dataId.equals(response.getDataId()))
                .sorted((a, b) -> b.getTimestamp().compareTo(a.getTimestamp()))
                .collect(java.util.stream.Collectors.toList());
    }

    @Override
    public AnalysisResponse getAnalysisDetail(String analysisId) throws Exception {
        AnalysisResponse response = analysisCache.get(analysisId);
        if (response == null) {
            throw new Exception("分析结果不存在: " + analysisId);
        }
        return response;
    }
}
