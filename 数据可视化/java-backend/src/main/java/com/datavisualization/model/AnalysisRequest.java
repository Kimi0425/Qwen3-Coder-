package com.datavisualization.model;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.NotNull;
import java.util.Map;

/**
 * 数据分析请求模型
 */
public class AnalysisRequest {

    @NotBlank(message = "数据ID不能为空")
    private String dataId;

    @NotNull(message = "分析类型不能为空")
    private String analysisType;

    private Map<String, Object> parameters;
    private String modelType = "qwen3-coder";
    private boolean enablePrediction = false;
    private int predictionSteps = 10;

    // 构造函数
    public AnalysisRequest() {}

    public AnalysisRequest(String dataId, String analysisType) {
        this.dataId = dataId;
        this.analysisType = analysisType;
    }

    // Getter和Setter方法
    public String getDataId() {
        return dataId;
    }

    public void setDataId(String dataId) {
        this.dataId = dataId;
    }

    public String getAnalysisType() {
        return analysisType;
    }

    public void setAnalysisType(String analysisType) {
        this.analysisType = analysisType;
    }

    public Map<String, Object> getParameters() {
        return parameters;
    }

    public void setParameters(Map<String, Object> parameters) {
        this.parameters = parameters;
    }

    public String getModelType() {
        return modelType;
    }

    public void setModelType(String modelType) {
        this.modelType = modelType;
    }

    public boolean isEnablePrediction() {
        return enablePrediction;
    }

    public void setEnablePrediction(boolean enablePrediction) {
        this.enablePrediction = enablePrediction;
    }

    public int getPredictionSteps() {
        return predictionSteps;
    }

    public void setPredictionSteps(int predictionSteps) {
        this.predictionSteps = predictionSteps;
    }

    @Override
    public String toString() {
        return "AnalysisRequest{" +
                "dataId='" + dataId + '\'' +
                ", analysisType='" + analysisType + '\'' +
                ", parameters=" + parameters +
                ", modelType='" + modelType + '\'' +
                ", enablePrediction=" + enablePrediction +
                ", predictionSteps=" + predictionSteps +
                '}';
    }
}
