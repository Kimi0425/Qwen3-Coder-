package com.datavisualization.model;

import java.util.List;
import java.util.Map;

/**
 * 数据分析响应模型
 */
public class AnalysisResponse {

    private String analysisId;
    private String dataId;
    private String analysisType;
    private String status;
    private String message;
    private Map<String, Object> results;
    private List<Map<String, Object>> visualizations;
    private Map<String, Object> predictions;
    private long processingTime;
    private String timestamp;

    // 构造函数
    public AnalysisResponse() {}

    public AnalysisResponse(String analysisId, String dataId, String analysisType) {
        this.analysisId = analysisId;
        this.dataId = dataId;
        this.analysisType = analysisType;
        this.status = "success";
        this.timestamp = java.time.Instant.now().toString();
    }

    // Getter和Setter方法
    public String getAnalysisId() {
        return analysisId;
    }

    public void setAnalysisId(String analysisId) {
        this.analysisId = analysisId;
    }

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

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public Map<String, Object> getResults() {
        return results;
    }

    public void setResults(Map<String, Object> results) {
        this.results = results;
    }

    public List<Map<String, Object>> getVisualizations() {
        return visualizations;
    }

    public void setVisualizations(List<Map<String, Object>> visualizations) {
        this.visualizations = visualizations;
    }

    public Map<String, Object> getPredictions() {
        return predictions;
    }

    public void setPredictions(Map<String, Object> predictions) {
        this.predictions = predictions;
    }

    public long getProcessingTime() {
        return processingTime;
    }

    public void setProcessingTime(long processingTime) {
        this.processingTime = processingTime;
    }

    public String getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(String timestamp) {
        this.timestamp = timestamp;
    }

    @Override
    public String toString() {
        return "AnalysisResponse{" +
                "analysisId='" + analysisId + '\'' +
                ", dataId='" + dataId + '\'' +
                ", analysisType='" + analysisType + '\'' +
                ", status='" + status + '\'' +
                ", message='" + message + '\'' +
                ", results=" + results +
                ", visualizations=" + visualizations +
                ", predictions=" + predictions +
                ", processingTime=" + processingTime +
                ", timestamp='" + timestamp + '\'' +
                '}';
    }
}
