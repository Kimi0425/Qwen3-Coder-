package com.datavisualization.service.impl;

import com.datavisualization.service.PythonApiService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

/**
 * Python API服务实现类
 */
@Service
public class PythonApiServiceImpl implements PythonApiService {

    @Value("${python.api.base-url:http://localhost:5000}")
    private String pythonApiBaseUrl;

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    public PythonApiServiceImpl() {
        this.restTemplate = new RestTemplate();
        this.objectMapper = new ObjectMapper();
    }

    @Override
    public Map<String, Object> analyzeData(Map<String, Object> request) throws Exception {
        String url = pythonApiBaseUrl + "/api/analyze";
        return makeApiCall(url, request);
    }

    @Override
    public Map<String, Object> generateVisualization(Map<String, Object> request) throws Exception {
        String url = pythonApiBaseUrl + "/api/visualize";
        return makeApiCall(url, request);
    }

    @Override
    public Map<String, Object> calculateStats(Map<String, Object> request) throws Exception {
        String url = pythonApiBaseUrl + "/api/stats";
        return makeApiCall(url, request);
    }

    @Override
    public Map<String, Object> preprocessData(Map<String, Object> request) throws Exception {
        String url = pythonApiBaseUrl + "/api/preprocess";
        return makeApiCall(url, request);
    }

    @Override
    public Map<String, Object> healthCheck() throws Exception {
        String url = pythonApiBaseUrl + "/api/health";
        return makeApiCall(url, Map.of());
    }

    /**
     * 发起API调用
     */
    private Map<String, Object> makeApiCall(String url, Map<String, Object> request) throws Exception {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.set("Content-Type", "application/json");
            
            String requestBody = objectMapper.writeValueAsString(request);
            HttpEntity<String> entity = new HttpEntity<>(requestBody, headers);
            
            ResponseEntity<String> response = restTemplate.exchange(
                    url, HttpMethod.POST, entity, String.class);
            
            if (response.getStatusCode().is2xxSuccessful()) {
                return objectMapper.readValue(response.getBody(), Map.class);
            } else {
                throw new Exception("Python API调用失败: " + response.getStatusCode());
            }
        } catch (Exception e) {
            throw new Exception("Python API通信异常: " + e.getMessage(), e);
        }
    }
}
