package com.datavisualization.controller;

import com.datavisualization.model.DataUploadRequest;
import com.datavisualization.model.AnalysisRequest;
import com.datavisualization.model.AnalysisResponse;
import com.datavisualization.service.DataService;
import com.datavisualization.service.AnalysisService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.validation.Valid;
import java.util.List;
import java.util.Map;

/**
 * 数据控制器
 * 处理数据上传、分析和可视化相关的API请求
 */
@RestController
@RequestMapping("/api/data")
@CrossOrigin(origins = "*")
public class DataController {

    @Autowired
    private DataService dataService;

    @Autowired
    private AnalysisService analysisService;

    /**
     * 上传数据文件
     * 
     * @param file 数据文件
     * @param request 上传请求参数
     * @return 上传结果
     */
    @PostMapping("/upload")
    public ResponseEntity<?> uploadData(
            @RequestParam("file") MultipartFile file,
            @Valid @ModelAttribute DataUploadRequest request) {
        try {
            String result = dataService.uploadData(file, request);
            return ResponseEntity.ok(Map.of("message", "数据上传成功", "dataId", result));
        } catch (Exception e) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "数据上传失败: " + e.getMessage()));
        }
    }

    /**
     * 获取数据列表
     * 
     * @return 数据列表
     */
    @GetMapping("/list")
    public ResponseEntity<?> getDataList() {
        try {
            List<Map<String, Object>> dataList = dataService.getDataList();
            return ResponseEntity.ok(Map.of("data", dataList));
        } catch (Exception e) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "获取数据列表失败: " + e.getMessage()));
        }
    }

    /**
     * 执行数据分析
     * 
     * @param request 分析请求
     * @return 分析结果
     */
    @PostMapping("/analyze")
    public ResponseEntity<?> analyzeData(@Valid @RequestBody AnalysisRequest request) {
        try {
            AnalysisResponse response = analysisService.analyzeData(request);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "数据分析失败: " + e.getMessage()));
        }
    }

    /**
     * 获取可视化图表数据
     * 
     * @param dataId 数据ID
     * @param chartType 图表类型
     * @param parameters 图表参数
     * @return 图表数据
     */
    @GetMapping("/visualize/{dataId}")
    public ResponseEntity<?> getVisualizationData(
            @PathVariable String dataId,
            @RequestParam String chartType,
            @RequestParam(required = false) Map<String, String> parameters) {
        try {
            Map<String, Object> chartData = dataService.getVisualizationData(dataId, chartType, parameters);
            return ResponseEntity.ok(chartData);
        } catch (Exception e) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "获取图表数据失败: " + e.getMessage()));
        }
    }

    /**
     * 获取数据统计信息
     * 
     * @param dataId 数据ID
     * @return 统计信息
     */
    @GetMapping("/stats/{dataId}")
    public ResponseEntity<?> getDataStats(@PathVariable String dataId) {
        try {
            Map<String, Object> stats = dataService.getDataStats(dataId);
            return ResponseEntity.ok(stats);
        } catch (Exception e) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "获取统计信息失败: " + e.getMessage()));
        }
    }

    /**
     * 删除数据
     * 
     * @param dataId 数据ID
     * @return 删除结果
     */
    @DeleteMapping("/{dataId}")
    public ResponseEntity<?> deleteData(@PathVariable String dataId) {
        try {
            dataService.deleteData(dataId);
            return ResponseEntity.ok(Map.of("message", "数据删除成功"));
        } catch (Exception e) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "数据删除失败: " + e.getMessage()));
        }
    }
}
