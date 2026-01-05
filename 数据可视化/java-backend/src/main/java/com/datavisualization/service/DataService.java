package com.datavisualization.service;

import com.datavisualization.model.DataUploadRequest;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Map;

/**
 * 数据服务接口
 * 处理数据上传、存储、查询和可视化数据生成
 */
public interface DataService {

    /**
     * 上传数据文件
     * 
     * @param file 数据文件
     * @param request 上传请求参数
     * @return 数据ID
     * @throws Exception 上传异常
     */
    String uploadData(MultipartFile file, DataUploadRequest request) throws Exception;

    /**
     * 获取数据列表
     * 
     * @return 数据列表
     * @throws Exception 查询异常
     */
    List<Map<String, Object>> getDataList() throws Exception;

    /**
     * 获取数据详情
     * 
     * @param dataId 数据ID
     * @return 数据详情
     * @throws Exception 查询异常
     */
    Map<String, Object> getDataDetail(String dataId) throws Exception;

    /**
     * 获取可视化图表数据
     * 
     * @param dataId 数据ID
     * @param chartType 图表类型
     * @param parameters 图表参数
     * @return 图表数据
     * @throws Exception 生成异常
     */
    Map<String, Object> getVisualizationData(String dataId, String chartType, Map<String, String> parameters) throws Exception;

    /**
     * 获取数据统计信息
     * 
     * @param dataId 数据ID
     * @return 统计信息
     * @throws Exception 计算异常
     */
    Map<String, Object> getDataStats(String dataId) throws Exception;

    /**
     * 删除数据
     * 
     * @param dataId 数据ID
     * @throws Exception 删除异常
     */
    void deleteData(String dataId) throws Exception;

    /**
     * 预处理数据
     * 
     * @param dataId 数据ID
     * @param preprocessingOptions 预处理选项
     * @return 预处理结果
     * @throws Exception 预处理异常
     */
    Map<String, Object> preprocessData(String dataId, Map<String, Object> preprocessingOptions) throws Exception;
}
