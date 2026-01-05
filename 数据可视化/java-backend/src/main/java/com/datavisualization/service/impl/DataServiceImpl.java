package com.datavisualization.service.impl;

import com.datavisualization.model.DataUploadRequest;
import com.datavisualization.service.DataService;
import com.datavisualization.service.PythonApiService;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 数据服务实现类
 */
@Service
public class DataServiceImpl implements DataService {

    @Autowired
    private PythonApiService pythonApiService;

    private final String UPLOAD_DIR = "uploads/";
    private final Map<String, Map<String, Object>> dataCache = new HashMap<>();

    @Override
    public String uploadData(MultipartFile file, DataUploadRequest request) throws Exception {
        // 创建上传目录
        Path uploadPath = Paths.get(UPLOAD_DIR);
        if (!Files.exists(uploadPath)) {
            Files.createDirectories(uploadPath);
        }

        // 生成唯一的数据ID
        String dataId = UUID.randomUUID().toString();
        String fileName = dataId + "_" + file.getOriginalFilename();
        Path filePath = uploadPath.resolve(fileName);

        // 保存文件
        Files.copy(file.getInputStream(), filePath);

        // 解析数据文件
        Map<String, Object> dataInfo = parseDataFile(filePath, request);
        dataInfo.put("dataId", dataId);
        dataInfo.put("fileName", fileName);
        dataInfo.put("originalName", file.getOriginalFilename());
        dataInfo.put("uploadTime", new Date());
        dataInfo.put("request", request);

        // 缓存数据信息
        dataCache.put(dataId, dataInfo);

        return dataId;
    }

    @Override
    public List<Map<String, Object>> getDataList() throws Exception {
        return dataCache.values().stream()
                .map(data -> {
                    Map<String, Object> summary = new HashMap<>();
                    summary.put("dataId", data.get("dataId"));
                    summary.put("name", ((DataUploadRequest) data.get("request")).getName());
                    summary.put("description", ((DataUploadRequest) data.get("request")).getDescription());
                    summary.put("dataType", ((DataUploadRequest) data.get("request")).getDataType());
                    summary.put("uploadTime", data.get("uploadTime"));
                    summary.put("rowCount", data.get("rowCount"));
                    summary.put("columnCount", data.get("columnCount"));
                    return summary;
                })
                .collect(Collectors.toList());
    }

    @Override
    public Map<String, Object> getDataDetail(String dataId) throws Exception {
        Map<String, Object> dataInfo = dataCache.get(dataId);
        if (dataInfo == null) {
            throw new Exception("数据不存在: " + dataId);
        }
        return dataInfo;
    }

    @Override
    public Map<String, Object> getVisualizationData(String dataId, String chartType, Map<String, String> parameters) throws Exception {
        Map<String, Object> dataInfo = dataCache.get(dataId);
        if (dataInfo == null) {
            throw new Exception("数据不存在: " + dataId);
        }

        // 调用Python API生成可视化数据
        Map<String, Object> request = new HashMap<>();
        request.put("dataId", dataId);
        request.put("chartType", chartType);
        request.put("parameters", parameters);
        request.put("data", dataInfo.get("data"));

        return pythonApiService.generateVisualization(request);
    }

    @Override
    public Map<String, Object> getDataStats(String dataId) throws Exception {
        Map<String, Object> dataInfo = dataCache.get(dataId);
        if (dataInfo == null) {
            throw new Exception("数据不存在: " + dataId);
        }

        // 调用Python API计算统计信息
        Map<String, Object> request = new HashMap<>();
        request.put("dataId", dataId);
        request.put("data", dataInfo.get("data"));

        return pythonApiService.calculateStats(request);
    }

    @Override
    public void deleteData(String dataId) throws Exception {
        Map<String, Object> dataInfo = dataCache.get(dataId);
        if (dataInfo == null) {
            throw new Exception("数据不存在: " + dataId);
        }

        // 删除文件
        String fileName = (String) dataInfo.get("fileName");
        Path filePath = Paths.get(UPLOAD_DIR + fileName);
        if (Files.exists(filePath)) {
            Files.delete(filePath);
        }

        // 从缓存中移除
        dataCache.remove(dataId);
    }

    @Override
    public Map<String, Object> preprocessData(String dataId, Map<String, Object> preprocessingOptions) throws Exception {
        Map<String, Object> dataInfo = dataCache.get(dataId);
        if (dataInfo == null) {
            throw new Exception("数据不存在: " + dataId);
        }

        // 调用Python API进行数据预处理
        Map<String, Object> request = new HashMap<>();
        request.put("dataId", dataId);
        request.put("data", dataInfo.get("data"));
        request.put("options", preprocessingOptions);

        Map<String, Object> result = pythonApiService.preprocessData(request);
        
        // 更新缓存中的数据
        dataInfo.put("data", result.get("processedData"));
        dataInfo.put("preprocessingInfo", result.get("preprocessingInfo"));

        return result;
    }

    /**
     * 解析数据文件
     */
    private Map<String, Object> parseDataFile(Path filePath, DataUploadRequest request) throws Exception {
        Map<String, Object> result = new HashMap<>();
        String fileName = filePath.getFileName().toString().toLowerCase();

        if (fileName.endsWith(".csv")) {
            result = parseCsvFile(filePath);
        } else if (fileName.endsWith(".xlsx") || fileName.endsWith(".xls")) {
            result = parseExcelFile(filePath);
        } else {
            throw new Exception("不支持的文件格式: " + fileName);
        }

        return result;
    }

    /**
     * 解析CSV文件
     */
    private Map<String, Object> parseCsvFile(Path filePath) throws Exception {
        Map<String, Object> result = new HashMap<>();
        List<Map<String, Object>> data = new ArrayList<>();

        try (Reader reader = Files.newBufferedReader(filePath);
             CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader())) {

            List<String> headers = csvParser.getHeaderNames();
            result.put("headers", headers);
            result.put("columnCount", headers.size());

            int rowCount = 0;
            for (CSVRecord record : csvParser) {
                Map<String, Object> row = new HashMap<>();
                for (String header : headers) {
                    row.put(header, record.get(header));
                }
                data.add(row);
                rowCount++;
            }
            result.put("rowCount", rowCount);
        }

        result.put("data", data);
        return result;
    }

    /**
     * 解析Excel文件
     */
    private Map<String, Object> parseExcelFile(Path filePath) throws Exception {
        Map<String, Object> result = new HashMap<>();
        List<Map<String, Object>> data = new ArrayList<>();

        try (FileInputStream fis = new FileInputStream(filePath.toFile());
             Workbook workbook = new XSSFWorkbook(fis)) {

            Sheet sheet = workbook.getSheetAt(0);
            Row headerRow = sheet.getRow(0);
            
            List<String> headers = new ArrayList<>();
            for (Cell cell : headerRow) {
                headers.add(getCellValueAsString(cell));
            }
            result.put("headers", headers);
            result.put("columnCount", headers.size());

            int rowCount = 0;
            for (int i = 1; i <= sheet.getLastRowNum(); i++) {
                Row row = sheet.getRow(i);
                if (row != null) {
                    Map<String, Object> rowData = new HashMap<>();
                    for (int j = 0; j < headers.size(); j++) {
                        Cell cell = row.getCell(j);
                        rowData.put(headers.get(j), getCellValueAsString(cell));
                    }
                    data.add(rowData);
                    rowCount++;
                }
            }
            result.put("rowCount", rowCount);
        }

        result.put("data", data);
        return result;
    }

    /**
     * 获取单元格值作为字符串
     */
    private String getCellValueAsString(Cell cell) {
        if (cell == null) {
            return "";
        }
        
        switch (cell.getCellType()) {
            case STRING:
                return cell.getStringCellValue();
            case NUMERIC:
                if (DateUtil.isCellDateFormatted(cell)) {
                    return cell.getDateCellValue().toString();
                } else {
                    return String.valueOf(cell.getNumericCellValue());
                }
            case BOOLEAN:
                return String.valueOf(cell.getBooleanCellValue());
            case FORMULA:
                return cell.getCellFormula();
            default:
                return "";
        }
    }
}
