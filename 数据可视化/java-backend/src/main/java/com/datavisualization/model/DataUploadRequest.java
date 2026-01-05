package com.datavisualization.model;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.NotNull;

/**
 * 数据上传请求模型
 */
public class DataUploadRequest {

    @NotBlank(message = "数据名称不能为空")
    private String name;

    @NotBlank(message = "数据描述不能为空")
    private String description;

    @NotNull(message = "数据类型不能为空")
    private String dataType;

    private String tags;
    private String category;
    private boolean isPublic = false;

    // 构造函数
    public DataUploadRequest() {}

    public DataUploadRequest(String name, String description, String dataType) {
        this.name = name;
        this.description = description;
        this.dataType = dataType;
    }

    // Getter和Setter方法
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getDataType() {
        return dataType;
    }

    public void setDataType(String dataType) {
        this.dataType = dataType;
    }

    public String getTags() {
        return tags;
    }

    public void setTags(String tags) {
        this.tags = tags;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public boolean isPublic() {
        return isPublic;
    }

    public void setPublic(boolean isPublic) {
        this.isPublic = isPublic;
    }

    @Override
    public String toString() {
        return "DataUploadRequest{" +
                "name='" + name + '\'' +
                ", description='" + description + '\'' +
                ", dataType='" + dataType + '\'' +
                ", tags='" + tags + '\'' +
                ", category='" + category + '\'' +
                ", isPublic=" + isPublic +
                '}';
    }
}
