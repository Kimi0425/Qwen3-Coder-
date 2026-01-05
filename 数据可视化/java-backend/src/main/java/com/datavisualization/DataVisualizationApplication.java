package com.datavisualization;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * 数据可视化系统主应用程序
 * 
 * @author Data Visualization Team
 * @version 1.0.0
 */
@SpringBootApplication
@EnableAsync
@EnableScheduling
public class DataVisualizationApplication {

    public static void main(String[] args) {
        SpringApplication.run(DataVisualizationApplication.class, args);
        System.out.println("数据可视化系统启动成功！");
        System.out.println("访问地址: http://localhost:8080");
        System.out.println("API文档: http://localhost:8080/swagger-ui.html");
    }
}
