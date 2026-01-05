#include "data_optimization.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// 计算统计信息
Statistics* calculate_statistics(const double *data, size_t length) {
    if (!data || length == 0) {
        return NULL;
    }
    
    Statistics *stats = (Statistics*)malloc(sizeof(Statistics));
    if (!stats) {
        return NULL;
    }
    
    // 初始化
    stats->count = length;
    stats->sum = 0.0;
    stats->min = data[0];
    stats->max = data[0];
    
    // 计算和、最小值、最大值
    #pragma omp parallel for reduction(+:stats->sum) reduction(min:stats->min) reduction(max:stats->max)
    for (size_t i = 0; i < length; i++) {
        stats->sum += data[i];
        if (data[i] < stats->min) stats->min = data[i];
        if (data[i] > stats->max) stats->max = data[i];
    }
    
    // 计算均值
    stats->mean = stats->sum / length;
    
    // 计算方差和标准差
    double variance_sum = 0.0;
    #pragma omp parallel for reduction(+:variance_sum)
    for (size_t i = 0; i < length; i++) {
        double diff = data[i] - stats->mean;
        variance_sum += diff * diff;
    }
    
    stats->variance = variance_sum / length;
    stats->std_dev = sqrt(stats->variance);
    
    // 计算中位数
    double *sorted_data = (double*)malloc(length * sizeof(double));
    if (sorted_data) {
        memcpy(sorted_data, data, length * sizeof(double));
        
        // 简单的冒泡排序（对于大数据集，应该使用更高效的排序算法）
        for (size_t i = 0; i < length - 1; i++) {
            for (size_t j = 0; j < length - i - 1; j++) {
                if (sorted_data[j] > sorted_data[j + 1]) {
                    double temp = sorted_data[j];
                    sorted_data[j] = sorted_data[j + 1];
                    sorted_data[j + 1] = temp;
                }
            }
        }
        
        if (length % 2 == 0) {
            stats->median = (sorted_data[length / 2 - 1] + sorted_data[length / 2]) / 2.0;
        } else {
            stats->median = sorted_data[length / 2];
        }
        
        free(sorted_data);
    }
    
    return stats;
}

// 释放统计信息
void statistics_free(Statistics *stats) {
    if (stats) {
        free(stats);
    }
}

// 计算均值
double calculate_mean(const double *data, size_t length) {
    if (!data || length == 0) {
        return 0.0;
    }
    
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < length; i++) {
        sum += data[i];
    }
    
    return sum / length;
}

// 计算中位数
double calculate_median(const double *data, size_t length) {
    if (!data || length == 0) {
        return 0.0;
    }
    
    double *sorted_data = (double*)malloc(length * sizeof(double));
    if (!sorted_data) {
        return 0.0;
    }
    
    memcpy(sorted_data, data, length * sizeof(double));
    
    // 使用快速排序
    qsort(sorted_data, length, sizeof(double), compare_doubles);
    
    double median;
    if (length % 2 == 0) {
        median = (sorted_data[length / 2 - 1] + sorted_data[length / 2]) / 2.0;
    } else {
        median = sorted_data[length / 2];
    }
    
    free(sorted_data);
    return median;
}

// 计算标准差
double calculate_std_dev(const double *data, size_t length) {
    if (!data || length == 0) {
        return 0.0;
    }
    
    double mean = calculate_mean(data, length);
    double variance_sum = 0.0;
    
    #pragma omp parallel for reduction(+:variance_sum)
    for (size_t i = 0; i < length; i++) {
        double diff = data[i] - mean;
        variance_sum += diff * diff;
    }
    
    return sqrt(variance_sum / length);
}

// 计算方差
double calculate_variance(const double *data, size_t length) {
    if (!data || length == 0) {
        return 0.0;
    }
    
    double mean = calculate_mean(data, length);
    double variance_sum = 0.0;
    
    #pragma omp parallel for reduction(+:variance_sum)
    for (size_t i = 0; i < length; i++) {
        double diff = data[i] - mean;
        variance_sum += diff * diff;
    }
    
    return variance_sum / length;
}

// 计算偏度
double calculate_skewness(const double *data, size_t length) {
    if (!data || length == 0) {
        return 0.0;
    }
    
    double mean = calculate_mean(data, length);
    double std_dev = calculate_std_dev(data, length);
    
    if (std_dev == 0.0) {
        return 0.0;
    }
    
    double skewness_sum = 0.0;
    #pragma omp parallel for reduction(+:skewness_sum)
    for (size_t i = 0; i < length; i++) {
        double normalized = (data[i] - mean) / std_dev;
        skewness_sum += normalized * normalized * normalized;
    }
    
    return skewness_sum / length;
}

// 计算峰度
double calculate_kurtosis(const double *data, size_t length) {
    if (!data || length == 0) {
        return 0.0;
    }
    
    double mean = calculate_mean(data, length);
    double std_dev = calculate_std_dev(data, length);
    
    if (std_dev == 0.0) {
        return 0.0;
    }
    
    double kurtosis_sum = 0.0;
    #pragma omp parallel for reduction(+:kurtosis_sum)
    for (size_t i = 0; i < length; i++) {
        double normalized = (data[i] - mean) / std_dev;
        double normalized_squared = normalized * normalized;
        kurtosis_sum += normalized_squared * normalized_squared;
    }
    
    return (kurtosis_sum / length) - 3.0; // 减去3得到超额峰度
}

// 计算Pearson相关系数
double calculate_pearson_correlation(const double *x, const double *y, size_t length) {
    if (!x || !y || length == 0) {
        return 0.0;
    }
    
    double mean_x = calculate_mean(x, length);
    double mean_y = calculate_mean(y, length);
    
    double numerator = 0.0;
    double sum_x_squared = 0.0;
    double sum_y_squared = 0.0;
    
    #pragma omp parallel for reduction(+:numerator,sum_x_squared,sum_y_squared)
    for (size_t i = 0; i < length; i++) {
        double diff_x = x[i] - mean_x;
        double diff_y = y[i] - mean_y;
        
        numerator += diff_x * diff_y;
        sum_x_squared += diff_x * diff_x;
        sum_y_squared += diff_y * diff_y;
    }
    
    double denominator = sqrt(sum_x_squared * sum_y_squared);
    if (denominator == 0.0) {
        return 0.0;
    }
    
    return numerator / denominator;
}

// 计算Spearman相关系数
double calculate_spearman_correlation(const double *x, const double *y, size_t length) {
    if (!x || !y || length == 0) {
        return 0.0;
    }
    
    // 创建排名数组
    double *rank_x = (double*)malloc(length * sizeof(double));
    double *rank_y = (double*)malloc(length * sizeof(double));
    
    if (!rank_x || !rank_y) {
        free(rank_x);
        free(rank_y);
        return 0.0;
    }
    
    // 计算排名
    for (size_t i = 0; i < length; i++) {
        rank_x[i] = 1.0;
        rank_y[i] = 1.0;
        
        for (size_t j = 0; j < length; j++) {
            if (x[j] < x[i]) rank_x[i]++;
            if (y[j] < y[i]) rank_y[i]++;
        }
    }
    
    // 使用排名计算Pearson相关系数
    double correlation = calculate_pearson_correlation(rank_x, rank_y, length);
    
    free(rank_x);
    free(rank_y);
    
    return correlation;
}

// 计算相关性矩阵
CorrelationMatrix* calculate_correlation_matrix(const Matrix *matrix) {
    if (!matrix || !matrix->data) {
        return NULL;
    }
    
    size_t n = matrix->cols;
    CorrelationMatrix *corr = (CorrelationMatrix*)malloc(sizeof(CorrelationMatrix));
    if (!corr) {
        return NULL;
    }
    
    corr->size = n;
    corr->correlation_matrix = (double*)aligned_malloc(n * n * sizeof(double), 32);
    if (!corr->correlation_matrix) {
        free(corr);
        return NULL;
    }
    
    // 计算每对列之间的相关性
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            if (i == j) {
                corr->correlation_matrix[i * n + j] = 1.0;
            } else {
                double *col_i = (double*)malloc(matrix->rows * sizeof(double));
                double *col_j = (double*)malloc(matrix->rows * sizeof(double));
                
                if (col_i && col_j) {
                    for (size_t k = 0; k < matrix->rows; k++) {
                        col_i[k] = matrix->data[k * n + i];
                        col_j[k] = matrix->data[k * n + j];
                    }
                    
                    corr->correlation_matrix[i * n + j] = calculate_pearson_correlation(col_i, col_j, matrix->rows);
                }
                
                free(col_i);
                free(col_j);
            }
        }
    }
    
    return corr;
}

// 释放相关性矩阵
void correlation_matrix_free(CorrelationMatrix *corr) {
    if (corr) {
        if (corr->correlation_matrix) {
            aligned_free(corr->correlation_matrix);
        }
        free(corr);
    }
}

// 比较函数（用于排序）
int compare_doubles(const void *a, const void *b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}
