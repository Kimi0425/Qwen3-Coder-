#include "data_optimization.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// 向量创建
Vector* vector_create(size_t length) {
    if (length == 0) {
        return NULL;
    }
    
    Vector *vector = (Vector*)malloc(sizeof(Vector));
    if (!vector) {
        return NULL;
    }
    
    vector->length = length;
    vector->capacity = length;
    
    // 使用对齐内存分配
    vector->data = (double*)aligned_malloc(length * sizeof(double), 32);
    if (!vector->data) {
        free(vector);
        return NULL;
    }
    
    // 初始化为零
    memset(vector->data, 0, length * sizeof(double));
    
    return vector;
}

// 向量释放
void vector_free(Vector *vector) {
    if (vector) {
        if (vector->data) {
            aligned_free(vector->data);
        }
        free(vector);
    }
}

// 设置向量元素
int vector_set(Vector *vector, size_t index, double value) {
    if (!vector || !vector->data) {
        return OPTIMIZATION_ERROR_NULL_POINTER;
    }
    
    if (index >= vector->length) {
        return OPTIMIZATION_ERROR_INVALID_SIZE;
    }
    
    vector->data[index] = value;
    return OPTIMIZATION_SUCCESS;
}

// 获取向量元素
double vector_get(const Vector *vector, size_t index) {
    if (!vector || !vector->data) {
        return 0.0;
    }
    
    if (index >= vector->length) {
        return 0.0;
    }
    
    return vector->data[index];
}

// 向量点积
double vector_dot_product(const Vector *a, const Vector *b) {
    if (!a || !b || !a->data || !b->data) {
        return 0.0;
    }
    
    if (a->length != b->length) {
        return 0.0;
    }
    
    double result = 0.0;
    
    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < a->length; i++) {
        result += a->data[i] * b->data[i];
    }
    
    return result;
}

// 向量范数
double vector_norm(const Vector *vector) {
    if (!vector || !vector->data) {
        return 0.0;
    }
    
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < vector->length; i++) {
        sum += vector->data[i] * vector->data[i];
    }
    
    return sqrt(sum);
}

// 向量加法
Vector* vector_add(const Vector *a, const Vector *b) {
    if (!a || !b || !a->data || !b->data) {
        return NULL;
    }
    
    if (a->length != b->length) {
        return NULL;
    }
    
    Vector *result = vector_create(a->length);
    if (!result) {
        return NULL;
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < a->length; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    
    return result;
}

// 向量减法
Vector* vector_subtract(const Vector *a, const Vector *b) {
    if (!a || !b || !a->data || !b->data) {
        return NULL;
    }
    
    if (a->length != b->length) {
        return NULL;
    }
    
    Vector *result = vector_create(a->length);
    if (!result) {
        return NULL;
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < a->length; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    
    return result;
}

// 向量缩放
Vector* vector_scale(const Vector *vector, double scalar) {
    if (!vector || !vector->data) {
        return NULL;
    }
    
    Vector *result = vector_create(vector->length);
    if (!result) {
        return NULL;
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < vector->length; i++) {
        result->data[i] = vector->data[i] * scalar;
    }
    
    return result;
}

// 时间序列创建
TimeSeries* timeseries_create(size_t length) {
    if (length == 0) {
        return NULL;
    }
    
    TimeSeries *timeseries = (TimeSeries*)malloc(sizeof(TimeSeries));
    if (!timeseries) {
        return NULL;
    }
    
    timeseries->length = length;
    timeseries->capacity = length;
    
    // 使用对齐内存分配
    timeseries->data = (double*)aligned_malloc(length * sizeof(double), 32);
    if (!timeseries->data) {
        free(timeseries);
        return NULL;
    }
    
    // 初始化为零
    memset(timeseries->data, 0, length * sizeof(double));
    
    return timeseries;
}

// 时间序列释放
void timeseries_free(TimeSeries *timeseries) {
    if (timeseries) {
        if (timeseries->data) {
            aligned_free(timeseries->data);
        }
        free(timeseries);
    }
}

// 设置时间序列元素
int timeseries_set(TimeSeries *timeseries, size_t index, double value) {
    if (!timeseries || !timeseries->data) {
        return OPTIMIZATION_ERROR_NULL_POINTER;
    }
    
    if (index >= timeseries->length) {
        return OPTIMIZATION_ERROR_INVALID_SIZE;
    }
    
    timeseries->data[index] = value;
    return OPTIMIZATION_SUCCESS;
}

// 获取时间序列元素
double timeseries_get(const TimeSeries *timeseries, size_t index) {
    if (!timeseries || !timeseries->data) {
        return 0.0;
    }
    
    if (index >= timeseries->length) {
        return 0.0;
    }
    
    return timeseries->data[index];
}

// 时间序列差分
TimeSeries* timeseries_difference(const TimeSeries *timeseries, int lag) {
    if (!timeseries || !timeseries->data) {
        return NULL;
    }
    
    if (lag <= 0 || (size_t)lag >= timeseries->length) {
        return NULL;
    }
    
    TimeSeries *result = timeseries_create(timeseries->length - lag);
    if (!result) {
        return NULL;
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < result->length; i++) {
        result->data[i] = timeseries->data[i + lag] - timeseries->data[i];
    }
    
    return result;
}

// 移动平均
TimeSeries* timeseries_moving_average(const TimeSeries *timeseries, size_t window) {
    if (!timeseries || !timeseries->data) {
        return NULL;
    }
    
    if (window == 0 || window > timeseries->length) {
        return NULL;
    }
    
    TimeSeries *result = timeseries_create(timeseries->length - window + 1);
    if (!result) {
        return NULL;
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < result->length; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < window; j++) {
            sum += timeseries->data[i + j];
        }
        result->data[i] = sum / window;
    }
    
    return result;
}

// 指数平滑
TimeSeries* timeseries_exponential_smoothing(const TimeSeries *timeseries, double alpha) {
    if (!timeseries || !timeseries->data) {
        return NULL;
    }
    
    if (alpha < 0.0 || alpha > 1.0) {
        return NULL;
    }
    
    TimeSeries *result = timeseries_create(timeseries->length);
    if (!result) {
        return NULL;
    }
    
    // 第一个值保持不变
    result->data[0] = timeseries->data[0];
    
    // 计算指数平滑值
    for (size_t i = 1; i < timeseries->length; i++) {
        result->data[i] = alpha * timeseries->data[i] + (1 - alpha) * result->data[i - 1];
    }
    
    return result;
}
