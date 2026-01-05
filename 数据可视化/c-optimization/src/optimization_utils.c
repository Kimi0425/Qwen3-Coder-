#include "data_optimization.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h> // AVX指令集

// 全局变量
static int num_threads = omp_get_max_threads();
static double precision = 1e-10;

// 对齐内存分配
void* aligned_malloc(size_t size, size_t alignment) {
    void *ptr = NULL;
    
    #ifdef _WIN32
        ptr = _aligned_malloc(size, alignment);
    #else
        if (posix_memalign(&ptr, alignment, size) != 0) {
            ptr = NULL;
        }
    #endif
    
    return ptr;
}

// 对齐内存释放
void aligned_free(void *ptr) {
    if (ptr) {
        #ifdef _WIN32
            _aligned_free(ptr);
        #else
            free(ptr);
        #endif
    }
}

// 快速指数函数（使用泰勒级数近似）
double fast_exp(double x) {
    if (x < -700.0) return 0.0;
    if (x > 700.0) return INFINITY;
    
    // 使用exp(x) = exp(x/2)^2的递归关系
    if (x < 0) {
        return 1.0 / fast_exp(-x);
    }
    
    // 将x分解为整数部分和小数部分
    int n = (int)(x * 1.4426950408889634); // 1/ln(2)
    double r = x - n * 0.6931471805599453; // ln(2)
    
    // 计算exp(r)使用泰勒级数
    double result = 1.0;
    double term = 1.0;
    
    for (int i = 1; i < 20; i++) {
        term *= r / i;
        result += term;
    }
    
    // 乘以2^n
    result *= pow(2.0, n);
    
    return result;
}

// 快速对数函数（使用牛顿法）
double fast_log(double x) {
    if (x <= 0.0) return -INFINITY;
    if (x == 1.0) return 0.0;
    
    // 使用log(x) = log(2) * log2(x)
    double log2_x = 0.0;
    double y = x;
    
    // 归一化到[1, 2)区间
    while (y >= 2.0) {
        y /= 2.0;
        log2_x += 1.0;
    }
    while (y < 1.0) {
        y *= 2.0;
        log2_x -= 1.0;
    }
    
    // 使用牛顿法计算log(y)
    double z = y - 1.0;
    double result = z;
    
    for (int i = 0; i < 10; i++) {
        double exp_z = fast_exp(result);
        result = result - (exp_z - y) / exp_z;
    }
    
    return result + log2_x * 0.6931471805599453; // ln(2)
}

// 快速平方根函数（使用牛顿法）
double fast_sqrt(double x) {
    if (x < 0.0) return NAN;
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;
    
    // 使用牛顿法
    double result = x;
    for (int i = 0; i < 10; i++) {
        result = 0.5 * (result + x / result);
    }
    
    return result;
}

// 快速幂函数
double fast_pow(double base, double exponent) {
    if (base == 0.0) return 0.0;
    if (exponent == 0.0) return 1.0;
    if (exponent == 1.0) return base;
    
    // 使用pow(x, y) = exp(y * ln(x))
    return fast_exp(exponent * fast_log(base));
}

// 并行矩阵乘法
void parallel_matrix_multiply(const Matrix *a, const Matrix *b, Matrix *result) {
    if (!a || !b || !result || !a->data || !b->data || !result->data) {
        return;
    }
    
    if (a->cols != b->rows || result->rows != a->rows || result->cols != b->cols) {
        return;
    }
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            double sum = 0.0;
            
            // 使用AVX指令集优化（如果可用）
            #ifdef __AVX__
            if (a->cols >= 4) {
                __m256d sum_vec = _mm256_setzero_pd();
                size_t k = 0;
                
                for (; k < a->cols - 3; k += 4) {
                    __m256d a_vec = _mm256_load_pd(&a->data[i * a->cols + k]);
                    __m256d b_vec = _mm256_load_pd(&b->data[k * b->cols + j]);
                    sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
                }
                
                // 水平求和
                double temp[4];
                _mm256_store_pd(temp, sum_vec);
                sum = temp[0] + temp[1] + temp[2] + temp[3];
                
                // 处理剩余元素
                for (; k < a->cols; k++) {
                    sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
                }
            } else {
                for (size_t k = 0; k < a->cols; k++) {
                    sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
                }
            }
            #else
            for (size_t k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            #endif
            
            result->data[i * b->cols + j] = sum;
        }
    }
}

// 并行向量加法
void parallel_vector_add(const Vector *a, const Vector *b, Vector *result) {
    if (!a || !b || !result || !a->data || !b->data || !result->data) {
        return;
    }
    
    if (a->length != b->length || result->length != a->length) {
        return;
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < a->length; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
}

// 并行统计计算
void parallel_statistics_calculate(const double *data, size_t length, Statistics *stats) {
    if (!data || !stats || length == 0) {
        return;
    }
    
    // 初始化
    stats->count = length;
    stats->sum = 0.0;
    stats->min = data[0];
    stats->max = data[0];
    
    // 并行计算和、最小值、最大值
    #pragma omp parallel for reduction(+:stats->sum) reduction(min:stats->min) reduction(max:stats->max)
    for (size_t i = 0; i < length; i++) {
        stats->sum += data[i];
        if (data[i] < stats->min) stats->min = data[i];
        if (data[i] > stats->max) stats->max = data[i];
    }
    
    // 计算均值
    stats->mean = stats->sum / length;
    
    // 并行计算方差
    double variance_sum = 0.0;
    #pragma omp parallel for reduction(+:variance_sum)
    for (size_t i = 0; i < length; i++) {
        double diff = data[i] - stats->mean;
        variance_sum += diff * diff;
    }
    
    stats->variance = variance_sum / length;
    stats->std_dev = sqrt(stats->variance);
}

// 缓存优化矩阵
void cache_optimize_matrix(Matrix *matrix) {
    if (!matrix || !matrix->data) {
        return;
    }
    
    // 重新排列数据以提高缓存局部性
    double *optimized_data = (double*)aligned_malloc(matrix->capacity * sizeof(double), 64);
    if (!optimized_data) {
        return;
    }
    
    // 按行优先顺序重新排列
    for (size_t i = 0; i < matrix->rows; i++) {
        for (size_t j = 0; j < matrix->cols; j++) {
            optimized_data[i * matrix->cols + j] = matrix->data[i * matrix->cols + j];
        }
    }
    
    aligned_free(matrix->data);
    matrix->data = optimized_data;
}

// 向量化操作
void vectorize_operations(double *data, size_t length) {
    if (!data || length == 0) {
        return;
    }
    
    // 使用AVX指令集进行向量化操作
    #ifdef __AVX__
    size_t i = 0;
    for (; i < length - 3; i += 4) {
        __m256d vec = _mm256_load_pd(&data[i]);
        vec = _mm256_sqrt_pd(vec); // 计算平方根
        _mm256_store_pd(&data[i], vec);
    }
    
    // 处理剩余元素
    for (; i < length; i++) {
        data[i] = sqrt(data[i]);
    }
    #else
    #pragma omp parallel for
    for (size_t i = 0; i < length; i++) {
        data[i] = sqrt(data[i]);
    }
    #endif
}

// 设置线程数
void set_num_threads(int threads) {
    if (threads > 0) {
        num_threads = threads;
        omp_set_num_threads(threads);
    }
}

// 获取线程数
int get_num_threads(void) {
    return num_threads;
}

// 设置精度
void set_precision(double prec) {
    if (prec > 0.0) {
        precision = prec;
    }
}

// 获取精度
double get_precision(void) {
    return precision;
}

// 错误信息
const char* optimization_error_string(OptimizationError error) {
    switch (error) {
        case OPTIMIZATION_SUCCESS:
            return "操作成功";
        case OPTIMIZATION_ERROR_NULL_POINTER:
            return "空指针错误";
        case OPTIMIZATION_ERROR_INVALID_SIZE:
            return "无效大小错误";
        case OPTIMIZATION_ERROR_MEMORY_ALLOCATION:
            return "内存分配错误";
        case OPTIMIZATION_ERROR_MATHEMATICAL:
            return "数学计算错误";
        case OPTIMIZATION_ERROR_CONVERGENCE:
            return "收敛错误";
        default:
            return "未知错误";
    }
}

// 性能监控结构
static PerformanceMetrics performance_metrics = {0};

// 获取性能指标
PerformanceMetrics* get_performance_metrics(void) {
    return &performance_metrics;
}

// 重置性能指标
void reset_performance_metrics(void) {
    performance_metrics.execution_time = 0.0;
    performance_metrics.memory_usage = 0;
    performance_metrics.cache_misses = 0;
    performance_metrics.instructions = 0;
}
