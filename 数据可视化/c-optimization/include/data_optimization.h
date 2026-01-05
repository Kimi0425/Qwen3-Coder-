#ifndef DATA_OPTIMIZATION_H
#define DATA_OPTIMIZATION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 数据结构定义
typedef struct {
    double *data;
    size_t rows;
    size_t cols;
    size_t capacity;
} Matrix;

typedef struct {
    double *data;
    size_t length;
    size_t capacity;
} Vector;

typedef struct {
    double *data;
    size_t length;
    size_t capacity;
} TimeSeries;

// 统计结构
typedef struct {
    double mean;
    double median;
    double std_dev;
    double variance;
    double min;
    double max;
    double sum;
    size_t count;
} Statistics;

// 相关性结构
typedef struct {
    double *correlation_matrix;
    size_t size;
} CorrelationMatrix;

// 预测结果结构
typedef struct {
    double *predictions;
    double *confidence_intervals;
    size_t length;
    double mse;
    double r2_score;
} PredictionResult;

// 矩阵操作函数
Matrix* matrix_create(size_t rows, size_t cols);
void matrix_free(Matrix *matrix);
int matrix_set(Matrix *matrix, size_t row, size_t col, double value);
double matrix_get(const Matrix *matrix, size_t row, size_t col);
Matrix* matrix_multiply(const Matrix *a, const Matrix *b);
Matrix* matrix_transpose(const Matrix *matrix);
double matrix_determinant(const Matrix *matrix);
Matrix* matrix_inverse(const Matrix *matrix);

// 向量操作函数
Vector* vector_create(size_t length);
void vector_free(Vector *vector);
int vector_set(Vector *vector, size_t index, double value);
double vector_get(const Vector *vector, size_t index);
double vector_dot_product(const Vector *a, const Vector *b);
double vector_norm(const Vector *vector);
Vector* vector_add(const Vector *a, const Vector *b);
Vector* vector_subtract(const Vector *a, const Vector *b);
Vector* vector_scale(const Vector *vector, double scalar);

// 时间序列操作函数
TimeSeries* timeseries_create(size_t length);
void timeseries_free(TimeSeries *timeseries);
int timeseries_set(TimeSeries *timeseries, size_t index, double value);
double timeseries_get(const TimeSeries *timeseries, size_t index);
TimeSeries* timeseries_difference(const TimeSeries *timeseries, int lag);
TimeSeries* timeseries_moving_average(const TimeSeries *timeseries, size_t window);
TimeSeries* timeseries_exponential_smoothing(const TimeSeries *timeseries, double alpha);

// 统计计算函数
Statistics* calculate_statistics(const double *data, size_t length);
void statistics_free(Statistics *stats);
double calculate_mean(const double *data, size_t length);
double calculate_median(const double *data, size_t length);
double calculate_std_dev(const double *data, size_t length);
double calculate_variance(const double *data, size_t length);
double calculate_skewness(const double *data, size_t length);
double calculate_kurtosis(const double *data, size_t length);

// 相关性分析函数
CorrelationMatrix* calculate_correlation_matrix(const Matrix *matrix);
void correlation_matrix_free(CorrelationMatrix *corr);
double calculate_pearson_correlation(const double *x, const double *y, size_t length);
double calculate_spearman_correlation(const double *x, const double *y, size_t length);

// 机器学习函数
PredictionResult* linear_regression_predict(const double *x, const double *y, size_t length, size_t prediction_steps);
PredictionResult* polynomial_regression_predict(const double *x, const double *y, size_t length, int degree, size_t prediction_steps);
PredictionResult* exponential_smoothing_predict(const double *data, size_t length, double alpha, size_t prediction_steps);
void prediction_result_free(PredictionResult *result);

// 优化算法函数
Vector* gradient_descent(const Matrix *X, const Vector *y, double learning_rate, size_t iterations);
Vector* least_squares(const Matrix *X, const Vector *y);
Vector* ridge_regression(const Matrix *X, const Vector *y, double lambda);

// 数值计算函数
double fast_exp(double x);
double fast_log(double x);
double fast_sqrt(double x);
double fast_pow(double base, double exponent);

// 并行计算函数
void parallel_matrix_multiply(const Matrix *a, const Matrix *b, Matrix *result);
void parallel_vector_add(const Vector *a, const Vector *b, Vector *result);
void parallel_statistics_calculate(const double *data, size_t length, Statistics *stats);

// 内存优化函数
void* aligned_malloc(size_t size, size_t alignment);
void aligned_free(void *ptr);
void cache_optimize_matrix(Matrix *matrix);
void vectorize_operations(double *data, size_t length);

// 错误处理
typedef enum {
    OPTIMIZATION_SUCCESS = 0,
    OPTIMIZATION_ERROR_NULL_POINTER = -1,
    OPTIMIZATION_ERROR_INVALID_SIZE = -2,
    OPTIMIZATION_ERROR_MEMORY_ALLOCATION = -3,
    OPTIMIZATION_ERROR_MATHEMATICAL = -4,
    OPTIMIZATION_ERROR_CONVERGENCE = -5
} OptimizationError;

const char* optimization_error_string(OptimizationError error);

// 性能监控
typedef struct {
    double execution_time;
    size_t memory_usage;
    size_t cache_misses;
    size_t instructions;
} PerformanceMetrics;

PerformanceMetrics* get_performance_metrics(void);
void reset_performance_metrics(void);

// 配置函数
void set_num_threads(int num_threads);
int get_num_threads(void);
void set_precision(double precision);
double get_precision(void);

#ifdef __cplusplus
}
#endif

#endif // DATA_OPTIMIZATION_H
