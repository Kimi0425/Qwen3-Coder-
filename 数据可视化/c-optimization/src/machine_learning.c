#include "data_optimization.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// 线性回归预测
PredictionResult* linear_regression_predict(const double *x, const double *y, size_t length, size_t prediction_steps) {
    if (!x || !y || length == 0 || prediction_steps == 0) {
        return NULL;
    }
    
    PredictionResult *result = (PredictionResult*)malloc(sizeof(PredictionResult));
    if (!result) {
        return NULL;
    }
    
    result->length = prediction_steps;
    result->predictions = (double*)aligned_malloc(prediction_steps * sizeof(double), 32);
    result->confidence_intervals = (double*)aligned_malloc(prediction_steps * sizeof(double), 32);
    
    if (!result->predictions || !result->confidence_intervals) {
        prediction_result_free(result);
        return NULL;
    }
    
    // 计算回归系数
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
    
    #pragma omp parallel for reduction(+:sum_x,sum_y,sum_xy,sum_x2)
    for (size_t i = 0; i < length; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }
    
    double n = (double)length;
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    double intercept = (sum_y - slope * sum_x) / n;
    
    // 计算R²和MSE
    double y_mean = sum_y / n;
    double ss_tot = 0.0, ss_res = 0.0;
    
    #pragma omp parallel for reduction(+:ss_tot,ss_res)
    for (size_t i = 0; i < length; i++) {
        double y_pred = slope * x[i] + intercept;
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
        ss_res += (y[i] - y_pred) * (y[i] - y_pred);
    }
    
    result->r2_score = 1.0 - (ss_res / ss_tot);
    result->mse = ss_res / n;
    
    // 计算标准误差
    double std_error = sqrt(result->mse);
    
    // 生成预测
    for (size_t i = 0; i < prediction_steps; i++) {
        double x_new = x[length - 1] + (i + 1);
        result->predictions[i] = slope * x_new + intercept;
        result->confidence_intervals[i] = 1.96 * std_error; // 95%置信区间
    }
    
    return result;
}

// 多项式回归预测
PredictionResult* polynomial_regression_predict(const double *x, const double *y, size_t length, int degree, size_t prediction_steps) {
    if (!x || !y || length == 0 || prediction_steps == 0 || degree < 1) {
        return NULL;
    }
    
    PredictionResult *result = (PredictionResult*)malloc(sizeof(PredictionResult));
    if (!result) {
        return NULL;
    }
    
    result->length = prediction_steps;
    result->predictions = (double*)aligned_malloc(prediction_steps * sizeof(double), 32);
    result->confidence_intervals = (double*)aligned_malloc(prediction_steps * sizeof(double), 32);
    
    if (!result->predictions || !result->confidence_intervals) {
        prediction_result_free(result);
        return NULL;
    }
    
    // 构建范德蒙德矩阵
    Matrix *X = matrix_create(length, degree + 1);
    if (!X) {
        prediction_result_free(result);
        return NULL;
    }
    
    for (size_t i = 0; i < length; i++) {
        for (int j = 0; j <= degree; j++) {
            matrix_set(X, i, j, pow(x[i], j));
        }
    }
    
    // 构建目标向量
    Vector *y_vec = vector_create(length);
    if (!y_vec) {
        matrix_free(X);
        prediction_result_free(result);
        return NULL;
    }
    
    for (size_t i = 0; i < length; i++) {
        vector_set(y_vec, i, y[i]);
    }
    
    // 使用最小二乘法求解系数
    Vector *coefficients = least_squares(X, y_vec);
    if (!coefficients) {
        matrix_free(X);
        vector_free(y_vec);
        prediction_result_free(result);
        return NULL;
    }
    
    // 计算R²和MSE
    double y_mean = calculate_mean(y, length);
    double ss_tot = 0.0, ss_res = 0.0;
    
    for (size_t i = 0; i < length; i++) {
        double y_pred = 0.0;
        for (int j = 0; j <= degree; j++) {
            y_pred += vector_get(coefficients, j) * pow(x[i], j);
        }
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
        ss_res += (y[i] - y_pred) * (y[i] - y_pred);
    }
    
    result->r2_score = 1.0 - (ss_res / ss_tot);
    result->mse = ss_res / length;
    
    // 计算标准误差
    double std_error = sqrt(result->mse);
    
    // 生成预测
    for (size_t i = 0; i < prediction_steps; i++) {
        double x_new = x[length - 1] + (i + 1);
        double y_pred = 0.0;
        for (int j = 0; j <= degree; j++) {
            y_pred += vector_get(coefficients, j) * pow(x_new, j);
        }
        result->predictions[i] = y_pred;
        result->confidence_intervals[i] = 1.96 * std_error;
    }
    
    // 清理资源
    matrix_free(X);
    vector_free(y_vec);
    vector_free(coefficients);
    
    return result;
}

// 指数平滑预测
PredictionResult* exponential_smoothing_predict(const double *data, size_t length, double alpha, size_t prediction_steps) {
    if (!data || length == 0 || prediction_steps == 0 || alpha < 0.0 || alpha > 1.0) {
        return NULL;
    }
    
    PredictionResult *result = (PredictionResult*)malloc(sizeof(PredictionResult));
    if (!result) {
        return NULL;
    }
    
    result->length = prediction_steps;
    result->predictions = (double*)aligned_malloc(prediction_steps * sizeof(double), 32);
    result->confidence_intervals = (double*)aligned_malloc(prediction_steps * sizeof(double), 32);
    
    if (!result->predictions || !result->confidence_intervals) {
        prediction_result_free(result);
        return NULL;
    }
    
    // 计算指数平滑值
    double *smoothed = (double*)malloc(length * sizeof(double));
    if (!smoothed) {
        prediction_result_free(result);
        return NULL;
    }
    
    smoothed[0] = data[0];
    for (size_t i = 1; i < length; i++) {
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1];
    }
    
    // 计算趋势
    double trend = 0.0;
    if (length > 1) {
        trend = smoothed[length - 1] - smoothed[length - 2];
    }
    
    // 计算MSE
    double mse_sum = 0.0;
    for (size_t i = 1; i < length; i++) {
        double error = data[i] - smoothed[i - 1];
        mse_sum += error * error;
    }
    result->mse = mse_sum / (length - 1);
    result->r2_score = 0.0; // 指数平滑不计算R²
    
    // 生成预测
    double last_smoothed = smoothed[length - 1];
    for (size_t i = 0; i < prediction_steps; i++) {
        result->predictions[i] = last_smoothed + trend * (i + 1);
        result->confidence_intervals[i] = 1.96 * sqrt(result->mse);
    }
    
    free(smoothed);
    return result;
}

// 释放预测结果
void prediction_result_free(PredictionResult *result) {
    if (result) {
        if (result->predictions) {
            aligned_free(result->predictions);
        }
        if (result->confidence_intervals) {
            aligned_free(result->confidence_intervals);
        }
        free(result);
    }
}

// 梯度下降算法
Vector* gradient_descent(const Matrix *X, const Vector *y, double learning_rate, size_t iterations) {
    if (!X || !y || !X->data || !y->data) {
        return NULL;
    }
    
    if (X->rows != y->length) {
        return NULL;
    }
    
    Vector *theta = vector_create(X->cols);
    if (!theta) {
        return NULL;
    }
    
    // 初始化参数
    for (size_t i = 0; i < X->cols; i++) {
        vector_set(theta, i, 0.0);
    }
    
    // 梯度下降迭代
    for (size_t iter = 0; iter < iterations; iter++) {
        Vector *predictions = vector_create(X->rows);
        if (!predictions) {
            vector_free(theta);
            return NULL;
        }
        
        // 计算预测值
        for (size_t i = 0; i < X->rows; i++) {
            double pred = 0.0;
            for (size_t j = 0; j < X->cols; j++) {
                pred += matrix_get(X, i, j) * vector_get(theta, j);
            }
            vector_set(predictions, i, pred);
        }
        
        // 计算梯度
        for (size_t j = 0; j < X->cols; j++) {
            double gradient = 0.0;
            for (size_t i = 0; i < X->rows; i++) {
                double error = vector_get(predictions, i) - vector_get(y, i);
                gradient += error * matrix_get(X, i, j);
            }
            gradient /= X->rows;
            
            // 更新参数
            double new_theta = vector_get(theta, j) - learning_rate * gradient;
            vector_set(theta, j, new_theta);
        }
        
        vector_free(predictions);
    }
    
    return theta;
}

// 最小二乘法
Vector* least_squares(const Matrix *X, const Vector *y) {
    if (!X || !y || !X->data || !y->data) {
        return NULL;
    }
    
    if (X->rows != y->length) {
        return NULL;
    }
    
    // 计算 X^T * X
    Matrix *X_transposed = matrix_transpose(X);
    if (!X_transposed) {
        return NULL;
    }
    
    Matrix *XTX = matrix_multiply(X_transposed, X);
    if (!XTX) {
        matrix_free(X_transposed);
        return NULL;
    }
    
    // 计算 (X^T * X)^(-1)
    Matrix *XTX_inv = matrix_inverse(XTX);
    if (!XTX_inv) {
        matrix_free(X_transposed);
        matrix_free(XTX);
        return NULL;
    }
    
    // 计算 X^T * y
    Vector *XTy = vector_create(X->cols);
    if (!XTy) {
        matrix_free(X_transposed);
        matrix_free(XTX);
        matrix_free(XTX_inv);
        return NULL;
    }
    
    for (size_t i = 0; i < X->cols; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < X->rows; j++) {
            sum += matrix_get(X_transposed, i, j) * vector_get(y, j);
        }
        vector_set(XTy, i, sum);
    }
    
    // 计算最终结果: (X^T * X)^(-1) * X^T * y
    Vector *result = vector_create(X->cols);
    if (!result) {
        matrix_free(X_transposed);
        matrix_free(XTX);
        matrix_free(XTX_inv);
        vector_free(XTy);
        return NULL;
    }
    
    for (size_t i = 0; i < X->cols; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < X->cols; j++) {
            sum += matrix_get(XTX_inv, i, j) * vector_get(XTy, j);
        }
        vector_set(result, i, sum);
    }
    
    // 清理资源
    matrix_free(X_transposed);
    matrix_free(XTX);
    matrix_free(XTX_inv);
    vector_free(XTy);
    
    return result;
}

// 岭回归
Vector* ridge_regression(const Matrix *X, const Vector *y, double lambda) {
    if (!X || !y || !X->data || !y->data) {
        return NULL;
    }
    
    if (X->rows != y->length) {
        return NULL;
    }
    
    // 计算 X^T * X + λI
    Matrix *X_transposed = matrix_transpose(X);
    if (!X_transposed) {
        return NULL;
    }
    
    Matrix *XTX = matrix_multiply(X_transposed, X);
    if (!XTX) {
        matrix_free(X_transposed);
        return NULL;
    }
    
    // 添加正则化项
    for (size_t i = 0; i < X->cols; i++) {
        double current = matrix_get(XTX, i, i);
        matrix_set(XTX, i, i, current + lambda);
    }
    
    // 计算 (X^T * X + λI)^(-1)
    Matrix *XTX_inv = matrix_inverse(XTX);
    if (!XTX_inv) {
        matrix_free(X_transposed);
        matrix_free(XTX);
        return NULL;
    }
    
    // 计算 X^T * y
    Vector *XTy = vector_create(X->cols);
    if (!XTy) {
        matrix_free(X_transposed);
        matrix_free(XTX);
        matrix_free(XTX_inv);
        return NULL;
    }
    
    for (size_t i = 0; i < X->cols; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < X->rows; j++) {
            sum += matrix_get(X_transposed, i, j) * vector_get(y, j);
        }
        vector_set(XTy, i, sum);
    }
    
    // 计算最终结果
    Vector *result = vector_create(X->cols);
    if (!result) {
        matrix_free(X_transposed);
        matrix_free(XTX);
        matrix_free(XTX_inv);
        vector_free(XTy);
        return NULL;
    }
    
    for (size_t i = 0; i < X->cols; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < X->cols; j++) {
            sum += matrix_get(XTX_inv, i, j) * vector_get(XTy, j);
        }
        vector_set(result, i, sum);
    }
    
    // 清理资源
    matrix_free(X_transposed);
    matrix_free(XTX);
    matrix_free(XTX_inv);
    vector_free(XTy);
    
    return result;
}
