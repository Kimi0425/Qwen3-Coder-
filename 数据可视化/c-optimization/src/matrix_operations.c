#include "data_optimization.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// 矩阵创建
Matrix* matrix_create(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return NULL;
    }
    
    Matrix *matrix = (Matrix*)malloc(sizeof(Matrix));
    if (!matrix) {
        return NULL;
    }
    
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->capacity = rows * cols;
    
    // 使用对齐内存分配以提高性能
    matrix->data = (double*)aligned_malloc(matrix->capacity * sizeof(double), 32);
    if (!matrix->data) {
        free(matrix);
        return NULL;
    }
    
    // 初始化为零
    memset(matrix->data, 0, matrix->capacity * sizeof(double));
    
    return matrix;
}

// 矩阵释放
void matrix_free(Matrix *matrix) {
    if (matrix) {
        if (matrix->data) {
            aligned_free(matrix->data);
        }
        free(matrix);
    }
}

// 设置矩阵元素
int matrix_set(Matrix *matrix, size_t row, size_t col, double value) {
    if (!matrix || !matrix->data) {
        return OPTIMIZATION_ERROR_NULL_POINTER;
    }
    
    if (row >= matrix->rows || col >= matrix->cols) {
        return OPTIMIZATION_ERROR_INVALID_SIZE;
    }
    
    matrix->data[row * matrix->cols + col] = value;
    return OPTIMIZATION_SUCCESS;
}

// 获取矩阵元素
double matrix_get(const Matrix *matrix, size_t row, size_t col) {
    if (!matrix || !matrix->data) {
        return 0.0;
    }
    
    if (row >= matrix->rows || col >= matrix->cols) {
        return 0.0;
    }
    
    return matrix->data[row * matrix->cols + col];
}

// 矩阵乘法（优化版本）
Matrix* matrix_multiply(const Matrix *a, const Matrix *b) {
    if (!a || !b || !a->data || !b->data) {
        return NULL;
    }
    
    if (a->cols != b->rows) {
        return NULL;
    }
    
    Matrix *result = matrix_create(a->rows, b->cols);
    if (!result) {
        return NULL;
    }
    
    // 使用OpenMP并行化矩阵乘法
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            double sum = 0.0;
            
            // 循环展开优化
            size_t k = 0;
            for (; k < a->cols - 3; k += 4) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
                sum += a->data[i * a->cols + k + 1] * b->data[(k + 1) * b->cols + j];
                sum += a->data[i * a->cols + k + 2] * b->data[(k + 2) * b->cols + j];
                sum += a->data[i * a->cols + k + 3] * b->data[(k + 3) * b->cols + j];
            }
            
            // 处理剩余元素
            for (; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            
            result->data[i * b->cols + j] = sum;
        }
    }
    
    return result;
}

// 矩阵转置
Matrix* matrix_transpose(const Matrix *matrix) {
    if (!matrix || !matrix->data) {
        return NULL;
    }
    
    Matrix *transposed = matrix_create(matrix->cols, matrix->rows);
    if (!transposed) {
        return NULL;
    }
    
    // 使用OpenMP并行化转置
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < matrix->rows; i++) {
        for (size_t j = 0; j < matrix->cols; j++) {
            transposed->data[j * matrix->rows + i] = matrix->data[i * matrix->cols + j];
        }
    }
    
    return transposed;
}

// 计算矩阵行列式（使用LU分解）
double matrix_determinant(const Matrix *matrix) {
    if (!matrix || !matrix->data) {
        return 0.0;
    }
    
    if (matrix->rows != matrix->cols) {
        return 0.0;
    }
    
    size_t n = matrix->rows;
    double det = 1.0;
    
    // 创建副本进行LU分解
    Matrix *lu = matrix_create(n, n);
    if (!lu) {
        return 0.0;
    }
    
    // 复制数据
    memcpy(lu->data, matrix->data, n * n * sizeof(double));
    
    // LU分解
    for (size_t k = 0; k < n - 1; k++) {
        if (fabs(lu->data[k * n + k]) < 1e-10) {
            matrix_free(lu);
            return 0.0;
        }
        
        for (size_t i = k + 1; i < n; i++) {
            double factor = lu->data[i * n + k] / lu->data[k * n + k];
            lu->data[i * n + k] = factor;
            
            for (size_t j = k + 1; j < n; j++) {
                lu->data[i * n + j] -= factor * lu->data[k * n + j];
            }
        }
    }
    
    // 计算行列式
    for (size_t i = 0; i < n; i++) {
        det *= lu->data[i * n + i];
    }
    
    matrix_free(lu);
    return det;
}

// 矩阵求逆（使用LU分解）
Matrix* matrix_inverse(const Matrix *matrix) {
    if (!matrix || !matrix->data) {
        return NULL;
    }
    
    if (matrix->rows != matrix->cols) {
        return NULL;
    }
    
    size_t n = matrix->rows;
    Matrix *inverse = matrix_create(n, n);
    if (!inverse) {
        return NULL;
    }
    
    // 创建单位矩阵
    for (size_t i = 0; i < n; i++) {
        inverse->data[i * n + i] = 1.0;
    }
    
    // 创建副本进行LU分解
    Matrix *lu = matrix_create(n, n);
    if (!lu) {
        matrix_free(inverse);
        return NULL;
    }
    
    memcpy(lu->data, matrix->data, n * n * sizeof(double));
    
    // LU分解
    for (size_t k = 0; k < n - 1; k++) {
        if (fabs(lu->data[k * n + k]) < 1e-10) {
            matrix_free(lu);
            matrix_free(inverse);
            return NULL;
        }
        
        for (size_t i = k + 1; i < n; i++) {
            double factor = lu->data[i * n + k] / lu->data[k * n + k];
            lu->data[i * n + k] = factor;
            
            for (size_t j = k + 1; j < n; j++) {
                lu->data[i * n + j] -= factor * lu->data[k * n + j];
            }
        }
    }
    
    // 前向替换
    for (size_t k = 0; k < n; k++) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < i; j++) {
                inverse->data[i * n + k] -= lu->data[i * n + j] * inverse->data[j * n + k];
            }
        }
    }
    
    // 后向替换
    for (size_t k = 0; k < n; k++) {
        for (int i = n - 1; i >= 0; i--) {
            for (size_t j = i + 1; j < n; j++) {
                inverse->data[i * n + k] -= lu->data[i * n + j] * inverse->data[j * n + k];
            }
            inverse->data[i * n + k] /= lu->data[i * n + i];
        }
    }
    
    matrix_free(lu);
    return inverse;
}
