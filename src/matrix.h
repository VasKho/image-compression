#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

typedef struct {
  size_t rows;
  size_t cols;
  double** content;
} matrix;

matrix* matrix_create(size_t rows, size_t cols);
void matrix_destroy(matrix* A);
void matrix_set_element(matrix* A, size_t row, size_t col, double value);
matrix* matrix_multiply(matrix* A, matrix* B);
matrix* matrix_substitute(matrix* A, matrix* B);
matrix* matrix_multiply_scalar(double num, matrix* A);
matrix* matrix_reshape(matrix* A, size_t new_rows, size_t new_cols);
matrix* matrix_transpose(matrix* A);
void matrix_copy(matrix* dst, matrix* src, size_t dst_start_row, size_t dst_start_col, size_t src_start_row, size_t src_start_col, size_t row_num, size_t col_num);

#endif
