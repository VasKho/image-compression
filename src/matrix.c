#include "matrix.h"

matrix* matrix_create(size_t rows, size_t cols) {
  matrix* A = calloc(1, sizeof(matrix));
  A->rows = rows;
  A->cols = cols;
  A->content = calloc(A->rows, sizeof(*A->content));
  for (size_t i = 0; i < A->rows; ++i)
    A->content[i] = calloc(A->cols, sizeof(**A->content));
  return A;
}

void matrix_destroy(matrix* A) {
  for (size_t i = 0; i < A->rows; ++i)
    free(A->content[i]);
  free(A->content);
  free(A);
}

void matrix_set_element(matrix* A, size_t row, size_t col, double value) {
  A->content[row][col] = value;
}

matrix* matrix_multiply(matrix* A, matrix* B) {
  matrix* C = matrix_create(A->rows, B->cols);
  for (size_t A_row_ind = 0; A_row_ind < A->rows; ++A_row_ind) {
    for (size_t col_ind = 0; col_ind < B->cols; ++col_ind) {
      for (size_t B_row_ind = 0; B_row_ind < B->rows; ++B_row_ind) {
	C->content[A_row_ind][col_ind] += A->content[A_row_ind][B_row_ind] * B->content[B_row_ind][col_ind];
      }
    }
  }
  return C;
}

matrix* matrix_substitute(matrix* A, matrix* B) {
  if (A->rows != B->rows || A->cols != B->cols) {
    printf("Invalid matrix sizes. Unable to substitute.\n");
    return NULL;
  }
  matrix* result = matrix_create(A->rows, A->cols);
  for (size_t row = 0; row < A->rows; ++row) {
    for (size_t col = 0; col < A->cols; ++col) {
      matrix_set_element(result, row, col, A->content[row][col] - B->content[row][col]);
    }
  }
  return result;
}

matrix* matrix_multiply_scalar(double num, matrix* A) {
  matrix* result = matrix_create(A->rows, A->cols);
  for (size_t row = 0; row < A->rows; ++row) {
    for (size_t col = 0; col < A->cols; ++col) {
      matrix_set_element(result, row, col, A->content[row][col] * num);
    }
  }
  return result;
}

matrix* matrix_reshape(matrix* A, size_t new_rows, size_t new_cols) {
  if (A->rows*A->cols != new_rows*new_cols) {
    printf("Invalid new sizes. Unable to reshape\n");
    return NULL;
  }
  matrix* new_matrix = matrix_create(new_rows, new_cols);
  size_t old_row = 0, old_col = 0;
  for (size_t new_row = 0; new_row < new_rows; ++new_row) {
    for (size_t new_col = 0; new_col < new_cols; ++new_col) {
      if (old_col == A->cols) {
	old_col = 0;
	++old_row;
      }
      matrix_set_element(new_matrix, new_row, new_col, A->content[old_row][old_col++]);
    }
  }
}

matrix* matrix_transpose(matrix* A) {
  matrix* A_T = matrix_create(A->cols, A->rows);
  for (size_t src_row = 0; src_row < A->rows; ++src_row) {
    for (size_t src_col = 0; src_col < A->cols; ++src_col) {
      matrix_set_element(A_T, src_col, src_row, A->content[src_row][src_col]);
    }
  }
  return A_T;
}

void matrix_copy(matrix* dst, matrix* src, size_t dst_start_row, size_t dst_start_col, size_t src_start_row, size_t src_start_col, size_t row_num, size_t col_num) {
  for (size_t row_read = 0; row_read < row_num; ++row_read) {
    for (size_t col_read = 0; col_read < col_num; ++col_read) {
      matrix_set_element(dst, dst_start_row+row_read, dst_start_col+col_read, src->content[src_start_row+row_read][src_start_col+col_read]);
    }
  }
}
