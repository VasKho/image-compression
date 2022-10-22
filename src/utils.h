#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <png.h>

gsl_matrix* png_read_to_matrix(char* file_name);
void png_write_from_matrix(char* file_name, gsl_matrix* A);
void matrix_print(gsl_matrix* A);
void matrix_normalize_colors(gsl_matrix* mtx);
void matrix_denormalize_colors(gsl_matrix* mtx);
gsl_matrix* gsl_matrix_multiply(gsl_matrix* A, gsl_matrix* B);
gsl_matrix* gsl_matrix_reshape(gsl_matrix* A, size_t new_rows, size_t new_cols);
void gsl_matrix_cp_submatrix(gsl_matrix* dst, gsl_matrix* src, size_t dst_start_row, size_t dst_start_col, size_t src_start_row, size_t src_start_col, size_t row_num, size_t col_num);

#endif
