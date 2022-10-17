#ifndef _UTILS_H_
#define _UTILS_H_

#include "matrix.h"
#include <stdio.h>
#include <png.h>

matrix* png_read_to_matrix(char* file_name);
void png_write_from_matrix(char* file_name, matrix* A);
void matrix_print(matrix* A);
void matrix_normalize_colors(matrix* mtx);
void matrix_denormalize_colors(matrix* mtx);

#endif
