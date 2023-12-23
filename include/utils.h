#ifndef _UTILS_H_
#define _UTILS_H_

#include <gsl/gsl_blas.h>

/* size_t get_num_of_h_splitted(gsl_matrix* mtx, size_t cols); */
/* size_t get_num_of_v_splitted(gsl_matrix* mtx, size_t rows); */
size_t get_num_of_parts_splitted(gsl_matrix* mtx, size_t rows, size_t cols);
/* gsl_matrix* png_read_to_matrix(char* file_name); */
/* void png_write_from_matrix(char* file_name, gsl_matrix* mtx); */
void matrix_print(gsl_matrix* mtx);
/* void matrix_normalize_colors(gsl_matrix* mtx); */
/* void matrix_denormalize_colors(gsl_matrix* mtx); */
/* gsl_matrix** load_images(char* path, size_t v_blocks_num, size_t h_blocks_num); */

#endif
