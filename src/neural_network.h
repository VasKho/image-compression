#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <time.h>
#include <math.h>
#include "utils.h"

typedef struct {
  size_t block_rows;
  size_t block_cols;
  size_t compression;
  double error;
} network_params;

gsl_matrix* generate_weights(size_t rows, size_t cols);
void load_weights(char* path, gsl_matrix* dest_encode, gsl_matrix* dest_decode);
int save_weights(char* path, gsl_matrix* src_encode, gsl_matrix* src_decode);
size_t get_num_of_h_splitted(gsl_matrix* A, size_t cols);
size_t get_num_of_v_splitted(gsl_matrix* A, size_t rows);
size_t get_num_of_parts_splitted(gsl_matrix* A, size_t rows, size_t cols);
gsl_matrix** split_image(gsl_matrix* img, size_t rows, size_t cols);
gsl_matrix* unite_image(gsl_matrix** arr, size_t rows, size_t cols);
gsl_matrix** encode(gsl_matrix* img, gsl_matrix* weights, size_t rows, size_t cols);
gsl_matrix* decode(gsl_matrix** compressed, gsl_matrix* src, gsl_matrix* weights, size_t rows, size_t cols);
double train(gsl_matrix* img, gsl_matrix* encode_weights, gsl_matrix* decode_weights, size_t m, size_t n);

#endif
