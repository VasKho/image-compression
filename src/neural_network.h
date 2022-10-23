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
} network_params;

gsl_matrix* generate_weights(size_t rows, size_t cols);
size_t get_num_of_parts_splitted(gsl_matrix* A, size_t rows, size_t cols);
gsl_matrix** split_image(gsl_matrix* img, network_params params);
gsl_matrix* unite_image(gsl_matrix** arr, size_t out_rows, size_t out_cols);
gsl_matrix** encode(gsl_matrix* img, gsl_matrix* weights, network_params params);
gsl_matrix* decode(gsl_matrix** compressed, gsl_matrix* weights, size_t num_of_parts, size_t out_rows, size_t out_cols, network_params params);
double train(gsl_matrix* img, gsl_matrix* encode_weights, gsl_matrix* decode_weights, network_params params, double aplha);

#endif
