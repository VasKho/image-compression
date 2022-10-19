#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#include "matrix.h"
#include <time.h>
#include <math.h>

matrix* generate_weights(size_t rows, size_t cols);
void load_weights(char* path, matrix* dest_encode, matrix* dest_decode);
int save_weights(char* path, matrix* src_encode, matrix* src_decode);
size_t get_num_of_h_splitted(matrix* A, size_t cols);
size_t get_num_of_v_splitted(matrix* A, size_t rows);
size_t get_num_of_parts_splitted(matrix* A, size_t rows, size_t cols);
matrix** split_image(matrix* img, size_t rows, size_t cols);
matrix* unite_image(matrix** arr, size_t rows, size_t cols);
matrix** encode(matrix* img, matrix* weights, size_t rows, size_t cols);
matrix* decode(matrix** compressed, matrix* src, matrix* weights, size_t rows, size_t cols);
double train(matrix* img, matrix* encode_weights, matrix* decode_weights, size_t m, size_t n);

#endif
