#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#include <gsl/gsl_blas.h>
#include <math.h>
#include "image.h"

typedef struct {
  size_t block_width;
  size_t block_height;
  size_t compression;
  gsl_matrix* encode_weights;
  gsl_matrix* decode_weights;
} neural_network_t;

typedef struct {
  size_t block_width;
  size_t block_height;
  size_t width;
  size_t height;
  gsl_matrix** data;
} encoded_image_t;


neural_network_t neural_network_init(size_t block_width, size_t block_height, size_t compression);
void neural_network_free(neural_network_t nn);
neural_network_t generate_weights(neural_network_t nn);
neural_network_t load_model(char* path);
void save_model(neural_network_t nn, char* path);
encoded_image_t encode(neural_network_t nn, image_t img);
image_t decode(neural_network_t nn, encoded_image_t enc);
double train(neural_network_t nn, double alpha, gsl_matrix** data, size_t data_size);

void encoded_image_free(encoded_image_t enc);

#endif
