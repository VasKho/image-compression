#ifndef _TRAINING_PIPELINE_H_
#define _TRAINING_PIPELINE_H_

#include <stdlib.h>
#include <gsl/gsl_matrix.h>

typedef struct {
  size_t max_num_of_blocks;
  size_t block_width;
  size_t block_height;
  double error;
  size_t compression;
} training_pipeline_params_t;

training_pipeline_params_t init_taining_params(size_t block_width, size_t block_height, size_t max_num_of_blocks, double error, size_t compression);
gsl_matrix** load_training_data(char* path, training_pipeline_params_t params, size_t* data_size);
void free_training_data(gsl_matrix** data, size_t data_size);

#endif
