#include "src/utils.h"
#include "src/neural_network.h"
#include <stdio.h>
#include <string.h>

size_t n = 16;
size_t m = 16;
size_t Z = 6;
double err = 0.15;

int action_generate(char* path_to_save, size_t rows, size_t cols);
int action_train(char* path_to_images, char* path_to_weights, size_t rows, size_t cols);
int action_test(char* path_to_image, char* path_to_weights, size_t rows, size_t cols);

int main(int argc, char* argv[]) {
  if (strncmp("generate", argv[1], strlen("generate")) == 0) {
    return action_generate(argv[2], 3*m*n, 3*m*n/Z);
  }
  if (strncmp("train", argv[1], strlen("train")) == 0) {
    return action_train(argv[2], argv[3], 3*m*n, 3*m*n/Z);
  }
  if (strncmp("test", argv[1], strlen("test")) == 0) {
    return action_test(argv[2], argv[3], 3*m*n, 3*m*n/Z);
  }
  return 0;
}

int action_generate(char* path_to_save, size_t rows, size_t cols) {
  gsl_matrix* weights = generate_weights(rows, cols);
  for (size_t row = 0; row < weights->size1; ++row) {
    for (size_t col = 0; col < weights->size2; ++col) {
      double elem = gsl_matrix_get(weights, row, col);
      gsl_matrix_set(weights, row, col, elem/cols);
    }
  }
  gsl_matrix* weightsT = gsl_matrix_alloc(weights->size2, weights->size1);
  gsl_matrix_transpose_memcpy(weightsT, weights);
  save_weights(path_to_save, weights, weightsT);
  gsl_matrix_free(weights);
  gsl_matrix_free(weightsT);
  return 0;
}

int action_train(char* path_to_images, char* path_to_weights, size_t rows, size_t cols) { 
  gsl_matrix* weights = gsl_matrix_alloc(rows, cols); 
  gsl_matrix* weightsT = gsl_matrix_alloc(cols, rows); 
  load_weights(path_to_weights, weights, weightsT); 
  char command_prefix[] = "find "; 
  char command_postfix[] = " -type f -name '*.png'"; 
  char* command = calloc(strlen(command_prefix)+strlen(path_to_images)+strlen(command_postfix), sizeof(char)); 
  strcat(command, command_prefix); 
  strcat(command, path_to_images); 
  strcat(command, command_postfix); 
  double sum_err = 0; 
  do {
    sum_err = 0; 
    FILE* pipe = popen(command, "r"); 
    if (!pipe) { 
      return -1; 
    } 
    char* path = calloc(2048, sizeof(char)); 
    char temp; 
    while (!feof(pipe)) { 
      temp = fgetc(pipe); 
      if (temp != '\n') { 
	strncat(path, &temp, 1); 
      } else { 
	printf("%s\n", path); 
	gsl_matrix* image = png_read_to_matrix(path); 
	matrix_normalize_colors(image); 
	sum_err += train(image, weights, weightsT, n, m); 
	save_weights(path_to_weights, weights, weightsT); 
	printf("%lf\n", sum_err); 
	memset(path, 0, 2048); 
	gsl_matrix_free(image); 
      } 
    } 
    pclose(pipe); 
    free(path); 
  } while (sum_err > err);
  gsl_matrix_free(weights); 
  gsl_matrix_free(weightsT); 
  free(command); 
  return 0; 
} 

int action_test(char* path_to_image, char* path_to_weights, size_t rows, size_t cols) { 
  gsl_matrix* image = png_read_to_matrix(path_to_image); 
  matrix_normalize_colors(image); 
  gsl_matrix* weights = gsl_matrix_alloc(rows, cols); 
  gsl_matrix* weightsT = gsl_matrix_alloc(cols, rows); 
  load_weights(path_to_weights, weights, weightsT); 
  gsl_matrix** encoded = encode(image, weights, n, m); 
  gsl_matrix* decoded = decode(encoded, image, weightsT, n, m); 
  matrix_denormalize_colors(decoded); 
  png_write_from_matrix("out.png", decoded); 
  gsl_matrix_free(image); 
  gsl_matrix_free(weights); 
  gsl_matrix_free(weightsT); 
  gsl_matrix_free(decoded); 
  size_t num_of_parts = get_num_of_parts_splitted(image, n, m*3); 
  for (size_t i = 0; i < num_of_parts; ++i) { 
    gsl_matrix_free(encoded[i]); 
  } 
  free(encoded); 
  return 0; 
} 
