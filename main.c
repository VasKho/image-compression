#include "src/matrix.h"
#include "src/utils.h"
#include "src/neural_network.h"
#include <stdio.h>
#include <string.h>

size_t n = 16;
size_t m = 16;
size_t Z = 3;
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
  matrix* weights = generate_weights(rows, cols);
  for (size_t row = 0; row < weights->rows; ++row) {
    for (size_t col = 0; col < weights->cols; ++col) {
      weights->content[row][col] /= cols;
    }
  }
  matrix* weightsT = matrix_transpose(weights);
  FILE* fp = fopen(path_to_save, "w+t");
  for (size_t row = 0; row < weights->rows; ++row) {
    for (size_t col = 0; col < weights->cols; ++col) {
      fprintf(fp, "%lf ", weights->content[row][col]);
    }
  }
  for (size_t row = 0; row < weightsT->rows; ++row) {
    for (size_t col = 0; col < weightsT->cols; ++col) {
      fprintf(fp, "%lf ", weightsT->content[row][col]);
    }
  }
  fclose(fp);
  matrix_destroy(weights);
  matrix_destroy(weightsT);
  return 0;
}

int action_train(char* path_to_images, char* path_to_weights, size_t rows, size_t cols) {
  matrix* weights = matrix_create(rows, cols);
  matrix* weightsT = matrix_create(cols, rows);
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
	matrix* image = png_read_to_matrix(path);
	matrix_normalize_colors(image);
	sum_err += train(image, weights, weightsT, n, m);
	save_weights(path_to_weights, weights, weightsT);
	printf("%lf\n", sum_err);
	memset(path, 0, 2048);
	matrix_destroy(image);
      }
    }
    pclose(pipe);
    free(path);
  } while (sum_err > err); 
  matrix_destroy(weights);
  matrix_destroy(weightsT);
  free(command);
  return 0;
}

int action_test(char* path_to_image, char* path_to_weights, size_t rows, size_t cols) {
  matrix* image = png_read_to_matrix(path_to_image);
  matrix_normalize_colors(image);
  matrix* weights = matrix_create(rows, cols);
  matrix* weightsT = matrix_create(cols, rows);
  load_weights(path_to_weights, weights, weightsT);
  matrix** encoded = encode(image, weights, n, m);
  matrix* decoded = decode(encoded, image, weightsT, n, m);
  matrix_denormalize_colors(decoded);
  png_write_from_matrix("out.png", decoded);
  matrix_destroy(image);
  matrix_destroy(weights);
  matrix_destroy(weightsT);
  matrix_destroy(decoded);
  size_t num_of_parts = get_num_of_parts_splitted(image, n, m*3);
  for (size_t i = 0; i < num_of_parts; ++i) {
    matrix_destroy(encoded[i]);
  }
  free(encoded);
  return 0;
}
