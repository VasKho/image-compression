#include "src/matrix.h"
#include "src/utils.h"
#include "src/neural_network.h"
#include <stdio.h>

size_t n = 2;
size_t m = 2;
size_t Z = 3;
double err = 0.15;

// Разделить веса после генерации на размер скрытого слоя

int main(int argc, char* argv[]) {
  matrix* image = png_read_to_matrix(argv[1]);
  matrix_normalize_colors(image);
  matrix* weights = generate_weights(3*m*n, 3*m*n/Z);
  matrix* weightsT = matrix_transpose(weights);
  double sum_err;
  do {
    sum_err = train(image, weights, weightsT, n, m);
    printf("%f\n", sum_err);
  } while (sum_err > err);
  matrix** encoded = encode(image, weights, n, m);
  matrix* decoded = decode(encoded, image, weightsT, n, m);
  for (size_t i = 0; i < get_num_of_parts_splitted(image, n, m*3); ++i) {
    matrix_destroy(encoded[i]);
  }
  free(encoded);
  matrix_denormalize_colors(decoded);
  png_write_from_matrix("./test1.png", decoded);
  matrix_destroy(image);
  matrix_destroy(weights);
  matrix_destroy(decoded);
  matrix_destroy(weightsT);
  return 0;
}
