#include "src/interface.h"
#include <string.h>
#include <stdio.h>

network_params default_params = {8, 8, 6};
double default_error = 0.15;
double default_aplpha = 0.0005;

int main(int argc, char* argv[]) {
  if (strncmp("generate", argv[1], strlen("generate")) == 0) {
    network_params params = default_params;
    if (argc >= 6) {
      sscanf(argv[3], "%zu", &params.block_rows);
      sscanf(argv[4], "%zu", &params.block_cols);
      sscanf(argv[5], "%zu", &params.compression);
    } else {
      printf("Less than 4 arguments were given. Default params will be used.\n");
    }
    return action_generate(argv[2], params);
  }
  if (strncmp("encode", argv[1], strlen("encode")) == 0) {
    if (argc < 5) {
      printf("Wrong number of arguments. Exitting program!\n");
      return 1;
    }
    return action_compress(argv[2], argv[3], argv[4]);
  }
  if (strncmp("decode", argv[1], strlen("decode")) == 0) {
    if (argc < 5) {
      printf("Wrong number of arguments. Exitting program!\n");
      return 1;
    }
    return action_decompress(argv[2], argv[3], argv[4]);
  }
  if (strncmp("train", argv[1], strlen("train")) == 0) {
    double alpha = default_aplpha;
    double error = default_error;
    if (argc < 4) {
      printf("Wrong number of arguments. Exitting program!\n");
      return 1;
    } else {
      if (argc > 4) sscanf(argv[4], "%lf", &error);
      if (argc >= 6) sscanf(argv[5], "%lf", &alpha);
    }
    return action_train(argv[2], argv[3], error, alpha);
  }
  if (strncmp("test", argv[1], strlen("test")) == 0) {
    if (argc < 4) {
      printf("Wrong number of arguments. Exitting program!\n");
      return 1;
    }
    return action_test(argv[2], argv[3]);
  }
  return 0;
}
