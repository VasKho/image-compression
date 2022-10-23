#include "src/interface.h"
#include <string.h>
#include <stdio.h>

network_params default_params = {8, 8, 6, 0.15};

int main(int argc, char* argv[]) {
  if (strncmp("generate", argv[1], strlen("generate")) == 0) {
    return action_generate(argv[2], default_params);
  }
  if (strncmp("train", argv[1], strlen("train")) == 0) {
    return action_train(argv[2], argv[3], default_params);
  }
  if (strncmp("test", argv[1], strlen("test")) == 0) {
    return action_test(argv[2], argv[3], default_params);
  }
  return 0;
}
