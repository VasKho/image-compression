#ifndef _INTERFACE_H_
#define _INTERFACE_H_

#include <stdio.h>
#include "neural_network.h"
#include <string.h>
#include "utils.h"

int action_generate(char* path_to_save, network_params params);
int action_train(char* path_to_images, char* path_to_weights, network_params params);
int action_test(char* path_to_image, char* path_to_weights, network_params params);

#endif
