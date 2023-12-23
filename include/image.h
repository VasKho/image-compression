#ifndef _IMAGE_ENCODER_H_
#define _IMAGE_ENCODER_H_

#include <gsl/gsl_matrix.h>
#include <png.h>
#include <math.h>

typedef struct {
  size_t width;
  size_t height;
  gsl_matrix* data;
} image_t;

typedef struct {
  size_t block_width;
  size_t block_height;
  size_t width;
  size_t height;
  gsl_matrix* data;
} splitted_image_t;


image_t image_init(size_t width, size_t height, gsl_matrix* data);
void image_free(image_t img);
void splitted_image_free(splitted_image_t img_arr);
image_t image_read_from_png(char* file_name);
int image_write_to_png(char* file_name, image_t img);
splitted_image_t split_image(image_t img, size_t block_width, size_t block_height);
image_t unite_image(splitted_image_t img_arr);

void image_normalize_colors(image_t img);
void image_denormalize_colors(image_t img);

#endif
