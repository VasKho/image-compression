#include "training_pipeline.h"
#include "utils.h"
#include <dirent.h>
#include <string.h>
#include <gsl/gsl_matrix.h>
#include <png.h>
#include <math.h>

typedef struct {
  size_t width;
  size_t height;
} image_info;

training_pipeline_params_t init_taining_params(size_t block_width, size_t block_height, size_t max_num_of_blocks, double error, size_t compression) {
  return (training_pipeline_params_t){
    .block_width = block_width*3,
    .block_height = block_height,
    .max_num_of_blocks = max_num_of_blocks,
    .error = error,
    .compression = compression
  };
}

image_info get_image_info(char* file_path) {
  FILE *fp = fopen(file_path, "rb");
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info_ptr = png_create_info_struct(png_ptr);
  png_init_io(png_ptr, fp);
  png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_STRIP_ALPHA, NULL);
  image_info res = {
    .width = 3*png_get_image_width(png_ptr, info_ptr),
    .height = png_get_image_height(png_ptr, info_ptr)
  };
  png_destroy_info_struct(png_ptr, &info_ptr);
  png_destroy_read_struct(&png_ptr, NULL, NULL);
  fclose(fp);
  return res;
}

void load_image(char* file_name, gsl_matrix* info) {
  FILE *fp = fopen(file_name, "rb");
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info_ptr = png_create_info_struct(png_ptr);
  png_init_io(png_ptr, fp);
  png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_STRIP_ALPHA, NULL);
  png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);
  
  for (size_t row = 0; row < info->size1; ++row) {
    for (size_t col = 0; col < info->size2; ++col) {
      gsl_matrix_set(info, row, col, (double)row_pointers[row][col]);
    }
  }
  png_destroy_info_struct(png_ptr, &info_ptr);
  png_destroy_read_struct(&png_ptr, NULL, NULL);
  fclose(fp);
}

void normalize_colors(gsl_matrix* mtx) {
  for (size_t col = 0; col < mtx->size2; ++col) {
    gsl_matrix_set(mtx, 0, col, gsl_matrix_get(mtx, 0, col)/255);
  }
}

void split_image_as_vector(gsl_matrix* img, size_t rows, size_t cols, gsl_matrix** result) {
  int part = 0;
  for (size_t row = 0; row < img->size1; row += rows) {
    for (size_t col = 0; col < img->size2; col += cols) {
      result[part] = gsl_matrix_calloc(1, rows*cols);
      gsl_matrix tmp1 = gsl_matrix_view_array(gsl_matrix_submatrix(img, row, col, rows, cols).matrix.data, 1, rows*cols).matrix;
      gsl_matrix_memcpy(result[part], &tmp1);
      normalize_colors(result[part++]);
    }
  }
}

gsl_matrix** load_training_data(char* path, training_pipeline_params_t params, size_t* data_size) {
  struct dirent *de;
  DIR *dr = opendir(path);
  if (dr == NULL) { 
    printf("Could not open current directory"); 
  }

  gsl_matrix** data = calloc(params.max_num_of_blocks, sizeof(gsl_matrix*));
  size_t base_path_len = strlen(path);
  char format[6] = "%s%s";
  if (path[base_path_len-1] != '/') {
    ++base_path_len;
    memcpy(format, "%s/%s", 5);
  }

  image_info info;
  
  while ((de = readdir(dr)) != NULL) {
    if (de->d_type == DT_REG) {
      char* buff = calloc(base_path_len+strlen(de->d_name)+1, sizeof(char));
      sprintf(buff, format, path, de->d_name);
      info = get_image_info(buff);
      free(buff);
      break;
    }
  }

  size_t blocks_per_image = ceil(info.width/(double)params.block_width) * ceil(info.height/(double)params.block_height);
  gsl_matrix* image_buff = gsl_matrix_calloc(info.height, info.width);

  size_t i = 0;
  for (; i+blocks_per_image < params.max_num_of_blocks && (de = readdir(dr)) != NULL; i += blocks_per_image) {
    if (de->d_type == DT_REG) {
      char* buff = calloc(base_path_len+strlen(de->d_name)+1, sizeof(char));
      sprintf(buff, format, path, de->d_name);
      load_image(buff, image_buff);
      split_image_as_vector(image_buff, params.block_height, params.block_width, data+i);
      free(buff);
    }
  }
  gsl_matrix_free(image_buff);
  closedir(dr);
  *data_size = i;
  return data;
}

void free_training_data(gsl_matrix** data, size_t data_size) {
  for (size_t i = 0; i < data_size; ++i) {
    gsl_matrix_free(data[i]);
  }
  free(data);
}
