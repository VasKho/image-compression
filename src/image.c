#include "image.h"
#include "utils.h"

image_t image_init(size_t width, size_t height, gsl_matrix* data) {
  return (image_t){
    .width = width,
    .height = height,
    .data = data
  };
}

void image_free(image_t img) {
  gsl_matrix_free(img.data);
}

void splitted_image_free(splitted_image_t img_arr) {
  free(img_arr.data);
}

image_t image_read_from_png(char* file_name) {
  FILE *fp = fopen(file_name, "rb");
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info_ptr = png_create_info_struct(png_ptr);
  png_init_io(png_ptr, fp);
  png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_STRIP_ALPHA, NULL);
  gsl_matrix* info =
    gsl_matrix_calloc(png_get_image_height(png_ptr, info_ptr), 3*png_get_image_width(png_ptr, info_ptr));
  png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);
  
  for (size_t row = 0; row < info->size1; ++row) {
    for (size_t col = 0; col < info->size2; ++col) {
      gsl_matrix_set(info, row, col, (double)row_pointers[row][col]);
    }
  }
  png_destroy_info_struct(png_ptr, &info_ptr);
  png_destroy_read_struct(&png_ptr, NULL, NULL);
  fclose(fp);
  return image_init(info->size2, info->size1, info);
}

int image_write_to_png(char* file_name, image_t img) {
  FILE *fp = fopen(file_name, "wb");
  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info_ptr = png_create_info_struct(png_ptr);
  png_set_IHDR(png_ptr,
	       info_ptr,
	       img.width/3,
	       img.height,
	       8,
	       PNG_COLOR_TYPE_RGB,
	       PNG_INTERLACE_NONE,
	       PNG_COMPRESSION_TYPE_DEFAULT,
	       PNG_FILTER_TYPE_DEFAULT);
  png_bytepp row_pointers = png_malloc(png_ptr, img.height*sizeof(png_bytep));
  for (size_t row = 0; row < img.height; ++row) {
    row_pointers[row] = png_malloc(png_ptr, img.width*sizeof(png_byte));
    for (size_t col = 0; col < img.width; ++col)
      row_pointers[row][col] = (png_byte)img.data->data[img.width*row + col];
  }
  png_init_io(png_ptr, fp);
  png_set_rows(png_ptr, info_ptr, row_pointers);
  png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
  for (size_t row = 0; row < img.height; ++row) {
    png_free(png_ptr, row_pointers[row]);
  }
  png_free(png_ptr, row_pointers);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(fp);
  return 0;
}

splitted_image_t split_image(image_t img, size_t block_width, size_t block_height) {
  size_t num_of_parts = get_num_of_parts_splitted(img.data, block_height, block_width);
  gsl_matrix* result = calloc(num_of_parts, sizeof(gsl_matrix));
  int part = 0;
  for (size_t row = 0; row < img.height; row += block_height) {
    for (size_t col = 0; col < img.width; col += block_width) {
      result[part++] = gsl_matrix_submatrix(img.data, row, col, block_height, block_width).matrix;
    }
  }
  return (splitted_image_t){
    .block_width = block_width,
    .block_height = block_height,
    .width = img.width,
    .height = img.height,
    .data = result
  };
}

image_t unite_image(splitted_image_t img_arr) {
  size_t ind = 0;
  size_t num_of_h_splitted = ceil(img_arr.width/(double)img_arr.block_width);
  size_t num_of_v_splitted = ceil(img_arr.height/(double)img_arr.block_height);
  gsl_matrix* result = gsl_matrix_calloc(img_arr.height, img_arr.width);
  for (size_t v_block = 0; v_block < num_of_v_splitted; ++v_block) {
    for (size_t row_ind = 0; row_ind < img_arr.block_height; ++row_ind) {
      for (size_t h_block = 0; h_block < num_of_h_splitted; ++h_block) {
	for (size_t col_ind = 0; col_ind < img_arr.block_width; ++col_ind) {
	  result->data[ind++] = gsl_matrix_get(&img_arr.data[num_of_h_splitted*v_block + h_block], row_ind, col_ind);
	}
      }
    }
  }
  return image_init(img_arr.width, img_arr.height, result);
}

void image_normalize_colors(image_t img) {
  for (size_t row = 0; row < img.data->size1; ++row) {
    for (size_t col = 0; col < img.data->size2; ++col) {
      gsl_matrix_set(img.data, row, col, gsl_matrix_get(img.data, row, col)/255);
    }
  }
}

void image_denormalize_colors(image_t img) {
  for (size_t row = 0; row < img.data->size1; ++row) {
    for (size_t col = 0; col < img.data->size2; ++col) {
      gsl_matrix_set(img.data, row, col, 255*gsl_matrix_get(img.data, row, col));
    }
  }
}
