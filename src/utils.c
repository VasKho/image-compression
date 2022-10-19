#include "utils.h"

matrix* png_read_to_matrix(char* file_name) {
  FILE *fp = fopen(file_name, "rb");
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info_ptr = png_create_info_struct(png_ptr);
  png_init_io(png_ptr, fp);
  png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_STRIP_ALPHA, NULL);
  matrix* info = matrix_create(png_get_image_height(png_ptr, info_ptr),
			       3*png_get_image_width(png_ptr, info_ptr));
  png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);
  for (size_t row = 0; row < info->rows; ++row) {
    for (size_t col = 0; col < info->cols; ++col) {
      matrix_set_element(info, row, col, (double)row_pointers[row][col]);
    }
  }
  png_destroy_info_struct(png_ptr, &info_ptr);
  png_destroy_read_struct(&png_ptr, NULL, NULL);
  fclose(fp);
  return info;
}

void png_write_from_matrix(char* file_name, matrix* A) {
  FILE *fp = fopen(file_name, "wb");
  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info_ptr = png_create_info_struct(png_ptr);
  png_set_IHDR(png_ptr, 
	       info_ptr, 
	       A->cols/3, 
	       A->rows, 
	       8,
	       PNG_COLOR_TYPE_RGB,
	       PNG_INTERLACE_NONE, 
	       PNG_COMPRESSION_TYPE_DEFAULT, 
	       PNG_FILTER_TYPE_DEFAULT);
  png_bytepp row_pointers = png_malloc(png_ptr, A->rows*sizeof(png_bytep)); 
  for (size_t row = 0; row < A->rows; ++row) {
    row_pointers[row] = png_malloc(png_ptr, A->cols*sizeof(png_byte)); 
    for (size_t col = 0; col < A->cols; ++col) 
      row_pointers[row][col] = (png_byte)A->content[row][col]; 
  } 
  png_init_io(png_ptr, fp);
  png_set_rows(png_ptr, info_ptr, row_pointers);
  png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
  for (size_t row = 0; row < A->rows; ++row) {
    png_free(png_ptr, row_pointers[row]);
  }
  png_free(png_ptr, row_pointers);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(fp);
}

void matrix_print(matrix* A) {
  for (size_t i = 0; i < A->rows; ++i) {
    for (size_t j = 0; j < A->cols; ++j) {
      printf("%lf ", A->content[i][j]);
    }
    printf("\n");
  }
}

void matrix_normalize_colors(matrix* mtx) {
  for (size_t row = 0; row < mtx->rows; ++row) {
    for (size_t col = 0; col < mtx->cols; ++col) {
      matrix_set_element(mtx, row, col, (2*mtx->content[row][col]/255)-1);
    }
  }
}

void matrix_denormalize_colors(matrix* mtx) {
  for (size_t row = 0; row < mtx->rows; ++row) {
    for (size_t col = 0; col < mtx->cols; ++col) {
      matrix_set_element(mtx, row, col, 256*(mtx->content[row][col]+1)/2);
    }
  }
}
