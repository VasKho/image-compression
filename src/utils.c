#include "utils.h"
/* #include <dirent.h> */
/* #include <string.h> */
#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <png.h>
#include <math.h>

/* size_t get_num_of_h_splitted(gsl_matrix* mtx, size_t cols) { */
/*   return ceil(mtx->size2/(double)cols); */
/* } */

/* size_t get_num_of_v_splitted(gsl_matrix* mtx, size_t rows) { */
/*   return ceil(mtx->size1/(double)rows); */
/* } */

size_t get_num_of_parts_splitted(gsl_matrix* mtx, size_t rows, size_t cols) {
  return ceil(mtx->size2/(double)cols) * ceil(mtx->size1/(double)rows);
}

gsl_matrix* png_read_to_matrix(char* file_name) {
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
  return info;
}

/* void png_write_from_matrix(char* file_name, gsl_matrix* mtx) { */
/*   FILE *fp = fopen(file_name, "wb"); */
/*   png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL); */
/*   png_infop info_ptr = png_create_info_struct(png_ptr); */
/*   png_set_IHDR(png_ptr, */
/* 	       info_ptr, */
/* 	       mtx->size2/3, */
/* 	       mtx->size1, */
/* 	       8, */
/* 	       PNG_COLOR_TYPE_RGB, */
/* 	       PNG_INTERLACE_NONE, */
/* 	       PNG_COMPRESSION_TYPE_DEFAULT, */
/* 	       PNG_FILTER_TYPE_DEFAULT); */
/*   png_bytepp row_pointers = png_malloc(png_ptr, mtx->size1*sizeof(png_bytep)); */
/*   for (size_t row = 0; row < mtx->size1; ++row) { */
/*     row_pointers[row] = png_malloc(png_ptr, mtx->size2*sizeof(png_byte)); */
/*     for (size_t col = 0; col < mtx->size2; ++col) */
/*       row_pointers[row][col] = (png_byte)mtx->data[mtx->size2*row + col]; */
/*   } */
/*   png_init_io(png_ptr, fp); */
/*   png_set_rows(png_ptr, info_ptr, row_pointers); */
/*   png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL); */
/*   for (size_t row = 0; row < mtx->size1; ++row) { */
/*     png_free(png_ptr, row_pointers[row]); */
/*   } */
/*   png_free(png_ptr, row_pointers); */
/*   png_destroy_write_struct(&png_ptr, &info_ptr); */
/*   fclose(fp); */
/* } */

/* void matrix_normalize_colors(gsl_matrix* mtx) { */
/*   for (size_t row = 0; row < mtx->size1; ++row) { */
/*     for (size_t col = 0; col < mtx->size2; ++col) { */
/*       gsl_matrix_set(mtx, row, col, (2*mtx->data[mtx->size2*row + col]/255)-1); */
/*     } */
/*   } */
/* } */

/* void matrix_denormalize_colors(gsl_matrix* mtx) { */
/*   for (size_t row = 0; row < mtx->size1; ++row) { */
/*     for (size_t col = 0; col < mtx->size2; ++col) { */
/*       gsl_matrix_set(mtx, row, col, 256*(mtx->data[mtx->size2*row + col]+1)/2); */
/*     } */
/*   } */
/* } */

void matrix_print(gsl_matrix* mtx) {
  for (size_t i = 0; i < mtx->size1; ++i) {
    for (size_t j = 0; j < mtx->size2; ++j) {
      printf("%-.4f ", mtx->data[mtx->size2*i + j]);
    }
    printf("\n");
  }
}
