#include "utils.h"

gsl_matrix* png_read_to_matrix(char* file_name) {
  FILE *fp = fopen(file_name, "rb");
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info_ptr = png_create_info_struct(png_ptr);
  png_init_io(png_ptr, fp);
  png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_STRIP_ALPHA, NULL);
  gsl_matrix* info = gsl_matrix_alloc(png_get_image_height(png_ptr, info_ptr),
				      3*png_get_image_width(png_ptr, info_ptr));
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

void png_write_from_matrix(char* file_name, gsl_matrix* A) {
  FILE *fp = fopen(file_name, "wb");
  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info_ptr = png_create_info_struct(png_ptr);
  png_set_IHDR(png_ptr, 
	       info_ptr, 
	       A->size2/3,
	       A->size1,
	       8,
	       PNG_COLOR_TYPE_RGB,
	       PNG_INTERLACE_NONE, 
	       PNG_COMPRESSION_TYPE_DEFAULT, 
	       PNG_FILTER_TYPE_DEFAULT);
  png_bytepp row_pointers = png_malloc(png_ptr, A->size1*sizeof(png_bytep)); 
  for (size_t row = 0; row < A->size1; ++row) {
    row_pointers[row] = png_malloc(png_ptr, A->size2*sizeof(png_byte)); 
    for (size_t col = 0; col < A->size2; ++col) 
      row_pointers[row][col] = (png_byte)gsl_matrix_get(A, row, col); 
  } 
  png_init_io(png_ptr, fp);
  png_set_rows(png_ptr, info_ptr, row_pointers);
  png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
  for (size_t row = 0; row < A->size1; ++row) {
    png_free(png_ptr, row_pointers[row]);
  }
  png_free(png_ptr, row_pointers);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(fp);
}

void matrix_print(gsl_matrix* A) {
  for (size_t i = 0; i < A->size1; ++i) { 
    for (size_t j = 0; j < A->size2; ++j) { 
      printf("%lf ", gsl_matrix_get(A, i, j)); 
    } 
    printf("\n"); 
  } 
}

void matrix_normalize_colors(gsl_matrix* mtx) {
  for (size_t row = 0; row < mtx->size1; ++row) {
    for (size_t col = 0; col < mtx->size2; ++col) {
      gsl_matrix_set(mtx, row, col, (2*gsl_matrix_get(mtx, row, col)/255)-1);
    }
  }
}

void matrix_denormalize_colors(gsl_matrix* mtx) {
  for (size_t row = 0; row < mtx->size1; ++row) {
    for (size_t col = 0; col < mtx->size2; ++col) {
      gsl_matrix_set(mtx, row, col, 256*(gsl_matrix_get(mtx, row, col)+1)/2);
    }
  }
}

gsl_matrix* gsl_matrix_multiply(gsl_matrix* A, gsl_matrix* B) {
  gsl_matrix* C = gsl_matrix_alloc(A->size1, B->size2);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
  return C;
} 

gsl_matrix* gsl_matrix_reshape(gsl_matrix* A, size_t new_rows, size_t new_cols) {
  if (A->size1*A->size2 != new_rows*new_cols) {
    printf("Invalid new sizes. Unable to reshape\n");
    return NULL;
  }
  gsl_matrix* new_matrix = gsl_matrix_alloc(new_rows, new_cols);
  size_t old_row = 0, old_col = 0;
  for (size_t new_row = 0; new_row < new_rows; ++new_row) {
    for (size_t new_col = 0; new_col < new_cols; ++new_col) {
      if (old_col == A->size2) {
	old_col = 0;
	++old_row;
      }
      gsl_matrix_set(new_matrix, new_row, new_col, gsl_matrix_get(A, old_row, old_col++));
    }
  }
  return new_matrix;
}

void gsl_matrix_cp_submatrix(gsl_matrix* dst, gsl_matrix* src, size_t dst_start_row, size_t dst_start_col, size_t src_start_row, size_t src_start_col, size_t row_num, size_t col_num) { 
  for (size_t row_read = 0; row_read < row_num; ++row_read) {
    for (size_t col_read = 0; col_read < col_num; ++col_read) {
      gsl_matrix_set(dst, dst_start_row+row_read, dst_start_col+col_read, gsl_matrix_get(src, src_start_row+row_read, src_start_col+col_read)); 
    } 
  } 
} 
