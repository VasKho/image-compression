#include "neural_network.h"

gsl_matrix* generate_weights(size_t rows, size_t cols) {
  srand(time(0));
  gsl_matrix* weights = gsl_matrix_alloc(rows, cols);
  for (size_t row = 0; row < weights->size1; ++row) {
    for (size_t col = 0; col < weights->size2; ++col) {
      gsl_matrix_set(weights, row, col, rand() % 3 - 1);
    }
  }
  return weights;
}

size_t get_num_of_parts_splitted(gsl_matrix* A, size_t rows, size_t cols) {
  return ceil(A->size2/(double)cols) * ceil(A->size1/(double)rows);
}

gsl_matrix* inline_transform_submatrix(gsl_matrix* img, size_t row_ind, size_t col_ind, size_t rows, size_t cols) {
  gsl_matrix_const_view view = gsl_matrix_const_submatrix(img, row_ind, col_ind, rows, cols);
  gsl_matrix* temp = gsl_matrix_alloc(view.matrix.size1, view.matrix.size2);
  gsl_matrix_memcpy(temp, &view.matrix);
  gsl_matrix* result = gsl_matrix_reshape(temp, 1, temp->size1*temp->size2);
  gsl_matrix_free(temp);
  return result;
}

gsl_matrix** split_image(gsl_matrix* img, network_params params) { 
  size_t rows = params.block_rows;
  size_t cols = params.block_cols*3;
  size_t num_of_parts = get_num_of_parts_splitted(img, rows, cols);
  size_t part = 0; 
  gsl_matrix** result = calloc(num_of_parts, sizeof(gsl_matrix_view*)); 
  for (size_t row_ind = 0; row_ind < img->size1; row_ind += rows) {  
    if (row_ind + rows >= img->size1) {
      row_ind = img->size1 - row_ind + rows; 
      for (size_t col_ind = 0; col_ind < img->size2; col_ind += cols) { 
	if (col_ind + cols >= img->size2) {
	  col_ind = img->size2 - col_ind + cols;
	  result[part++] = inline_transform_submatrix(img, row_ind, col_ind, rows, cols);
	  break;
	}
	result[part++] = inline_transform_submatrix(img, row_ind, col_ind, rows, cols);
      } 
      break; 
    } 
    for (size_t col_ind = 0; col_ind < img->size2; col_ind += cols) {
      if (col_ind + cols >= img->size2) {
	col_ind = img->size2 - col_ind + cols;
	result[part++] = inline_transform_submatrix(img, row_ind, col_ind, rows, cols);
	break; 
      }
      result[part++] = inline_transform_submatrix(img, row_ind, col_ind, rows, cols);
    } 
  }  
  return result;  
} 

gsl_matrix* unite_image(gsl_matrix** arr, size_t out_rows, size_t out_cols) {
  size_t part = 0;
  size_t d_row = arr[0]->size1;
  size_t d_col = arr[0]->size2;
  gsl_matrix* result = gsl_matrix_alloc(out_rows, out_cols);
  for (size_t row_ind = 0; row_ind < result->size1; row_ind += d_row) { 
    if (row_ind + d_row >= result->size1) { 
      row_ind = result->size1 - row_ind + d_row; 
      for (size_t col_ind = 0; col_ind < result->size2; col_ind += d_col) { 
	if (col_ind + d_col >= result->size2) { 
	col_ind = result->size2 - col_ind + d_col; 
	gsl_matrix_cp_submatrix(result, arr[part++], row_ind, col_ind, 0, 0, d_row, d_col);
	break;
      }
	gsl_matrix_cp_submatrix(result, arr[part++], row_ind, col_ind, 0, 0, d_row, d_col);
      } 
      break; 
    } 
    for (size_t col_ind = 0; col_ind < result->size2; col_ind += d_col) { 
      if (col_ind + d_col >= result->size2) { 
	col_ind = result->size2 - col_ind + d_col; 
	gsl_matrix_cp_submatrix(result, arr[part++], row_ind, col_ind, 0, 0, d_row, d_col);
	break; 
      } 
      gsl_matrix_cp_submatrix(result, arr[part++], row_ind, col_ind, 0, 0, d_row, d_col);
    } 
  } 
  return result;
} 

gsl_matrix** encode(gsl_matrix* img, gsl_matrix* weights, network_params params) { 
  size_t rows = params.block_rows;
  size_t cols = params.block_cols*3;
  size_t num_of_parts = get_num_of_parts_splitted(img, rows, cols); 
  gsl_matrix** splitted_img = split_image(img, params);
  gsl_matrix** out_mtx = calloc(num_of_parts, sizeof(gsl_matrix*));
  for (size_t i = 0; i < num_of_parts; ++i) {
    out_mtx[i] = gsl_matrix_multiply(splitted_img[i], weights);
  }
  for (size_t i = 0; i < num_of_parts; ++i) {
    gsl_matrix_free(splitted_img[i]);
  }
  free(splitted_img);
  return out_mtx;
} 

gsl_matrix* decode(gsl_matrix** compressed, gsl_matrix* weights, size_t num_of_parts, size_t out_rows, size_t out_cols, network_params params) { 
  size_t rows = params.block_rows;
  size_t cols = params.block_cols*3;
  gsl_matrix** out_mtx = calloc(num_of_parts, sizeof(gsl_matrix*)); 
  for (size_t i = 0; i < num_of_parts; ++i) {
    gsl_matrix* temp_out = gsl_matrix_multiply(compressed[i], weights);
    out_mtx[i] = gsl_matrix_reshape(temp_out, rows, cols);
    gsl_matrix_free(temp_out);
  }
  gsl_matrix* new_img = unite_image(out_mtx, out_rows, out_cols);
  for (size_t i = 0; i < num_of_parts; ++i) {
    gsl_matrix_free(out_mtx[i]);
  }
  free(out_mtx);
  return new_img;
} 

void modify_output_weights(gsl_matrix* decode_weights, double alpha, gsl_matrix* output_transposed, gsl_matrix* error) {
  gsl_matrix* mul_1 = gsl_matrix_multiply(output_transposed, error);
  gsl_matrix_scale(mul_1, alpha);
  gsl_matrix_sub(decode_weights, mul_1);
  gsl_matrix_free(mul_1);
}

void modify_input_weights(gsl_matrix* encode_weights, gsl_matrix* decode_weights, double alpha, gsl_matrix* error, gsl_matrix* input) {
  gsl_matrix* input_transposed = gsl_matrix_alloc(input->size2, input->size1);
  gsl_matrix_transpose_memcpy(input_transposed, input);
  gsl_matrix* decode_transposed = gsl_matrix_alloc(decode_weights->size2, decode_weights->size1);
  gsl_matrix_transpose_memcpy(decode_transposed, decode_weights);
  gsl_matrix* mul_1 = gsl_matrix_multiply(input_transposed, error);
  gsl_matrix* mul_2 = gsl_matrix_multiply(mul_1, decode_transposed);
  gsl_matrix_scale(mul_2, alpha);
  gsl_matrix_sub(encode_weights, mul_2);
  gsl_matrix_free(input_transposed);
  gsl_matrix_free(decode_transposed);
  gsl_matrix_free(mul_1);
  gsl_matrix_free(mul_2);
}

gsl_matrix* modify_weights(gsl_matrix* img, gsl_matrix* encode_weights, gsl_matrix* decode_weights, double alpha) {
  gsl_matrix* encoded = gsl_matrix_multiply(img, encode_weights);
  gsl_matrix* decoded = gsl_matrix_multiply(encoded, decode_weights);
  gsl_matrix* error = gsl_matrix_alloc(decoded->size1, decoded->size2);
  gsl_matrix_memcpy(error, decoded);
  gsl_matrix_sub(error, img); // Now decoded stores error matrix
  gsl_matrix* output_transposed = gsl_matrix_alloc(encoded->size2, encoded->size1);
  gsl_matrix_transpose_memcpy(output_transposed, encoded);
  modify_output_weights(decode_weights, alpha, output_transposed, error);
  modify_input_weights(encode_weights, decode_weights, alpha, error, img);
  gsl_matrix_free(encoded);
  gsl_matrix_free(decoded);
  gsl_matrix_free(output_transposed);
  return error;
} 

double avg_error(gsl_matrix* error) { 
  double avg = 0;
  for (size_t row = 0; row < error->size1; ++row) { 
    for (size_t col = 0; col < error->size2; ++col) { 
      avg += gsl_matrix_get(error, row, col) * gsl_matrix_get(error, row, col); 
    }
  }
  return avg; 
} 

double train(gsl_matrix* img, gsl_matrix* encode_weights, gsl_matrix* decode_weights, network_params params, double alpha) {
  size_t num_of_parts = get_num_of_parts_splitted(img, params.block_rows, 3*params.block_cols);
  gsl_matrix** splitted_img = split_image(img, params); 
  double sum_avg_error = 0;
  for (size_t part = 0; part < num_of_parts; ++part) {
    gsl_matrix* error = modify_weights(splitted_img[part], encode_weights, decode_weights, alpha);
    sum_avg_error += avg_error(error);
    gsl_matrix_free(error);
  } 
  for (size_t i = 0; i < num_of_parts; ++i) { 
    gsl_matrix_free(splitted_img[i]); 
  } 
  free(splitted_img); 
  return sum_avg_error; 
}
