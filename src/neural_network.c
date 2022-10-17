#include "neural_network.h"

double decode_alpha = 5e-6;

matrix* generate_weights(size_t rows, size_t cols) {
  srand(time(0));
  matrix* weights = matrix_create(rows, cols);
  for (size_t row = 0; row < weights->rows; ++row) {
    for (size_t col = 0; col < weights->cols; ++col) {
      matrix_set_element(weights, row, col, rand() % 3 - 1);
    }
  }
  return weights;
}

size_t get_num_of_h_splitted(matrix* A, size_t cols) {
  return ceil(A->cols/(double)cols);
}

size_t get_num_of_v_splitted(matrix* A, size_t rows) {
  return ceil(A->rows/(double)rows);
}

size_t get_num_of_parts_splitted(matrix* A, size_t rows, size_t cols) {
  return ceil(A->cols/(double)cols) * ceil(A->rows/(double)rows);
}

matrix** split_image(matrix* img, size_t rows, size_t cols) {
  cols *= 3;
  size_t num_of_parts = get_num_of_parts_splitted(img, rows, cols);
  size_t part = 0;
  matrix** result = calloc(num_of_parts, sizeof(matrix*));
  for (size_t row_ind = 0; row_ind < img->rows; row_ind += rows) { 
    if (row_ind + rows >= img->rows) { 
      row_ind = img->rows - row_ind + rows;
      for (size_t col_ind = 0; col_ind < img->cols; col_ind += cols) {
	result[part] = matrix_create(rows, cols);
	if (col_ind + cols >= img->cols) { 
	  col_ind = img->cols - col_ind + cols;
	  matrix_copy(result[part++], img, 0, 0, row_ind, col_ind, rows, cols);
	  break;
	}
	matrix_copy(result[part++], img, 0, 0, row_ind, col_ind, rows, cols);
      }
      break;
    }
    for (size_t col_ind = 0; col_ind < img->cols; col_ind += cols) {
      result[part] = matrix_create(rows, cols);
      if (col_ind + cols >= img->cols) { 
	col_ind = img->cols - col_ind + cols;
	matrix_copy(result[part++], img, 0, 0, row_ind, col_ind, rows, cols);
	break;
      }
      matrix_copy(result[part++], img, 0, 0, row_ind, col_ind, rows, cols);
    }
  } 
  return result; 
}

matrix* unite_image(matrix** arr, size_t rows, size_t cols) {
  cols *= 3;
  size_t part = 0;
  size_t d_row = arr[0]->rows;
  size_t d_col = arr[0]->cols;
  matrix* result = matrix_create(rows, cols);
  for (size_t row_ind = 0; row_ind < result->rows; row_ind += d_row) {
    if (row_ind + d_row >= result->rows) {
      row_ind = result->rows - row_ind + d_row;
      for (size_t col_ind = 0; col_ind < result->cols; col_ind += d_col) {
	if (col_ind + d_col >= result->cols) {
	col_ind = result->cols - col_ind + d_col;
	matrix_copy(result, arr[part++], row_ind, col_ind, 0, 0, d_row, d_col);
	break;
      }
      matrix_copy(result, arr[part++], row_ind, col_ind, 0, 0, d_row, d_col);
      }
      break;
    }
    for (size_t col_ind = 0; col_ind < result->cols; col_ind += d_col) {
      if (col_ind + d_col >= result->cols) {
	col_ind = result->cols - col_ind + d_col;
	matrix_copy(result, arr[part++], row_ind, col_ind, 0, 0, d_row, d_col);
	break;
      }
      matrix_copy(result, arr[part++], row_ind, col_ind, 0, 0, d_row, d_col);
    }
  }
  return result;
}

matrix** encode(matrix* img, matrix* weights, size_t rows, size_t cols) {
  cols *= 3;
  size_t num_of_parts = get_num_of_parts_splitted(img, rows, cols);
  size_t num_of_v_parts = get_num_of_v_splitted(img, rows);
  size_t num_of_h_parts = get_num_of_h_splitted(img, cols);
  matrix** splitted_img = split_image(img, rows, cols/3);
  matrix** out_mtx = calloc(num_of_parts, sizeof(matrix*));
  for (size_t i = 0; i < num_of_parts; ++i) {
    matrix* transformed = matrix_reshape(splitted_img[i], 1, rows*cols);
    out_mtx[i] = matrix_multiply(transformed, weights);
    matrix_destroy(splitted_img[i]);
    matrix_destroy(transformed);
  }
  free(splitted_img);
  return out_mtx;
}

matrix* decode(matrix** compressed, matrix* img, matrix* weights, size_t rows, size_t cols) {
  cols *= 3;
  size_t num_of_parts = get_num_of_parts_splitted(img, rows, cols);
  size_t num_of_v_parts = get_num_of_v_splitted(img, rows);
  size_t num_of_h_parts = get_num_of_h_splitted(img, cols);
  matrix** out_mtx = calloc(num_of_parts, sizeof(matrix*));
  for (size_t i = 0; i < num_of_parts; ++i) {
    matrix* temp_out = matrix_multiply(compressed[i], weights);
    out_mtx[i] = matrix_reshape(temp_out, rows, cols);
    matrix_destroy(temp_out);
  }
  matrix* new_img = unite_image(out_mtx, num_of_v_parts*rows, num_of_h_parts*cols/3);
  for (size_t i = 0; i < num_of_parts; ++i) {
    matrix_destroy(out_mtx[i]);
  }
  free(out_mtx);
  return new_img;
}

void modify_output_weights(matrix* decode_weights, double alpha, matrix* output_transposed, matrix* error) {
  matrix* mul_1 = matrix_multiply(output_transposed, error);
  matrix* mul_2 = matrix_multiply_scalar(alpha, mul_1);
  matrix* result = matrix_substitute(decode_weights, mul_2);
  matrix_copy(decode_weights, result, 0, 0, 0, 0, result->rows, result->cols);
  matrix_destroy(mul_1);
  matrix_destroy(mul_2);
  matrix_destroy(result);
}

matrix* modify_input_weights(matrix* encode_weights, matrix* decode_weights, double alpha, matrix* error, matrix* input) {
  matrix* input_transposed = matrix_transpose(input);
  matrix* decode_transposed = matrix_transpose(decode_weights);
  matrix* mul_1 = matrix_multiply(input_transposed, error);
  matrix* mul_2 = matrix_multiply(mul_1, decode_transposed);
  matrix* mul_3 = matrix_multiply_scalar(decode_alpha, mul_2);
  matrix* result = matrix_substitute(encode_weights, mul_3);
  matrix_copy(encode_weights, result, 0, 0, 0, 0, result->rows, result->cols);
  matrix_destroy(input_transposed);
  matrix_destroy(decode_transposed);
  matrix_destroy(mul_1);
  matrix_destroy(mul_2);
  matrix_destroy(mul_3);
  matrix_destroy(result);
}

matrix* modify_weights(matrix* img, matrix* encode_weights, matrix* decode_weights) {
  matrix* transformed = matrix_reshape(img, 1, img->rows*img->cols);
  matrix* encoded = matrix_multiply(transformed, encode_weights);
  matrix* decoded = matrix_multiply(encoded, decode_weights);
  matrix* error = matrix_substitute(decoded, transformed);
  matrix* output_transposed = matrix_transpose(encoded);
  matrix* out_mul = matrix_multiply(encoded, output_transposed);
  modify_output_weights(decode_weights, decode_alpha, output_transposed, error);
  modify_input_weights(encode_weights, decode_weights, decode_alpha, error, transformed);
  matrix_destroy(transformed);
  matrix_destroy(encoded);
  matrix_destroy(decoded);
  matrix_destroy(output_transposed);
  matrix_destroy(out_mul);
  return error;
}

double avg_error(matrix* error) {
  double avg = 0;
  for (size_t row = 0; row < error->rows; ++row) {
    for (size_t col = 0; col < error->cols; ++col) {
      avg += error->content[row][col] * error->content[row][col];
    }
  }
  return avg;
}

double train(matrix* img, matrix* encode_weights, matrix* decode_weights, size_t n, size_t m) {
  size_t num_of_parts = get_num_of_parts_splitted(img, n, m*3);
  matrix** splitted_img = split_image(img, n, m);
  double sum_avg_error = 0;
  for (size_t part = 0; part < num_of_parts; ++part) {
    matrix* error = modify_weights(splitted_img[part], encode_weights, decode_weights);
    sum_avg_error += avg_error(error);
    matrix_destroy(error);
  }
  for (size_t i = 0; i < num_of_parts; ++i) {
    matrix_destroy(splitted_img[i]);
  }
  free(splitted_img);
  return sum_avg_error;
}
