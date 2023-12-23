#include "neural_network.h"
#include "utils.h"
#include <time.h>
#include "image.h"
#include <string.h>

encoded_image_t encoded_image_init(size_t block_width, size_t block_height, size_t width, size_t height, gsl_matrix** data) {
  return (encoded_image_t){
    .block_width = block_width,
    .block_height = block_height,
    .width = width,
    .height = height,
    .data = data
  };
}

void encoded_image_free(encoded_image_t enc) {
  size_t num_of_parts = ceil(enc.width/(double)enc.block_width) * ceil(enc.height/(double)enc.block_height);
  for (size_t i = 0; i < num_of_parts; ++i) {
    gsl_matrix_free(enc.data[i]);
  }
  free(enc.data);
}

neural_network_t neural_network_init(size_t block_width, size_t block_height, size_t compression) {
  return (neural_network_t){
    .block_width = 3*block_width,
    .block_height = block_height,
    .compression = compression,
    .encode_weights = gsl_matrix_calloc(3*block_height*block_width, 3*block_height*block_width/compression),
    .decode_weights = gsl_matrix_calloc(3*block_height*block_width/compression, 3*block_height*block_width)
  };
}

void neural_network_free(neural_network_t nn) {
  gsl_matrix_free(nn.encode_weights);
  gsl_matrix_free(nn.decode_weights);
}

neural_network_t generate_weights(neural_network_t nn) {
  srand(time(0));
  size_t rows = nn.block_height*nn.block_width;
  size_t cols = nn.block_height*nn.block_width/nn.compression;
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      gsl_matrix_set(nn.encode_weights, row, col, rand() % 3 - 1);
    }
  }
  gsl_matrix_transpose_memcpy(nn.decode_weights, nn.encode_weights);
  return nn;
}

neural_network_t load_model(char* path) {
  size_t block_width, block_height, compression;
  FILE* fp = fopen(path, "rb");
  fread(&block_width, sizeof(size_t), 1, fp);
  fread(&block_height, sizeof(size_t), 1, fp);
  fread(&compression, sizeof(size_t), 1, fp);
  neural_network_t nn = neural_network_init(block_width/3, block_height, compression);
  gsl_matrix_fread(fp, nn.encode_weights);
  gsl_matrix_fread(fp, nn.decode_weights);
  fclose(fp);
  return nn;
}

void save_model(neural_network_t nn, char* path) {
  FILE* fp = fopen(path, "wb");
  fwrite(&nn.block_width, sizeof(size_t), 1, fp);
  fwrite(&nn.block_height, sizeof(size_t), 1, fp);
  fwrite(&nn.compression, sizeof(size_t), 1, fp);
  gsl_matrix_fwrite(fp, nn.encode_weights);
  gsl_matrix_fwrite(fp, nn.decode_weights);
  fclose(fp);
}

encoded_image_t encode(neural_network_t nn, image_t img) {
  image_normalize_colors(img);
  size_t num_of_parts = get_num_of_parts_splitted(img.data, nn.block_height, nn.block_width);
  splitted_image_t splitted_img = split_image(img, nn.block_width, nn.block_height);
  gsl_matrix** out_mtx = calloc(num_of_parts, sizeof(gsl_matrix*));
  for (size_t i = 0; i < num_of_parts; ++i) {
    gsl_matrix transformed =
      gsl_matrix_view_array(splitted_img.data[i].data, 1, nn.block_height*nn.block_width).matrix;
    out_mtx[i] = gsl_matrix_calloc(1, nn.encode_weights->size2);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &transformed, nn.encode_weights, 0.0, out_mtx[i]);
  }
  splitted_image_free(splitted_img);
  image_denormalize_colors(img);
  return encoded_image_init(nn.block_width, nn.block_height, img.width, img.height, out_mtx);
}

image_t decode(neural_network_t nn, encoded_image_t enc) {
  size_t num_of_v_parts = ceil(enc.height/(double)enc.block_height);
  size_t num_of_h_parts = ceil(enc.width/(double)enc.block_width);
  size_t num_of_parts = num_of_h_parts * num_of_v_parts;
  splitted_image_t out = (splitted_image_t){
    .block_width = enc.block_width,
    .block_height = enc.block_height,
    .width = enc.width,
    .height = enc.height,
    .data = calloc(num_of_parts, sizeof(gsl_matrix))
  };
  gsl_matrix** out_mtx = calloc(num_of_parts, sizeof(gsl_matrix*));
  for (size_t i = 0; i < num_of_parts; ++i) {
    out_mtx[i] = gsl_matrix_calloc(1, nn.decode_weights->size2);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, enc.data[i], nn.decode_weights, 0.0, out_mtx[i]);
    out.data[i] = gsl_matrix_view_array(out_mtx[i]->data, nn.block_height, nn.block_width).matrix;
  }
  image_t new_img = unite_image(out);
  image_denormalize_colors(new_img);
  for (size_t i = 0; i < num_of_parts; ++i) {
    gsl_matrix_free(out_mtx[i]);
  }
  free(out_mtx);
  splitted_image_free(out);
  return new_img;
}

gsl_matrix* modify_weights(neural_network_t nn, double alpha, gsl_matrix* in_vector) {
  gsl_matrix* encoded = gsl_matrix_calloc(1, nn.encode_weights->size2);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, in_vector, nn.encode_weights, 0.0, encoded);
  gsl_matrix* decoded = gsl_matrix_calloc(1, nn.decode_weights->size2);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, encoded, nn.decode_weights, 0.0, decoded);
  gsl_matrix_sub(decoded, in_vector); // now decoded is an error matrix

  gsl_matrix* diff = gsl_matrix_calloc(nn.encode_weights->size2, decoded->size2);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, alpha, encoded, decoded, 0.0, diff);
  gsl_matrix_sub(nn.decode_weights, diff);
  gsl_matrix_free(diff);

  gsl_matrix* tmp = gsl_matrix_calloc(in_vector->size2, decoded->size2);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, alpha, in_vector, decoded, 0.0, tmp);
  gsl_matrix* diff_1 = gsl_matrix_calloc(tmp->size1, nn.decode_weights->size1);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, tmp, nn.decode_weights, 0.0, diff_1);
  gsl_matrix_sub(nn.encode_weights, diff_1);
  gsl_matrix_free(diff_1);
  
  gsl_matrix_free(encoded);
  return decoded;
}

double avg_error(gsl_matrix* error) {
  double avg = 0;
  for (size_t col = 0; col < error->size2; ++col) {
    avg += pow(gsl_matrix_get(error, 0, col), 2.0);
  }
  return avg/pow(error->size2, 2.0);
}

double train(neural_network_t nn, double alpha, gsl_matrix** data, size_t data_size) {
  double sum_avg_error = 0;
  gsl_matrix* encoded = gsl_matrix_calloc(1, nn.encode_weights->size2);
  gsl_matrix* decoded = gsl_matrix_calloc(1, nn.decode_weights->size2);
  gsl_matrix* diff = gsl_matrix_calloc(nn.encode_weights->size2, decoded->size2);
  gsl_matrix* tmp = gsl_matrix_calloc(data[0]->size2, decoded->size2);
  gsl_matrix* diff_1 = gsl_matrix_calloc(tmp->size1, nn.decode_weights->size1);
  for (size_t part = 0; part < data_size; ++part) {
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, data[part], nn.encode_weights, 0.0, encoded);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, encoded, nn.decode_weights, 0.0, decoded);
    gsl_matrix_sub(decoded, data[part]); // now decoded is an error matrix
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, alpha, encoded, decoded, 0.0, diff);
    gsl_matrix_sub(nn.decode_weights, diff);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, alpha, data[part], decoded, 0.0, tmp);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, tmp, nn.decode_weights, 0.0, diff_1);
    gsl_matrix_sub(nn.encode_weights, diff_1);
    sum_avg_error += avg_error(decoded);
  }
  gsl_matrix_free(diff_1);
  gsl_matrix_free(tmp);
  gsl_matrix_free(diff);
  gsl_matrix_free(decoded);
  gsl_matrix_free(encoded);
  return sum_avg_error;
}
