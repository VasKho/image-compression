#include "interface.h"

int save_weights(char* path, gsl_matrix* src_encode, gsl_matrix* src_decode, network_params params) {
  FILE* fp = fopen(path, "w+b");
  fwrite(&params, sizeof(network_params), 1, fp);
  gsl_matrix_fwrite(fp, src_encode);
  gsl_matrix_fwrite(fp, src_decode);
  fclose(fp);
  return 0;
}

void load_weights(char* path, gsl_matrix** dest_encode, gsl_matrix** dest_decode, network_params* params) {
  FILE* fp = fopen(path, "rb");
  fread(params, sizeof(network_params), 1, fp);
  size_t rows = 3*params->block_rows*params->block_cols;
  size_t cols = rows/params->compression;
  if (dest_encode == NULL) {
    gsl_matrix* temp = gsl_matrix_alloc(rows, cols);
    gsl_matrix_fread(fp, temp);
    gsl_matrix_free(temp);
  } else {
    *dest_encode = gsl_matrix_alloc(rows, cols);
    gsl_matrix_fread(fp, *dest_encode);
  }
  if (dest_decode == NULL) {
    gsl_matrix* temp = gsl_matrix_alloc(cols, rows);
    gsl_matrix_fread(fp, temp);
    gsl_matrix_free(temp);
  } else {
    *dest_decode = gsl_matrix_alloc(cols, rows);
    gsl_matrix_fread(fp, *dest_decode);
  }  
  fclose(fp);
}

int action_generate(char* path_to_save, network_params params) {
  size_t rows = 3*params.block_rows*params.block_cols;
  size_t cols = rows/params.compression;
  gsl_matrix* weights = generate_weights(rows, cols);
  for (size_t row = 0; row < weights->size1; ++row) {
    for (size_t col = 0; col < weights->size2; ++col) {
      double elem = gsl_matrix_get(weights, row, col);
      gsl_matrix_set(weights, row, col, elem/cols);
    }
  }
  gsl_matrix* weightsT = gsl_matrix_alloc(weights->size2, weights->size1);
  gsl_matrix_transpose_memcpy(weightsT, weights);
  save_weights(path_to_save, weights, weightsT, params);
  gsl_matrix_free(weights);
  gsl_matrix_free(weightsT);
  return 0;
}

int action_compress(char* path_to_input, char* path_to_weights, char* path_to_output) {
  network_params params;
  gsl_matrix* image = png_read_to_matrix(path_to_input); 
  matrix_normalize_colors(image);
  gsl_matrix* weights;
  load_weights(path_to_weights, &weights, NULL, &params);
  gsl_matrix** encoded = encode(image, weights, params);
  FILE* f_out = fopen(path_to_output, "w+b");
  size_t num_of_parts = get_num_of_parts_splitted(image, params.block_rows, 3*params.block_cols);
  fwrite(&num_of_parts, sizeof(size_t), 1, f_out);
  fwrite(&image->size1, sizeof(size_t), 1, f_out);
  fwrite(&image->size2, sizeof(size_t), 1, f_out);
  fwrite(&encoded[0]->size1, sizeof(size_t), 1, f_out);
  fwrite(&encoded[0]->size2, sizeof(size_t), 1, f_out);
  for (size_t i = 0; i < num_of_parts; ++i) {
    gsl_matrix_fwrite(f_out, encoded[i]);
    gsl_matrix_free(encoded[i]);
  }
  fclose(f_out);
  free(encoded);
  gsl_matrix_free(image);
  gsl_matrix_free(weights);
  return 0; 
}

int action_decompress(char* path_to_input, char* path_to_weights, char* path_to_output) {
  network_params params;
  gsl_matrix* weights;
  load_weights(path_to_weights, NULL, &weights, &params);
  size_t num_of_parts, out_rows, out_cols, enc_rows, enc_cols;
  FILE* f_in = fopen(path_to_input, "rb");
  fread(&num_of_parts, sizeof(size_t), 1, f_in);
  fread(&out_rows, sizeof(size_t), 1, f_in);
  fread(&out_cols, sizeof(size_t), 1, f_in);
  fread(&enc_rows, sizeof(size_t), 1, f_in);
  fread(&enc_cols, sizeof(size_t), 1, f_in);
  gsl_matrix** compressed = calloc(num_of_parts, sizeof(gsl_matrix*));
  for (size_t i = 0; i < num_of_parts; ++i) {
    compressed[i] = gsl_matrix_alloc(enc_rows, enc_cols);
    gsl_matrix_fread(f_in, compressed[i]);
  }
  fclose(f_in); 
  gsl_matrix* decompressed = decode(compressed, weights, num_of_parts, out_rows, out_cols, params); 
  png_write_from_matrix(path_to_output, decompressed); 
  for (size_t i = 0; i < num_of_parts; ++i) { 
    gsl_matrix_free(compressed[i]); 
  } 
  free(compressed); 
  gsl_matrix_free(weights); 
  gsl_matrix_free(decompressed); 
  return 0; 
}

int action_train(char* path_to_images, char* path_to_weights, double error, double alpha) {
  network_params params;
  gsl_matrix* weights;
  gsl_matrix* weightsT; 
  load_weights(path_to_weights, &weights, &weightsT, &params);
  char command_prefix[] = "find "; 
  char command_postfix[] = " -type f -name '*.png'"; 
  char* command = calloc(strlen(command_prefix)+strlen(path_to_images)+strlen(command_postfix)+1, sizeof(char)); 
  strcat(command, command_prefix); 
  strcat(command, path_to_images); 
  strcat(command, command_postfix); 
  double sum_err = 0;
  do {
    sum_err = 0; 
    FILE* pipe = popen(command, "r");
    if (!pipe) {
      return -1; 
    } 
    char* path = calloc(2048, sizeof(char)); 
    char temp;
    while (!feof(pipe)) {
      temp = fgetc(pipe);
      if (temp != '\n') {
	strncat(path, &temp, 1);
      } else { 
	printf("%s\n", path); 
	gsl_matrix* image = png_read_to_matrix(path); 
	matrix_normalize_colors(image); 
	sum_err += train(image, weights, weightsT, params, alpha);
	save_weights(path_to_weights, weights, weightsT, params);
	printf("%lf\n", sum_err); 
	memset(path, 0, 2048);
	gsl_matrix_free(image); 
      } 
    } 
    pclose(pipe); 
    free(path); 
  } while (sum_err > error);
  gsl_matrix_free(weights); 
  gsl_matrix_free(weightsT); 
  free(command); 
  return 0; 
} 

int action_test(char* path_to_image, char* path_to_weights) {
  network_params params;
  gsl_matrix* image = png_read_to_matrix(path_to_image);
  size_t num_of_parts = get_num_of_parts_splitted(image, params.block_rows, 3*params.block_cols);
  matrix_normalize_colors(image);
  gsl_matrix* weights, *weightsT;
  load_weights(path_to_weights, &weights, &weightsT, &params);
  gsl_matrix** encoded = encode(image, weights, params);
  gsl_matrix* decoded = decode(encoded, weightsT, num_of_parts, image->size1, image->size2, params);
  matrix_denormalize_colors(decoded);
  png_write_from_matrix("out.png", decoded);
  gsl_matrix_free(image);
  gsl_matrix_free(weights);
  gsl_matrix_free(weightsT);
  gsl_matrix_free(decoded);
  for (size_t i = 0; i < num_of_parts; ++i) {
    gsl_matrix_free(encoded[i]);
  }
  free(encoded);
  return 0; 
}
