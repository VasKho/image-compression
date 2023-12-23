#include <gsl/gsl_blas.h>
#include "utils.h"
#include "training_pipeline.h"
#include "neural_network.h"
#include "image.h"

size_t n = 8;
size_t m = 8;
size_t Z = 3;
double err_threshhold = 1;

int main(int argc, char* argv[]) {
  /* training_pipeline_params_t params = init_taining_params(n, m, 20000, err_threshhold, Z); */
  /* size_t data_size = 20000; */
  /* gsl_matrix** data = load_training_data("./pngs", params, &data_size); */
  /* /\* neural_network_t nn = neural_network_init(n, m, Z); *\/ */
  /* /\* nn = generate_weights(nn); *\/ */

  /* neural_network_t nn = load_model("./weights_denorm.model"); */
  
  /* double err = 100; */
  /* while (err > err_threshhold) { */
  /*   err = train(nn, 0.00005, data, data_size); */
  /*   printf("%f\n", err); */
  /* } */
  /* save_model(nn, "./weights_denorm.model"); */
  /* free_training_data(data, data_size); */
  
  neural_network_t nn = load_model("./weights_denorm.model");
  image_t img = image_read_from_png("./pngs/1.png");
  encoded_image_t enc = encode(nn, img);
  image_t img1 = decode(nn, enc);
  image_write_to_png("./tmp.png", img1);
  encoded_image_free(enc);
  image_free(img);
  image_free(img1);

  neural_network_free(nn);
  return 0;
}
