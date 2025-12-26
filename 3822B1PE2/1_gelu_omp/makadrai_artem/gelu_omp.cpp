#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float> &input) {
  std::vector<float> output(input.size());

  constexpr float kalf = 0.044715f;
  constexpr float k = 0.7978845608028654f;

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < input.size(); ++i) {
    float x = input[i];
    float x3 = x * x * x;

    float t = k * (x + kalf * x3);

    float exp_term = expf(-2.0f * t);
    float tanh_approx = 2.0f / (1.0f + exp_term) - 1.0f;

    output[i] = 0.5f * x * (1.0f + tanh_approx);
  }

  return output;
}