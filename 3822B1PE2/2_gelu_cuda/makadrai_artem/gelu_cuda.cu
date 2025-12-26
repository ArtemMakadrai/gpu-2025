#include "gelu_cuda.h"

__global__ void gelu_kernel(const float *input, float *output, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = input[idx];
    float x3 = x * x * x;

    constexpr float kalf = 0.044715f;
    constexpr float k = 0.7978845608028654f;

    float t = k * (x + kalf * x3);
    float exp_term = expf(-2.0f * t);
    float tanh_approx = 2.0f / (1.0f + exp_term) - 1.0f;

    output[idx] = 0.5f * x * (1.0f + tanh_approx);
  }
}

std::vector<float> GeluCUDA(const std::vector<float> &input) {
  size_t n = input.size();
  size_t bytes = n * sizeof(float);

  float *d_input, *d_output;

  cudaMalloc(&d_input, bytes);
  cudaMalloc(&d_output, bytes);

  cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

  int bsize = 256;
  int gsize = (n + bsize - 1) / bsize;

  gelu_kernel<<<gsize, bsize>>>(d_input, d_output, n);

  std::vector<float> output(n);
  cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);

  return output;
}