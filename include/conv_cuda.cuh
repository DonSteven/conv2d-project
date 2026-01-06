#pragma once
#include <vector>

// host API：internal will call CUDA kernel
void conv2d_cuda_naive(
    const std::vector<float>& in,
    std::vector<float>& out,
    const std::vector<float>& kernel,
    int H, int W, int K,
    float* kernel_ms = nullptr
);

void conv2d_cuda_tiled(
    const std::vector<float>& in,
    std::vector<float>& out,
    const std::vector<float>& kernel_h,
    int H, int W, int K,
    float* kernel_ms = nullptr
);