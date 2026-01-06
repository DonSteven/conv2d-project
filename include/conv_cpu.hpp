#pragma once
#include <vector>

// simple 2D convolution api
void conv2d_cpu(
    const std::vector<float>& in,
    std::vector<float>& out,
    const std::vector<float>& kernel,
    int H, int W, int K
);
