#pragma once
#include <vector>

void conv2d_omp(
    const std::vector<float>& in,
    std::vector<float>& out,
    const std::vector<float>& kernel,
    int H, int W, int K
);
