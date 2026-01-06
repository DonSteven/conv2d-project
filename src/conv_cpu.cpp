#include "conv_cpu.hpp"
#include <algorithm>
#include <cassert>

void conv2d_cpu(
    const std::vector<float>& in,
    std::vector<float>& out,
    const std::vector<float>& kernel,
    int H, int W, int K
) {
    assert((int)in.size() == H * W); 
    out.assign(H * W, 0.0f); // initialize output to zeros
    assert((int)kernel.size() == K * K); 

    int r = K / 2;  // kernel radius

    for (int y = 0; y < H; ++y) { // iterate over input image rows
        for (int x = 0; x < W; ++x) { // iterate over input image columns
            float sum = 0.0f;
            for (int ky = -r; ky <= r; ++ky) { // iterate over kernel rows
                int yy = std::min(std::max(y + ky, 0), H - 1); // 0 <= yy < H
                for (int kx = -r; kx <= r; ++kx) {
                    int xx = std::min(std::max(x + kx, 0), W - 1);
                    float w = kernel[(ky + r) * K + (kx + r)]; // kernel weight at (ky, kx)
                    sum += w * in[yy * W + xx]; // accumulate convolution result
                }
            }
            out[y * W + x] = sum;
        }
    }
}
