#include "conv_omp.hpp"
#include <algorithm>
#include <cassert>
#include <omp.h>

void conv2d_omp(
    const std::vector<float>& in,
    std::vector<float>& out,
    const std::vector<float>& kernel,
    int H, int W, int K
) {
    assert((int)in.size() == H * W);
    out.assign(H * W, 0.0f);
    assert((int)kernel.size() == K * K);

    int r = K / 2;

    #pragma omp parallel for // parallelize over rows
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (int ky = -r; ky <= r; ++ky) {
                int yy = std::min(std::max(y + ky, 0), H - 1);
                for (int kx = -r; kx <= r; ++kx) {
                    int xx = std::min(std::max(x + kx, 0), W - 1);
                    float w = kernel[(ky + r) * K + (kx + r)];
                    sum += w * in[yy * W + xx];
                }
            }
            out[y * W + x] = sum;
        }
    }
}
