#include "conv_cpu.hpp"
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    int H = 1024, W = 1024, K = 7;
    if (argc >= 3) {
        H = std::stoi(argv[1]);
        W = std::stoi(argv[2]);
    }

    std::vector<float> in(H * W), out(H * W);
    std::vector<float> kernel(K * K);

    // simple fill: input with [0,1) random, kernel with box blur / Laplace
    for (int i = 0; i < H * W; ++i) in[i] = float(i % 255) / 255.0f;
    for (int i = 0; i < K * K; ++i) kernel[i] = 1.0f / (K * K);

    auto t0 = std::chrono::high_resolution_clock::now();
    conv2d_cpu(in, out, kernel, H, W, K);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cerr << "CPU conv time = " << ms << " ms\n";

    // print a few points for sanity check
    std::cout << out[0] << " " << out[H/2 * W + W/2] << " " << out.back() << "\n";
    return 0;
}


// simple cpu: H = 1024, W = 1024: Cpu conv time = 55.3311ms
// 0.0168067 0.161665 0.0420168