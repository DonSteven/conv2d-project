#include "conv_cpu.hpp"
#include "conv_cuda.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::abs(double(a[i]) - double(b[i])));
    return m;
}

int main(int argc, char** argv) {
    int H = 2048, W = 2048, K = 7;
    if (argc >= 3) { H = std::stoi(argv[1]); W = std::stoi(argv[2]); }

    std::vector<float> in(H * W), out_cpu(H * W), out_gpu(H * W), out_gpu_tiled(H * W);
    std::vector<float> kernel(K * K, 1.0f / (K * K));
    for (int i = 0; i < H * W; ++i) in[i] = float(i % 255) / 255.0f;

    std::vector<float> dummy_out(H*W);
    conv2d_cuda_naive(in, dummy_out, kernel, H, W, K, nullptr); 

    auto t0 = std::chrono::high_resolution_clock::now();
    conv2d_cpu(in, out_cpu, kernel, H, W, K);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto t2 = std::chrono::high_resolution_clock::now();
    conv2d_cuda_naive(in, out_gpu, kernel, H, W, K);
    auto t3 = std::chrono::high_resolution_clock::now();

    auto t4 = std::chrono::high_resolution_clock::now();
    conv2d_cuda_tiled(in, out_gpu_tiled, kernel, H, W, K);
    auto t5 = std::chrono::high_resolution_clock::now();

    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double gpu_tiled_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
    double diff = max_abs_diff(out_cpu, out_gpu);
    double diff_tiled = max_abs_diff(out_cpu, out_gpu_tiled);

    std::cout << "CPU ms: " << cpu_ms << "\n";
    std::cout << "GPU naive ms (H2D+D2H included): " << gpu_ms << "\n";
    std::cout << "Speedup (end-to-end): " << cpu_ms / gpu_ms << "x\n";
    std::cout << "Max abs diff: " << diff << "\n";
    std::cout << "------------" << "\n";
    std::cout << "GPU tiled ms (H2D+D2H included): " << gpu_tiled_ms << "\n";
    std::cout << "Speedup tiled (end-to-end): " << cpu_ms / gpu_tiled_ms << "x\n";
    std::cout << "Max abs diff tiled: " << diff_tiled << "\n";
    return 0;
}


// ./conv_gpu_naive 1024 1024
// CPU ms: 55.7092
// GPU naive ms (H2D+D2H included): 3.74823
// Speedup (end-to-end): 14.8628x
// Max abs diff: 5.96046e-08
// ------------
// GPU tiled ms (H2D+D2H included): 3.6317
// Speedup tiled (end-to-end): 15.3397x
// Max abs diff tiled: 5.96046e-08

// ./conv_gpu_naive 2048 2048
// CPU ms: 222.304
// GPU naive ms (H2D+D2H included): 11.1387
// Speedup (end-to-end): 19.9578x
// Max abs diff: 1.19209e-07
// ------------
// GPU tiled ms (H2D+D2H included): 11.165
// Speedup tiled (end-to-end): 19.9109x
// Max abs diff tiled: 1.19209e-07

// ./conv_gpu_naive 4096 4096
// CPU ms: 884.74
// GPU naive ms (H2D+D2H included): 38.8977
// Speedup (end-to-end): 22.7453x
// Max abs diff: 1.19209e-07
// ------------
// GPU tiled ms (H2D+D2H included): 38.8195
// Speedup tiled (end-to-end): 22.7911x
// Max abs diff tiled: 1.19209e-07

// ./conv_gpu_naive 8192 8192
// CPU ms: 3559.76
// GPU naive ms (H2D+D2H included): 150.057
// Speedup (end-to-end): 23.7227x
// Max abs diff: 5.96046e-08
// ------------
// GPU tiled ms (H2D+D2H included): 149.04
// Speedup tiled (end-to-end): 23.8845x
// Max abs diff tiled: 5.96046e-08

// --------------- data before modification ------------------ 

// ./conv_gpu_naive 1024 1024
// CPU ms: 58.5223
// GPU naive ms (H2D+D2H included): 262.808
// Speedup (end-to-end): 0.222681x
// Max abs diff: 5.96046e-08
// ------------
// GPU tiled ms (H2D+D2H included): 3.58211
// Speedup tiled (end-to-end): 73.3668x
// Max abs diff tiled: 5.96046e-08

//  ./conv_gpu_naive 2048 2048
// CPU ms: 233.443
// GPU naive ms (H2D+D2H included): 171.548
// Speedup (end-to-end): 1.3608x
// Max abs diff: 1.19209e-07
// ------------
// GPU tiled ms (H2D+D2H included): 11.0129
// Speedup tiled (end-to-end): 15.5771x
// Max abs diff tiled: 1.19209e-07

// ./conv_gpu_naive 4096 4096
// CPU ms: 929.33
// GPU naive ms (H2D+D2H included): 188.697
// Speedup (end-to-end): 4.92499x
// Max abs diff: 1.19209e-07
// ------------
// GPU tiled ms (H2D+D2H included): 38.3575
// Speedup tiled (end-to-end): 4.91943x
// Max abs diff tiled: 1.19209e-07

// ./conv_gpu_naive 8192 8192
// CPU ms: 3691.57
// GPU naive ms (H2D+D2H included): 296.892
// Speedup (end-to-end): 12.4341x
// Max abs diff: 5.96046e-08
// ------------
// GPU tiled ms (H2D+D2H included): 144.282
// Speedup tiled (end-to-end): 2.05772x
// Max abs diff tiled: 5.96046e-08