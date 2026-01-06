#include "conv_cpu.hpp"
#include "conv_cuda.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip> // for std::setw

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::abs(double(a[i]) - double(b[i])));
    return m;
}

double compute_flops(int H, int W, int K) {
    return double(H) * double(W) * double(2 * K * K);
}

void run_one_size(int N, int K) {
    int H = N, W = N;

    std::vector<float> in(H * W), out_cpu(H * W),
                       out_gpu_naive(H * W), out_gpu_tiled(H * W);
    std::vector<float> kernel(K * K, 1.0f / (K * K));

    for (int i = 0; i < H * W; ++i) in[i] = float(i % 255) / 255.0f;

    // ==== 1. Warm-up ====
    // eliminate CUDA Context initialization cold start delay
    std::vector<float> dummy_out(H * W);
    conv2d_cuda_naive(in, dummy_out, kernel, H, W, K, nullptr);

    // ==== 2. CPU Benchmark ====
    auto t0 = std::chrono::high_resolution_clock::now();
    conv2d_cpu(in, out_cpu, kernel, H, W, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ==== 3. GPU Naive Benchmark ====
    float naive_kernel_ms = 0.0f; // receive kernel time
    auto t2 = std::chrono::high_resolution_clock::now();
    conv2d_cuda_naive(in, out_gpu_naive, kernel, H, W, K, &naive_kernel_ms);
    auto t3 = std::chrono::high_resolution_clock::now();
    double gpu_naive_total_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // ==== 4. GPU Tiled Benchmark ====
    float tiled_kernel_ms = 0.0f; 
    auto t4 = std::chrono::high_resolution_clock::now();
    conv2d_cuda_tiled(in, out_gpu_tiled, kernel, H, W, K, &tiled_kernel_ms);
    auto t5 = std::chrono::high_resolution_clock::now();
    double gpu_tiled_total_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    // ==== 5. check correctness ====
    double diff_naive = max_abs_diff(out_cpu, out_gpu_naive);
    double diff_tiled = max_abs_diff(out_cpu, out_gpu_tiled);

    double flops = compute_flops(H, W, K);
    double cpu_gflops = flops / (cpu_ms / 1000.0) / 1e9;
    
    // notice: compute GPU GFLOP/s using kernel time
    double naive_gflops = flops / (naive_kernel_ms / 1000.0) / 1e9;
    double tiled_gflops = flops / (tiled_kernel_ms / 1000.0) / 1e9;

    std::cout << "===== N = " << N << " (K=" << K << ") =====\n";
    std::cout << "CPU:        " << cpu_ms << " ms, " << cpu_gflops << " GFLOP/s\n";
    std::cout << "GPU Naive:  Total " << gpu_naive_total_ms << " ms, Kernel " << naive_kernel_ms << " ms, " << naive_gflops << " GFLOP/s\n";
    std::cout << "GPU Tiled:  Total " << gpu_tiled_total_ms << " ms, Kernel " << tiled_kernel_ms << " ms, " << tiled_gflops << " GFLOP/s\n";
    std::cout << "Check Diff: Naive=" << diff_naive << ", Tiled=" << diff_tiled << "\n";

    // CSV Output (Total Time, Kernel Time, Kernel GFLOPs)
    std::cout << "CSV," << N << ","
              << cpu_ms << "," << cpu_gflops << ","
              << gpu_naive_total_ms << "," << naive_kernel_ms << "," << naive_gflops << ","
              << gpu_tiled_total_ms << "," << tiled_kernel_ms << "," << tiled_gflops << "\n\n";
}

int main(int argc, char** argv) {
    int K = 7;
    if (argc >= 2) {
        int N = std::stoi(argv[1]);
        run_one_size(N, K);
    } else {
        int sizes[] = {1024, 2048, 4096, 8192};
        for (int N : sizes) run_one_size(N, K);
    }
    return 0;
}

// ===== N = 1024 (H=W=1024, K=7) =====
// CPU: 56.1289 ms, 1.83079 GFLOP/s
// GPU naive: total 3.6791 ms, kernel 0.125504 ms, 818.782 GFLOP/s
// GPU tiled: total 3.59347 ms, kernel 0.15888 ms, 646.78 GFLOP/s
// Max abs diff naive = 5.96046e-08, tiled = 5.96046e-08
// CSV,1024,56.1289,1.83079,3.6791,0.125504,818.782,3.59347,0.15888,646.78

// ===== N = 2048 (H=W=2048, K=7) =====
// CPU: 223.601 ms, 1.83828 GFLOP/s
// GPU naive: total 10.8808 ms, kernel 0.426336 ms, 964.126 GFLOP/s
// GPU tiled: total 10.7082 ms, kernel 0.43376 ms, 947.625 GFLOP/s
// Max abs diff naive = 1.19209e-07, tiled = 1.19209e-07
// CSV,2048,223.601,1.83828,10.8808,0.426336,964.126,10.7082,0.43376,947.625

// ===== N = 4096 (H=W=4096, K=7) =====
// CPU: 891.228 ms, 1.84483 GFLOP/s
// GPU naive: total 39.7844 ms, kernel 1.64659 ms, 998.527 GFLOP/s
// GPU tiled: total 39.4599 ms, kernel 1.67731 ms, 980.239 GFLOP/s
// Max abs diff naive = 1.19209e-07, tiled = 1.19209e-07
// CSV,4096,891.228,1.84483,39.7844,1.64659,998.527,39.4599,1.67731,980.239

// ===== N = 8192 (H=W=8192, K=7) =====
// CPU: 3545.71 ms, 1.85482 GFLOP/s
// GPU naive: total 146.3 ms, kernel 6.53107 ms, 1006.98 GFLOP/s
// GPU tiled: total 146.7 ms, kernel 6.67843 ms, 984.762 GFLOP/s
// Max abs diff naive = 5.96046e-08, tiled = 5.96046e-08
// CSV,8192,3545.71,1.85482,146.3,6.53107,1006.98,146.7,6.67843,984.762

// **** data before modification: increase warmup, reduce GPU naive kernel time, improve tiled GFLOP/s ****

// ===== N = 1024 (H=W=1024, K=7) =====
// CPU: 55.5659 ms, 1.84934 GFLOP/s
// GPU naive: total 157.391 ms, kernel 5.41379 ms, 18.9812 GFLOP/s
// GPU tiled: total 3.5834 ms, kernel 0.150784 ms, 681.508 GFLOP/s
// Max abs diff naive = 5.96046e-08, tiled = 5.96046e-08
// CSV,1024,55.5659,1.84934,157.391,5.41379,18.9812,3.5834,0.150784,681.508

// ===== N = 2048 (H=W=2048, K=7) =====
// CPU: 222.513 ms, 1.84727 GFLOP/s
// GPU naive: total 11.2557 ms, kernel 0.426336 ms, 964.126 GFLOP/s
// GPU tiled: total 11.1464 ms, kernel 0.434528 ms, 945.95 GFLOP/s
// Max abs diff naive = 1.19209e-07, tiled = 1.19209e-07
// CSV,2048,222.513,1.84727,11.2557,0.426336,964.126,11.1464,0.434528,945.95

// ===== N = 4096 (H=W=4096, K=7) =====
// CPU: 893.991 ms, 1.83913 GFLOP/s
// GPU naive: total 60.6786 ms, kernel 1.6512 ms, 995.741 GFLOP/s
// GPU tiled: total 49.666 ms, kernel 1.68141 ms, 977.851 GFLOP/s
// Max abs diff naive = 1.19209e-07, tiled = 1.19209e-07
// CSV,4096,893.991,1.83913,60.6786,1.6512,995.741,49.666,1.68141,977.851

// ===== N = 8192 (H=W=8192, K=7) =====
// CPU: 3635.19 ms, 1.80917 GFLOP/s
// GPU naive: total 233.453 ms, kernel 6.53738 ms, 1006.01 GFLOP/s
// GPU tiled: total 185.618 ms, kernel 6.68253 ms, 984.159 GFLOP/s
// Max abs diff naive = 5.96046e-08, tiled = 5.96046e-08
// CSV,8192,3635.19,1.80917,233.453,6.53738,1006.01,185.618,6.68253,984.159