#include "conv_cpu.hpp"
#include "conv_omp.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <omp.h>

double compute_flops(int H, int W, int K) {
    return double(H) * double(W) * double(2 * K * K);
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::abs(double(a[i]) - double(b[i])));
    return m;
}

int main(int argc, char** argv) {
    int N = 4096;
    int K = 7;
    if (argc >= 2) N = std::stoi(argv[1]);

    int H = N, W = N;

    std::vector<float> in(H * W), out_cpu(H * W), out_omp(H * W);
    std::vector<float> kernel(K * K, 1.0f / (K * K));

    for (int i = 0; i < H * W; ++i)
        in[i] = float(i % 255) / 255.0f;

    // baseline: CPU single thread
    auto t0 = std::chrono::high_resolution_clock::now();
    conv2d_cpu(in, out_cpu, kernel, H, W, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double flops = compute_flops(H, W, K);
    double cpu_gflops = flops / (cpu_ms / 1000.0) / 1e9;

    std::cout << "===== OpenMP scaling, N = " << N << " (K=" << K << ") =====\n";
    std::cout << "CPU 1 thread: " << cpu_ms << " ms, "
              << cpu_gflops << " GFLOP/s\n\n";

    // modify threads list based on machine core count
    int thread_list[] = {1, 2, 4, 8, 16};

    for (int T : thread_list) {
        omp_set_num_threads(T);

        auto t2 = std::chrono::high_resolution_clock::now();
        conv2d_omp(in, out_omp, kernel, H, W, K);
        auto t3 = std::chrono::high_resolution_clock::now();
        double omp_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        double omp_gflops = flops / (omp_ms / 1000.0) / 1e9;
        double speedup = cpu_ms / omp_ms;

        double diff = max_abs_diff(out_cpu, out_omp);

        std::cout << "threads = " << T
                  << " : " << omp_ms << " ms, "
                  << omp_gflops << " GFLOP/s, "
                  << "speedup = " << speedup << "x, "
                  << "max diff = " << diff << "\n";

        std::cout << "CSV," << N << "," << T << ","
                  << omp_ms << "," << omp_gflops << "," << speedup << "\n";
    }

    return 0;
}

// ./conv_omp_experiment 4096
// ===== OpenMP scaling, N = 4096 (K=7) =====
// CPU 1 thread: 887.265 ms, 1.85307 GFLOP/s

// threads = 1 : 897.076 ms, 1.83281 GFLOP/s, speedup = 0.989064x, max diff = 0
// CSV,4096,1,897.076,1.83281,0.989064
// threads = 2 : 455.127 ms, 3.61255 GFLOP/s, speedup = 1.94949x, max diff = 0
// CSV,4096,2,455.127,3.61255,1.94949
// threads = 4 : 232.785 ms, 7.06302 GFLOP/s, speedup = 3.81152x, max diff = 0
// CSV,4096,4,232.785,7.06302,3.81152
// threads = 8 : 241.349 ms, 6.8124 GFLOP/s, speedup = 3.67627x, max diff = 0
// CSV,4096,8,241.349,6.8124,3.67627
// threads = 16 : 238.47 ms, 6.89465 GFLOP/s, speedup = 3.72066x, max diff = 0
// CSV,4096,16,238.47,6.89465,3.72066