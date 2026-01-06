#include "conv_cuda.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <nvToolsExt.h> 

#define CUDA_CHECK(call) \
    do { \
        cudaError_t e = (call); \
        if (e != cudaSuccess) { \
            throw std::runtime_error(cudaGetErrorString(e)); \
        } \
    } while (0)

__device__ __forceinline__
int clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ================== Kernels ==================

__global__
void conv2d_naive_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    const float* __restrict__ kernel,
    int H, int W, int K
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    int r = K / 2;
    float sum = 0.0f;

    for (int ky = -r; ky <= r; ++ky) {
        int yy = clamp(y + ky, 0, H - 1);
        for (int kx = -r; kx <= r; ++kx) {
            int xx = clamp(x + kx, 0, W - 1);
            float w = kernel[(ky + r) * K + (kx + r)];
            sum += w * in[yy * W + xx];
        }
    }
    out[y * W + x] = sum;
}

template<int K, int BLOCK>
__global__
void conv2d_tiled_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    const float* __restrict__ kernel,
    int H, int W
) {
    constexpr int R = K / 2;
    constexpr int TILE_W = BLOCK + 2 * R;
    constexpr int TILE_H = BLOCK + 2 * R;

    __shared__ float tile[TILE_H][TILE_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x0 = blockIdx.x * BLOCK;
    int y0 = blockIdx.y * BLOCK;
    int gx = x0 + tx - R;
    int gy = y0 + ty - R;

    gx = clamp(gx, 0, W - 1);
    gy = clamp(gy, 0, H - 1);

    if (ty < TILE_H && tx < TILE_W) {
        tile[ty][tx] = in[gy * W + gx];
    }
    __syncthreads();

    int out_x = x0 + tx;
    int out_y = y0 + ty;

    if (tx < BLOCK && ty < BLOCK && out_x < W && out_y < H) {
        float sum = 0.0f;
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                float w = kernel[ky * K + kx];
                float v = tile[ty + ky][tx + kx];
                sum += w * v;
            }
        }
        out[out_y * W + out_x] = sum;
    }
}

// ================== Host Functions ==================

void conv2d_cuda_naive(
    const std::vector<float>& in,
    std::vector<float>& out,
    const std::vector<float>& kernel_h,
    int H, int W, int K,
    float* kernel_ms 
) {
    nvtxRangePushA("GPU Naive Total"); // NVTX tag for total time

    assert((int)in.size() == H * W);
    assert((int)kernel_h.size() == K * K);
    out.assign(H * W, 0.0f);

    size_t img_bytes = H * W * sizeof(float);
    size_t ker_bytes = K * K * sizeof(float);

    float *d_in = nullptr, *d_out = nullptr, *d_kernel = nullptr;

    nvtxRangePushA("Malloc");
    CUDA_CHECK(cudaMalloc(&d_in, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, ker_bytes));
    nvtxRangePop();

    nvtxRangePushA("H2D Copy");
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), img_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel_h.data(), ker_bytes, cudaMemcpyHostToDevice));
    nvtxRangePop();

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    // ---- Setup Events ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---- Kernel Execution ----
    nvtxRangePushA("Naive Kernel Compute"); // NVTX: compute segment
    cudaEventRecord(start);                 // Event: record start time
    
    conv2d_naive_kernel<<<grid, block>>>(d_in, d_out, d_kernel, H, W, K);
    
    cudaEventRecord(stop);                  // Event: end time
    CUDA_CHECK(cudaGetLastError());
    cudaEventSynchronize(stop);             // synchronize first
    nvtxRangePop();                         // NVTX: end compute segment

    // Calculate Time
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    if (kernel_ms) *kernel_ms = ms;         // store time if pointer provided

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    nvtxRangePushA("D2H Copy");
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, img_bytes, cudaMemcpyDeviceToHost));
    nvtxRangePop();

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);
    
    nvtxRangePop(); // End Total
}

void conv2d_cuda_tiled(
    const std::vector<float>& in,
    std::vector<float>& out,
    const std::vector<float>& kernel_h,
    int H, int W, int K,
    float* kernel_ms
) {
    nvtxRangePushA("GPU Tiled Total");

    assert(K == 7); 
    out.assign(H * W, 0.0f);

    size_t img_bytes = H * W * sizeof(float);
    size_t ker_bytes = K * K * sizeof(float);

    float *d_in = nullptr, *d_out = nullptr, *d_kernel = nullptr;

    nvtxRangePushA("Malloc");
    CUDA_CHECK(cudaMalloc(&d_in, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, ker_bytes));
    nvtxRangePop();

    nvtxRangePushA("H2D Copy");
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), img_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel_h.data(), ker_bytes, cudaMemcpyHostToDevice));
    nvtxRangePop();

    constexpr int BLOCK = 16;
    dim3 block(BLOCK + K - 1, BLOCK + K - 1);
    dim3 grid((W + BLOCK - 1) / BLOCK, (H + BLOCK - 1) / BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---- Kernel Execution ----
    nvtxRangePushA("Tiled Kernel Compute");
    cudaEventRecord(start);

    conv2d_tiled_kernel<7, BLOCK><<<grid, block>>>(d_in, d_out, d_kernel, H, W);
    
    cudaEventRecord(stop);
    CUDA_CHECK(cudaGetLastError());
    cudaEventSynchronize(stop);
    nvtxRangePop();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    if (kernel_ms) *kernel_ms = ms;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    nvtxRangePushA("D2H Copy");
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, img_bytes, cudaMemcpyDeviceToHost));
    nvtxRangePop();

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);

    nvtxRangePop(); // End Total
}