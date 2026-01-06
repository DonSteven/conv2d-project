// src/main_mpi.cpp
#include "conv_cpu.hpp"

#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono> 

// Compute the FLOPs of conv2d:
// For each output pixel, roughly 2 * K^2 floating-point ops (1 mul + 1 add per weight)
double compute_flops(int H, int W, int K) {
    return double(H) * double(W) * double(2 * K * K);
}

// Maximum absolute difference between two arrays (for correctness check)
double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::abs(double(a[i]) - double(b[i])));
    return m;
}

/*
    Perform convolution on the local region WITH halo padding.
    The input 'in' has dimensions (local_H + 2R) × W:

        Rows [0, R-1]               : top halo region
        Rows [R, R+local_H-1]       : the actual rows owned by this MPI rank
        Rows [R+local_H, ...]       : bottom halo region

    Only the middle local_H rows will be written to 'out'.
    This allows independent convolution without communication during compute.

    (Extra explanation)
    - Halo ensures each rank has the neighbor rows needed by the kernel.
    - Kernel indexing uses (ky + R) and (kx + R) because ky ∈ [-R, R].
    - yy = y + ky + R shifts local row y to the correct halo-offset row in 'in'.
*/
void conv2d_local_with_halo(
    const std::vector<float>& in, // size: (local_H + 2R) * W
    std::vector<float>& out, // size: local_H * W
    const std::vector<float>& kernel,
    int local_H,   // number of rows owned by this rank (excluding halos)
    int W,
    int K,
    int R
) {
    int ext_H = local_H + 2 * R;
    assert((int)in.size() == ext_H * W);
    out.assign(local_H * W, 0.0f);
    assert((int)kernel.size() == K * K);

    for (int y = 0; y < local_H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;

            for (int ky = -R; ky <= R; ++ky) {
                int yy = y + ky + R;   // shift by R to account for halo offset
                yy = std::max(0, std::min(yy, ext_H - 1)); // clamp inside extended region

                for (int kx = -R; kx <= R; ++kx) {
                    int xx = x + kx;
                    xx = std::max(0, std::min(xx, W - 1));

                    float w = kernel[(ky + R) * K + (kx + R)];
                    sum += w * in[yy * W + xx];
                }
            }

            out[y * W + x] = sum;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 4096;
    int K = 7;
    if (argc >= 2) N = std::stoi(argv[1]);
    if (argc >= 3) K = std::stoi(argv[2]);

    int H = N, W = N;
    int R = K / 2;

    // ----- 1. All ranks create the same kernel -----
    // No need to broadcast: kernel is small and deterministic.
    std::vector<float> kernel(K * K, 1.0f / (K * K));

    // ----- 2. Rank 0 creates the global input and computes sendcounts/displacements -----
    std::vector<float> global_in;
    std::vector<float> global_out;
    std::vector<float> out_cpu;

    std::vector<int> sendcounts, displs;

    if (rank == 0) {
        global_in.resize(H * W);

        // Generate a simple pattern for testing
        for (int i = 0; i < H * W; ++i)
            global_in[i] = float(i % 255) / 255.0f;

        global_out.resize(H * W);
        out_cpu.resize(H * W);

        sendcounts.resize(world_size);
        displs.resize(world_size);

        /*
            Compute the number of rows each rank receives.
            The first 'rem' ranks receive one extra row (classic block distribution).
        */
        int base_rows = H / world_size;
        int rem = H % world_size;

        int offset_rows = 0;
        for (int r = 0; r < world_size; ++r) {
            int rows = base_rows + (r < rem ? 1 : 0);
            sendcounts[r] = rows * W;      // number of float elements to send
            displs[r] = offset_rows * W;   // starting offset in global_in
            offset_rows += rows;
        }
    }

    // ----- 3. Each rank computes its local height and starting row -----
    int base_rows = H / world_size;
    int rem = H % world_size;
    int local_H = base_rows + (rank < rem ? 1 : 0);

    /*
        start_row describes where this rank's block starts in the global image.
        It is consistent with sendcounts/displs computed above.
    */
    int start_row = rank * base_rows + std::min(rank, rem);
    (void)start_row; // Only for documentation; not used later.

    int ext_H = local_H + 2 * R;  // height including halo rows

    std::vector<float> local_in(ext_H * W, 0.0f);
    std::vector<float> local_out(local_H * W);

    // ----- 4. Scatter the owned rows into the middle region of local_in -----
    float* sendbuf = nullptr;
    int* sendcounts_ptr = nullptr;
    int* displs_ptr = nullptr;
    if (rank == 0) {
        sendbuf = global_in.data();
        sendcounts_ptr = sendcounts.data();
        displs_ptr = displs.data();
    }

    /*
        MPI_Scatterv sends each rank exactly its contiguous block of rows.
        They are placed starting at local_in[R * W], leaving room for the top halo.

        local_in layout after Scatterv:
            top halo rows [0 .. R-1]        : uninitialized yet
            owned rows   [R .. R+local_H-1] : filled by Scatterv
            bottom halo  [R+local_H .. end] : uninitialized yet
    */
    MPI_Scatterv(
        sendbuf, sendcounts_ptr, displs_ptr, MPI_FLOAT,
        local_in.data() + R * W, local_H * W, MPI_FLOAT,
        0, MPI_COMM_WORLD
    );

    // ----- 5. Build halo regions by exchanging boundary rows with neighbors -----
    MPI_Status status;

    // ---- Top halo ----
    if (rank == 0) {
        /*
            No upper neighbor → perform edge replication.
            Copy the first owned row into each top-halo row.
        */
        for (int i = 0; i < R; ++i) {
            std::copy(
                local_in.begin() + R * W,
                local_in.begin() + (R + 1) * W,
                local_in.begin() + i * W
            );
        }
    } else {
        /*
            Exchange boundary:
            Send top R owned rows upward, receive bottom R rows from rank-1
            into the top halo region.

            Explanation:
            - send buffer: local_in + R*W (first owned row block)
            - receive buffer: local_in (start of halo region)
        */
        MPI_Sendrecv(
            local_in.data() + R * W,     R * W, MPI_FLOAT, rank - 1, 0,
            local_in.data(),             R * W, MPI_FLOAT, rank - 1, 1,
            MPI_COMM_WORLD, &status
        );
    }

    // ---- Bottom halo ----
    if (rank == world_size - 1) {
        /*
            No lower neighbor → replicate the last owned row.
        */
        for (int i = 0; i < R; ++i) {
            std::copy(
                local_in.begin() + (R + local_H - 1) * W,
                local_in.begin() + (R + local_H) * W,
                local_in.begin() + (R + local_H + i) * W
            );
        }
    } else {
        /*
            Exchange boundary with rank+1:
            Send bottom R owned rows,
            receive top R rows of the next rank into bottom halo region.
        */
        MPI_Sendrecv(
            local_in.data() + local_H * W,       R * W, MPI_FLOAT, rank + 1, 1,
            local_in.data() + (R + local_H) * W, R * W, MPI_FLOAT, rank + 1, 0,
            MPI_COMM_WORLD, &status
        );
    }

    // ----- 6. Timing the MPI total time + pure compute time -----
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    double t_local_start = MPI_Wtime();
    conv2d_local_with_halo(local_in, local_out, kernel, local_H, W, K, R);
    double t_local_end = MPI_Wtime();
    double local_compute_ms = (t_local_end - t_local_start) * 1000.0;

    /*
        Reduction: rank 0 wants to know the slowest rank's compute time.
        This matters because MPI performance is bottlenecked by the slowest process.
    */
    double mpi_compute_max_ms = 0.0;
    MPI_Reduce(
        &local_compute_ms, &mpi_compute_max_ms,
        1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD
    );

    // ----- 7. Gather all outputs back to rank 0 -----
    float* recvbuf = nullptr;
    if (rank == 0) {
        recvbuf = global_out.data();
    }

    /*
        MPI_Gatherv places each rank's output into the correct location in global_out.
        Displacements and counts mirror the Scatterv step.
    */
    MPI_Gatherv(
        local_out.data(), local_H * W, MPI_FLOAT,
        recvbuf, sendcounts_ptr, displs_ptr, MPI_FLOAT,
        0, MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double mpi_total_ms = (t1 - t0) * 1000.0;

    // ----- 8. Rank 0 computes CPU baseline & verifies correctness -----
    if (rank == 0) {
        double flops = compute_flops(H, W, K);

        auto c0 = std::chrono::high_resolution_clock::now();
        conv2d_cpu(global_in, out_cpu, kernel, H, W, K);
        auto c1 = std::chrono::high_resolution_clock::now();
        double cpu_ms =
            std::chrono::duration<double, std::milli>(c1 - c0).count();

        // Performance metrics
        double cpu_gflops = flops / (cpu_ms / 1000.0) / 1e9;
        double mpi_total_gflops = flops / (mpi_total_ms / 1000.0) / 1e9;
        double mpi_compute_gflops = flops / (mpi_compute_max_ms / 1000.0) / 1e9;

        double diff = max_abs_diff(out_cpu, global_out);

        std::cout << "===== MPI convolution =====\n";
        std::cout << "Image N = " << N << " (H=W), K = " << K
                  << ", world_size = " << world_size << "\n\n";

        std::cout << "CPU (single process): " << cpu_ms << " ms, "
                  << cpu_gflops << " GFLOP/s\n";

        std::cout << "MPI total time:       " << mpi_total_ms << " ms, "
                  << mpi_total_gflops << " GFLOP/s\n";

        std::cout << "MPI compute (max):    " << mpi_compute_max_ms << " ms, "
                  << mpi_compute_gflops << " GFLOP/s\n";

        std::cout << "Speedup (CPU / MPI total): "
                  << cpu_ms / mpi_total_ms << "x\n";

        std::cout << "Max abs diff vs CPU:  " << diff << "\n";

        // Single-line CSV output for scaling plots
        std::cout << "CSV,"
                  << N << "," << world_size << ","
                  << cpu_ms << "," << cpu_gflops << ","
                  << mpi_total_ms << "," << mpi_total_gflops << ","
                  << mpi_compute_max_ms << "," << mpi_compute_gflops << ","
                  << (cpu_ms / mpi_total_ms) << "\n";
    }

    MPI_Finalize();
    return 0;
}
