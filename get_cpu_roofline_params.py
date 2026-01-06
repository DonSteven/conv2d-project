# current node：2 x AMD EPYC 7413 24-Core
# compute the theoretical FP32 peak FLOPs and memory bandwidth (for Roofline)

def get_epyc_7413_cpu_roofline_params(
    use_turbo_freq: bool = False,
):
    """
    Returns:
      - cpu_name
      - peak_flops_fp32: the theoretical FP32 peak FLOPs (double, unit FLOP/s)
      - peak_bw_bytes: the theoretical memory bandwidth (double, unit Byte/s)

    Args:
      - use_turbo_freq:
          False: use base frequency 2.65 GHz to compute (more conservative, closer to long-term sustainable performance)
          True:  use max turbo 3.60 GHz to compute (theoretical absolute peak)
    """

    cpu_name = "2 x AMD EPYC 7413 (Milan, 24C each)"

    # ---------- 1) frequency ----------
    base_freq_ghz = 2.65   # EPYC 7413 base clock
    turbo_freq_ghz = 3.60  # EPYC 7413 max boost
    freq_ghz = turbo_freq_ghz if use_turbo_freq else base_freq_ghz

    # ---------- 2) core count ----------
    cores_per_socket = 24
    num_sockets = 2
    total_cores = cores_per_socket * num_sockets 

    # ---------- 3) each core per cycle FP32 FLOPs ----------
    # AMD Milan: AVX2 256-bit + 2x FMA units per core
    # 256 bit / 32 bit = 8 lanes (FP32)
    # each FMA: 2 FLOPs（mul + add）
    # each core per cycle: 8 lanes * 2 FMA * 2 FLOPs/FMA = 32 FLOPs
    flops_per_cycle_per_core_fp32 = 32

    # Peak FLOPs = total_cores * frequency(Hz) * flops_per_cycle_per_core_fp32
    peak_flops_fp32 = (
        total_cores
        * freq_ghz
        * 1e9
        * flops_per_cycle_per_core_fp32
    )

    # ---------- 4) memory bandwidth ----------
    # official per-socket memory bandwidth = 204.8 GB/s (DDR4-3200, 8 channels)
    bw_per_socket_gbs = 204.8
    total_bw_gbs = bw_per_socket_gbs * num_sockets
    peak_bw_bytes = total_bw_gbs * 1e9

    # print human-readable info
    print("CPU name               :", cpu_name)
    print("Frequency used (GHz)   :", freq_ghz)
    print("Total physical cores   :", total_cores)
    print("FP32 FLOPs/core/cycle  :", flops_per_cycle_per_core_fp32)
    print(f"Peak FP32 FLOPs        ≈ {peak_flops_fp32 / 1e12:.2f} TFLOP/s")
    print(f"Peak memory BW         ≈ {total_bw_gbs:.1f} GB/s")

    return cpu_name, peak_flops_fp32, peak_bw_bytes


if __name__ == "__main__":
    # by default use base frequency (2.65 GHz) for Roofline, more conservative than turbo
    get_epyc_7413_cpu_roofline_params(use_turbo_freq=False)
