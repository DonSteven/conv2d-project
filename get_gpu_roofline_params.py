import subprocess


def get_gpu_roofline_params():
    """
    use nvidia-smi and manual table lookup, get GPU specs:
      - name_str: GPU name (string)
      - peak_flops_fp32: theoretical FP32 peak FLOP/s (float, unit FLOP/s)
      - peak_bw_bytes: theoretical memory bandwidth (float, unit Byte/s)

    Note:
    - In Oscar, nvidia-smi does not support memory.bus_width field, so we have to
      look it up manually based on GPU name.
    - clocks.max.memory reports "DDR clock", GDDR6X needs to be multiplied by 2 to get effective rate
    """
    # 1) first from nvidia-smi get GPU name + max memory clock
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,clocks.max.memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(
            cmd,
            encoding="utf-8",
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        print("nvidia-smi failed, output below:")
        print(e.output)
        raise

    line = out.strip().splitlines()[0]
    name_str, mem_clock_str = [x.strip() for x in line.split(",")]

    mem_clock_mhz = int(mem_clock_str)  # in MHz

    # 2) check the table for bus width and FP32 peak FLOPs
    #    this can be extended to more GPUs as needed
    gpu_specs = {
        "RTX 3090": {
            "bus_width_bits": 384,
            "peak_flops_fp32": 35.58e12,  # ≈ 35.6 TFLOP/s
        },
    }

    bus_width_bits = None
    peak_flops_fp32 = None
    for key, spec in gpu_specs.items():
        if key in name_str:
            bus_width_bits = spec["bus_width_bits"]
            peak_flops_fp32 = spec["peak_flops_fp32"]
            break

    if bus_width_bits is None:
        raise RuntimeError(
            f"cannot identify GPU '{name_str}' specs, please add to gpu_specs."
        )

    # 3) compute theoretical bandwidth
    # For Rtx 3090：
    #   nvidia-smi reports DDR clock (≈ 9751 MHz)
    #   effective transfer rate needs to be multiplied by 2
    effective_mem_clock_mhz = mem_clock_mhz * 2

    # BW[GB/s] ≈ mem_clock_MHz * bus_width_bits / 8000
    peak_bw_gbs = effective_mem_clock_mhz * bus_width_bits / 8000.0
    peak_bw_bytes = peak_bw_gbs * 1e9

    print("GPU name             :", name_str)
    print("Bus width (bits)     :", bus_width_bits)
    print("Max mem clock (MHz)  :", mem_clock_mhz)
    print("Effective mem clock  :", effective_mem_clock_mhz, "MHz")
    print(f"Peak BW              ≈ {peak_bw_gbs:.1f} GB/s")
    if peak_flops_fp32 is not None:
        print(f"Peak FP32 FLOPs      ≈ {peak_flops_fp32 / 1e12:.2f} TFLOP/s")

    return name_str, peak_flops_fp32, peak_bw_bytes


if __name__ == "__main__":
    get_gpu_roofline_params()
