import matplotlib.pyplot as plt
import numpy as np

def plot_roofline(hardware_name, peak_flops, peak_bw, measured_points, filename):
    """
    绘制 Roofline Model
    peak_flops: GFLOP/s
    peak_bw: GB/s
    measured_points: list of dict {'name': str, 'ai': float, 'perf': float, 'color': str}
    """
    
    # 1. set the axis ranges (log scale)
    ai_min = 0.1  # 0.1 FLOP/Byte
    ai_max = 1000 # 1000 FLOP/Byte
    
    ridge_point = peak_flops / peak_bw
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 2. draw the roof
    # Memory Bound (y = x * bandwidth)
    x_mem = np.linspace(ai_min, ridge_point, 100)
    y_mem = x_mem * peak_bw
    ax.plot(x_mem, y_mem, 'k-', linewidth=2, label='Memory Bandwidth Bound')
    
    # Compute Bound (y = peak_flops)
    x_compute = np.linspace(ridge_point, ai_max, 100)
    y_compute = [peak_flops] * len(x_compute)
    ax.plot(x_compute, y_compute, 'k--', linewidth=2, label='Compute Bound (Peak)')
    
    # hardware parameters
    ax.text(ai_min * 1.2, peak_bw * ai_min * 1.5, f'Peak BW: {peak_bw:.1f} GB/s', rotation=45, fontsize=10)
    ax.text(ai_max / 5, peak_flops * 1.1, f'Peak: {peak_flops:.0f} GFLOP/s', fontsize=10)

    # 3. plot measured data points (The Dots)
    for p in measured_points:
        ax.plot(p['ai'], p['perf'], 'o', color=p['color'], markersize=10, label=p['name'], markeredgecolor='black')
        # add a label to the point
        ax.annotate(f"{p['perf']:.1f} G", (p['ai'], p['perf']), 
                    xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)

    # 4. format the chart
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=12)
    ax.set_title(f'Roofline Model: {hardware_name}', fontsize=14)
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend(loc='lower right')
    
    plt.savefig(filename, dpi=300)
    print(f"Graph saved to {filename}")
    plt.show()

# ==========================================
# 1. GPU Roofline (RTX 3090)
# ==========================================
gpu_points = [
    # Naive: theoretical AI ~0.49 (K=7). Reality Perf ~998.5
    {'name': 'GPU Naive (Theoretical AI)', 'ai': 0.49, 'perf': 998.5, 'color': 'red'},
    
    # Tiled: theoretical AI ~8.5 (K=7, Block=16). Reality Perf ~980.2
    {'name': 'GPU Tiled (Theoretical AI)', 'ai': 8.5,  'perf': 980.2, 'color': 'green'}
]

plot_roofline(
    hardware_name="NVIDIA RTX 3090", 
    peak_flops=35580,  # 35.58 TFLOPs
    peak_bw=936.1,     # 936.1 GB/s
    measured_points=gpu_points, 
    filename="roofline_gpu.png"
)

# ==========================================
# 2. CPU Roofline (2x EPYC 7413)
# ==========================================
cpu_points = [
    # CPU: Naive AI ~0.49. Reality Perf ~1.84 (N=4096)
    {'name': 'CPU Base (N=4096)', 'ai': 0.49, 'perf': 1.84, 'color': 'blue'}
]

plot_roofline(
    hardware_name="2x AMD EPYC 7413", 
    peak_flops=4070,   # 4.07 TFLOPs
    peak_bw=409.6,     # 409.6 GB/s
    measured_points=cpu_points, 
    filename="roofline_cpu.png"
)