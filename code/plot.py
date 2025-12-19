import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ------------------------------
# Data from your results
# ------------------------------

# Context length scalability
context_lengths = [128, 256, 512, 1024, 2048, 4096]

# Latency (ms/query)
latency_baseline = [618.55, 819.35, 1023.34, 1430.65, 2266.81, 3940.64]
latency_paged    = [1172.16, 1377.78, 1578.74, 2002.98, 2101.57, 3819.25]

# Peak GPU memory allocated (GB)
gpu_memory_baseline = [5.08, 7.41, 9.73, 14.39, 23.70, 42.33]
gpu_memory_paged    = [5.15, 7.47, 9.79, 14.45, 23.75, 42.36]

# Batch size efficiency
batch_sizes = [1, 2, 4, 8, 16, 32]
throughput_baseline = [4.58, 3.81, 3.90, 3.95, 3.97, 3.97]
throughput_paged    = [3.45, 3.29, 3.62, 3.79, 3.88, 3.91]

# Colors for consistency
color_baseline = '#1f77b4'  # blue
color_paged    = '#ff7f0e'  # orange

# ------------------------------
# Figure 2: Latency vs Context Length
# ------------------------------
plt.figure(figsize=(6,4))
plt.plot(context_lengths, latency_baseline, marker='o', color=color_baseline, linestyle='-', linewidth=2, label='Baseline')
plt.plot(context_lengths, latency_paged, marker='s', color=color_paged, linestyle='--', linewidth=2, label='Paged')
plt.xlabel('Context Length (tokens)', fontsize=12)
plt.ylabel('Latency (ms/query)', fontsize=12)
plt.title('Context Length vs Latency', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.xticks(context_lengths)  # show only desired ticks
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('../figures/fig_latency_vs_context.png', dpi=300)
# plt.show()

# ------------------------------
# Figure 3: Peak GPU Memory vs Context Length
# ------------------------------
plt.figure(figsize=(6,4))
plt.plot(context_lengths, gpu_memory_baseline, marker='o', color=color_baseline, linestyle='-', linewidth=2, label='Baseline')
plt.plot(context_lengths, gpu_memory_paged, marker='s', color=color_paged, linestyle='--', linewidth=2, label='Paged')
plt.xlabel('Context Length (tokens)', fontsize=12)
plt.ylabel('Peak GPU Memory (GB)', fontsize=12)
plt.title('Context Length vs Peak GPU Memory', fontsize=14)
plt.xscale('log')
plt.xticks(context_lengths)  # show only desired ticks
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.gca().xaxis.get_major_formatter().set_scientific(False)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('../figures/fig_gpu_memory_vs_context.png', dpi=300)
# plt.show()

# ------------------------------
# Figure 4: Throughput vs Batch Size
# ------------------------------
plt.figure(figsize=(6,4))
plt.plot(batch_sizes, throughput_baseline, marker='o', color=color_baseline, linestyle='-', linewidth=2, label='Baseline')
plt.plot(batch_sizes, throughput_paged, marker='s', color=color_paged, linestyle='--', linewidth=2, label='Paged')
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Throughput (queries/sec)', fontsize=12)
plt.title('Batch Size vs Throughput', fontsize=14)
plt.xscale('log', base=2)
plt.xticks(batch_sizes)  # show only desired ticks
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('../figures/fig_throughput_vs_batch.png', dpi=300)
# plt.show()
