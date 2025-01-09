import os
import matplotlib.pyplot as plt
import numpy as np

# Define benchmark parameters
versions = ['cuda']
base_path = ''
plt.rcParams.update({'font.size': 18})

# Function to extract thread count and throughput data in MB/sec
def extract_thread_throughput(file_path):
    thread_throughput = {}
    try:
        with open(file_path, "r") as file:
            for line in file:
                if "Threads:" in line and "Bytes per second:" in line:
                    parts = line.split(",")
                    threads = int(parts[0].split("Threads:")[-1].strip())
                    throughput = float(parts[2].split("Bytes per second:")[-1].split("MB/sec")[0].strip())
                    thread_throughput.setdefault(threads, []).append(throughput)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return thread_throughput

# Function to find the TXT file for each version
def find_txt_file(base_path, version):
    possible_paths = [
        os.path.join(base_path, version + '-implementation', 'benchmark_thoughput', 'build', 'benchmark_results.txt'),
        os.path.join(base_path, version + '-implementation', 'benchmark_throughput', 'build', 'benchmark_results.txt')
    ]
    for path in possible_paths:
        print(f"Checking path: {path}")
        if os.path.exists(path):
            return path
    return None  # Return None if no file is found

# Collect throughput data per thread count
throughput_data = {}

for version in versions:
    results_txt = find_txt_file(base_path, version)
    if results_txt:
        print(f"Processing file: {results_txt}")
        throughput_data[version] = extract_thread_throughput(results_txt)

        # Print throughput data for debugging
        if throughput_data[version]:
            print(f"Throughput data for {version}: {throughput_data[version]}")
        else:
            print(f"No throughput data found for {version}")
    else:
        print(f"TXT file not found for version {version}.")

# Plot speedup vs. number of threads
# Plot speedup vs. number of threads
if throughput_data:
    plt.figure(figsize=(12, 8))

    for version, data in throughput_data.items():
        thread_counts = sorted(data.keys())
        mean_throughputs = [np.mean(data[threads]) for threads in thread_counts]
        speedups = [throughput / mean_throughputs[0] for throughput in mean_throughputs]  # Speedup w.r.t 1 thread

        plt.plot(thread_counts, speedups, marker="o", label=version)

    # Set x-axis to log scale and define custom ticks
    plt.xscale('log', base=2)
    plt.gca().set_xticks(thread_counts)  # Set specific tick positions
    plt.gca().set_xticklabels(thread_counts, fontsize=18)  # Ensure the tick labels are visible

    plt.title("Speedup vs. Number of Threads for CUDA", fontsize=18)
    plt.xlabel("Number of Threads (Log Scale)", fontsize=18)
    plt.ylabel("Speedup (w.r.t 1 Thread)", fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot to a file
    png_path = os.path.join(base_path, 'plots', 'speedup_vs_threads.png')
    svg_path = os.path.join(base_path, 'plots', 'speedup_vs_threads.svg')
    pdf_path = os.path.join(base_path, 'plots', 'speedup_vs_threads.pdf')
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path)
    plt.savefig(svg_path)
    plt.savefig(pdf_path)
    print(f"Speedup plot saved as '{png_path}'")
    plt.show()
else:
    print("No valid throughput data to plot.")

