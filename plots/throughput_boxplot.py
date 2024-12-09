import matplotlib.pyplot as plt

results_file = "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/zune-implementation/benchmark_throughput/build/benchmark_results.txt"

def extract_throughput_mb(file_path):
    """Extract throughput values in MB/sec from the benchmark results text file."""
    throughputs = []
    with open(file_path, "r") as file:
        for line in file:
            if "Bytes per second" in line:
                throughput = float(line.split("Bytes per second:")[-1].split("MB/sec")[0].strip())
                throughputs.append(throughput)
    return throughputs

# Extract throughput data in MB/sec
throughput_data_mb = extract_throughput_mb(results_file)

if throughput_data_mb:
    # Create the boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        throughput_data_mb,
        patch_artist=True,  
        boxprops=dict(facecolor="skyblue", color="blue"),  
        medianprops=dict(color="red", linewidth=2), 
        whiskerprops=dict(color="blue", linewidth=1.5),  
        capprops=dict(color="blue", linewidth=1.5)  
    )
    plt.title("Throughput Distribution for Zune JPEG Decoder", fontsize=14)
    plt.ylabel("Throughput (MB/sec)", fontsize=12)  # Updated to MB/sec
    plt.xticks([1], ["Zune JPEG Decoder"], fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    output_path = "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/plots/throughput_boxplot_mb.png"
    plt.savefig(output_path)

    plt.show()

    print(f"Box plot saved as '{output_path}'")
else:
    print("No throughput data available to plot.")
