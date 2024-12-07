import matplotlib.pyplot as plt

# Define the file path to the benchmark results text file
results_file = "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/zune-implementation/benchmark_throughput/build/benchmark_results.txt"

def extract_throughput(file_path):
    """Extract throughput values from the benchmark results text file."""
    throughputs = []
    with open(file_path, "r") as file:
        for line in file:
            if "Total Throughput" in line:
                # Extract throughput value from the line
                throughput = float(line.split("images/sec")[0].split(":")[-1].strip())
                throughputs.append(throughput)
    return throughputs

# Extract throughput data
throughput_data = extract_throughput(results_file)

if throughput_data:
    # Create the boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        throughput_data,
        patch_artist=True,  
        boxprops=dict(facecolor="skyblue", color="blue"),  
        medianprops=dict(color="red", linewidth=2), 
        whiskerprops=dict(color="blue", linewidth=1.5),  
        capprops=dict(color="blue", linewidth=1.5)  
    )
    plt.title("Throughput Distribution for Zune JPEG Decoder", fontsize=14)
    plt.ylabel("Throughput (images/sec)", fontsize=12)
    plt.xticks([1], ["Zune JPEG Decoder"], fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot as an image file
    output_path = "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/plots/throughput_boxplot.png"
    plt.savefig(output_path)

    # Display the plot
    plt.show()

    print(f"Box plot saved as '{output_path}'")
else:
    print("No throughput data available to plot.")
