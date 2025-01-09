import os
import matplotlib.pyplot as plt
import numpy as np

# Define benchmark parameters
versions = ['zune', 'jpeglib', 'nvjpeg', 'cuda']
base_path = ''

plt.rcParams.update({'font.size': 18})

# Function to extract throughput data in MB/sec from the benchmark results text file
def extract_throughput_mb(file_path):
    throughputs = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                if "Bytes per second" in line:
                    throughput = float(line.split("Bytes per second:")[-1].split("MB/sec")[0].strip())
                    throughputs.append(throughput)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return throughputs

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

# Loop through all versions and collect throughput data
throughput_data = {}

for version in versions:
    results_txt = find_txt_file(base_path, version)
    if results_txt:
        print(f"Processing file: {results_txt}")
        throughput_data[version] = extract_throughput_mb(results_txt)
        
        # Print throughput data for debugging
        if throughput_data[version]:
            print(f"Throughput data for {version}: {throughput_data[version]}")
        else:
            print(f"No throughput data found for {version}")
    else:
        print(f"TXT file not found for version {version}.")

# Prepare data for the boxplot
box_data = [throughput_data[version] for version in versions if version in throughput_data and throughput_data[version]]

# Ensure all versions have at least some range for visibility
box_data = [data if np.ptp(data) > 0 else [min(data) - 1e-2, max(data) + 1e-2] for data in box_data]

means = [np.mean(data) for data in box_data]
print(means)

# Check if there's valid data to plot
if box_data:
    # Create the boxplot
    plt.figure(figsize=(10, 6))

    plt.boxplot(
        box_data,
        patch_artist=True,
        boxprops=dict(facecolor="skyblue", color="blue"),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(color="blue", linewidth=1.5),
        capprops=dict(color="blue", linewidth=1.5),
        showfliers=False
    )

    # Adjust the y-axis to show all values clearly
    y_min = min(min(data) for data in box_data) * 0.9  # Add 10% padding
    y_max = max(max(data) for data in box_data) * 1.1
    plt.ylim(y_min, y_max)

    # Set plot labels and title
    plt.title("Throughput Distribution for Various JPEG Decoders", fontsize=18)
    plt.ylabel("Throughput (MB/sec)", fontsize=18)
    plt.xticks(range(1, len(box_data) + 1), [v for v in versions if v in throughput_data and throughput_data[v]], fontsize=18)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot in multiple formats
    output_dir = os.path.join(base_path, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    output_png = os.path.join(output_dir, 'throughput_boxplot_reduced.png')
    output_pdf = os.path.join(output_dir, 'throughput_boxplot_reduced.pdf')
    output_svg = os.path.join(output_dir, 'throughput_boxplot_reduced.svg')

    plt.savefig(output_png)
    plt.savefig(output_pdf)
    plt.savefig(output_svg)

    plt.legend()

    print(f"Box plot saved as PNG: {output_png}")
    print(f"Box plot saved as PDF: {output_pdf}")
    print(f"Box plot saved as SVG: {output_svg}")
else:
    print("No valid throughput data to plot.")
