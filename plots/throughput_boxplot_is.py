import os
import matplotlib.pyplot as plt
import numpy as np

# Define benchmark parameters
versions = ['cuda', 'nvjpeg']
base_path = ''
plt.rcParams.update({'font.size': 18})

# Function to extract batch size and throughput data in MB/sec
# This assumes lines are in the format "Batchsize: <batchsize>, Throughput: <value> images/sec, Bytes per second: <value> MB/sec"
def extract_batch_throughput(file_path):
    batch_throughput = {}
    try:
        with open(file_path, "r") as file:
            for line in file:
                if "Batchsize:" in line and "Bytes per second:" in line:
                    parts = line.split(",")
                    batch_size = int(parts[0].split("Batchsize:")[-1].strip())
                    throughput = float(parts[2].split("Bytes per second:")[-1].split("MB/sec")[0].strip())
                    batch_throughput.setdefault(batch_size, []).append(throughput)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return batch_throughput

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

# Collect throughput data per batch size
throughput_data = {}

for version in versions:
    results_txt = find_txt_file(base_path, version)
    if results_txt:
        print(f"Processing file: {results_txt}")
        throughput_data[version] = extract_batch_throughput(results_txt)

        # Print throughput data for debugging
        if throughput_data[version]:
            print(f"Throughput data for {version}: {throughput_data[version]}")
        else:
            print(f"No throughput data found for {version}")
    else:
        print(f"TXT file not found for version {version}.")

# Plot throughput vs. batch size
if throughput_data:
    plt.figure(figsize=(12, 8))

    for version, data in throughput_data.items():
        batch_sizes = sorted(data.keys())
        mean_throughputs = [np.mean(data[batch]) for batch in batch_sizes]
        std_devs = [np.std(data[batch]) for batch in batch_sizes]

        plt.plot(batch_sizes, mean_throughputs, marker="o", label=version)
        plt.fill_between(
            batch_sizes,
            [mean - std for mean, std in zip(mean_throughputs, std_devs)],
            [mean + std for mean, std in zip(mean_throughputs, std_devs)],
            alpha=0.2
        )

    plt.title("Throughput vs. Image Size for Various JPEG Decoders", fontsize=18)
    plt.xlabel("Number of Images in Single Batch", fontsize=18)
    plt.ylabel("Throughput (MB/sec)", fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot to a file
    png_path = os.path.join(base_path, 'plots', 'throughput_vs_image_size.png')
    svg_path = os.path.join(base_path, 'plots', 'throughput_vs_image_size.svg')
    pdf_path = os.path.join(base_path, 'plots', 'throughput_vs_image_size.pdf')
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path)
    plt.savefig(svg_path)
    plt.savefig(pdf_path)
    print(f"Throughput plot saved as '{png_path}'")
    plt.show()
else:
    print("No valid throughput data to plot.")
