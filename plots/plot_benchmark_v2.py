import os
import json
import matplotlib.pyplot as plt

# Define benchmark parameters
versions = ['zune', 'cudaO', 'cudaU', 'cpp', 'jpeglib']
sizes = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

def find_json_file(base_path, version):
    """
    Searches for the JSON file under `implementation/build/` or `implementation/benchmark/build/`.
    """
    possible_paths = [
        os.path.join(base_path, version + '-implementation', 'build', 'benchmark_results.json'),
        os.path.join(base_path, version + '-implementation', 'benchmark', 'build', 'benchmark_results.json')
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"JSON file not found for version {version} in specified directories.")

def readings(json_path):
    """
    Reads benchmark results from a JSON file and computes average decoding times for each size.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Initialize totals and counts for each size
    totals = [0 for _ in range(len(sizes))]
    counts = [0 for _ in range(len(sizes))]

    for bench in data["benchmarks"]:
        idx = bench["family_index"]
        totals[idx] += bench["real_time"]
        counts[idx] += 1

    # Compute average decoding times
    results = [totals[i] / counts[i] if counts[i] > 0 else 0 for i in range(len(sizes))]
    return results

# Base path for the project
base_path = '/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/'

# Extract average decoding times for each version
average_times = {}
for version in versions:
    try:
        results_json = find_json_file(base_path, version)
        print(f"Processing file: {results_json}")
        average_times[version] = readings(results_json)
        print(f"Averages for {version}: {average_times[version]}")
    except FileNotFoundError as e:
        print(e)

# Plot the results
plt.figure(figsize=(10, 6))
size_labels = [str(x) + "x" + str(x) for x in sizes]
for version in versions:
    if version in average_times:
        plt.plot(size_labels, average_times[version], marker='o', label=version)

plt.legend()
plt.title("JPEG Decoding Time vs. Image Size")
plt.xlabel("Image Size (WxH)")
plt.ylabel("Average Decoding Time (ms)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Save the plot to a file
output_path = os.path.join(base_path, 'plots', 'benchmark_decoding_time_plot.png')
plt.savefig(output_path)  # Save as PNG file

print(f"Plot saved to {output_path}")
