import os
import json
import numpy as np  # To compute standard deviation
import matplotlib.pyplot as plt

versions = ['cpp', 'cuda', 'cudaunified']
sizes = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

def readings(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Initialize lists to store times per size
    times_per_size = [[] for _ in range(len(sizes))]
    
    # Collect the real_time for each size
    for bench in data["benchmarks"]:
        idx = bench["family_index"]
        times_per_size[idx].append(bench["real_time"])
    
    # Calculate averages and standard deviations
    averages = [np.mean(times) if times else 0 for times in times_per_size]
    std_devs = [np.std(times) if times else 0 for times in times_per_size]
    
    return averages, std_devs

# Dictionary to store average times and standard deviations
average_times = {}
std_devs = {}

for version in versions:
    results_json = os.path.join('/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/', version + '-implementation', 'benchmark', 'build', 'benchmark_results.json')
    # results_json = os.path.join('/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/', version + '-implementation', 'build/benchmark_results.json')
    avg, std = readings(results_json)
    average_times[version] = avg
    std_devs[version] = std
    print(f"{version} average times: {avg}")
    print(f"{version} std deviations: {std}")

# Plotting the results with standard deviation shading
plt.figure(figsize=(10, 6))
size_labels = [str(x) + "x" + str(x) for x in sizes]

for version in versions:
    avg = average_times[version]
    std = std_devs[version]
    
    # Plot the average line
    plt.plot(size_labels, avg, marker='o', label=version)
    
    # Add the shaded region for standard deviation
    plt.fill_between(size_labels, np.array(avg) - np.array(std), np.array(avg) + np.array(std), alpha=0.2)

plt.legend()
plt.title("JPEG Decoding Time vs. Image Size (with Standard Deviation)")
plt.xlabel("Image Size (WxH)")
plt.ylabel("Average Decoding Time (ms)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Save the plot to a file
output_path = '/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/plots/benchmark_decoding_time_plot_with_std.png'
plt.savefig(output_path)  # Save as PNG file

# Display the plot
# plt.show()
