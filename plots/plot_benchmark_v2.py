import os
import matplotlib.pyplot as plt
import numpy as np

# Define benchmark parameters
#versions = ['jpeglib', 'zune','nvjpeg', 'cpp', 'cudaH']
#versions = ['jpeglib', 'zune', 'cpp', 'nvjpeg', 'cudaH', 'cudaO']
#versions = ['jpeglib', 'nvjpeg', 'cpp', 'cudaU', 'cudaB']
versions = ['jpeglib', 'nvjpeg', 'cudaO']


sizes = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

def find_txt_file(base_path, version):
    """
    Searches for the TXT file under `implementation/build/` or `implementation/benchmark/build/`.
    """
    possible_paths = [
        # os.path.join(base_path, version + '-implementation', 'build', 'benchmark_results.txt'),
        os.path.join(base_path, version + '-implementation', 'benchmark', 'build', 'benchmark_results.txt')
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"TXT file not found for version {version} in specified directories.")

def read_txt_file(txt_path):
    """
    Reads benchmark results from a TXT file and computes average decoding times and standard deviations for each size.
    """
    image_sizes = {size: [] for size in sizes} 

    with open(txt_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            image_path = parts[0]
            time = float(parts[1])

            size_str = image_path.split('/')[-2]  
            size = int(size_str.split('x')[0])  

            if size in image_sizes:
                image_sizes[size].append(time)

    return image_sizes

def calculate_stats(image_sizes):
    """
    Calculate average decoding times and standard deviations for each size.
    """
    avg_times = {}
    std_devs = {}
    
    for size, times in image_sizes.items():
        #print(f"size: {size}")
        #print(f"times: {times}")

        if times:  
            avg_times[size] = np.mean(times)
            std_devs[size] = np.std(times)
        else:
            avg_times[size] = 0
            std_devs[size] = 0

    return avg_times, std_devs

# Base path for the project
base_path = '/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/'

# Extract average decoding times and standard deviations for each version
average_times = {}
std_devs = {}
for version in versions:
    try:
        results_txt = find_txt_file(base_path, version)
        print(f"Processing file: {results_txt}")
        image_sizes = read_txt_file(results_txt)
        avg_times, std_dev = calculate_stats(image_sizes)
        average_times[version] = avg_times
        std_devs[version] = std_dev
        print(f"Averages for {version}: {avg_times}")
        print(f"Standard Deviations for {version}: {std_dev}")
    except FileNotFoundError as e:
        print(e)

# Plot the results with standard deviation as shaded area behind each line
plt.figure(figsize=(10, 6))
size_labels = [str(x) + "x" + str(x) for x in sizes]
for version in versions:
    if version in average_times:
        avg_times = [average_times[version].get(size, 0) for size in sizes]
        std_devs_for_version = [std_devs[version].get(size, 0) for size in sizes]

        color = plt.cm.tab10(versions.index(version) % 10)  
        
        # Only add shaded area if standard deviation is non-zero
        if np.any(np.array(std_devs_for_version) > 0): 
            plt.fill_between(size_labels, 
                             np.subtract(avg_times, std_devs_for_version), 
                             np.add(avg_times, std_devs_for_version), 
                             alpha=0.3, color=color)  

        plt.plot(size_labels, avg_times, marker='o', label=version, color=color)  

plt.legend()
plt.title("JPEG Decoding Time vs. Image Size with Standard Deviation")
plt.xlabel("Image Size (WxH)")
plt.ylabel("Average Decoding Time (ms)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Save the plot to a file
output_path = os.path.join(base_path, 'plots', 'benchmark_decoding_time_with_stddev_shaded_plot.png')
plt.savefig(output_path)  # Save as PNG file

print(f"Plot saved to {output_path}")
