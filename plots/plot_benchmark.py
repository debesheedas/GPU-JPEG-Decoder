import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

versions = ['cpp','cuda','cudaunified']

sizes = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

def readings(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Print the data
    # print(data["benchmarks"])
    totals = [0 for _ in range(len(sizes))]
    counts = [0 for _ in range(len(sizes))]
    # results= [0 for _ in range(len(sizes))]
    for bench in data["benchmarks"]:
        idx = bench["family_index"]
        totals[idx] += bench["real_time"]
        counts[idx] += 1
    results = [totals[i]/counts[i] for i in range(len(sizes))]
    return results



average_times = {}
for version in versions:
    results_json = os.path.join('/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/', version+'-implementation', 'build/benchmark_results.json')
    average_times[version] = readings(results_json)
    print(average_times[version])

# # Function to extract image size from the file path
# def get_image_size(filename):
#     parts = filename.split('_')
#     size_str = parts[-1].split('.')[0]  # Extracts '200x200' part
#     return size_str

# # Dictionary to store decoding times for each image size
# image_sizes = defaultdict(list)

# # Assuming the benchmark results are stored in this path
# benchmark_file_path = '/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cpp-implementation/build/benchmark_results.txt'

# # Read the benchmark results from the text file
# with open(benchmark_file_path, 'r') as file:
#     for line in file:
#         parts = line.strip().split()
#         if len(parts) == 2:
#             image_path = parts[0]
#             decoding_time = float(parts[1])
#             size = get_image_size(image_path)  # Extract the image size
#             image_sizes[size].append(decoding_time)

# # Calculate average decoding times for each image size
# average_times = {size: sum(times) / len(times) for size, times in image_sizes.items()}

# # Sort the sizes for better plotting (optional, to keep the plot tidy)
# sorted_sizes = sorted(average_times.keys(), key=lambda x: [int(i) for i in x.split('x')])

# Plot the results
plt.figure(figsize=(10, 6))
size_labels = [str(x)+"x"+str(x) for x in sizes]
for version in versions:
    plt.plot(size_labels, average_times[version], marker='o', label=version)
plt.legend()
plt.title("JPEG Decoding Time vs. Image Size")
plt.xlabel("Image Size (WxH)")
plt.ylabel("Average Decoding Time (ms)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Save the plot to a file
output_path = '/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/plots/benchmark_decoding_time_plot.png'
plt.savefig(output_path)  # Save as PNG file

# Display the plot
# plt.show()
