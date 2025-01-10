import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from statistics import mean

versions = ['cudaH', 'nvjpeg']

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
    results_json = os.path.join('GPU-JPEG-Decoder/', version+'-implementation', 'benchmark/build/benchmark_results.json')
    average_times[version] = readings(results_json)
    print(average_times[version])

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
output_path = 'GPU-JPEG-Decoder/plots/benchmark_decoding_time_plot.png'
plt.savefig(output_path)  # Save as PNG file

# avg = []
# for i in range(10):
#     avg.append(average_times['nvjpeg'][i]/average_times['cudaH'][i])
# print(avg)
# print("average speedup: ", mean(avg))
# print("max speedup: ", max(avg))

# Display the plot
# plt.show()
