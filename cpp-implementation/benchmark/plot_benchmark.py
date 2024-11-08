import matplotlib.pyplot as plt
from collections import defaultdict

# Function to extract image size from the file path
def get_image_size(filename):
    parts = filename.split('_')
    size_str = parts[-1].split('.')[0]  # Extracts '200x200' part
    return size_str

# Dictionary to store decoding times for each image size
image_sizes = defaultdict(list)

# Assuming the benchmark results are stored in this path
benchmark_file_path = '/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cpp-implementation/benchmark_results.txt'

# Read the benchmark results from the text file
with open(benchmark_file_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) == 2:
            image_path = parts[0]
            decoding_time = float(parts[1])
            size = get_image_size(image_path)  # Extract the image size
            image_sizes[size].append(decoding_time)

# Calculate average decoding times for each image size
average_times = {size: sum(times) / len(times) for size, times in image_sizes.items()}

# Sort the sizes for better plotting (optional, to keep the plot tidy)
sorted_sizes = sorted(average_times.keys(), key=lambda x: [int(i) for i in x.split('x')])

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sorted_sizes, [average_times[size] for size in sorted_sizes], marker='o')
plt.title("JPEG Decoding Time vs. Image Size")
plt.xlabel("Image Size (WxH)")
plt.ylabel("Average Decoding Time (ms)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Save the plot to a file
output_path = '/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cpp-implementation/benchmark_decoding_time_plot.png'
plt.savefig(output_path)  # Save as PNG file

# Display the plot
plt.show()
