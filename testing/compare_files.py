def read_numbers_from_file(file_path):
    """Reads a file with numbers separated by spaces and returns them as a list of floats."""
    with open(file_path, 'r') as file:
        content = file.read()
        # Convert content to a list of numbers
        numbers = [float(num) for num in content.split()]
    return numbers

def compare_files(file1, file2):
    """Compares two files with lists of numbers and returns indices and values of differences."""
    numbers1 = read_numbers_from_file(file1)
    numbers2 = read_numbers_from_file(file2)

    # Find the shorter length to prevent index errors
    min_length = min(len(numbers1), len(numbers2))
    
    # Collect differences
    differences = []
    for i in range(min_length):
        if numbers1[i] != numbers2[i]:
            differences.append((i, numbers1[i], numbers2[i]))

    # Check for additional numbers in longer list
    if len(numbers1) != len(numbers2):
        longer_list = numbers1 if len(numbers1) > len(numbers2) else numbers2
        for i in range(min_length, len(longer_list)):
            differences.append((i, longer_list[i], None if len(numbers1) > len(numbers2) else longer_list[i]))

    return differences

# Specify the paths to your files
file1_path = '/home/dphpc2024_jpeg_1/cfernand/GPU-JPEG-Decoder/testing/ground_truth/7_3264x2448.array'
file2_path = '/home/dphpc2024_jpeg_1/cfernand/GPU-JPEG-Decoder/testing/cuda1_output_arrays/7_3264x2448.array'

# Run comparison and print results
differences = compare_files(file1_path, file2_path)
if differences:
    print("Differences found at indices:")
    for index, num1, num2 in differences:
        print(f"Index {index}: File1 has {num1}, File2 has {num2}")
else:
    print("The files are identical.")
