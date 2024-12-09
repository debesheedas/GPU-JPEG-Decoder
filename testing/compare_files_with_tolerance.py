def read_numbers_from_file(file_path):
    """Reads a file with numbers separated by spaces and returns them as a list of floats."""
    with open(file_path, 'r') as file:
        content = file.read()
        # Convert content to a list of numbers
        numbers = [float(num) for num in content.split()]
    return numbers

def compare_files(file1, file2, tolerance=1e-5):
    """
    Compares two files with lists of numbers and returns indices and values of differences.
    
    Args:
        file1 (str): Path to the first file.
        file2 (str): Path to the second file.
        tolerance (float): Acceptable difference between numbers.
    
    Returns:
        tuple: Differences and similarity percentages (with and without tolerance).
    """
    numbers1 = read_numbers_from_file(file1)
    numbers2 = read_numbers_from_file(file2)

    # Find the shorter length to prevent index errors
    min_length = min(len(numbers1), len(numbers2))
    
    # Collect differences (with and without tolerance)
    differences = []
    total_comparisons = min_length
    within_tolerance = 0
    exact_matches = 0

    for i in range(min_length):
        if abs(numbers1[i] - numbers2[i]) > tolerance:
            differences.append((i, numbers1[i], numbers2[i]))
        else:
            within_tolerance += 1
            if numbers1[i] == numbers2[i]:
                exact_matches += 1

    # Check for additional numbers in the longer list
    if len(numbers1) != len(numbers2):
        longer_list = numbers1 if len(numbers1) > len(numbers2) else numbers2
        for i in range(min_length, len(longer_list)):
            differences.append((i, longer_list[i], None if len(numbers1) > len(numbers2) else longer_list[i]))

    # Calculate similarity percentages
    similarity_with_tolerance = (within_tolerance / total_comparisons) * 100
    similarity_exact = (exact_matches / total_comparisons) * 100

    return differences, similarity_with_tolerance, similarity_exact

# Specify the paths to your files
file1_path = '/home/dphpc2024_jpeg_1/cfernand/GPU-JPEG-Decoder/testing/cuda1_output_arrays/profile.array'
file2_path = '/home/dphpc2024_jpeg_1/cfernand/GPU-JPEG-Decoder/testing/jpeglib_output_arrays/profile.array'

# Set tolerance
tolerance_value = 3  # Adjust as needed

# Run comparison
differences, similarity_with_tolerance, similarity_exact = compare_files(file1_path, file2_path, tolerance=tolerance_value)

# Print results
print(f"Similarity with tolerance ({tolerance_value}): {similarity_with_tolerance:.2f}%")
print(f"Exact similarity (no tolerance): {similarity_exact:.2f}%")

#if differences:
#    print("Differences found:")
#    for index, num1, num2 in differences:
#        print(f"Index {index}: File1 has {num1}, File2 has {num2}")
#else:
#    print("The files are identical (within tolerance).")
