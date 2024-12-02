import unittest
import subprocess
import sys
import os


TOL = 8

def read_output(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f:
            elements = line.strip().split()
            data.append([float(element) for element in elements])

    return data

def test_array_equality(implementation_folder, image_path):    
    decoder_executable = os.path.join(implementation_folder, "decoder")

    subprocess.run(["./"+decoder_executable, image_path], check=True)
    
    implementation_type = os.path.basename(implementation_folder).split('-')[0]

    image_name = os.path.basename(image_path).replace(".jpg", ".array")
    ground_truth = read_output(os.path.join("./ground_truth/", image_name))
    decoder_output = read_output(os.path.join(implementation_type+"_output_arrays", image_name))
    assert(len(ground_truth) == len(decoder_output))
    differences = []
    for i in range(len(ground_truth[1])):
        if abs(ground_truth[1][i] - decoder_output[1][i]) > TOL:
            differences.append((i, ground_truth[1][i], decoder_output[1][i]))

    if not differences:
        print("Congratulations! Output matches ground truth!", image_name)
        return True
    else:
        print(f"Differences exceeding tolerance ({TOL}):", differences)
        print("Output does not match the ground truth!", image_name)
        return False


if __name__ == '__main__':
    num_args = len(sys.argv) - 1
    print(num_args)
    if num_args == 0:
        print("Please enter command line arguments: Path to the folder with decoder implementation, and then optionallly path to single image")
    elif num_args == 1:
        results = []
        implementation_folder = sys.argv[1]
        for image in os.listdir("./images"):
            image_path = os.path.join("./images", image)
            if os.path.isfile(image_path):
                results.append(test_array_equality(implementation_folder, image_path))
        if all(results):
            print("All test cases passed!") 
        else:
            print(f'{results.count(False)} test cases failed')
    elif num_args == 2:
        implementation_folder = sys.argv[1]
        image_path = sys.argv[2]
        test_array_equality(implementation_folder, image_path)
    else:
        print("Too many arguments")
        sys.exit()