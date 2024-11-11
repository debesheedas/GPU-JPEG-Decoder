import unittest
import subprocess
import sys
import os


TOL = 2

def read_output(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f:
            elements = line.strip().split()
            data.append([float(element) for element in elements])

    return data

def test_array_equality(implementation_folder, image_path):
    # print("Testing equality with ground truth")
    # print(implementation_folder, image_path)
    
    decoder_executable = os.path.join(implementation_folder, "decoder")

    subprocess.run(["./"+decoder_executable, image_path], check=True)
    # subprocess.run(["python3", "./baseline.py", "python_output.txt"], check=True)
    
    implementation_type = os.path.basename(implementation_folder).split('-')[0]

    image_name = os.path.basename(image_path).replace(".jpg", ".array")
    # print(implementation_type)
    # print(image_name)
    ground_truth = read_output(os.path.join("./ground_truth/", image_name))
    decoder_output = read_output(os.path.join(implementation_type+"_output_arrays", image_name))
    assert(len(ground_truth) == len(decoder_output))
    for row1, row2 in zip(ground_truth, decoder_output):
        assert(len(row1) == len(row2)), "Mismatch in row lengths"
        for val1, val2 in zip(row1, row2):
            if abs(val1 - val2) > TOL:
                return False 

    print("Congratulations! Output matches ground truth with tolerance!", image_name)
    return True


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

    