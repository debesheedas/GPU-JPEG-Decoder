import unittest
import subprocess
import sys
import os
import numpy as np
import cv2

TOL = 1e-4

def read_output(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f:
            elements = line.strip().split()
            data.append([float(element) for element in elements])
    return data

def load_and_save_array_image(array_file_path, output_image_path):
    with open(array_file_path, "r") as file:
        line = file.readline().strip()
        height, width = map(int, line.split(" "))
        channel_R = file.readline().strip().split(" ")
        channel_G = file.readline().strip().split(" ")
        channel_B = file.readline().strip().split(" ")

    # Convert channels to numpy arrays and reshape to image dimensions
    red_channel = np.array(channel_R, dtype=np.uint8).reshape(height, width)
    green_channel = np.array(channel_G, dtype=np.uint8).reshape(height, width)
    blue_channel = np.array(channel_B, dtype=np.uint8).reshape(height, width)

    # Merge the channels into a single BGR image
    image = cv2.merge([blue_channel, green_channel, red_channel])

    # Save the image
    cv2.imwrite(output_image_path, image)

def test_array_equality(implementation_folder, image_path):    
    decoder_executable = os.path.join(implementation_folder, "decoder")

    subprocess.run(["./"+decoder_executable, image_path], check=True)
    
    implementation_type = os.path.basename(implementation_folder.strip('/')).split('-')[0]
    
    image_name = os.path.basename(image_path).replace(".jpg", ".array")
    ground_truth_path = os.path.join("./ground_truth/", image_name)
    decoder_output_path = os.path.join(implementation_type+"_output_arrays", image_name)

    ground_truth = read_output(ground_truth_path)
    decoder_output = read_output(decoder_output_path)

    # Save ground truth and decoder output images
    #load_and_save_array_image(ground_truth_path, f"ground_truth_{image_name}.jpeg")
    load_and_save_array_image(decoder_output_path, f"decoder_output_{image_name}.jpeg")

    assert len(ground_truth) == len(decoder_output)
    if ground_truth == decoder_output:
        print("Congratulations! Output matches ground truth!", image_name)
        return True
    else:
        differences = [(abs(ground_truth[1][i] - decoder_output[1][i]), ground_truth[1][i], decoder_output[1][i]) for i in range(len(ground_truth[1])) if abs(ground_truth[1][i] - decoder_output[1][i]) > TOL]
        abs_diff = [diff[0] for diff in differences]
        print("Max difference:", max(abs_diff))       
        print("Output does not match the ground truth!", image_name)
    return False

if __name__ == '__main__':
    num_args = len(sys.argv) - 1
    if num_args == 0:
        print("Please enter command line arguments: Path to the folder with decoder implementation, and then optionally path to single image")
    elif num_args == 1:
        results = []
        implementation_folder = sys.argv[1]
        os.makedirs("./images_saved", exist_ok=True)  # Create directory for saved images
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
        os.makedirs("./images_saved", exist_ok=True)  # Create directory for saved images
        test_array_equality(implementation_folder, image_path)
    else:
        print("Too many arguments")
        sys.exit()
