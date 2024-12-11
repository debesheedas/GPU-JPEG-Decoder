import os
from image_converter import convert_to_444_jpeg

# This script reads all the images in the bechmark_dataset folder and converts each image into the 444 format. 
# It also renames the files by a simple index number

dataset_path = '../benchmarking_dataset'

for folder in os.listdir('../benchmarking_dataset'):
    index = 0
    for image in os.listdir(os.path.join(dataset_path, folder)):
        # print(image)
        image_path = os.path.join(dataset_path, folder, image)
        out_path = os.path.join(dataset_path, folder, str(index)+".jpeg")
        convert_to_444_jpeg(image_path, out_path)
        index += 1