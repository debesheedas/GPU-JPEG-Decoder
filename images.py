import os
from PIL import Image
from shutil import move
from collections import defaultdict

def organize_images_by_size(folder_path):
    # Dictionary to store images by size
    size_dict = defaultdict(list)
    
    # Loop through files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it is a file (not a directory)
        if os.path.isfile(file_path):
            # Check if it's an image by opening it with PIL
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    size_dict[(width, height)].append(file_path)
            except Exception as e:
                print(f"Skipping file {filename}: {e}")
        else:
            print(f"Skipping directory {filename}: Not a file")

    # Create folders for each size and move images
    for size, file_paths in size_dict.items():
        size_folder = os.path.join(folder_path, f"{size[0]}x{size[1]}")
        
        # Create the folder if it doesn't exist
        os.makedirs(size_folder, exist_ok=True)
        
        # Move each image to the corresponding size folder
        for file_path in file_paths:
            move(file_path, os.path.join(size_folder, os.path.basename(file_path)))
        print(f"Moved images to folder: {size_folder}")

# Folder path containing the images
folder_path = "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/Images/n02085620-Chihuahua"

# Organize images by size
organize_images_by_size(folder_path)
