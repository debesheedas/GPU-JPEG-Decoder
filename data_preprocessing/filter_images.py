import os
from PIL import Image
from shutil import move, rmtree
from collections import defaultdict

def organize_images_by_size(folder_path):
    # Dictionary to store images by size
    size_dict = defaultdict(list)
    
    # Walk through all files in the folder and subfolders
    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            
            # Check if it is a file and an image
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    size_dict[(width, height)].append(file_path)
            except Exception as e:
                print(f"Skipping file {filename}: {e}")

    # Create folders for each size and move images
    for size, file_paths in size_dict.items():
        size_folder = os.path.join(folder_path, f"{size[0]}x{size[1]}")
        
        # Create the folder if it doesn't exist
        os.makedirs(size_folder, exist_ok=True)
        
        # Move each image to the corresponding size folder
        for file_path in file_paths:
            move(file_path, os.path.join(size_folder, os.path.basename(file_path)))
        
        # Check the number of images in the folder, and delete if less than 6
        if len(file_paths) <= 50:
            print(f"Removing folder {size_folder} (less than 20 images)")
            rmtree(size_folder)  # Remove the folder and its contents
        else:
            print(f"Moved images to folder: {size_folder}")

# Folder path containing the images
folder_path = "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/Images"

# Organize images by size
organize_images_by_size(folder_path)
