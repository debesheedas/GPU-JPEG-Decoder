import argparse
import cv2
import numpy as np

def load_and_display_array_image(array_file_path):

    with open(array_file_path, "r") as file:

        line = file.readline().strip()
        height, width = map(int, line.split(" "))
        print(f"Image dimensions: {height}x{width}")


        channel_R = file.readline().strip().split(" ")
        channel_G = file.readline().strip().split(" ")
        channel_B = file.readline().strip().split(" ")

        print("Channel lengths:", len(channel_R), len(channel_G), len(channel_B))

    # Convert channels to numpy arrays and reshape to image dimensions
    red_channel = np.array(channel_R, dtype=np.uint8).reshape(height, width)
    green_channel = np.array(channel_G, dtype=np.uint8).reshape(height, width)
    blue_channel = np.array(channel_B, dtype=np.uint8).reshape(height, width)

    # Merge the channels into a single BGR image
    image = cv2.merge([blue_channel, green_channel, red_channel])

    cv2.imwrite("image.jpeg",image)
    # cv2.imshow("RGB Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display an image from a .array file.")
    parser.add_argument("array_file_path", type=str, help="Path to the .array file containing image data")

    args = parser.parse_args()

    load_and_display_array_image(args.array_file_path)