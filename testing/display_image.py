import cv2
import numpy as np

with open("./cpp_output_arrays/dawg.array", "r") as file:
    line = file.readline().strip()
    height, width = line.split(" ")
    print(height, width)
    channel_R = file.readline().strip().split(" ")
    channel_G = file.readline().strip().split(" ")
    channel_B = file.readline().strip().split(" ")

    print(len(channel_R), len(channel_G), len(channel_B))

red_channel = np.array(channel_R, dtype=np.uint8).reshape(int(height), int(width))  
green_channel = np.array(channel_G, dtype=np.uint8).reshape(int(height), int(width))
blue_channel = np.array(channel_B, dtype=np.uint8).reshape(int(height), int(width))
# Merge the channels into a single BGR image
image = cv2.merge([blue_channel, green_channel, red_channel])

# Display the image
cv2.imshow("RGB Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()