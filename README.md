# GPU-JPEG-Decoder
High performance JPEG Decoder for GPU

## Introduction

Goal: Implement a decoder (and optionally an encoder) for the JPEG image format that runs on a GPU.

# Step 1:
Implement a JPEG decoder using python.

## How to test decoder implementation

Please see the [testing](testing/) folder which contains the following scripts. PLEASE ``cd`` into the testing directory before running these scripts and be careful of the relative addressing used:
1. [display_image.py](testing/display_image.py) is for displaying the decoded channel arrays of the image. 
To avoid installing opencv on the cluster, we write the decoded channel arrays to a file. Then we read this file using this python script and display the image using opencv-python bindings which is installed on the cluster. To use this python script, modify the path to the file. For example:

```python   
with open("./cpp_output_arrays/5_200x200.array", "r") as file:
```
    
```python3 display_image.py```

2. [compare.py](testing/compare.py) is for comparing the decoded channel arrays of the image with the ground truth expected output of the decoder. For each implementation, the output files are collected in a separate folder, for example [cpp_output_arrays](testing/cpp_output_arrays). The expected output files are collected in the folder [ground_truth](testing/ground_truth). 
NOTE: This python script takes command line arguments. The first argument is the path to the folder containing the decoder implementation to be tested, eg. ``../cpp-implementation`` in case of testing the cpp implementation fo the decoder. If you pass only this argument to the script as follows, it will run the decoder on all the images in [testing/images](testing/images) and compare the output to the ground truth. The output files will also be written to the corresponding output directory.

```python3 compare.py ../cpp-implementation```

If you want to compare only a particular image, you can pass the second argument as the name of the image file in the [testing/images](testing/images) folder. For example, to compare only the image named 5_200x200.jpg, you can run the script as follows:

```python3 compare.py ../cpp-implementation ./images/5_200x200.jpg```




### Contributors:
1. Debeshee Das
2. Beste Guney
3. Emmy Zhou
4. Nayanika Debnath
5. Carlos Fernandez
