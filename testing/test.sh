cd ..
cd cudaU-implementation
make
./decoder /home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/testing/images/9_64x64.jpg
cd ..
cd testing
python3 display_image.py /home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/testing/cudaU_output_arrays/9_64x64.array
# python3 compare.py ../cudaU-implementation /home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/testing/images/9_64x64.jpg