#include <stdio.h>
#include<iostream>
#include <cuda_runtime.h>
#include "src/parser.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Please provide the name of the image file to be decompressed." << std::endl;
        return 1;
    }

    // Reading the image bytes
    std::string imagePath = argv[1];
    // Extract the file name of the image file from the file path
    fs::path file_path(imagePath);
    std::string filename = file_path.filename().string();

    uint8_t* quantTables;
    int16_t* yCrCbChannels;
    int16_t* rgbChannels;
    int16_t* outputChannels;
    int* zigzagLocations;

    std::vector<uint8_t> imageData;
    // Stream* imageStream = new Stream(imageData);
    int width = 0;
    int height = 0;
    std::unordered_map<int,HuffmanTree*> huffmanTrees;
    // std::cout << "debug 1" << std::endl;
    // Extracting the byte chunks
    extract(imagePath, quantTables, imageData, width, height, huffmanTrees);
    // std::cout << imageData[0] << " is the image data" << std::endl;
    // Allocating memory for the arrays
    // std::cout << "debug 2" << std::endl;
    allocate(yCrCbChannels, rgbChannels, outputChannels, width, height, zigzagLocations);
    // std::cout << "debug 3" << std::endl;
    performHuffmanDecoding(imageData, yCrCbChannels, huffmanTrees, width, height);
    // std::cout << "debug 4" << std::endl;
    decodeKernel<<<1, 1024>>>(yCrCbChannels, rgbChannels, outputChannels, width, height, quantTables, zigzagLocations);
    // std::cout << "debug 5" << std::endl;
    cudaDeviceSynchronize();
    
    write(outputChannels, width, height, filename);
    // std::cout << "debug 6" << std::endl;
    clean(quantTables, yCrCbChannels, rgbChannels, outputChannels, zigzagLocations, huffmanTrees);
    // std::cout << "debug 7" << std::endl;
}