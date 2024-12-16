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

    uint16_t* hfCodes; 
    int* hfLengths;
    uint8_t* quantTables;
    int16_t* yCrCbChannels;
    int16_t* rgbChannels;
    int16_t* outputChannels;
    int* zigzagLocations;

    uint8_t* imageData;
    int width = 0;
    int height = 0;
    std::unordered_map<int,HuffmanTree*> huffmanTrees;

    // Extracting the byte chunks
    extract(imagePath, quantTables, imageData, width, height, huffmanTrees);
    // Allocating memory for the arrays
    allocate(hfCodes, hfLengths, huffmanTrees, yCrCbChannels, rgbChannels, outputChannels, width, height, zigzagLocations);
    
    decodeKernel<<<1, 256>>>(imageData, yCrCbChannels, rgbChannels, outputChannels, width, height, quantTables, hfCodes, hfLengths, zigzagLocations);
    cudaDeviceSynchronize();
    
    write(outputChannels, width, height, filename);
    clean(hfCodes, hfLengths, quantTables, yCrCbChannels, rgbChannels, outputChannels, zigzagLocations, imageData, huffmanTrees);
}