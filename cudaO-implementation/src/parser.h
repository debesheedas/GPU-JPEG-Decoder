#include <fstream>
#include <vector>
#include <iterator>
#include <unordered_map>
// #include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include "../utils/utils.h"
#include "huffmanTree.h"
#include <cuda_runtime.h>

#ifdef __APPLE__
    // #include <filesystem>
    namespace fs = std::__fs::filesystem;
#else
    // #include <filesystem>
    namespace fs = std::filesystem;
#endif

struct HostData {
    std::string imagePath;
    std::unordered_map<int,HuffmanTree*> huffmanTrees;
};

struct DeviceData {
    uint8_t* imageData;   
    uint16_t* hfCodes; 
    int* hfLengths;
    uint8_t* quantTables;
    int* yCrCbChannels;
    int* rgbChannels;
    int* outputChannels;
    int* zigzagLocations;         
    int width;                  // Image width
    int height;                 // Image height
};

const std::vector<uint16_t> MARKERS = {0xffd8, 0xffe0, 0xffdb, 0xffc0, 0xffc4, 0xffda};
const int C1 = 2841; // 2048*sqrt(2)*cos(1*pi/16)
const int C2 = 2676; // 2048*sqrt(2)*cos(2*pi/16)
const int C3 = 2408; // 2048*sqrt(2)*cos(3*pi/16)
const int C5 = 1609; // 2048*sqrt(2)*cos(5*pi/16)
const int C6 = 1108; // 2048*sqrt(2)*cos(6*pi/16)
const int C7 = 565;  // 2048*sqrt(2)*cos(7*pi/16)

// __global__ void decodeKernel(uint8_t* imageData, int* arr_l, int* arr_r, int* arr_y, int* zigzag_l, int* zigzag_r, 
//                                 int* zigzag_y, double* idctTable, int validHeight, 
//                                 int validWidth, int width, int height, int xBlocks, int yBlocks, int* redOutput, 
//                                 int* greenOutput, int* blueOutput, uint8_t* quant1, uint8_t* quant2, 
//                                 uint16_t* hf0codes, uint16_t* hf1codes, uint16_t* hf16codes, uint16_t* hf17codes,
//                                 int* hf0lengths, int* hf1lengths, int* hf16lengths, int* hf17lengths);

__global__ void batchDecodeKernel(DeviceData* deviceStructs);
__device__ void decodeImage(uint8_t* imageData, int* yCrCbChannels, int* rgbChannels, int* outputChannels, int width, int height, uint8_t* quantTables, uint16_t* hfCodes, int* hfLengths, int* zigzagLocations, int threadId, int blockSize);
void allocate(uint16_t*& hfCodes, int*& hfLengths, std::unordered_map<int,HuffmanTree*>& huffmanTrees, int*& yCrCbChannels, int*& rgbChannels, int*& outputChannels, int width, int height, int*& zigzagLocations);
void extract(std::string imagePath, uint8_t*& quantTables, uint8_t*& imageData, int& width, int& height, std::unordered_map<int,HuffmanTree*>& huffmanTrees);
void clean(uint16_t*& hfCodes, int*& hfLengths, uint8_t*& quantTables, int*& yCrCbChannels, int*& rgbChannels, int*& outputChannels, int*& zigzagLocations, uint8_t*& imageData, std::unordered_map<int,HuffmanTree*>& huffmanTrees);
/*
    Class for accessing the image channels of an image.
*/
const int zigzagEntries[64] = {
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63
    };

struct ImageChannels {
    std::vector<std::vector<int>> channels;

    ImageChannels(int size)
        : channels(6, std::vector<int>(size)) {}  // 6 channels, each with `size` elements

    // Accessor methods to get a reference to a specific channel
    std::vector<int>& getY() { return channels[0]; }
    std::vector<int>& getCr() { return channels[1]; }
    std::vector<int>& getCb() { return channels[2]; }
    std::vector<int>& getR() { return channels[3]; }
    std::vector<int>& getG() { return channels[4]; }
    std::vector<int>& getB() { return channels[5]; }
};

