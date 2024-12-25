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
    int16_t* yCrCbChannels;
    int16_t* rgbChannels;
    int16_t* outputChannels;
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

__global__ void batchDecodeKernel(DeviceData* deviceStructs);
__device__ void decodeImage(uint8_t* imageData, int16_t* yCrCbChannels, int16_t* rgbChannels, int16_t* outputChannels, int width, int height, uint8_t* quantTables, uint16_t* hfCodes, int* hfLengths, int* zigzagLocations, int threadId, int blockSize, int* zigzagMap, int16_t* outputBlocks, int16_t* inputBlocks);
__global__ void decodeKernel(uint8_t* imageData, int16_t* yCrCbChannels, int16_t* rgbChannels, int16_t* outputChannels, int width, int height, uint8_t* quantTables, uint16_t* hfCodes, int* hfLengths, int* zigzagLocations);
void allocate(uint16_t*& hfCodes, int*& hfLengths, std::unordered_map<int,HuffmanTree*>& huffmanTrees, int16_t*& yCrCbChannels, int16_t*& rgbChannels, int16_t*& outputChannels, int width, int height, int*& zigzagLocations);
void extract(std::string imagePath, uint8_t*& quantTables, uint8_t*& imageData, int& width, int& height, std::unordered_map<int,HuffmanTree*>& huffmanTrees);
void clean(uint16_t*& hfCodes, int*& hfLengths, uint8_t*& quantTables, int16_t*& yCrCbChannels, int16_t*& rgbChannels, int16_t*& outputChannels, int*& zigzagLocations, uint8_t*& imageData, std::unordered_map<int,HuffmanTree*>& huffmanTrees);
void write(int16_t* outputChannels, int width, int height, std::string filename);

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

/*
    Class for accessing the image channels of an image.
*/
struct ImageChannels {
    std::vector<std::vector<int16_t>> channels;

    ImageChannels(int size)
        : channels(6, std::vector<int16_t>(size)) {}  // 6 channels, each with `size` elements

    // Accessor methods to get a reference to a specific channel
    std::vector<int16_t>& getY() { return channels[0]; }
    std::vector<int16_t>& getCr() { return channels[1]; }
    std::vector<int16_t>& getCb() { return channels[2]; }
    std::vector<int16_t>& getR() { return channels[3]; }
    std::vector<int16_t>& getG() { return channels[4]; }
    std::vector<int16_t>& getB() { return channels[5]; }
};

