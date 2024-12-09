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

const std::vector<uint16_t> MARKERS = {0xffd8, 0xffe0, 0xffdb, 0xffc0, 0xffc4, 0xffda};
const int C1 = 2841; // 2048*sqrt(2)*cos(1*pi/16)
const int C2 = 2676; // 2048*sqrt(2)*cos(2*pi/16)
const int C3 = 2408; // 2048*sqrt(2)*cos(3*pi/16)
const int C5 = 1609; // 2048*sqrt(2)*cos(5*pi/16)
const int C6 = 1108; // 2048*sqrt(2)*cos(6*pi/16)
const int C7 = 565;  // 2048*sqrt(2)*cos(7*pi/16)

/*
    Class for accessing the image channels of an image.
*/
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

class JPEGParser {
    private:
        // Parts of the jpeg file.
        std::string filename;
        uint8_t* readBytes;
        uint8_t* applicationHeader;
        uint8_t* startOfFrame;
        uint8_t* startOfScan;
        uint8_t* imageData;

        // Huffman Tables
        uint8_t *huffmanTable1, *huffmanTable2, *huffmanTable3, *huffmanTable4;
        std::unordered_map<int,HuffmanTree*> huffmanTrees;


        // Quant Tables
        uint8_t *quantTable1, *quantTable2;

        // Huffman Lookup Tables
        uint16_t* hf0codes;
        uint16_t* hf1codes; 
        uint16_t* hf16codes;
        uint16_t* hf17codes;
        int* hf0lengths; 
        int* hf1lengths; 
        int* hf16lengths; 
        int* hf17lengths;

        ImageChannels* channels;
        // Image features.
        int height;
        int width;
        int paddedWidth, paddedHeight, xBlocks, yBlocks;
        int imageDataLength;

        double* idctTable;
        int* zigzag;

        int *luminous, *chromRed, *chromYel;
        int *redOutput, *greenOutput, *blueOutput;

        // Methods for extracting and building blocks.
        __device__ int buildMCU(int* outBuffer, uint8_t* imageData, int bitOffset, uint8_t* quant, int& oldCoeff, uint16_t* dcHfcodes, int* dcHflengths, uint16_t* acHfcodes, int* acHflengths);
        __device__ int match_huffman_code(uint8_t* stream, int bit_offset, uint16_t* huff_codes, int* huff_bits, int &code, int &length);
        
    public:
        JPEGParser(std::string& imagePath);
        ~JPEGParser();
        void extract();
        void move();
        void decode();
        void write();
};