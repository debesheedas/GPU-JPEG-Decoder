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

struct JPEGParserData {
    uint8_t* imageData;   // Pointer to image data
    int* luminous;            // Pointer to luminance data
    int* chromRed;            // Pointer to chroma red data
    int* chromYel;            // Pointer to chroma yellow data
    int* zigzag_l;
    int* zigzag_r;
    int* zigzag_y;            
    double* idctTable;           // Pointer to IDCT table
    int idctWidth;              // Width of IDCT (e.g., 8)
    int idctHeight;             // Height of IDCT (e.g., 8)
    int width;                  // Image width
    int height;                 // Image height
    int xBlocks;                // Number of horizontal blocks
    int yBlocks;                // Number of vertical blocks
    int* redOutput;           // Pointer to red channel output
    int* greenOutput;         // Pointer to green channel output
    int* blueOutput;          // Pointer to blue channel output
    uint8_t* quantTable1;         // Pointer to first quantization table
    uint8_t* quantTable2;         // Pointer to second quantization table
    uint16_t* hf0codes;    // Huffman table 0 codes
    uint16_t* hf1codes;    // Huffman table 1 codes
    uint16_t* hf16codes;   // Huffman table 16 codes
    uint16_t* hf17codes;   // Huffman table 17 codes
    int* hf0lengths;            // Huffman table 0 lengths
    int* hf1lengths;            // Huffman table 1 lengths
    int* hf16lengths;           // Huffman table 16 lengths
    int* hf17lengths;           // Huffman table 17 lengths
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

__global__ void batchDecodeKernel(JPEGParserData* deviceStructs);
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
    public:
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
        int *zigzag_l, *zigzag_r, *zigzag_y;
        int *redOutput, *greenOutput, *blueOutput;

        // Methods for extracting and building blocks.
        __device__ int buildMCU(int* outBuffer, uint8_t* imageData, int bitOffset, uint8_t* quant, int& oldCoeff, uint16_t* dcHfcodes, int* dcHflengths, uint16_t* acHfcodes, int* acHflengths);
        __device__ int match_huffman_code(uint8_t* stream, int bit_offset, uint16_t* huff_codes, int* huff_bits, int &code, int &length);
        
    
        JPEGParser(std::string& imagePath);
        ~JPEGParser();
        void extract();
        void move();
        void decode();
        void write();
};