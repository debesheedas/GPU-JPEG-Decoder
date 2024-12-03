#include <fstream>
#include <vector>
#include <iterator>
#include <unordered_map>
// #include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include "../utils/color.h"
#include "idct.h"
#include "huffmanTree2.h"
#include <cuda_runtime.h>

#ifdef __APPLE__
    // #include <filesystem>
    namespace fs = std::__fs::filesystem;
#else
    // #include <filesystem>
    namespace fs = std::filesystem;
#endif

const std::vector<uint16_t> MARKERS = {0xffd8, 0xffe0, 0xffdb, 0xffc0, 0xffc4, 0xffda};

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

        ImageChannels* channels;
        // Image features.
        int height;
        int width;
        int imageDataLength;

        double* idctTable;
        int* zigzag;

        // Methods for extracting and building blocks.
        void buildMCU(int* arr, Stream* imageStream, int hf, int quant, int& oldcoeff);
    public:
        JPEGParser(std::string& imagePath);
        ~JPEGParser();
        void extract();
        void decode();
        void write();
};