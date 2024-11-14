#include <fstream>
#include <vector>
#include <iterator>
#include <unordered_map>
// #include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include "../utils/color.h"
#include "idct.h"
#include "huffmanTree.h"

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
        std::vector<uint8_t> readBytes;
        std::vector<uint8_t> applicationHeader;
        std::unordered_map<int,std::vector<uint8_t>> quantTables;
        std::vector<uint8_t> startOfFrame;
        std::unordered_map<int,std::vector<uint8_t>> huffmanTables;
        std::unordered_map<int,HuffmanTree*> huffmanTrees;
        std::vector<uint8_t> startOfScan;
        std::vector<uint8_t> imageData;
        ImageChannels* channels;

        // Image features.
        int height;
        int width;

        // Methods for extracting and building blocks.
        
        void buildMCU(std::vector<int>& arr, Stream* imageStream, int hf, int quant, int& oldcoeff, int validWidth, int validHeight);

    public:
        JPEGParser(std::string& imagePath);
        void extract();
        void decode();
        void write();
};