#include <fstream>
#include <vector>
#include <iterator>
#include <unordered_map>
#include <iostream>
#include <filesystem>

#include "../utils/color.h"
#include "../utils/stream.h"
#include <jpeglib.h>

#ifdef __APPLE__
    // #include <filesystem>
    namespace fs = std::__fs::filesystem;
#else
    // #include <filesystem>
    namespace fs = std::filesystem;
#endif

const std::vector<uint16_t> MARKERS = {0xffd8, 0xffe0, 0xffdb, 0xffc0, 0xffc4, 0xffda};

class JPEGParser {
    private:
        std::string filename;
        std::vector<uint8_t> readBytes;      // Raw image data from file
        std::vector<uint8_t> decompressedData; // Decompressed pixel data
        int width, height;

    public:
        JPEGParser(std::string& imagePath);
        void decode();
        void write();
};