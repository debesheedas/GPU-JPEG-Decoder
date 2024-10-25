#include <stdio.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "utils.h"
#include "huffmanTree.h"

class JPEGParser {
    public:
    JPEGParser(std::string& imagePath);
    void extract(std::vector<uint8_t>& bytes);
    void decode_start_of_scan();
    void buildMCU(std::vector<int8_t>& arr, Stream* imageStream, int hf, int quant, int oldcoeff);
    int data;
    std::vector<uint8_t> applicationHeader;
    // std::vector<uint8_t> quantTable1;
    // std::vector<uint8_t> quantTable2;
    std::unordered_map<int,std::vector<uint8_t>> quantTables;
    std::vector<uint8_t> startOfFrame;
    std::unordered_map<int,std::vector<uint8_t>> huffmanTables;
    std::unordered_map<int,HuffmanTree*> huffmanTrees;
    std::vector<uint8_t> startOfScan;
    std::vector<uint8_t> imageData;
    int height;
    int width;
};