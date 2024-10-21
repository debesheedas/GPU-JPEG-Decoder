#include <stdio.h>
#include <vector>
#include <string>

class JPEGParser {
    public:
    JPEGParser(std::string& imagePath);
    void extract(std::vector<char> bytes);
    int data;
    std::vector<char> applicationHeader;
    std::vector<char> quantTable1;
    std::vector<char> quantTable2;
    std::vector<char> startOfFrame;
    std::vector<char> huffmanTable1;
    std::vector<char> huffmanTable2;
    std::vector<char> huffmanTable3;
    std::vector<char> huffmanTable4;
    std::vector<char> startOfScan;
    std::vector<char> imageData;
};