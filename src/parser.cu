#include <stdio.h>
#include <fstream>
#include <vector>
#include <iterator>
#include <iostream>
#include <algorithm>
#include "parser.h"
#include "utils.h"

// TODO(bguney): Implement a stream class for global stream state rather than copy.

JPEGParser::JPEGParser(std::string& imagePath){
        std::ifstream input(imagePath, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()));
        input.close();
        JPEGParser::extract(bytes);
}

void JPEGParser::extract(std::vector<char> bytes) {        
    int i = 0;
    int tableSize = 0;
    
    while(1) {
        // Application header starts with 0xffe0
        if((unsigned char) bytes[i] == 0xff && (unsigned char) bytes[i+1] == 0xe0){
            std::cout<< "Extracting Application Header" << std::endl;
            tableSize = ByteUtil::getSize(bytes[i+2], bytes[i+3]);
            std::copy(bytes.begin() + (i+4), bytes.begin() + (i+2+tableSize), std::back_inserter(this->applicationHeader));
            i = i+1+tableSize; // Increment i by one less than extracted segment because of the i++ at the end of this while loop
        } 
        else if ((unsigned char) bytes[i] == 0xff && (unsigned char) bytes[i+1] == 0xdb) {
            std::cout<< "Extracting Quant Tables" << std::endl;
            // Extract Quantization Table 1
            std::copy(bytes.begin() + (i+5), bytes.begin() + (i+5+64), std::back_inserter(this->quantTable1));
            i = i + 5 + 64;
            // Extract Quantization Table 2
            std::copy(bytes.begin() + (i+5), bytes.begin() + (i+5+64), std::back_inserter(this->quantTable2));
            i = i + 5 + 63; // Increment i by one less than extracted segment because of the i++ at the end of this while loop
        }
        else if ((unsigned char) bytes[i] == 0xff && (unsigned char) bytes[i+1] == 0xc0) {
            std::cout<< "Extracting Start of Frame" << std::endl;
            std::copy(bytes.begin() + (i+4), bytes.begin() + (i+19), std::back_inserter(this->startOfFrame));
            i = i + 18; // Increment i by one less than extracted segment because of the i++ at the end of this while loop
        }
        else if ((unsigned char) bytes[i] == 0xff && (unsigned char) bytes[i+1] == 0xc4) {
            std::cout<< "Extracting Huffman Tables" << std::endl;
            // Get the table size for Huffman Table 1
            tableSize = ByteUtil::getSize(bytes[i+2], bytes[i+3]);
            std::copy(bytes.begin() + (i+4), bytes.begin() + (i+2+tableSize), std::back_inserter(this->huffmanTable1));
            i = i+2+tableSize;
            // Identify table size and extract Huffman Table 2
            tableSize = ByteUtil::getSize(bytes[i+2], bytes[i+3]);
            std::copy(bytes.begin() + (i+4), bytes.begin() + (i+2+tableSize), std::back_inserter(this->huffmanTable2));
            i = i+2+tableSize;
            // Identify table size and extract Huffman Table 3
            tableSize = ByteUtil::getSize(bytes[i+2], bytes[i+3]);
            std::copy(bytes.begin() + (i+4), bytes.begin() + (i+2+tableSize), std::back_inserter(this->huffmanTable3));
            i = i+2+tableSize;
            // Identify table size and extract Huffman Table 4
            tableSize = ByteUtil::getSize(bytes[i+2], bytes[i+3]);
            std::copy(bytes.begin() + (i+4), bytes.begin() + (i+2+tableSize), std::back_inserter(this->huffmanTable4));
            i = i+1+tableSize; // Increment i by one less than extracted segment because of the i++ at the end of this while loop
        }
        else if ((unsigned char) bytes[i] == 0xff && (unsigned char) bytes[i+1] == 0xda) {
            std::cout<< "Start of Scan" << std::endl;
            tableSize = ByteUtil::getSize(bytes[i+2], bytes[i+3]);
            std::copy(bytes.begin() + (i+4), bytes.begin() + (i+2+tableSize), std::back_inserter(this->startOfScan));
            i = i+2+tableSize;
            std::cout<< "Extracting Image Data" << std::endl;
            while (!((unsigned char) bytes[i] == 0xff && (unsigned char) bytes[i+1] == 0xd9)) {
                if (!((unsigned char) bytes[i] == 0x00 && (unsigned char) bytes[i+1] == 0xff)) {
                    this->imageData.push_back(bytes[i]);
                }
                i++;
            }
            // TODO: Remove 0x00 if it precedes 0xff
            break;
        }
        i++;
    }
}