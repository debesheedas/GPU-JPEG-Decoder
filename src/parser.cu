#include <stdio.h>
#include <fstream>
#include <vector>
#include <iterator>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include "parser.h"


// TODO(bguney): Implement a stream class for global stream state rather than copy.

JPEGParser::JPEGParser(std::string& imagePath){
        std::ifstream input(imagePath, std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()));
        input.close();
        JPEGParser::extract(bytes);
}

void JPEGParser::extract(std::vector<uint8_t>& bytes) {        
    uint16_t tableSize = 0;
    uint8_t header = 0;

    // Using the Stream class for reading bytes.
    Stream* stream = new Stream(bytes);

    while (true) {
        uint16_t marker = stream->getMarker();
        // std::cout << std::hex << (int)marker << std::endl;

        if (marker == 0xffd8) {
            std::cout << "Start of the image " << std::endl;
        } else if (marker == 0xffe0) {
            std::cout<< "Extracting Application Header" << std::endl;
            tableSize = stream->getMarker();
            stream->getNBytes(this->applicationHeader, int(tableSize - 2));
        } else if (marker == 0xffdb) {
            std::cout<< "Extracting Quant Tables" << std::endl;
            stream->getMarker();
            uint8_t destination = stream->getByte();
            stream->getNBytes(this->quantTables[0], 64);
            if(stream->getMarker() == 0xffdb) {
                stream->getMarker();
                destination = stream->getByte();
                stream->getNBytes(this->quantTables[1], 64);
            } else {
                std::cout << " Something went wrong at parsing second quant table." << std::endl;
            }
        } else if (marker == 0xffc0) {
            std::cout<< "Extracting Start of Frame" << std::endl;
            tableSize = stream->getMarker();
            stream->getNBytes(this->startOfFrame, (int) tableSize - 2);
            Stream* frame = new Stream(this->startOfFrame);
            int precision = frame->getByte();
            this->height = frame->getMarker();
            this->width = frame->getMarker();
        } else if (marker == 0xffc4) {
            std::cout<< "Extracting Huffman Tables" << std::endl;
            tableSize = stream->getMarker();
            header = stream->getByte();
            stream->getNBytes(this->huffmanTables[0], (int) tableSize - 3);
            this->huffmanTrees[header] = new HuffmanTree(this->huffmanTables[0]);

            if (stream->getMarker() == 0xffc4) {
                tableSize = stream->getMarker();
                header = stream->getByte();
                stream->getNBytes(this->huffmanTables[1], (int) tableSize - 3);
                this->huffmanTrees[header] = new HuffmanTree(this->huffmanTables[1]); 
            }

            if (stream->getMarker() == 0xffc4) {
                tableSize = stream->getMarker();
                header = stream->getByte();
                stream->getNBytes(this->huffmanTables[2], (int) tableSize - 3);
                this->huffmanTrees[header] = new HuffmanTree(this->huffmanTables[2]);
            }

            if (stream->getMarker() == 0xffc4) {
                tableSize = stream->getMarker();
                header = stream->getByte();
                stream->getNBytes(this->huffmanTables[3], (int) tableSize - 3);
                this->huffmanTrees[header] = new HuffmanTree(this->huffmanTables[3]);
            }
        } else if (marker == 0xffda) {
            std::cout<< "Start of Scan" << std::endl;
            tableSize = stream->getMarker();
            stream->getNBytes(this->startOfScan, (int) tableSize - 2);
            std::cout<< "Extracting Image Data" << std::endl;
            uint8_t curByte, nextByte = 0;

            do{
                curByte = stream->getByte();
                nextByte = stream->getByte();

                if (curByte != 0xff && nextByte != 0x00) {
                    this->imageData.push_back(curByte);
                }
            } while (curByte != 0xff && nextByte != 0xd9);

            break;
        }
    }   
}

void JPEGParser::buildMCU(std::vector<int8_t>& arr, Stream* imageStream, int hf, int quant, int oldCoeff) {
    uint8_t code = this->huffmanTrees[hf]->getCode(imageStream);
    int bits = imageStream->getNBits(code);
    int decoded = ByteUtil::DecodeNumber(code, bits);
    int dcCoeff = decoded + oldCoeff;

    arr[0] = dcCoeff * (int) this->quantTables[quant][0];
    int length = 1;
    while(length < 64) {
        code = this->huffmanTrees[16+hf]->getCode(imageStream);

        if(code == 0) {
            break;
        }

        // The first part of the AC key_len 
        // is the number of leading zeros
        if (code > 15) {
            length += (code >> 4);
            code = code & 0x0f;
        }

        bits = imageStream->getNBits(code);

        if (length < 64) {
            decoded = ByteUtil::DecodeNumber(code, bits);
            arr[length] = decoded * this->quantTables[quant][length];
            length++;
        }
        length++;
    }
    // TODO: Perform zigzag on mat and call IDCT

}

void JPEGParser::decode_start_of_scan(){
    int oldLumCoeff = 0;
    int oldCbdCoeff = 0;
    int oldCrdCoeff = 0;
    int yBlocks = this->height / 8;
    int xBlocks = this->width / 8;

    Stream* imageStream = new Stream(this->imageData);
    std::vector<std::vector<std::vector<int8_t>>> luminous(xBlocks, std::vector<std::vector<int8_t>>(yBlocks, std::vector<int8_t>(64,0)));
    std::vector<std::vector<std::vector<int8_t>>> chromRed(xBlocks, std::vector<std::vector<int8_t>>(yBlocks, std::vector<int8_t>(64,0)));
    std::vector<std::vector<std::vector<int8_t>>> chromYel(xBlocks, std::vector<std::vector<int8_t>>(yBlocks, std::vector<int8_t>(64,0)));
    for (int y = 0; y < yBlocks; y++) {
        for (int x = 0; x < xBlocks; x++) {
            this->buildMCU(luminous[x][y], imageStream, 0, 0, oldLumCoeff);
            this->buildMCU(chromRed[x][y], imageStream, 1, 1, oldCbdCoeff);
            this->buildMCU(chromYel[x][y], imageStream, 1, 1, oldCrdCoeff);
        }
    }
}
