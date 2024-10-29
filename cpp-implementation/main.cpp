#include <stdio.h>
#include<iostream>
#include "parser.h"

int main() {
    std::string imagePath = "../profile.jpg";
    JPEGParser* parser = new JPEGParser(imagePath);
    parser->decode_start_of_scan();
    // HuffmanTree* tree = new HuffmanTree(parser->huffmanTable1);
}
