#include <stdio.h>
#include<iostream>
#include "src/parser.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Please provide the name of the image file to be decompressed." << std::endl;
        return 1;
    }
    std::string imagePath = argv[1];
    JPEGParser* parser = new JPEGParser(imagePath);
    parser->extract();
    parser->decode();
    parser->write();
    return 0;
}
