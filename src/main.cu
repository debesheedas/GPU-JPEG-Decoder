#include <stdio.h>
#include<iostream>
#include "parser.h"
#include "huffmanTree.h"


int main() {
    std::string imagePath = "../profile.jpg";
    const JPEGParser* parser = new JPEGParser(imagePath);
    HuffmanTree* tree = new HuffmanTree(parser->huffmanTable1);
}
