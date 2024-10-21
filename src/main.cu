#include <stdio.h>
#include<iostream>
#include "parser.h"
#include "huffmanTree.h"


int main() {
    // std::cout<<"Hello";
    std::string imagePath = "../profile.jpg";
    const JPEGParser* parser = new JPEGParser(imagePath);
    const HuffmanTree* tree = new HuffmanTree(parser->huffmanTable1);
}
