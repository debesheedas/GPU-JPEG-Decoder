#include <stdio.h>
#include <string>
#include <iostream>
#include <unordered_map>
#include <bitset>
#include <vector>
#include "utils/stream.h"

struct HuffmanTreeNode {
    HuffmanTreeNode* left;
    HuffmanTreeNode* right;
    uint8_t val;
    int length;
    bool isLeaf;

    HuffmanTreeNode(uint8_t val, int length, bool isLeaf) {
        this->val = val;
        this->length = length;
        this->isLeaf = isLeaf;
        this->left = NULL;
        this->right = NULL;
    }
};

class HuffmanTree {
    std::vector<HuffmanTreeNode*> nodes;
    std::vector<uint8_t> bytes;

    public:
        HuffmanTree(const std::vector<uint8_t>& bytes);
        void createNodes();
        bool addToTree(HuffmanTreeNode* root, HuffmanTreeNode* node, int position);
        void decodeTree(HuffmanTreeNode* node, std::string currentString);
        std::unordered_map<uint8_t, std::string> getCodes();
        uint8_t traverseTree(HuffmanTreeNode* cur, Stream* st);
        uint8_t getCode(Stream* st);
        HuffmanTreeNode* root;
        std::unordered_map<uint8_t, std::string> codes;
};