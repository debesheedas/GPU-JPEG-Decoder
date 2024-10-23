#include <stdio.h>
#include <string>
#include <iostream>
#include <unordered_map>
#include <bitset>
#include <vector>

struct HuffmanTreeNode {
    HuffmanTreeNode* left;
    HuffmanTreeNode* right;
    char val;
    int length;
    bool isLeaf;

    HuffmanTreeNode(char val, int length, bool isLeaf) {
        this->val = val;
        this->length = length;
        this->isLeaf = isLeaf;
        this->left = NULL;
        this->right = NULL;
    }
};

class HuffmanTree {
    std::vector<HuffmanTreeNode*> nodes;
    HuffmanTreeNode* root;
    std::vector<char> bytes;
    std::unordered_map<char, std::string> codes;

    public:
        HuffmanTree(const std::vector<char>& bytes);
        void createNodes();
        bool addToTree(HuffmanTreeNode* root, HuffmanTreeNode* node, int position);
        void decodeTree(HuffmanTreeNode* node, std::string currentString);
        std::unordered_map<char, std::string> getCodes();
};