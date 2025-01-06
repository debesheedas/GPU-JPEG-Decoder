#ifndef HUFFMAN_TREE_H
#define HUFFMAN_TREE_H

#include <string>
#include <unordered_map>
#include <vector>
#include "../utils/stream.h"

/*
    Struct representing the nodes constructing the HuffmanTree.
    They carry the values in unsigned integers.
*/
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

    // ~HuffmanTreeNode() {
    //     delete left;
    //     delete right;
    // }
};

/*
    HuffmanTree class for creating the codes from the JPEG huffman tables.
*/
class HuffmanTree {
    std::vector<HuffmanTreeNode*> nodes;
    std::vector<uint8_t> bytes;
    HuffmanTreeNode* root;
    std::unordered_map<uint8_t, std::string> codes;

    void createNodes();
    bool addToTree(HuffmanTreeNode* root, HuffmanTreeNode* node, int position);
    void decodeTree(HuffmanTreeNode* node, std::string currentString);
    void clearTree(HuffmanTreeNode* node);  // Helper to clean up the tree

    public:
        HuffmanTree(const std::vector<uint8_t>& bytes);
        ~HuffmanTree();
        std::unordered_map<uint8_t, std::string> getCodes();
        uint8_t traverseTree(HuffmanTreeNode* cur, Stream* st);
        uint8_t getCode(Stream* st);
};

#endif