#include <iostream>
#include "huffmanTree.h"

// Constructor for HuffmanTree
HuffmanTree::HuffmanTree(std::vector<uint8_t>& bytes) {
    for (int i = 0; i < bytes.size(); i++) {
        this->bytes.push_back(bytes[i]);
    }
    // this->bytes = bytes;
    this->createNodes();
    this->root = new HuffmanTreeNode(0,0,false);
    this->codes = new uint16_t[256];
    this->codeLengths = new int[256];

    for(const auto& n: this->nodes) {
        this->addToTree(this->root, n, n->length);
    }
    this->decodeTree(this->root, "");
    this->createCodes(this->root, 0, 0);
}

// HuffmanTree::~HuffmanTree() {
//     clearTree(this->root);
//     delete[] codes;
//     delete[] codeLengths;
// }

// Creates the nodes for the characters.
void HuffmanTree::createNodes() {
    // Extracting the length information.
    char lengths[16]; 
    for (int i = 0; i < 16; i++) {
        lengths[i] = bytes[i];
    }
    int offset = 16;
    
    for (int i = 0; i < 16; i++) {
        char curLength = lengths[i];
        for (int j = 0; j < curLength; j++) {
            char curVal = this->bytes[offset++];
            this->nodes.push_back(new HuffmanTreeNode(curVal, i+1, true));
        }
    }
}

// Create codes for the tree
void HuffmanTree::createCodes(HuffmanTreeNode* node, uint16_t value, int length) {
    if (node->isLeaf) {
        this->codes[node->val] = value;
        this->codeLengths[node->val] = length;
        return;
    }
    if (node->left != NULL) {
        this->createCodes(node->left, value << 1, length + 1);
    }
    if (node->right != NULL) {
        this->createCodes(node->right, (value << 1) + 1, length + 1);
    }
    return;
}

bool HuffmanTree::addToTree(HuffmanTreeNode* root, HuffmanTreeNode* node, int position) {
    if (root->isLeaf){
        return false;
    }
    if (position == 1) {
        if (root->left == NULL) {
            root->left = node;
        } else if (root->right == NULL) {
            root->right = node;
        } else {
            return false;
        }
        return true;
    }
    std::vector<int> indeces = {0,1};
    for (const int& i: indeces) {
        HuffmanTreeNode* internalNode = new HuffmanTreeNode(i, 0, false);
        if (i == 0) {
            if (root->left == NULL) {
                root->left = internalNode;
            } else {
                delete internalNode;
            }
            if (this->addToTree(root->left, node, position-1)) {
                return true;
            }
        } else {
            if (root->right == NULL) {
                root->right = internalNode;
            } else {
                delete internalNode;
            }
            if (this->addToTree(root->right, node, position - 1)) {
                return true;
            }
        }
    }
    return false;
}

void HuffmanTree::decodeTree(HuffmanTreeNode* node, std::string currentString) {
    // If we reach a leaf node we return the value.
    if (node->isLeaf) {
        this->codesString[node->val] = currentString;
        return;
    }
    if (node->left != NULL) {
        this->decodeTree(node->left, currentString + "0");
    }
    if (node->right != NULL) {
        this->decodeTree(node->right, currentString + "1");
    }
    return;
}

uint8_t HuffmanTree::traverseTree(HuffmanTreeNode* cur, Stream* st) {
    if (cur->isLeaf) {
        return cur->val;
    }
    int val = (int)(st->getBit());

    if ((cur->right != NULL) && val == 1) {
        return this->traverseTree(cur->right, st);
    } else if ((cur->left != NULL) && val == 0) {
        return this->traverseTree(cur->left, st);
    }

    return cur->val;
}

void HuffmanTree::clearTree(HuffmanTreeNode* node) {
    if (!node) return;

    clearTree(node->left);
    clearTree(node->right);
    delete node;
}

uint8_t HuffmanTree::getCode(Stream* st) {
    return this->traverseTree(this->root, st);
} 

void HuffmanTree::printCodes() {
    for (int i = 0; i < 256; i++) {
        std::cout << i << "  " << (int)this->codes[i] << " " << (int) this->codeLengths[i] << std::endl;
    }
}