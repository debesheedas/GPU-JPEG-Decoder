#include "huffmanTree.h"

// Constructor for HuffmanTree
HuffmanTree::HuffmanTree(const std::vector<char>& bytes) {
    this->bytes = bytes;
    this->createNodes();
    this->root = new HuffmanTreeNode(0,0,false);

    for(const auto& n: this->nodes) {
        std::cout << n->val << " value " << std::endl;
        this->addToTree(this->root, n, n->length);
    }
    this->decodeTree(this->root, "");
    for(const auto& [key,val]: this->codes) {
    std::cout << int(key) << " is key " << val << std::endl;
    }
}

// Creates the nodes for the characters.
void HuffmanTree::createNodes() {
    int offset = 0;
    
    // The first byte belongs to the header.
    char header = this->bytes[offset];
    offset = offset + 1;

    // Extracting the length information.
    std::vector<char> lengths(this->bytes.begin(), this->bytes.begin() + 16);
    offset = offset + 16;

    for (int i = 0; i < 16; i++) {
        char curLength = lengths[i];
        std::cout << (int)(curLength) << " *** " << std::endl;
        for (int j = 0; j < curLength; j++) {
            char curVal = this->bytes[offset++];
            std::cout << " cur val " << std::hex << (int)(unsigned char)(curVal) << " " << i + 1 << std::endl;
            this->nodes.push_back(new HuffmanTreeNode(curVal, i+1, true));
        }
    }
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
                std::cout << " adding to the left " << std::endl;
                root->left = internalNode;
            }
            if (this->addToTree(root->left, node, position-1)) {
                return true;
            }
        } else {
            if (root->right == NULL) {
                std::cout << " adding to the right " << std::endl;
                root->right = internalNode;
            }
            if (this->addToTree(root->right, node, position - 1)) {
                return true;
            }
        }
    }
    return false;
}

void HuffmanTree::decodeTree(HuffmanTreeNode* node, std::string currentString) {
    // If we reach a leaf node
    if (node->isLeaf) {
        this->codes[node->val] = currentString;
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