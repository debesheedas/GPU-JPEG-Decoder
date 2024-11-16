#include "huffmanTree.h"

// Constructor for HuffmanTree
HuffmanTree::HuffmanTree(const std::vector<uint8_t>& bytes) {
    this->bytes = bytes;
    this->createNodes();
    this->root = new HuffmanTreeNode(0,0,false);

    for(const auto& n: this->nodes) {
        this->addToTree(this->root, n, n->length);
    }
    this->decodeTree(this->root, "");
}

// Creates the nodes for the characters.
void HuffmanTree::createNodes() {
    // Extracting the length information.
    std::vector<char> lengths(this->bytes.begin(), this->bytes.begin() + 16);
    int offset = 16;

    for (int i = 0; i < 16; i++) {
        char curLength = lengths[i];
        for (int j = 0; j < curLength; j++) {
            char curVal = this->bytes[offset++];
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
                root->left = internalNode;
            }
            if (this->addToTree(root->left, node, position-1)) {
                return true;
            }
        } else {
            if (root->right == NULL) {
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
    // If we reach a leaf node we return the value.
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

uint8_t HuffmanTree::getCode(Stream* st) {
    return this->traverseTree(this->root, st);
}
