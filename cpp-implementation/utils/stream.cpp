#include "stream.h"

Stream::Stream(std::vector<uint8_t>& data) {
    this->data = data;
    this->position = 0;
}

int Stream::getBit() {
    char curVal = this->data[this->position >> 3];
    int bitShift = 7 - (this->position & 0x7);
    this->position++;
    return (curVal >> bitShift) & 1;
}

int Stream::getNBits(int n) {
    int curVal = 0;
    for (int i = 0; i < n; i++) {
        curVal = curVal * 2;
        curVal += this->getBit();
    }
    return curVal;
}

uint8_t Stream::getByte() {
    uint8_t curVal = this->data[this->position >> 3];
    this->position += 8;
    return curVal;
}

uint16_t Stream::getMarker() {
    int actualPos = this->position >> 3;
    uint16_t firstByte = this->data[actualPos];
    uint16_t secondByte = this->data[actualPos + 1];
    this->position += 16;
    return (firstByte << 8) | secondByte;
}

void Stream::getNBytes(std::vector<uint8_t>& arr, int length) {
    for (int i = 0; i < length; i++) {
        arr.push_back(this->getByte());
    }
}