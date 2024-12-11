/*
Utils class for functions that should perform on the device.
*/
#include "utils.h"

__device__ uint8_t getBit(uint8_t* data, int& position) {
    uint8_t curVal = data[position >> 3];
    int bitShift = 7 - (position & 0x07);
    position++;
    return (curVal >> bitShift) & 0x01;
}

__device__ uint16_t getNBits(uint8_t* data, int& position, int n) {
    int curVal = 0;
    for (int i = 0; i < n; i++) {
        curVal = curVal * 2;
        curVal += getBit(data, position);
    }
    return curVal;
}

__device__ uint8_t getByte(uint8_t* data, int& position) {
    uint8_t curVal = data[position >> 3];
    position += 8;
    return curVal;
}

__device__ void getNBytes(uint8_t* arr, int length, uint8_t* data, int& position) {
    for (int i = 0; i < length; i++) {
        arr[i] = getByte(data, position);
    }
}

__device__ int decodeNumber(uint8_t code, int bits) {
    int l = 1 << (code - 1);  // Calculate 2^(code - 1) using bit shift
    
    if (bits >= l) {
        return bits;
    } else {
        return bits - ((l << 1) - 1);  // Equivalent to bits - (2 * l - 1)
    }
}
