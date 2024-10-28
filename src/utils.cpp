#include <stdio.h>
#include <fstream>
#include <vector>
#include <iterator>
#include <iostream>
#include "utils.h"


// int ByteUtil::getSize() {
//     std::cout << "Static method called!" << std::endl;
// }

// int ByteUtil::getSize(int byte1, int byte2) {
//     return ((int)(unsigned char) byte1 << 8)+ (int)(unsigned char) byte2;
// };

int ByteUtil::DecodeNumber(int code, int bits) {
    int l = 1 << (code - 1);  // Calculate 2^(code - 1) using bit shift
    
    if (bits >= l) {
        return bits;
    } else {
        return bits - ((l << 1) - 1);  // Equivalent to bits - (2 * l - 1)
    }
}