#include <stdio.h>
#include <fstream>
#include <vector>
#include <iterator>
#include <iostream>
#include "utils.h"


// int ByteUtil::getSize() {
//     std::cout << "Static method called!" << std::endl;
// }

int ByteUtil::getSize(int byte1, int byte2) {
    return ((int)(unsigned char) byte1 << 8)+ (int)(unsigned char) byte2;
};