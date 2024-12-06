#ifndef IDCT_H
#define IDCT_H

#include "idct.h"
#include <vector>
#include <cmath>
#include <algorithm>

const int COS[] = {
    23170,  // cos(pi/4) * 2^16
    32138,  // cos(pi/8) * 2^16
    27246,  // cos(3pi/8) * 2^16
    18204,  // cos(5pi/8) * 2^16
    12540,  // cos(7pi/8) * 2^16
};
const int SCALE = 65536; // Scaling factor for fixed-point arithmetic


// Constructor
IDCT::IDCT(std::vector<int>& base)
    : base(base),
      zigzag{
          {0, 1, 5, 6, 14, 15, 27, 28},
          {2, 4, 7, 13, 16, 26, 29, 42},
          {3, 8, 12, 17, 25, 30, 41, 43},
          {9, 11, 18, 24, 31, 40, 44, 53},
          {10, 19, 23, 32, 39, 45, 52, 54},
          {20, 22, 33, 38, 46, 51, 55, 60},
          {21, 34, 37, 47, 50, 56, 59, 61},
          {35, 36, 48, 49, 57, 58, 62, 63}
      } {}

// Rearrange the block using Zigzag order
void IDCT::rearrangeUsingZigzag() {
    std::vector<int> temp(64, 0);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int zigzagIndex = zigzag[i][j];
            temp[i * 8 + j] = base[zigzagIndex];
        }
    }
    base = temp;
}

// Perform 1D IDCT on a single row or column
void IDCT::idct1D(std::vector<int>& data) {
    int temp[8];

    // Butterfly operations
    int s0 = data[0] + data[4];
    int s1 = data[0] - data[4];
    int s2 = (data[2] * COS[1] + data[6] * COS[2]) / SCALE;
    int s3 = (data[2] * COS[2] - data[6] * COS[1]) / SCALE;

    int d0 = data[1] + data[7];
    int d1 = data[3] + data[5];
    int d2 = data[3] - data[5];
    int d3 = data[1] - data[7];

    // Combine even terms
    temp[0] = s0 + s2;
    temp[4] = s0 - s2;
    temp[2] = s1 + s3;
    temp[6] = s1 - s3;

    // Combine odd terms
    temp[1] = (d0 * COS[0] + d1 * COS[3] + d2 * COS[4]) / SCALE;
    temp[3] = (d0 * COS[3] - d1 * COS[0] - d3 * COS[4]) / SCALE;
    temp[5] = (d0 * COS[4] - d2 * COS[3] + d3 * COS[0]) / SCALE;
    temp[7] = (d1 * COS[4] - d2 * COS[0] + d3 * COS[3]) / SCALE;

    // Copy results back
    for (int i = 0; i < 8; i++) {
        data[i] = temp[i];
    }
}

// Perform 2D IDCT on an 8x8 block
void IDCT::idct2D(std::vector<std::vector<int>>& block) {
    // Perform 1D IDCT on each row
    for (int i = 0; i < 8; i++) {
        idct1D(block[i]);
    }

    // Perform 1D IDCT on each column
    for (int j = 0; j < 8; j++) {
        std::vector<int> column(8);
        for (int i = 0; i < 8; i++) {
            column[i] = block[i][j];
        }
        idct1D(column);
        for (int i = 0; i < 8; i++) {
            block[i][j] = column[i];
        }
    }
}

// Perform full IDCT on the block
void IDCT::performIDCT() {
    rearrangeUsingZigzag();

    // Convert the 1D base array into a 2D block
    std::vector<std::vector<int>> block(8, std::vector<int>(8, 0));
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            block[i][j] = base[i * 8 + j];
        }
    }

    // Apply 2D IDCT
    idct2D(block);

    // Convert back to 1D base array
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            base[i * 8 + j] = block[i][j];
        }
    }
}

#endif