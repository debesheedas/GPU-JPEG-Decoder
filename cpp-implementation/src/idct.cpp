#ifndef IDCT_H
#define IDCT_H

#include <algorithm>
#include "idct.h"

IDCT::IDCT(std::vector<int>& base)
    : zigzag{
          {0, 1, 5, 6, 14, 15, 27, 28},
          {2, 4, 7, 13, 16, 26, 29, 42},
          {3, 8, 12, 17, 25, 30, 41, 43},
          {9, 11, 18, 24, 31, 40, 44, 53},
          {10, 19, 23, 32, 39, 45, 52, 54},
          {20, 22, 33, 38, 46, 51, 55, 60},
          {21, 34, 37, 47, 50, 56, 59, 61},
          {35, 36, 48, 49, 57, 58, 62, 63}} {
    this->base = base;
}

int IDCT::clip(int value) {
        return std::clamp(value, -256, 255);
}

void IDCT::rearrangeUsingZigzag() {
    std::vector<int> temp(64, 0);
    for (int x = 0; x < IDCT_PRECISION; x++) {
        for (int y = 0; y < IDCT_PRECISION; y++) {
            temp[8 * x + y] = base[zigzag[x][y]];
        }
    }
    base = temp;
}

void IDCT::idctRow(int* block) {
    int x0, x1, x2, x3, x4, x5, x6, x7;

    // Shortcut: if all AC terms are zero, directly scale the DC term
    if (!((x1 = block[4]<<11) | (x2 = block[6]) | (x3 = block[2]) | (x4 = block[1]) | (x5 = block[7]) | (x6 = block[5]) | (x7 = block[3]))) {
        block[0] = block[1] = block[2] = block[3] = block[4] = block[5] = block[6] = block[7] = block[0]<<3;
        return;
    }
    // Scale the DC coefficient
    x0 = (block[0]<<11) + 128;

    int x8 = C7 * (x4 + x5);
    x4 = x8 + (C1 - C7) * x4;
    x5 = x8 - (C1 + C7) * x5;
    x8 = C3 * (x6 + x7);
    x6 = x8 - (C3 - C5) * x6;
    x7 = x8 - (C3 + C5) * x7;

    x8 = x0 + x1;
    x0 -= x1;
    x1 = C6 * (x3 + x2);
    x2 = x1 - (C2 + C6) * x2;
    x3 = x1 + (C2 - C6) * x3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;

    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;

    block[0] = (x7 + x1) >> 8;
    block[1] = (x3 + x2) >> 8;
    block[2] = (x0 + x4) >> 8;
    block[3] = (x8 + x6) >> 8;
    block[4] = (x8 - x6) >> 8;
    block[5] = (x0 - x4) >> 8;
    block[6] = (x3 - x2) >> 8;
    block[7] = (x7 - x1) >> 8;
}

void IDCT::idctCol(int* block) {
    int x0, x1, x2, x3, x4, x5, x6, x7;

    // Shortcut: if all AC terms are zero, directly scale the DC term
    if (!((x1 = (block[8*4]<<8)) | (x2 = block[8*6]) | (x3 = block[8*2]) | (x4 = block[8*1]) | (x5 = block[8*7]) | (x6 = block[8*5]) | (x7 = block[8*3]))) {
        block[8*0] = block[8*1] = block[8*2] = block[8*3] = block[8*4] = block[8*5] = block[8*6] = block[8*7] = clip((block[8*0]+32)>>6);
        return;
    }
    // Scale the DC coefficient
    x0 = (block[8*0]<<8) + 8192;

    int x8 = C7 * (x4 + x5) + 4;
    x4 = (x8 + (C1 - C7) * x4) >> 3;
    x5 = (x8 - (C1 + C7) * x5) >> 3;
    x8 = C3 * (x6 + x7) + 4;
    x6 = (x8 - (C3 - C5) * x6) >> 3;
    x7 = (x8 - (C3 + C5) * x7) >> 3;
    
    x8 = x0 + x1;
    x0 -= x1;
    x1 = C6 * (x3 + x2) + 4;
    x2 = (x1 - (C2 + C6) * x2) >> 3;
    x3 = (x1 + (C2 - C6) * x3) >> 3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;

    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;

    block[8 * 0] = clip((x7 + x1) >> 14);
    block[8 * 1] = clip((x3 + x2) >> 14);
    block[8 * 2] = clip((x0 + x4) >> 14);
    block[8 * 3] = clip((x8 + x6) >> 14);
    block[8 * 4] = clip((x8 - x6) >> 14);
    block[8 * 5] = clip((x0 - x4) >> 14);
    block[8 * 6] = clip((x3 - x2) >> 14);
    block[8 * 7] = clip((x7 - x1) >> 14);
}

void IDCT::performIDCT() {
    // Perform IDCT for rows
    for (int i = 0; i < 8; i++) {
        idctRow(&base[8 * i]);
    }
    // Perform IDCT for columns
    for (int i = 0; i < 8; i++) {
        idctCol(&base[i]);
    }
}

#endif


