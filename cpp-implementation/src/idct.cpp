#ifndef IDCT_H
#define IDCT_H

#include <iostream>
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

void IDCT::rearrangeUsingZigzag(int validWidth, int validHeight) {
    std::vector<int> temp(64, 0);
    for (int x = 0; x < validWidth; x++) {
        for (int y = 0; y < validHeight; y++) {
            temp[8 * x + y] = base[zigzag[x][y]];
        }
    }
    base = temp;
}

void IDCT::idctRow(int* blk) {
    int x0 = (blk[0] << 11) + 128; // Bias for proper rounding
    int x1 = blk[4] << 11;
    int x2 = blk[6];
    int x3 = blk[2];
    int x4 = blk[1];
    int x5 = blk[7];
    int x6 = blk[5];
    int x7 = blk[3];

    // if (!(x1 | x2 | x3 | x4 | x5 | x6 | x7)) {
    //     for (int i = 0; i < 8; i++) blk[i] = blk[0]<<3; // Proper scaling
    //     return;
    // }

    // int x0, x1, x2, x3, x4, x5, x6, x7;

    // /* shortcut */
    // if (!((x1 = blk[4]<<11) | (x2 = blk[6]) | (x3 = blk[2]) |
    //         (x4 = blk[1]) | (x5 = blk[7]) | (x6 = blk[5]) | (x7 = blk[3])))
    // {
    //     blk[0]=blk[1]=blk[2]=blk[3]=blk[4]=blk[5]=blk[6]=blk[7]=blk[0]<<3;
    //     return;
    // }

    // x0 = (blk[0]<<11) + 128; /* for proper rounding in the fourth stage */


    // First stage
    int x8 = W7 * (x4 + x5);
    x4 = x8 + (W1 - W7) * x4;
    x5 = x8 - (W1 + W7) * x5;
    x8 = W3 * (x6 + x7);
    x6 = x8 - (W3 - W5) * x6;
    x7 = x8 - (W3 + W5) * x7;

    // Second stage
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2);
    x2 = x1 - (W2 + W6) * x2;
    x3 = x1 + (W2 - W6) * x3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;

    // Third stage
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;

    // Fourth stage
    blk[0] = (x7 + x1) >> 8;
    blk[1] = (x3 + x2) >> 8;
    blk[2] = (x0 + x4) >> 8;
    blk[3] = (x8 + x6) >> 8;
    blk[4] = (x8 - x6) >> 8;
    blk[5] = (x0 - x4) >> 8;
    blk[6] = (x3 - x2) >> 8;
    blk[7] = (x7 - x1) >> 8;
}

void IDCT::idctCol(int* blk) {
    int x0 = (blk[8 * 0] << 8) + 8192;
    int x1 = blk[8 * 4] << 8;
    int x2 = blk[8 * 6];
    int x3 = blk[8 * 2];
    int x4 = blk[8 * 1];
    int x5 = blk[8 * 7];
    int x6 = blk[8 * 5];
    int x7 = blk[8 * 3];

    // if (!(x1 | x2 | x3 | x4 | x5 | x6 | x7)) {
    //     for (int i = 0; i < 8; i++) blk[8*i] = clip(blk[8*0]+32)>>6; // Proper scaling
    //     return;
    // }
    // int x0, x1, x2, x3, x4, x5, x6, x7;

    // /* shortcut */
    // if (!((x1 = (blk[8*4]<<8)) | (x2 = blk[8*6]) | (x3 = blk[8*2]) |
    //         (x4 = blk[8*1]) | (x5 = blk[8*7]) | (x6 = blk[8*5]) | (x7 = blk[8*3])))
    // {
    //     blk[8*0]=blk[8*1]=blk[8*2]=blk[8*3]=blk[8*4]=blk[8*5]=blk[8*6]=blk[8*7]=
    //     (blk[8*0]+32)>>6;
    //     return;
    // }

    // x0 = (blk[8*0]<<8) + 8192;


    int x8 = W7 * (x4 + x5) + 4;
    x4 = (x8 + (W1 - W7) * x4) >> 3;
    x5 = (x8 - (W1 + W7) * x5) >> 3;
    x8 = W3 * (x6 + x7) + 4;
    x6 = (x8 - (W3 - W5) * x6) >> 3;
    x7 = (x8 - (W3 + W5) * x7) >> 3;

    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2) + 4;
    x2 = (x1 - (W2 + W6) * x2) >> 3;
    x3 = (x1 + (W2 - W6) * x3) >> 3;
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

    blk[8 * 0] = (x7 + x1) >> 14;
    blk[8 * 1] = (x3 + x2) >> 14;
    blk[8 * 2] = (x0 + x4) >> 14;
    blk[8 * 3] = (x8 + x6) >> 14;
    blk[8 * 4] = (x8 - x6) >> 14;
    blk[8 * 5] = (x0 - x4) >> 14;
    blk[8 * 6] = (x3 - x2) >> 14;
    blk[8 * 7] = (x7 - x1) >> 14;
}

void IDCT::performIDCT(int validWidth, int validHeight) {
    int block[64] = {0};

    for (int i = 0; i < 64; i++) {
        block[i] = static_cast<int>(base[i]);
    }

    for (int i = 0; i < 8; i++) {
        idctRow(block + 8 * i);
    }

    for (int i = 0; i < 8; i++) {
        idctCol(block + i);
    }

    for (int i = 0; i < 64; i++) {
        base[i] = static_cast<int>(block[i]);
    }
}

#endif


// #ifndef IDCT_H
// #define IDCT_H

// #include <iostream>
// #include "idct.h"

// const int kIDCTMatrix[64] = {
//   8192,  11363,  10703,   9633,   8192,   6437,   4433,   2260,
//   8192,   9633,   4433,  -2259,  -8192, -11362, -10704,  -6436,
//   8192,   6437,  -4433, -11362,  -8192,   2261,  10704,   9633,
//   8192,   2260, -10703,  -6436,   8192,   9633,  -4433, -11363,
//   8192,  -2260, -10703,   6436,   8192,  -9633,  -4433,  11363,
//   8192,  -6437,  -4433,  11362,  -8192,  -2261,  10704,  -9633,
//   8192,  -9633,   4433,   2259,  -8192,  11362, -10704,   6436,
//   8192, -11363,  10703,  -9633,   8192,  -6437,   4433,  -2260,
// };


// IDCT::IDCT(std::vector<int>& base): zigzag {
//             {0, 1, 5, 6, 14, 15, 27, 28},
//             {2, 4, 7, 13, 16, 26, 29, 42},
//             {3, 8, 12, 17, 25, 30, 41, 43},
//             {9, 11, 18, 24, 31, 40, 44, 53},
//             {10, 19, 23, 32, 39, 45, 52, 54},
//             {20, 22, 33, 38, 46, 51, 55, 60},
//             {21, 34, 37, 47, 50, 56, 59, 61},
//             {35, 36, 48, 49, 57, 58, 62, 63}
//         } {
//     this->base = base;
// }

// void IDCT::rearrangeUsingZigzag(int validWidth, int validHeight) {
//     std::vector<int> temp(64, 0);
//     for (int x = 0; x < 8; x++) {
//         for (int y = 0; y < 8; y++) {
//             temp[8 * x + y] = base[zigzag[x][y]];
//         }
//     }
//     base = temp;
// }

// void Compute1dIDCT(const int in[8], const int stride, int out[8]) {
//     int tmp0, tmp1, tmp2, tmp3, tmp4;

//     tmp1 = kIDCTMatrix[0] * in[0];
//     out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = out[6] = out[7] = tmp1;

//     tmp0 = in[stride];
//     tmp1 = kIDCTMatrix[ 1] * tmp0;
//     tmp2 = kIDCTMatrix[ 9] * tmp0;
//     tmp3 = kIDCTMatrix[17] * tmp0;
//     tmp4 = kIDCTMatrix[25] * tmp0;
//     out[0] += tmp1;
//     out[1] += tmp2;
//     out[2] += tmp3;
//     out[3] += tmp4;
//     out[4] -= tmp4;
//     out[5] -= tmp3;
//     out[6] -= tmp2;
//     out[7] -= tmp1;

//     tmp0 = in[2 * stride];
//     tmp1 = kIDCTMatrix[ 2] * tmp0;
//     tmp2 = kIDCTMatrix[10] * tmp0;
//     out[0] += tmp1;
//     out[1] += tmp2;
//     out[2] -= tmp2;
//     out[3] -= tmp1;
//     out[4] -= tmp1;
//     out[5] -= tmp2;
//     out[6] += tmp2;
//     out[7] += tmp1;

//     tmp0 = in[3 * stride];
//     tmp1 = kIDCTMatrix[ 3] * tmp0;
//     tmp2 = kIDCTMatrix[11] * tmp0;
//     tmp3 = kIDCTMatrix[19] * tmp0;
//     tmp4 = kIDCTMatrix[27] * tmp0;
//     out[0] += tmp1;
//     out[1] += tmp2;
//     out[2] += tmp3;
//     out[3] += tmp4;
//     out[4] -= tmp4;
//     out[5] -= tmp3;
//     out[6] -= tmp2;
//     out[7] -= tmp1;

//     tmp0 = in[4 * stride];
//     tmp1 = kIDCTMatrix[ 4] * tmp0;
//     out[0] += tmp1;
//     out[1] -= tmp1;
//     out[2] -= tmp1;
//     out[3] += tmp1;
//     out[4] += tmp1;
//     out[5] -= tmp1;
//     out[6] -= tmp1;
//     out[7] += tmp1;

//     tmp0 = in[5 * stride];
//     tmp1 = kIDCTMatrix[ 5] * tmp0;
//     tmp2 = kIDCTMatrix[13] * tmp0;
//     tmp3 = kIDCTMatrix[21] * tmp0;
//     tmp4 = kIDCTMatrix[29] * tmp0;
//     out[0] += tmp1;
//     out[1] += tmp2;
//     out[2] += tmp3;
//     out[3] += tmp4;
//     out[4] -= tmp4;
//     out[5] -= tmp3;
//     out[6] -= tmp2;
//     out[7] -= tmp1;

//     tmp0 = in[6 * stride];
//     tmp1 = kIDCTMatrix[ 6] * tmp0;
//     tmp2 = kIDCTMatrix[14] * tmp0;
//     out[0] += tmp1;
//     out[1] += tmp2;
//     out[2] -= tmp2;
//     out[3] -= tmp1;
//     out[4] -= tmp1;
//     out[5] -= tmp2;
//     out[6] += tmp2;
//     out[7] += tmp1;

//     tmp0 = in[7 * stride];
//     tmp1 = kIDCTMatrix[ 7] * tmp0;
//     tmp2 = kIDCTMatrix[15] * tmp0;
//     tmp3 = kIDCTMatrix[23] * tmp0;
//     tmp4 = kIDCTMatrix[31] * tmp0;
//     out[0] += tmp1;
//     out[1] += tmp2;
//     out[2] += tmp3;
//     out[3] += tmp4;
//     out[4] -= tmp4;
//     out[5] -= tmp3;
//     out[6] -= tmp2;
//     out[7] -= tmp1;
// }

// void IDCT::performIDCT(int validWidth, int validHeight) {
//     int colidcts[64];
//     const int kColScale = 11;
//     const int kColRound = 1 << (kColScale - 1);

//     for (int x = 0; x < 8; ++x) {
//         int colbuf[8] = {0};
//         Compute1dIDCT(&base[x], 8, colbuf);
//         for (int y = 0; y < 8; ++y) {
//             colidcts[8 * y + x] = (colbuf[y] + kColRound) >> kColScale;
//         }
//     }

//     const int kRowScale = 18;
//     const int kRowRound = 257 << (kRowScale - 1);  // includes offset by 128
//     for (int y = 0; y < 8; ++y) {
//         const int rowidx = 8 * y;
//         int rowbuf[8] = {0};
//         Compute1dIDCT(&colidcts[rowidx], 1, rowbuf);
//         for (int x = 0; x < 8; ++x) {
//             base[rowidx + x] = std::max(0, std::min(255, (rowbuf[x] + kRowRound) >> kRowScale));
//         }
//     }
// }

// #endif