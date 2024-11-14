#ifndef IDCT_H
#define IDCT_H

#include <iostream>
#include "idct.h"


IDCT::IDCT(std::vector<int>& base): idctTable(8, std::vector<float>(8,0)), zigzag {
            {0, 1, 5, 6, 14, 15, 27, 28},
            {2, 4, 7, 13, 16, 26, 29, 42},
            {3, 8, 12, 17, 25, 30, 41, 43},
            {9, 11, 18, 24, 31, 40, 44, 53},
            {10, 19, 23, 32, 39, 45, 52, 54},
            {20, 22, 33, 38, 46, 51, 55, 60},
            {21, 34, 37, 47, 50, 56, 59, 61},
            {35, 36, 48, 49, 57, 58, 62, 63}
        } {
    this->initializeIDCTTable();
    this->base = base;
}

void IDCT::initializeIDCTTable() {
    for (int u = 0; u < IDCT_PRECISION; u++) {
        for (int x = 0; x < IDCT_PRECISION; x++) {
            float normCoeff = (u == 0) ? (1.0f / sqrtf(2.0f)) : 1.0f; 
            this->idctTable[u][x] = normCoeff * cosf(((2.0f * x + 1.0f) * u * M_PI) / 16.0f); 
        }
    }
    //  for (int i = 0; i < IDCT_PRECISION; i++) {
    //     for (int j =0; j < IDCT_PRECISION; j++) {
    //         std::cout << idctTable[i][j] << " ";
    //     }
    //     }
    //     std::cout << std::endl;
}

void IDCT::rearrangeUsingZigzag(int validWidth, int validHeight) {
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            // Check if the position is within the valid image bounds
            if (x < validWidth && y < validHeight) {
                this->zigzag[x][y] = this->base[zigzag[x][y]];
            } else {
                this->zigzag[x][y] = 0; // Assign zero for padding
            }
        }
    }
}

void IDCT::performIDCT(int validWidth, int validHeight) {
    std::vector<std::vector<float>> out(8, std::vector<float>(8, 0));

    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            if (x < validWidth && y < validHeight) {
                float localSum = 0.0f;
                for (int u = 0; u < IDCT_PRECISION; u++) {
                    for (int v = 0; v < IDCT_PRECISION; v++) {
                        localSum += static_cast<float>(this->zigzag[v][u]) * this->idctTable[u][x] * this->idctTable[v][y];
                    }
                }
                out[y][x] = std::floor(localSum / 4.0f);
            }
        }
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (i < validHeight && j < validWidth) {
                float value = out[i][j];
                int convertedValue = static_cast<int>(value);
                this->base[i * 8 + j] = convertedValue;
            }
        }
    }
    for (int i = 0; i < IDCT_PRECISION * IDCT_PRECISION; i++) {
            std::cout << base[i] << " ";
        }
        std::cout << std::endl;
}

#endif