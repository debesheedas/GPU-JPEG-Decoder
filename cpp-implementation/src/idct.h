#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

const int IDCT_PRECISION = 8;
const int C1 = 2841; // 2048*sqrt(2)*cos(1*pi/16)
const int C2 = 2676; // 2048*sqrt(2)*cos(2*pi/16)
const int C3 = 2408; // 2048*sqrt(2)*cos(3*pi/16)
const int C5 = 1609; // 2048*sqrt(2)*cos(5*pi/16)
const int C6 = 1108; // 2048*sqrt(2)*cos(6*pi/16)
const int C7 = 565;  // 2048*sqrt(2)*cos(7*pi/16)

class IDCT {
private:
    std::vector<std::vector<int>> zigzag;
    int clip(int value);
    void idctRow(int* blk);
    void idctCol(int* blk);
public:
    std::vector<int> base;

    IDCT(std::vector<int>& base);
    void rearrangeUsingZigzag(int validWidth, int validHeight);
    void performIDCT(int validWidth, int validHeight);
};