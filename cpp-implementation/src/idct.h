#pragma once

#include <vector>
#include <cmath>
#include <algorithm> // For std::clamp

const int IDCT_PRECISION = 8;
const int W1 = 2841; // 2048*sqrt(2)*cos(1*pi/16)
const int W2 = 2676; // 2048*sqrt(2)*cos(2*pi/16)
const int W3 = 2408; // 2048*sqrt(2)*cos(3*pi/16)
const int W5 = 1609; // 2048*sqrt(2)*cos(5*pi/16)
const int W6 = 1108; // 2048*sqrt(2)*cos(6*pi/16)
const int W7 = 565;  // 2048*sqrt(2)*cos(7*pi/16)

class IDCT {
private:
    std::vector<std::vector<int>> zigzag;
    inline int clip(int value) {
        return std::clamp(value, 0, 255); // Clamp value to range [0, 255]
    }
    void idctRow(int* blk);
    void idctCol(int* blk);
public:
    std::vector<int> base;

    IDCT(std::vector<int>& base);
    void rearrangeUsingZigzag(int validWidth, int validHeight);
    void performIDCT(int validWidth, int validHeight);
};

// #pragma

// #include <vector>
// #include <cmath>

// const int IDCT_PRECISION = 8;

// /*
//     The class representing the Inverse Discrete Cosine Transform.
// */
// class IDCT {
//     private:
//         std::vector<std::vector<int>> zigzag;

//     public:
//         IDCT(std::vector<int>& base);
//         std::vector<int> base;
        
//         void rearrangeUsingZigzag(int validWidth, int validHeight);
//         void performIDCT(int validWidth, int validHeight);
//         void initializeIDCTTable();
// };