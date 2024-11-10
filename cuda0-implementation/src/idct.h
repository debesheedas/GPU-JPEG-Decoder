#pragma once

#include <vector>
#include <cmath>
#include <cuda_runtime.h>

const int IDCT_PRECISION = 8;

/*
    The class representing the Inverse Discrete Cosine Transform.
*/
class IDCT {
    private:
        std::vector<int> zigzag;

    public:
        IDCT(std::vector<int>& base);
        std::vector<float> idctTable;
        std::vector<int> base;
        
        void rearrangeUsingZigzag(int validWidth, int validHeight);
        void performIDCT(int validWidth, int validHeight);
        void initializeIDCTTable();
};
