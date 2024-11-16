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
        // std::vector<int> zigzag;
        int* zigzag;
        int blockSize;
        int gridSize;

    public:
        IDCT(std::vector<int>& base);
        double* idctTable;
        int* base;
        
        void rearrangeUsingZigzag(int validWidth, int validHeight);
        void performIDCT(int validWidth, int validHeight);
        void initializeIDCTTable();
};
