#pragma once

#include <vector>

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
        
        void rearrangeUsingZigzag();
        void performIDCT();
        void initializeIDCTTable();
};
