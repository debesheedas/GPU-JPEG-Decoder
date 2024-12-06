#include <vector>
#include <cmath>
#include <algorithm>

class IDCT {
private:
    int zigzag[8][8];
    void idct1D(std::vector<int>& data);
    void idct2D(std::vector<std::vector<int>>& block);

public:
    std::vector<int> base;
    IDCT(std::vector<int>& base);
    void rearrangeUsingZigzag();
    void performIDCT();

};