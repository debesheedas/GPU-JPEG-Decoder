
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

int clamp(int col) {
    return col > 255 ? 255 : (col < 0 ? 0 : col);
}

void GetArray(unsigned char* output, const unsigned char* input, int length) {
    for (int i = 0; i < length; ++i) {
        output[i] = input[i];
    }
}

int decode_number(int code, int bits) {
    int l = 1 << (code - 1);
    return (bits >= l) ? bits : bits - (2 * l - 1);
}

void color_conversion(const std::vector<float>& Y, const std::vector<float>& Cr, const std::vector<float>& Cb,
                      std::vector<int>& R, std::vector<int>& G, std::vector<int>& B, int length) {
    for (int i = 0; i < length; ++i) {
        float r = Cr[i] * (2 - 2 * 0.299) + Y[i];
        float b = Cb[i] * (2 - 2 * 0.114) + Y[i];
        float g = (Y[i] - 0.114 * b - 0.299 * r) / 0.587;

        R[i] = clamp(static_cast<int>(r + 128));
        G[i] = clamp(static_cast<int>(g + 128));
        B[i] = clamp(static_cast<int>(b + 128));
    }
}
