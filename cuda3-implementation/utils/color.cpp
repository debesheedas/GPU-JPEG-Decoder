
#include "color.h"

int clamp(int col) {
    return col > 255 ? 255 : (col < 0 ? 0 : col);
}

void colorConversion(const std::vector<int>& Y, const std::vector<int>& Cr, const std::vector<int>& Cb,
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
