#ifndef COLOR_H
#define COLOR_H

#include <vector>

int clamp(int col);
void GetArray(unsigned char* output, const unsigned char* input, int length);
int decode_number(int code, int bits);
void color_conversion(const std::vector<float>& Y, const std::vector<float>& Cr, const std::vector<float>& Cb,
                      std::vector<int>& R, std::vector<int>& G, std::vector<int>& B, int length);

#endif // COLOR_H