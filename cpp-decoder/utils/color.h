#ifndef COLOR_H
#define COLOR_H

#include <vector>

/*
    This util class is for color related procedures. Especially used for YCrCb to RGB conversion.
*/

int clamp(int col);
void colorConversion(const std::vector<int>& Y, const std::vector<int>& Cr, const std::vector<int>& Cb,
                      std::vector<int>& R, std::vector<int>& G, std::vector<int>& B, int length);

#endif // COLOR_H