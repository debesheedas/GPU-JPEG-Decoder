#include "idct.h"
#include <iostream>

__global__ void rearrangeUsingZigzagkernel(int *dZigzag, int* initialZigzag, const int *dBase, int numElements, int validWidth, int validHeight)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements) {
        if ((idx/8) < validWidth && (idx%8) < validHeight) {
            dZigzag[idx] = dBase[initialZigzag[idx]];
        } else {
            dZigzag[idx] = 0;
        }
    }
}

__global__ void performIDCTKernel(int *dOut, const int *dZigzag, const double *dIdctTable, int precision, int validWidth, int validHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO(carlos): Further parallelize here.
    if (x < precision && y < precision && x < validWidth && y < validHeight) {
        double localSum = 0.0;
        for (int u = 0; u < precision; u++) {
            for (int v = 0; v < precision; v++) {
                localSum += dZigzag[v * precision + u] * dIdctTable[u * precision + x] * dIdctTable[v * precision + y];
            }
        }

        dOut[y * precision + x] = static_cast<int>(std::floor(localSum / 4.0));
    }
}


IDCT::IDCT(int* baseValue, double* idct, int* zigzag, int* initialZigzag): base(baseValue), idctTable(idct), zigzag(zigzag), initialZigzag(initialZigzag) {
    blockSize = 64;
    gridSize = (64 + blockSize - 1) / blockSize;    
}

void IDCT::rearrangeUsingZigzag(int validWidth, int validHeight)
{    
    rearrangeUsingZigzagkernel<<<gridSize, blockSize>>>(zigzag, initialZigzag, base, 64, validWidth, validHeight);
}

void IDCT::performIDCT(int validWidth, int validHeight)
{
    int precision = IDCT_PRECISION;

    dim3 blockSize2(8, 8);
    dim3 gridSize2((precision + blockSize2.x - 1) / blockSize2.x, (precision + blockSize2.y - 1) / blockSize2.y);

    performIDCTKernel<<<gridSize2, blockSize2>>>(base, zigzag, idctTable, precision, validWidth, validHeight);
}