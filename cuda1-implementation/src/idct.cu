#include "idct.h"
#include <iostream>

__global__ void rearrangeUsingZigzagkernel(int *dZigzag, const int *dBase, int numElements, int validWidth, int validHeight)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements) {
        if ((idx/8) < validWidth && (idx%8) < validHeight) {
            dZigzag[idx] = dBase[dZigzag[idx]];
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


IDCT::IDCT(int* baseValue, double* idct): base(baseValue), idctTable(idct) {
    
    int zigzagEntries[64] = {
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63
    };

    cudaMalloc((void**)&zigzag, 64 * sizeof(int));
    cudaMemcpy(zigzag, zigzagEntries, 64 * sizeof(int), cudaMemcpyHostToDevice);

    blockSize = 64;
    gridSize = (64 + blockSize - 1) / blockSize;    
}

void IDCT::rearrangeUsingZigzag(int validWidth, int validHeight)
{    
    rearrangeUsingZigzagkernel<<<gridSize, blockSize>>>(zigzag, base, 64, validWidth, validHeight);
}

void IDCT::performIDCT(int validWidth, int validHeight)
{
    int precision = IDCT_PRECISION;

    dim3 blockSize2(8, 8);
    dim3 gridSize2((precision + blockSize2.x - 1) / blockSize2.x, (precision + blockSize2.y - 1) / blockSize2.y);

    performIDCTKernel<<<gridSize2, blockSize2>>>(base, zigzag, idctTable, precision, validWidth, validHeight);
}