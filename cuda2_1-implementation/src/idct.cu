#include "idct.h"
#include <iostream>

__global__ void rearrangeUsingZigzagkernel(int *dZigzag, const int *dBase, int N, int validWidth, int validHeight)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int idx = x * N + y;

    if (idx < N * N) {
        if (x < validWidth && y < validHeight) {
            dZigzag[idx] = dBase[dZigzag[idx]];
        } else {
            dZigzag[idx] = 0;
        }
    }
}

__global__ void initializeIDCTTableKernel(double *dIdctTable, int numThreads)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (id < numThreads) {
        double normCoeff = ((id / 8) == 0) ? (1.0 / sqrt(2.0)) : 1.0;
        dIdctTable[id] = normCoeff * cos(((2.0 * (id%8) + 1.0) * (id/8) * M_PI) / 16.0);
    }
}

__global__ void performIDCTKernel(int *dOut, const int *dZigzag, const double *dIdctTable, int precision, int validWidth, int validHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

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


IDCT::IDCT(std::vector<int>& baseValues) {
    
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
    cudaMalloc((void**)&base, 64 * sizeof(int));
    cudaMalloc((void **)&idctTable, 64 * sizeof(double));
    cudaMemcpy(zigzag, zigzagEntries, 64 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(base, baseValues.data(), 64 * sizeof(int), cudaMemcpyHostToDevice);


    this->initializeIDCTTable();
    //this->base = base;
}

void IDCT::initializeIDCTTable()
{
    int blockSize = 64;
    int gridSize = (64 + blockSize - 1) / blockSize;
    initializeIDCTTableKernel<<<gridSize, blockSize>>>(idctTable, 64);
    // dim3 blockSize(8, 8);
    // dim3 gridSize((precision + blockSize.x - 1) / blockSize.x, (precision + blockSize.y - 1) / blockSize.y);

    //initializeIDCTTableKernel<<<gridSize, blockSize>>>(dIdctTable, precision);
    //cudaMemcpy(idctTable.data(), dIdctTable, precision * precision * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaFree(dIdctTable);
}

void IDCT::rearrangeUsingZigzag(int validWidth, int validHeight)
{
    
    dim3 blockSize(8, 8);
    dim3 gridSize(1, 1);

    rearrangeUsingZigzagkernel<<<gridSize, blockSize>>>(zigzag, base, 8, validWidth, validHeight);
}

void IDCT::performIDCT(int validWidth, int validHeight)
{
    int precision = IDCT_PRECISION;

    dim3 blockSize(8, 8);
    dim3 gridSize((precision + blockSize.x - 1) / blockSize.x, (precision + blockSize.y - 1) / blockSize.y);

    performIDCTKernel<<<gridSize, blockSize>>>(base, zigzag, idctTable, precision, validWidth, validHeight);

    //cudaFree(dIdctTable);
}