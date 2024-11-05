#include "idct.h"
#include <cuda_runtime.h>
#include <cmath>
#include <math.h>

__global__ void rearrangeUsingZigzagkernel(int *d_zigzag, const int *d_base, int N)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int idx = x * N + y;
    
    if (idx == 0) {
        for (int i = 0; i < N*N; x++) {    
            d_zigzag[i] = d_base[d_zigzag[i]];
        }   
    }
}

__global__ void initializeIDCTTableKernel(float *d_idctTable, int precision)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (u == 0 && x == 0) {
        for (int i = 0; i < precision; i++) {
            for (int j = 0; j < precision; j++) {
                float normCoeff = (i == 0) ? (1.0f / sqrtf(2)) : 1.0f;
                d_idctTable[i * precision + j] = normCoeff * cosf(((2.0f * j + 1.0f) * i * M_PI) / 16.0f);
            }
        }
    }
}

__global__ void performIDCTKernel(int *d_out, const int *d_zigzag, const float *d_idctTable, int precision)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x == 0 && y == 0) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                float local_sum = 0.0f;
                for (int u = 0; u < precision; u++) {
                    for (int v = 0; v < precision; v++) {
                        local_sum += static_cast<float>(d_zigzag[v * precision + u]) * d_idctTable[u * precision + i] * d_idctTable[v * precision + j];
                    }
                }
                    d_out[i * precision + j] = static_cast<int>(std::floor(local_sum / 4.0f));
            }
        }
    }
}

IDCT::IDCT(std::vector<int>& base): idctTable(8*8), zigzag {
            0, 1, 5, 6, 14, 15, 27, 28,
            2, 4, 7, 13, 16, 26, 29, 42,
            3, 8, 12, 17, 25, 30, 41, 43,
            9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 51, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63
        } {
    this->initializeIDCTTable();
    this->base = base;
}

void IDCT::initializeIDCTTable()
{
    int precision = IDCT_PRECISION;
    idctTable.resize(precision*precision);
    float *d_idctTable;
    cudaMalloc((void **)&d_idctTable, precision * precision * sizeof(float));

    dim3 blockSize(1, 1);
    dim3 gridSize(1,1);

    initializeIDCTTableKernel<<<gridSize, blockSize>>>(d_idctTable, precision);
    cudaMemcpy(idctTable.data(), d_idctTable, precision * precision * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_idctTable);
}

void IDCT::rearrangeUsingZigzag()
{
    int N = 8;

    int *d_zigzag, *d_base;
    cudaMalloc((void **)&d_zigzag, N * N * sizeof(int));
    cudaMalloc((void **)&d_base, N * N * sizeof(int));

    cudaMemcpy(d_zigzag, zigzag.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_base, base.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(1, 1);
    dim3 gridSize(1, 1);

    rearrangeUsingZigzagkernel<<<gridSize, blockSize>>>(d_zigzag, d_base, N);

    cudaMemcpy(zigzag.data(), d_zigzag, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_zigzag);
    cudaFree(d_base);
}

void IDCT::performIDCT()
{
    int precision = IDCT_PRECISION;

    int* d_zigzag,*d_out;
    float* d_idctTable;
    cudaMalloc((void **)&d_zigzag, precision * precision * sizeof(int));
    cudaMalloc((void **)&d_idctTable, precision * precision * sizeof(float));
    cudaMalloc((void **)&d_out, precision * precision * sizeof(float));

    cudaMemcpy(d_zigzag, zigzag.data(), precision * precision * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idctTable, idctTable.data(), precision * precision * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(1, 1);
    dim3 gridSize(1,1);

    performIDCTKernel<<<gridSize, blockSize>>>(d_out, d_zigzag, d_idctTable, precision);
    cudaMemcpy(base.data(), d_out, precision * precision * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_zigzag);
    cudaFree(d_idctTable);
    cudaFree(d_out);
}