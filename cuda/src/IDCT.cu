#include "idct.h"
#include <cuda_runtime.h>
#include <cmath>
#include <math.h>

__global__ void rearrangeUsingZigzagkernel(int *d_zigzag, const int *d_base, int N)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int idx = x * N + y;
    
    if (idx < N * N) {
        d_zigzag[idx] = d_base[d_zigzag[idx]];
    }
}

__global__ void initializeIDCTTablekernel(float *d_idct_table, int precision)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (u < precision && x < precision) {
        float normCoeff = (u == 0) ? (1.0f / sqrtf(2)) : 1.0f;
        d_idct_table[u * precision + x] = normCoeff * cosf(((2.0f * x + 1.0f) * u * M_PI) / 16.0f);
    }
}

__global__ void performIDCTkernel(float *d_out, const int *d_zigzag, const float *d_idct_table, int precision)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < precision && y < precision) {
        float local_sum = 0;

        for (int u = 0; u < precision; u++) {
            for (int v = 0; v < precision; v++) {
                local_sum += d_zigzag[v * precision + u] * d_idct_table[u * precision + x] * d_idct_table[v * precision + y];
            }
        }

        d_out[y * precision + x] = local_sum / 4.0f;
    }
}

IDCT::IDCT(std::vector<int>& base): idctTable(8, std::vector<float>(8,0)), zigzag {
            {0, 1, 5, 6, 14, 15, 27, 28},
            {2, 4, 7, 13, 16, 26, 29, 42},
            {3, 8, 12, 17, 25, 30, 41, 43},
            {9, 11, 18, 24, 31, 40, 44, 53},
            {10, 19, 23, 32, 39, 45, 52, 54},
            {20, 22, 33, 38, 46, 51, 55, 60},
            {21, 34, 37, 47, 50, 56, 59, 61},
            {35, 36, 48, 49, 57, 58, 62, 63}
        } {
    this->initializeIDCTTable();
    this->base = base;
}

void IDCT::initializeIDCTTable()
{
    int precision = this->IDCT_PRECISION;
    float *d_idct_table;
    cudaMalloc((void **)&d_idct_table, precision * precision * sizeof(float));

    dim3 blockSize(8, 8);
    dim3 gridSize((precision + blockSize.x - 1) / blockSize.x, (precision + blockSize.y - 1) / blockSize.y);

    initializeIDCTTableKernel<<<gridSize, blockSize>>>(d_idct_table, precision);
    cudaMemcpy(idct_table, d_idct_table, precision * precision * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_idct_table);
}

int* IDCT::rearrangeUsingZigzag()
{
    int h_zigzag[64];
    int N = 8;

    // Flattening
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_zigzag[i * N + j] = zigzag[i][j];
        }
    }

    int *d_zigzag, *d_base;
    cudaMalloc((void **)&d_zigzag, N * N * sizeof(int));
    cudaMalloc((void **)&d_base, N * N * sizeof(int));

    cudaMemcpy(d_zigzag, h_zigzag, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_base, base, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(N, N);
    dim3 gridSize(1, 1);

    rearrangeUsingZigzagkernel<<<gridSize, blockSize>>>(d_zigzag, d_base, N);

    cudaMemcpy(h_zigzag, d_zigzag, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Unflatten
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            zigzag[i][j] = h_zigzag[i * N + j];
        }
    }
    cudaFree(d_zigzag);
    cudaFree(d_base);

    return &zigzag[0][0];
}

void IDCT::performIDCT()
{
    int precision = this->idct_precision;
    float out[precision * precision];
    int h_zigzag[64];
    int N = 8;
    // Flattening
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_zigzag[i * N + j] = zigzag[i][j];
        }
    }

    int* d_zigzag;
    float* d_idct_table, *d_out;
    cudaMalloc((void **)&d_zigzag, precision * precision * sizeof(int));
    cudaMalloc((void **)&d_idct_table, precision * precision * sizeof(float));
    cudaMalloc((void **)&d_out, precision * precision * sizeof(float));

    cudaMemcpy(d_zigzag, h_zigzag, precision * precision * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idct_table, idct_table, precision * precision * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(8, 8);
    dim3 gridSize((precision + blockSize.x - 1) / blockSize.x, (precision + blockSize.y - 1) / blockSize.y);

    perform_IDCT_kernel<<<gridSize, blockSize>>>(d_out, d_zigzag, d_idct_table, precision);

    cudaMemcpy(out, d_out, precision * precision * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy output back to base
    for (int i = 0; i < precision * precision; i++) {
        base[i] = static_cast<int>(out[i]);
    }

    cudaFree(d_zigzag);
    cudaFree(d_idct_table);
    cudaFree(d_out);
}