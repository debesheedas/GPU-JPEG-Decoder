#include "idct.h"

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

__global__ void initializeIDCTTableKernel(float *dIdctTable, int precision)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (u < precision && x < precision) {
        float normCoeff = (u == 0) ? (1.0f / sqrtf(2)) : 1.0f;
        dIdctTable[u * precision + x] = normCoeff * cosf(((2.0f * x + 1.0f) * u * M_PI) / 16.0f);
    }
}

__global__ void performIDCTKernel(int *dOut, const int *dZigzag, const float *dIdctTable, int precision, int validWidth, int validHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < precision && y < precision && x < validWidth && y < validHeight) {
        float localSum = 0.0f;
        for (int u = 0; u < precision; u++) {
            for (int v = 0; v < precision; v++) {
                localSum += static_cast<float>(dZigzag[v * precision + u]) * dIdctTable[u * precision + x] * dIdctTable[v * precision + y];
            }
        }

        dOut[y * precision + x] = static_cast<int>(std::floor(localSum / 4.0f));
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
    idctTable.resize(precision * precision);
    float *dIdctTable;
    cudaMalloc((void **)&dIdctTable, precision * precision * sizeof(float));

    // Use a smaller block size (2, 2) to reduce occupancy and slow down the kernel
    dim3 blockSize(2, 2);
    dim3 gridSize((precision + blockSize.x - 1) / blockSize.x, (precision + blockSize.y - 1) / blockSize.y);

    initializeIDCTTableKernel<<<gridSize, blockSize>>>(dIdctTable, precision);
    cudaMemcpy(idctTable.data(), dIdctTable, precision * precision * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dIdctTable);
}

void IDCT::rearrangeUsingZigzag(int validWidth, int validHeight)
{
    int N = 8;

    int *dZigzag, *dBase;
    cudaMalloc((void **)&dZigzag, N * N * sizeof(int));
    cudaMalloc((void **)&dBase, N * N * sizeof(int));

    cudaMemcpy(dZigzag, zigzag.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dBase, base.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(N, N);
    dim3 gridSize(1, 1);

    rearrangeUsingZigzagkernel<<<gridSize, blockSize>>>(dZigzag, dBase, N, validWidth, validHeight);

    cudaMemcpy(zigzag.data(), dZigzag, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dZigzag);
    cudaFree(dBase);
}

void IDCT::performIDCT(int validWidth, int validHeight)
{
    int precision = IDCT_PRECISION;

    int* dZigzag, *dOut;
    float* dIdctTable;
    cudaMalloc((void **)&dZigzag, precision * precision * sizeof(int));
    cudaMalloc((void **)&dIdctTable, precision * precision * sizeof(float));
    cudaMalloc((void **)&dOut, precision * precision * sizeof(float));

    cudaMemcpy(dZigzag, zigzag.data(), precision * precision * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dIdctTable, idctTable.data(), precision * precision * sizeof(float), cudaMemcpyHostToDevice);

    // Use smaller block size (2, 2) to reduce occupancy
    dim3 blockSize(2, 2);
    dim3 gridSize((precision + blockSize.x - 1) / blockSize.x, (precision + blockSize.y - 1) / blockSize.y);

    performIDCTKernel<<<gridSize, blockSize>>>(dOut, dZigzag, dIdctTable, precision, validWidth, validHeight);
    cudaMemcpy(base.data(), dOut, precision * precision * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dZigzag);
    cudaFree(dIdctTable);
    cudaFree(dOut);
}