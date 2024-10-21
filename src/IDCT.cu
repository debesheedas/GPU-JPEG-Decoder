#include "IDCT.h"
#include <cmath>
#include <math.h>

__global__ void rearrange_using_zigzag_kernel(int *d_zigzag, cont int *d_base, int N)
{
    int x = threadIdx.x;
    int y = threadIdx.y;

    int idx = x * N + y;

    d_zigzag[idx] = d_base[zigzag[idx]];
}

__global__ void initialize_idct_table_kernel(float *d_idct_table, int precision)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (u < precision)
    {
        if (x < precision)
        {
            float normCoeff = (u == 0) ? (1.0f / sqrtf(2)) : 1.0f;
            d_idct_table[u * precision + x] = normCoeff * cosf(((2.0f * x + 1.0f) * u * M_PI) / 16.0f);
        }
    }
}

__global__ void perform_IDCT_kernel(float *d_out, const float *d_zigzag, const float *d_idct_table, int precision)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < precision)
    {
        if (y < precision)
        {
            float local_sum = 0;

            for (int u = 0; u < precision; u++)
            {
                for (int v = 0; v < precision; v++)
                {
                    local_sum += d_zigzag[v * precision + u] * d_idct_table[u * precision + x] * d_idct_table[v * precision + y];
                }
            }

            d_out[y * precision + x] = local_sum / 4.0f;
        }
    }
}

void IDCT::initialize_idct_table()
{
    int precision = this->idct_precision;
    float *d_idct_table;
    cudaMalloc((void **)&d_idct_table, precision * precision * sizeof(float));

    dim3 blockSize(8, 8);
    dim3 gridSize((precision + blockSize.x - 1) / blockSize.x, (precision + blockSize.y - 1) / blockSize.y);

    initialize_idct_table_kernel<<<gridSize, blockSize>>>(d_idct_table, precision);
    cudaMemcpy(idct_table, d_idct_table, precision * precision * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_idct_table);
}

int IDCT::rearrange_using_zigzag()
{
    int h_zigzag[64];
    int N = 8;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_zigzag[i * N + j] = zigzag[i][j];
        }
    }

    int *d_zigzag;
    int *d_base;
    cudaMalloc((void **)&d_zigzag, N * N * sizeof(int));
    cudaMalloc((void **)&d_base, N * N * sizeof(int));

    cudaMemcpy(d_zigzag, h_zigzag, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_base, base, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(N, N);
    dim3 gridSize(1, 1);

    rearrange_using_zigzag_kernel<<<gridSize, blockSize>>>(d_zigzag, d_base, N);

    cudaMemcpy(h_zigzag, d_zigzag, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            zigzag[i][j] = h_zigzag[i * N + j];
        }
    }
    cudaFree(d_zigzag);
    cudaFree(d_base);
    return zigzag;
}

void IDCT::perform_IDCT()
{
    int N = 8;
    int precision = self.idct_precision;
    float out[precision * precision];

    int *d_zigzag;
    float *d_idct_table;
    float *d_out;
    cudaMalloc((void **)&d_zigzag, precision * precision * sizeof(float));
    cudaMalloc((void **)&d_idct_table, precision * precision * sizeof(float));
    cudaMalloc((void **)&d_out, precision * precision * sizeof(float));

    cudaMemcpy(d_zigzag, h_zigzag, precision * precision * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idct_table, h_idct_table, precision * precision * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(8, 8);
    dim3 gridSize((precision + blockSize.x - 1) / blockSize.x, (precision + blockSize.y - 1) / blockSize.y);

    performIDCTKernel<<<gridSize, blockSize>>>(d_out, d_zigzag, d_idct_table, precision);

    cudaMemcpy(out, d_out, precision * precision * sizeof(float), cudaMemcpyDeviceToHost);
    this->base = out;
}
