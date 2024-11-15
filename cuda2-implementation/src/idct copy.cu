#include "idct.h"

// __global__ void rearrangeUsingZigzagkernel(int *dZigzag, const int *dBase, int N, int validWidth, int validHeight)
// {
//     int x = threadIdx.x;
//     int y = threadIdx.y;
//     int idx = x * N + y;

//     if (idx < N * N) {
//         if (x < validWidth && y < validHeight) {
//             dZigzag[idx] = dBase[dZigzag[idx]];
//         } else {
//             dZigzag[idx] = 0;
//         }
//     }
// }

// __global__ void rearrangeUsingZigzagkernel(const int *dBase, int N, int validWidth, int validHeight)
// {
//     int x = threadIdx.x;
//     int y = threadIdx.y;
//     int idx = x * N + y;

//     if (idx < N * N) {
//         if (x < validWidth && y < validHeight) {
//             d_zigzag[idx] = dBase[d_zigzag[idx]];
//         } else {
//             d_zigzag[idx] = 0;
//         }
//     }
// }

__global__ void rearrangeUsingZigzagkernel(int *dZigzag, int N, int validWidth, int validHeight)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int idx = x * N + y;

    if (idx < N * N) {
        if (x < validWidth && y < validHeight) {
            dZigzag[idx] = d_base[dZigzag[idx]];
        } else {
            dZigzag[idx] = 0;
        }
    }
}

__global__ void initializeIDCTTableKernel(double *dIdctTable, int precision)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (u < precision && x < precision) {
        double normCoeff = (u == 0) ? (1.0 / sqrt(2.0)) : 1.0;
        dIdctTable[u * precision + x] = normCoeff * cos(((2.0 * x + 1.0) * u * M_PI) / 16.0);
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
    // if (x < precision && y < precision && x < validWidth && y < validHeight) {
    //     extern __shared__ double sharedSum[];
    //     int tid = threadIdx.y * blockDim.x + threadIdx.x;

    //     for (int i = tid; i < precision * precision; i += blockDim.x * blockDim.y) {
    //         int u = i % precision;
    //         int v = i / precision;
    //         sharedSum[tid] += dZigzag[v * precision + u] * dIdctTable[u * precision + x] * dIdctTable[v * precision + y];
    //     }
    //     __syncthreads();

    //     for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1) {
    //         if (tid < stride) {
    //             sharedSum[tid] += sharedSum[tid + stride];
    //         }
    //         __syncthreads();  // Ensure all threads complete each reduction step
    //     }

    //     // Thread 0 in each block writes the final result to dOut
    //     if (tid == 0) {
    //         dOut[y * precision + x] = static_cast<int>(std::floor(sharedSum[0] / 4.0));
    //     }
    // }
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
    double *dIdctTable;
    cudaMalloc((void **)&dIdctTable, precision * precision * sizeof(double));

    dim3 blockSize(8, 8);
    dim3 gridSize((precision + blockSize.x - 1) / blockSize.x, (precision + blockSize.y - 1) / blockSize.y);

    initializeIDCTTableKernel<<<gridSize, blockSize>>>(dIdctTable, precision);
    cudaMemcpy(idctTable.data(), dIdctTable, precision * precision * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dIdctTable);
}

void IDCT::rearrangeUsingZigzag(int validWidth, int validHeight)
{
    int N = 8;

    //int *dZigzag, *dBase;
    int *dBase;
    //cudaMalloc((void **)&dZigzag, N * N * sizeof(int));
    //cudaMalloc((void **)&dBase, N * N * sizeof(int));

    //cudaMemcpy(dZigzag, zigzag.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
   // cudaMemcpy(dBase, base.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(N, N);
    dim3 gridSize(1, 1);

    //rearrangeUsingZigzagkernel<<<gridSize, blockSize>>>(dZigzag, dBase, N, validWidth, validHeight);
    //rearrangeUsingZigzagkernel<<<gridSize, blockSize>>>(dBase, N, validWidth, validHeight);
//    rearrangeUsingZigzagkernel<<<gridSize, blockSize>>>(N, validWidth, validHeight);

    cudaMemcpy(zigzag.data(), d_zigzag, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dZigzag);
    cudaFree(dBase);
}

void IDCT::performIDCT(int validWidth, int validHeight)
{

    //cudaMalloc((void **)&dZigzag, N * N * sizeof(int));
    //cudaMalloc((void **)&dBase, N * N * sizeof(int));

    //cudaMemcpy(dZigzag, zigzag.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
   // cudaMemcpy(dBase, base.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);


    int *dZigzag, *dBase;
    int *dOut;
    double* dIdctTable;
    cudaMalloc((void **)&dIdctTable, precision * precision * sizeof(double));
    cudaMalloc((void **)&dOut, precision * precision * sizeof(int));
    cudaMalloc((void **)&dZigzag, N * N * sizeof(int));
    cudaMalloc((void **)&dBase, N * N * sizeof(int));

    dim3 blockSize(8, 8);
    dim3 gridSize(1, 1);

    //rearrangeUsingZigzagkernel<<<gridSize, blockSize>>>(dZigzag, dBase, N, validWidth, validHeight);
    //rearrangeUsingZigzagkernel<<<gridSize, blockSize>>>(dBase, N, validWidth, validHeight);
    cudaMemcpy(dZigzag, zigzag.data(), precision * precision * sizeof(int), cudaMemcpyHostToDevice);
    rearrangeUsingZigzagkernel<<<gridSize, blockSize>>>(dZigzag, N, validWidth, validHeight);

    


    

    //cudaMemcpy(dZigzag, zigzag.data(), precision * precision * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dIdctTable, idctTable.data(), precision * precision * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockSize(8, 8);
    dim3 gridSize((precision + blockSize.x - 1) / blockSize.x, (precision + blockSize.y - 1) / blockSize.y);

    performIDCTKernel<<<gridSize, blockSize>>>(dOut, dZigzag, dIdctTable, precision, validWidth, validHeight);
    cudaMemcpy(base.data(), dOut, precision * precision * sizeof(int), cudaMemcpyDeviceToHost);

    //cudaFree(dZigzag);
    cudaFree(dIdctTable);
    cudaFree(dOut);
}