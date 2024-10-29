#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__device__ int clamp(int col) {
    return col > 255 ? 255 : (col < 0 ? 0 : col);
}

// 1. GetArray: Unpacks an array of elements based on type. Here, `type` is assumed as a single-byte element.
__global__ void GetArrayKernel(unsigned char* output, unsigned char* input, int length) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < length) {
        output[i] = input[i];
    }
}

void GetArray(unsigned char* host_output, unsigned char* host_input, int length) {
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, length * sizeof(unsigned char));
    cudaMalloc(&d_output, length * sizeof(unsigned char));

    cudaMemcpy(d_input, host_input, length * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (length + blockSize - 1) / blockSize;
    GetArrayKernel<<<numBlocks, blockSize>>>(d_output, d_input, length);

    cudaMemcpy(host_output, d_output, length * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

// 2. decode_number function
__device__ int decode_number(int code, int bits) {
    int l = 1 << (code - 1);
    return (bits >= l) ? bits : bits - (2 * l - 1);
}

// 3. clamp function: Defined above as `__device__ int clamp(int col)`

// 4. color_conversion function
__global__ void color_conversion_kernel(float* Y, float* Cr, float* Cb, int* R, int* G, int* B, int length) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < length) {
        float r = Cr[i] * (2 - 2 * 0.299) + Y[i];
        float b = Cb[i] * (2 - 2 * 0.114) + Y[i];
        float g = (Y[i] - 0.114 * b - 0.299 * r) / 0.587;

        R[i] = clamp(static_cast<int>(r + 128));
        G[i] = clamp(static_cast<int>(g + 128));
        B[i] = clamp(static_cast<int>(b + 128));
    }
}

void color_conversion(float* host_Y, float* host_Cr, float* host_Cb, int* host_R, int* host_G, int* host_B, int length) {
    float *d_Y, *d_Cr, *d_Cb;
    int *d_R, *d_G, *d_B;

    cudaMalloc(&d_Y, length * sizeof(float));
    cudaMalloc(&d_Cr, length * sizeof(float));
    cudaMalloc(&d_Cb, length * sizeof(float));
    cudaMalloc(&d_R, length * sizeof(int));
    cudaMalloc(&d_G, length * sizeof(int));
    cudaMalloc(&d_B, length * sizeof(int));

    cudaMemcpy(d_Y, host_Y, length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cr, host_Cr, length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cb, host_Cb, length * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (length + blockSize - 1) / blockSize;
    color_conversion_kernel<<<numBlocks, blockSize>>>(d_Y, d_Cr, d_Cb, d_R, d_G, d_B, length);

    cudaMemcpy(host_R, d_R, length * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_G, d_G, length * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_B, d_B, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_Y);
    cudaFree(d_Cr);
    cudaFree(d_Cb);
    cudaFree(d_R);
    cudaFree(d_G);
    cudaFree(d_B);
}
