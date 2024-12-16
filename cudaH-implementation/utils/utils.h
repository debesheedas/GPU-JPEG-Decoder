#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <cuda_runtime.h>

// Device functions
__device__ uint8_t getBit(uint8_t* data, int& position);
__device__ uint16_t getNBits(uint8_t* data, int& position, int n);
__device__ uint8_t getByte(uint8_t* data, int& position);
__device__ void getNBytes(uint8_t* arr, int length, uint8_t* data, int& position);
__device__ int16_t decodeNumber(uint8_t code, int bits);

#endif