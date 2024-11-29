#ifndef STREAM_H
#define STREAM_H

#include <cstdint>

/*
    Stream class is used for the easy manipulation of the byte stream.
*/

class Stream {
    uint8_t* data; // The stream data is read as unsigned integers.
    int position; // Current location in the bit stream.

    public:
        __device__ Stream(uint8_t* data);
        __device__ uint8_t getBit(); 
        __device__ uint16_t getNBits(int n);
        __device__ uint8_t getByte();
        __device__ uint16_t getMarker();
        __device__ void getNBytes(uint8_t* arr, int n);

        // Converts the value in byte stream to correct 2's complement.
        __device__ int decodeNumber(uint8_t code, int bits);
};

#endif
