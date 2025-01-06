#ifndef STREAM_H
#define STREAM_H

#include <vector>
#include <cstdint>

/*
    Stream class is used for the easy manipulation of the byte stream.
*/

class Stream {
    std::vector<uint8_t> data; // The stream data is read as unsigned integers.
    int position; // Current location in the bit stream.

    public:
        Stream(std::vector<uint8_t>& data);
        uint8_t getBit(); 
        uint16_t getNBits(int n);
        uint8_t getByte();
        uint16_t getMarker();
        void getNBytes(std::vector<uint8_t>& arr, int n);

        // Converts the value in byte stream to correct 2's complement.
        static int16_t decodeNumber(uint8_t code, int bits);
};

#endif
