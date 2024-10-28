#include <iostream>
#include <vector>

class Stream {
    std::vector<uint8_t> data;
    int position; // Current location in the bit stream.

    public:
        Stream(std::vector<uint8_t>& data);
        int getBit(); 
        int getNBits(int n);
        uint8_t getByte();
        uint16_t getMarker();
        void getNBytes(std::vector<uint8_t>& arr, int n);
};