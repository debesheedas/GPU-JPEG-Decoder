#include "parser.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>


__device__ int match_huffman_code(uint8_t* stream, int bit_offset, uint16_t* huff_codes, int* huff_bits, int &code, int &length) {
    unsigned int extracted_bits = getNBits(stream, bit_offset, 16);
    // Compare against Huffman table
    for (int i = 0; i < 256; ++i) {
        if (huff_bits[i] > 0 && huff_bits[i] <= 16) { // Valid bit length
            unsigned int mask = (1 << huff_bits[i]) - 1;
            if ((extracted_bits >> (16 - huff_bits[i]) & mask) == huff_codes[i]) {
                code = i;
                length = huff_bits[i];
                return i;
            }
        }
    }
    return -1;
}

__device__ void koggeStonePrefixSumRange(int *arr, int n, int i, int k, int j) {
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int ind = tid * k + i;
    int numSegments = (((j - i) / k) + blockSize - 2) / (blockSize - 1);

    int start = i;
    int end = min(start + blockSize * k, j);

    for (int seg = 0; seg < numSegments; seg++) {
        if (ind >= start && ind < end) {
            for (int offset = k; offset < end - start; offset *= 2) {
                int temp = 0;
                if (ind >= start + offset) {
                    temp = arr[ind - offset];
                }
                __syncthreads();
                if (ind < n) {
                    arr[ind] += temp;
                }
                __syncthreads();
            }
        }

        ind += (blockSize - 1) * k;
        start += (blockSize - 1) * k;
        end = min(start + blockSize * k, j);
    }
}

__device__ void koggeStonePrefixSumRange16(int16_t *arr, int n, int i, int k, int j) {
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int ind = tid * k + i;
    int numSegments = (((j - i) / k) + blockSize - 2) / (blockSize - 1);

    int start = i;
    int end = min(start + blockSize * k, j);

    for (int seg = 0; seg < numSegments; seg++) {
        if (ind >= start && ind < end) {
            for (int offset = k; offset < end - start; offset *= 2) {
                int16_t temp = 0;
                if (ind >= start + offset) {
                    temp = arr[ind - offset];
                }
                __syncthreads();
                if (ind < n) {
                    arr[ind] += temp;
                }
                __syncthreads();
            }
        }

        ind += (blockSize - 1) * k;
        start += (blockSize - 1) * k;
        end = min(start + blockSize * k, j);
    }
}

__global__ void parallel_prefix_sum(int *arr, int n) {
    int tid = threadIdx.x; 
    for (int step = 1; step < n; step *= 2) {
        int temp = 0;
        if (tid >= step) {
            temp = arr[tid - step];
        }
        __syncthreads(); 
        arr[tid] += temp;
        __syncthreads();
    }
}

__device__ void parallelPrefixSumWithThrust(int* arr, int n) {
    thrust::device_ptr<int> dev_ptr(arr);
    thrust::exclusive_scan(dev_ptr, dev_ptr + n, dev_ptr);  // Perform the exclusive scan (prefix sum)
}

__global__ void parallelPrefixSumWithThrustDevice(int *arr, int n) {
    __shared__ int temp[256];  // Shared memory for each block

    int tid = threadIdx.x;  // Thread ID
    int index = threadIdx.x + blockIdx.x * blockDim.x;  // Global index for arr

    // Step 1: Load data into shared memory
    if (index < n) {
        temp[tid] = arr[index];
    } else {
        temp[tid] = 0;  // Padding if out of bounds
    }
    __syncthreads();

    // Step 2: Perform exclusive scan using shared memory (Kogge-Stone-like approach)
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int val = 0;
        if (tid >= offset) {
            val = temp[tid - offset];
        }
        __syncthreads(); // Synchronize before update
        temp[tid] += val;
        __syncthreads(); // Synchronize after update
    }

    // Step 3: Write back to global memory
    if (index < n) {
        arr[index] = temp[tid];  // Write the result to the output array
    }
}



__device__ void decodeSequence(
    int seqInd, int start, int end, bool overflow, bool write,
    int16_t* outBuffer, int* sInfo, uint8_t* imageData, uint16_t* hfCodes, int* hfLengths, int* returnState, int width, int height, int imageDataLength) {

    // decode state variables
    int p = start; // start bit offset for the subsequence
    int n = 0;     // number of decoded symbols
    int c = 0;     // current component (Y, assuming)
    int z = 0;     // zig-zag index
    int posInOutput = 0;

    // load the state from the previous sequence if in overflow mode
    if (write && seqInd > 0) {
        posInOutput = sInfo[(seqInd - 1) * 4 + 1];
    }
    if (overflow) {
        p = sInfo[(seqInd - 1) * 4 + 0];
        c = sInfo[(seqInd - 1) * 4 + 2];
        z = sInfo[(seqInd - 1) * 4 + 3];
    }
    // decode symbols in the subsequence
    while (p < end) {
        int code = 1, codeLength = 1;

        // select huffman table based on the current component and zig-zag index
        uint16_t dcOffset = (c > 0) * 256;
        uint16_t acOffset = (z > 0) * 512;

        match_huffman_code(imageData, p, hfCodes + dcOffset + acOffset, hfLengths + dcOffset + acOffset, code, codeLength);

        p += codeLength;

        if (p > imageDataLength) break;

        int leadingZeros = 0;
        if ((code == 0) && (z > 0)) {
            leadingZeros = 63 - z;
        }
        else if ((code > 15) && (z > 0)) {
            leadingZeros += (code >> 4);
            code = code & 0x0F;
        }

        if (write) {
            uint16_t bits = getNBits(imageData, p, code);
            int16_t decoded = decodeNumber(code, bits);
            int writeInd = c * width * height + (posInOutput / 192) * 64 + (posInOutput % 64);
            outBuffer[writeInd] = decoded;
        }
        else {
            p += code;
        }

        //increment the state values
        n += leadingZeros + 1;
        z += leadingZeros + 1;
        posInOutput += leadingZeros + 1;

        if (z >= 64) {
            z = 0;
            c = (c + 1) % 3;
        }
    }

    returnState[0] = p;
    returnState[1] = n;
    returnState[2] = c;
    returnState[3] = z;
}

__device__ bool isSynchronized(int* returnState, int* storedState) {
    // synchronisation occurs when all key state variables match
    return storedState[0] == returnState[0] && // bit offset
           storedState[2] == returnState[2] && // component
           storedState[3] == returnState[3];   // zig-zag index
}

__device__ void parallelHuffManDecode(
    uint8_t* imageData, int imageDataLength, int16_t* outBuffer, 
    uint16_t* hfCodes, int* hfLengths, int width, int height, 
    int blockSize, int* sInfo) {

    int threadId = threadIdx.x;

    // divide the bitstream into even parts
    int segmentSize = (imageDataLength + blockSize - 1) / blockSize;
    int start = threadId * segmentSize;
    int end = min(start + segmentSize, imageDataLength);

    //printf("imageDataLength:%d\n", imageDataLength);

    // decode the segment assigned to this thread
    int returnState[4]; // array to hold decoding state [p, n, c, z]
    decodeSequence(threadId, start, end, false, false, outBuffer, sInfo, imageData, hfCodes, hfLengths, returnState, width, height, imageDataLength);

    sInfo[threadId * 4 + 0] = returnState[0]; // bit offset
    sInfo[threadId * 4 + 1] = returnState[1]; // symbols decoded
    sInfo[threadId * 4 + 2] = returnState[2]; // component (Y/Cb/Cr)
    sInfo[threadId * 4 + 3] = returnState[3]; // zig-zag index

    __syncthreads();

    int seqInd = threadId + 1; // overflow to the next segment
    bool isSynced = false;
    for (int i = 0; i < blockSize; i++) {
        if ((!isSynced) && (seqInd < blockSize)) {
            start = seqInd * segmentSize;
            end = min(start + segmentSize, imageDataLength);
            // decode the current sequence
            decodeSequence(seqInd, start, end, true, false, outBuffer, sInfo, imageData, hfCodes, hfLengths, returnState, width, height, imageDataLength);

            isSynced = isSynchronized(returnState, &sInfo[seqInd * 4]);

            // // Print the current sequence index and returnState for debugging
            // if (threadId == 0) {
            //     printf("Thread: %d, Sequence: %d\n", threadId, seqInd);
            //     printf("ReturnState: [p: %d, n: %d, c: %d, z: %d]\n", returnState[0], returnState[1], returnState[2], returnState[3]);

            //     printf("sInfo:\n");
            //     for (int i = 0; i < blockSize; i++) {
            //         printf("  sInfo[%d]: [p: %d, n: %d, c: %d, z: %d]\n",
            //             i, sInfo[i * 4 + 0], sInfo[i * 4 + 1], sInfo[i * 4 + 2], sInfo[i * 4 + 3]);
            //     }
            // }

            sInfo[seqInd * 4 + 0] = returnState[0]; // bit offset
            sInfo[seqInd * 4 + 1] = returnState[1]; // symbols decoded
            sInfo[seqInd * 4 + 2] = returnState[2]; // current component
            sInfo[seqInd * 4 + 3] = returnState[3]; // zigzag index

            seqInd++;
        }
        __syncthreads();
    }

    // Perform exclusive prefix sum to calculate offsets
    //koggeStonePrefixSumRange(sInfo, 4 * blockSize, 1, 4, 4 * blockSize);

    // if (threadId == 0) {
    //     int sum = 0;
    //     int ind = 1;
    //     while (ind < 4 * blockSize) {
    //         sum += sInfo[ind];
    //         sInfo[ind] = sum;
    //         ind += 4;
    //     }
    // }

    // __syncthreads();
    if (threadId == 0) {
    parallelPrefixSumWithThrustDevice(sInfo, 4 * blockSize);
    }
    
    __syncthreads();
    if (threadId == 0) {
        printf("sInfo after prefix sum\n");
        for (int i = 0; i < blockSize; i++) {
            printf("  sInfo[%d]: [p: %d, n: %d, c: %d, z: %d]\n",
                i, sInfo[i * 4 + 0], sInfo[i * 4 + 1], sInfo[i * 4 + 2], sInfo[i * 4 + 3]);
        }
    }
    __syncthreads();
    // re-decode with corrected offsets

    start = threadId * segmentSize;
    end = min(start + segmentSize, imageDataLength);
    if (threadId == 0) {
        decodeSequence(threadId, start, end, false, true, outBuffer, sInfo, imageData, hfCodes, hfLengths, returnState, width, height, imageDataLength);
    } else {
        decodeSequence(threadId, start, end, true, true, outBuffer, sInfo, imageData, hfCodes, hfLengths, returnState, width, height, imageDataLength);
    }
    __syncthreads();
    int totalPixels = (width * height);

    // THIS GETS STUCK :(
    // for (int c = 0; c < 3; ++c) { // Process each channel (Y, Cb, Cr)
    //
    //     koggeStonePrefixSumRange16(outBuffer, 3 * totalPixels, c * totalPixels, 64, (c+1) * totalPixels);
    // }

    if (threadId < 3) {
        int16_t dcCoeff = 0;
        int ind = threadId * totalPixels;
        while (ind < (threadId + 1) * totalPixels) {
            dcCoeff += outBuffer[ind];
            outBuffer[ind] = dcCoeff;
            ind += 64;
        }
    }
}



__device__ int16_t clip(int16_t value) {
    value = max(-256, value);
    value = min(255, value);
    return value;
}
__device__ void idctRow(int16_t* block) {
    int x0, x1, x2, x3, x4, x5, x6, x7;

    // Extract and scale coefficients
    x0 = (block[0] << 11) + 128; // Scale DC coefficient
    x1 = block[4] << 11;
    x2 = block[6];
    x3 = block[2];
    x4 = block[1];
    x5 = block[7];
    x6 = block[5];
    x7 = block[3];

    // Perform IDCT calculations
    int x8 = C7 * (x4 + x5);
    x4 = x8 + (C1 - C7) * x4;
    x5 = x8 - (C1 + C7) * x5;
    x8 = C3 * (x6 + x7);
    x6 = x8 - (C3 - C5) * x6;
    x7 = x8 - (C3 + C5) * x7;

    x8 = x0 + x1;
    x0 -= x1;
    x1 = C6 * (x3 + x2);
    x2 = x1 - (C2 + C6) * x2;
    x3 = x1 + (C2 - C6) * x3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;

    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;

    // Store results
    block[0] = (x7 + x1) >> 8;
    block[1] = (x3 + x2) >> 8;
    block[2] = (x0 + x4) >> 8;
    block[3] = (x8 + x6) >> 8;
    block[4] = (x8 - x6) >> 8;
    block[5] = (x0 - x4) >> 8;
    block[6] = (x3 - x2) >> 8;
    block[7] = (x7 - x1) >> 8;
}

__device__ void idctCol(int16_t* block) {
    int x0, x1, x2, x3, x4, x5, x6, x7;

    // Extract and scale coefficients
    x0 = (block[8 * 0] << 8) + 8192; // Scale DC coefficient
    x1 = block[8 * 4] << 8;
    x2 = block[8 * 6];
    x3 = block[8 * 2];
    x4 = block[8 * 1];
    x5 = block[8 * 7];
    x6 = block[8 * 5];
    x7 = block[8 * 3];

    // Perform IDCT calculations
    int x8 = C7 * (x4 + x5) + 4;
    x4 = (x8 + (C1 - C7) * x4) >> 3;
    x5 = (x8 - (C1 + C7) * x5) >> 3;
    x8 = C3 * (x6 + x7) + 4;
    x6 = (x8 - (C3 - C5) * x6) >> 3;
    x7 = (x8 - (C3 + C5) * x7) >> 3;

    x8 = x0 + x1;
    x0 -= x1;
    x1 = C6 * (x3 + x2) + 4;
    x2 = (x1 - (C2 + C6) * x2) >> 3;
    x3 = (x1 + (C2 - C6) * x3) >> 3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;

    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;

    // Store results with clipping
    block[8 * 0] = clip((x7 + x1) >> 14);
    block[8 * 1] = clip((x3 + x2) >> 14);
    block[8 * 2] = clip((x0 + x4) >> 14);
    block[8 * 3] = clip((x8 + x6) >> 14);
    block[8 * 4] = clip((x8 - x6) >> 14);
    block[8 * 5] = clip((x0 - x4) >> 14);
    block[8 * 6] = clip((x3 - x2) >> 14);
    block[8 * 7] = clip((x7 - x1) >> 14);
}


void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

/* Function to allocate the GPU space. */
void allocate(uint16_t*& hfCodes, int*& hfLengths, std::unordered_map<int,HuffmanTree*>& huffmanTrees,
              int16_t*& yCrCbChannels, int16_t*& rgbChannels, int16_t*& outputChannels, int width, int height,
              int*& zigzagLocations, int*& sInfo, int maxThreads) {

    const size_t codeSize = 256 * sizeof(uint16_t);
    const size_t lengthSize = 256 * sizeof(int);
    const size_t imageSize = width * height * sizeof(int16_t);

    checkCudaError(cudaMalloc((void**)&hfCodes, codeSize * 4), "Failed to allocate device memory for huffman codes.");
    checkCudaError(cudaMalloc((void**)&hfLengths, lengthSize * 4), "Failed to allocate device memory for huffman lengths.");

    checkCudaError(cudaMalloc((void**)&zigzagLocations, 256 * sizeof(int)), "Failed to allocate device memory for zigzag table.");
    checkCudaError(cudaMemcpy(zigzagLocations, zigzagEntries, sizeof(int) * 64, cudaMemcpyHostToDevice), "Failed to copy entries for the zigzag table.");

    int index = 0;

    for (int i = 0; i < 4; ++i) {
        if (i > 1 && index < 16)
            index = 16;

        checkCudaError(cudaMemcpy(hfCodes + i * 256, huffmanTrees[index]->codes, codeSize, cudaMemcpyHostToDevice), "Failed to copy data to device for huffman codes");
        checkCudaError(cudaMemcpy(hfLengths + i * 256, huffmanTrees[index]->codeLengths, lengthSize, cudaMemcpyHostToDevice), "Failed to copy data to device for huffman lengths");
        index++;
    }

    // Allocating the channels in the GPU memory.
    checkCudaError(cudaMalloc((void**)&yCrCbChannels, imageSize * 3), "Failed to allocate device memory for one yCrCb channel.");
    checkCudaError(cudaMalloc((void**)&rgbChannels, imageSize * 3), "Failed to allocate device memory for one rgb channel.");
    checkCudaError(cudaMalloc((void**)&outputChannels, imageSize * 3), "Failed to allocate device memory for one output channel.");

    // Allocate memory for sInfo
    checkCudaError(cudaMalloc((void**)&sInfo, 4 * maxThreads * sizeof(int)), "Failed to allocate memory for sInfo.");
}

void extract(std::string imagePath, uint8_t*& quantTables, uint8_t*& imageData, int& numBits, int& width, int& height, std::unordered_map<int,HuffmanTree*>& huffmanTrees) {  

    fs::path file_path(imagePath);
    std::string filename = file_path.filename().string();
    std::ifstream input(imagePath, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()));
    input.close();      

    uint16_t tableSize = 0;
    uint8_t header = 0;
    int imageDataLength = 0;

    // Using the Stream class for reading bytes.
    Stream stream(bytes);
    while (true) {
        uint16_t marker = stream.getMarker();
        if (marker == MARKERS[0]) {
            continue;
        } else if (marker == MARKERS[1]) {
            tableSize = stream.getMarker();
            std::vector<uint8_t> appHeader((int) tableSize - 2);
            stream.getNBytes(appHeader, int(tableSize - 2));
        } else if (marker == MARKERS[2]) {
            stream.getMarker();
            uint8_t destination = stream.getByte();
            checkCudaError(cudaMalloc((void**)& quantTables, 128 * sizeof(uint8_t)), "Failed to allocate device memory for first quant table.");

            std::vector<uint8_t> hostQuantTable(64);
            stream.getNBytes(hostQuantTable, 64);

            checkCudaError(cudaMemcpy(quantTables, hostQuantTable.data(), 64 * sizeof(uint8_t), cudaMemcpyHostToDevice),"Failed to copy data to device for first quant");

            if(stream.getMarker() == MARKERS[2]) {
                stream.getMarker();
                destination = stream.getByte();
                stream.getNBytes(hostQuantTable, 64);
                checkCudaError(cudaMemcpy(quantTables + 64, hostQuantTable.data(), 64 * sizeof(uint8_t), cudaMemcpyHostToDevice),"Failed to copy data to device for second quant");
            } else {
                std::cout << " Something went wrong at parsing second quant table." << std::endl;
            }
        } else if (marker == MARKERS[3]) {
            tableSize = stream.getMarker();
            std::vector<uint8_t> startOfFrame((int) tableSize - 2);
            stream.getNBytes(startOfFrame, (int) tableSize - 2);
            Stream frame(startOfFrame);
            int precision = frame.getByte();
            height = frame.getMarker();
            width = frame.getMarker();
        } else if (marker == MARKERS[4]) {
            tableSize = stream.getMarker();
            header = stream.getByte();

            std::vector<uint8_t> huffmanTable((int) tableSize - 3);

            stream.getNBytes(huffmanTable, (int) tableSize - 3);
            huffmanTrees[(int)header] = new HuffmanTree(huffmanTable);

            if (stream.getMarker() ==  MARKERS[4]) {
                tableSize = stream.getMarker();
                header = stream.getByte();
                huffmanTable.resize((int) tableSize - 3);
                stream.getNBytes(huffmanTable, (int) tableSize - 3);
                huffmanTrees[(int)header] = new HuffmanTree(huffmanTable);
            }

            if (stream.getMarker() ==  MARKERS[4]) {
                tableSize = stream.getMarker();
                header = stream.getByte();
                huffmanTable.resize((int) tableSize - 3);
                stream.getNBytes(huffmanTable, (int) tableSize - 3);
                huffmanTrees.emplace((int)header, new HuffmanTree(huffmanTable));
            }

            if (stream.getMarker() ==  MARKERS[4]) {
                tableSize = stream.getMarker();
                header = stream.getByte();
                huffmanTable.resize((int) tableSize - 3);
                stream.getNBytes(huffmanTable, (int) tableSize - 3);
                huffmanTrees.emplace((int)header, new HuffmanTree(huffmanTable));
            }

        } else if (marker == MARKERS[5]) {
            tableSize = stream.getMarker();

            std::vector<uint8_t> startOfScan((int) tableSize - 2);
            stream.getNBytes(startOfScan, (int) tableSize - 2);
            uint8_t curByte, prevByte = 0x00;
            size_t size = 5 * 1024 * 1024;
            std::vector<uint8_t> hostImageData(size);
            
            while (true) {
                curByte = stream.getByte();
                if ((prevByte == 0xff) && (curByte == 0xd9))
                    break;
                if (curByte == 0x00) {
                    if (prevByte != 0xff) {
                        hostImageData[imageDataLength++] = curByte;
                    }
                } else {
                    hostImageData[imageDataLength++] = curByte;
                }
                prevByte = curByte;
            }
            
            imageDataLength--; // We remove the ending byte because it is extra 0xff.
            checkCudaError(cudaMalloc((void**)& imageData, imageDataLength * sizeof(uint8_t)), "Failed to allocate device memory for image bytes.");
            checkCudaError(cudaMemcpy(imageData, hostImageData.data(), imageDataLength * sizeof(uint8_t), cudaMemcpyHostToDevice),"Failed to copy data to device for image data.");
            break;
        }
    } 
    numBits = imageDataLength * 8;
}

__device__ int buildMCU(int16_t* outBuffer, uint8_t* imageData, int bitOffset,
                        int& oldCoeff, uint16_t* dcHfcodes, int* dcHflengths, uint16_t* acHfcodes, int* acHflengths) {
    int code = 0;
    int code_length = 0;
    match_huffman_code(imageData, bitOffset, dcHfcodes, dcHflengths, code, code_length);
    // printf("dc code %d:\n", code);
    bitOffset += code_length;
    uint16_t bits = getNBits(imageData, bitOffset, code);

    int16_t decoded = decodeNumber(code, bits); 
    int16_t dcCoeff = decoded + oldCoeff;
    // printf("dc coeff %d:\n", dcCoeff);
    outBuffer[0] = dcCoeff;

    int length = 1;
    while (length < 64) {
        match_huffman_code(imageData, bitOffset, acHfcodes, acHflengths, code, code_length);
        // printf("code %d:\n", code);
        bitOffset += code_length;
        if (code == 0) {
            break;
        }
        // The first part of the AC key length is the number of leading zeros
        if (code > 15) {
            length += (code >> 4);
            code = code & 0x0f;
        }
        bits = getNBits(imageData, bitOffset, code);
        if (length < 64) {
            decoded = decodeNumber(code, bits);
            // printf("ac coeff %d:\n", decoded);
            outBuffer[length] = decoded;
            length++;
        }
    }
    // Update oldCoeff for the next MCU
    oldCoeff = dcCoeff;
    return bitOffset;
}

__device__ void performHuffmanDecoding(uint8_t* imageData, int16_t* yCrCbChannels, uint16_t* hfCodes, int* hfLengths, int width, int height) {
    int16_t* curLuminous = yCrCbChannels;
    int16_t* curChromRed = yCrCbChannels + width * height;
    int16_t* curChromYel = yCrCbChannels + 2 * width * height;
    int oldLumCoeff = 0, oldCbdCoeff = 0, oldCrdCoeff = 0;
    int bitOffset = 0;

    int xBlocks = width / 8;
    int yBlocks = height / 8;

    for (int y = 0; y < yBlocks; y++) {
        for (int x = 0; x < xBlocks; x++) {
            bitOffset = buildMCU(curLuminous, imageData, bitOffset, oldLumCoeff, hfCodes, hfLengths, hfCodes + 512, hfLengths+ 512);
            bitOffset = buildMCU(curChromRed, imageData, bitOffset, oldCbdCoeff, hfCodes + 256, hfLengths + 256, hfCodes + 768, hfLengths + 768);
            bitOffset = buildMCU(curChromYel, imageData, bitOffset, oldCrdCoeff, hfCodes + 256, hfLengths + 256, hfCodes + 768, hfLengths + 768);
            curLuminous += 64;
            curChromRed += 64;
            curChromYel += 64;
        }
    }
}

// New idea: Let 32 threads handle 4 8x8 blocks simulatenously instead of 1 block at a time, this way we can avoid syncs
__device__ void performZigzagReordering(int16_t* yCrCbChannels, int16_t* rgbChannels, uint8_t* quantTables,
                                        int blockIndex, int startIndex, int totalPixels, int channelId, int* zigzagLocations) {
                                            
    for (int index = startIndex; index < startIndex + 8; index++) {
        rgbChannels[channelId * totalPixels + index] = yCrCbChannels[channelId * totalPixels + blockIndex * 64 + zigzagLocations
        [index % 64]] * quantTables[(64 & -(channelId > 0)) + zigzagLocations[index % 64]];
    }
}

__device__ void performColorConversion(int16_t* rgbChannels, int16_t* outputChannels,
                                       int totalPixels, int width, int threadId, int blockSize) {
    int pixelIndex = threadId;
    while (pixelIndex * 8 < totalPixels) {
        // do colour conversion row-wise
        int start = (pixelIndex / 8) * 64 + (pixelIndex % 8) * 8;
        for (int i = start; i < start + 8; i++) {
            // Compute global row and column directly
            // Explanation:
            // - Each block contains 8x8 pixels (64 pixels total).
            // - i / 64 gives the block ID.
            // - (i / 64) / (width / 8) computes the block's row index in the grid (each block is 8 rows high).
            // - (i / 64) % (width / 8) computes the block's column index in the grid (each block is 8 columns wide).
            // - (i % 64) / 8 gives the row within the block, and (i % 64) % 8 gives the column within the block.
            int globalRow = ((i / 64) / (width / 8)) * 8 + (i % 64) / 8;
            int globalColumn = ((i / 64) % (width / 8)) * 8 + (i % 64) % 8;

            int actualIndex = globalRow * width + globalColumn;

            float red = rgbChannels[2 * totalPixels + i] * (2 - 2 * 0.299) + rgbChannels[i];
            float blue = rgbChannels[totalPixels + i] * (2 - 2 * 0.114) + rgbChannels[i];
            float green = (rgbChannels[i] - 0.114 * blue - 0.299 * red) / 0.587;

            outputChannels[actualIndex] = min(max(static_cast<int16_t>(red + 128), 0), 255);
            outputChannels[totalPixels + actualIndex] = min(max(static_cast<int16_t>(green + 128), 0), 255);
            outputChannels[2 * totalPixels + actualIndex] = min(max(static_cast<int16_t>(blue + 128), 0), 255);
        }
        pixelIndex += blockSize;
    }
}

__global__ void decodeKernel(
    uint8_t* imageData,
    int imageDataLength,
    int16_t* yCrCbChannels,
    int16_t* rgbChannels,
    int16_t* outputChannels,
    int width,
    int height,
    uint8_t* quantTables,
    uint16_t* hfCodes,
    int* hfLengths,
    int* zigzagLocations,
    int* sInfo) // Pass global memory for sInfo
{
    int threadId = threadIdx.x;  // Thread ID within the block
    int blockSize = blockDim.x;  // Total number of threads in the block

    // Call decodeImage with all necessary parameters
    decodeImage(
        imageData,
        imageDataLength,
        yCrCbChannels,
        rgbChannels,
        outputChannels,
        width,
        height,
        quantTables,
        hfCodes,
        hfLengths,
        zigzagLocations,
        sInfo,        // Pass the global memory pointer for sInfo
        threadId,
        blockSize
    );
}
__device__ void decodeImage(uint8_t* imageData, int imageDataLength, int16_t* yCrCbChannels, int16_t* rgbChannels, int16_t* outputChannels, int width, int height, uint8_t* quantTables, uint16_t* hfCodes, int* hfLengths, int* zigzagLocations, int* sInfo, int threadId, int blockSize) {

    __shared__ int zigzagMap[1024];

    int pixelIndex = threadId;
    while (pixelIndex < 64) {
        zigzagMap[pixelIndex] = zigzagLocations[pixelIndex];
        pixelIndex += blockSize;
    }

    int totalPixels = width * height;
    
    parallelHuffManDecode(imageData, imageDataLength, yCrCbChannels, hfCodes, hfLengths, width, height, blockSize, sInfo);

    __syncthreads();

    for (int channel = 0; channel < 3; channel++) {
        pixelIndex = threadId;
        while (pixelIndex * 8 < totalPixels) {
            int startIndex = (pixelIndex / 8) * 64 + (pixelIndex % 8) * 8;
            int blockIndex = startIndex / 64;

            performZigzagReordering(yCrCbChannels, rgbChannels, quantTables,
                                    blockIndex, startIndex, totalPixels, channel, zigzagMap);

            pixelIndex += blockSize;
        }
    }

    for (int channel = 0; channel < 3; channel++) {
        pixelIndex = threadId;
        while (pixelIndex * 8 < totalPixels) {        
            int start = (pixelIndex / 8) * 64 + (pixelIndex % 8) * 8;
            idctRow(rgbChannels + channel * totalPixels + start);
            pixelIndex += blockSize;
        }

        pixelIndex = threadId;
        while (pixelIndex * 8 < totalPixels) {
            int start = (pixelIndex / 8) * 64 + (pixelIndex % 8);
            idctCol(rgbChannels + channel * totalPixels + start);
            pixelIndex += blockSize;
        }
    }

    performColorConversion(rgbChannels, outputChannels, totalPixels, width, threadId, blockSize);

}

__global__ void batchDecodeKernel(DeviceData* deviceStructs) {
    int imageId = blockIdx.x;
    int threadId = threadIdx.x;
    int blockSize = blockDim.x;

    decodeImage(deviceStructs[imageId].imageData,
                deviceStructs[imageId].imageDataLength,
                deviceStructs[imageId].yCrCbChannels,
                deviceStructs[imageId].rgbChannels,
                deviceStructs[imageId].outputChannels,
                deviceStructs[imageId].width,
                deviceStructs[imageId].height,
                deviceStructs[imageId].quantTables,
                deviceStructs[imageId].hfCodes,
                deviceStructs[imageId].hfLengths,
                deviceStructs[imageId].zigzagLocations,
                deviceStructs[imageId].sInfo, // Use sInfo from DeviceData
                threadId,
                blockSize);
}

void clean(uint16_t*& hfCodes, int*& hfLengths, uint8_t*& quantTables, int16_t*& yCrCbChannels, int16_t*& rgbChannels, int16_t*& outputChannels, int*& zigzagLocations, uint8_t*& imageData, std::unordered_map<int,HuffmanTree*>& huffmanTrees, int*& sInfo) {
    // Freeing the memory
    cudaFree(hfCodes);
    cudaFree(hfLengths);
    cudaFree(quantTables);
    cudaFree(imageData);
    cudaFree(zigzagLocations);
    cudaFree(yCrCbChannels);
    cudaFree(rgbChannels);
    cudaFree(outputChannels);
    cudaFree(sInfo);
    for(auto& [key, item]: huffmanTrees) {
        if (item != NULL) {
            delete item;
        }
    }
}

void write(int16_t* outputChannels, int width, int height, std::string filename) {
    ImageChannels channels(height * width);
    size_t channelSize = width * height * sizeof(int16_t);
    cudaMemcpy(channels.getR().data(), outputChannels, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(channels.getG().data(), outputChannels+width*height, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(channels.getB().data(), outputChannels+2*width*height, channelSize, cudaMemcpyDeviceToHost);

    // Writing the decoded channels to a file instead of displaying using opencv
    fs::path output_dir = "../testing/cudaO_output_arrays";
    // fs::path output_dir = "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/testing/bench"; // Change the directory name here for future CUDA implementations
    fs::path full_path = output_dir / filename;
    full_path.replace_extension(".array");
    std::ofstream outfile(full_path);
    outfile << height << " " << width << std::endl;
    std::copy(channels.getR().begin(), channels.getR().end(), std::ostream_iterator<int>(outfile, " "));
    outfile << std::endl;
    std::copy(channels.getG().begin(), channels.getG().end(), std::ostream_iterator<int>(outfile, " "));
    outfile << std::endl;
    std::copy(channels.getB().begin(), channels.getB().end(), std::ostream_iterator<int>(outfile, " "));
    outfile.close();
}