#include "parser.h"

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

__device__ void idctCol(int16_t* block, bool print) {
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

    if (print) {
        printf("%d %d %d %d %d %d %d %d \n", x0, x1, x2, x3, x4, x5, x6, x7);
    }

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
void allocate(uint16_t*& hfCodes, int*& hfLengths, std::unordered_map<int,HuffmanTree*>& huffmanTrees, int16_t*& yCrCbChannels, int16_t*& rgbChannels, int16_t*& outputChannels, int width, int height, int*& zigzagLocations) {

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
   
}

void extract(std::string imagePath, uint8_t*& quantTables, uint8_t*& imageData, int& width, int& height, std::unordered_map<int,HuffmanTree*>& huffmanTrees) {  

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
}

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
    // int16_t* curLuminous = yCrCbChannels;
    // int16_t* curChromRed = yCrCbChannels + width * height;
    // int16_t* curChromYel = yCrCbChannels + 2 * width * height;
    int oldLumCoeff = 0, oldCbdCoeff = 0, oldCrdCoeff = 0;
    int bitOffset = 0;

    int xBlocks = width / 8;
    int yBlocks = height / 8;

    for (int y = 0; y < yBlocks; y++) {
        for (int x = 0; x < xBlocks; x++) {
            bitOffset = buildMCU(yCrCbChannels, imageData, bitOffset, oldLumCoeff, hfCodes, hfLengths, hfCodes + 512, hfLengths+ 512);
            yCrCbChannels = yCrCbChannels + 64;
            bitOffset = buildMCU(yCrCbChannels, imageData, bitOffset, oldCbdCoeff, hfCodes + 256, hfLengths + 256, hfCodes + 768, hfLengths + 768);
            yCrCbChannels = yCrCbChannels + 64;
            bitOffset = buildMCU(yCrCbChannels, imageData, bitOffset, oldCrdCoeff, hfCodes + 256, hfLengths + 256, hfCodes + 768, hfLengths + 768);
            yCrCbChannels = yCrCbChannels + 64;
        }
    }
}

// __device__ void performHuffmanDecoding(uint8_t* imageData, int16_t* yCrCbChannels, uint16_t* hfCodes, int* hfLengths, int width, int height, int& oldLumCoeff, int& oldCbdCoeff, int& oldCrdCoeff, int bitoffset) {
//     bitOffset = buildMCU(yCrCbChannels, imageData, bitOffset, oldLumCoeff, hfCodes, hfLengths, hfCodes + 512, hfLengths+ 512);
//     yCrCbChannels = yCrCbChannels + 64;
//     bitOffset = buildMCU(yCrCbChannels, imageData, bitOffset, oldCbdCoeff, hfCodes + 256, hfLengths + 256, hfCodes + 768, hfLengths + 768);
//     yCrCbChannels = yCrCbChannels + 64;
//     bitOffset = buildMCU(yCrCbChannels, imageData, bitOffset, oldCrdCoeff, hfCodes + 256, hfLengths + 256, hfCodes + 768, hfLengths + 768);
//     yCrCbChannels = yCrCbChannels + 64;
// }

// __device__ void performZigzagReordering(int16_t* yCrCbChannels, int16_t* rgbChannels, uint8_t* quantTables,
//                                         int blockIndex, int withinBlock, int channelId, int inBlockIndex, int pixelIndex, int* zigzagLocations) {

//     rgbChannels[pixelIndex] = yCrCbChannels[channelId * 64+blockIndex * 192 + zigzagLocations
//     [inBlockIndex]] * quantTables[(64 & -(channelId > 0)) + zigzagLocations[inBlockIndex]];
// }

__device__ void performZigzagReordering(int16_t* yCrCbChannels, int16_t* rgbChannels, uint8_t* quantTables,
                                       int channelId, int location, int threadId, int* zigzagLocations) {

    rgbChannels[location] = yCrCbChannels[channelId * 64 + zigzagLocations
    [threadId]] * quantTables[(64 & -(channelId > 0)) + zigzagLocations[threadId]];
}

// __device__ void performColorConversion(int16_t* rgbChannels, int16_t* outputChannels,
//                                        int totalPixels, int width, int threadId, int blockDimGridDim) {
//     for (int i = threadId; i < totalPixels; i += blockDimGridDim) {
//         int blockId = i / 64;
//         int blockRow = blockId / (width / 8);
//         int blockColumn = blockId % (width / 8);

//         int rowStart = blockRow * 8;
//         int columnStart = blockColumn * 8;

//         int pixelIndexInBlock = i % 64;
//         int rowInBlock = pixelIndexInBlock / 8;
//         int columnInBlock = pixelIndexInBlock % 8;

//         int globalRow = rowStart + rowInBlock;
//         int globalColumn = columnStart + columnInBlock;

//         int actualIndex = globalRow * width + globalColumn;

//         // Retrieve pixel data and perform the color conversion
//         int ij = i / 64;
//         ij = ij * 192;
//         ij = ij + (i%64);

//         float red = rgbChannels[2*64+ij] * (2 - 2 * 0.299) + rgbChannels[ij];
//         float blue = rgbChannels[64 + ij] * (2 - 2 * 0.114) + rgbChannels[ij];
//         float green = (rgbChannels[ij] - 0.114 * blue - 0.299 * red) / 0.587;

//         // Clamp values to [0, 255]
//         outputChannels[3*actualIndex] = min(max(static_cast<int16_t>(red + 128), 0), 255);
//         outputChannels[3*actualIndex+1] = min(max(static_cast<int16_t>(green + 128), 0), 255);
//         outputChannels[3*actualIndex+2] = min(max(static_cast<int16_t>(blue + 128), 0), 255);
//     }
// }

__device__ void performColorConversion(int16_t* rgbChannels, int16_t* outputChannels,
                                      int width, int threadId, int blockId) {
    int blockRow = blockId / (width / 8);
    int blockColumn = blockId % (width / 8);

    int rowStart = blockRow * 8;
    int columnStart = blockColumn * 8;

    int pixelIndexInBlock = threadId;
    int rowInBlock = pixelIndexInBlock / 8;
    int columnInBlock = pixelIndexInBlock % 8;

    int globalRow = rowStart + rowInBlock;
    int globalColumn = columnStart + columnInBlock;

    int actualIndex = globalRow * width + globalColumn;


    float red = rgbChannels[128+threadId] * (2 - 2 * 0.299) + rgbChannels[threadId];
    float blue = rgbChannels[64 + threadId] * (2 - 2 * 0.114) + rgbChannels[threadId];
    float green = (rgbChannels[threadId] - 0.114 * blue - 0.299 * red) / 0.587;

    // Clamp values to [0, 255]
    outputChannels[3*actualIndex] = min(max(static_cast<int16_t>(red + 128), 0), 255);
    outputChannels[3*actualIndex+1] = min(max(static_cast<int16_t>(green + 128), 0), 255);
    outputChannels[3*actualIndex+2] = min(max(static_cast<int16_t>(blue + 128), 0), 255);
}

__global__ void decodeKernel(uint8_t* imageData, int16_t* yCrCbChannels, int16_t* rgbChannels, int16_t* outputChannels, int width, int height, uint8_t* quantTables, uint16_t* hfCodes, int* hfLengths, int* zigzagLocations) {
    // int imageId = blockIdx.x;
    int threadId = threadIdx.x;
    int blockSize = blockDim.x;
    decodeImage(imageData,
                yCrCbChannels,
                rgbChannels,
                outputChannels,
                width,
                height,
                quantTables,
                hfCodes,
                hfLengths,
                zigzagLocations,
                threadId, blockSize);
}

// __device__ void decodeImage(uint8_t* imageData, int16_t* yCrCbChannels, int16_t* rgbChannels, int16_t* outputChannels, int width, int height, uint8_t* quantTables, uint16_t* hfCodes, int* hfLengths, int* zigzagLocations, int threadId, int blockSize) {

//     __shared__ int zigzagMap[64];

//     int pixelIndex = threadId;
//     while (pixelIndex < 64) {
//         zigzagMap[pixelIndex] = zigzagLocations[pixelIndex];
//         pixelIndex += blockSize;
//     }


//     __syncthreads();
//     int totalPixels = width * height;
//     pixelIndex = threadId;
//     if (threadId==0) {
//         performHuffmanDecoding(imageData, yCrCbChannels, hfCodes, hfLengths, width, height);
//     }
//     __syncthreads();

//     pixelIndex = threadId;
//     while (pixelIndex < 3 * totalPixels) {
//         int block = pixelIndex / 192;
//         int withinBlock = pixelIndex % 192;
//         int channel = withinBlock / 64;
//         int inBlockIndex = withinBlock % 64;


//         performZigzagReordering(yCrCbChannels, rgbChannels, quantTables,
//                                 block, withinBlock, channel, inBlockIndex, pixelIndex, zigzagMap);

//         // int channel = pixelIndex / totalPixels;
//         // int index = pixelIndex % totalPixels;
//         // int threadIndexInBlock = index % 64;
//         // int blockIndex = index / 64;

//         // performZigzagReordering(yCrCbChannels, rgbChannels, quantTables,
//         //                         blockIndex, threadIndexInBlock, threadId, index, totalPixels, channel, zigzagMap);

//         pixelIndex += blockSize;
//     }

//     __syncthreads();

//     pixelIndex = threadId;
//     while (pixelIndex * 8 < 3 * totalPixels) {     
//         int index = pixelIndex % totalPixels;
//         int start = (index / 8) * 64 + (index % 8) * 8;
//         idctRow(rgbChannels + (pixelIndex / totalPixels) * totalPixels + start);
//         start = (index / 8) * 64 + (index % 8);
//         idctCol(rgbChannels + (pixelIndex / totalPixels) * totalPixels + start);
//         pixelIndex += blockSize;
//     }

//     __syncthreads();

//     performColorConversion(rgbChannels, outputChannels, totalPixels, width, threadId, blockSize);
// }

__device__ void decodeImage(uint8_t* imageData, int16_t* yCrCbChannels, int16_t* rgbChannels, int16_t* outputChannels, int width, int height, uint8_t* quantTables, uint16_t* hfCodes, int* hfLengths, int* zigzagLocations, int threadId, int blockSize) {

    __shared__ int zigzagMap[64];
    __shared__ int16_t outputBlocks[256];
    __shared__ int16_t inputBlocks[192];

    int16_t* curStart = inputBlocks;

    int totalPixels = width * height;
    int oldLumCoeff = 0, oldCbdCoeff = 0, oldCrdCoeff = 0;
    int bitOffset = 0;

    int xBlocks = width / 8;
    int yBlocks = height / 8;
    
    zigzagMap[threadId] = zigzagLocations[threadId];
    __syncthreads();

    // int pixelIndex = threadId;

    for (int i = 0; i < xBlocks*yBlocks; i++) {
        if (threadId==0) {
            bitOffset = buildMCU(curStart, imageData, bitOffset, oldLumCoeff, hfCodes, hfLengths, hfCodes + 512, hfLengths+ 512);
            curStart = curStart + 64;
            bitOffset = buildMCU(curStart, imageData, bitOffset, oldCbdCoeff, hfCodes + 256, hfLengths + 256, hfCodes + 768, hfLengths + 768);
            curStart = curStart + 64;
            bitOffset = buildMCU(curStart, imageData, bitOffset, oldCrdCoeff, hfCodes + 256, hfLengths + 256, hfCodes + 768, hfLengths + 768);
            curStart = inputBlocks;
            // if (i == 30) {
            //     printf("%d %d %d \n", inputBlocks[0], inputBlocks[64], inputBlocks[128]);
            // }
            // printf("Thread %d performing serial task for item %d\n", threadId, i);
        }
       __syncthreads();
        // printf("Processing item %d by thread %d\n", i, threadId);
        int pixelIndex = threadId;
        while (pixelIndex < 64) {
            performZigzagReordering(inputBlocks, outputBlocks, quantTables, 0, pixelIndex, pixelIndex, zigzagMap);
            performZigzagReordering(inputBlocks, outputBlocks, quantTables,
                                1, pixelIndex + 64, pixelIndex, zigzagMap);
            performZigzagReordering(inputBlocks, outputBlocks, quantTables,
                                2, pixelIndex + 128, pixelIndex, zigzagMap);
            pixelIndex+= 32;
        }   
        // if (i == 0 && threadId == 0) {
        //     printf("%d %d %d %d %d %d \n", outputBlocks[0], outputBlocks[1], outputBlocks[64], outputBlocks[65], outputBlocks[128], outputBlocks[129]);
        // }
        // (index / 8) * 64 + (index % 8) * 8
        // idctRow(outputBlocks + (threadId * 8));
        idctRow(outputBlocks + (threadId / 8) * 64 + (threadId % 8) * 8);
        if (i == 0 && threadId == 0) {
            printf("%d %d %d %d %d %d \n", outputBlocks[0], outputBlocks[1], outputBlocks[64], outputBlocks[65], outputBlocks[128], outputBlocks[129]);
        }
        __syncthreads();     
        // (index / 8) * 64 + (index % 8);
        idctCol(outputBlocks + ((threadId / 8)*64 + (threadId % 8)), threadId == 0 & i==0);

        //__syncthreads();        
        if (i == 0 && threadId == 0) {
            printf("%d %d %d %d %d %d \n", outputBlocks[0], outputBlocks[1], outputBlocks[64], outputBlocks[65], outputBlocks[128], outputBlocks[129]);
        }
        __syncthreads(); 

        pixelIndex = threadId;
        while (pixelIndex < 64) {
            performColorConversion(outputBlocks, outputChannels, width, pixelIndex, i);       
            pixelIndex += 32; 
        }
        __syncthreads();
    }
}

__global__ void batchDecodeKernel(DeviceData* deviceStructs) {
    // int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int imageId = blockIdx.x;
    int threadId = threadIdx.x;
    int blockSize = blockDim.x;
    decodeImage(deviceStructs[imageId].imageData,
                deviceStructs[imageId].yCrCbChannels,
                deviceStructs[imageId].rgbChannels,
                deviceStructs[imageId].outputChannels,
                deviceStructs[imageId].width,
                deviceStructs[imageId].height,
                deviceStructs[imageId].quantTables,
                deviceStructs[imageId].hfCodes,
                deviceStructs[imageId].hfLengths,
                deviceStructs[imageId].zigzagLocations,
                threadId, blockSize);
}

void clean(uint16_t*& hfCodes, int*& hfLengths, uint8_t*& quantTables, int16_t*& yCrCbChannels, int16_t*& rgbChannels, int16_t*& outputChannels, int*& zigzagLocations, uint8_t*& imageData, std::unordered_map<int,HuffmanTree*>& huffmanTrees) {
    // Freeing the memory
    cudaFree(hfCodes);
    cudaFree(hfLengths);
    cudaFree(quantTables);
    cudaFree(imageData);
    cudaFree(zigzagLocations);
    cudaFree(yCrCbChannels);
    cudaFree(rgbChannels);
    cudaFree(outputChannels);
    for(auto& [key, item]: huffmanTrees) {
        if (item != NULL) {
            delete item;
        }
    }
}

void write(int16_t* outputChannels, int width, int height, std::string filename) {
    ImageChannels channels(height * width);
    size_t channelSize = width * height * sizeof(int16_t);

    std::vector<int16_t> tempChannels(width*height*3);
    cudaMemcpy(tempChannels.data(), outputChannels, channelSize*3, cudaMemcpyDeviceToHost);
    //cudaMemcpy(channels.getG().data(), outputChannels+width*height, channelSize, cudaMemcpyDeviceToHost);
    //cudaMemcpy(channels.getB().data(), outputChannels+2*width*height, channelSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < width*height; i++) {
        channels.getR()[i] = tempChannels[3*i];
        channels.getG()[i] = tempChannels[3*i+1];
        channels.getB()[i] = tempChannels[3*i+2];
    }
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