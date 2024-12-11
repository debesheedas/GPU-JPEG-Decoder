#include "parser.h"

__constant__ int initialZigzag[64]; 

__device__ int clip(int value) {
    if (value < -256) return -256;
    if (value > 255) return 255;
    return value;
}

__device__ void idctRow(int* block) {
    int x0, x1, x2, x3, x4, x5, x6, x7;

    // Shortcut: if all AC terms are zero, directly scale the DC term
    if (!((x1 = block[4]<<11) | (x2 = block[6]) | (x3 = block[2]) | (x4 = block[1]) | (x5 = block[7]) | (x6 = block[5]) | (x7 = block[3]))) {
        block[0] = block[1] = block[2] = block[3] = block[4] = block[5] = block[6] = block[7] = block[0]<<3;
        return;
    }
    // Scale the DC coefficient
    x0 = (block[0]<<11) + 128;

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

    block[0] = (x7 + x1) >> 8;
    block[1] = (x3 + x2) >> 8;
    block[2] = (x0 + x4) >> 8;
    block[3] = (x8 + x6) >> 8;
    block[4] = (x8 - x6) >> 8;
    block[5] = (x0 - x4) >> 8;
    block[6] = (x3 - x2) >> 8;
    block[7] = (x7 - x1) >> 8;
}

__device__ void idctCol(int* block) {
    int x0, x1, x2, x3, x4, x5, x6, x7;

    // Shortcut: if all AC terms are zero, directly scale the DC term
    if (!((x1 = (block[8*4]<<8)) | (x2 = block[8*6]) | (x3 = block[8*2]) | (x4 = block[8*1]) | (x5 = block[8*7]) | (x6 = block[8*5]) | (x7 = block[8*3]))) {
        block[8*0] = block[8*1] = block[8*2] = block[8*3] = block[8*4] = block[8*5] = block[8*6] = block[8*7] = clip((block[8*0]+32)>>6);
        return;
    }
    // Scale the DC coefficient
    x0 = (block[8*0]<<8) + 8192;

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

    block[8 * 0] = clip((x7 + x1) >> 14);
    block[8 * 1] = clip((x3 + x2) >> 14);
    block[8 * 2] = clip((x0 + x4) >> 14);
    block[8 * 3] = clip((x8 + x6) >> 14);
    block[8 * 4] = clip((x8 - x6) >> 14);
    block[8 * 5] = clip((x0 - x4) >> 14);
    block[8 * 6] = clip((x3 - x2) >> 14);
    block[8 * 7] = clip((x7 - x1) >> 14);
}

__global__ void initializeIDCTTableKernel(double *dIdctTable, int numThreads)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (id < numThreads) {
        double normCoeff = ((id / 8) == 0) ? (1.0 / sqrt(2.0)) : 1.0;
        dIdctTable[id] = normCoeff * cos(((2.0 * (id%8) + 1.0) * (id/8) * M_PI) / 16.0);
    }
}

JPEGParser::JPEGParser(std::string& imagePath) {
    // Extract the file name of the image file from the file path
    fs::path file_path(imagePath);
    this->filename = file_path.filename().string();
    std::ifstream input(imagePath, std::ios::binary);
    
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()));
    this->readBytes = new uint8_t[bytes.size()];
    for (int i = 0; i < bytes.size(); i++) {
        readBytes[i] = bytes[i];
    }
    input.close();

    imageDataLength = 0;

    int zigzagEntries[64] = {
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63
    };

    cudaMalloc((void**)&zigzag, 64 * sizeof(int));
    cudaMemcpyToSymbol(initialZigzag, zigzagEntries, sizeof(int) * 64);

    int blockSize = 64;
    int gridSize = (64 + blockSize - 1) / blockSize;
    cudaMalloc((void**)&idctTable, 64 * sizeof(double));
    initializeIDCTTableKernel<<<blockSize, gridSize>>>(idctTable, 64);
}

/* Function to allocate the GPU space. */
void JPEGParser::move() {
    cudaMalloc((uint16_t**)&this->hf0codes, 256 * sizeof(uint16_t));
    cudaMalloc((uint16_t**)&this->hf1codes, 256 * sizeof(uint16_t));
    cudaMalloc((uint16_t**)&this->hf16codes, 256 * sizeof(uint16_t));
    cudaMalloc((uint16_t**)&this->hf17codes, 256 * sizeof(uint16_t));

    cudaMalloc((int**)&this->hf0lengths, 256 * sizeof(int));
    cudaMalloc((int**)&this->hf1lengths, 256 * sizeof(int));
    cudaMalloc((int**)&this->hf16lengths, 256 * sizeof(int));
    cudaMalloc((int**)&this->hf17lengths, 256 * sizeof(int));

    cudaMemcpy(this->hf0codes, this->huffmanTrees[0]->codes, 256 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(this->hf1codes, this->huffmanTrees[1]->codes, 256 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(this->hf16codes, this->huffmanTrees[16]->codes, 256 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(this->hf17codes, this->huffmanTrees[17]->codes, 256 * sizeof(uint16_t), cudaMemcpyHostToDevice);

    cudaMemcpy(this->hf0lengths, this->huffmanTrees[0]->codeLengths, 256 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->hf1lengths, this->huffmanTrees[1]->codeLengths, 256 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->hf16lengths, this->huffmanTrees[16]->codeLengths, 256 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->hf17lengths, this->huffmanTrees[17]->codeLengths, 256 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Allocating the channels in the GPU memory.
    cudaMalloc((void**)&this->luminous, 64 * xBlocks * yBlocks * sizeof(int));
    cudaMalloc((void**)&this->chromRed, 64 * xBlocks * yBlocks * sizeof(int));
    cudaMalloc((void**)&this->chromYel, 64 * xBlocks * yBlocks * sizeof(int));

    cudaMalloc((void**)&this->zigzag_l, 64 * xBlocks * yBlocks * sizeof(int));
    cudaMalloc((void**)&this->zigzag_r, 64 * xBlocks * yBlocks * sizeof(int));
    cudaMalloc((void**)&this->zigzag_y, 64 * xBlocks * yBlocks * sizeof(int));

    cudaMalloc((void**)&this->redOutput, 64 * xBlocks * yBlocks * sizeof(int));
    cudaMalloc((void**)&this->greenOutput, 64 * xBlocks * yBlocks * sizeof(int));
    cudaMalloc((void**)&this->blueOutput, 64 * xBlocks * yBlocks * sizeof(int));
}

void JPEGParser::extract() {        
    uint16_t tableSize = 0;
    uint8_t header = 0;

    // Using the Stream class for reading bytes.
    Stream* stream = new Stream(this->readBytes);
    while (true) {
        uint16_t marker = stream->getMarker();

        if (marker == MARKERS[0]) {
            continue;
        } else if (marker == MARKERS[1]) {
            tableSize = stream->getMarker();
            this->applicationHeader = new uint8_t[(int) tableSize - 2];
            stream->getNBytes(this->applicationHeader, int(tableSize - 2));
        } else if (marker == MARKERS[2]) {
            stream->getMarker();
            uint8_t destination = stream->getByte();
            uint8_t* host_quantTable1 = new uint8_t[64];
            stream->getNBytes(host_quantTable1, 64);
            cudaMalloc((void**)&this->quantTable1, 64 * sizeof(uint8_t));
            cudaMemcpy(this->quantTable1, host_quantTable1, 64 * sizeof(uint8_t), cudaMemcpyHostToDevice);

            if(stream->getMarker() == MARKERS[2]) {
                stream->getMarker();
                destination = stream->getByte();
                this->quantTable2 = new uint8_t[64];
                uint8_t* host_quantTable2 = new uint8_t[64];
                stream->getNBytes(host_quantTable2, 64);
                cudaMalloc((void**)&this->quantTable2, 64 * sizeof(uint8_t));
                cudaMemcpy(this->quantTable2, host_quantTable2, 64 * sizeof(uint8_t), cudaMemcpyHostToDevice);
                
            } else {
                std::cout << " Something went wrong at parsing second quant table." << std::endl;
            }
        } else if (marker == MARKERS[3]) {
            tableSize = stream->getMarker();
            this->startOfFrame = new uint8_t[(int) tableSize - 2];
            stream->getNBytes(this->startOfFrame, (int) tableSize - 2);
            Stream* frame = new Stream(this->startOfFrame);
            int precision = frame->getByte();
            this->height = frame->getMarker();
            this->width = frame->getMarker();
            this->paddedWidth = ((this->width + 7) / 8) * 8;
            this->paddedHeight = ((this->height + 7) / 8) * 8;
            this->xBlocks = this->paddedWidth / 8;
            this->yBlocks = this->paddedHeight / 8;
            delete frame;

        } else if (marker == MARKERS[4]) {
            tableSize = stream->getMarker();
            header = stream->getByte();
            this->huffmanTable1 = new uint8_t[(int) tableSize - 3];
            stream->getNBytes(this->huffmanTable1, (int) tableSize - 3);
            this->huffmanTrees[header] = new HuffmanTree(this->huffmanTable1);

            if (stream->getMarker() ==  MARKERS[4]) {
                tableSize = stream->getMarker();
                header = stream->getByte();
                this->huffmanTable2 = new uint8_t[(int) tableSize - 3];
                stream->getNBytes(this->huffmanTable2, (int) tableSize - 3);
                this->huffmanTrees[header] = new HuffmanTree(this->huffmanTable2);
            }

            if (stream->getMarker() ==  MARKERS[4]) {
                tableSize = stream->getMarker();
                header = stream->getByte();
                this->huffmanTable3 = new uint8_t[(int) tableSize - 3];
                stream->getNBytes(this->huffmanTable3, (int) tableSize - 3);
                this->huffmanTrees[header] = new HuffmanTree(this->huffmanTable3);
            }

            if (stream->getMarker() ==  MARKERS[4]) {
                tableSize = stream->getMarker();
                header = stream->getByte();
                this->huffmanTable4 = new uint8_t[(int) tableSize - 3];
                stream->getNBytes(this->huffmanTable4, (int) tableSize - 3);
                this->huffmanTrees[header] = new HuffmanTree(this->huffmanTable4);
            }

        } else if (marker == MARKERS[5]) {
            tableSize = stream->getMarker();
            this->startOfScan = new uint8_t[(int) tableSize - 2];
            stream->getNBytes(this->startOfScan, (int) tableSize - 2);
            uint8_t curByte, prevByte = 0x00;
            size_t size = 5 * 1024 * 1024;
            uint8_t* host_imageData = new uint8_t[size];
            
            while (true) {
                curByte = stream->getByte();
                if ((prevByte == 0xff) && (curByte == 0xd9))
                    break;
                if (curByte == 0x00) {
                    if (prevByte != 0xff) {
                        host_imageData[imageDataLength++] = curByte;
                    }
                } else {
                    host_imageData[imageDataLength++] = curByte;
                }
                prevByte = curByte;
            }
            
            imageDataLength--; // We remove the ending byte because it is extra 0xff.
            cudaMalloc((void**)&this->imageData, imageDataLength * sizeof(uint8_t));
            cudaMemcpy(this->imageData, host_imageData, imageDataLength * sizeof(uint8_t), cudaMemcpyHostToDevice);
            break;
        }
    } 
    delete stream;  
    move(); 
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

__device__ int buildMCU(int* outBuffer, uint8_t* imageData, int bitOffset, uint8_t* quant, 
                        int& oldCoeff, uint16_t* dcHfcodes, int* dcHflengths, uint16_t* acHfcodes, int* acHflengths) {

    int code = 0;
    int code_length = 0;
    match_huffman_code(imageData, bitOffset, dcHfcodes, dcHflengths, code, code_length);
    bitOffset += code_length;
    uint16_t bits = getNBits(imageData, bitOffset, code);

    int decoded = decodeNumber(code, bits); 
    int dcCoeff = decoded + oldCoeff;

    outBuffer[0] = dcCoeff;

    int length = 1;
    while (length < 64) {
        match_huffman_code(imageData, bitOffset, acHfcodes, acHflengths, code, code_length);
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
            outBuffer[length] = decoded;
            length++;
        }
    }
    // Update oldCoeff for the next MCU
    oldCoeff = dcCoeff;
    return bitOffset;
}

JPEGParser::~JPEGParser() {
    cudaFree(idctTable);
    // delete channels;
    for (auto& tree : huffmanTrees) {
        delete tree.second;
    }
    cudaFree(this->quantTable1);
    cudaFree(this->quantTable2);
    cudaFree(this->imageData);
    delete[]  this->readBytes;
    delete[]  this->applicationHeader;
    delete[]  this->startOfFrame;
    delete[]  this->startOfScan;
    delete[]  this->huffmanTable1;
    delete[]  this->huffmanTable2;
    delete[]  this->huffmanTable3;
    delete[]  this->huffmanTable4;
}

__device__ void performHuffmanDecoding(uint8_t* imageData, int* arr_l, int* arr_r, int* arr_y,
                                       uint8_t* quant1, uint8_t* quant2,
                                       uint16_t* hf0codes, int* hf0lengths, uint16_t* hf16codes, int* hf16lengths,
                                       uint16_t* hf1codes, int* hf1lengths, uint16_t* hf17codes, int* hf17lengths,
                                       int yBlocks, int xBlocks) {
    int* curLuminous = arr_l;
    int* curChromRed = arr_r;
    int* curChromYel = arr_y;
    int oldLumCoeff = 0, oldCbdCoeff = 0, oldCrdCoeff = 0;
    int bitOffset = 0;

    for (int y = 0; y < yBlocks; y++) {
        for (int x = 0; x < xBlocks; x++) {
            bitOffset = buildMCU(curLuminous, imageData, bitOffset, quant1, oldLumCoeff, hf0codes, hf0lengths, hf16codes, hf16lengths);
            bitOffset = buildMCU(curChromRed, imageData, bitOffset, quant2, oldCbdCoeff, hf1codes, hf1lengths, hf17codes, hf17lengths);
            bitOffset = buildMCU(curChromYel, imageData, bitOffset, quant2, oldCrdCoeff, hf1codes, hf1lengths, hf17codes, hf17lengths);
            curLuminous += 64;
            curChromRed += 64;
            curChromYel += 64;
        }
    }
}

__device__ void performZigzagReordering(int* arr_l, int* arr_r, int* arr_y, 
                                        int* zigzag_l, int* zigzag_r, int* zigzag_y,
                                        int blockIndex, int threadIndexInBlock, int threadId,
                                        const int* initialZigzag, int pixelIndex, uint8_t* quant1, uint8_t* quant2) {
    zigzag_l[pixelIndex] = arr_l[blockIndex * 64 + initialZigzag[threadIndexInBlock]] * quant1[initialZigzag[threadIndexInBlock]];
    zigzag_r[pixelIndex] = arr_r[blockIndex * 64 + initialZigzag[threadIndexInBlock]] * quant2[initialZigzag[threadIndexInBlock]];
    zigzag_y[pixelIndex] = arr_y[blockIndex * 64 + initialZigzag[threadIndexInBlock]] * quant2[initialZigzag[threadIndexInBlock]];
}

__device__ void performColorConversion(int* arr_l, int* arr_r, int* arr_y,
                                       int* redOutput, int* greenOutput, int* blueOutput,
                                       int totalPixels, int width, int threadId, int blockDimGridDim) {
    for (int i = threadId; i < totalPixels; i += blockDimGridDim) {
        int blockId = i / 64;
        int blockRow = blockId / (width / 8);
        int blockColumn = blockId % (width / 8);

        int rowStart = blockRow * 8;
        int columnStart = blockColumn * 8;

        int pixelIndexInBlock = i % 64;
        int rowInBlock = pixelIndexInBlock / 8;
        int columnInBlock = pixelIndexInBlock % 8;

        int globalRow = rowStart + rowInBlock;
        int globalColumn = columnStart + columnInBlock;

        int actualIndex = globalRow * width + globalColumn;

        // Retrieve pixel data and perform the color conversion
        float red = arr_y[i] * (2 - 2 * 0.299) + arr_l[i];
        float blue = arr_r[i] * (2 - 2 * 0.114) + arr_l[i];
        float green = (arr_l[i] - 0.114 * blue - 0.299 * red) / 0.587;

        // Clamp values to [0, 255]
        redOutput[actualIndex] = min(max(static_cast<int>(red + 128), 0), 255);
        greenOutput[actualIndex] = min(max(static_cast<int>(green + 128), 0), 255);
        blueOutput[actualIndex] = min(max(static_cast<int>(blue + 128), 0), 255);
    }
}

__device__ void performIDCT(const int* zigzag, double* idctTable, int threadCol, int threadRow, double& localSum, int threshold) {
    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            localSum += zigzag[threshold + v * 8 + u] * idctTable[u * 8 + threadCol] * idctTable[v * 8 + threadRow];
        }
    }
}

__global__ void decodeKernel(uint8_t* imageData, int* arr_l, int* arr_r, int* arr_y, int* zigzag_l, int* zigzag_r, 
                                int* zigzag_y, double* idctTable, int validHeight, 
                                int validWidth, int width, int height, int xBlocks, int yBlocks, int* redOutput, 
                                int* greenOutput, int* blueOutput, uint8_t* quant1, uint8_t* quant2, 
                                uint16_t* hf0codes, uint16_t* hf1codes, uint16_t* hf16codes, uint16_t* hf17codes,
                                int* hf0lengths, int* hf1lengths, int* hf16lengths, int* hf17lengths) {

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("blockdim.x %d\n", blockDim.x);
    // printf("blockidx.x %d\n", blockIdx.x);
    // printf("threadidx.x %d\n", threadIdx.x);

    int pixelIndex = threadId;
    int totalPixels = width * height;

    if (threadId==0) {
        performHuffmanDecoding(imageData, arr_l, arr_r, arr_y, quant1, quant2, 
                               hf0codes, hf0lengths, hf16codes, hf16lengths, 
                               hf1codes, hf1lengths, hf17codes, hf17lengths, yBlocks, xBlocks);
    }
    __syncthreads();

    while (pixelIndex < totalPixels) {
        int threadIndexInBlock = pixelIndex % 64;
        int blockIndex = pixelIndex / 64;

        performZigzagReordering(arr_l, arr_r, arr_y, zigzag_l, zigzag_r, zigzag_y,
                                blockIndex, threadIndexInBlock, threadId, initialZigzag, pixelIndex, quant1, quant2);

        pixelIndex += blockDim.x * gridDim.x;
    }

    __syncthreads();

    pixelIndex = threadId;

    while (pixelIndex * 8 < totalPixels) {
        // int threadIndexInBlock = pixelIndex % 64;
        // int blockIndex = pixelIndex / 64;
        
        idctRow(zigzag_l + pixelIndex * 8);
        idctRow(zigzag_r + pixelIndex * 8);
        idctRow(zigzag_y + pixelIndex * 8);

        pixelIndex += blockDim.x * gridDim.x;
    }

    __syncthreads();

    pixelIndex = threadId;

     while (pixelIndex * 8 < totalPixels) {
        // int threadIndexInBlock = pixelIndex % 64;
        // int blockIndex = pixelIndex / 64;


        int start = pixelIndex / 8;
        start = start * 64;
        start = start + (pixelIndex % 8);
        
        idctCol(zigzag_l + start);
        idctCol(zigzag_r + start);
        idctCol(zigzag_y + start);

        pixelIndex += blockDim.x * gridDim.x;
    }
    __syncthreads();

    // Iterate over pixels handled by this thread
    performColorConversion(zigzag_l, zigzag_r, zigzag_y, redOutput, greenOutput, blueOutput, 
                           totalPixels, width, threadId, blockDim.x * gridDim.x);
}



__device__ void decodeImage(uint8_t* imageData, int* arr_l, int* arr_r, int* arr_y, int* zigzag_l, int* zigzag_r, 
                                int* zigzag_y, double* idctTable, int validHeight, 
                                int validWidth, int width, int height, int xBlocks, int yBlocks, int* redOutput, 
                                int* greenOutput, int* blueOutput, uint8_t* quant1, uint8_t* quant2, 
                                uint16_t* hf0codes, uint16_t* hf1codes, uint16_t* hf16codes, uint16_t* hf17codes,
                                int* hf0lengths, int* hf1lengths, int* hf16lengths, int* hf17lengths, int threadId, int blockSize) {

    // int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    //int threadId = threadIdx.x;
    // printf("blockdim.x %d\n", blockDim.x);
    // printf("blockidx.x %d\n", blockIdx.x);
    // printf("threadidx.x %d\n", threadIdx.x);

    int pixelIndex = threadId;
    int totalPixels = width * height;

    if (threadId==0) {
        performHuffmanDecoding(imageData, arr_l, arr_r, arr_y, quant1, quant2, 
                               hf0codes, hf0lengths, hf16codes, hf16lengths, 
                               hf1codes, hf1lengths, hf17codes, hf17lengths, yBlocks, xBlocks);
    }
    __syncthreads();

    while (pixelIndex < totalPixels) {
        int threadIndexInBlock = pixelIndex % 64;
        int blockIndex = pixelIndex / 64;

        performZigzagReordering(arr_l, arr_r, arr_y, zigzag_l, zigzag_r, zigzag_y,
                                blockIndex, threadIndexInBlock, threadId, initialZigzag, pixelIndex, quant1, quant2);

        pixelIndex += blockSize;
    }

    __syncthreads();

    pixelIndex = threadId;

    while (pixelIndex * 8 < totalPixels) {
        // int threadIndexInBlock = pixelIndex % 64;
        // int blockIndex = pixelIndex / 64;
        
        idctRow(zigzag_l + pixelIndex * 8);
        idctRow(zigzag_r + pixelIndex * 8);
        idctRow(zigzag_y + pixelIndex * 8);

        pixelIndex += blockSize;
    }

    __syncthreads();

    pixelIndex = threadId;

     while (pixelIndex * 8 < totalPixels) {
        // int threadIndexInBlock = pixelIndex % 64;
        // int blockIndex = pixelIndex / 64;


        int start = pixelIndex / 8;
        start = start * 64;
        start = start + (pixelIndex % 8);
        
        idctCol(zigzag_l + start);
        idctCol(zigzag_r + start);
        idctCol(zigzag_y + start);

        pixelIndex += blockSize;
    }
    __syncthreads();

    // Iterate over pixels handled by this thread
    performColorConversion(zigzag_l, zigzag_r, zigzag_y, redOutput, greenOutput, blueOutput, 
                           totalPixels, width, threadId, blockSize);
}

__global__ void batchDecodeKernel(JPEGParserData* deviceStructs) {
    // int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int imageId = blockIdx.x;
    int threadId = threadIdx.x;
    int blockSize = blockDim.x;
    decodeImage(deviceStructs[imageId].imageData, 
                deviceStructs[imageId].luminous, 
                deviceStructs[imageId].chromRed, 
                deviceStructs[imageId].chromYel, 
                deviceStructs[imageId].zigzag_l,
                deviceStructs[imageId].zigzag_r,
                deviceStructs[imageId].zigzag_y, 
                deviceStructs[imageId].idctTable, 
                8, 8,  
                deviceStructs[imageId].width, 
                deviceStructs[imageId].height, 
                deviceStructs[imageId].xBlocks, 
                deviceStructs[imageId].yBlocks, 
                deviceStructs[imageId].redOutput, 
                deviceStructs[imageId].greenOutput, 
                deviceStructs[imageId].blueOutput,
                deviceStructs[imageId].quantTable1, 
                deviceStructs[imageId].quantTable2, 
                deviceStructs[imageId].hf0codes, 
                deviceStructs[imageId].hf1codes, 
                deviceStructs[imageId].hf16codes, 
                deviceStructs[imageId].hf17codes, 
                deviceStructs[imageId].hf0lengths, 
                deviceStructs[imageId].hf1lengths, 
                deviceStructs[imageId].hf16lengths, 
                deviceStructs[imageId].hf17lengths,
                threadId, blockSize);
}

void JPEGParser::decode() {
    decodeKernel<<<1, 1024>>>(this->imageData, this->luminous, this->chromRed, this->chromYel, this->zigzag_l, this->zigzag_r, this->zigzag_y, this->idctTable, 8, 8,  
                                            this->width, this->height, this->xBlocks, this->yBlocks, this->redOutput, this->greenOutput, this->blueOutput,
                                            this->quantTable1, this->quantTable2, this->hf0codes, this->hf1codes, this->hf16codes, this->hf17codes, 
                                            this->hf0lengths, this->hf1lengths, this->hf16lengths, this->hf17lengths);


    if (luminous) cudaFree(luminous);
    if (chromRed) cudaFree(chromRed);
    if (chromYel) cudaFree(chromYel);
    if (hf0codes) cudaFree(hf0codes);
    if (hf1codes) cudaFree(hf1codes);
    if (hf16codes) cudaFree(hf16codes);
    if (hf17codes) cudaFree(hf17codes);
    if (hf0lengths) cudaFree(hf0lengths);
    if (hf1lengths) cudaFree(hf1lengths);
    if (hf16lengths) cudaFree(hf16lengths);
    if (hf17lengths) cudaFree(hf17lengths); 
}

void JPEGParser::write() {
    this->channels = new ImageChannels(this->height * this->width);
    size_t channelSize = this->width * this->height * sizeof(int);
    cudaMemcpy(channels->getR().data(), this->redOutput, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(channels->getG().data(), this->greenOutput, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(channels->getB().data(), this->blueOutput, channelSize, cudaMemcpyDeviceToHost);
    cudaFree(redOutput);
    cudaFree(greenOutput);
    cudaFree(blueOutput);

    // Writing the decoded channels to a file instead of displaying using opencv
    fs::path output_dir = "../testing/cudaO_output_arrays";
    // fs::path output_dir = "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/testing/bench"; // Change the directory name here for future CUDA implementations
    fs::path full_path = output_dir / this->filename;
    full_path.replace_extension(".array");
    std::ofstream outfile(full_path);
    outfile << this->height << " " << this->width << std::endl;
    std::copy(this->channels->getR().begin(), this->channels->getR().end(), std::ostream_iterator<int>(outfile, " "));
    outfile << std::endl;
    std::copy(this->channels->getG().begin(), this->channels->getG().end(), std::ostream_iterator<int>(outfile, " "));
    outfile << std::endl;
    std::copy(this->channels->getB().begin(), this->channels->getB().end(), std::ostream_iterator<int>(outfile, " "));
    outfile.close();
}