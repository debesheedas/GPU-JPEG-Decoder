#include "parser.h"

__constant__ int initialZigzag[64]; 

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

void JPEGParser::move() {
    // allocate all GPU space required here.
    //copy whatever data possible to copy at this point  

    // Copy the Huffman Loookup Tables here
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
    // printf("huffman codes\n");
    // for (int idx=0; idx < 20; idx++) {
    //     printf("arr[%d] = %u\n", idx, static_cast<unsigned int>(this->huffmanTrees[0]->codes[idx]));
    // }

    cudaMemcpy(this->hf0lengths, this->huffmanTrees[0]->codeLengths, 256 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->hf1lengths, this->huffmanTrees[1]->codeLengths, 256 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->hf16lengths, this->huffmanTrees[16]->codeLengths, 256 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->hf17lengths, this->huffmanTrees[17]->codeLengths, 256 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Allocating the channels in the GPU memory.
    cudaMalloc((void**)&this->luminous, 64 * xBlocks * yBlocks * sizeof(int));
    cudaMalloc((void**)&this->chromRed, 64 * xBlocks * yBlocks * sizeof(int));
    cudaMalloc((void**)&this->chromYel, 64 * xBlocks * yBlocks * sizeof(int));
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
            // for (int idx=0; idx < 20; idx++) {
            //     printf("arr[%d] = %u\n", idx, static_cast<unsigned int>(host_imageData[idx]));
            // }
            // printf("__________________");
            break;
        }
    } 
    delete stream;  
    move(); 
}

// __device__ uint8_t getCode(uint8_t* imageData, int bitOffset, uint16_t* hfcodes, int* hflengths) {
//     // TODO: Beste
//     // This function takes a pointer to the image data stream on the gpu and its bitoffset and pointers to the two huffman lookup tables on the GPU
//     // It returns the huffman decoded value
// }
__device__ int match_huffman_code(uint8_t* stream, int bit_offset, uint16_t* huff_codes, int* huff_bits, int &code, int &length) {
    // printf("bit_offset %d:\n", bit_offset);
    // for (int idx=0; idx < 20; idx++) {
    //     printf("arr[%d] = %u\n", idx, static_cast<unsigned int>(stream[idx]));
    // }
    // printf("huffman codes\n");
    // for (int idx=0; idx < 20; idx++) {
    //     printf("arr[%d] = %u\n", idx, static_cast<unsigned int>(huff_codes[idx]));
    // }
    // printf("huffman  bits\n");
    // for (int idx=0; idx < 20; idx++) {
    //     printf("arr[%d] = %u\n", idx, static_cast<unsigned int>(huff_bits[idx]));
    // }
    // Extract up to 16 bits from the stream (max Huffman code length)
    unsigned int extracted_bits = getNBits(stream, bit_offset, 16);

    // Compare against Huffman table
    for (int i = 0; i < 256; ++i) {
        if (huff_bits[i] > 0 && huff_bits[i] <= 16) { // Valid bit length
            unsigned int mask = (1 << huff_bits[i]) - 1;
            if ((extracted_bits >> (16 - huff_bits[i]) & mask) == huff_codes[i]) {
                // return i; // Return the index of the matched Huffman code
                code = i;
                length = huff_bits[i];
                return;
            }
        }
    }
    // return -1; // No match found
}

__device__ int buildMCU(int* outBuffer, uint8_t* imageData, int bitOffset, uint8_t* quant, 
                        int& oldCoeff, uint16_t* dcHfcodes, int* dcHflengths, uint16_t* acHfcodes, int* acHflengths) {

    // uint8_t code = getCode(imageData, bitOffset, dcHfcodes, dcHflengths);
    int code = 0;
    int code_length = 0;
    match_huffman_code(imageData, bitOffset, dcHfcodes, dcHflengths, code, code_length);
    bitOffset += code_length;
    printf("dc code %d:\n", code);
    //std::cout << code << " is the code " << std::endl;
    //int code = 200;
    uint16_t bits = getNBits(imageData, bitOffset, code);
    bitOffset += code;

    int decoded = decodeNumber(code, bits); 
    int dcCoeff = decoded + oldCoeff;
    printf("dc coeff %d:\n", dcCoeff);
    // if (quant == 0) {
    //     hostBuffer[0] = dcCoeff * (int) this->quantTable1[0];
    // } else {
    //     hostBuffer[0] = dcCoeff * (int) this->quantTable2[0];
    // }
    outBuffer[0] = dcCoeff * (int) quant[0];

    int length = 1;

    while (length < 64) {
        // code = getCode(imageData, bitOffset, acHfcodes, acHflengths);
        match_huffman_code(imageData, bitOffset, acHfcodes, acHflengths, code, code_length);
        bitOffset += code_length;
        printf("code %d:\n", code);
        if (code == 0) {
            break;
        }

        // The first part of the AC key length is the number of leading zeros
        if (code > 15) {
            length += (code >> 4);
            code = code & 0x0f;
        }

        // bits = imageStream->getNBits(code);
        bits = getNBits(imageData, bitOffset, code);
        bitOffset += code;

        if (length < 64) {
            decoded = decodeNumber(code, bits);
            int val;
            // if (quant == 0) {
            //     val = decoded * (int) this->quantTable1[length];
            // } else {
            //     val = decoded * (int) this->quantTable2[length];
            // }
            printf("ac coeff %d:\n", decoded);
            val = decoded * (int) quant[length];
            outBuffer[length] = val;
            length++;
        }
    }

    // Update oldCoeff for the next MCU
    oldCoeff = dcCoeff;
    // printf("Returning Bitoffset %d:\n", bitOffset);
    return bitOffset;
}

JPEGParser::~JPEGParser() {

    cudaFree(idctTable);

    delete[] quantTable1;
    delete[] quantTable2;

    delete channels;

    for (auto& tree : huffmanTrees) {
        delete tree.second;
    }

    delete this->applicationHeader;
    delete this->quantTable1;
    delete this->quantTable2;
    delete this->startOfFrame;
    delete this->startOfScan;
    delete this->huffmanTable1;
    delete this->huffmanTable2;
    delete this->huffmanTable3;
    delete this->huffmanTable4;
    delete this->imageData;
}

__global__ void decodeKernel(uint8_t* imageData, int* arr_l, int* arr_r, int* arr_y, double* idctTable, int validHeight, 
                                int validWidth, int width, int height, int xBlocks, int yBlocks, int* redOutput, 
                                int* greenOutput, int* blueOutput, uint8_t* quant1, uint8_t* quant2, 
                                uint16_t* hf0codes, uint16_t* hf1codes, uint16_t* hf16codes, uint16_t* hf17codes,
                                int* hf0lengths, int* hf1lengths, int* hf16lengths, int* hf17lengths) {

    int globalBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
    int blockStart = globalBlockIndex * 64;

    // Identify the thread's position in the 8x8 grid
    int threadRow = threadIdx.y; // Row index (0-7)
    int threadCol = threadIdx.x; // Column index (0-7)
    int threadIndexInBlock = threadRow * 8 + threadCol; // Flattened index

    // Calculate the global index for this thread
    int globalIndex = blockStart + threadIndexInBlock;

    if (globalIndex==0) {
        printf("Thread %d:\n", globalIndex);
        // printf("huffman codes of hf0codes\n");
        // for (int idx=0; idx < 20; idx++) {
        //     printf("arr[%d] = %u\n", idx, static_cast<unsigned int>(hf0codes[idx]));
        // }
        int *curLuminous = arr_l;
        int *curChromRed = arr_r;
        int *curChromYel = arr_y;
        int oldLumCoeff = 0;
        int oldCbdCoeff = 0;
        int oldCrdCoeff = 0;
        int bitOffset = 0;
        for (int y = 0; y < yBlocks; y++) {
            for (int x = 0; x < xBlocks; x++) {
                // Determine the valid width and height for this block to account for padding
                // int blockWidth = (x == xBlocks - 1 && paddedWidth != this->width) ? this->width % 8 : 8;
                // int blockHeight = (y == yBlocks - 1 && paddedHeight != this->height) ? this->height % 8 : 8;

                // bitOffset = buildMCU(curLuminous, imageStream, 0, 0, oldLumCoeff);
                // printf("oldLumcoefficient %d:\n", oldLumCoeff);
                bitOffset = buildMCU(curLuminous, imageData, bitOffset, quant1, oldLumCoeff, hf0codes, hf0lengths, hf16codes, hf16lengths);
                // printf("Bitoffset %d:\n", bitOffset);
                // printf("oldCbdcoefficient %d:\n", oldCbdCoeff);
                bitOffset = buildMCU(curChromRed, imageData, bitOffset, quant2, oldCbdCoeff, hf1codes, hf1lengths, hf17codes, hf17lengths);
                // printf("Bitoffset %d:\n", bitOffset);
                // printf("oldCrdcoefficient %d:\n", oldCrdCoeff);
                bitOffset = buildMCU(curChromYel, imageData, bitOffset, quant2, oldCrdCoeff, hf1codes, hf1lengths, hf17codes, hf17lengths);
                // bitOffset = buildMCU(curChromRed, imageStream, 1, 1, oldCbdCoeff);
                // bitOffset = buildMCU(curChromYel, imageStream, 1, 1, oldCrdCoeff);

                curLuminous += 64;
                curChromRed += 64;
                curChromYel += 64;
            }
        }
    }
    __syncthreads();

    // Shared memory for zigzag arrays
    __shared__ int sharedZigzag[3 * 64];
    int* zigzag_l = &sharedZigzag[0];
    int* zigzag_r = &sharedZigzag[64];
    int* zigzag_y = &sharedZigzag[128];

    if (threadCol < validWidth && threadRow < validHeight) {
        zigzag_l[threadIndexInBlock] = arr_l[blockStart + initialZigzag[threadIndexInBlock]];
        zigzag_r[threadIndexInBlock] = arr_r[blockStart + initialZigzag[threadIndexInBlock]];
        zigzag_y[threadIndexInBlock] = arr_y[blockStart + initialZigzag[threadIndexInBlock]];
    } else {
        zigzag_l[threadIndexInBlock] = 0;
        zigzag_r[threadIndexInBlock] = 0;
        zigzag_y[threadIndexInBlock] = 0;
    }

    __syncthreads();

    if (threadCol < validWidth && threadRow < validHeight) {
        double localSum_l = 0.0;
        double localSum_r = 0.0;
        double localSum_y = 0.0;
        for (int u = 0; u < 8; u++) {
            for (int v = 0; v < 8; v++) {
                localSum_l += zigzag_l[v * 8 + u] * idctTable[u * 8 + threadCol] * idctTable[v * 8 + threadRow];
                localSum_r += zigzag_r[v * 8 + u] * idctTable[u * 8 + threadCol] * idctTable[v * 8 + threadRow];
                localSum_y += zigzag_y[v * 8 + u] * idctTable[u * 8 + threadCol] * idctTable[v * 8 + threadRow];
            }
        }

        arr_l[globalIndex] = static_cast<int>(std::floor(localSum_l / 4.0)); //luminuous
        arr_y[globalIndex] = static_cast<int>(std::floor(localSum_y / 4.0)); //chromyel
        arr_r[globalIndex] = static_cast<int>(std::floor(localSum_r / 4.0)); // chromred
    }

    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate
    int y = blockIdx.y * blockDim.y + threadIdx.y; // y-coordinate
    int i = y * width + x;

    if (x < width && y < height) {
        int blockIndex = (y / 8) * xBlocks + (x / 8); // Index of the current 8x8 block
        int pixelIndexInBlock = threadIdx.y * 8 + threadIdx.x;  // Position within the block

        float red = arr_y[blockIndex * 64 + pixelIndexInBlock] * (2 - 2 * 0.299) + arr_l[blockIndex * 64 + pixelIndexInBlock];
        float blue = arr_r[blockIndex * 64 + pixelIndexInBlock] * (2 - 2 * 0.114) + arr_l[blockIndex * 64 + pixelIndexInBlock];
        float green = (arr_l[blockIndex * 64 + pixelIndexInBlock] - 0.114 * blue - 0.299 * red) / 0.587;

        int castedRed = static_cast<int>(red + 128);
        int castedGreen = static_cast<int>(green + 128);
        int castedBlue = static_cast<int>(blue + 128);

        if (castedRed > 255) {
            redOutput[i] = 255;
        } else if (castedRed < 0) {
            redOutput[i] = 0;
        } else {
            redOutput[i] = castedRed;
        }

        if (castedGreen > 255) {
            greenOutput[i] = 255;
        } else if (castedGreen < 0) {
            greenOutput[i] = 0;
        } else {
            greenOutput[i] = castedGreen;
        }

        if (castedBlue > 255) {
            blueOutput[i] = 255;
        } else if (castedBlue < 0) {
            blueOutput[i] = 0;
        } else {
            blueOutput[i] = castedBlue;
        }
    }

}

void JPEGParser::decode() {
    // int oldLumCoeff = 0;
    // int oldCbdCoeff = 0;
    // int oldCrdCoeff = 0;

    // Pad the image dimension if it is not divisible by 8
    // int paddedWidth = ((this->width + 7) / 8) * 8;
    // int paddedHeight = ((this->height + 7) / 8) * 8;

    // int xBlocks = paddedWidth / 8;
    // int yBlocks = paddedHeight / 8;

    // Stream* imageStream = new Stream(this->imageData);

    // Allocating the channels in the GPU memory.
    // int *luminous, *chromRed, *chromYel;
    // int *redOutput, *greenOutput, *blueOutput;
    // cudaMalloc((void**)&this->luminous, 64 * xBlocks * yBlocks * sizeof(int));
    // cudaMalloc((void**)&this->chromRed, 64 * xBlocks * yBlocks * sizeof(int));
    // cudaMalloc((void**)&this->chromYel, 64 * xBlocks * yBlocks * sizeof(int));
    // cudaMalloc((void**)&this->redOutput, 64 * xBlocks * yBlocks * sizeof(int));
    // cudaMalloc((void**)&this->greenOutput, 64 * xBlocks * yBlocks * sizeof(int));
    // cudaMalloc((void**)&this->blueOutput, 64 * xBlocks * yBlocks * sizeof(int));

    // int* hostBuffer_l = new int[64 * xBlocks * yBlocks * sizeof(int)];
    // int* hostBuffer_y = new int[64 * xBlocks * yBlocks * sizeof(int)];
    // int* hostBuffer_r = new int[64 * xBlocks * yBlocks * sizeof(int)];

    // int *curLuminous = hostBuffer_l;
    // int *curChromRed = hostBuffer_r;
    // int *curChromYel = hostBuffer_y;

    // for (int y = 0; y < yBlocks; y++) {
    //     for (int x = 0; x < xBlocks; x++) {
    //         // Determine the valid width and height for this block to account for padding
    //         // int blockWidth = (x == xBlocks - 1 && paddedWidth != this->width) ? this->width % 8 : 8;
    //         // int blockHeight = (y == yBlocks - 1 && paddedHeight != this->height) ? this->height % 8 : 8;

    //         this->buildMCU(curLuminous, imageStream, 0, 0, oldLumCoeff);
    //         this->buildMCU(curChromRed, imageStream, 1, 1, oldCbdCoeff);
    //         this->buildMCU(curChromYel, imageStream, 1, 1, oldCrdCoeff);
    //         curLuminous += 64;
    //         curChromRed += 64;
    //         curChromYel += 64;
    //     }
    // }

    // cudaMemcpy(luminous, hostBuffer_l, 64 * xBlocks * yBlocks * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(chromRed, hostBuffer_r, 64 * xBlocks * yBlocks * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(chromYel, hostBuffer_y, 64 * xBlocks * yBlocks * sizeof(int), cudaMemcpyHostToDevice);

    // int numBlocks = xBlocks * yBlocks; // Number of CUDA blocks
    // dim3 threadsPerBlock(8, 8);

    dim3 blockSize(8, 8);
    dim3 gridSize(this->xBlocks, this->yBlocks);

    decodeKernel<<<gridSize, blockSize>>>(this->imageData, this->luminous, this->chromRed, this->chromYel, idctTable, 8, 8,  
                                            this->width, this->height, this->xBlocks, this->yBlocks, this->redOutput, this->greenOutput, this->blueOutput,
                                            this->quantTable1, this->quantTable2, this->hf0codes, this->hf1codes, this->hf16codes, this->hf17codes, 
                                            this->hf0lengths, this->hf1lengths, this->hf16lengths, this->hf17lengths);
    this->channels = new ImageChannels(this->height * this->width);

    // cudaMemcpy(channels->getR().data(), redOutput, channelSize, cudaMemcpyDeviceToHost);
    // cudaMemcpy(channels->getG().data(), greenOutput, channelSize, cudaMemcpyDeviceToHost);
    // cudaMemcpy(channels->getB().data(), blueOutput, channelSize, cudaMemcpyDeviceToHost);

    // FOR DEBUGGING PURPOSES ONLY
    int bytes = 64 * xBlocks * yBlocks * sizeof(int);
    int *h_array = (int *)malloc(bytes);

    cudaMemcpy(h_array, luminous, bytes, cudaMemcpyDeviceToHost);
    std::cout << "Array contents copied from GPU to CPU:" << std::endl;
    for (int i = 0; i < (64 * xBlocks * yBlocks); i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;
    cudaMemcpy(h_array, chromRed, bytes, cudaMemcpyDeviceToHost);
    std::cout << "Array contents copied from GPU to CPU:" << std::endl;
    for (int i = 0; i < (64 * xBlocks * yBlocks); i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;
    cudaMemcpy(h_array, chromYel, bytes, cudaMemcpyDeviceToHost);
    std::cout << "Array contents copied from GPU to CPU:" << std::endl;
    for (int i = 0; i < (64 * xBlocks * yBlocks); i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    if (luminous) cudaFree(luminous);
    if (chromRed) cudaFree(chromRed);
    if (chromYel) cudaFree(chromYel);

    // delete imageStream;
    // delete hostBuffer_l;
    // delete hostBuffer_r;
    // delete hostBuffer_y;
    
}

void JPEGParser::write() {

    size_t channelSize = this->width * this->height * sizeof(int);
    cudaMemcpy(channels->getR().data(), redOutput, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(channels->getG().data(), greenOutput, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(channels->getB().data(), blueOutput, channelSize, cudaMemcpyDeviceToHost);
    cudaFree(redOutput);
    cudaFree(greenOutput);
    cudaFree(blueOutput);

    // Writing the decoded channels to a file instead of displaying using opencv
    fs::path output_dir = "../testing/cudaU_output_arrays"; // Change the directory name here for future CUDA implementations
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