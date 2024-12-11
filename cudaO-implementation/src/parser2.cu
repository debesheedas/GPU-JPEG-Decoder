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
                return;
            }
        }
    }
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
    outBuffer[0] = dcCoeff * (int) quant[0];

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
            int val;
            val = decoded * (int) quant[length];
            outBuffer[length] = val;
            length++;
        }
    }
    // Update oldCoeff for the next MCU
    oldCoeff = dcCoeff;
    return bitOffset;
}

JPEGParser::~JPEGParser() {
    cudaFree(idctTable);
    delete channels;
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
                                        const int* initialZigzag) {
    zigzag_l[threadId] = arr_l[blockIndex * 64 + initialZigzag[threadIndexInBlock]];
    zigzag_r[threadId] = arr_r[blockIndex * 64 + initialZigzag[threadIndexInBlock]];
    zigzag_y[threadId] = arr_y[blockIndex * 64 + initialZigzag[threadIndexInBlock]];
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

__global__ void decodeKernel(uint8_t* imageData, int* arr_l, int* arr_r, int* arr_y, double* idctTable, int validHeight, 
                                int validWidth, int width, int height, int xBlocks, int yBlocks, int* redOutput, 
                                int* greenOutput, int* blueOutput, uint8_t* quant1, uint8_t* quant2, 
                                uint16_t* hf0codes, uint16_t* hf1codes, uint16_t* hf16codes, uint16_t* hf17codes,
                                int* hf0lengths, int* hf1lengths, int* hf16lengths, int* hf17lengths) {

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
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

        __shared__ int sharedZigzag[3 * 256];
        int* zigzag_l = &sharedZigzag[0];
        int* zigzag_r = &sharedZigzag[256];
        int* zigzag_y = &sharedZigzag[512];

        performZigzagReordering(arr_l, arr_r, arr_y, zigzag_l, zigzag_r, zigzag_y,
                                blockIndex, threadIndexInBlock, threadId, initialZigzag);

        __syncthreads();

        int threadPos = pixelIndex % 64;
        int threadRow = threadPos / 8;
        int threadCol = threadPos % 8;
        double localSum_l = 0.0;
        double localSum_r = 0.0;
        double localSum_y = 0.0;

        performIDCT(zigzag_l, idctTable, threadCol, threadRow, localSum_l, threadId - (threadId % 64));
        performIDCT(zigzag_r, idctTable, threadCol, threadRow, localSum_r, threadId - (threadId % 64));
        performIDCT(zigzag_y, idctTable, threadCol, threadRow, localSum_y, threadId - (threadId % 64));


        arr_l[pixelIndex] = static_cast<int>(std::floor(localSum_l / 4.0));
        arr_y[pixelIndex] = static_cast<int>(std::floor(localSum_y / 4.0));
        arr_r[pixelIndex] = static_cast<int>(std::floor(localSum_r / 4.0));

        pixelIndex += blockDim.x * gridDim.x;
    }

    __syncthreads();

    // Iterate over pixels handled by this thread
    performColorConversion(arr_l, arr_r, arr_y, redOutput, greenOutput, blueOutput, 
                           totalPixels, width, threadId, blockDim.x * gridDim.x);
}

void JPEGParser::decode() {
    decodeKernel<<<1, 256>>>(this->imageData, this->luminous, this->chromRed, this->chromYel, idctTable, 8, 8,  
                                            this->width, this->height, this->xBlocks, this->yBlocks, this->redOutput, this->greenOutput, this->blueOutput,
                                            this->quantTable1, this->quantTable2, this->hf0codes, this->hf1codes, this->hf16codes, this->hf17codes, 
                                            this->hf0lengths, this->hf1lengths, this->hf16lengths, this->hf17lengths);

    this->channels = new ImageChannels(this->height * this->width);

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

    size_t channelSize = this->width * this->height * sizeof(int);
    cudaMemcpy(channels->getR().data(), redOutput, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(channels->getG().data(), greenOutput, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(channels->getB().data(), blueOutput, channelSize, cudaMemcpyDeviceToHost);
    cudaFree(redOutput);
    cudaFree(greenOutput);
    cudaFree(blueOutput);

    // Writing the decoded channels to a file instead of displaying using opencv
    fs::path output_dir = "../testing/cudaO_output_arrays"; // Change the directory name here for future CUDA implementations
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