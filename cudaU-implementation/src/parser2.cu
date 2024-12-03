#include "parser2.h"

__constant__ int initialZigzag[64]; 

__device__ int match_huffman_code(const unsigned uint8_t* stream, int bit_offset, 
                                  const uint16_t* huff_codes, 
                                  const int* huff_bits) {
    // Extract up to 16 bits from the stream (max Huffman code length)
    unsigned int extracted_bits = getNBits(stream, bit_offset, 16);

    // Compare against Huffman table
    for (int i = 0; i < 256; ++i) {
        if (huff_bits[i] > 0 && huff_bits[i] <= 16) { // Valid bit length
            unsigned int mask = (1 << huff_bits[i]) - 1;
            if ((extracted_bits >> (16 - huff_bits[i])) == huff_codes[i]) {
                return i; // Return the index of the matched Huffman code
            }
        }
    }

    return -1; // No match found
}

// __device__ int parallel_match_huffman_code(const uint8_t* stream, int bit_offset, 
//                                            const uint16_t* huff_codes, const int* huff_bits) {
//     // Extract up to 16 bits from the stream (max Huffman code length)
//     unsigned int extracted_bits = getNBits(stream, bit_offset, 16);

//     // Allocate shared memory for threads to indicate match
//     __shared__ int match_index;
//     if (threadIdx.x == 0) {
//         match_index = -1; // Initialize to no match
//     }
//     __syncthreads();

//     // Each thread checks one Huffman code
//     int thread_code_index = threadIdx.x;
//     if (thread_code_index < 256) {
//         if (huff_bits[thread_code_index] > 0 && huff_bits[thread_code_index] <= 16) {
//             unsigned int mask = (1 << huff_bits[thread_code_index]) - 1;
//             if ((extracted_bits >> (16 - huff_bits[thread_code_index])) == huff_codes[thread_code_index]) {
//                 atomicMin(&match_index, thread_code_index); // Record the match index
//             }
//         }
//     }

//     __syncthreads();

//     // Return the matching index
//     return match_index;
// }

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

void JPEGParser::extract() {        
    uint16_t tableSize = 0;
    uint8_t header = 0;

    // Using the Stream class for reading bytes.
    Stream* stream = new Stream(this->readBytes);

    while (true) {
        uint16_t marker = stream->getMarker();

        if (marker == MARKERS[0]) {
            std::cout << "beginning" << std::endl;
            continue;
        } else if (marker == MARKERS[1]) {
            std::cout << "other beginning" << std::endl;
            tableSize = stream->getMarker();
            this->applicationHeader = new uint8_t[(int) tableSize - 2];
            stream->getNBytes(this->applicationHeader, int(tableSize - 2));
        } else if (marker == MARKERS[2]) {
            std::cout << "other marker" << std::endl;
            stream->getMarker();
            uint8_t destination = stream->getByte();
            this->quantTable1 = new uint8_t[64];
            stream->getNBytes(quantTable1, 64);
            if(stream->getMarker() == MARKERS[2]) {
                std::cout << "other other beginning" << std::endl;
                stream->getMarker();
                destination = stream->getByte();
                this->quantTable2 = new uint8_t[64];
                stream->getNBytes(quantTable2, 64);
            } else {
                std::cout << " Something went wrong at parsing second quant table." << std::endl;
            }
        } else if (marker == MARKERS[3]) {
            std::cout << "makrer start if frame" << std::endl;
            tableSize = stream->getMarker();
            this->startOfFrame = new uint8_t[(int) tableSize - 2];
            stream->getNBytes(this->startOfFrame, (int) tableSize - 2);
            Stream* frame = new Stream(this->startOfFrame);
            int precision = frame->getByte();
            this->height = frame->getMarker();
            this->width = frame->getMarker();
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
                std::cout << "printing the codes " << std::endl;
                this->huffmanTrees[header]->printCodes();
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
            this->imageData = new uint8_t[size];
            
            while (true) {
                curByte = stream->getByte();
                if ((prevByte == 0xff) && (curByte == 0xd9))
                    break;
                if (curByte == 0x00) {
                    if (prevByte != 0xff) {
                        this->imageData[imageDataLength++] = curByte;
                    }
                } else {
                    this->imageData[imageDataLength++] = curByte;
                }
                prevByte = curByte;
            }
            
            imageDataLength--; // We remove the ending byte because it is extra 0xff.
            break;
        }
    } 
    delete stream;   
}

void JPEGParser::buildMCU(int* hostBuffer, Stream* imageStream, int hf, int quant, int& oldCoeff) {
    //uint8_t code = this->huffmanTrees[hf]->getCode(imageStream);
    uint16_t bits = imageStream->getNBits(code);
    int decoded = Stream::decodeNumber(code, bits);
    int dcCoeff = decoded + oldCoeff;

    if (quant == 0) {
        hostBuffer[0] = dcCoeff * (int) this->quantTable1[0];
    } else {
        hostBuffer[0] = dcCoeff * (int) this->quantTable2[0];
    }
    int length = 1;

    while (length < 64) {
        code = this->huffmanTrees[16 + hf]->getCode(imageStream);

        if (code == 0) {
            break;
        }

        // The first part of the AC key length is the number of leading zeros
        if (code > 15) {
            length += (code >> 4);
            code = code & 0x0f;
        }

        bits = imageStream->getNBits(code);
        if (length < 64) {
            decoded = Stream::decodeNumber(code, bits);
            int val;
            if (quant == 0) {
                val = decoded * (int) this->quantTable1[length];
            } else {
                val = decoded * (int) this->quantTable2[length];
            }
            hostBuffer[length] = val;
            length++;
        }
    }

    // Update oldCoeff for the next MCU
    oldCoeff = dcCoeff;
}

JPEGParser::~JPEGParser() {
    // if (this->channels) {
    //     delete this->channels;
    // }

    cudaFree(idctTable);

    delete[] quantTable1;
    delete[] quantTable2;

    delete channels;

    for (auto& tree : huffmanTrees) {
        delete tree.second;
    }

    delete this->applicationHeader;
    //delete this->quantTable1;
    //delete this->quantTable2;
    delete this->startOfFrame;
    delete this->startOfScan;
    delete this->huffmanTable1;
    delete this->huffmanTable2;
    delete this->huffmanTable3;
    delete this->huffmanTable4;
    delete this->imageData;
}

__global__ void decodeKernel(int* arr_l, int* arr_r, int* arr_y, double* idctTable, int validHeight, int validWidth,  int width, int height, int xBlocks, int yBlocks, int* redOutput, int* greenOutput, int* blueOutput) {
    // Shared memory for zigzag arrays
    __shared__ int sharedZigzag[3 * 64];
    int* zigzag_l = &sharedZigzag[0];
    int* zigzag_r = &sharedZigzag[64];
    int* zigzag_y = &sharedZigzag[128];
    

    int globalBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
    int blockStart = globalBlockIndex * 64;

    // Identify the thread's position in the 8x8 grid
    int threadRow = threadIdx.y; // Row index (0-7)
    int threadCol = threadIdx.x; // Column index (0-7)
    int threadIndexInBlock = threadRow * 8 + threadCol; // Flattened index

    // Calculate the global index for this thread
    int globalIndex = blockStart + threadIndexInBlock;

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
    int oldLumCoeff = 0;
    int oldCbdCoeff = 0;
    int oldCrdCoeff = 0;

    // Pad the image dimension if it is not divisible by 8
    int paddedWidth = ((this->width + 7) / 8) * 8;
    int paddedHeight = ((this->height + 7) / 8) * 8;

    int xBlocks = paddedWidth / 8;
    int yBlocks = paddedHeight / 8;

    Stream* imageStream = new Stream(this->imageData);

    // Allocating the channels in the GPU memory.
    int *luminous, *chromRed, *chromYel;
    int *redOutput, *greenOutput, *blueOutput;
    cudaMalloc((void**)&luminous, 64 * xBlocks * yBlocks * sizeof(int));
    cudaMalloc((void**)&chromRed, 64 * xBlocks * yBlocks * sizeof(int));
    cudaMalloc((void**)&chromYel, 64 * xBlocks * yBlocks * sizeof(int));
    cudaMalloc((void**)&redOutput, 64 * xBlocks * yBlocks * sizeof(int));
    cudaMalloc((void**)&greenOutput, 64 * xBlocks * yBlocks * sizeof(int));
    cudaMalloc((void**)&blueOutput, 64 * xBlocks * yBlocks * sizeof(int));

    int* hostBuffer_l = new int[64 * xBlocks * yBlocks * sizeof(int)];
    int* hostBuffer_y = new int[64 * xBlocks * yBlocks * sizeof(int)];
    int* hostBuffer_r = new int[64 * xBlocks * yBlocks * sizeof(int)];

    int *curLuminous = hostBuffer_l;
    int *curChromRed = hostBuffer_r;
    int *curChromYel = hostBuffer_y;

    for (int y = 0; y < yBlocks; y++) {
        for (int x = 0; x < xBlocks; x++) {
            // Determine the valid width and height for this block to account for padding
            // int blockWidth = (x == xBlocks - 1 && paddedWidth != this->width) ? this->width % 8 : 8;
            // int blockHeight = (y == yBlocks - 1 && paddedHeight != this->height) ? this->height % 8 : 8;

            this->buildMCU(curLuminous, imageStream, 0, 0, oldLumCoeff);
            this->buildMCU(curChromRed, imageStream, 1, 1, oldCbdCoeff);
            this->buildMCU(curChromYel, imageStream, 1, 1, oldCrdCoeff);
            curLuminous += 64;
            curChromRed += 64;
            curChromYel += 64;
        }
    }

    cudaMemcpy(luminous, hostBuffer_l, 64 * xBlocks * yBlocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(chromRed, hostBuffer_r, 64 * xBlocks * yBlocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(chromYel, hostBuffer_y, 64 * xBlocks * yBlocks * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = xBlocks * yBlocks; // Number of CUDA blocks
    dim3 threadsPerBlock(8, 8);

    // Convert YCbCr channels to RGB
    dim3 blockSize(8, 8);
    dim3 gridSize(xBlocks, yBlocks);
    size_t channelSize = width * height * sizeof(int);
    decodeKernel<<<gridSize, blockSize>>>(luminous, chromRed, chromYel, idctTable, 8, 8,  width, height, xBlocks, yBlocks, redOutput, greenOutput, blueOutput);
    this->channels = new ImageChannels(this->height * this->width);

    cudaMemcpy(channels->getR().data(), redOutput, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(channels->getG().data(), greenOutput, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(channels->getB().data(), blueOutput, channelSize, cudaMemcpyDeviceToHost);

    if (luminous) cudaFree(luminous);
    if (chromRed) cudaFree(chromRed);
    if (chromYel) cudaFree(chromYel);

    delete imageStream;
    delete hostBuffer_l;
    delete hostBuffer_r;
    delete hostBuffer_y;
    cudaFree(redOutput);
    cudaFree(greenOutput);
    cudaFree(blueOutput);
}

void JPEGParser::write() {
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