#include "parser.h"

__constant__ int initialZigzag[64]; 

__global__ void colorConversionKernel(int* luminous, int* chromRed, int* chromYel, int width, int height, int xBlocks, int yBlocks) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate
    int y = blockIdx.y * blockDim.y + threadIdx.y; // y-coordinate
    int i = y * width + x;

    if (x < width && y < height) {
        int blockIndex = (y / 8) * xBlocks + (x / 8); // Index of the current 8x8 block
        int pixelIndexInBlock = threadIdx.y * 8 + threadIdx.x;  // Position within the block

        float red = chromRed[blockIndex * 64 + pixelIndexInBlock] * (2 - 2 * 0.299) + luminous[blockIndex * 64 + pixelIndexInBlock];
        float blue = chromYel[blockIndex * 64 + pixelIndexInBlock] * (2 - 2 * 0.114) + luminous[blockIndex * 64 + pixelIndexInBlock];
        float green = (luminous[blockIndex * 64 + pixelIndexInBlock] - 0.114 * blue - 0.299 * red) / 0.587;

        int castedRed = static_cast<int>(red + 128);
        int castedGreen = static_cast<int>(green + 128);
        int castedBlue = static_cast<int>(blue + 128);

        if (castedRed > 255) {
            chromRed[i] = 255;
        } else if (castedRed < 0) {
            chromRed[i] = 0;
        } else {
            chromRed[i] = castedRed;
        }

        if (castedGreen > 255) {
            chromYel[i] = 255;
        } else if (castedGreen < 0) {
            chromYel[i] = 0;
        } else {
            chromYel[i] = castedGreen;
        }

        if (castedBlue > 255) {
            luminous[i] = 255;
        } else if (castedBlue < 0) {
            luminous[i] = 0;
        } else {
            luminous[i] = castedBlue;
        }
    }
}

__global__ void initializeIDCTTableKernel(double *dIdctTable, int numThreads)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (id < numThreads) {
        double normCoeff = ((id / 8) == 0) ? (1.0 / sqrt(2.0)) : 1.0;
        dIdctTable[id] = normCoeff * cos(((2.0 * (id%8) + 1.0) * (id/8) * M_PI) / 16.0);
    }
}

JPEGParser::JPEGParser(std::string& imagePath): quantTables(2) {
    // Extract the file name of the image file from the file path
    fs::path file_path(imagePath);
    this->filename = file_path.filename().string();
    std::ifstream input(imagePath, std::ios::binary);
    
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()));
    this->readBytes = bytes;
    input.close();


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

    quantTables[0] = new uint8_t[64];
    quantTables[1] = new uint8_t[64];

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
            continue;
        } else if (marker == MARKERS[1]) {
            tableSize = stream->getMarker();
            stream->getNBytes(this->applicationHeader, int(tableSize - 2));
        } else if (marker == MARKERS[2]) {
            stream->getMarker();
            uint8_t destination = stream->getByte();
            stream->getNBytes(quantTables[0], 64);
            if(stream->getMarker() == MARKERS[2]) {
                stream->getMarker();
                destination = stream->getByte();
                stream->getNBytes(quantTables[1], 64);
            } else {
                std::cout << " Something went wrong at parsing second quant table." << std::endl;
            }
        } else if (marker == MARKERS[3]) {
            tableSize = stream->getMarker();
            stream->getNBytes(this->startOfFrame, (int) tableSize - 2);
            Stream* frame = new Stream(this->startOfFrame);
            int precision = frame->getByte();
            this->height = frame->getMarker();
            this->width = frame->getMarker();
        } else if (marker == MARKERS[4]) {
            tableSize = stream->getMarker();
            header = stream->getByte();
            stream->getNBytes(this->huffmanTables[0], (int) tableSize - 3);
            this->huffmanTrees[header] = new HuffmanTree(this->huffmanTables[0]);

            int huffmanCount = 1;
            while(huffmanCount < 4) {
                if (stream->getMarker() ==  MARKERS[4]) {
                    tableSize = stream->getMarker();
                    header = stream->getByte();
                    stream->getNBytes(this->huffmanTables[huffmanCount], (int) tableSize - 3);
                    this->huffmanTrees[header] = new HuffmanTree(this->huffmanTables[huffmanCount]);
                    huffmanCount++; 
                }
            }
        } else if (marker == MARKERS[5]) {
            tableSize = stream->getMarker();
            stream->getNBytes(this->startOfScan, (int) tableSize - 2);
            uint8_t curByte, prevByte = 0x00;

            while (true) {
                curByte = stream->getByte();
                if ((prevByte == 0xff) && (curByte == 0xd9))
                    break;
                if (curByte == 0x00) {
                    if (prevByte != 0xff) {
                        this->imageData.push_back(curByte);
                    }
                } else {
                    this->imageData.push_back(curByte);
                }
                prevByte = curByte;
            }
            
            imageData.pop_back(); // We remove the ending byte because it is extra 0xff.
            break;
        }
    }   
}

void JPEGParser::buildMCU(int* hostBuffer, Stream* imageStream, int hf, int quant, int& oldCoeff) {
    uint8_t code = this->huffmanTrees[hf]->getCode(imageStream);
    uint16_t bits = imageStream->getNBits(code);
    int decoded = Stream::decodeNumber(code, bits);
    int dcCoeff = decoded + oldCoeff;

    hostBuffer[0] = dcCoeff * (int) this->quantTables[quant][0];
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
            int val = decoded * (int) this->quantTables[quant][length];
            hostBuffer[length] = val;
            length++;
        }
    }

    // Update oldCoeff for the next MCU
    oldCoeff = dcCoeff;
}

__global__ void performIDCTKernel(int* arr_l, int* arr_r, int* arr_y, double* idctTable, int validHeight, int validWidth) {
    // Shared memory for zigzag arrays
    __shared__ int sharedZigzag[3 * 64];
    int* zigzag_l = &sharedZigzag[0];
    int* zigzag_r = &sharedZigzag[64];
    int* zigzag_y = &sharedZigzag[128];
    
    int blockStart = blockIdx.x * 64;

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

        arr_l[globalIndex] = static_cast<int>(std::floor(localSum_l / 4.0));
        arr_y[globalIndex] = static_cast<int>(std::floor(localSum_y / 4.0));
        arr_r[globalIndex] = static_cast<int>(std::floor(localSum_r / 4.0));
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
    cudaMalloc((void**)&luminous, 64 * xBlocks * yBlocks * sizeof(int));
    cudaMalloc((void**)&chromRed, 64 * xBlocks * yBlocks * sizeof(int));
    cudaMalloc((void**)&chromYel, 64 * xBlocks * yBlocks * sizeof(int));

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

    performIDCTKernel<<<numBlocks, threadsPerBlock>>>(luminous, chromRed, chromYel, idctTable, 8, 8);

    this->channels = new ImageChannels(this->height * this->width);

    // Convert YCbCr channels to RGB
    dim3 blockSize(8, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    size_t channelSize = width * height * sizeof(int);
    colorConversionKernel<<<gridSize, blockSize>>>(luminous, chromYel, chromRed, width, height, xBlocks, yBlocks);

    cudaMemcpy(channels->getR().data(), chromYel, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(channels->getG().data(), chromRed, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(channels->getB().data(), luminous, channelSize, cudaMemcpyDeviceToHost);

    if (luminous) cudaFree(luminous);
    if (chromRed) cudaFree(chromRed);
    if (chromYel) cudaFree(chromYel);
}

void JPEGParser::write() {
    // Writing the decoded channels to a file instead of displaying using opencv
    fs::path output_dir = "../testing/cuda_output_arrays"; // Change the directory name here for future CUDA implementations
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