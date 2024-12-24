#include "parser.h"

__device__ int16_t clip(int16_t value) {
    if (value < -256) return -256;
    if (value > 255) return 255;
    return value;
}

__device__ void idctRow(int16_t* block) {
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

__device__ void idctCol(int16_t* block) {
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

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

/* Function to allocate the GPU space. */
void allocate(int16_t*& yCrCbChannels, int16_t*& rgbChannels, int16_t*& outputChannels, int width, int height, int*& zigzagLocations) {

    // const size_t codeSize = 256 * sizeof(uint16_t);
    // const size_t lengthSize = 256 * sizeof(int);
    const size_t imageSize = width * height * sizeof(int16_t);

    // checkCudaError(cudaMalloc((void**)&hfCodes, codeSize * 4), "Failed to allocate device memory for huffman codes.");
    // checkCudaError(cudaMalloc((void**)&hfLengths, lengthSize * 4), "Failed to allocate device memory for huffman lengths.");

    checkCudaError(cudaMalloc((void**)&zigzagLocations, 256 * sizeof(int)), "Failed to allocate device memory for zigzag table.");
    checkCudaError(cudaMemcpy(zigzagLocations, zigzagEntries, sizeof(int) * 64, cudaMemcpyHostToDevice), "Failed to copy entries for the zigzag table.");

    // int index = 0;

    // for (int i = 0; i < 4; ++i) {
    //     if (i > 1 && index < 16)
    //         index = 16;

    //     checkCudaError(cudaMemcpy(hfCodes + i * 256, huffmanTrees[index]->codes, codeSize, cudaMemcpyHostToDevice), "Failed to copy data to device for huffman codes");
    //     checkCudaError(cudaMemcpy(hfLengths + i * 256, huffmanTrees[index]->codeLengths, lengthSize, cudaMemcpyHostToDevice), "Failed to copy data to device for huffman lengths");
    //     index++;
    // }

    // Allocating the channels in the GPU memory.
    checkCudaError(cudaMalloc((void**)&yCrCbChannels, imageSize * 3), "Failed to allocate device memory for one yCrCb channel.");
    checkCudaError(cudaMalloc((void**)&rgbChannels, imageSize * 3), "Failed to allocate device memory for one rgb channel.");
    checkCudaError(cudaMalloc((void**)&outputChannels, imageSize * 3), "Failed to allocate device memory for one output channel.");
}

void extract(std::string imagePath, uint8_t*& quantTables, std::vector<uint8_t>& imageData, int& width, int& height, std::unordered_map<int,HuffmanTree*>& huffmanTrees) {  

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
            // size_t size = 5 * 1024 * 1024;
            
            while (true) {
                curByte = stream.getByte();
                if ((prevByte == 0xff) && (curByte == 0xd9))
                    break;
                if (curByte == 0x00) {
                    if (prevByte != 0xff) {
                        imageData.push_back(curByte);
                        imageDataLength++;
                    }
                } else {
                    imageData.push_back(curByte);
                    imageDataLength++;
                }
                prevByte = curByte;
            }
            
            imageDataLength--; // We remove the ending byte because it is extra 0xff.
            break;
        }
    } 

}

void buildMCU(int16_t* hostBuffer, Stream* imageStream, int& oldCoeff, int hf, std::unordered_map<int,HuffmanTree*> huffmanTrees) {
    // if (huffmanTrees[hf]) std::cout << "not null" << std::endl;
    uint8_t code = huffmanTrees[hf]->getCode(imageStream);
    // printf("dc code %d:\n", code);
    uint16_t bits = imageStream->getNBits(code);
    int decoded = Stream::decodeNumber(code, bits);
    int dcCoeff = decoded + oldCoeff;
    // printf("dc coeff %d:\n", dcCoeff);
    hostBuffer[0] = dcCoeff;
    int length = 1;
    while (length < 64) {
        code = huffmanTrees[16 + hf]->getCode(imageStream);
        // printf("code %d:\n", code);
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
            // printf("ac coeff %d:\n", decoded);
            int val = decoded;
            hostBuffer[length] = val;
            length++;
        }
    }
    // Update oldCoeff for the next MCU
    oldCoeff = dcCoeff;
}

void performHuffmanDecoding(std::vector<uint8_t> imageData, int16_t* yCrCbChannels, std::unordered_map<int,HuffmanTree*> huffmanTrees, int width, int height) {
    std::vector<int16_t> host_yCrCbChannels(3*width*height, 0);
    Stream* imageStream = new Stream(imageData);

    int16_t* curLuminous = host_yCrCbChannels.data();
    int16_t* curChromRed = host_yCrCbChannels.data() + width * height;
    int16_t* curChromYel = host_yCrCbChannels.data() + 2 * width * height;

    int oldLumCoeff = 0, oldCbdCoeff = 0, oldCrdCoeff = 0;
    int xBlocks = width / 8;
    int yBlocks = height / 8;

    for (int y = 0; y < yBlocks; y++) {
        for (int x = 0; x < xBlocks; x++) {
            buildMCU(curLuminous, imageStream, oldLumCoeff, 0, huffmanTrees);
            buildMCU(curChromRed, imageStream, oldCbdCoeff, 1, huffmanTrees);
            buildMCU(curChromYel, imageStream, oldCrdCoeff, 1, huffmanTrees);
            curLuminous += 64;
            curChromRed += 64;
            curChromYel += 64;
        }
    }
    // for(int i = 0; i < 10; i++) {
    //     // printf("Hi\n");
    //     printf("%d %d %d\n",  host_yCrCbChannels[i],  host_yCrCbChannels[width * height + i], host_yCrCbChannels[2*width * height + i]);
    //     // printf("%d\n",  yCrCbChannels[i]);
    //     // printf("%d %d %d\n",  arr_l[i], arr_r[i], arr_y[i]);
    // }
    checkCudaError(cudaMemcpy(yCrCbChannels, host_yCrCbChannels.data(), sizeof(int16_t) * host_yCrCbChannels.size(), cudaMemcpyHostToDevice), "Failed to copy entries for the zigzag table.");
    delete imageStream;
}

__device__ void performZigzagReordering(int16_t* yCrCbChannels, int16_t* rgbChannels, uint8_t* quantTables,
                                        int blockIndex, int threadIndexInBlock, int threadId, int pixelIndex, int totalPixels, int* zigzagLocations) {
    rgbChannels[pixelIndex] = yCrCbChannels[blockIndex * 64 + zigzagLocations
[threadIndexInBlock]] * quantTables[zigzagLocations
[threadIndexInBlock]];
    rgbChannels[totalPixels + pixelIndex] = yCrCbChannels[totalPixels+blockIndex * 64 + zigzagLocations
[threadIndexInBlock]] * quantTables[64+zigzagLocations
[threadIndexInBlock]];
    rgbChannels[2*totalPixels + pixelIndex] = yCrCbChannels[2*totalPixels+blockIndex * 64 + zigzagLocations
[threadIndexInBlock]] * quantTables[64+zigzagLocations
[threadIndexInBlock]];
}

__device__ void performColorConversion(int16_t* rgbChannels, int16_t* outputChannels,
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
        float red = rgbChannels[2*totalPixels+i] * (2 - 2 * 0.299) + rgbChannels[i];
        float blue = rgbChannels[totalPixels + i] * (2 - 2 * 0.114) + rgbChannels[i];
        float green = (rgbChannels[i] - 0.114 * blue - 0.299 * red) / 0.587;

        // Clamp values to [0, 255]
        outputChannels[actualIndex] = min(max(static_cast<int16_t>(red + 128), 0), 255);
        outputChannels[totalPixels+ actualIndex] = min(max(static_cast<int16_t>(green + 128), 0), 255);
        outputChannels[2*totalPixels+actualIndex] = min(max(static_cast<int16_t>(blue + 128), 0), 255);
        // if (actualIndex == 355) {
        //     printf("the values %f, %f, %f, %d, %d, %d, \n", red, blue, green, outputChannels[actualIndex], outputChannels[totalPixels+ actualIndex], outputChannels[2*totalPixels+actualIndex]);
        // }
    }
}

__global__ void decodeKernel(int16_t* yCrCbChannels, int16_t* rgbChannels, int16_t* outputChannels, int width, int height, uint8_t* quantTables, int* zigzagLocations) {
    // int imageId = blockIdx.x;
    int threadId = threadIdx.x;
    int blockSize = blockDim.x;
    decodeImage(yCrCbChannels,
                rgbChannels,
                outputChannels,
                width,
                height,
                quantTables,
                zigzagLocations,
                threadId, blockSize);
}

__device__ void decodeImage(int16_t* yCrCbChannels, int16_t* rgbChannels, int16_t* outputChannels, int width, int height, uint8_t* quantTables, int* zigzagLocations, int threadId, int blockSize) {
    
    int totalPixels = width * height;
    __shared__ int zigzagMap[1024];

    int pixelIndex = threadId;
    while (pixelIndex < 64) {
        zigzagMap[pixelIndex] = zigzagLocations[pixelIndex];
        pixelIndex += blockSize;
    }
    __syncthreads();

    // pixelIndex = threadId;
    // if (threadId==0) {
    //     performHuffmanDecoding(imageData, yCrCbChannels, hfCodes, hfLengths, width, height);
    // }
    // __syncthreads();

    while (pixelIndex < totalPixels) {
        int threadIndexInBlock = pixelIndex % 64;
        int blockIndex = pixelIndex / 64;

        performZigzagReordering(yCrCbChannels, rgbChannels, quantTables,
                                blockIndex, threadIndexInBlock, threadId, pixelIndex, totalPixels, zigzagMap);

        // pixelIndex += blockDim.x * gridDim.x;
        pixelIndex += blockSize;
    }

    __syncthreads();

    pixelIndex = threadId;

    while (pixelIndex * 8 < totalPixels) {        
        idctRow(rgbChannels + pixelIndex * 8);
        idctRow(rgbChannels + totalPixels + pixelIndex * 8);
        idctRow(rgbChannels + 2*totalPixels + pixelIndex * 8);

        // pixelIndex += blockDim.x * gridDim.x;
        pixelIndex += blockSize;
    }

    __syncthreads();

    pixelIndex = threadId;

     while (pixelIndex * 8 < totalPixels) {
        int start = pixelIndex / 8;
        start = start * 64;
        start = start + (pixelIndex % 8);
        
        idctCol(rgbChannels + start);
        idctCol(rgbChannels + totalPixels + start);
        idctCol(rgbChannels + 2*totalPixels + start);

        // pixelIndex += blockDim.x * gridDim.x;
        pixelIndex += blockSize;
    }
    __syncthreads();

    // Iterate over pixels handled by this thread
    performColorConversion(rgbChannels, outputChannels, totalPixels, width, threadId, blockSize);
}

__global__ void batchDecodeKernel(DeviceData* deviceStructs) {
    // int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int imageId = blockIdx.x;
    int threadId = threadIdx.x;
    int blockSize = blockDim.x;
    decodeImage(deviceStructs[imageId].yCrCbChannels,
                deviceStructs[imageId].rgbChannels,
                deviceStructs[imageId].outputChannels,
                deviceStructs[imageId].width,
                deviceStructs[imageId].height,
                deviceStructs[imageId].quantTables,
                deviceStructs[imageId].zigzagLocations,
                threadId, blockSize);
}

void clean(uint8_t*& quantTables, int16_t*& yCrCbChannels, int16_t*& rgbChannels, int16_t*& outputChannels, int*& zigzagLocations, std::unordered_map<int,HuffmanTree*>& huffmanTrees) {
    // Freeing the memory
    cudaFree(quantTables);
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
    cudaMemcpy(channels.getR().data(), outputChannels, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(channels.getG().data(), outputChannels+width*height, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(channels.getB().data(), outputChannels+2*width*height, channelSize, cudaMemcpyDeviceToHost);

    // Writing the decoded channels to a file instead of displaying using opencv
    fs::path output_dir = "../testing/cudaH_output_arrays";
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
