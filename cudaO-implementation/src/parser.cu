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
    bitOffset += code_length;
    uint16_t bits = getNBits(imageData, bitOffset, code);

    int decoded = decodeNumber(code, bits); 
    int16_t dcCoeff = decoded + oldCoeff;

    outBuffer[0] = dcCoeff;

    if (dcCoeff > 32767) {
        printf("dc coeff is larger \n");
    }

    if (decoded > 32767) {
        printf("decoded is larger \n");
    }

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
    }
}

__global__ void decodeKernel(uint8_t* imageData, int16_t* yCrCbChannels, int16_t* rgbChannels, int16_t* outputChannels, int width, int height, uint8_t* quantTables, uint16_t* hfCodes, int* hfLengths, int* zigzagLocations) {
    int imageId = blockIdx.x;
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

__device__ void decodeImage(uint8_t* imageData, int16_t* yCrCbChannels, int16_t* rgbChannels, int16_t* outputChannels, int width, int height, uint8_t* quantTables, uint16_t* hfCodes, int* hfLengths, int* zigzagLocations, int threadId, int blockSize) {

    __shared__ int zigzagMap[256];

    // int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    zigzagMap[threadId] = zigzagLocations[threadId];

    __syncthreads();

    int pixelIndex = threadId;
    int totalPixels = width * height;
    if (threadId==0) {
        performHuffmanDecoding(imageData, yCrCbChannels, hfCodes, hfLengths, width, height);
    }
    __syncthreads();

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

// __global__ void batchDecodeKernel(DeviceData* deviceStructs) {
//     // int globalId = blockIdx.x * blockDim.x + threadIdx.x;
//     int imageId = blockIdx.x;
//     int threadId = threadIdx.x;
//     int blockSize = blockDim.x;
//     decodeImage(deviceStructs[imageId].imageData,
//                 deviceStructs[imageId].yCrCbChannels,
//                 deviceStructs[imageId].rgbChannels,
//                 deviceStructs[imageId].outputChannels,
//                 deviceStructs[imageId].width,
//                 deviceStructs[imageId].height,
//                 deviceStructs[imageId].quantTables,
//                 deviceStructs[imageId].hfCodes,
//                 deviceStructs[imageId].hfLengths,
//                 deviceStructs[imageId].zigzagLocations,
//                 threadId, blockSize);
// }

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
    for(const auto& [key, item]: huffmanTrees) {
        delete item;
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
