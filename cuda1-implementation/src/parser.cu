#include "parser.h"

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
    
    quantTables[0] = new uint8_t[64];
    quantTables[1] = new uint8_t[64];

    int blockSize = 64;
    int gridSize = (64 + blockSize - 1) / blockSize;
    cudaMalloc((void**)&idctTable, 64 * sizeof(double));
    initializeIDCTTableKernel<<<blockSize, gridSize>>>(idctTable, 64);

    //cudaMalloc((void**)&quantTables[0], 64 * sizeof(uint8_t));
    //cudaMalloc((void**)&quantTables[1], 64 * sizeof(uint8_t));
    // JPEGParser::extract(bytes);
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
            // std::cout<< "Extracting Application Header" << std::endl;
            tableSize = stream->getMarker();
            stream->getNBytes(this->applicationHeader, int(tableSize - 2));
        } else if (marker == MARKERS[2]) {
            // std::cout<< "Extracting Quant Tables" << std::endl;
            stream->getMarker();
            uint8_t destination = stream->getByte();
            stream->getNBytes(quantTables[0], 64);
            //std::cout << " got the goods **** " << std::endl;
            ///cudaMemcpy(quantTables[0], temp, 64 * sizeof(uint8_t), cudaMemcpyHostToDevice);
            if(stream->getMarker() == MARKERS[2]) {
                stream->getMarker();
                destination = stream->getByte();
                stream->getNBytes(quantTables[1], 64);
                //std::cout << " got the goods " << std::endl;
                //cudaMemcpy(quantTables[1], temp, 64 * sizeof(uint8_t), cudaMemcpyHostToDevice);
            } else {
                std::cout << " Something went wrong at parsing second quant table." << std::endl;
            }
        } else if (marker == MARKERS[3]) {
            // std::cout<< "Extracting Start of Frame" << std::endl;
            tableSize = stream->getMarker();
            stream->getNBytes(this->startOfFrame, (int) tableSize - 2);
            Stream* frame = new Stream(this->startOfFrame);
            int precision = frame->getByte();
            this->height = frame->getMarker();
            this->width = frame->getMarker();
        } else if (marker == MARKERS[4]) {
            // std::cout<< "Extracting Huffman Tables" << std::endl;
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
            // std::cout<< "Start of Scan" << std::endl;
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

void JPEGParser::buildMCU(int* arr, Stream* imageStream, int hf, int quant, int& oldCoeff, int validWidth = 8, int validHeight = 8) {
    std::vector<int> hostBuffer(64,0);
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

    // Create and process the IDCT for this block with the valid dimensions
    cudaMemcpy(arr, hostBuffer.data(), 64*sizeof(int), cudaMemcpyHostToDevice);
    IDCT* idct = new IDCT(arr, idctTable);
    idct->rearrangeUsingZigzag(validWidth, validHeight);
    idct->performIDCT(validWidth, validHeight);

    // Update oldCoeff for the next MCU
    oldCoeff = dcCoeff;

    delete idct;
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
    std::vector<std::vector<std::vector<int>>> luminous(xBlocks, std::vector<std::vector<int>>(yBlocks, std::vector<int>(64,0)));
    std::vector<std::vector<std::vector<int>>> chromRed(xBlocks, std::vector<std::vector<int>>(yBlocks, std::vector<int>(64,0)));
    std::vector<std::vector<std::vector<int>>> chromYel(xBlocks, std::vector<std::vector<int>>(yBlocks, std::vector<int>(64,0)));

    int* temp;
    cudaMalloc((void**)&temp, 64 * sizeof(int));

    for (int y = 0; y < yBlocks; y++) {
        for (int x = 0; x < xBlocks; x++) {
            // Determine the valid width and height for this block to account for padding
            int blockWidth = (x == xBlocks - 1 && paddedWidth != this->width) ? this->width % 8 : 8;
            int blockHeight = (y == yBlocks - 1 && paddedHeight != this->height) ? this->height % 8 : 8;

            this->buildMCU(temp, imageStream, 0, 0, oldLumCoeff, blockWidth, blockHeight);
            cudaMemcpy(luminous[x][y].data(), temp, 64 * sizeof(int), cudaMemcpyDeviceToHost);
            this->buildMCU(temp, imageStream, 1, 1, oldCbdCoeff, blockWidth, blockHeight);
            cudaMemcpy(chromRed[x][y].data(), temp, 64 * sizeof(int), cudaMemcpyDeviceToHost);
            this->buildMCU(temp, imageStream, 1, 1, oldCrdCoeff, blockWidth, blockHeight);
            cudaMemcpy(chromYel[x][y].data(), temp, 64 * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }

    this->channels = new ImageChannels(this->height * this->width);

    // Write the processed data into the channels, ignoring padded regions
    for (int y = 0; y < yBlocks; y++) {
        for (int x = 0; x < xBlocks; x++) {
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int pixelY = y * 8 + i;
                    int pixelX = x * 8 + j;

                    if (pixelY < this->height && pixelX < this->width) {
                        int index = i * 8 + j;
                        int pixelIndex = pixelY * this->width + pixelX;

                        this->channels->getY()[pixelIndex] = luminous[x][y][index];
                        this->channels->getCr()[pixelIndex] = chromYel[x][y][index];
                        this->channels->getCb()[pixelIndex] = chromRed[x][y][index];
                    }
                }
            }
        }
    }

    // Convert YCbCr channels to RGB
    colorConversion(this->channels->getY(), this->channels->getCr(), this->channels->getCb(), this->channels->getR(), this->channels->getG(), this->channels->getB(), this->height * this->width);

}

void JPEGParser::write() {
    // Writing the decoded channels to a file instead of displaying using opencv
    fs::path output_dir = "../testing/cuda1_output_arrays"; // Change the directory name here for future CUDA implementations
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