#include "parser.h"

JPEGParser::JPEGParser(std::string& imagePath){
    std::ifstream input(imagePath, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()));
    input.close();
    JPEGParser::extract(bytes);
}

void JPEGParser::extract(std::vector<uint8_t>& bytes) {        
    uint16_t tableSize = 0;
    uint8_t header = 0;

    // Using the Stream class for reading bytes.
    Stream* stream = new Stream(bytes);

    while (true) {
        uint16_t marker = stream->getMarker();

        if (marker == MARKERS[0]) {
            continue;
        } else if (marker == MARKERS[1]) {
            std::cout<< "Extracting Application Header" << std::endl;
            tableSize = stream->getMarker();
            stream->getNBytes(this->applicationHeader, int(tableSize - 2));
        } else if (marker == MARKERS[2]) {
            std::cout<< "Extracting Quant Tables" << std::endl;
            stream->getMarker();
            uint8_t destination = stream->getByte();
            stream->getNBytes(this->quantTables[0], 64);
            if(stream->getMarker() == MARKERS[2]) {
                stream->getMarker();
                destination = stream->getByte();
                stream->getNBytes(this->quantTables[1], 64);
            } else {
                std::cout << " Something went wrong at parsing second quant table." << std::endl;
            }
        } else if (marker == MARKERS[3]) {
            std::cout<< "Extracting Start of Frame" << std::endl;
            tableSize = stream->getMarker();
            stream->getNBytes(this->startOfFrame, (int) tableSize - 2);
            Stream* frame = new Stream(this->startOfFrame);
            int precision = frame->getByte();
            this->height = frame->getMarker();
            this->width = frame->getMarker();
        } else if (marker == MARKERS[4]) {
            std::cout<< "Extracting Huffman Tables" << std::endl;
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
            std::cout<< "Start of Scan" << std::endl;
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

void JPEGParser::buildMCU(std::vector<int>& arr, Stream* imageStream, int hf, int quant, int& oldCoeff) {
    uint8_t code = this->huffmanTrees[hf]->getCode(imageStream);
    uint16_t bits = imageStream->getNBits(code);
    int decoded = Stream::decodeNumber(code, bits);
    int dcCoeff = decoded + oldCoeff;

    
    arr[0] = dcCoeff * (int) this->quantTables[quant][0];
    int length = 1;

    while(length < 64) {
        code = this->huffmanTrees[16+hf]->getCode(imageStream);

        if(code == 0) {
            break;
        }

        // The first part of the AC key_len 
        // is the number of leading zeros
        if (code > 15) {
            length += (code >> 4);
            code = code & 0x0f;
        }

        bits = imageStream->getNBits(code);
        if (length < 64) {
            decoded = Stream::decodeNumber(code, bits);
            int val = decoded * this->quantTables[quant][length];
            arr[length] = val;
            length++;
        }
    }
    
    IDCT* idct = new IDCT(arr);
    idct->rearrangeUsingZigzag();
    idct->performIDCT();
    arr.assign(idct->base.begin(), idct->base.end());
    oldCoeff = dcCoeff;
}

void JPEGParser::decode(){
    int oldLumCoeff = 0;
    int oldCbdCoeff = 0;
    int oldCrdCoeff = 0;
    int yBlocks = this->height / 8;
    int xBlocks = this->width / 8;

    Stream* imageStream = new Stream(this->imageData);
    std::vector<std::vector<std::vector<int>>> luminous(xBlocks, std::vector<std::vector<int>>(yBlocks, std::vector<int>(64,0)));
    std::vector<std::vector<std::vector<int>>> chromRed(xBlocks, std::vector<std::vector<int>>(yBlocks, std::vector<int>(64,0)));
    std::vector<std::vector<std::vector<int>>> chromYel(xBlocks, std::vector<std::vector<int>>(yBlocks, std::vector<int>(64,0)));

    for (int y = 0; y < yBlocks; y++) {
        for (int x = 0; x < xBlocks; x++) {
            this->buildMCU(luminous[x][y], imageStream, 0, 0, oldLumCoeff);
            this->buildMCU(chromRed[x][y], imageStream, 1, 1, oldCbdCoeff);
            this->buildMCU(chromYel[x][y], imageStream, 1, 1, oldCrdCoeff);
        }
    }

    int size = this->height * this->width;

    ImageChannels* channels = new ImageChannels(size);

    for (int y = 0; y < yBlocks; y++) {
        for (int x = 0; x < xBlocks; x++) {
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int index = i * 8 + j;
                    int pixelIndex = (y * 8 + i) * this->width + (x * 8 + j);

                    channels->getY()[pixelIndex] = luminous[x][y][index];
                    channels->getCr()[pixelIndex] = chromYel[x][y][index];
                    channels->getCb()[pixelIndex] = chromRed[x][y][index];
                }
            }
        }
    }

    colorConversion(channels->getY(), channels->getCr(), channels->getCb(), channels->getR(), channels->getG(), channels->getB(), size);

    // Displaying the converted image.
    cv::Mat image(this->height, this->width, CV_8UC3);
    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            int idx = i * this->width + j;
            image.at<cv::Vec3b>(i, j) = cv::Vec3b((channels->getB()[idx]), channels->getG()[idx], channels->getR()[idx]);
        }
    }

    cv::imshow("Decoded Image", image);
    cv::waitKey(0);
    std::string outputFilename = "decoded_image.jpg";
    cv::imwrite(outputFilename, image);
    std::cout << "Image saved as " << outputFilename << std::endl;
}
