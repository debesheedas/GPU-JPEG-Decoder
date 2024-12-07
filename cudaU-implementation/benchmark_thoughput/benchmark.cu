#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cudaU-implementation/src/parser.h"

namespace fs = std::filesystem;

// Function to get all images in the dataset
std::vector<std::string> getAllImages(const std::string& datasetPath) {
    std::vector<std::string> imagePaths;
    for (const auto& entry : fs::recursive_directory_iterator(datasetPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpeg") {
            imagePaths.push_back(entry.path().string());
        }
    }
    return imagePaths;
}

// __global__ void myKernel(int size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) {
//         idx = idx; // Example operation
//     }
// }
// CUDA kernel for parallel image processing (dummy example, replace with actual implementation)

struct JPEGParserData {
    unsigned char* imageData;   // Pointer to image data
    int* luminous;            // Pointer to luminance data
    int* chromRed;            // Pointer to chroma red data
    int* chromYel;            // Pointer to chroma yellow data            // Pointer to chroma yellow data
    double* idctTable;           // Pointer to IDCT table
    int idctWidth;              // Width of IDCT (e.g., 8)
    int idctHeight;             // Height of IDCT (e.g., 8)
    int width;                  // Image width
    int height;                 // Image height
    int xBlocks;                // Number of horizontal blocks
    int yBlocks;                // Number of vertical blocks
    int* redOutput;           // Pointer to red channel output
    int* greenOutput;         // Pointer to green channel output
    int* blueOutput;          // Pointer to blue channel output
    uint8_t* quantTable1;         // Pointer to first quantization table
    uint8_t* quantTable2;         // Pointer to second quantization table
    uint16_t* hf0codes;    // Huffman table 0 codes
    uint16_t* hf1codes;    // Huffman table 1 codes
    uint16_t* hf16codes;   // Huffman table 16 codes
    uint16_t* hf17codes;   // Huffman table 17 codes
    int* hf0lengths;            // Huffman table 0 lengths
    int* hf1lengths;            // Huffman table 1 lengths
    int* hf16lengths;           // Huffman table 16 lengths
    int* hf17lengths;           // Huffman table 17 lengths
};

JPEGParserData copyToStruct(JPEGParser* parser) {
    JPEGParserData data;

    // Copy the pointers and scalar values
    data.imageData = parser->imageData;
    data.luminous = parser->luminous;
    data.chromRed = parser->chromRed;
    data.chromYel = parser->chromYel;
    data.idctTable = parser->idctTable;
    data.idctWidth = 8;  // Fixed width
    data.idctHeight = 8; // Fixed height
    data.width = parser->width;
    data.height = parser->height;
    data.xBlocks = parser->xBlocks;
    data.yBlocks = parser->yBlocks;
    data.redOutput = parser->redOutput;
    data.greenOutput = parser->greenOutput;
    data.blueOutput = parser->blueOutput;
    data.quantTable1 = parser->quantTable1;
    data.quantTable2 = parser->quantTable2;
    data.hf0codes = parser->hf0codes;
    data.hf1codes = parser->hf1codes;
    data.hf16codes = parser->hf16codes;
    data.hf17codes = parser->hf17codes;
    data.hf0lengths = parser->hf0lengths;
    data.hf1lengths = parser->hf1lengths;
    data.hf16lengths = parser->hf16lengths;
    data.hf17lengths = parser->hf17lengths;


    return data;
}

__global__ void processImagesKernel(JPEGParserData* deviceStructs, int numImages) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numImages) {
        // Process each image buffer (dummy operation, replace with decoding logic)
        // jpegParsers[idx].decode();
        // myKernel<<<1, 256>>>(idx);
        dim3 blockSize(8, 8);
        dim3 gridSize((deviceStructs[idx].width + blockSize.x - 1) / blockSize.x, (deviceStructs[idx].height + blockSize.y - 1) / blockSize.y);

        decodeKernel<<<gridSize, blockSize>>>(deviceStructs[idx].imageData, 
                                                deviceStructs[idx].luminous, 
                                                deviceStructs[idx].chromRed, 
                                                deviceStructs[idx].chromYel, 
                                                deviceStructs[idx].idctTable, 
                                                8, 8,  
                                                deviceStructs[idx].width, 
                                                deviceStructs[idx].height, 
                                                deviceStructs[idx].xBlocks, 
                                                deviceStructs[idx].yBlocks, 
                                                deviceStructs[idx].redOutput, 
                                                deviceStructs[idx].greenOutput, 
                                                deviceStructs[idx].blueOutput,
                                                deviceStructs[idx].quantTable1, 
                                                deviceStructs[idx].quantTable2, 
                                                deviceStructs[idx].hf0codes, 
                                                deviceStructs[idx].hf1codes, 
                                                deviceStructs[idx].hf16codes, 
                                                deviceStructs[idx].hf17codes, 
                                                deviceStructs[idx].hf0lengths, 
                                                deviceStructs[idx].hf1lengths, 
                                                deviceStructs[idx].hf16lengths, 
                                                deviceStructs[idx].hf17lengths
                                                );
    }
}

// Benchmark function for throughput measurement
void JPEGDecoderBenchmark(benchmark::State& state, std::vector<std::string> imagePaths) {

    size_t numImages = imagePaths.size();
    std::ofstream resultFile("benchmark_results.txt", std::ios_base::app);

    JPEGParser** jpegParsers = new JPEGParser*[numImages];
    JPEGParserData* structs = new JPEGParserData[numImages];

    // Create a new JPEGParser object for each image and store the pointer
    for (size_t i = 0; i < numImages; ++i) {
        jpegParsers[i] = new JPEGParser(imagePaths[i]);
        JPEGParser& parser = *jpegParsers[i];
        parser.extract();
        structs[i] = copyToStruct(jpegParsers[i]);
    }
    JPEGParserData* deviceStructs;
    cudaMalloc(&deviceStructs, numImages * sizeof(JPEGParserData));
    cudaMemcpy(deviceStructs, structs, numImages * sizeof(JPEGParserData), cudaMemcpyHostToDevice);

    for (auto _ : state) {
        // Start timer
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Launch kernel
        int threadsPerBlock = 1;
        int numBlocks = numImages;
        // processImagesKernel<<<numBlocks, threadsPerBlock>>>(jpegParsers, numImages);
        processImagesKernel<<<numBlocks, threadsPerBlock>>>(deviceStructs, numImages);
        cudaDeviceSynchronize();

        // Stop timer
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Calculate throughput
        double seconds = milliseconds / 1000.0;
        double throughput = numImages / seconds; // images per second
        double totalBytesProcessed = 0.0;
        for (const auto& path : imagePaths) {
            totalBytesProcessed += fs::file_size(path);  // Calculate total bytes processed
        }
        double bytesPerSecond = totalBytesProcessed / seconds; // bytes per second

        // Set iteration metrics
        state.SetIterationTime(seconds);
        state.counters["throughput_images_per_sec"] = throughput;
        state.counters["bytes_per_sec"] = bytesPerSecond;

        // Log results
        resultFile << "Throughput: " << throughput << " images/sec, "
                   << "Bytes per second: " << bytesPerSecond / (1024 * 1024) << " MB/sec\n";
    }
    resultFile.close();
    delete[] structs;
    // for (size_t i = 0; i < numImages; ++i) {
    //     delete jpegParsers[i]; // Delete each JPEGParser object
    // }
    // delete[] jpegParsers;
    cudaFree(deviceStructs);
}

int main(int argc, char** argv) {
    std::string datasetPath = "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/benchmarking_dataset";

    std::vector<std::string> imagePaths = getAllImages(datasetPath);

    if (imagePaths.empty()) {
        std::cout << "No images found in the dataset directory." << std::endl;
        return 1;
    }

    benchmark::RegisterBenchmark("BM_JPEGDecoder_Throughput", [imagePaths](benchmark::State& state) {
        JPEGDecoderBenchmark(state, imagePaths);
    })
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
