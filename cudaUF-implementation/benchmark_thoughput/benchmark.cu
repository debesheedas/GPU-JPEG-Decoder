#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cudaUF-implementation/src/parser.h"

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
void JPEGDecoderBenchmark(benchmark::State& state, std::vector<std::string> imagePaths, size_t batchSize) {
    size_t numImages = imagePaths.size();
    std::ofstream resultFile("benchmark_results.txt", std::ios_base::app);

    // Define batch size (adjust based on available memory)
    // size_t batchSize =  512; // Example batch size
    size_t numBatches = (numImages + batchSize - 1) / batchSize;
    std::cout<< "num batches" << numBatches << "numImages" << numImages << std::endl;

    for (auto _ : state) {
        float totalKernelTime = 0.0f; // Total time across all batches
        JPEGParser** jpegParsers = new JPEGParser*[batchSize];
        JPEGParserData* structs = new JPEGParserData[batchSize];
        JPEGParserData* deviceStructs;
        cudaMalloc(&deviceStructs, batchSize * sizeof(JPEGParserData));
        // std::cout<<"Allocate Complete"<<std::endl;

        for (size_t batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
            size_t startIdx = batchIdx * batchSize;
            size_t endIdx = std::min(startIdx + batchSize, numImages);
            size_t currentBatchSize = endIdx - startIdx;

            // Allocate JPEGParser objects and structs for the current batch
            // JPEGParser** jpegParsers = new JPEGParser*[currentBatchSize];
            // JPEGParserData* structs = new JPEGParserData[currentBatchSize];
            // std::cout << "debug1" <<std::endl;
            for (size_t i = 0; i < currentBatchSize; ++i) {
                size_t globalIdx = startIdx + i;
                // std::cout << "debug2" <<std::endl;
                jpegParsers[i] = new JPEGParser(imagePaths[globalIdx]);
                // std::cout << "debug3" <<std::endl;
                jpegParsers[i]->extract();
                // std::cout << "debug4" <<std::endl;
                structs[i] = copyToStruct(jpegParsers[i]);
                // std::cout << "debug5" <<std::endl;
            }
            // std::cout << "debug6" <<std::endl;
            // Allocate memory for the current batch on the GPU
            cudaMemcpy(deviceStructs, structs, currentBatchSize * sizeof(JPEGParserData), cudaMemcpyHostToDevice);
            // std::cout<<"Copy"<<std::endl;
            // Measure kernel execution time
            cudaEvent_t batchStart, batchStop;
            cudaEventCreate(&batchStart);
            cudaEventCreate(&batchStop);

            cudaEventRecord(batchStart);
            processImagesKernel<<<currentBatchSize, 1>>>(deviceStructs, currentBatchSize);
            cudaEventRecord(batchStop);
            cudaEventSynchronize(batchStop);

            // Calculate time for this batch
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, batchStart, batchStop);
            totalKernelTime += milliseconds;

            // Cleanup for this batch
            for (size_t i = 0; i < currentBatchSize; ++i) {
                // if (structs[i].luminous) std::cout<<"lim" << std::endl;
                if (structs[i].luminous) cudaFree(structs[i].luminous);
                if (structs[i].chromRed) cudaFree(structs[i].chromRed);
                if (structs[i].chromYel) cudaFree(structs[i].chromYel);
                if (structs[i].hf0codes) cudaFree(structs[i].hf0codes);
                if (structs[i].hf1codes) cudaFree(structs[i].hf1codes);
                if (structs[i].hf16codes) cudaFree(structs[i].hf16codes);
                if (structs[i].hf17codes) cudaFree(structs[i].hf17codes);
                if (structs[i].hf0lengths) cudaFree(structs[i].hf0lengths);
                if (structs[i].hf1lengths) cudaFree(structs[i].hf1lengths);
                if (structs[i].hf16lengths) cudaFree(structs[i].hf16lengths);
                if (structs[i].hf17lengths) cudaFree(structs[i].hf17lengths);
                // if (structs[i].redOutput) std::cout<<"red" << std::endl;
                if (structs[i].redOutput) cudaFree(structs[i].redOutput);
                if (structs[i].greenOutput) cudaFree(structs[i].greenOutput);
                if (structs[i].blueOutput) cudaFree(structs[i].blueOutput);
            }
            
            cudaEventDestroy(batchStart);
            cudaEventDestroy(batchStop);
            for (size_t i = 0; i < currentBatchSize; ++i) {
                // jpegParsers[i]->~JPEGParser();  // Call the destructor explicitly
                delete jpegParsers[i];
            }
            // delete[] jpegParsers;
            // delete[] structs;
            // std::cout<<"Free"<<std::endl;
        }
        cudaFree(deviceStructs);
        delete[] jpegParsers;
        delete[] structs;
        // Convert total kernel time to seconds
        double seconds = totalKernelTime / 1000.0;

        // Calculate throughput
        double throughput = numImages / seconds; // Images per second
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
}

// int main(int argc, char** argv) {
//     std::string datasetPath = "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/benchmarking_dataset_mini";

//     std::vector<std::string> imagePaths = getAllImages(datasetPath);

//     if (imagePaths.empty()) {
//         std::cout << "No images found in the dataset directory." << std::endl;
//         return 1;
//     }

//     benchmark::RegisterBenchmark("BM_JPEGDecoder_Throughput", [imagePaths](benchmark::State& state) {
//         JPEGDecoderBenchmark(state, imagePaths);
//     })
//     ->Unit(benchmark::kMillisecond)
//     ->Iterations(10);

//     benchmark::Initialize(&argc, argv);
//     benchmark::RunSpecifiedBenchmarks();
//     return 0;
// }

int main(int argc, char** argv) {
    // std::vector<std::string> datasetPaths = {
    //     "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/benchmarking_dataset_mini",
    //     "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/benchmarking_dataset_old"
    // };
    std::vector<std::string> datasetPaths = {
        "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/benchmarking_dataset_mini",
    };
    std::vector<size_t> batchSizes = {256, 512};
    //std::vector<size_t> batchSizes = {16, 32, 64, 128, 256, 512};


    double bestThroughput = 0.0;
    std::string bestDatasetPath;
    size_t bestBatchSize = 0;

    for (const auto& datasetPath : datasetPaths) {
        std::vector<std::string> imagePaths = getAllImages(datasetPath);

        if (imagePaths.empty()) {
            std::cout << "No images found in dataset: " << datasetPath << std::endl;
            continue;
        }

        for (const auto& batchSize : batchSizes) {
            std::cout << "Testing batchSize: " << batchSize
                      << " with dataset: " << datasetPath << std::endl;

            benchmark::RegisterBenchmark("BM_JPEGDecoder_Throughput",
                [&imagePaths, batchSize](benchmark::State& state) {
                    // Benchmarking function
                    JPEGDecoderBenchmark(state, imagePaths, batchSize);
                    
                    // Retrieve throughput from counters within the benchmarking function
                    double throughput = state.counters["throughput_images_per_sec"].value;
                    state.SetLabel("Throughput: " + std::to_string(throughput));
                })
                ->Unit(benchmark::kMillisecond)
                ->Iterations(1);

            benchmark::RunSpecifiedBenchmarks();

            // After benchmarking, you can directly use the state in the benchmark function
            // So no need to access it outside of the benchmark loop
        }
    }

    std::cout << "Best throughput: " << bestThroughput
              << " images/sec with batchSize: " << bestBatchSize
              << " and dataset: " << bestDatasetPath << std::endl;

    return 0;
}