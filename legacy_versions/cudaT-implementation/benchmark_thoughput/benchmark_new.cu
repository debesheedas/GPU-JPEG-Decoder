#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cudaO-implementation/src/parser.h"

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


JPEGParserData copyToStruct(JPEGParser* parser) {
    JPEGParserData data;

    // Copy the pointers and scalar values
    data.imageData = parser->imageData;
    data.luminous = parser->luminous;
    data.chromRed = parser->chromRed;
    data.chromYel = parser->chromYel;
    data.zigzag_l = parser->zigzag_l;
    data.zigzag_r = parser->zigzag_r;
    data.zigzag_y = parser->zigzag_y;
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

// __global__ void processImagesKernel(JPEGParserData* deviceStructs, int numImages) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < numImages) {
//         // Process each image buffer (dummy operation, replace with decoding logic)
//         // jpegParsers[idx].decode();
//         // myKernel<<<1, 256>>>(idx);
//         dim3 blockSize(8, 8);
//         dim3 gridSize((deviceStructs[idx].width + blockSize.x - 1) / blockSize.x, (deviceStructs[idx].height + blockSize.y - 1) / blockSize.y);

//         decodeKernel<<<1,1024>>>(deviceStructs[idx].imageData, 
//                                                 deviceStructs[idx].luminous, 
//                                                 deviceStructs[idx].chromRed, 
//                                                 deviceStructs[idx].chromYel, 
//                                                 deviceStructs[idx].zigzag_l,
//                                                 deviceStructs[idx].zigzag_r,
//                                                 deviceStructs[idx].zigzag_y, 
//                                                 deviceStructs[idx].idctTable, 
//                                                 8, 8,  
//                                                 deviceStructs[idx].width, 
//                                                 deviceStructs[idx].height, 
//                                                 deviceStructs[idx].xBlocks, 
//                                                 deviceStructs[idx].yBlocks, 
//                                                 deviceStructs[idx].redOutput, 
//                                                 deviceStructs[idx].greenOutput, 
//                                                 deviceStructs[idx].blueOutput,
//                                                 deviceStructs[idx].quantTable1, 
//                                                 deviceStructs[idx].quantTable2, 
//                                                 deviceStructs[idx].hf0codes, 
//                                                 deviceStructs[idx].hf1codes, 
//                                                 deviceStructs[idx].hf16codes, 
//                                                 deviceStructs[idx].hf17codes, 
//                                                 deviceStructs[idx].hf0lengths, 
//                                                 deviceStructs[idx].hf1lengths, 
//                                                 deviceStructs[idx].hf16lengths, 
//                                                 deviceStructs[idx].hf17lengths
//                                                 );
//     }
// }

// Benchmark function for throughput measurement
void JPEGDecoderBenchmark(benchmark::State& state, std::vector<std::string> imagePaths, size_t batchSize) {
    size_t numImages = imagePaths.size();
    std::ofstream resultFile("benchmark_results.txt", std::ios_base::app);

    // Define batch size (adjust based on available memory)
    size_t batchSize =  400; // Example batch size
    int threads = 64; 
    size_t numBatches = (numImages + batchSize - 1) / batchSize;
    std::cout<< "num batches " << numBatches << " | numImages " << numImages << std::endl;

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
            // processImagesKernel<<<currentBatchSize, 1>>>(deviceStructs, currentBatchSize);
            batchDecodeKernel<<<currentBatchSize,threads>>>(deviceStructs);
            cudaEventRecord(batchStop);
            cudaEventSynchronize(batchStop);
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
            }

            // Calculate time for this batch
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, batchStart, batchStop);
            totalKernelTime += milliseconds;

            // Temporarily write output
            // for (size_t i = 0; i < currentBatchSize; ++i) {
            //     // size_t globalIdx = startIdx + i;
            //     // std::cout << "debug2" <<std::endl;
            //     jpegParsers[i]->write();
            // }

            // Cleanup for this batch
            for (size_t i = 0; i < currentBatchSize; ++i) {
                // if (structs[i].luminous) std::cout<<"lim" << std::endl;
                if (structs[i].luminous) cudaFree(structs[i].luminous);
                if (structs[i].chromRed) cudaFree(structs[i].chromRed);
                if (structs[i].chromYel) cudaFree(structs[i].chromYel);
                if (structs[i].zigzag_l) cudaFree(structs[i].zigzag_l);
                if (structs[i].zigzag_r) cudaFree(structs[i].zigzag_r);
                if (structs[i].zigzag_y) cudaFree(structs[i].zigzag_y);
                if (structs[i].hf0codes) cudaFree(structs[i].hf0codes);
                if (structs[i].hf1codes) cudaFree(structs[i].hf1codes);
                if (structs[i].hf16codes) cudaFree(structs[i].hf16codes);
                if (structs[i].hf17codes) cudaFree(structs[i].hf17codes);
                if (structs[i].hf0lengths) cudaFree(structs[i].hf0lengths);
                if (structs[i].hf1lengths) cudaFree(structs[i].hf1lengths);
                if (structs[i].hf16lengths) cudaFree(structs[i].hf16lengths);
                if (structs[i].hf17lengths) cudaFree(structs[i].hf17lengths);
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

int main(int argc, char** argv) {
    // Default values
    size_t batchSize = 512;
    std::string datasetPath = "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/benchmarking_dataset_mini";

    // Process command-line arguments
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--batchsize" && i + 1 < argc) {
            batchSize = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--datasetpath" && i + 1 < argc) {
            datasetPath = argv[++i];
        }
    }

    std::vector<std::string> imagePaths = getAllImages(datasetPath);

    if (imagePaths.empty()) {
        std::cout << "No images found in the dataset directory." << std::endl;
        return 1;
    }

    benchmark::RegisterBenchmark("BM_JPEGDecoder_Throughput", [imagePaths, batchSize](benchmark::State& state) {
        JPEGDecoderBenchmark(state, imagePaths, batchSize);
    })
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}