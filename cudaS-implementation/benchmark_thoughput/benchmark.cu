#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cudaS-implementation/src/parser.h"

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


// void copyToDevice(HostData* parser, DeviceData* data) {
//     data->imageData = parser->imageData;
//     data->yCrCbChannels = parser->yCrCbChannels;
//     data->outputChannels = parser->outputChannels;
//     data->zigzagLocations = parser->zigzagLocations;
//     data->width = parser->width;
//     data->height = parser->height;
//     data->rgbChannels = parser->rgbChannels;
//     data->quantTables = parser->quantTables;
//     data->hfCodes = parser->hfCodes;
//     data->hfLengths = parser->hfLengths;
// }

// Benchmark function for throughput measurement
void JPEGDecoderBenchmark(benchmark::State& state, std::vector<std::string> imagePaths) {
    size_t numImages = imagePaths.size();
    std::ofstream resultFile("benchmark_results.txt", std::ios_base::app);

    // Define batch size (adjust based on available memory)
    size_t batchSize = 1000; // Example batch size
    int threads = 32;
    size_t numBatches = (numImages + batchSize - 1) / batchSize;
    std::cout<< "num batches " << numBatches << " | numImages " << numImages << std::endl;
    
    for (auto _ : state) {
        float totalKernelTime = 0.0f; // Total time across all batches
        
        DeviceData structs[batchSize];
        DeviceData* deviceStructs;
        cudaMalloc(&deviceStructs, batchSize * sizeof(DeviceData));
        // std::cout<<"Allocate Complete"<<std::endl;
        
        for (size_t batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
            HostData hosts[batchSize];
            size_t startIdx = batchIdx * batchSize;
            size_t endIdx = std::min(startIdx + batchSize, numImages);
            size_t currentBatchSize = endIdx - startIdx;

            for (size_t i = 0; i < currentBatchSize; ++i) {
                
                size_t globalIdx = startIdx + i;
                HostData* host_data = &hosts[i];
                DeviceData* data = &structs[i];
        
                host_data->imagePath = imagePaths[globalIdx];
                extract(host_data->imagePath, data->quantTables, data->imageData, data->width, data->height, host_data->huffmanTrees);
                allocate(data->hfCodes, data->hfLengths, host_data->huffmanTrees, data->yCrCbChannels, data->rgbChannels, data->outputChannels, data->width, data->height, data->zigzagLocations);
            }
            // Allocate memory for the current batch on the GPU
            cudaMemcpy(deviceStructs, structs, currentBatchSize * sizeof(DeviceData), cudaMemcpyHostToDevice);
            cudaEvent_t batchStart, batchStop;
            cudaEventCreate(&batchStart);
            cudaEventCreate(&batchStop);

            //nvtxRangePushA("BatchDecodeKernel Execution");
            cudaEventRecord(batchStart);
            batchDecodeKernel<<<currentBatchSize, threads>>>(deviceStructs);
            cudaEventRecord(batchStop);
            //nvtxRangePop();  // End NVTX marker
            
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
            cudaEventDestroy(batchStart);
            cudaEventDestroy(batchStop);
            // Cleanup for this batch
            for (size_t i = 0; i < currentBatchSize; ++i) {
                HostData* host_data = &hosts[i];
                DeviceData* data = &structs[i];
                // std::cout << host_data->huffmanTrees[0]->codes[0] << std::endl;
                clean(data->hfCodes, data->hfLengths, data->quantTables, data->yCrCbChannels, data->rgbChannels, data->outputChannels, data->zigzagLocations, data->imageData, host_data->huffmanTrees);
            }
        }
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
        if (deviceStructs) cudaFree(deviceStructs);

        // Log results
        resultFile << "Throughput: " << throughput << " images/sec, "
                   << "Bytes per second: " << bytesPerSecond / (1024 * 1024) << " MB/sec\n";
    }
    
    resultFile.close();
}

int main(int argc, char** argv) {
    std::string datasetPath = "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/benchmarking_dataset_through";

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
