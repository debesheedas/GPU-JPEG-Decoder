#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
#include "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cudaT-implementation/src/parser.h"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdexcept>
#include <unordered_map>

namespace fs = std::filesystem;

// Function to get all images with a specified size
std::vector<std::string> getImagesBySize(const std::string& datasetPath, int size) {
    std::string folderPath = datasetPath + "/" + std::to_string(size) + "x" + std::to_string(size);
    std::vector<std::string> imagePaths;

    for (const auto& entry : fs::recursive_directory_iterator(folderPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpeg") {
            imagePaths.push_back(entry.path().string());
        }
    }
    return imagePaths;
}

// Benchmark function template for JPEG decoding
void JPEGDecoderBenchmark(benchmark::State& state, const std::vector<std::string>& imagePaths) {
    std::string imagePath = imagePaths[state.range(0)];
    std::ofstream resultFile("benchmark_results.txt", std::ios_base::app);
    fs::path file_path(imagePath);
    std::string filename = file_path.filename().string();
    
    for (auto _ : state) {
        // Start CUDA timer for GPU-based operations
        cudaEvent_t start, stop;
        cudaError_t err;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        nvtxRangePush("Full");

        // Declare pointers for memory
        uint16_t* hfCodes = nullptr; 
        int* hfLengths = nullptr;
        uint8_t* quantTables = nullptr;
        int16_t* yCrCbChannels = nullptr;
        int16_t* rgbChannels = nullptr;
        int16_t* outputChannels = nullptr;
        int* zigzagLocations = nullptr;

        uint8_t* imageData = nullptr;
        int width = 0;
        int height = 0;
        std::unordered_map<int, HuffmanTree*> huffmanTrees;

        printf("hello_0\n");
        // Extracting the byte chunks
        try {
            extract(imagePath, quantTables, imageData, width, height, huffmanTrees);
            if (width <= 0 || height <= 0) {
                throw std::runtime_error("Invalid image dimensions.");
            }
            printf("hi_1\n");
        } catch (const std::exception& e) {
            fprintf(stderr, "Extraction error: %s\n", e.what());
            return;
        }

        // Allocating memory for the arrays
        try {
            allocate(hfCodes, hfLengths, huffmanTrees, yCrCbChannels, rgbChannels, outputChannels, width, height, zigzagLocations);
            if (!hfCodes || !hfLengths || !quantTables || !yCrCbChannels || !rgbChannels || !outputChannels || !zigzagLocations) {
                throw std::runtime_error("Memory allocation failed for one or more pointers.");
            }
            printf("hi_2\n");
        } catch (const std::exception& e) {
            fprintf(stderr, "Allocation error: %s\n", e.what());
            return;
        }

        cudaEventRecord(start);

        // Launch the kernel for decoding
        decodeKernel<<<1, 1024>>>(imageData, yCrCbChannels, rgbChannels, outputChannels, width, height, quantTables, hfCodes, hfLengths, zigzagLocations);

        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
            clean(hfCodes, hfLengths, quantTables, yCrCbChannels, rgbChannels, outputChannels, zigzagLocations, imageData, huffmanTrees);
            return;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        nvtxRangePop();

        // Write the decoded output (optional debugging step)
        try {
            write(outputChannels, width, height, filename);
        } catch (const std::exception& e) {
            fprintf(stderr, "Write error: %s\n", e.what());
        }

        // Clean up GPU memory
        try {
            clean(hfCodes, hfLengths, quantTables, yCrCbChannels, rgbChannels, outputChannels, zigzagLocations, imageData, huffmanTrees);
        } catch (const std::exception& e) {
            fprintf(stderr, "Cleanup error: %s\n", e.what());
        }

        // Calculate time for this iteration
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Record the time taken for this benchmark iteration
        state.SetIterationTime(milliseconds / 1000.0);
        resultFile << imagePath << " " << milliseconds << " ms\n";

        // Debugging info
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error during iteration: %s\n", cudaGetErrorString(err));
        }
    }
    
    resultFile.close();
}

// Register benchmarks for different image sizes
void RegisterBenchmarks(const std::string& datasetPath) {
    const std::vector<int> imageSizes = {200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000};
    
    for (int size : imageSizes) {
        auto imagePaths = getImagesBySize(datasetPath, size);
        
        if (!imagePaths.empty()) {
            std::string benchmarkName = "BM_JPEGDecoder_" + std::to_string(size);
            
            benchmark::RegisterBenchmark(benchmarkName.c_str(), [imagePaths](benchmark::State& state) {
                JPEGDecoderBenchmark(state, imagePaths);
            })
            ->Unit(benchmark::kMillisecond)
            ->Iterations(10)
            ->DenseRange(0, imagePaths.size() - 1, 1);
        }
    }
}

int main(int argc, char** argv) {
    std::string datasetPath = "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/benchmarking_dataset_mini";
    RegisterBenchmarks(datasetPath);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
