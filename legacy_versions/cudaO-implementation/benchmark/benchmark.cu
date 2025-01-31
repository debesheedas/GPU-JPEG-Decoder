#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
#include "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cudaO-implementation/src/parser.h"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

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

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        nvtxRangePush("Full");

        uint16_t* hfCodes; 
        int* hfLengths;
        uint8_t* quantTables;
        int16_t* yCrCbChannels;
        int16_t* rgbChannels;
        int16_t* outputChannels;
        int* zigzagLocations;

        uint8_t* imageData;
        int width = 0;
        int height = 0;
        std::unordered_map<int,HuffmanTree*> huffmanTrees;

        // Extracting the byte chunks
        extract(imagePath, quantTables, imageData, width, height, huffmanTrees);
        // Allocating memory for the arrays
        allocate(hfCodes, hfLengths, huffmanTrees, yCrCbChannels, rgbChannels, outputChannels, width, height, zigzagLocations);

        cudaEventRecord(start);
        decodeKernel<<<1, 32>>>(imageData, yCrCbChannels, rgbChannels, outputChannels, width, height, quantTables, hfCodes, hfLengths, zigzagLocations);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        nvtxRangePop();

        write(outputChannels, width, height, filename);
        clean(hfCodes, hfLengths, quantTables, yCrCbChannels, rgbChannels, outputChannels, zigzagLocations, imageData, huffmanTrees);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        state.SetIterationTime(milliseconds / 1000.0); // Set iteration time for benchmark
        resultFile << imagePath << " " << milliseconds << "\n"; // Time in milliseconds
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