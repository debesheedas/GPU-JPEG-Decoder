#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
#include "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cpp-implementation/src/parser.h"

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
    
    for (auto _ : state) {
        // reading of jpeg here
        JPEGParser parser(imagePath);
        auto start_time = std::chrono::high_resolution_clock::now();
        parser.extract();
        parser.decode();
        auto end_time = std::chrono::high_resolution_clock::now();
        parser.write();
        // writing of output file here
        std::chrono::duration<double> decode_duration = end_time - start_time;
        state.SetIterationTime(decode_duration.count());
        resultFile << imagePath << " " << decode_duration.count() * 1000 << "\n";  // time in ms
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
    std::string datasetPath = "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/datasets/benchmarking_dataset";
    RegisterBenchmarks(datasetPath);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
