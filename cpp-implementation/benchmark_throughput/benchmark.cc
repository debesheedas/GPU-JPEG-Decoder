#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
#include "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cpp-implementation/src/parser.h"

namespace fs = std::filesystem;

// Function to get all images with a specified size
std::vector<std::string> getAllImages(const std::string& datasetPath) {
    std::vector<std::string> imagePaths;

    for (const auto& entry : fs::recursive_directory_iterator(datasetPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpeg") {
            imagePaths.push_back(entry.path().string());
        }
    }
    return imagePaths;
}

// Benchmark function template for JPEG decoding
void JPEGDecoderBenchmark(benchmark::State& state, const std::vector<std::string>& imagePaths) {
    std::ofstream resultFile("benchmark_results.txt", std::ios_base::app);
    size_t numImages = imagePaths.size();

    for (auto _ : state) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Decode all images in the current iteration
        for (const std::string& imagePath : imagePaths) {
            std::string mutablePath = imagePath;
            JPEGParser parser(mutablePath);
            parser.extract();
            parser.decode();
            parser.write();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;

        // Calculate throughput metrics
        double imagesPerSecond = numImages / duration.count();
        double totalBytesProcessed = 0.0;
        for (const auto& path : imagePaths) {
            totalBytesProcessed += fs::file_size(path);
        }
        double bytesPerSecond = totalBytesProcessed / duration.count();

        // Log the results in the required format
        resultFile << "Throughput: " << imagesPerSecond << " images/sec, Bytes per second: "
                   << bytesPerSecond / (1024 * 1024) << " MB/sec\n";

        state.SetIterationTime(duration.count());
        state.counters["throughput_images_per_sec"] = imagesPerSecond;
        state.counters["bytes_per_sec"] = bytesPerSecond;
    }

    resultFile.close();
}


int main(int argc, char** argv) {
    std::string datasetPath = "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/benchmarking_dataset_mini";
    
    // Get all images in the dataset
    auto imagePaths = getAllImages(datasetPath);

    if (imagePaths.empty()) {
        std::cout << "No images found in the dataset directory." << std::endl;
        return 1;
    }

    // Run the benchmark on all images
    benchmark::RegisterBenchmark("BM_JPEGDecoder_AllImages", [imagePaths](benchmark::State& state) {
        JPEGDecoderBenchmark(state, imagePaths);
    })
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

    // Initialize and run the benchmark
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
