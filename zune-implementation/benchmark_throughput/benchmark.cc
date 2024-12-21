#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <iostream>

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

// Benchmark function template for JPEG decoding using the executable
void JPEGDecoderBenchmark(benchmark::State& state, const std::vector<std::string>& imagePaths) {
    std::ofstream resultFile("benchmark_results.txt", std::ios_base::app);
    size_t numImages = imagePaths.size();
    double totalTime = 0.0;

    for (auto _ : state) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Process all images in the current benchmark iteration
        for (size_t i = 0; i < numImages; ++i) {
            std::string imagePath = imagePaths[i];
            std::string command = "../../zune -i" + imagePath;
            int exit_code = std::system(command.c_str());
            if (exit_code != 0) {
                state.SkipWithError("Executable failed to run.");
                return;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> decode_duration = end_time - start_time;
        totalTime += decode_duration.count();  // Total time for the current iteration
        
        // Calculate throughput
        double throughput = numImages / decode_duration.count();  // images per second
        double totalBytesProcessed = 0.0;
        for (const auto& path : imagePaths) {
            totalBytesProcessed += fs::file_size(path);  // Calculate total bytes processed
        }
        double bytesPerSecond = totalBytesProcessed / decode_duration.count();  // bytes per second

        // Set iteration time based on FPS
        state.SetIterationTime(decode_duration.count());
        resultFile << "Total Throughput for " << numImages << " images: " << throughput << " images/sec, "
                   << "Bytes per second: " << bytesPerSecond / (1024 * 1024) << " MB/sec\n";  // MB/sec for readability
        state.counters["throughput_images_per_sec"] = throughput;
        state.counters["bytes_per_sec"] = bytesPerSecond;
    }

    resultFile.close();
}

int main(int argc, char** argv) {
    std::string datasetPath = "/home/dphpc2024_jpeg_1/cfernand/GPU-JPEG-Decoder/cudaO-implementation/benchmark_thoughput/benchmarking_dataset_through";
    
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
