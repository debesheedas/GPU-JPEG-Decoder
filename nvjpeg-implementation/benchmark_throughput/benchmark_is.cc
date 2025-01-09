#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <iostream>

namespace fs = std::filesystem;

// Path to the nvjpeg executable
std::string path_to_decoder = "";

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

// Benchmark function template for JPEG decoding
void JPEGDecoderBenchmark(benchmark::State& state, const std::vector<std::string>& imagePaths, const std::string datasetPath) {
    std::ofstream resultFile("benchmark_results.txt", std::ios_base::app);
    double totalTime = 0.0;
    size_t imageSizes[] = {1, 10, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000};
    for(auto imageSize: imageSizes){
        size_t numImages = imageSize;
        size_t batchSize = imageSize;
        for (auto _ : state) {
            auto start_time = std::chrono::high_resolution_clock::now();
            // std::string command = path_to_decoder + " -fmt rgb -b 100 -t 32 -i " + datasetPath + " > /dev/null 2>&1";
            std::string command = path_to_decoder + " -fmt rgb -b " + std::to_string(batchSize) + " -t " + std::to_string(batchSize) + " -j 16 -i " + datasetPath + " > /dev/null 2>&1";
            int ret_code = system(command.c_str());
            if (ret_code != 0) {
                throw std::runtime_error("Command execution failed with code: " + std::to_string(ret_code));
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> decode_duration = end_time - start_time;
            totalTime += decode_duration.count();

            // Calculate throughput
            double throughput = numImages / decode_duration.count();
            double totalBytesProcessed = 0.0;
            int count = 0;
            for (const auto& path : imagePaths) {
                totalBytesProcessed += fs::file_size(path);
                count++;
                if(count==imageSize) break;
            }
            double bytesPerSecond = totalBytesProcessed / decode_duration.count();

            // Set iteration time based on FPS
            state.SetIterationTime(decode_duration.count());
            resultFile << "Batchsize: " << batchSize << ", "
                        << "Throughput: " << throughput << " images/sec, "
                        << "Bytes per second: " << bytesPerSecond / (1024 * 1024) << " MB/sec\n";
            state.counters["throughput_images_per_sec"] = throughput;
            state.counters["bytes_per_sec"] = bytesPerSecond;
        }
    }
    resultFile.close();
}

int main(int argc, char** argv) {
    std::string datasetPath = "";

    // Get all images in the dataset
    auto imagePaths = getAllImages(datasetPath);

    if (imagePaths.empty()) {
        std::cout << "No images found in the dataset directory." << std::endl;
        return 1;
    }

    // Run the benchmark on all images
    benchmark::RegisterBenchmark("BM_JPEGDecoder_AllImages", [imagePaths, datasetPath](benchmark::State& state) {
        JPEGDecoderBenchmark(state, imagePaths, datasetPath);
    })
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

    // Initialize and run the benchmark
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
