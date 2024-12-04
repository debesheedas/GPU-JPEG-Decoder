 #include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <thread>
#include <iostream>
#include <cmath>

namespace fs = std::filesystem;

const std::string path_to_decoder = "/home/dphpc2024_jpeg_1/cfernand/GPU-JPEG-Decoder/jpeglib-implementation/libjpeg_install/build/djpeg";

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

// Benchmark function for throughput measurement
void JPEGDecoderThroughput(benchmark::State& state, const std::vector<std::string>& imagePaths, int batchSize) {
    const int totalImages = imagePaths.size();
    std::ofstream resultFile("benchmark_results_throughput.csv", std::ios_base::app);

    for (auto _ : state) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Process images in batches
        int completedImages = 0;
        while (completedImages < totalImages) {
            int currentBatchSize = std::min(batchSize, totalImages - completedImages);

            std::vector<std::thread> threads;
            for (int i = 0; i < currentBatchSize; ++i) {
                threads.emplace_back([&, i]() {
                    std::string command = path_to_decoder + " " + imagePaths[completedImages + i];
                    int ret_code = system(command.c_str());
                    if (ret_code != 0) {
                        std::cerr << "Command execution failed for " << imagePaths[completedImages + i]
                                  << " with code: " << ret_code << '\n';
                    }
                });
            }

            // Wait for all threads in the current batch to complete
            for (auto& thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }

            completedImages += currentBatchSize;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> decode_duration = end_time - start_time;

        // Calculate throughput
        double total_time_sec = decode_duration.count();
        double throughput = totalImages / total_time_sec;

        // Log results
        resultFile << "Batch Size: " << batchSize
                   << ", Total Images: " << totalImages
                   << ", Total Time: " << total_time_sec << " s"
                   << ", Throughput: " << throughput << " images/sec\n";

        // Report throughput to benchmark framework
        state.counters["Throughput (images/sec)"] = throughput;
    }

    resultFile.close();
}

// Register benchmarks for different image sizes and batch sizes
void RegisterThroughputBenchmarks(const std::string& datasetPath) {
    const std::vector<int> imageSizes = {200};//, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000};
    const std::vector<int> batchSizes = {10};//, 20, 40, 80, 100}; // Batch sizes to test

    for (int size : imageSizes) {
        auto imagePaths = getImagesBySize(datasetPath, size);

        if (!imagePaths.empty()) {
            for (int batchSize : batchSizes) {
                std::string benchmarkName = "BM_Throughput_" + std::to_string(size) + "_Batch" + std::to_string(batchSize);

                benchmark::RegisterBenchmark(benchmarkName.c_str(), [imagePaths, batchSize](benchmark::State& state) {
                    JPEGDecoderThroughput(state, imagePaths, batchSize);
                })
                ->Unit(benchmark::kSecond) // Set unit to seconds for throughput calculation
                ->Iterations(10);
            }
        }
    }
}

int main(int argc, char** argv) {
    std::ofstream resultFile("benchmark_results_throughput.csv", std::ios_base::trunc);
    resultFile << "Batch Size, Total Images, Total Time (s), Throughput (images/sec)\n"; // Add header
    resultFile.close();

    std::string datasetPath = "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/benchmarking_dataset";
    RegisterThroughputBenchmarks(datasetPath);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}