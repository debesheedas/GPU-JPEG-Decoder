#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cuda0-implementation/src/parser.h"
#include <cuda_runtime.h>

// Benchmark function to measure JPEG decoding performance for various image sizes (CUDA)
static void BM_JPEGDecoderCUDA(benchmark::State& state) {
    // Define image sizes to benchmark (e.g., 120x120, 800x600, etc.)
    std::vector<std::string> imagePaths = {
        "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cpp-implementation/benchmark/images/1_320x240.jpg", 
        "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cpp-implementation/benchmark/images/2_400x400.jpg",
        "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cpp-implementation/benchmark/images/3_120x120.jpg", 
        "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cpp-implementation/benchmark/images/4_800x600.jpg",
        "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cpp-implementation/benchmark/images/5_200x200.jpg",
        "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cpp-implementation/benchmark/images/6_500x298.jpg",
        "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cpp-implementation/benchmark/images/7_600x607.jpg",
        "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cpp-implementation/benchmark/images/8_640x853.jpg",
        "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cpp-implementation/benchmark/images/9_887x629.jpg",
        "/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/cpp-implementation/benchmark/images/10_3264x2448.jpg"
    };

    // Dynamically select the image path based on the benchmark state
    std::string imagePath = imagePaths[state.range(0)];

    // Output file to store results
    std::ofstream resultFile("benchmark_results.txt", std::ios_base::app);
    
    // Measure the decoding time for each iteration
    for (auto _ : state) {
        // CUDA-specific logic for handling image decoding
        JPEGParser parser(imagePath); // Initialize the parser

        // Start CUDA timer for GPU-based operations
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        parser.decode(); // Run the decoding on the GPU (Make sure parser.decodeCUDA() is implemented for GPU)

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        state.SetIterationTime(milliseconds / 1000.0); // Set iteration time for benchmark

        // Save the image size and time to the result file
        resultFile << imagePath << " " << milliseconds << "\n"; // Time in milliseconds
    }

    resultFile.close();
}

// Register the benchmark for different image sizes
BENCHMARK(BM_JPEGDecoderCUDA)
    ->Unit(benchmark::kMillisecond)  // Set the time unit to milliseconds
    ->Iterations(100)  // Number of iterations for each benchmark
    ->DenseRange(0, 9, 1);  // Image size index range (0 to 9) based on imagePaths size

// Main function to run all registered benchmarks
BENCHMARK_MAIN();
