cmake_minimum_required(VERSION 3.10)
project(JPEGDecoderBenchmark)

# Enable testing
enable_testing()

# Set the path to the locally installed Google Benchmark library
set(BENCHMARK_ROOT "$ENV{HOME}/benchmark_install")

# Find the Benchmark library in the specified path
find_package(benchmark REQUIRED PATHS ${BENCHMARK_ROOT} NO_DEFAULT_PATH)

# Add the benchmark executable for benchmarking
add_executable(benchmark_test benchmark.cc)

# Link the benchmark library to the benchmark test
target_link_libraries(benchmark_test benchmark::benchmark)

# Register the benchmark test with CTest
add_test(NAME benchmark COMMAND benchmark_test)

# Set compiler options for Release configuration if necessary
set(CMAKE_BUILD_TYPE Release)
