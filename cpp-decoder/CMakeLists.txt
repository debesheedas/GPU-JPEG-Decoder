cmake_minimum_required(VERSION 3.10)
project(JPEGDecoder)

# Enable testing
enable_testing()

# Add source files for the decoder executable
add_executable(decoder
    main.cpp
    src/parser.cpp
    src/huffmanTree.cpp
    src/idct.cpp
    utils/stream.cpp
    utils/color.cpp
)

# Set the path to the locally installed Google Benchmark library
set(BENCHMARK_ROOT "$ENV{HOME}/benchmark_install")

# Find the Benchmark library in the specified path
find_package(benchmark REQUIRED PATHS ${BENCHMARK_ROOT} NO_DEFAULT_PATH)

# Add the benchmark test executable and its source files
add_executable(benchmark_test
    benchmark/benchmark.cc
    src/parser.cpp
    src/huffmanTree.cpp
    src/idct.cpp
    utils/stream.cpp
    utils/color.cpp
)

# Add the src directory to the include path for benchmark_test
target_include_directories(benchmark_test PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Link the benchmark library to the benchmark test
target_link_libraries(benchmark_test benchmark::benchmark)

# Register the benchmark test with CTest
add_test(NAME benchmark COMMAND benchmark_test)

# Set compiler options for Release configuration if necessary
set(CMAKE_BUILD_TYPE Release)
