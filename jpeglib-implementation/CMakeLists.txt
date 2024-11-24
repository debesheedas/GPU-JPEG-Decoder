cmake_minimum_required(VERSION 3.10)
project(JPEGDecoder)

# Enable testing
enable_testing()

# Set the path to the locally installed libjpeg library
set(LIBJPEG_INCLUDE_DIR "$ENV{HOME}/libjpeg_install/install/include")
set(LIBJPEG_LIB_DIR "$ENV{HOME}/libjpeg_install/install/lib")

# Add the source files for the decoder executable
add_executable(decoder
    main.cpp
    src/parser.cpp
    utils/stream.cpp
    utils/color.cpp
)

# Include the libjpeg headers
target_include_directories(decoder PRIVATE ${LIBJPEG_INCLUDE_DIR})

# Link the libjpeg library
target_link_libraries(decoder PRIVATE ${LIBJPEG_LIB_DIR}/libjpeg.so)

# Set the path to the locally installed Google Benchmark library
set(BENCHMARK_ROOT "$ENV{HOME}/benchmark_install")

# Find the Benchmark library in the specified path
find_package(benchmark REQUIRED PATHS ${BENCHMARK_ROOT} NO_DEFAULT_PATH)

# Add the benchmark test executable and its source files
add_executable(benchmark_test
    benchmark/benchmark.cc
    src/parser.cpp
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
