cmake_minimum_required(VERSION 3.10)
project(JPEGDecoder LANGUAGES CXX CUDA)

# Enable testing
enable_testing()

# Find CUDAToolkit explicitly
find_package(CUDAToolkit REQUIRED)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Add the CUDA source file (idct.cu) as an object library
add_library(idct_obj OBJECT ../src/idct.h)
set_target_properties(idct_obj PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(parser_obj OBJECT ../src/parser.h)
set_target_properties(parser_obj PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Explicitly set the CUDA library path (optional)
list(APPEND CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/usr/local/cuda/lib64")

# Add executables (decoder and benchmark_test)

add_executable(decoder
    ../main.cpp
    ../src/parser.cu
    ../src/huffmanTree.cpp
    ../src/idct.cu
    ../utils/stream.cpp
    ../utils/color.cpp
)

# Link the object file and CUDA runtime library to the decoder executable
target_link_libraries(decoder PRIVATE $<TARGET_OBJECTS:idct_obj> $<TARGET_OBJECTS:parser_obj> /usr/local/cuda/lib64/libcudart.so)

add_executable(benchmark_test
    benchmark.cu
    ../src/parser.cu
    ../src/huffmanTree.cpp
    ../src/idct.cu
    ../utils/stream.cpp
    ../utils/color.cpp
)

# Link the object file and CUDA runtime library to the benchmark_test executable
target_link_libraries(benchmark_test PRIVATE $<TARGET_OBJECTS:idct_obj> $<TARGET_OBJECTS:parser_obj> /usr/local/cuda/lib64/libcudart.so)

# Include CUDA include directories
target_include_directories(decoder PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(benchmark_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# target_link_libraries(decoder PRIVATE $<TARGET_OBJECTS:idct_obj> /usr/local/cuda/lib64/libcudart.so)
# target_link_libraries(benchmark_test PRIVATE $<TARGET_OBJECTS:idct_obj> /usr/local/cuda/lib64/libcudart.so)

# target_link_libraries(decoder PRIVATE src/idct.o)
target_link_libraries(decoder PRIVATE $<TARGET_OBJECTS:idct_obj>)
target_link_libraries(benchmark_test PRIVATE $<TARGET_OBJECTS:idct_obj>)
# target_link_libraries(benchmark_test PRIVATE src/idct.o)



# Link Google Benchmark library
set(BENCHMARK_ROOT "$ENV{HOME}/benchmark_install")
find_package(benchmark REQUIRED PATHS ${BENCHMARK_ROOT} NO_DEFAULT_PATH)
target_link_libraries(benchmark_test PRIVATE benchmark::benchmark)

# Register benchmark test with CTest
add_test(NAME benchmark COMMAND benchmark_test)

# Set compiler options for Release configuration
set(CMAKE_BUILD_TYPE Release)
