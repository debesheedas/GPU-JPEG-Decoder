cmake_minimum_required(VERSION 3.10)
project(GPUJPEGDecoder LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Source directories
set(SRC_DIR ${CMAKE_SOURCE_DIR}/../src)
set(UTILS_DIR ${SRC_DIR}/utils)

# Include directories
include_directories(${SRC_DIR} ${UTILS_DIR})

# Sources for libraries
set(LIBRARY_SOURCES
    ${SRC_DIR}/huffmanTree.cpp
    ${SRC_DIR}/idct.cu
    ${SRC_DIR}/parser.cu
    ${SRC_DIR}/parser2.cu
)

set(UTILS_SOURCES
    ${UTILS_DIR}/stream.cpp
    ${UTILS_DIR}/color.cpp
)

# Add libraries
add_library(idct_obj ${SRC_DIR}/idct.cu)
add_library(parser_obj ${SRC_DIR}/parser.cu ${SRC_DIR}/parser2.cu)
add_library(huffman_obj ${SRC_DIR}/huffmanTree.cpp)
add_library(utils_obj ${UTILS_SOURCES})

# Add decoder executable
add_executable(decoder ${SRC_DIR}/main.cpp)
target_link_libraries(decoder PRIVATE idct_obj parser_obj huffman_obj utils_obj)

# CUDA-specific properties
set_target_properties(decoder PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
