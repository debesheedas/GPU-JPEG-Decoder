# Compiler for CUDA code
NVCC = nvcc
NVCCFLAGS = --fmad=false --prec-div=true --prec-sqrt=true

# Compiler for C++ code
CXX = g++
CXXFLAGS = -std=c++17 -I/usr/local/cuda/include  # Add CUDA include path

# CUDA library path
LDFLAGS = -L/usr/local/cuda/lib64

# Sources
CU_SOURCES = src/parser.cu src/idct.cu
CPP_SOURCES = main.cpp src/huffmanTree.cpp utils/stream.cpp utils/color.cpp
OBJECTS = $(CU_SOURCES:.cu=.o) $(CPP_SOURCES:.cpp=.o)

# Output binary
TARGET = decoder

# Default target
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJECTS)

# Compile CUDA files (.cu)
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C++ files (.cpp)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean target
clean:
	rm -f $(OBJECTS) $(TARGET)
