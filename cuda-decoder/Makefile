decoder: main.cu src/parser.cu src/huffmanTree.cpp utils/stream.cpp utils/utils.cu
	/usr/local/cuda/bin/nvcc --relocatable-device-code true -o decoder main.cu src/parser.cu src/huffmanTree.cpp utils/stream.cpp utils/utils.cu

clean:
	rm -f decoder

# -----------------------------------------------------------------------
# MAKEFILE FOR ENABLING LINEWISE PROFILING

# CUDA_PATH := /usr/local/cuda
# NVCC := $(CUDA_PATH)/bin/nvcc
# CXXFLAGS := -O3 -lineinfo -rdc=true -lnvToolsExt

# SRC := main.cu src/parser.cu src/huffmanTree.cpp utils/stream.cpp utils/utils.cu
# TARGET := decoder

# all: $(TARGET)

# $(TARGET): $(SRC)
# 	$(NVCC) $(CXXFLAGS) -o $(TARGET) $(SRC) -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64

# clean:
# 	rm -f $(TARGET)
