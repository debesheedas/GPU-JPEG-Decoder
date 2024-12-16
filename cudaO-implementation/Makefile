decoder: src/parser.cu src/huffmanTree.cpp utils/stream.cpp utils/utils.cu
	/usr/local/cuda/bin/nvcc --relocatable-device-code true -o decoder src/parser.cu src/huffmanTree.cpp utils/stream.cpp utils/utils.cu

clean:
	rm -f decoder