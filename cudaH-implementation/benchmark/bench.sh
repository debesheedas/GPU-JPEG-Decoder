export CUDACXX=/usr/local/cuda/bin/nvcc
rm -rf build
mkdir build
cd build
cmake ..
cmake --build . --config Release
