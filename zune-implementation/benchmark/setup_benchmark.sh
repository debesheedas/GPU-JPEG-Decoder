#!/bin/bash

# Install Google Benchmark to a specified directory
mkdir -p ~/benchmark_install
cd ~
git clone https://github.com/google/benchmark.git
cd benchmark
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=~/benchmark_install ..
cmake --install .

echo "Benchmark library installed to ~/benchmark_install"
