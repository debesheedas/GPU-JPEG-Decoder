rm -f CMakeCache.txt
rm -rf CMakeFiles/
rm -rf build
mkdir build
cd build
cmake ..
cmake --build . --config Release
./benchmark_test