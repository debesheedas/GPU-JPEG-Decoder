rm -f CMakeCache.txt
rm -rf CMakeFiles/
rm -rf build
mkdir build
cd build
cmake ..
cmake --build . --config Release

# Nsight Compute Profiling
echo "Running benchmark with Nsight Compute profiling..."

# Set TMPDIR to avoid lock file issues
export TMPDIR="/home/$USER/ncu_tmp"
mkdir -p "$TMPDIR"

# Run Nsight Compute with the benchmark executable
echo "Running benchmark with Nsight Compute profiling..."
TMPDIR="$TMPDIR" /usr/local/cuda/bin/ncu --set full --nvtx --target-processes all \
    -f --export ../report_throughput_bench.ncu-rep ./benchmark_test --benchmark_out=benchmark_results.json --benchmark_out_format=json

if [ $? -eq 0 ]; then
    echo "Profiling complete. Results saved to 'report_throughput_bench.ncu-rep'."
else
    echo "Profiling failed."
    exit 1
fi