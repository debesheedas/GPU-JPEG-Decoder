#!/bin/bash

# List of batch sizes and datasets
BATCH_SIZES=(32 64 128 256 512 1024)  # Example batch sizes
DATASETS=("/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/benchmarking_dataset_mini/400x400")  # Example dataset paths

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Loop through each combination of batch size and dataset
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
  for DATASET_PATH in "${DATASETS[@]}"; do
    echo "------------------------------------------------"
    echo "Building with batch size: $BATCH_SIZE and dataset: $DATASET_PATH"

    # Run cmake with the current batch size and dataset path
    cmake .. -DBATCH_SIZE=${BATCH_SIZE} -DDATASET_PATH="${DATASET_PATH}"

    # Build the project
    cmake --build . --config Release

    # Run the benchmark test with the current parameters
    echo "Running benchmark for batch size ${BATCH_SIZE} and dataset ${DATASET_PATH}..."
    ./benchmark_test --benchmark_out="benchmark_results_${BATCH_SIZE}_${DATASET_PATH//\//_}.json" --benchmark_out_format=json --batchsize ${BATCH_SIZE} --datasetpath "${DATASET_PATH}"

    echo "Benchmark completed for batch size ${BATCH_SIZE} and dataset ${DATASET_PATH}. Results are saved."
    echo "------------------------------------------------"
  done
done

echo "All benchmarks completed."
