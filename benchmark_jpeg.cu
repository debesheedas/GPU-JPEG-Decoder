#include <benchmark/benchmark.h>
#include "decoder.h"
#include <cuda_runtime.h>

static void BM_Decoder_FullPipeline(benchmark::State& state) {
    IDCT idct;
    idct.idct_precision = 8;

    int test_zigzag[8][8] = {
        {0, 1, 5, 6, 14, 15, 27, 28},
        {2, 4, 7, 13, 16, 26, 29, 42},
        {3, 8, 12, 17, 25, 30, 41, 43},
        {9, 11, 18, 24, 31, 40, 44, 53},
        {10, 19, 23, 32, 39, 45, 52, 54},
        {20, 22, 33, 38, 46, 51, 55, 60},
        {21, 34, 37, 47, 50, 56, 59, 61},
        {35, 36, 48, 49, 57, 58, 62, 63},
    };
    
    memcpy(idct.zigzag, test_zigzag, sizeof(test_zigzag));

    for (auto _ : state) {
        idct.initialize_idct_table();
        
        idct.rearrange_using_zigzag();

        idct.perform_IDCT();

        cudaDeviceSynchronize();
    }
}
BENCHMARK(BM_Decoder_FullPipeline);

BENCHMARK_MAIN();
