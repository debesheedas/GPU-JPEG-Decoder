#include "IDCT.h"
#include <cmath>

void IDCT::initialize_idct_table(){
    idct_table = new float*[idct_precision];
    
    for(int i=0; i<idct_precision; i++){
        idct_table[i] = new float[idct_precision];
    }

    for(int u=0; u<idct_precision; u++){
        for(int x=0; x<idct_precision; x++){
            float normCoeff = (u == 0) ? (1.0f / sqrtf(2)) : 1.0f;
            idct_table[u][x] = normCoeff * cosf(((2.0f * x + 1.0f) * u * M_PI) / 16.0f);
        }
    }
}

int* IDCT::rearrange_using_zigzag(){
    for(int x=0; x<8; x++){
        for(int y=0; y<8; y++){
            zigzag[x][y] = base[zigzag[x][y]];
        }
    }
    return &zigzag[0][0];
}

std::vector<int8_t> IDCT::perform_IDCT(){
    float** out = new float*[8];
    for(int i=0; i<8; i++){
        out[i] = new float[8];
    }

    for(int x=0; x<8; x++){
        for(int y=0; y<8; y++){
            float local_sum = 0;
            for(int u=0; u<idct_precision; u++){
                for(int v=0; v<idct_precision; v++){
                    local_sum += zigzag[v][u] * idct_table[u][x] * idct_table[v][y];
                }
            }
            //Is the order of the variables correct here?
            out[y][x] = local_sum / 4.0f;
        }
    }
    
    for(int i=0; i<8; i++){
        for(int j=0; j<8; j++){
            float value = out[i][j];
            int8_t convertedValue = static_cast<int8_t>(std::round(value));
            base[i*8+j] = convertedValue;
        }
    }

    return base;
}