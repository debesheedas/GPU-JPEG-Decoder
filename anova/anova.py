import numpy as np
import scipy.stats as stats

def read_throughput(filename):
    with open(filename, 'r') as file:
        throughput_data = []
        for line in file:
            if "Bytes per second:" in line:  
                bytes_per_second = line.split("Bytes per second:")[1].strip().split()[0]
                throughput_data.append(float(bytes_per_second))
        return throughput_data


cuda_decoder = read_throughput('/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/anova/cuda_throughput.txt')
nvjpeg = read_throughput('/home/dphpc2024_jpeg_1/GPU-JPEG-Decoder/anova/nvjpeg_throughput.txt')

all_data = np.concatenate([cuda_decoder, nvjpeg])

n_cuda = len(cuda_decoder)
n_nvjpeg = len(nvjpeg)
n_total = n_cuda + n_nvjpeg

mean_cuda = np.mean(cuda_decoder)
mean_nvjpeg = np.mean(nvjpeg)
mean_total = np.mean(all_data)

ssb = n_cuda * (mean_cuda - mean_total)**2 + n_nvjpeg * (mean_nvjpeg - mean_total)**2

ssw_cuda = np.sum((np.array(cuda_decoder) - mean_cuda)**2)
ssw_nvjpeg = np.sum((np.array(nvjpeg) - mean_nvjpeg)**2)
ssw = ssw_cuda + ssw_nvjpeg

# Degrees of freedom
dfb = 2 - 1  # Between-group degrees of freedom
dfw = n_total - 2  # Within-group degrees of freedom

# Mean squares
msb = ssb / dfb
msw = ssw / dfw

# F-statistic
f_stat = msb / msw

# p-value
p_value = 1 - stats.f.cdf(f_stat, dfb, dfw)

# Print results
print("CUDA-Decoder Mean Throughput:", mean_cuda)
print("NVJPEG Mean Throughput:", mean_nvjpeg)
print("Overall Mean Throughput:", mean_total)
print("F-statistic:", f_stat)
print("p-value:", p_value)

# Interpretation
if p_value < 0.05:
    print("\nResult: The difference in throughput is statistically significant (reject H0).")
    if mean_cuda > mean_nvjpeg:
        print("CUDA-Decoder has significantly better throughput.")
    else:
        print("NVJPEG has significantly better throughput.")
else:
    print("\nResult: The difference in throughput is not statistically significant (fail to reject H0).")
