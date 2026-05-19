#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

struct GpuFeature {
    unsigned long long n;
    unsigned int total_steps;
    unsigned int odd_steps;
    unsigned int even_steps;
    unsigned long long peak;
    unsigned int flags;
};

__device__ unsigned int ctz64(unsigned long long value) {
    return static_cast<unsigned int>(__ffsll(value) - 1);
}

__global__ void scan_kernel(unsigned long long start, unsigned long long count, unsigned int max_steps, GpuFeature *out) {
    const unsigned long long offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= count) {
        return;
    }

    const unsigned long long n = start + offset;
    unsigned long long value = n;
    unsigned long long peak = n;
    unsigned int total = 0;
    unsigned int odd = 0;
    unsigned int even = 0;
    unsigned int flags = 0;

    while (value != 1 && total < max_steps) {
        if ((value & 1ull) == 0) {
            const unsigned int run = ctz64(value);
            value >>= run;
            total += run;
            even += run;
        } else {
            if (value > (0xffffffffffffffffull - 1ull) / 3ull) {
                flags |= 2u;
                break;
            }
            value = value * 3ull + 1ull;
            total += 1;
            odd += 1;
            if (value > peak) {
                peak = value;
            }
        }
    }

    if (value == 1) {
        flags |= 1u;
    }
    if (total >= max_steps) {
        flags |= 4u;
    }

    out[offset] = {n, total, odd, even, peak, flags};
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "usage: collatz_scan_cuda START END OUTPUT_CSV\n";
        return 1;
    }

    const unsigned long long start = std::strtoull(argv[1], nullptr, 10);
    const unsigned long long end = std::strtoull(argv[2], nullptr, 10);
    const std::string output = argv[3];
    if (start == 0 || end < start) {
        std::cerr << "invalid range\n";
        return 1;
    }

    const unsigned long long count64 = end - start + 1ull;
    if (count64 > 10000000ull) {
        std::cerr << "CUDA v1 safety cap is 10,000,000 starts per process; split larger scans into chunks\n";
        return 1;
    }
    const std::size_t count = static_cast<std::size_t>(count64);

    GpuFeature *device = nullptr;
    cudaMalloc(&device, count * sizeof(GpuFeature));
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    scan_kernel<<<grid, block>>>(start, count64, 10000000u, device);
    const cudaError_t kernel_error = cudaDeviceSynchronize();
    if (kernel_error != cudaSuccess) {
        std::cerr << "CUDA kernel failed: " << cudaGetErrorString(kernel_error) << "\n";
        cudaFree(device);
        return 1;
    }

    auto *host = new GpuFeature[count];
    cudaMemcpy(host, device, count * sizeof(GpuFeature), cudaMemcpyDeviceToHost);
    cudaFree(device);

    std::ofstream out(output);
    out << "n,total_steps,odd_steps,even_steps,peak_low,flags\n";
    for (std::size_t i = 0; i < count; ++i) {
        out << host[i].n << ','
            << host[i].total_steps << ','
            << host[i].odd_steps << ','
            << host[i].even_steps << ','
            << host[i].peak << ','
            << host[i].flags << '\n';
    }

    delete[] host;
    return 0;
}
