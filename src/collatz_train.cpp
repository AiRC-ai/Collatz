#include "collatz/core.hpp"

#include <iostream>
#include <string>

#if defined(COLLATZ_WITH_TORCH)
#include <torch/torch.h>
#endif

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

#if defined(COLLATZ_WITH_TORCH)
    std::cout << "collatz_train LibTorch runtime is enabled.\n";
    std::cout << "Next implementation slice: load compact feature batches, train metric heads, and write checkpoints.\n";
    std::cout << "CUDA available: " << (torch::cuda::is_available() ? "yes" : "no") << "\n";
    return 0;
#else
    std::cout << "collatz_train is compiled as a C++ placeholder.\n";
    std::cout << "Rebuild with -DCOLLATZ_ENABLE_TORCH=ON and Torch_DIR pointing at LibTorch to enable training.\n";
    std::cout << "The scanner, validator, dashboard, and control CLI do not require Python.\n";
    return 0;
#endif
}
