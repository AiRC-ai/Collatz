# Build Notes

The default configuration builds the portable CPU path and does not require
CUDA or LibTorch:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Run validation with:

```sh
ctest --test-dir build --output-on-failure
```

The classical scanner, source alignment, evidence publisher, and dashboard are
C++/shell. Python is used for lightweight research validation and
evidence/schema checks.

Optional targets can be enabled explicitly:

```sh
cmake -S . -B build-cuda -DCOLLATZ_ENABLE_CUDA=ON
cmake -S . -B build-torch -DCOLLATZ_ENABLE_TORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
```

The `Makefile` remains a convenience wrapper around the default CMake build and
test commands.
