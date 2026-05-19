cmake_minimum_required(VERSION 3.20)

project(3xN1 VERSION 0.1.0 LANGUAGES CXX)

option(COLLATZ_ENABLE_CUDA "Build the CUDA scanner" OFF)
option(COLLATZ_ENABLE_TORCH "Build collatz_train with LibTorch" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(collatz_core
    src/collatz_core.cpp
    src/collatz_feature_io.cpp
    src/collatz_ml.cpp
)
target_include_directories(collatz_core PUBLIC include)
target_compile_options(collatz_core PRIVATE
    $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-Wall -Wextra>
)

add_executable(collatz_plot src/collatz_plot.cpp)
target_link_libraries(collatz_plot PRIVATE collatz_core)

add_executable(collatz_scan_cpu src/collatz_scan_cpu.cpp)
target_link_libraries(collatz_scan_cpu PRIVATE collatz_core)

add_executable(collatz_validate_sources src/collatz_validate_sources.cpp)
target_link_libraries(collatz_validate_sources PRIVATE collatz_core)

add_executable(collatz_web src/collatz_web.cpp)
target_link_libraries(collatz_web PRIVATE collatz_core)

add_executable(collatzctl src/collatzctl.cpp)
target_link_libraries(collatzctl PRIVATE collatz_core)

add_executable(collatz_train src/collatz_train.cpp)
target_link_libraries(collatz_train PRIVATE collatz_core)

add_executable(collatz_embed_export src/collatz_embed_export.cpp)
target_link_libraries(collatz_embed_export PRIVATE collatz_core)

add_executable(collatz_path_image src/collatz_path_image.cpp)
target_link_libraries(collatz_path_image PRIVATE collatz_core)

add_executable(collatz_select_representatives src/collatz_select_representatives.cpp)
target_link_libraries(collatz_select_representatives PRIVATE collatz_core)

add_executable(collatz_stratified_sample src/collatz_stratified_sample.cpp)
target_link_libraries(collatz_stratified_sample PRIVATE collatz_core)

add_executable(collatz_graph_export src/collatz_graph_export.cpp)
target_link_libraries(collatz_graph_export PRIVATE collatz_core)

add_executable(collatz_embedding_analyze src/collatz_embedding_analyze.cpp)
target_link_libraries(collatz_embedding_analyze PRIVATE collatz_core)

add_executable(collatz_neighborhood_analyze src/collatz_neighborhood_analyze.cpp)
target_link_libraries(collatz_neighborhood_analyze PRIVATE collatz_core)

add_executable(collatz_insight_analyze src/collatz_insight_analyze.cpp)
target_link_libraries(collatz_insight_analyze PRIVATE collatz_core)

add_executable(collatz_hypothesis_analyze src/collatz_hypothesis_analyze.cpp)
target_link_libraries(collatz_hypothesis_analyze PRIVATE collatz_core)

add_executable(collatz_source_align src/collatz_source_align.cpp)
target_link_libraries(collatz_source_align PRIVATE collatz_core)

add_executable(collatz_source_targets src/collatz_source_targets.cpp)
target_link_libraries(collatz_source_targets PRIVATE collatz_core)

add_executable(collatz_selftest tests/collatz_selftest.cpp)
target_link_libraries(collatz_selftest PRIVATE collatz_core)

enable_testing()
add_test(NAME collatz_selftest COMMAND collatz_selftest)

if(COLLATZ_ENABLE_TORCH)
    find_package(Torch REQUIRED)
    target_compile_definitions(collatz_train PRIVATE COLLATZ_WITH_TORCH=1)
    target_link_libraries(collatz_train PRIVATE "${TORCH_LIBRARIES}")
endif()

if(COLLATZ_ENABLE_CUDA)
    enable_language(CUDA)
    add_executable(collatz_scan_cuda src/collatz_scan_cuda.cu)
    set_target_properties(collatz_scan_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_include_directories(collatz_scan_cuda PRIVATE include)
endif()
