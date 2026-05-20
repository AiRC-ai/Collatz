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

add_executable(collatz_family_labels src/collatz_family_labels.cpp)
target_link_libraries(collatz_family_labels PRIVATE collatz_core)

add_executable(collatz_path_image src/collatz_path_image.cpp)
target_link_libraries(collatz_path_image PRIVATE collatz_core)

add_executable(collatz_path_image_atlas src/collatz_path_image_atlas.cpp)
target_link_libraries(collatz_path_image_atlas PRIVATE collatz_core)

add_executable(collatz_select_representatives src/collatz_select_representatives.cpp)
target_link_libraries(collatz_select_representatives PRIVATE collatz_core)

add_executable(collatz_stratified_sample src/collatz_stratified_sample.cpp)
target_link_libraries(collatz_stratified_sample PRIVATE collatz_core)

add_executable(collatz_full_audit src/collatz_full_audit.cpp)
target_link_libraries(collatz_full_audit PRIVATE collatz_core)

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

add_executable(collatz_evidence_publish src/collatz_evidence_publish.cpp)
target_link_libraries(collatz_evidence_publish PRIVATE collatz_core)

add_executable(collatz_selftest tests/collatz_selftest.cpp)
target_link_libraries(collatz_selftest PRIVATE collatz_core)

enable_testing()
add_test(NAME collatz_selftest COMMAND collatz_selftest)

add_test(NAME collatz_full_audit_fixture_scan
    COMMAND collatz_scan_cpu
        --start 1
        --end 100
        --output "${CMAKE_BINARY_DIR}/full_audit_fixture.bin"
        --metadata "${CMAKE_BINARY_DIR}/full_audit_fixture.metadata.json"
        --progress "${CMAKE_BINARY_DIR}/full_audit_fixture.progress.jsonl"
        --format bin
        --threads 1
        --chunk-size 25
)

add_test(NAME collatz_full_audit_fixture
    COMMAND collatz_full_audit
        --input "${CMAKE_BINARY_DIR}/full_audit_fixture.bin"
        --output "${CMAKE_BINARY_DIR}/full_audit_fixture.summary.json"
        --range-bands 4
        --top-count 4
)
set_tests_properties(collatz_full_audit_fixture PROPERTIES DEPENDS collatz_full_audit_fixture_scan)

set(COLLATZ_SOURCE_FIXTURES "${CMAKE_SOURCE_DIR}/tests/fixtures/source_records")
add_test(NAME collatz_source_targets_fixtures
    COMMAND collatz_source_targets
        --oeis-stopping "${COLLATZ_SOURCE_FIXTURES}/b006577_fixture.txt"
        --oeis-path-records "${COLLATZ_SOURCE_FIXTURES}/b006884_fixture.txt"
        --roosendaal-path-records "${COLLATZ_SOURCE_FIXTURES}/roosendaal_path_records_fixture.html"
        --roosendaal-delay-records "${COLLATZ_SOURCE_FIXTURES}/roosendaal_delay_records_fixture.html"
        --barina-path-records "${COLLATZ_SOURCE_FIXTURES}/barina_path_records_fixture.html"
        --oliveira-max-excursion-records "${COLLATZ_SOURCE_FIXTURES}/oliveira_max_excursion_fixture.txt"
        --oliveira-stopping-records "${COLLATZ_SOURCE_FIXTURES}/oliveira_stopping_fixture.txt"
        --output "${CMAKE_BINARY_DIR}/source_targets_fixture.csv"
        --metadata "${CMAKE_BINARY_DIR}/source_targets_fixture.metadata.json"
        --max-n 100000000
        --stopping-limit 10
        --path-record-limit 10
        --generic-record-limit 10
)

add_test(NAME collatz_source_align_extended_fixture
    COMMAND collatz_source_align
        --projection "${COLLATZ_SOURCE_FIXTURES}/projection_fixture.csv"
        --source-samples "${CMAKE_BINARY_DIR}/source_targets_fixture.csv"
        --output-dir "${CMAKE_BINARY_DIR}/source_alignment_extended_fixture"
        --neighbors 3
)
set_tests_properties(collatz_source_align_extended_fixture PROPERTIES DEPENDS collatz_source_targets_fixtures)

add_test(NAME collatz_source_align_old_header_fixture
    COMMAND collatz_source_align
        --projection "${COLLATZ_SOURCE_FIXTURES}/projection_fixture.csv"
        --source-samples "${COLLATZ_SOURCE_FIXTURES}/source_targets_old_header_fixture.csv"
        --output-dir "${CMAKE_BINARY_DIR}/source_alignment_old_header_fixture"
        --neighbors 3
)

add_test(NAME collatz_hypothesis_multi_source_fixture
    COMMAND collatz_hypothesis_analyze
        --validation-metrics "${COLLATZ_SOURCE_FIXTURES}/validation_metrics_fixture.json"
        --source-alignment "${COLLATZ_SOURCE_FIXTURES}/source_alignment_multi_fixture.json"
        --output-dir "${CMAKE_BINARY_DIR}/hypothesis_multi_source_fixture"
)

add_test(NAME collatz_source_align_taxonomy_fixture
    COMMAND collatz_source_align
        --projection "${COLLATZ_SOURCE_FIXTURES}/projection_fixture.csv"
        --source-samples "${COLLATZ_SOURCE_FIXTURES}/source_alignment_taxonomy_targets.csv"
        --feature-bin "${CMAKE_BINARY_DIR}/full_audit_fixture.bin"
        --output-dir "${CMAKE_BINARY_DIR}/source_alignment_taxonomy_fixture"
        --neighbors 3
)
set_tests_properties(collatz_source_align_taxonomy_fixture PROPERTIES DEPENDS collatz_full_audit_fixture_scan)

add_test(NAME collatz_embed_export_safe_fixture
    COMMAND collatz_embed_export
        --input "${CMAKE_BINARY_DIR}/full_audit_fixture.bin"
        --sample-file "${COLLATZ_SOURCE_FIXTURES}/sample_starts_fixture.csv"
        --output-dir "${CMAKE_BINARY_DIR}/ml_safe_fixture"
        --sketch-dims 16
        --metric-mode safe
)
set_tests_properties(collatz_embed_export_safe_fixture PROPERTIES DEPENDS collatz_full_audit_fixture_scan)

add_test(NAME collatz_family_labels_fixture
    COMMAND collatz_family_labels
        --input "${CMAKE_BINARY_DIR}/full_audit_fixture.bin"
        --sample-file "${COLLATZ_SOURCE_FIXTURES}/sample_starts_fixture.csv"
        --source-targets "${COLLATZ_SOURCE_FIXTURES}/source_targets_old_header_fixture.csv"
        --output "${CMAKE_BINARY_DIR}/ml_labels_fixture/families.csv"
        --metadata "${CMAKE_BINARY_DIR}/ml_labels_fixture/metadata.json"
        --max-steps 1000
        --tail-cap 32
)
set_tests_properties(collatz_family_labels_fixture PROPERTIES DEPENDS collatz_full_audit_fixture_scan)

add_test(NAME collatz_pair_sampler_fixture
    COMMAND python3 "${CMAKE_SOURCE_DIR}/research/pair_sampler.py"
        --labels "${CMAKE_BINARY_DIR}/ml_labels_fixture/families.csv"
        --metrics "${CMAKE_BINARY_DIR}/ml_safe_fixture/metrics_safe.csv"
        --sample "${COLLATZ_SOURCE_FIXTURES}/sample_starts_fixture.csv"
        --output-dir "${CMAKE_BINARY_DIR}/ml_pairs_fixture"
        --hard-negatives 1
        --max-pairs-per-type 16
)
set_tests_properties(collatz_pair_sampler_fixture PROPERTIES DEPENDS "collatz_embed_export_safe_fixture;collatz_family_labels_fixture")

add_test(NAME collatz_path_image_tensor_fixture
    COMMAND collatz_path_image
        --sample-file "${COLLATZ_SOURCE_FIXTURES}/sample_starts_fixture.csv"
        --format tensor
        --output-dir "${CMAKE_BINARY_DIR}/image_tensor_fixture"
        --size 8
        --count 6
        --max-steps 1000
)

add_test(NAME collatz_graph_export_membership_fixture
    COMMAND collatz_graph_export
        --starts-file "${COLLATZ_SOURCE_FIXTURES}/sample_starts_fixture.csv"
        --output-dir "${CMAKE_BINARY_DIR}/graph_membership_fixture"
        --count 6
        --max-steps 1000
        --preview-nodes 64
        --preview-edges 128
)

add_test(NAME path_family_tools_fixture
    COMMAND python3 "${CMAKE_SOURCE_DIR}/tools/test_path_family_tools.py"
        "${CMAKE_BINARY_DIR}/ml_safe_fixture"
        "${CMAKE_BINARY_DIR}/ml_labels_fixture"
        "${CMAKE_BINARY_DIR}/ml_pairs_fixture"
        "${CMAKE_BINARY_DIR}/image_tensor_fixture"
        "${CMAKE_BINARY_DIR}/graph_membership_fixture"
)
set_tests_properties(path_family_tools_fixture PROPERTIES DEPENDS "collatz_pair_sampler_fixture;collatz_path_image_tensor_fixture;collatz_graph_export_membership_fixture")

add_test(NAME collatz_evidence_publish_fixture
    COMMAND collatz_evidence_publish
        --full-audit "${CMAKE_BINARY_DIR}/full_audit_fixture.summary.json"
        --stratified-metadata "${CMAKE_SOURCE_DIR}/fixtures/evidence_small/stratified_metadata.json"
        --topology-manifest "${CMAKE_SOURCE_DIR}/fixtures/evidence_small/topology.json"
        --validation-metrics "${CMAKE_SOURCE_DIR}/fixtures/evidence_small/evidence_validation_metrics.json"
        --ablation-report "${CMAKE_SOURCE_DIR}/fixtures/evidence_small/ablation_report.csv"
        --source-alignment "${CMAKE_SOURCE_DIR}/fixtures/evidence_small/source_alignment.json"
        --output "${CMAKE_BINARY_DIR}/latest_public_summary.json"
        --git-commit fixture
)
set_tests_properties(collatz_evidence_publish_fixture PROPERTIES DEPENDS collatz_full_audit_fixture)

add_test(NAME evidence_summary_validation_fixture
    COMMAND "${CMAKE_SOURCE_DIR}/tools/validate_evidence_summary.py"
        "${CMAKE_SOURCE_DIR}/fixtures/evidence_small/expected_latest_public_summary.json"
)

add_test(NAME readme_snapshot_fixture
    COMMAND "${CMAKE_SOURCE_DIR}/tools/render_readme_snapshot.py"
        "--input" "${CMAKE_SOURCE_DIR}/fixtures/evidence_small/expected_latest_public_summary.json"
        "--expect" "${CMAKE_SOURCE_DIR}/fixtures/evidence_small/expected_readme_snapshot.md"
)

add_test(NAME readme_snapshot_committed_block
    COMMAND "${CMAKE_SOURCE_DIR}/tools/render_readme_snapshot.py"
        "--input" "${CMAKE_SOURCE_DIR}/data/generated/evidence/latest_public_summary.json"
        "--readme" "${CMAKE_SOURCE_DIR}/README.md"
        "--check-readme"
)

add_test(NAME dashboard_summary_svg_fixture
    COMMAND "${CMAKE_SOURCE_DIR}/tools/render_dashboard_summary.py"
        "--input" "${CMAKE_SOURCE_DIR}/fixtures/evidence_small/expected_latest_public_summary.json"
        "--output" "${CMAKE_BINARY_DIR}/dashboard-summary-fixture.svg"
)

add_test(NAME evidence_contract_fixture
    COMMAND "${CMAKE_SOURCE_DIR}/tools/test_evidence_contract.py"
)

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
