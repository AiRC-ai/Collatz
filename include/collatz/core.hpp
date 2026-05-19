#pragma once

#include <array>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace collatz {

using UInt128 = unsigned __int128;

constexpr std::uint32_t kFeatureVersion = 1;
constexpr std::size_t kParityPrefixBits = 256;
constexpr std::size_t kParityWords = kParityPrefixBits / 64;
constexpr std::size_t kHalvingHistogramBuckets = 16;

enum FeatureFlags : std::uint32_t {
    FeatureReachedOne = 1u << 0,
    FeatureOverflow = 1u << 1,
    FeatureMaxSteps = 1u << 2,
    FeatureFirstDropKnown = 1u << 3,
};

struct UInt128Parts {
    std::uint64_t high = 0;
    std::uint64_t low = 0;
};

struct FeatureRow {
    std::uint64_t n = 0;
    std::uint32_t total_steps = 0;
    std::uint32_t first_drop_time = 0;
    std::uint32_t odd_steps = 0;
    std::uint32_t even_steps = 0;
    std::uint32_t accelerated_steps = 0;
    std::uint32_t peak_step = 0;
    long double peak_log2 = 0.0L;
    long double peak_ratio_log2 = 0.0L;
    long double steps_per_input_bit = 0.0L;
    UInt128Parts peak = {};
    std::uint8_t residue_mod3 = 0;
    std::uint8_t residue_mod4 = 0;
    std::uint8_t residue_mod8 = 0;
    std::uint8_t residue_mod16 = 0;
    std::uint8_t residue_mod32 = 0;
    std::array<std::uint64_t, kParityWords> parity_prefix = {};
    std::array<std::uint16_t, kHalvingHistogramBuckets> halving_histogram = {};
    std::uint32_t flags = 0;
    std::uint64_t checksum = 0;
};

struct PathPoint {
    std::uint32_t step = 0;
    UInt128 value = 0;
    long double log2_value = 0.0L;
};

FeatureRow compute_feature(std::uint64_t n, std::uint32_t max_steps);
std::vector<PathPoint> generate_path(std::uint64_t n, std::uint32_t max_steps, bool *overflow);

std::string feature_csv_header();
std::string feature_to_csv(const FeatureRow &feature);
std::string feature_to_json(const FeatureRow &feature);
std::string parity_prefix_hex(const FeatureRow &feature);
std::uint64_t checksum_feature(const FeatureRow &feature);

UInt128Parts split_uint128(UInt128 value);
std::string uint128_to_decimal(UInt128 value);
long double log2_uint128(UInt128 value);

std::string now_iso8601();
std::string json_escape(std::string_view value);

std::optional<std::uint64_t> parse_u64(std::string_view text);
std::optional<std::uint32_t> parse_u32(std::string_view text);
void ensure_parent_dir(const std::string &path);

std::string read_last_nonempty_line(const std::string &path);
std::vector<std::string> read_last_lines(const std::string &path, std::size_t max_lines);

} // namespace collatz
