#pragma once

#include "collatz/core.hpp"

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace collatz {

constexpr std::uint32_t kBinaryFeatureVersion = 1;

#pragma pack(push, 1)
struct BinaryFeatureHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_size;
    std::uint32_t record_size;
    std::uint32_t endian_marker;
    std::uint64_t range_start;
    std::uint64_t range_end;
    std::uint32_t max_steps;
    char created_utc[32];
    char reserved[32];
};

struct BinaryFeatureRecord {
    std::uint64_t n;
    std::uint32_t total_steps;
    std::uint32_t first_drop_time;
    std::uint32_t odd_steps;
    std::uint32_t even_steps;
    std::uint32_t accelerated_steps;
    std::uint32_t peak_step;
    double peak_log2;
    double peak_ratio_log2;
    double steps_per_input_bit;
    std::uint64_t peak_high;
    std::uint64_t peak_low;
    std::uint8_t residue_mod3;
    std::uint8_t residue_mod4;
    std::uint8_t residue_mod8;
    std::uint8_t residue_mod16;
    std::uint8_t residue_mod32;
    std::uint8_t reserved0[3];
    std::uint64_t parity_prefix[kParityWords];
    std::uint16_t halving_histogram[kHalvingHistogramBuckets];
    std::uint32_t flags;
    std::uint64_t checksum;
};
#pragma pack(pop)

BinaryFeatureRecord to_binary_record(const FeatureRow &feature);
FeatureRow from_binary_record(const BinaryFeatureRecord &record);
void write_binary_header(std::ostream &out, std::uint64_t range_start, std::uint64_t range_end, std::uint32_t max_steps);
BinaryFeatureHeader read_binary_header(const std::string &path);
void update_binary_header(const std::string &path, std::uint64_t range_end, std::uint32_t max_steps);
std::uint64_t binary_record_count(const std::string &path);
std::optional<std::uint64_t> binary_completed_through(const std::string &path, std::uint64_t requested_start);
bool valid_binary_header(const BinaryFeatureHeader &header);
std::vector<BinaryFeatureRecord> read_binary_records(const std::string &path, std::uint64_t limit = 0);

} // namespace collatz
