#include "collatz/feature_io.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace collatz {
namespace {

constexpr char kMagic[8] = {'3', 'X', 'N', '1', 'F', 'T', 'R', '\0'};
constexpr std::uint32_t kEndianMarker = 0x01020304u;

} // namespace

BinaryFeatureRecord to_binary_record(const FeatureRow &feature) {
    BinaryFeatureRecord record{};
    record.n = feature.n;
    record.total_steps = feature.total_steps;
    record.first_drop_time = feature.first_drop_time;
    record.odd_steps = feature.odd_steps;
    record.even_steps = feature.even_steps;
    record.accelerated_steps = feature.accelerated_steps;
    record.peak_step = feature.peak_step;
    record.peak_log2 = static_cast<double>(feature.peak_log2);
    record.peak_ratio_log2 = static_cast<double>(feature.peak_ratio_log2);
    record.steps_per_input_bit = static_cast<double>(feature.steps_per_input_bit);
    record.peak_high = feature.peak.high;
    record.peak_low = feature.peak.low;
    record.residue_mod3 = feature.residue_mod3;
    record.residue_mod4 = feature.residue_mod4;
    record.residue_mod8 = feature.residue_mod8;
    record.residue_mod16 = feature.residue_mod16;
    record.residue_mod32 = feature.residue_mod32;
    std::copy(feature.parity_prefix.begin(), feature.parity_prefix.end(), record.parity_prefix);
    std::copy(feature.halving_histogram.begin(), feature.halving_histogram.end(), record.halving_histogram);
    record.flags = feature.flags;
    record.checksum = feature.checksum;
    return record;
}

FeatureRow from_binary_record(const BinaryFeatureRecord &record) {
    FeatureRow feature{};
    feature.n = record.n;
    feature.total_steps = record.total_steps;
    feature.first_drop_time = record.first_drop_time;
    feature.odd_steps = record.odd_steps;
    feature.even_steps = record.even_steps;
    feature.accelerated_steps = record.accelerated_steps;
    feature.peak_step = record.peak_step;
    feature.peak_log2 = record.peak_log2;
    feature.peak_ratio_log2 = record.peak_ratio_log2;
    feature.steps_per_input_bit = record.steps_per_input_bit;
    feature.peak.high = record.peak_high;
    feature.peak.low = record.peak_low;
    feature.residue_mod3 = record.residue_mod3;
    feature.residue_mod4 = record.residue_mod4;
    feature.residue_mod8 = record.residue_mod8;
    feature.residue_mod16 = record.residue_mod16;
    feature.residue_mod32 = record.residue_mod32;
    std::copy(std::begin(record.parity_prefix), std::end(record.parity_prefix), feature.parity_prefix.begin());
    std::copy(std::begin(record.halving_histogram), std::end(record.halving_histogram), feature.halving_histogram.begin());
    feature.flags = record.flags;
    feature.checksum = record.checksum;
    return feature;
}

void write_binary_header(std::ostream &out, std::uint64_t range_start, std::uint64_t range_end, std::uint32_t max_steps) {
    BinaryFeatureHeader header{};
    std::memcpy(header.magic, kMagic, sizeof(header.magic));
    header.version = kBinaryFeatureVersion;
    header.header_size = sizeof(BinaryFeatureHeader);
    header.record_size = sizeof(BinaryFeatureRecord);
    header.endian_marker = kEndianMarker;
    header.range_start = range_start;
    header.range_end = range_end;
    header.max_steps = max_steps;

    const std::string timestamp = now_iso8601();
    std::memcpy(header.created_utc, timestamp.data(), std::min(timestamp.size(), sizeof(header.created_utc) - 1));

    out.write(reinterpret_cast<const char *>(&header), sizeof(header));
    if (!out) {
        throw std::runtime_error("failed to write binary feature header");
    }
}

BinaryFeatureHeader read_binary_header(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open binary feature file: " + path);
    }
    BinaryFeatureHeader header{};
    in.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!in) {
        throw std::runtime_error("failed to read binary feature header: " + path);
    }
    return header;
}

void update_binary_header(const std::string &path, std::uint64_t range_end, std::uint32_t max_steps) {
    std::fstream io(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!io) {
        throw std::runtime_error("failed to open binary feature file for header update: " + path);
    }

    BinaryFeatureHeader header{};
    io.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!io) {
        throw std::runtime_error("failed to read binary feature header: " + path);
    }
    if (!valid_binary_header(header)) {
        throw std::runtime_error("binary feature file has an incompatible header: " + path);
    }

    header.range_end = range_end;
    header.max_steps = max_steps;
    io.clear();
    io.seekp(0);
    io.write(reinterpret_cast<const char *>(&header), sizeof(header));
    if (!io) {
        throw std::runtime_error("failed to write binary feature header: " + path);
    }
}

std::uint64_t binary_record_count(const std::string &path) {
    if (!std::filesystem::exists(path)) {
        return 0;
    }
    const auto size = std::filesystem::file_size(path);
    if (size < sizeof(BinaryFeatureHeader)) {
        return 0;
    }
    return static_cast<std::uint64_t>((size - sizeof(BinaryFeatureHeader)) / sizeof(BinaryFeatureRecord));
}

std::optional<std::uint64_t> binary_completed_through(const std::string &path, std::uint64_t requested_start) {
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }
    const auto header = read_binary_header(path);
    if (!valid_binary_header(header)) {
        throw std::runtime_error("binary feature file has an incompatible header: " + path);
    }
    if (header.range_start != requested_start) {
        throw std::runtime_error("binary resume requires the same --start used to create the file");
    }
    const auto count = binary_record_count(path);
    if (count == 0) {
        return std::nullopt;
    }
    return requested_start + count - 1;
}

bool valid_binary_header(const BinaryFeatureHeader &header) {
    return std::memcmp(header.magic, kMagic, sizeof(header.magic)) == 0 &&
           header.version == kBinaryFeatureVersion &&
           header.header_size == sizeof(BinaryFeatureHeader) &&
           header.record_size == sizeof(BinaryFeatureRecord) &&
           header.endian_marker == kEndianMarker;
}

std::vector<BinaryFeatureRecord> read_binary_records(const std::string &path, std::uint64_t limit) {
    const auto header = read_binary_header(path);
    if (!valid_binary_header(header)) {
        throw std::runtime_error("binary feature file has an incompatible header: " + path);
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open binary feature file: " + path);
    }
    in.seekg(static_cast<std::streamoff>(sizeof(BinaryFeatureHeader)));

    const auto available = binary_record_count(path);
    const auto wanted = limit == 0 ? available : std::min<std::uint64_t>(available, limit);
    std::vector<BinaryFeatureRecord> records;
    records.resize(static_cast<std::size_t>(wanted));
    if (wanted == 0) {
        return records;
    }

    in.read(
        reinterpret_cast<char *>(records.data()),
        static_cast<std::streamsize>(records.size() * sizeof(BinaryFeatureRecord)));
    if (!in) {
        throw std::runtime_error("failed to read binary feature records: " + path);
    }
    return records;
}

} // namespace collatz
