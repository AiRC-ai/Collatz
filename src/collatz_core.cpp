#include "collatz/core.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace collatz {
namespace {

constexpr UInt128 kUInt128Max = ~static_cast<UInt128>(0);

std::uint32_t ctz128(UInt128 value) {
    const auto low = static_cast<std::uint64_t>(value);
    if (low != 0) {
        return static_cast<std::uint32_t>(__builtin_ctzll(low));
    }
    const auto high = static_cast<std::uint64_t>(value >> 64);
    return 64u + static_cast<std::uint32_t>(__builtin_ctzll(high));
}

void set_parity_bit(FeatureRow &feature, std::uint32_t step_index, bool odd) {
    if (!odd || step_index >= kParityPrefixBits) {
        return;
    }
    const std::size_t word = step_index / 64u;
    const std::size_t bit = step_index % 64u;
    feature.parity_prefix[word] |= (std::uint64_t{1} << bit);
}

void fnv_bytes(std::uint64_t &hash, const void *data, std::size_t size) {
    const auto *bytes = static_cast<const unsigned char *>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
}

template <typename T>
void fnv_value(std::uint64_t &hash, const T &value) {
    fnv_bytes(hash, &value, sizeof(value));
}

std::string hex64(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

void maybe_note_first_drop(FeatureRow &feature, UInt128 value, UInt128 start, std::uint32_t step) {
    if ((feature.flags & FeatureFirstDropKnown) == 0 && value < start) {
        feature.first_drop_time = step;
        feature.flags |= FeatureFirstDropKnown;
    }
}

void update_peak(FeatureRow &feature, UInt128 value, std::uint32_t step, UInt128 &peak) {
    if (value > peak) {
        peak = value;
        feature.peak = split_uint128(value);
        feature.peak_step = step;
    }
}

} // namespace

FeatureRow compute_feature(std::uint64_t n, std::uint32_t max_steps) {
    if (n == 0) {
        throw std::invalid_argument("Collatz start must be greater than zero");
    }

    FeatureRow feature;
    feature.n = n;
    feature.residue_mod3 = static_cast<std::uint8_t>(n % 3u);
    feature.residue_mod4 = static_cast<std::uint8_t>(n % 4u);
    feature.residue_mod8 = static_cast<std::uint8_t>(n % 8u);
    feature.residue_mod16 = static_cast<std::uint8_t>(n % 16u);
    feature.residue_mod32 = static_cast<std::uint8_t>(n % 32u);

    const UInt128 start = static_cast<UInt128>(n);
    UInt128 value = start;
    UInt128 peak = value;
    feature.peak = split_uint128(peak);

    while (value != 1 && feature.total_steps < max_steps) {
        if ((value & 1u) == 0) {
            std::uint32_t run = ctz128(value);
            const std::uint32_t remaining = max_steps - feature.total_steps;
            bool truncated = false;
            if (run > remaining) {
                run = remaining;
                truncated = true;
            }

            for (std::uint32_t i = 1; i <= run; ++i) {
                maybe_note_first_drop(feature, value >> i, start, feature.total_steps + i);
            }

            value >>= run;
            feature.total_steps += run;
            feature.even_steps += run;
            feature.accelerated_steps += 1;
            const std::size_t bucket = std::min<std::size_t>(run, kHalvingHistogramBuckets) - 1u;
            feature.halving_histogram[bucket] += 1;

            if (truncated) {
                feature.flags |= FeatureMaxSteps;
                break;
            }
            continue;
        }

        set_parity_bit(feature, feature.total_steps, true);
        if (value > (kUInt128Max - 1u) / 3u) {
            feature.flags |= FeatureOverflow;
            break;
        }
        value = value * 3u + 1u;
        feature.total_steps += 1;
        feature.odd_steps += 1;
        feature.accelerated_steps += 1;
        maybe_note_first_drop(feature, value, start, feature.total_steps);
        update_peak(feature, value, feature.total_steps, peak);
    }

    if (value == 1) {
        feature.flags |= FeatureReachedOne;
    } else if (feature.total_steps >= max_steps) {
        feature.flags |= FeatureMaxSteps;
    }

    feature.peak_log2 = log2_uint128(peak);
    feature.peak_ratio_log2 = feature.peak_log2 - std::log2(static_cast<long double>(n));
    const long double input_bits = std::max(1.0L, std::log2(static_cast<long double>(n)));
    feature.steps_per_input_bit = static_cast<long double>(feature.total_steps) / input_bits;
    feature.checksum = checksum_feature(feature);
    return feature;
}

std::vector<PathPoint> generate_path(std::uint64_t n, std::uint32_t max_steps, bool *overflow) {
    if (n == 0) {
        throw std::invalid_argument("Collatz start must be greater than zero");
    }
    if (overflow) {
        *overflow = false;
    }

    std::vector<PathPoint> points;
    UInt128 value = static_cast<UInt128>(n);
    points.push_back({0, value, log2_uint128(value)});

    for (std::uint32_t step = 1; value != 1 && step <= max_steps; ++step) {
        if ((value & 1u) == 0) {
            value >>= 1u;
        } else {
            if (value > (kUInt128Max - 1u) / 3u) {
                if (overflow) {
                    *overflow = true;
                }
                break;
            }
            value = value * 3u + 1u;
        }
        points.push_back({step, value, log2_uint128(value)});
    }
    return points;
}

std::string feature_csv_header() {
    std::ostringstream out;
    out << "n,total_steps,first_drop_time,odd_steps,even_steps,accelerated_steps,"
        << "peak_step,peak_log2,peak_ratio_log2,steps_per_input_bit,"
        << "peak_high,peak_low,residue_mod3,residue_mod4,residue_mod8,residue_mod16,residue_mod32,"
        << "parity_prefix_hex";
    for (std::size_t i = 0; i < kHalvingHistogramBuckets; ++i) {
        out << ",halving_run_" << (i + 1);
    }
    out << ",flags,checksum";
    return out.str();
}

std::string feature_to_csv(const FeatureRow &feature) {
    std::ostringstream out;
    out << std::setprecision(10)
        << feature.n << ','
        << feature.total_steps << ','
        << feature.first_drop_time << ','
        << feature.odd_steps << ','
        << feature.even_steps << ','
        << feature.accelerated_steps << ','
        << feature.peak_step << ','
        << static_cast<double>(feature.peak_log2) << ','
        << static_cast<double>(feature.peak_ratio_log2) << ','
        << static_cast<double>(feature.steps_per_input_bit) << ','
        << feature.peak.high << ','
        << feature.peak.low << ','
        << static_cast<unsigned>(feature.residue_mod3) << ','
        << static_cast<unsigned>(feature.residue_mod4) << ','
        << static_cast<unsigned>(feature.residue_mod8) << ','
        << static_cast<unsigned>(feature.residue_mod16) << ','
        << static_cast<unsigned>(feature.residue_mod32) << ','
        << parity_prefix_hex(feature);
    for (const auto count : feature.halving_histogram) {
        out << ',' << count;
    }
    out << ',' << feature.flags << ',' << feature.checksum;
    return out.str();
}

std::string feature_to_json(const FeatureRow &feature) {
    std::ostringstream out;
    out << std::setprecision(10)
        << "{\"n\":" << feature.n
        << ",\"total_steps\":" << feature.total_steps
        << ",\"first_drop_time\":" << feature.first_drop_time
        << ",\"odd_steps\":" << feature.odd_steps
        << ",\"even_steps\":" << feature.even_steps
        << ",\"accelerated_steps\":" << feature.accelerated_steps
        << ",\"peak_step\":" << feature.peak_step
        << ",\"peak_log2\":" << static_cast<double>(feature.peak_log2)
        << ",\"peak_ratio_log2\":" << static_cast<double>(feature.peak_ratio_log2)
        << ",\"peak_decimal\":\"" << uint128_to_decimal((static_cast<UInt128>(feature.peak.high) << 64) | feature.peak.low) << "\""
        << ",\"residue_mod3\":" << static_cast<unsigned>(feature.residue_mod3)
        << ",\"residue_mod4\":" << static_cast<unsigned>(feature.residue_mod4)
        << ",\"residue_mod8\":" << static_cast<unsigned>(feature.residue_mod8)
        << ",\"residue_mod16\":" << static_cast<unsigned>(feature.residue_mod16)
        << ",\"residue_mod32\":" << static_cast<unsigned>(feature.residue_mod32)
        << ",\"parity_prefix_hex\":\"" << parity_prefix_hex(feature) << "\""
        << ",\"flags\":" << feature.flags
        << ",\"checksum\":" << feature.checksum
        << '}';
    return out.str();
}

std::string parity_prefix_hex(const FeatureRow &feature) {
    std::ostringstream out;
    for (std::size_t i = kParityWords; i-- > 0;) {
        out << hex64(feature.parity_prefix[i]);
    }
    return out.str();
}

std::uint64_t checksum_feature(const FeatureRow &feature) {
    std::uint64_t hash = 14695981039346656037ull;
    fnv_value(hash, kFeatureVersion);
    fnv_value(hash, feature.n);
    fnv_value(hash, feature.total_steps);
    fnv_value(hash, feature.first_drop_time);
    fnv_value(hash, feature.odd_steps);
    fnv_value(hash, feature.even_steps);
    fnv_value(hash, feature.accelerated_steps);
    fnv_value(hash, feature.peak_step);
    fnv_value(hash, feature.peak.high);
    fnv_value(hash, feature.peak.low);
    fnv_value(hash, feature.residue_mod3);
    fnv_value(hash, feature.residue_mod4);
    fnv_value(hash, feature.residue_mod8);
    fnv_value(hash, feature.residue_mod16);
    fnv_value(hash, feature.residue_mod32);
    for (const auto word : feature.parity_prefix) {
        fnv_value(hash, word);
    }
    for (const auto bucket : feature.halving_histogram) {
        fnv_value(hash, bucket);
    }
    fnv_value(hash, feature.flags);
    return hash;
}

UInt128Parts split_uint128(UInt128 value) {
    return {
        static_cast<std::uint64_t>(value >> 64),
        static_cast<std::uint64_t>(value),
    };
}

std::string uint128_to_decimal(UInt128 value) {
    if (value == 0) {
        return "0";
    }

    std::string out;
    while (value > 0) {
        const auto digit = static_cast<unsigned>(value % 10u);
        out.push_back(static_cast<char>('0' + digit));
        value /= 10u;
    }
    std::reverse(out.begin(), out.end());
    return out;
}

long double log2_uint128(UInt128 value) {
    if (value == 0) {
        return 0.0L;
    }
    const UInt128Parts parts = split_uint128(value);
    const long double high = static_cast<long double>(parts.high);
    const long double low = static_cast<long double>(parts.low);
    return std::log2(high * 18446744073709551616.0L + low);
}

std::string now_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t raw = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &raw);
#else
    gmtime_r(&raw, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

std::string json_escape(std::string_view value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const char ch : value) {
        switch (ch) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out.push_back(ch);
            break;
        }
    }
    return out;
}

std::optional<std::uint64_t> parse_u64(std::string_view text) {
    if (text.empty()) {
        return std::nullopt;
    }
    std::uint64_t value = 0;
    const auto *begin = text.data();
    const auto *end = text.data() + text.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        return std::nullopt;
    }
    return value;
}

std::optional<std::uint32_t> parse_u32(std::string_view text) {
    const auto value = parse_u64(text);
    if (!value || *value > std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(*value);
}

void ensure_parent_dir(const std::string &path) {
    const std::filesystem::path file_path(path);
    const auto parent = file_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

std::string read_last_nonempty_line(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        return {};
    }
    std::string line;
    std::string last;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            last = line;
        }
    }
    return last;
}

std::vector<std::string> read_last_lines(const std::string &path, std::size_t max_lines) {
    std::ifstream in(path);
    if (!in) {
        return {};
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        lines.push_back(line);
        if (lines.size() > max_lines) {
            lines.erase(lines.begin());
        }
    }
    return lines;
}

} // namespace collatz
