#include "collatz/feature_io.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string input;
    std::string output = "data/generated/full_audit/summary.json";
    std::size_t range_bands = 16;
    std::size_t top_count = 16;
    std::uint64_t limit = 0;
};

struct RecordSummary {
    std::uint64_t n = 0;
    std::uint32_t total_steps = 0;
    std::uint32_t first_drop_time = 0;
    double peak_log2 = 0.0;
    double peak_ratio_log2 = 0.0;
    std::uint8_t residue_mod32 = 0;
};

struct BandStats {
    std::uint64_t count = 0;
    std::uint64_t min_n = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_n = 0;
    long double total_steps_sum = 0.0L;
    std::uint32_t max_total_steps = 0;
    std::uint64_t max_total_steps_n = 0;
    double max_peak_ratio_log2 = -std::numeric_limits<double>::infinity();
    std::uint64_t max_peak_ratio_n = 0;
    std::vector<std::uint64_t> total_steps_hist;
};

struct ResidueStats {
    std::uint64_t count = 0;
    long double total_steps_sum = 0.0L;
    std::uint32_t max_total_steps = 0;
    std::uint64_t max_total_steps_n = 0;
};

void usage(std::ostream &out) {
    out << "usage: collatz_full_audit --input FILE [options]\n\n"
        << "options:\n"
        << "  --output FILE       JSON output (default data/generated/full_audit/summary.json)\n"
        << "  --range-bands N     number of numeric range bands (default 16)\n"
        << "  --top-count N       top records to include per ladder (default 16)\n"
        << "  --limit N           max binary records to read, 0 means all records\n";
}

std::uint64_t parse_u64_required(const std::string &value, const char *name, bool allow_zero = false) {
    const auto parsed = collatz::parse_u64(value);
    if (!parsed || (!allow_zero && *parsed == 0)) {
        throw std::runtime_error(std::string(name) + " must be a positive integer");
    }
    return *parsed;
}

Options parse_args(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string(name) + " requires a value");
            }
            return argv[++i];
        };
        if (arg == "--input") {
            options.input = need_value("--input");
        } else if (arg == "--output") {
            options.output = need_value("--output");
        } else if (arg == "--range-bands") {
            options.range_bands = static_cast<std::size_t>(parse_u64_required(need_value("--range-bands"), "--range-bands"));
        } else if (arg == "--top-count") {
            options.top_count = static_cast<std::size_t>(parse_u64_required(need_value("--top-count"), "--top-count", true));
        } else if (arg == "--limit") {
            options.limit = parse_u64_required(need_value("--limit"), "--limit", true);
        } else if (arg == "--help" || arg == "-h") {
            usage(std::cout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.input.empty()) {
        throw std::runtime_error("--input is required");
    }
    return options;
}

void mix_checksum(std::uint64_t &hash, std::uint64_t value) {
    for (std::size_t i = 0; i < sizeof(value); ++i) {
        hash ^= static_cast<unsigned char>((value >> (i * 8u)) & 0xffu);
        hash *= 1099511628211ull;
    }
}

void increment_hist(std::vector<std::uint64_t> &hist, std::uint32_t value) {
    if (hist.size() <= value) {
        hist.resize(static_cast<std::size_t>(value) + 1, 0);
    }
    ++hist[value];
}

double mean_or_zero(long double sum, std::uint64_t count) {
    if (count == 0) {
        return 0.0;
    }
    return static_cast<double>(sum / static_cast<long double>(count));
}

std::uint32_t quantile_from_hist(const std::vector<std::uint64_t> &hist, std::uint64_t total, long double q) {
    if (total == 0 || hist.empty()) {
        return 0;
    }
    std::uint64_t target = static_cast<std::uint64_t>(q * static_cast<long double>(total - 1)) + 1;
    if (target == 0) {
        target = 1;
    }
    std::uint64_t cumulative = 0;
    for (std::size_t i = 0; i < hist.size(); ++i) {
        cumulative += hist[i];
        if (cumulative >= target) {
            return static_cast<std::uint32_t>(i);
        }
    }
    return static_cast<std::uint32_t>(hist.size() - 1);
}

std::size_t range_band_for(
    std::uint64_t range_start,
    std::uint64_t range_end,
    std::uint64_t n,
    std::size_t bands) {
    if (bands == 0 || range_end <= range_start || n < range_start) {
        return 0;
    }
    const long double offset = static_cast<long double>(n - range_start);
    const long double span = static_cast<long double>(range_end - range_start + 1);
    std::size_t band = static_cast<std::size_t>((offset / span) * static_cast<long double>(bands));
    if (band >= bands) {
        band = bands - 1;
    }
    return band;
}

template <typename Better>
void consider_top(std::vector<RecordSummary> &items, const RecordSummary &record, std::size_t keep, Better better) {
    if (keep == 0) {
        return;
    }
    items.push_back(record);
    std::sort(items.begin(), items.end(), better);
    if (items.size() > keep) {
        items.resize(keep);
    }
}

void write_record_array(std::ostream &out, const std::vector<RecordSummary> &records) {
    out << '[';
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto &record = records[i];
        if (i != 0) {
            out << ',';
        }
        out << "{\"n\":" << record.n
            << ",\"total_steps\":" << record.total_steps
            << ",\"first_drop_time\":" << record.first_drop_time
            << ",\"peak_log2\":" << record.peak_log2
            << ",\"peak_ratio_log2\":" << record.peak_ratio_log2
            << ",\"residue_mod32\":" << static_cast<unsigned>(record.residue_mod32)
            << '}';
    }
    out << ']';
}

void write_quantiles(std::ostream &out, const std::vector<std::uint64_t> &hist, std::uint64_t total) {
    out << "{\"p50\":" << quantile_from_hist(hist, total, 0.50L)
        << ",\"p90\":" << quantile_from_hist(hist, total, 0.90L)
        << ",\"p99\":" << quantile_from_hist(hist, total, 0.99L)
        << ",\"p999\":" << quantile_from_hist(hist, total, 0.999L)
        << '}';
}

void write_audit(
    const Options &options,
    const collatz::BinaryFeatureHeader &header,
    std::uint64_t available_records,
    std::uint64_t records_read,
    std::uint64_t reached_one,
    std::uint64_t overflow,
    std::uint64_t max_steps_hit,
    std::uint64_t first_drop_known,
    long double total_steps_sum,
    long double first_drop_sum,
    long double peak_log2_sum,
    long double peak_ratio_sum,
    const RecordSummary &max_total_steps,
    const RecordSummary &max_peak_ratio,
    const RecordSummary &max_peak_log2,
    const std::vector<std::uint64_t> &total_steps_hist,
    const std::vector<std::uint64_t> &first_drop_hist,
    const std::array<std::uint64_t, collatz::kHalvingHistogramBuckets> &halving_hist,
    const std::array<ResidueStats, 32> &residues,
    const std::vector<BandStats> &bands,
    const std::vector<RecordSummary> &top_total_steps,
    const std::vector<RecordSummary> &top_peak_ratio,
    std::uint64_t checksum) {
    collatz::ensure_parent_dir(options.output);
    std::ofstream out(options.output);
    if (!out) {
        throw std::runtime_error("failed to open output: " + options.output);
    }
    out << std::setprecision(10);
    const std::uint64_t effective_range_end =
        records_read == 0 ? header.range_end : header.range_start + records_read - 1;
    out << "{\n"
        << "  \"dataset_type\": \"collatz_full_dataset_audit\",\n"
        << "  \"tool\": \"collatz_full_audit\",\n"
        << "  \"created_utc\": \"" << collatz::now_iso8601() << "\",\n"
        << "  \"input_range_start\": " << header.range_start << ",\n"
        << "  \"input_range_end\": " << header.range_end << ",\n"
        << "  \"effective_range_end\": " << effective_range_end << ",\n"
        << "  \"input_records_available\": " << available_records << ",\n"
        << "  \"records_read\": " << records_read << ",\n"
        << "  \"coverage_ratio\": " << (available_records == 0 ? 0.0 : static_cast<double>(records_read) / static_cast<double>(available_records)) << ",\n"
        << "  \"reached_one_count\": " << reached_one << ",\n"
        << "  \"overflow_count\": " << overflow << ",\n"
        << "  \"max_steps_hit_count\": " << max_steps_hit << ",\n"
        << "  \"first_drop_known_count\": " << first_drop_known << ",\n"
        << "  \"total_steps_mean\": " << mean_or_zero(total_steps_sum, records_read) << ",\n"
        << "  \"first_drop_time_mean\": " << mean_or_zero(first_drop_sum, first_drop_known) << ",\n"
        << "  \"peak_log2_mean\": " << mean_or_zero(peak_log2_sum, records_read) << ",\n"
        << "  \"peak_ratio_log2_mean\": " << mean_or_zero(peak_ratio_sum, records_read) << ",\n"
        << "  \"max_total_steps\": " << max_total_steps.total_steps << ",\n"
        << "  \"max_total_steps_n\": " << max_total_steps.n << ",\n"
        << "  \"max_peak_ratio_log2\": " << max_peak_ratio.peak_ratio_log2 << ",\n"
        << "  \"max_peak_ratio_n\": " << max_peak_ratio.n << ",\n"
        << "  \"max_peak_log2\": " << max_peak_log2.peak_log2 << ",\n"
        << "  \"max_peak_n\": " << max_peak_log2.n << ",\n"
        << "  \"total_steps_quantiles\": ";
    write_quantiles(out, total_steps_hist, records_read);
    out << ",\n  \"first_drop_time_quantiles\": ";
    write_quantiles(out, first_drop_hist, first_drop_known);
    out << ",\n  \"halving_run_histogram\": [";
    for (std::size_t i = 0; i < halving_hist.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << halving_hist[i];
    }
    out << "],\n  \"residue_mod32\": [\n";
    for (std::size_t i = 0; i < residues.size(); ++i) {
        const auto &residue = residues[i];
        out << "    {\"residue\":" << i
            << ",\"count\":" << residue.count
            << ",\"total_steps_mean\":" << mean_or_zero(residue.total_steps_sum, residue.count)
            << ",\"max_total_steps\":" << residue.max_total_steps
            << ",\"max_total_steps_n\":" << residue.max_total_steps_n << '}';
        if (i + 1 != residues.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  ],\n  \"range_bands\": [\n";
    for (std::size_t i = 0; i < bands.size(); ++i) {
        const auto &band = bands[i];
        out << "    {\"band\":" << i
            << ",\"count\":" << band.count
            << ",\"min_n\":" << (band.count == 0 ? 0 : band.min_n)
            << ",\"max_n\":" << band.max_n
            << ",\"total_steps_mean\":" << mean_or_zero(band.total_steps_sum, band.count)
            << ",\"total_steps_p99\":" << quantile_from_hist(band.total_steps_hist, band.count, 0.99L)
            << ",\"max_total_steps\":" << band.max_total_steps
            << ",\"max_total_steps_n\":" << band.max_total_steps_n
            << ",\"max_peak_ratio_log2\":" << (band.count == 0 ? 0.0 : band.max_peak_ratio_log2)
            << ",\"max_peak_ratio_n\":" << band.max_peak_ratio_n << '}';
        if (i + 1 != bands.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  ],\n  \"top_total_steps\": ";
    write_record_array(out, top_total_steps);
    out << ",\n  \"top_peak_ratio\": ";
    write_record_array(out, top_peak_ratio);
    out << ",\n"
        << "  \"range_band_count\": " << bands.size() << ",\n"
        << "  \"feature_schema_version\": " << collatz::kFeatureVersion << ",\n"
        << "  \"binary_feature_version\": " << collatz::kBinaryFeatureVersion << ",\n"
        << "  \"checksum_fnv1a64\": " << checksum << "\n"
        << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_args(argc, argv);
        const auto header = collatz::read_binary_header(options.input);
        if (!collatz::valid_binary_header(header)) {
            throw std::runtime_error("input has an invalid binary feature header");
        }
        const auto available = collatz::binary_record_count(options.input);
        const auto wanted = options.limit == 0 ? available : std::min<std::uint64_t>(available, options.limit);

        std::ifstream in(options.input, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open input: " + options.input);
        }
        in.seekg(static_cast<std::streamoff>(sizeof(collatz::BinaryFeatureHeader)));

        std::uint64_t records_read = 0;
        std::uint64_t reached_one = 0;
        std::uint64_t overflow = 0;
        std::uint64_t max_steps_hit = 0;
        std::uint64_t first_drop_known = 0;
        long double total_steps_sum = 0.0L;
        long double first_drop_sum = 0.0L;
        long double peak_log2_sum = 0.0L;
        long double peak_ratio_sum = 0.0L;
        RecordSummary max_total_steps{};
        RecordSummary max_peak_ratio{};
        RecordSummary max_peak_log2{};
        std::vector<std::uint64_t> total_steps_hist;
        std::vector<std::uint64_t> first_drop_hist;
        std::array<std::uint64_t, collatz::kHalvingHistogramBuckets> halving_hist{};
        std::array<ResidueStats, 32> residues{};
        std::vector<BandStats> bands(options.range_bands);
        std::vector<RecordSummary> top_total_steps;
        std::vector<RecordSummary> top_peak_ratio;
        std::uint64_t checksum = 14695981039346656037ull;

        const std::uint64_t effective_range_end =
            wanted == 0 ? header.range_end : header.range_start + wanted - 1;

        for (; records_read < wanted; ++records_read) {
            collatz::BinaryFeatureRecord record{};
            in.read(reinterpret_cast<char *>(&record), sizeof(record));
            if (!in) {
                throw std::runtime_error("failed while reading binary feature record");
            }
            mix_checksum(checksum, record.n);
            mix_checksum(checksum, record.checksum);

            RecordSummary summary{
                record.n,
                record.total_steps,
                record.first_drop_time,
                record.peak_log2,
                record.peak_ratio_log2,
                record.residue_mod32,
            };

            reached_one += (record.flags & collatz::FeatureReachedOne) != 0 ? 1 : 0;
            overflow += (record.flags & collatz::FeatureOverflow) != 0 ? 1 : 0;
            max_steps_hit += (record.flags & collatz::FeatureMaxSteps) != 0 ? 1 : 0;
            const bool has_first_drop = (record.flags & collatz::FeatureFirstDropKnown) != 0;
            first_drop_known += has_first_drop ? 1 : 0;
            total_steps_sum += record.total_steps;
            if (has_first_drop) {
                first_drop_sum += record.first_drop_time;
                increment_hist(first_drop_hist, record.first_drop_time);
            }
            peak_log2_sum += record.peak_log2;
            peak_ratio_sum += record.peak_ratio_log2;
            increment_hist(total_steps_hist, record.total_steps);

            if (record.total_steps > max_total_steps.total_steps) {
                max_total_steps = summary;
            }
            if (record.peak_ratio_log2 > max_peak_ratio.peak_ratio_log2 || max_peak_ratio.n == 0) {
                max_peak_ratio = summary;
            }
            if (record.peak_log2 > max_peak_log2.peak_log2 || max_peak_log2.n == 0) {
                max_peak_log2 = summary;
            }
            consider_top(top_total_steps, summary, options.top_count, [](const auto &left, const auto &right) {
                if (left.total_steps == right.total_steps) {
                    return left.n < right.n;
                }
                return left.total_steps > right.total_steps;
            });
            consider_top(top_peak_ratio, summary, options.top_count, [](const auto &left, const auto &right) {
                if (left.peak_ratio_log2 == right.peak_ratio_log2) {
                    return left.n < right.n;
                }
                return left.peak_ratio_log2 > right.peak_ratio_log2;
            });

            for (std::size_t i = 0; i < halving_hist.size(); ++i) {
                halving_hist[i] += record.halving_histogram[i];
            }

            auto &residue = residues[record.residue_mod32];
            ++residue.count;
            residue.total_steps_sum += record.total_steps;
            if (record.total_steps > residue.max_total_steps) {
                residue.max_total_steps = record.total_steps;
                residue.max_total_steps_n = record.n;
            }

            if (!bands.empty()) {
                auto &band = bands[range_band_for(header.range_start, effective_range_end, record.n, bands.size())];
                ++band.count;
                band.min_n = std::min(band.min_n, record.n);
                band.max_n = std::max(band.max_n, record.n);
                band.total_steps_sum += record.total_steps;
                increment_hist(band.total_steps_hist, record.total_steps);
                if (record.total_steps > band.max_total_steps) {
                    band.max_total_steps = record.total_steps;
                    band.max_total_steps_n = record.n;
                }
                if (record.peak_ratio_log2 > band.max_peak_ratio_log2 || band.max_peak_ratio_n == 0) {
                    band.max_peak_ratio_log2 = record.peak_ratio_log2;
                    band.max_peak_ratio_n = record.n;
                }
            }
        }

        write_audit(
            options,
            header,
            available,
            records_read,
            reached_one,
            overflow,
            max_steps_hit,
            first_drop_known,
            total_steps_sum,
            first_drop_sum,
            peak_log2_sum,
            peak_ratio_sum,
            max_total_steps,
            max_peak_ratio,
            max_peak_log2,
            total_steps_hist,
            first_drop_hist,
            halving_hist,
            residues,
            bands,
            top_total_steps,
            top_peak_ratio,
            checksum);

        std::cout << "records_read=" << records_read
                  << " max_total_steps=" << max_total_steps.total_steps
                  << " max_total_steps_n=" << max_total_steps.n
                  << " output=" << options.output << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
