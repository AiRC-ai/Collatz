#include "collatz/core.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kOeisStoppingSource = "OEIS_A006577_total_stopping_time";
constexpr std::string_view kOeisPathSource = "OEIS_A006884_path_record";
constexpr std::string_view kRoosendaalPathSource = "Roosendaal_path_record";
constexpr std::string_view kRoosendaalDelaySource = "Roosendaal_delay_record";
constexpr std::string_view kBarinaPathSource = "Barina_path_record";
constexpr std::string_view kOliveiraPathSource = "Oliveira_e_Silva_max_excursion_record";
constexpr std::string_view kOliveiraStoppingSource = "Oliveira_e_Silva_stopping_record";

struct Options {
    std::string oeis_stopping;
    std::string oeis_path_records;
    std::string roosendaal_path_records;
    std::string roosendaal_delay_records;
    std::string barina_path_records;
    std::string oliveira_max_excursion_records;
    std::string oliveira_stopping_records;
    std::string output = "data/source_validation/public_source_targets.csv";
    std::string metadata;
    std::uint64_t max_n = 100000000;
    std::uint32_t max_steps = 10000000;
    std::size_t stopping_limit = 5000;
    std::size_t path_record_limit = 100;
    std::size_t generic_record_limit = 250;
};

struct FamilyCounter {
    std::size_t rows_read = 0;
    std::size_t rows_written = 0;
    std::size_t rows_skipped_above_max_n = 0;
    std::size_t rows_bad = 0;
    std::size_t rows_mismatch = 0;
    std::size_t overflow_tokens = 0;
};

struct Counters {
    std::map<std::string, FamilyCounter> family;
    std::set<std::string> families_written;
    std::size_t skipped_overflow_or_unfinished = 0;
};

struct TargetRow {
    std::string source;
    std::uint64_t n = 0;
    std::uint32_t total_steps = 0;
    std::uint64_t peak_low = 0;
    std::string source_kind;
    std::uint64_t source_rank = 0;
    std::string source_url;
    std::string retrieved_utc;
    std::string parser;
};

void usage(std::ostream &out) {
    out << "usage: collatz_source_targets [options]\n\n"
        << "Build a public source-validation target CSV from OEIS and record-holder files.\n\n"
        << "options:\n"
        << "  --oeis-stopping FILE                  OEIS A006577 b-file (n total_steps)\n"
        << "  --oeis-path-records FILE              OEIS A006884 b-file (index n)\n"
        << "  --roosendaal-path-records FILE        Roosendaal path-record page/text\n"
        << "  --roosendaal-delay-records FILE       Roosendaal delay-record page/text\n"
        << "  --barina-path-records FILE            Barina path-record page/text\n"
        << "  --oliveira-max-excursion-records FILE Decompressed Oliveira e Silva t(n) record file\n"
        << "  --oliveira-stopping-records FILE      Decompressed Oliveira e Silva s(n) record file\n"
        << "  --output FILE                         output CSV (default data/source_validation/public_source_targets.csv)\n"
        << "  --metadata FILE                       metadata JSON (default OUTPUT.metadata.json)\n"
        << "  --max-n N                             keep starts <= N (default 100000000)\n"
        << "  --max-steps N                         Collatz compute guard (default 10000000)\n"
        << "  --stopping-limit N                    max A006577 rows to write (default 5000)\n"
        << "  --path-record-limit N                 max path-record rows per source to write (default 100)\n"
        << "  --generic-record-limit N              max non-OEIS rows per source to write (default 250)\n";
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
        if (arg == "--oeis-stopping") {
            options.oeis_stopping = need_value("--oeis-stopping");
        } else if (arg == "--oeis-path-records") {
            options.oeis_path_records = need_value("--oeis-path-records");
        } else if (arg == "--roosendaal-path-records") {
            options.roosendaal_path_records = need_value("--roosendaal-path-records");
        } else if (arg == "--roosendaal-delay-records") {
            options.roosendaal_delay_records = need_value("--roosendaal-delay-records");
        } else if (arg == "--barina-path-records") {
            options.barina_path_records = need_value("--barina-path-records");
        } else if (arg == "--oliveira-max-excursion-records") {
            options.oliveira_max_excursion_records = need_value("--oliveira-max-excursion-records");
        } else if (arg == "--oliveira-stopping-records") {
            options.oliveira_stopping_records = need_value("--oliveira-stopping-records");
        } else if (arg == "--output") {
            options.output = need_value("--output");
        } else if (arg == "--metadata") {
            options.metadata = need_value("--metadata");
        } else if (arg == "--max-n") {
            const auto value = collatz::parse_u64(need_value("--max-n"));
            if (!value || *value == 0) {
                throw std::runtime_error("--max-n must be a positive integer");
            }
            options.max_n = *value;
        } else if (arg == "--max-steps") {
            const auto value = collatz::parse_u32(need_value("--max-steps"));
            if (!value || *value == 0) {
                throw std::runtime_error("--max-steps must be a positive integer");
            }
            options.max_steps = *value;
        } else if (arg == "--stopping-limit") {
            const auto value = collatz::parse_u64(need_value("--stopping-limit"));
            if (!value) {
                throw std::runtime_error("--stopping-limit must be a non-negative integer");
            }
            options.stopping_limit = static_cast<std::size_t>(std::min<std::uint64_t>(*value, std::numeric_limits<std::size_t>::max()));
        } else if (arg == "--path-record-limit") {
            const auto value = collatz::parse_u64(need_value("--path-record-limit"));
            if (!value) {
                throw std::runtime_error("--path-record-limit must be a non-negative integer");
            }
            options.path_record_limit = static_cast<std::size_t>(std::min<std::uint64_t>(*value, std::numeric_limits<std::size_t>::max()));
        } else if (arg == "--generic-record-limit") {
            const auto value = collatz::parse_u64(need_value("--generic-record-limit"));
            if (!value) {
                throw std::runtime_error("--generic-record-limit must be a non-negative integer");
            }
            options.generic_record_limit = static_cast<std::size_t>(std::min<std::uint64_t>(*value, std::numeric_limits<std::size_t>::max()));
        } else if (arg == "--help" || arg == "-h") {
            usage(std::cout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    const bool has_input = !options.oeis_stopping.empty() || !options.oeis_path_records.empty() ||
                           !options.roosendaal_path_records.empty() || !options.roosendaal_delay_records.empty() ||
                           !options.barina_path_records.empty() || !options.oliveira_max_excursion_records.empty() ||
                           !options.oliveira_stopping_records.empty();
    if (!has_input) {
        throw std::runtime_error("at least one source input file is required");
    }
    if (options.metadata.empty()) {
        options.metadata = options.output + ".metadata.json";
    }
    return options;
}

bool comment_or_blank(const std::string &line) {
    for (const char ch : line) {
        if (ch == '#') {
            return true;
        }
        if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '-') {
            return false;
        }
    }
    return true;
}

bool read_two_u64(const std::string &line, std::uint64_t &left, std::uint64_t &right) {
    std::string left_text;
    std::string right_text;
    std::stringstream stream(line);
    if (!(stream >> left_text >> right_text)) {
        return false;
    }
    const auto parsed_left = collatz::parse_u64(left_text);
    const auto parsed_right = collatz::parse_u64(right_text);
    if (!parsed_left || !parsed_right) {
        return false;
    }
    left = *parsed_left;
    right = *parsed_right;
    return true;
}

std::vector<std::uint64_t> numeric_tokens(const std::string &line, FamilyCounter &counter) {
    std::vector<std::uint64_t> tokens;
    std::string token;
    auto flush = [&]() {
        if (token.empty()) {
            return;
        }
        std::string digits;
        digits.reserve(token.size());
        for (const char ch : token) {
            if (std::isdigit(static_cast<unsigned char>(ch)) != 0) {
                digits.push_back(ch);
            }
        }
        token.clear();
        if (digits.empty()) {
            return;
        }
        const auto parsed = collatz::parse_u64(digits);
        if (!parsed) {
            ++counter.overflow_tokens;
            return;
        }
        tokens.push_back(*parsed);
    };

    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (std::isdigit(static_cast<unsigned char>(ch)) != 0) {
            token.push_back(ch);
        } else if (ch == ',' && !token.empty() && i + 1 < line.size() &&
                   std::isdigit(static_cast<unsigned char>(line[i + 1])) != 0) {
            token.push_back(ch);
        } else {
            flush();
        }
    }
    flush();
    return tokens;
}

bool feature_is_usable(const collatz::FeatureRow &feature) {
    return (feature.flags & collatz::FeatureReachedOne) != 0 && (feature.flags & collatz::FeatureOverflow) == 0 &&
           (feature.flags & collatz::FeatureMaxSteps) == 0 && feature.peak.high == 0;
}

void write_target(std::ofstream &out, const TargetRow &row) {
    out << row.source << ',' << row.n << ',' << row.total_steps << ',' << row.peak_low
        << ',' << row.source_kind << ',' << row.source_rank
        << ',' << row.source_url << ',' << row.retrieved_utc << ',' << row.parser << '\n';
}

bool write_computed_target(
    const Options &options,
    std::ofstream &out,
    Counters &counters,
    std::string_view source,
    std::string_view source_kind,
    std::uint64_t source_rank,
    std::string_view source_url,
    std::string_view parser,
    std::uint64_t n,
    const std::string &retrieved_utc) {
    auto &counter = counters.family[std::string(source)];
    if (n == 0) {
        ++counter.rows_bad;
        return false;
    }
    if (n > options.max_n) {
        ++counter.rows_skipped_above_max_n;
        return false;
    }

    const auto feature = collatz::compute_feature(n, options.max_steps);
    if (!feature_is_usable(feature)) {
        ++counters.skipped_overflow_or_unfinished;
        return false;
    }

    write_target(out, {std::string(source), feature.n, feature.total_steps, feature.peak.low,
                       std::string(source_kind), source_rank, std::string(source_url), retrieved_utc,
                       std::string(parser)});
    ++counter.rows_written;
    counters.families_written.insert(std::string(source));
    return true;
}

void process_stopping_file(const Options &options, std::ofstream &out, Counters &counters, const std::string &retrieved_utc) {
    if (options.oeis_stopping.empty() || options.stopping_limit == 0) {
        return;
    }
    std::ifstream in(options.oeis_stopping);
    if (!in) {
        throw std::runtime_error("failed to open OEIS stopping file: " + options.oeis_stopping);
    }

    auto &counter = counters.family[std::string(kOeisStoppingSource)];
    std::string line;
    while (std::getline(in, line) && counter.rows_written < options.stopping_limit) {
        if (comment_or_blank(line)) {
            continue;
        }
        ++counter.rows_read;
        std::uint64_t n = 0;
        std::uint64_t steps = 0;
        if (!read_two_u64(line, n, steps) || n == 0 || steps > std::numeric_limits<std::uint32_t>::max()) {
            ++counter.rows_bad;
            continue;
        }
        if (n > options.max_n) {
            ++counter.rows_skipped_above_max_n;
            continue;
        }

        const auto feature = collatz::compute_feature(n, options.max_steps);
        if (!feature_is_usable(feature)) {
            ++counters.skipped_overflow_or_unfinished;
            continue;
        }
        if (feature.total_steps != static_cast<std::uint32_t>(steps)) {
            ++counter.rows_mismatch;
            continue;
        }

        write_target(out, {std::string(kOeisStoppingSource), feature.n, feature.total_steps, feature.peak.low,
                           "total_stopping_time", n, "https://oeis.org/A006577/b006577.txt", retrieved_utc, "oeis_b_file_v1"});
        ++counter.rows_written;
        counters.families_written.insert(std::string(kOeisStoppingSource));
    }
}

void process_oeis_path_record_file(const Options &options, std::ofstream &out, Counters &counters, const std::string &retrieved_utc) {
    if (options.oeis_path_records.empty() || options.path_record_limit == 0) {
        return;
    }
    std::ifstream in(options.oeis_path_records);
    if (!in) {
        throw std::runtime_error("failed to open OEIS path-record file: " + options.oeis_path_records);
    }

    auto &counter = counters.family[std::string(kOeisPathSource)];
    std::string line;
    while (std::getline(in, line) && counter.rows_written < options.path_record_limit) {
        if (comment_or_blank(line)) {
            continue;
        }
        ++counter.rows_read;
        std::uint64_t index = 0;
        std::uint64_t n = 0;
        if (!read_two_u64(line, index, n) || index == 0 || n == 0) {
            ++counter.rows_bad;
            continue;
        }
        write_computed_target(options, out, counters, kOeisPathSource, "path_record", index,
                              "https://oeis.org/A006884/b006884.txt", "oeis_b_file_v1", n, retrieved_utc);
    }
}

bool starts_with_numeric_column(const std::string &line) {
    std::size_t pos = 0;
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos])) != 0) {
        ++pos;
    }
    if (pos >= line.size() || std::isdigit(static_cast<unsigned char>(line[pos])) == 0) {
        return false;
    }
    while (pos < line.size() && std::isdigit(static_cast<unsigned char>(line[pos])) != 0) {
        ++pos;
    }
    return pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos])) != 0;
}

std::vector<std::string> record_candidate_rows(std::ifstream &in) {
    std::vector<std::string> rows;
    std::string line;
    std::string row;
    bool in_html_row = false;
    while (std::getline(in, line)) {
        const bool starts_html_row = line.find("<tr") != std::string::npos;
        if (starts_html_row || in_html_row) {
            in_html_row = true;
            row += ' ';
            row += line;
            if (line.find("</tr") != std::string::npos) {
                if (row.find("<td") != std::string::npos && row.find("colspan") == std::string::npos) {
                    rows.push_back(row);
                }
                row.clear();
                in_html_row = false;
            }
            continue;
        }
        if (starts_with_numeric_column(line)) {
            rows.push_back(line);
        }
    }
    if (in_html_row && row.find("<td") != std::string::npos && row.find("colspan") == std::string::npos) {
        rows.push_back(row);
    }
    return rows;
}

void process_ranked_record_file(
    const Options &options,
    std::ofstream &out,
    Counters &counters,
    const std::string &retrieved_utc,
    const std::string &path,
    std::string_view source,
    std::string_view source_kind,
    std::string_view source_url,
    std::string_view parser) {
    if (path.empty() || options.generic_record_limit == 0) {
        return;
    }
    std::ifstream in(path);
    if (!in) {
        return;
    }

    auto &counter = counters.family[std::string(source)];
    const auto rows = record_candidate_rows(in);
    for (const auto &line : rows) {
        if (counter.rows_written >= options.generic_record_limit) {
            break;
        }
        if (comment_or_blank(line)) {
            continue;
        }
        ++counter.rows_read;
        const auto tokens = numeric_tokens(line, counter);
        if (tokens.size() < 2) {
            ++counter.rows_bad;
            continue;
        }
        const std::uint64_t rank = tokens[0];
        const std::uint64_t n = tokens[1];
        if (rank > 100000 || n == 0) {
            ++counter.rows_bad;
            continue;
        }
        write_computed_target(options, out, counters, source, source_kind, rank, source_url, parser, n, retrieved_utc);
    }
}

void process_oliveira_record_file(
    const Options &options,
    std::ofstream &out,
    Counters &counters,
    const std::string &retrieved_utc,
    const std::string &path,
    std::string_view source,
    std::string_view source_kind,
    std::string_view source_url) {
    if (path.empty() || options.generic_record_limit == 0) {
        return;
    }
    std::ifstream in(path);
    if (!in) {
        return;
    }

    auto &counter = counters.family[std::string(source)];
    const auto rows = record_candidate_rows(in);
    std::uint64_t rank = 0;
    for (const auto &line : rows) {
        if (counter.rows_written >= options.generic_record_limit) {
            break;
        }
        if (comment_or_blank(line)) {
            continue;
        }
        ++counter.rows_read;
        const auto tokens = numeric_tokens(line, counter);
        if (tokens.size() < 2) {
            ++counter.rows_bad;
            continue;
        }
        const std::uint64_t n = tokens[0];
        if (n == 0) {
            ++counter.rows_bad;
            continue;
        }
        ++rank;
        write_computed_target(options, out, counters, source, source_kind, rank, source_url, "oliveira_record_file_v1", n, retrieved_utc);
    }
}

std::size_t rows_written(const Counters &counters) {
    std::size_t total = 0;
    for (const auto &[_, counter] : counters.family) {
        total += counter.rows_written;
    }
    return total;
}

std::size_t future_source_targets(const Counters &counters) {
    std::size_t total = 0;
    for (const auto &[_, counter] : counters.family) {
        total += counter.rows_skipped_above_max_n;
    }
    return total;
}

void write_family_counter_json(std::ofstream &out, const Counters &counters) {
    out << "  \"families\": {\n";
    std::size_t i = 0;
    for (const auto &[family, counter] : counters.family) {
        out << "    \"" << collatz::json_escape(family) << "\": {"
            << "\"rows_read\":" << counter.rows_read
            << ",\"rows_written\":" << counter.rows_written
            << ",\"future_source_targets\":" << counter.rows_skipped_above_max_n
            << ",\"rows_bad\":" << counter.rows_bad
            << ",\"rows_mismatch\":" << counter.rows_mismatch
            << ",\"overflow_tokens\":" << counter.overflow_tokens
            << "}";
        if (++i != counters.family.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  },\n";
}

void write_metadata(const Options &options, const Counters &counters) {
    collatz::ensure_parent_dir(options.metadata);
    std::ofstream out(options.metadata);
    if (!out) {
        throw std::runtime_error("failed to open metadata output: " + options.metadata);
    }
    out << "{\n"
        << "  \"dataset_type\": \"collatz_public_source_targets\",\n"
        << "  \"tool\": \"collatz_source_targets\",\n"
        << "  \"schema_version\": 2,\n"
        << "  \"created_utc\": \"" << collatz::now_iso8601() << "\",\n"
        << "  \"output\": \"" << collatz::json_escape(options.output) << "\",\n"
        << "  \"max_n\": " << options.max_n << ",\n"
        << "  \"max_steps\": " << options.max_steps << ",\n"
        << "  \"sources\": [\n"
        << "    {\"id\":\"OEIS_A006577\",\"url\":\"https://oeis.org/A006577/b006577.txt\",\"role\":\"total stopping-time validation\"},\n"
        << "    {\"id\":\"OEIS_A006884\",\"url\":\"https://oeis.org/A006884/b006884.txt\",\"role\":\"path-record validation\"},\n"
        << "    {\"id\":\"Roosendaal_path_records\",\"url\":\"https://www.ericr.nl/wondrous/pathrecs.html\",\"role\":\"path-record validation\"},\n"
        << "    {\"id\":\"Roosendaal_delay_records\",\"url\":\"https://www.ericr.nl/wondrous/delrecs.html\",\"role\":\"delay-record validation\"},\n"
        << "    {\"id\":\"Barina_path_records\",\"url\":\"https://pcbarina.fit.vutbr.cz/path-records.htm\",\"role\":\"path-record validation\"},\n"
        << "    {\"id\":\"Oliveira_e_Silva_max_excursion_records\",\"url\":\"https://sweet.ua.pt/tos/3x%2B1.html\",\"role\":\"maximum-excursion record validation\"},\n"
        << "    {\"id\":\"Oliveira_e_Silva_stopping_records\",\"url\":\"https://sweet.ua.pt/tos/3x%2B1.html\",\"role\":\"stopping-record validation\"}\n"
        << "  ],\n"
        << "  \"rows_written\": " << rows_written(counters) << ",\n"
        << "  \"source_family_count\": " << counters.families_written.size() << ",\n"
        << "  \"future_source_targets\": " << future_source_targets(counters) << ",\n"
        << "  \"skipped_overflow_or_unfinished\": " << counters.skipped_overflow_or_unfinished << ",\n";
    write_family_counter_json(out, counters);
    out << "  \"header\": \"source,n,total_steps,peak_low,source_kind,source_rank,source_url,retrieved_utc,parser\"\n"
        << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto options = parse_args(argc, argv);
        const std::string retrieved_utc = collatz::now_iso8601();
        collatz::ensure_parent_dir(options.output);
        std::ofstream out(options.output);
        if (!out) {
            throw std::runtime_error("failed to open output: " + options.output);
        }
        out << "source,n,total_steps,peak_low,source_kind,source_rank,source_url,retrieved_utc,parser\n";

        Counters counters;
        process_stopping_file(options, out, counters, retrieved_utc);
        process_oeis_path_record_file(options, out, counters, retrieved_utc);
        process_ranked_record_file(options, out, counters, retrieved_utc, options.roosendaal_path_records,
                                   kRoosendaalPathSource, "path_record",
                                   "https://www.ericr.nl/wondrous/pathrecs.html", "ranked_record_table_v1");
        process_ranked_record_file(options, out, counters, retrieved_utc, options.roosendaal_delay_records,
                                   kRoosendaalDelaySource, "delay_record",
                                   "https://www.ericr.nl/wondrous/delrecs.html", "ranked_record_table_v1");
        process_ranked_record_file(options, out, counters, retrieved_utc, options.barina_path_records,
                                   kBarinaPathSource, "path_record",
                                   "https://pcbarina.fit.vutbr.cz/path-records.htm", "ranked_record_table_v1");
        process_oliveira_record_file(options, out, counters, retrieved_utc, options.oliveira_max_excursion_records,
                                     kOliveiraPathSource, "maximum_excursion_record",
                                     "https://sweet.ua.pt/tos/3x%2B1.html");
        process_oliveira_record_file(options, out, counters, retrieved_utc, options.oliveira_stopping_records,
                                     kOliveiraStoppingSource, "stopping_record",
                                     "https://sweet.ua.pt/tos/3x%2B1.html");
        out.close();
        write_metadata(options, counters);

        std::cout << "source_targets=" << rows_written(counters)
                  << " source_families=" << counters.families_written.size()
                  << " future_source_targets=" << future_source_targets(counters)
                  << " output=" << options.output
                  << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
