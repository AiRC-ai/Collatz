#include "collatz/core.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ExpectedRow {
    std::string source;
    std::uint64_t n = 0;
    std::uint32_t total_steps = 0;
    std::uint64_t peak_low = 0;
};

std::vector<std::string> split_csv_line(const std::string &line) {
    std::vector<std::string> fields;
    std::string current;
    std::istringstream input(line);
    while (std::getline(input, current, ',')) {
        fields.push_back(current);
    }
    return fields;
}

std::vector<ExpectedRow> load_expected(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open source validation file: " + path);
    }

    std::vector<ExpectedRow> rows;
    std::string line;
    bool header = true;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (header) {
            header = false;
            continue;
        }
        const auto fields = split_csv_line(line);
        if (fields.size() < 4) {
            throw std::runtime_error("bad validation row: " + line);
        }
        ExpectedRow row;
        row.source = fields[0];
        row.n = collatz::parse_u64(fields[1]).value_or(0);
        row.total_steps = collatz::parse_u32(fields[2]).value_or(std::numeric_limits<std::uint32_t>::max());
        row.peak_low = collatz::parse_u64(fields[3]).value_or(0);
        if (row.n == 0 || row.total_steps == std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("bad numeric validation row: " + line);
        }
        rows.push_back(row);
    }
    return rows;
}

} // namespace

int main(int argc, char **argv) {
    std::string path = "data/source_validation/reference_samples.csv";
    if (argc == 3 && std::string(argv[1]) == "--samples") {
        path = argv[2];
    } else if (argc != 1) {
        std::cerr << "usage: collatz_validate_sources [--samples FILE]\n";
        return 1;
    }

    try {
        const auto rows = load_expected(path);
        std::size_t failures = 0;
        for (const auto &row : rows) {
            const auto feature = collatz::compute_feature(row.n, 10000000);
            const bool ok = feature.total_steps == row.total_steps &&
                            feature.peak.high == 0 &&
                            feature.peak.low == row.peak_low;
            std::cout << (ok ? "ok " : "FAIL ")
                      << row.source
                      << " n=" << row.n
                      << " steps expected=" << row.total_steps << " actual=" << feature.total_steps
                      << " peak expected=" << row.peak_low << " actual=" << feature.peak.low
                      << "\n";
            if (!ok) {
                ++failures;
            }
        }

        if (failures != 0) {
            std::cerr << failures << " source validation rows failed\n";
            return 1;
        }
        std::cout << "validated " << rows.size() << " source sample rows\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
