#include "collatz/core.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::uint64_t n = 0;
    std::uint32_t max_steps = 1000000;
    std::size_t width = 100;
    std::size_t height = 24;
    std::string csv;
};

void usage(std::ostream &out) {
    out << "usage: collatz_plot N [--width N] [--height N] [--max-steps N] [--csv FILE]\n";
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
        if (arg == "--width") {
            auto value = collatz::parse_u64(need_value("--width"));
            if (!value || *value < 10) {
                throw std::runtime_error("--width must be at least 10");
            }
            options.width = static_cast<std::size_t>(*value);
        } else if (arg == "--height") {
            auto value = collatz::parse_u64(need_value("--height"));
            if (!value || *value < 5) {
                throw std::runtime_error("--height must be at least 5");
            }
            options.height = static_cast<std::size_t>(*value);
        } else if (arg == "--max-steps") {
            auto value = collatz::parse_u32(need_value("--max-steps"));
            if (!value || *value == 0) {
                throw std::runtime_error("--max-steps must be positive");
            }
            options.max_steps = *value;
        } else if (arg == "--csv") {
            options.csv = need_value("--csv");
        } else if (arg == "--help" || arg == "-h") {
            usage(std::cout);
            std::exit(0);
        } else if (options.n == 0) {
            auto value = collatz::parse_u64(arg);
            if (!value || *value == 0) {
                throw std::runtime_error("N must be a positive uint64 integer for the C++ fast path");
            }
            options.n = *value;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.n == 0) {
        throw std::runtime_error("N is required");
    }
    return options;
}

void write_csv(const std::string &path, const std::vector<collatz::PathPoint> &points) {
    collatz::ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open CSV: " + path);
    }
    out << "step,value,log2_value\n";
    for (const auto &point : points) {
        out << point.step << ','
            << collatz::uint128_to_decimal(point.value) << ','
            << static_cast<double>(point.log2_value) << '\n';
    }
}

void print_plot(const std::vector<collatz::PathPoint> &points, std::size_t width, std::size_t height) {
    if (points.empty()) {
        return;
    }

    long double max_log = 0.0L;
    for (const auto &point : points) {
        max_log = std::max(max_log, point.log2_value);
    }
    if (max_log <= 0.0L) {
        max_log = 1.0L;
    }

    std::vector<std::string> canvas(height, std::string(width, ' '));
    for (std::size_t x = 0; x < width; ++x) {
        const std::size_t index = points.size() == 1
            ? 0
            : (x * (points.size() - 1)) / (width - 1);
        const auto y_scaled = static_cast<std::size_t>((points[index].log2_value / max_log) * static_cast<long double>(height - 1));
        const std::size_t y = height - 1 - std::min<std::size_t>(y_scaled, height - 1);
        canvas[y][x] = '#';
    }

    for (const auto &line : canvas) {
        std::cout << line << '\n';
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_args(argc, argv);
        bool overflow = false;
        const auto points = collatz::generate_path(options.n, options.max_steps, &overflow);
        const auto feature = collatz::compute_feature(options.n, options.max_steps);

        std::cout << "n: " << options.n << "\n"
                  << "steps: " << feature.total_steps << "\n"
                  << "peak: " << collatz::uint128_to_decimal((static_cast<collatz::UInt128>(feature.peak.high) << 64) | feature.peak.low) << "\n"
                  << "peak step: " << feature.peak_step << "\n"
                  << "flags: " << feature.flags << "\n";
        if (overflow) {
            std::cout << "warning: path stopped before completion because the uint128 fast path overflowed\n";
        }

        print_plot(points, options.width, options.height);
        if (!options.csv.empty()) {
            write_csv(options.csv, points);
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
