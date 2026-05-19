#include "collatz/ml.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace collatz {
namespace {

double clamp01(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

std::vector<double> normalize_series(const std::vector<double> &series, std::size_t size) {
    std::vector<double> sampled(size, 0.0);
    if (series.empty()) {
        return sampled;
    }
    for (std::size_t i = 0; i < size; ++i) {
        const std::size_t index = size == 1 ? 0 : (i * (series.size() - 1)) / (size - 1);
        sampled[i] = series[index];
    }

    const auto [min_it, max_it] = std::minmax_element(sampled.begin(), sampled.end());
    const double min_value = *min_it;
    const double max_value = *max_it;
    const double span = max_value - min_value;
    if (span <= std::numeric_limits<double>::epsilon()) {
        std::fill(sampled.begin(), sampled.end(), 0.5);
        return sampled;
    }
    for (auto &value : sampled) {
        value = (value - min_value) / span;
    }
    return sampled;
}

std::vector<double> series_for_image(const std::vector<double> &series, std::size_t size) {
    return normalize_series(series, size);
}

std::uint8_t to_byte(double value) {
    return static_cast<std::uint8_t>(std::lround(clamp01(value) * 255.0));
}

} // namespace

std::vector<double> metric_vector(const BinaryFeatureRecord &record) {
    std::vector<double> out;
    out.reserve(kMetricVectorDims);
    const double total_steps = static_cast<double>(record.total_steps);
    const double first_drop = static_cast<double>(record.first_drop_time);
    const double odd_steps = static_cast<double>(record.odd_steps);
    const double even_steps = static_cast<double>(record.even_steps);
    const double accelerated = static_cast<double>(record.accelerated_steps);
    const double peak_step = static_cast<double>(record.peak_step);
    const double denom = std::max(1.0, total_steps);
    const double input_bits = std::max(1.0, std::log2(static_cast<double>(record.n)));

    out.push_back(std::log2(static_cast<double>(record.n)) / 64.0);
    out.push_back(total_steps / 4096.0);
    out.push_back(first_drop / 4096.0);
    out.push_back(odd_steps / denom);
    out.push_back(even_steps / denom);
    out.push_back(accelerated / denom);
    out.push_back(peak_step / denom);
    out.push_back(record.peak_log2 / 128.0);
    out.push_back(record.peak_ratio_log2 / 128.0);
    out.push_back(record.steps_per_input_bit / std::max(1.0, input_bits));
    out.push_back(static_cast<double>(record.residue_mod3) / 2.0);
    out.push_back(static_cast<double>(record.residue_mod4) / 3.0);
    out.push_back(static_cast<double>(record.residue_mod8) / 7.0);
    out.push_back(static_cast<double>(record.residue_mod16) / 15.0);
    out.push_back(static_cast<double>(record.residue_mod32) / 31.0);
    out.push_back(static_cast<double>(record.flags & FeatureReachedOne ? 1 : 0));
    out.push_back(static_cast<double>(record.flags & FeatureFirstDropKnown ? 1 : 0));

    double histogram_total = 0.0;
    for (const auto count : record.halving_histogram) {
        histogram_total += count;
    }
    histogram_total = std::max(1.0, histogram_total);
    for (const auto count : record.halving_histogram) {
        out.push_back(static_cast<double>(count) / histogram_total);
    }

    const auto parity_bits = parity_bits_from_record(record);
    const double parity_density = std::accumulate(parity_bits.begin(), parity_bits.end(), 0.0) /
                                  std::max<std::size_t>(1, parity_bits.size());
    out.push_back(parity_density);
    out.push_back(static_cast<double>(record.checksum & 0xffffu) / 65535.0);
    out.push_back(static_cast<double>((record.checksum >> 16) & 0xffffu) / 65535.0);
    out.push_back(static_cast<double>((record.checksum >> 32) & 0xffffu) / 65535.0);

    return out;
}

std::vector<std::uint8_t> parity_bits_from_record(const BinaryFeatureRecord &record, std::size_t max_bits) {
    const std::size_t count = std::min<std::size_t>({max_bits, kParityPrefixBits, std::max<std::uint32_t>(1, record.total_steps)});
    std::vector<std::uint8_t> bits(count, 0);
    for (std::size_t i = 0; i < count; ++i) {
        const auto word = i / 64u;
        const auto bit = i % 64u;
        bits[i] = static_cast<std::uint8_t>((record.parity_prefix[word] >> bit) & 1u);
    }
    return bits;
}

std::vector<std::uint16_t> parity_run_tokens(const BinaryFeatureRecord &record, std::size_t max_steps) {
    const auto bits = parity_bits_from_record(record, max_steps);
    std::vector<std::uint16_t> runs;
    if (bits.empty()) {
        return runs;
    }

    std::uint8_t current = bits.front();
    std::uint16_t length = 0;
    for (const auto bit : bits) {
        if (bit == current && length < 32767) {
            ++length;
            continue;
        }
        runs.push_back(static_cast<std::uint16_t>((current ? 0x8000u : 0u) | length));
        current = bit;
        length = 1;
    }
    runs.push_back(static_cast<std::uint16_t>((current ? 0x8000u : 0u) | length));
    return runs;
}

std::vector<double> log_path_sketch(std::uint64_t n, std::uint32_t max_steps, std::size_t dims) {
    bool overflow = false;
    const auto path = generate_path(n, max_steps, &overflow);
    std::vector<double> logs;
    logs.reserve(path.size());
    for (const auto &point : path) {
        logs.push_back(static_cast<double>(point.log2_value));
    }
    return normalize_series(logs, dims);
}

std::vector<std::uint8_t> residue_sequence(std::uint64_t n, std::uint32_t max_steps, std::size_t dims, std::uint8_t modulus) {
    bool overflow = false;
    const auto path = generate_path(n, max_steps, &overflow);
    std::vector<std::uint8_t> residues(dims, 0);
    if (path.empty() || modulus == 0) {
        return residues;
    }
    for (std::size_t i = 0; i < dims; ++i) {
        const std::size_t index = dims == 1 ? 0 : (i * (path.size() - 1)) / (dims - 1);
        residues[i] = static_cast<std::uint8_t>(path[index].value % modulus);
    }
    return residues;
}

std::vector<std::uint8_t> recurrence_image(const std::vector<double> &series, std::size_t size) {
    const auto values = series_for_image(series, size);
    std::vector<std::uint8_t> pixels(size * size, 0);
    for (std::size_t y = 0; y < size; ++y) {
        for (std::size_t x = 0; x < size; ++x) {
            pixels[y * size + x] = to_byte(1.0 - std::abs(values[x] - values[y]));
        }
    }
    return pixels;
}

std::vector<std::uint8_t> gramian_angular_field(const std::vector<double> &series, std::size_t size) {
    auto values = series_for_image(series, size);
    for (auto &value : values) {
        value = value * 2.0 - 1.0;
    }

    std::vector<double> phi(size, 0.0);
    for (std::size_t i = 0; i < size; ++i) {
        phi[i] = std::acos(std::clamp(values[i], -1.0, 1.0));
    }

    std::vector<std::uint8_t> pixels(size * size, 0);
    for (std::size_t y = 0; y < size; ++y) {
        for (std::size_t x = 0; x < size; ++x) {
            const double value = (std::cos(phi[x] + phi[y]) + 1.0) * 0.5;
            pixels[y * size + x] = to_byte(value);
        }
    }
    return pixels;
}

std::vector<std::uint8_t> markov_transition_field(const std::vector<double> &series, std::size_t size, std::size_t bins) {
    const auto values = series_for_image(series, size);
    bins = std::max<std::size_t>(2, bins);
    std::vector<std::size_t> states(size, 0);
    for (std::size_t i = 0; i < size; ++i) {
        states[i] = std::min<std::size_t>(bins - 1, static_cast<std::size_t>(values[i] * bins));
    }

    std::vector<double> transitions(bins * bins, 0.0);
    for (std::size_t i = 1; i < states.size(); ++i) {
        transitions[states[i - 1] * bins + states[i]] += 1.0;
    }
    for (std::size_t from = 0; from < bins; ++from) {
        double row_total = 0.0;
        for (std::size_t to = 0; to < bins; ++to) {
            row_total += transitions[from * bins + to];
        }
        if (row_total > 0.0) {
            for (std::size_t to = 0; to < bins; ++to) {
                transitions[from * bins + to] /= row_total;
            }
        }
    }

    std::vector<std::uint8_t> pixels(size * size, 0);
    for (std::size_t y = 0; y < size; ++y) {
        for (std::size_t x = 0; x < size; ++x) {
            pixels[y * size + x] = to_byte(transitions[states[y] * bins + states[x]]);
        }
    }
    return pixels;
}

std::vector<std::uint8_t> parity_raster(const std::vector<std::uint8_t> &bits, std::size_t size) {
    std::vector<std::uint8_t> pixels(size * size, 0);
    if (bits.empty()) {
        return pixels;
    }
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = bits[i % bits.size()] ? 255 : 32;
    }
    return pixels;
}

std::vector<std::uint8_t> residue_raster(const std::vector<std::uint8_t> &residues, std::size_t size, std::uint8_t modulus) {
    std::vector<std::uint8_t> pixels(size * size, 0);
    if (residues.empty() || modulus == 0) {
        return pixels;
    }
    const double denom = std::max(1, static_cast<int>(modulus) - 1);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = to_byte(static_cast<double>(residues[i % residues.size()]) / denom);
    }
    return pixels;
}

void write_pgm(const std::string &path, const std::vector<std::uint8_t> &pixels, std::size_t width, std::size_t height) {
    if (pixels.size() != width * height) {
        throw std::runtime_error("PGM pixel buffer size does not match dimensions");
    }
    ensure_parent_dir(path);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open image output: " + path);
    }
    out << "P5\n" << width << ' ' << height << "\n255\n";
    out.write(reinterpret_cast<const char *>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
}

} // namespace collatz
