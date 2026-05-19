#pragma once

#include "collatz/core.hpp"
#include "collatz/feature_io.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace collatz {

constexpr std::size_t kMetricVectorDims = 37;
constexpr std::size_t kDefaultSketchLength = 128;
constexpr std::size_t kDefaultImageSize = 64;

std::vector<double> metric_vector(const BinaryFeatureRecord &record);
std::vector<std::uint16_t> parity_run_tokens(const BinaryFeatureRecord &record, std::size_t max_steps = kParityPrefixBits);
std::vector<std::uint8_t> parity_bits_from_record(const BinaryFeatureRecord &record, std::size_t max_bits = kParityPrefixBits);
std::vector<double> log_path_sketch(std::uint64_t n, std::uint32_t max_steps, std::size_t dims);
std::vector<std::uint8_t> residue_sequence(std::uint64_t n, std::uint32_t max_steps, std::size_t dims, std::uint8_t modulus);

std::vector<std::uint8_t> recurrence_image(const std::vector<double> &series, std::size_t size);
std::vector<std::uint8_t> gramian_angular_field(const std::vector<double> &series, std::size_t size);
std::vector<std::uint8_t> markov_transition_field(const std::vector<double> &series, std::size_t size, std::size_t bins);
std::vector<std::uint8_t> parity_raster(const std::vector<std::uint8_t> &bits, std::size_t size);
std::vector<std::uint8_t> residue_raster(const std::vector<std::uint8_t> &residues, std::size_t size, std::uint8_t modulus);
void write_pgm(const std::string &path, const std::vector<std::uint8_t> &pixels, std::size_t width, std::size_t height);

} // namespace collatz
