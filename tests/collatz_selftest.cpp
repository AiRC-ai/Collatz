#include "collatz/core.hpp"
#include "collatz/feature_io.hpp"
#include "collatz/ml.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    try {
        const std::array<std::uint32_t, 11> expected_steps = {
            0, 0, 1, 7, 2, 5, 8, 16, 3, 19, 6
        };
        for (std::uint64_t n = 1; n <= 10; ++n) {
            const auto feature = collatz::compute_feature(n, 1000000);
            require(feature.total_steps == expected_steps[n], "OEIS A006577 sample mismatch");
            require((feature.flags & collatz::FeatureReachedOne) != 0, "sample did not reach one");
            require(feature.checksum != 0, "checksum must be nonzero");
        }

        const auto seven = collatz::compute_feature(7, 1000000);
        require(seven.total_steps == 16, "7 should have 16 total steps");
        require(seven.peak.high == 0 && seven.peak.low == 52, "7 should peak at 52");

        const auto twenty_seven = collatz::compute_feature(27, 1000000);
        require(twenty_seven.total_steps == 111, "27 should have 111 total steps");
        require(twenty_seven.peak.high == 0 && twenty_seven.peak.low == 9232, "27 should peak at 9232");
        const auto record = collatz::to_binary_record(twenty_seven);
        require(record.n == 27, "binary record should preserve n");
        require(record.total_steps == 111, "binary record should preserve total steps");
        require(record.peak_low == 9232, "binary record should preserve peak");
        require(record.checksum == twenty_seven.checksum, "binary record should preserve checksum");

        const auto metrics = collatz::metric_vector(record);
        require(metrics.size() == collatz::kMetricVectorDims, "metric vector dimension mismatch");

        const auto parity_runs = collatz::parity_run_tokens(record);
        require(!parity_runs.empty(), "parity run tokens should not be empty");

        const auto sketch = collatz::log_path_sketch(27, 1000000, 32);
        require(sketch.size() == 32, "log path sketch dimension mismatch");

        const auto recurrence_a = collatz::recurrence_image(sketch, 16);
        const auto recurrence_b = collatz::recurrence_image(sketch, 16);
        require(recurrence_a.size() == 16 * 16, "recurrence image size mismatch");
        require(recurrence_a == recurrence_b, "recurrence image should be deterministic");

        const auto gaf = collatz::gramian_angular_field(sketch, 16);
        const auto mtf = collatz::markov_transition_field(sketch, 16, 8);
        const auto residues = collatz::residue_sequence(27, 1000000, 16, 32);
        const auto residue = collatz::residue_raster(residues, 16, 32);
        const auto parity = collatz::parity_raster(collatz::parity_bits_from_record(record), 16);
        require(gaf.size() == 16 * 16, "GAF image size mismatch");
        require(mtf.size() == 16 * 16, "MTF image size mismatch");
        require(residue.size() == 16 * 16, "residue raster size mismatch");
        require(parity.size() == 16 * 16, "parity raster size mismatch");

        std::cout << "collatz_selftest passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "collatz_selftest failed: " << error.what() << "\n";
        return 1;
    }
}
