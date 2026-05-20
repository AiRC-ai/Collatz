#include "collatz/feature_io.hpp"
#include "collatz/ml.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::uint64_t n = 27;
    std::string output = "docs/media/path-image-atlas.svg";
    std::uint32_t max_steps = 1000000;
    std::size_t size = 32;
    std::size_t mtf_bins = 16;
    std::uint8_t residue_modulus = 32;
};

struct Panel {
    std::string title;
    std::string subtitle;
    std::vector<std::uint8_t> pixels;
};

void usage(std::ostream &out) {
    out << "usage: collatz_path_image_atlas [options]\n\n"
        << "Render a README SVG atlas from the real Collatz path-image encoders.\n\n"
        << "options:\n"
        << "  --n N                 start value (default 27)\n"
        << "  --output FILE         SVG output (default docs/media/path-image-atlas.svg)\n"
        << "  --max-steps N         max path steps (default 1000000)\n"
        << "  --size N              generated image size in pixels (default 32)\n"
        << "  --mtf-bins N          Markov transition bins (default 16)\n"
        << "  --residue-modulus N   residue raster modulus (default 32)\n";
}

std::size_t parse_size_arg(const std::string &value, const char *name) {
    const auto parsed = collatz::parse_u64(value);
    if (!parsed || *parsed == 0 || *parsed > 128) {
        throw std::runtime_error(std::string(name) + " must be in [1,128]");
    }
    return static_cast<std::size_t>(*parsed);
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

        if (arg == "--n") {
            const auto value = collatz::parse_u64(need_value("--n"));
            if (!value || *value == 0) {
                throw std::runtime_error("--n must be a positive uint64 integer");
            }
            options.n = *value;
        } else if (arg == "--output") {
            options.output = need_value("--output");
        } else if (arg == "--max-steps") {
            const auto value = collatz::parse_u32(need_value("--max-steps"));
            if (!value || *value == 0) {
                throw std::runtime_error("--max-steps must be a positive integer");
            }
            options.max_steps = *value;
        } else if (arg == "--size") {
            options.size = parse_size_arg(need_value("--size"), "--size");
        } else if (arg == "--mtf-bins") {
            options.mtf_bins = parse_size_arg(need_value("--mtf-bins"), "--mtf-bins");
        } else if (arg == "--residue-modulus") {
            const auto value = collatz::parse_u32(need_value("--residue-modulus"));
            if (!value || *value < 2 || *value > 255) {
                throw std::runtime_error("--residue-modulus must be in [2,255]");
            }
            options.residue_modulus = static_cast<std::uint8_t>(*value);
        } else if (arg == "--help" || arg == "-h") {
            usage(std::cout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
}

std::string hex2(int value) {
    std::ostringstream out;
    out << std::hex << std::setw(2) << std::setfill('0') << std::clamp(value, 0, 255);
    return out.str();
}

std::string color_for(std::uint8_t value) {
    const double t = static_cast<double>(value) / 255.0;
    const int r0 = 5;
    const int g0 = 8;
    const int b0 = 22;
    const int r1 = 94;
    const int g1 = 234;
    const int b1 = 212;
    const int r2 = 250;
    const int g2 = 204;
    const int b2 = 21;

    int r = 0;
    int g = 0;
    int b = 0;
    if (t < 0.68) {
        const double u = t / 0.68;
        r = static_cast<int>(r0 + (r1 - r0) * u);
        g = static_cast<int>(g0 + (g1 - g0) * u);
        b = static_cast<int>(b0 + (b1 - b0) * u);
    } else {
        const double u = (t - 0.68) / 0.32;
        r = static_cast<int>(r1 + (r2 - r1) * u);
        g = static_cast<int>(g1 + (g2 - g1) * u);
        b = static_cast<int>(b1 + (b2 - b1) * u);
    }
    return "#" + hex2(r) + hex2(g) + hex2(b);
}

std::string fixed(double value, int digits = 2) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(digits) << value;
    return out.str();
}

void render_pixels(std::ostream &out, const Panel &panel, std::size_t size, int x, int y, int pixel_size) {
    out << "    <g transform=\"translate(" << x << "," << y << ")\">\n";
    out << "      <rect width=\"" << (size * pixel_size) << "\" height=\"" << (size * pixel_size)
        << "\" fill=\"#050816\"/>\n";
    for (std::size_t row = 0; row < size; ++row) {
        for (std::size_t col = 0; col < size; ++col) {
            const auto value = panel.pixels[row * size + col];
            out << "      <rect x=\"" << (col * pixel_size) << "\" y=\"" << (row * pixel_size)
                << "\" width=\"" << pixel_size << "\" height=\"" << pixel_size
                << "\" fill=\"" << color_for(value) << "\"/>\n";
        }
    }
    out << "    </g>\n";
}

void write_svg(
    const Options &options,
    const collatz::FeatureRow &feature,
    const std::vector<Panel> &panels) {
    const int panel_width = 204;
    const int panel_height = 250;
    const int margin = 54;
    const int gap = 22;
    const int pixel_size = 4;
    const int image_size = static_cast<int>(options.size) * pixel_size;
    const int width = margin * 2 + static_cast<int>(panels.size()) * panel_width + static_cast<int>(panels.size() - 1) * gap;
    const int height = 420;

    collatz::ensure_parent_dir(options.output);
    std::ofstream out(options.output);
    if (!out) {
        throw std::runtime_error("failed to open SVG output: " + options.output);
    }

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height
        << "\" viewBox=\"0 0 " << width << ' ' << height
        << "\" role=\"img\" aria-labelledby=\"title desc\">\n"
        << "  <title id=\"title\">Real Collatz path-image atlas for n=" << feature.n << "</title>\n"
        << "  <desc id=\"desc\">Generated from the actual Collatz path-image encoders: recurrence plot, Gramian Angular Field, Markov Transition Field, parity raster, and residue raster.</desc>\n"
        << "  <defs>\n"
        << "    <linearGradient id=\"bg\" x1=\"0\" x2=\"1\" y1=\"0\" y2=\"1\"><stop offset=\"0\" stop-color=\"#090e1b\"/><stop offset=\"1\" stop-color=\"#121b33\"/></linearGradient>\n"
        << "  </defs>\n"
        << "  <rect width=\"" << width << "\" height=\"" << height << "\" fill=\"url(#bg)\"/>\n"
        << "  <text x=\"54\" y=\"48\" fill=\"#f8fafc\" font-family=\"Inter,ui-sans-serif,system-ui,sans-serif\" font-size=\"28\" font-weight=\"800\">Real Path-As-Image Tensor Atlas</text>\n"
        << "  <text x=\"54\" y=\"76\" fill=\"#aab6d3\" font-family=\"Inter,ui-sans-serif,system-ui,sans-serif\" font-size=\"14\">n="
        << feature.n << ", total steps=" << feature.total_steps << ", peak=" << feature.peak.low
        << ", peak log2=" << fixed(feature.peak_log2, 3) << ". Every pixel is generated from the computed trajectory.</text>\n";

    for (std::size_t i = 0; i < panels.size(); ++i) {
        const int panel_x = margin + static_cast<int>(i) * (panel_width + gap);
        const int panel_y = 112;
        out << "  <g transform=\"translate(" << panel_x << "," << panel_y << ")\">\n"
            << "    <rect width=\"" << panel_width << "\" height=\"" << panel_height
            << "\" rx=\"8\" fill=\"#080c18\" stroke=\"#2b385e\"/>\n"
            << "    <text x=\"14\" y=\"26\" fill=\"#dbe7ff\" font-family=\"Inter,ui-sans-serif,system-ui,sans-serif\" font-size=\"13\" font-weight=\"800\">"
            << panels[i].title << "</text>\n"
            << "    <text x=\"14\" y=\"44\" fill=\"#8ea0c4\" font-family=\"Inter,ui-sans-serif,system-ui,sans-serif\" font-size=\"10\">"
            << panels[i].subtitle << "</text>\n";
        const int image_x = 14 + (panel_width - 28 - image_size) / 2;
        render_pixels(out, panels[i], options.size, image_x, 62, pixel_size);
        out << "  </g>\n";
    }

    out << "  <text x=\"54\" y=\"386\" fill=\"#7384ad\" font-family=\"Inter,ui-sans-serif,system-ui,sans-serif\" font-size=\"12\">Generated by build/collatz_path_image_atlas using the same C++ encoder functions as collatz_path_image; this is data representation, not a proof.</text>\n"
        << "</svg>\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto options = parse_args(argc, argv);
        const auto feature = collatz::compute_feature(options.n, options.max_steps);
        if ((feature.flags & collatz::FeatureReachedOne) == 0 || (feature.flags & collatz::FeatureOverflow) != 0 ||
            (feature.flags & collatz::FeatureMaxSteps) != 0) {
            throw std::runtime_error("start value did not produce a usable bounded trajectory");
        }
        const auto record = collatz::to_binary_record(feature);
        const auto sketch = collatz::log_path_sketch(options.n, options.max_steps, options.size);
        const auto parity_bits = collatz::parity_bits_from_record(record);
        const auto residues = collatz::residue_sequence(options.n, options.max_steps, options.size, options.residue_modulus);

        std::vector<Panel> panels;
        panels.push_back({"Recurrence Plot", "|log path_i - log path_j|", collatz::recurrence_image(sketch, options.size)});
        panels.push_back({"Gramian Angular Field", "angular transform of log path", collatz::gramian_angular_field(sketch, options.size)});
        panels.push_back({"Markov Transition Field", "transition matrix over log bins", collatz::markov_transition_field(sketch, options.size, options.mtf_bins)});
        panels.push_back({"Parity Raster", "even/odd prefix repeated as pixels", collatz::parity_raster(parity_bits, options.size)});
        panels.push_back({"Residue Raster", "path values modulo residue base", collatz::residue_raster(residues, options.size, options.residue_modulus)});

        write_svg(options, feature, panels);
        std::cout << "path_image_atlas=" << options.output
                  << " n=" << feature.n
                  << " total_steps=" << feature.total_steps
                  << " peak=" << feature.peak.low
                  << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
