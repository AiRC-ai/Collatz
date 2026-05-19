#include "collatz/core.hpp"
#include "collatz/feature_io.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage(std::ostream &out) {
    out << "usage: collatzctl <command> [options]\n\n"
        << "commands:\n"
        << "  sample N                  print one computed feature as JSON\n"
        << "  status [--progress FILE]  print latest scanner progress JSON\n"
        << "  inspect-bin FILE          print compact binary feature file metadata\n"
        << "  patch-bin-header FILE --range-end N [--max-steps N]\n"
        << "  sources                   print validation and research source list\n";
}

void print_sources() {
    std::cout
        << "Roosendaal: https://www.ericr.nl/wondrous/\n"
        << "Oliveira e Silva: https://sweet.ua.pt/tos/3x%2B1.html\n"
        << "Barina project: https://pcbarina.fit.vutbr.cz/\n"
        << "Barina CUDA repository: https://github.com/xbarin02/collatz/\n"
        << "Springer Barina paper: https://link.springer.com/article/10.1007/s11227-025-06961-0\n"
        << "OEIS A006577: https://oeis.org/A006577\n"
        << "OEIS A006884: https://oeis.org/A006884\n"
        << "Math StackExchange stopping curves: https://math.stackexchange.com/questions/4678861/collatz-stopping-time-curves\n"
        << "MDPI clustering paper: https://www.mdpi.com/2227-7390/9/4/314\n"
        << "Kaggle comparison dataset: https://www.kaggle.com/datasets/clmentscipion/collatz-sequences-and-metrics-dataset\n"
        << "McDaMastR simulator reference: https://github.com/McDaMastR/CollatzConjectureSimulator\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 2) {
            usage(std::cerr);
            return 1;
        }

        const std::string command = argv[1];
        if (command == "sample") {
            if (argc != 3) {
                throw std::runtime_error("sample requires N");
            }
            const auto n = collatz::parse_u64(argv[2]);
            if (!n || *n == 0) {
                throw std::runtime_error("N must be a positive uint64 integer");
            }
            std::cout << collatz::feature_to_json(collatz::compute_feature(*n, 10000000)) << "\n";
            return 0;
        }

        if (command == "status") {
            std::string progress = "logs/progress.jsonl";
            if (argc == 4 && std::string(argv[2]) == "--progress") {
                progress = argv[3];
            } else if (argc != 2) {
                throw std::runtime_error("status accepts only --progress FILE");
            }
            const auto line = collatz::read_last_nonempty_line(progress);
            if (line.empty()) {
                std::cout << "{\"status\":\"no progress yet\",\"progress\":\"" << collatz::json_escape(progress) << "\"}\n";
            } else {
                std::cout << line << "\n";
            }
            return 0;
        }

        if (command == "inspect-bin") {
            if (argc != 3) {
                throw std::runtime_error("inspect-bin requires FILE");
            }
            const auto header = collatz::read_binary_header(argv[2]);
            if (!collatz::valid_binary_header(header)) {
                throw std::runtime_error("invalid or incompatible binary feature header");
            }
            std::cout << "{\"file\":\"" << collatz::json_escape(argv[2]) << "\""
                      << ",\"version\":" << header.version
                      << ",\"header_size\":" << header.header_size
                      << ",\"record_size\":" << header.record_size
                      << ",\"range_start\":" << header.range_start
                      << ",\"range_end\":" << header.range_end
                      << ",\"max_steps\":" << header.max_steps
                      << ",\"created_utc\":\"" << collatz::json_escape(header.created_utc) << "\""
                      << ",\"records\":" << collatz::binary_record_count(argv[2])
                      << "}\n";
            return 0;
        }

        if (command == "patch-bin-header") {
            if (argc != 5 && argc != 7) {
                throw std::runtime_error("patch-bin-header requires FILE --range-end N [--max-steps N]");
            }
            const std::string path = argv[2];
            if (std::string(argv[3]) != "--range-end") {
                throw std::runtime_error("patch-bin-header requires --range-end N");
            }
            const auto range_end = collatz::parse_u64(argv[4]);
            if (!range_end || *range_end == 0) {
                throw std::runtime_error("--range-end must be a positive uint64 integer");
            }

            auto header = collatz::read_binary_header(path);
            if (!collatz::valid_binary_header(header)) {
                throw std::runtime_error("invalid or incompatible binary feature header");
            }
            std::uint32_t max_steps = header.max_steps;
            if (argc == 7) {
                if (std::string(argv[5]) != "--max-steps") {
                    throw std::runtime_error("patch-bin-header optional argument is --max-steps N");
                }
                const auto parsed_max_steps = collatz::parse_u32(argv[6]);
                if (!parsed_max_steps || *parsed_max_steps == 0) {
                    throw std::runtime_error("--max-steps must be a positive integer");
                }
                max_steps = *parsed_max_steps;
            }

            const auto records = collatz::binary_record_count(path);
            if (*range_end < header.range_start || *range_end - header.range_start + 1 < records) {
                throw std::runtime_error("--range-end would be smaller than the record count in the file");
            }
            collatz::update_binary_header(path, *range_end, max_steps);
            header = collatz::read_binary_header(path);
            std::cout << "{\"file\":\"" << collatz::json_escape(path) << "\""
                      << ",\"range_start\":" << header.range_start
                      << ",\"range_end\":" << header.range_end
                      << ",\"max_steps\":" << header.max_steps
                      << ",\"records\":" << records
                      << "}\n";
            return 0;
        }

        if (command == "sources") {
            print_sources();
            return 0;
        }

        throw std::runtime_error("unknown command: " + command);
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
