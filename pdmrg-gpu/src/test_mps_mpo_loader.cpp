// Test program for MPS/MPO binary file loader
// Verifies that data serialized by Python can be correctly loaded in C++

#include "../include/mps_mpo_loader.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

void print_tensor_summary(const MPSTensor& t, int site) {
    std::cout << "  Site " << site << ": shape (" << t.D_left << ", " << t.d << ", " << t.D_right << ")\n";

    // Print some statistics
    double norm2 = 0.0;
    double max_abs = 0.0;
    for (const auto& z : t.data) {
        norm2 += std::norm(z);
        max_abs = std::max(max_abs, std::abs(z));
    }

    std::cout << "    Norm: " << std::sqrt(norm2) << "\n";
    std::cout << "    Max |element|: " << max_abs << "\n";

    // Print first few elements
    std::cout << "    First 3 elements: ";
    for (size_t i = 0; i < std::min(size_t(3), t.data.size()); ++i) {
        std::cout << t.data[i] << " ";
    }
    std::cout << "\n";
}

void print_tensor_summary(const MPOTensor& t, int site) {
    std::cout << "  Site " << site << ": shape (" << t.D_mpo_left << ", "
              << t.d_bra << ", " << t.d_ket << ", " << t.D_mpo_right << ")\n";

    // Print some statistics
    double norm2 = 0.0;
    double max_abs = 0.0;
    for (const auto& z : t.data) {
        norm2 += std::norm(z);
        max_abs = std::max(max_abs, std::abs(z));
    }

    std::cout << "    Norm: " << std::sqrt(norm2) << "\n";
    std::cout << "    Max |element|: " << max_abs << "\n";

    // Print first few elements
    std::cout << "    First 3 elements: ";
    for (size_t i = 0; i < std::min(size_t(3), t.data.size()); ++i) {
        std::cout << t.data[i] << " ";
    }
    std::cout << "\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mps|mpo> <filepath>\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  " << argv[0] << " mps benchmark_data/heisenberg_L12_chi10_mps.bin\n";
        std::cerr << "  " << argv[0] << " mpo benchmark_data/heisenberg_L12_mpo.bin\n";
        return 1;
    }

    std::string type = argv[1];
    std::string filepath = argv[2];

    std::cout << std::fixed << std::setprecision(6);

    try {
        if (type == "mps") {
            std::cout << "\n=== Loading MPS ===\n\n";
            auto tensors = MPSLoader::load(filepath);

            std::cout << "\n=== Tensor Summary ===\n\n";
            for (size_t i = 0; i < tensors.size(); ++i) {
                print_tensor_summary(tensors[i], i);
            }

            // Compute total number of parameters
            size_t total_params = 0;
            for (const auto& t : tensors) {
                total_params += t.total_elements();
            }
            std::cout << "\nTotal parameters: " << total_params << "\n";

        } else if (type == "mpo") {
            std::cout << "\n=== Loading MPO ===\n\n";
            auto tensors = MPOLoader::load(filepath);

            std::cout << "\n=== Tensor Summary ===\n\n";
            for (size_t i = 0; i < tensors.size(); ++i) {
                print_tensor_summary(tensors[i], i);
            }

            // Compute total number of parameters
            size_t total_params = 0;
            for (const auto& t : tensors) {
                total_params += t.total_elements();
            }
            std::cout << "\nTotal parameters: " << total_params << "\n";

        } else {
            std::cerr << "Error: type must be 'mps' or 'mpo'\n";
            return 1;
        }

        std::cout << "\n=== Success ===\n";
        std::cout << "Loaded data from " << filepath << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
