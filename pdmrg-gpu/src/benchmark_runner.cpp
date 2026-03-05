// Benchmark Runner for PDMRG vs PDMRG2
// Tests with 1,2,4,8 streams and compares performance

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>

int main() {
    std::cout << "\n====================================================\n";
    std::cout << "  PDMRG vs PDMRG2 Benchmark - MI300X\n";
    std::cout << "====================================================\n\n";

    std::vector<int> streams = {1, 2, 4, 8};
    std::vector<std::string> problems = {"heisenberg", "josephson"};

    for (const auto& prob : problems) {
        std::cout << "\n=== " << prob << " ===\n\n";

        std::cout << "PDMRG (Lanczos/BLAS-2):\n";
        for (int s : streams) {
            std::string cmd = "./pdmrg_complete " + prob + " " + std::to_string(s);
            std::cout << "  Streams=" << s << ": ";
            system(cmd.c_str());
        }

        std::cout << "\nPDMRG2 (Block-Davidson/BLAS-3):\n";
        for (int s : streams) {
            std::string cmd = "./pdmrg2_complete " + prob + " " + std::to_string(s);
            std::cout << "  Streams=" << s << ": ";
            system(cmd.c_str());
        }
    }

    return 0;
}
