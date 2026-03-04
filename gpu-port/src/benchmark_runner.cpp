// Benchmark Runner for PDMRG vs PDMRG2
// Tests both algorithms with different stream counts
// Compares performance and accuracy

#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>

#define HIP_CHECK(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

struct BenchmarkResult {
    std::string algorithm;
    std::string problem;
    int n_streams;
    double energy;
    double time_sec;
    double error;
    double energy_per_site;
};

void print_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n====================================================\n";
    std::cout << "Benchmark Results Summary\n";
    std::cout << "====================================================\n\n";

    std::cout << std::setw(10) << "Algorithm"
              << std::setw(12) << "Problem"
              << std::setw(10) << "Streams"
              << std::setw(14) << "Energy"
              << std::setw(10) << "Time (s)"
              << std::setw(12) << "Error"
              << "\n";
    std::cout << std::string(68, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::setw(10) << r.algorithm
                  << std::setw(12) << r.problem
                  << std::setw(10) << r.n_streams
                  << std::setw(14) << std::fixed << std::setprecision(8) << r.energy
                  << std::setw(10) << std::fixed << std::setprecision(4) << r.time_sec
                  << std::setw(12) << std::scientific << std::setprecision(2) << r.error
                  << "\n";
    }

    std::cout << "\n";
}

void save_csv(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream f(filename);
    f << "algorithm,problem,streams,energy,time_sec,error,energy_per_site\n";

    for (const auto& r : results) {
        f << r.algorithm << ","
          << r.problem << ","
          << r.n_streams << ","
          << std::fixed << std::setprecision(12) << r.energy << ","
          << r.time_sec << ","
          << std::scientific << std::setprecision(10) << r.error << ","
          << std::fixed << std::setprecision(10) << r.energy_per_site << "\n";
    }

    std::cout << "Results saved to: " << filename << "\n";
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "PDMRG vs PDMRG2 Benchmark Suite\n";
    std::cout << "AMD MI300X GPU Performance Comparison\n";
    std::cout << "====================================================\n\n";

    // GPU info
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Memory: " << (prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0) << " GB\n";
    std::cout << "Compute Units: " << prop.multiProcessorCount << "\n\n";

    std::vector<BenchmarkResult> results;

    // Test configurations
    std::vector<std::string> algorithms = {"PDMRG", "PDMRG2"};
    std::vector<std::string> problems = {"heisenberg", "josephson"};
    std::vector<int> stream_counts = {1, 2, 4, 8};

    std::cout << "Running benchmarks...\n";
    std::cout << "Algorithms: PDMRG (Lanczos), PDMRG2 (Block-Davidson)\n";
    std::cout << "Problems: Heisenberg, Josephson\n";
    std::cout << "Stream counts: 1, 2, 4, 8\n\n";

    // Expected energies
    double heisenberg_expected = -5.317755183336;
    double josephson_expected = -18.0;  // Approximate for L=12

    for (const auto& algo : algorithms) {
        for (const auto& prob : problems) {
            for (int streams : stream_counts) {
                std::cout << "Running: " << algo << " / " << prob << " / " << streams << " streams...";
                std::cout.flush();

                BenchmarkResult result;
                result.algorithm = algo;
                result.problem = prob;
                result.n_streams = streams;

                // Build command
                std::string cmd;
                if (algo == "PDMRG") {
                    cmd = "./pdmrg_complete " + prob + " " + std::to_string(streams);
                } else {
                    cmd = "./pdmrg2_complete " + prob + " " + std::to_string(streams);
                }

                cmd += " > /tmp/benchmark_output.txt 2>&1";

                auto t_start = std::chrono::high_resolution_clock::now();
                int status = system(cmd.c_str());
                auto t_end = std::chrono::high_resolution_clock::now();

                if (status != 0) {
                    std::cout << " FAILED\n";
                    continue;
                }

                result.time_sec = std::chrono::duration<double>(t_end - t_start).count();

                // Parse output for energy
                std::ifstream output("/tmp/benchmark_output.txt");
                std::string line;
                bool found = false;

                while (std::getline(output, line)) {
                    if (line.find("Final E:") != std::string::npos) {
                        size_t pos = line.find(":");
                        if (pos != std::string::npos) {
                            result.energy = std::stod(line.substr(pos + 1));
                            found = true;
                        }
                    }
                }

                if (!found) {
                    std::cout << " NO ENERGY FOUND\n";
                    continue;
                }

                double expected = (prob == "heisenberg") ? heisenberg_expected : josephson_expected;
                result.error = std::abs(result.energy - expected);
                result.energy_per_site = result.energy / 11.0;  // L-1 = 11 bonds

                results.push_back(result);

                std::cout << " DONE (E=" << std::fixed << std::setprecision(6) << result.energy
                          << ", t=" << std::setprecision(3) << result.time_sec << "s)\n";
            }
        }
    }

    std::cout << "\n";
    print_results(results);
    save_csv(results, "benchmark_results.csv");

    // Performance analysis
    std::cout << "\n====================================================\n";
    std::cout << "Performance Analysis\n";
    std::cout << "====================================================\n\n";

    // Compare PDMRG vs PDMRG2 speedup
    for (const auto& prob : problems) {
        std::cout << "Problem: " << prob << "\n";
        for (int streams : stream_counts) {
            double pdmrg_time = 0, pdmrg2_time = 0;

            for (const auto& r : results) {
                if (r.problem == prob && r.n_streams == streams) {
                    if (r.algorithm == "PDMRG") pdmrg_time = r.time_sec;
                    if (r.algorithm == "PDMRG2") pdmrg2_time = r.time_sec;
                }
            }

            if (pdmrg_time > 0 && pdmrg2_time > 0) {
                double speedup = pdmrg_time / pdmrg2_time;
                std::cout << "  Streams " << streams << ": PDMRG2 is "
                          << std::fixed << std::setprecision(2) << speedup << "x "
                          << (speedup > 1 ? "faster" : "slower") << " than PDMRG\n";
            }
        }
        std::cout << "\n";
    }

    // Stream scaling
    std::cout << "Stream Scaling Efficiency:\n";
    for (const auto& algo : algorithms) {
        for (const auto& prob : problems) {
            std::cout << algo << " / " << prob << ":\n";

            double time_1 = 0;
            for (const auto& r : results) {
                if (r.algorithm == algo && r.problem == prob && r.n_streams == 1) {
                    time_1 = r.time_sec;
                    break;
                }
            }

            if (time_1 == 0) continue;

            for (int streams : stream_counts) {
                if (streams == 1) continue;

                for (const auto& r : results) {
                    if (r.algorithm == algo && r.problem == prob && r.n_streams == streams) {
                        double speedup = time_1 / r.time_sec;
                        double efficiency = speedup / streams * 100;
                        std::cout << "  " << streams << " streams: " << std::fixed << std::setprecision(2)
                                  << speedup << "x speedup (" << efficiency << "% efficiency)\n";
                    }
                }
            }
            std::cout << "\n";
        }
    }

    std::cout << "====================================================\n";

    return 0;
}
