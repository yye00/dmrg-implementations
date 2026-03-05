#include "stream_coordinator.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <iostream>

// Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error in %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            throw std::runtime_error("HIP error"); \
        } \
    } while(0)

//==============================================================================
// StreamCoordinator Implementation
//==============================================================================

StreamCoordinator::StreamCoordinator(int n_streams, int chain_length,
                                     int chi_max, int d, int D_mpo, int max_bond)
    : n_streams_(n_streams), chain_length_(chain_length),
      chi_max_(chi_max), d_(d), D_mpo_(D_mpo), max_bond_(max_bond),
      total_energy_(0.0)
{
    if (n_streams_ < 1) {
        throw std::runtime_error("StreamCoordinator: n_streams must be >= 1");
    }

    if (chain_length_ < n_streams_ * 2) {
        throw std::runtime_error("StreamCoordinator: chain too short for n_streams");
    }

    // Create HIP streams
    streams_.resize(n_streams_);
    for (int i = 0; i < n_streams_; i++) {
        HIP_CHECK(hipStreamCreate(&streams_[i]));
    }

    // Create boundary events
    boundary_events_.resize(n_streams_ - 1);  // n-1 boundaries between n segments
    for (int i = 0; i < n_streams_ - 1; i++) {
        HIP_CHECK(hipEventCreate(&boundary_events_[i]));
    }

    // Distribute sites and create segments
    distribute_sites();

    // Create boundary mergers (one for each adjacent pair)
    mergers_.reserve(n_streams_ - 1);
    for (int i = 0; i < n_streams_ - 1; i++) {
        mergers_.push_back(new BoundaryMergeGPU(max_bond_, 30, 1e-10));
    }

    // Initialize energy tracking
    segment_energies_.resize(n_streams_, 0.0);

    std::cout << "StreamCoordinator created with " << n_streams_ << " streams" << std::endl;
    std::cout << "Chain length: " << chain_length_ << std::endl;
    for (int i = 0; i < n_streams_; i++) {
        std::cout << "  Segment " << i << ": sites ["
                  << segments_[i]->get_start_site() << ", "
                  << segments_[i]->get_end_site() << "]" << std::endl;
    }
}

StreamCoordinator::~StreamCoordinator() {
    // Delete segments
    for (auto seg : segments_) {
        delete seg;
    }

    // Delete mergers
    for (auto merger : mergers_) {
        delete merger;
    }

    // Destroy streams
    for (auto stream : streams_) {
        HIP_CHECK(hipStreamDestroy(stream));
    }

    // Destroy events
    for (auto event : boundary_events_) {
        HIP_CHECK(hipEventDestroy(event));
    }
}

void StreamCoordinator::distribute_sites() {
    // Simple load-balanced distribution
    // Each segment gets approximately chain_length / n_streams sites

    int sites_per_segment = chain_length_ / n_streams_;
    int remainder = chain_length_ % n_streams_;

    int current_site = 0;
    segments_.reserve(n_streams_);

    for (int i = 0; i < n_streams_; i++) {
        int start_site = current_site;
        int num_sites = sites_per_segment + (i < remainder ? 1 : 0);
        int end_site = start_site + num_sites - 1;

        StreamSegment* seg = new StreamSegment(
            i, start_site, end_site,
            chi_max_, d_, D_mpo_,
            streams_[i]
        );

        segments_.push_back(seg);
        current_site = end_site + 1;
    }
}

//==============================================================================
// Main Iteration
//==============================================================================

double StreamCoordinator::run_iteration(int iter) {
    std::cout << "\n=== Iteration " << iter << " ===" << std::endl;

    // Phase 1: Forward sweep (all segments in parallel)
    std::cout << "  Forward sweep..." << std::endl;
    sweep_forward();

    // Phase 2: Even boundary merges (0↔1, 2↔3, ...)
    if (iter % 2 == 0) {
        std::cout << "  Even boundary merges..." << std::endl;
        merge_even_boundaries(iter);
    } else {
        std::cout << "  Odd boundary merges..." << std::endl;
        merge_odd_boundaries(iter);
    }

    // Phase 3: Backward sweep (all segments in parallel)
    std::cout << "  Backward sweep..." << std::endl;
    sweep_backward();

    // Phase 4: Opposite boundary merges
    if (iter % 2 == 0) {
        std::cout << "  Odd boundary merges..." << std::endl;
        merge_odd_boundaries(iter);
    } else {
        std::cout << "  Even boundary merges..." << std::endl;
        merge_even_boundaries(iter);
    }

    // Collect energy
    collect_energy();

    std::cout << "  Total energy: " << total_energy_ << std::endl;

    return total_energy_;
}

//==============================================================================
// Sweep Operations
//==============================================================================

void StreamCoordinator::sweep_forward() {
    // Launch forward sweep on all segments in parallel
    for (int i = 0; i < n_streams_; i++) {
        // Note: sweep_left_to_right is async on the segment's stream
        segments_[i]->sweep_left_to_right();
    }

    // Wait for all sweeps to complete
    for (int i = 0; i < n_streams_; i++) {
        HIP_CHECK(hipStreamSynchronize(streams_[i]));
    }
}

void StreamCoordinator::sweep_backward() {
    // Launch backward sweep on all segments in parallel
    for (int i = 0; i < n_streams_; i++) {
        segments_[i]->sweep_right_to_left();
    }

    // Wait for all sweeps to complete
    for (int i = 0; i < n_streams_; i++) {
        HIP_CHECK(hipStreamSynchronize(streams_[i]));
    }
}

//==============================================================================
// Merge Operations
//==============================================================================

void StreamCoordinator::merge_even_boundaries(int iter) {
    // Merge boundaries: (0↔1), (2↔3), (4↔5), ...
    // These can happen in parallel since they don't overlap

    // First, rebuild environments at boundaries
    rebuild_boundary_environments();

    // Update V matrices
    update_boundary_v();

    // Perform merges in parallel
    for (int i = 0; i < n_streams_ - 1; i += 2) {
        double energy = merge_boundary(i, i + 1);
        segment_energies_[i] = energy;
    }

    // Synchronize all streams involved
    for (int i = 0; i < n_streams_; i += 2) {
        HIP_CHECK(hipStreamSynchronize(streams_[i]));
        if (i + 1 < n_streams_) {
            HIP_CHECK(hipStreamSynchronize(streams_[i + 1]));
        }
    }
}

void StreamCoordinator::merge_odd_boundaries(int iter) {
    // Merge boundaries: (1↔2), (3↔4), (5↔6), ...

    rebuild_boundary_environments();
    update_boundary_v();

    for (int i = 1; i < n_streams_ - 1; i += 2) {
        double energy = merge_boundary(i, i + 1);
        segment_energies_[i] = energy;
    }

    // Synchronize
    for (int i = 1; i < n_streams_; i += 2) {
        HIP_CHECK(hipStreamSynchronize(streams_[i]));
        if (i + 1 < n_streams_) {
            HIP_CHECK(hipStreamSynchronize(streams_[i + 1]));
        }
    }
}

double StreamCoordinator::merge_boundary(int left_idx, int right_idx) {
    if (left_idx < 0 || right_idx >= n_streams_ || left_idx + 1 != right_idx) {
        throw std::runtime_error("StreamCoordinator::merge_boundary: invalid indices");
    }

    StreamSegment* left = segments_[left_idx];
    StreamSegment* right = segments_[right_idx];

    // Get boundary data
    BoundaryData* left_boundary = left->get_right_boundary();
    BoundaryData* right_boundary = right->get_left_boundary();

    if (!left_boundary || !right_boundary) {
        throw std::runtime_error("StreamCoordinator::merge_boundary: null boundary data");
    }

    // Extract boundary tensors into BoundaryData
    // TODO: Implement extract_boundary_tensors() in StreamSegment

    // Perform merge using the appropriate merger
    double energy = 0.0;
    double trunc_err = 0.0;

    BoundaryMergeGPU* merger = mergers_[left_idx];  // Use merger for this boundary
    merger->merge(left_boundary, right_boundary, energy, trunc_err,
                  false,  // skip_optimization = false (do optimize)
                  streams_[left_idx]);

    return energy;
}

//==============================================================================
// Environment and V Updates
//==============================================================================

void StreamCoordinator::rebuild_boundary_environments() {
    // Rebuild L_env at right boundaries and R_env at left boundaries
    for (int i = 0; i < n_streams_; i++) {
        if (i < n_streams_ - 1) {
            segments_[i]->rebuild_right_boundary_env();
        }
        if (i > 0) {
            segments_[i]->rebuild_left_boundary_env();
        }
    }

    // Synchronize all streams
    for (int i = 0; i < n_streams_; i++) {
        HIP_CHECK(hipStreamSynchronize(streams_[i]));
    }
}

void StreamCoordinator::update_boundary_v() {
    // Recompute V = Lambda^-1 at all boundaries
    for (int i = 0; i < n_streams_; i++) {
        if (i < n_streams_ - 1) {
            segments_[i]->recompute_boundary_v(false);  // Right boundary
        }
        if (i > 0) {
            segments_[i]->recompute_boundary_v(true);   // Left boundary
        }
    }

    // Synchronize
    for (int i = 0; i < n_streams_; i++) {
        HIP_CHECK(hipStreamSynchronize(streams_[i]));
    }
}

//==============================================================================
// Energy Collection
//==============================================================================

void StreamCoordinator::collect_energy() {
    // For now: simple sum of segment energies
    // TODO: Proper energy accounting with boundary corrections

    total_energy_ = 0.0;
    for (int i = 0; i < n_streams_; i++) {
        total_energy_ += segment_energies_[i];
    }

    // For multi-segment DMRG, need to avoid double-counting at boundaries
    // This requires proper accounting of which terms belong to which segment
    // For now: simple average
    if (n_streams_ > 1) {
        total_energy_ /= (n_streams_ - 1);  // Approximate correction
    }
}

//==============================================================================
// MPO and MPS Setup
//==============================================================================

void StreamCoordinator::set_mpo(double** d_mpo_tensors) {
    // Distribute MPO tensors to segments
    for (int i = 0; i < n_streams_; i++) {
        int start = segments_[i]->get_start_site();
        int end = segments_[i]->get_end_site();

        for (int site = start; site <= end; site++) {
            int local_idx = site - start;
            double* d_mpo_local = segments_[i]->get_mpo_tensor(site);

            // Copy MPO tensor for this site
            size_t mpo_size = D_mpo_ * d_ * d_ * D_mpo_;
            HIP_CHECK(hipMemcpy(d_mpo_local, d_mpo_tensors[site],
                               mpo_size * sizeof(double), hipMemcpyDeviceToDevice));
        }
    }
}

void StreamCoordinator::initialize_mps_random() {
    // Each segment initializes its own MPS (already done in constructor)
    // This method is for consistency with external interface
    std::cout << "MPS initialized randomly in each segment" << std::endl;
}
