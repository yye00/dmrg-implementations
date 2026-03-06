#pragma once

#include <hip/hip_runtime.h>
#include <vector>
#include "stream_segment.h"
#include "boundary_merge_gpu.h"

/**
 * StreamCoordinator: Orchestrates multi-stream domain-decomposed DMRG
 *
 * Maps CPU PDMRG's MPI parallelism to GPU stream parallelism:
 * - MPI ranks → HIP streams
 * - MPI_Send/Recv → hipMemcpyAsync + Events
 * - MPI_Barrier → hipStreamSynchronize
 *
 * Algorithm (from CPU PDMRG):
 * 1. Sweep Phase (parallel): Each segment does local QR sweeps on its stream
 * 2. Even Merge Phase (sync): Merge boundaries (0↔1, 2↔3, 4↔5, ...)
 * 3. Sweep Phase (parallel): Reverse sweep
 * 4. Odd Merge Phase (sync): Merge boundaries (1↔2, 3↔4, ...)
 * 5. Repeat until convergence
 *
 * Critical: Preserves exact SVD boundary reconciliation from CPU implementation
 */
class StreamCoordinator {
public:
    /**
     * Constructor
     *
     * @param n_streams      Number of streams (segments)
     * @param chain_length   Total number of sites in MPS chain
     * @param chi_max        Maximum bond dimension
     * @param d              Physical dimension
     * @param D_mpo          MPO bond dimension
     * @param max_bond       Maximum bond at boundaries (for merges)
     */
    StreamCoordinator(int n_streams, int chain_length,
                      int chi_max, int d, int D_mpo, int max_bond);

    ~StreamCoordinator();

    /**
     * Run one DMRG iteration (sweep + merge cycle)
     *
     * @param iter  Iteration number (determines even/odd merge pattern)
     * @return      Total energy from all segments
     */
    double run_iteration(int iter);

    /**
     * Get current total energy
     */
    double get_energy() const { return total_energy_; }

    /**
     * Get segment for debugging/inspection
     */
    StreamSegment* get_segment(int i) {
        if (i < 0 || i >= n_streams_) return nullptr;
        return segments_[i];
    }

    /**
     * Set MPO tensors for all segments
     *
     * @param d_mpo_tensors  Array of MPO tensors [chain_length]
     *                       Each tensor is (D_mpo, d, d, D_mpo)
     */
    void set_mpo(double** d_mpo_tensors);

    /**
     * Initialize MPS from random state or product state
     */
    void initialize_mps_random();
    
    /**
     * Load MPS from binary file
     * @param filename Path to binary file containing MPS tensors
     * @return true on success, false on failure
     */
    bool load_mps_from_binary(const char* filename);
    void build_all_environments();  // Build environments after MPS+MPO loaded

private:
    // Configuration
    int n_streams_;
    int chain_length_;
    int chi_max_;
    int d_;
    int D_mpo_;
    int max_bond_;
    std::vector<int> bond_dims_;  // Bond dimensions for MPS loading

    // Segments and streams
    std::vector<hipStream_t> streams_;
    std::vector<StreamSegment*> segments_;
    std::vector<BoundaryMergeGPU*> mergers_;

    // Synchronization events for boundaries
    std::vector<hipEvent_t> boundary_events_;

    // Energy tracking
    double total_energy_;
    std::vector<double> segment_energies_;

    // ============================================================================
    // Sweep Operations
    // ============================================================================

    /**
     * Forward sweep: All segments sweep left-to-right in parallel
     */
    void sweep_forward();

    /**
     * Backward sweep: All segments sweep right-to-left in parallel
     */
    void sweep_backward();

    // ============================================================================
    // Merge Operations
    // ============================================================================

    /**
     * Even boundary merge: (0↔1, 2↔3, 4↔5, ...)
     *
     * Segments with even indices merge with right neighbor.
     * Requires synchronization before and after merge.
     *
     * @param iter  Iteration number (for logging/debugging)
     */
    void merge_even_boundaries(int iter);

    /**
     * Odd boundary merge: (1↔2, 3↔4, 5↔6, ...)
     *
     * Segments with odd indices merge with right neighbor.
     *
     * @param iter  Iteration number (for logging/debugging)
     */
    void merge_odd_boundaries(int iter);

    /**
     * Merge two adjacent segments at their shared boundary
     *
     * @param left_idx   Index of left segment
     * @param right_idx  Index of right segment
     * @return           Energy from the merge
     */
    double merge_boundary(int left_idx, int right_idx);

    // ============================================================================
    // Energy Collection
    // ============================================================================

    /**
     * Collect energy from all segments
     *
     * For domain-decomposed DMRG, total energy = sum of segment energies
     * minus boundary double-counting corrections.
     */
    void collect_energy();

    /**
     * Compute full-chain energy by evaluating ⟨ψ|H|ψ⟩
     *
     * Contracts MPS with MPO for all bonds in the chain.
     * This gives the actual Hamiltonian expectation value,
     * not just boundary optimization energies.
     *
     * @return  Total energy of the full MPS chain
     */
    double compute_full_chain_energy();

    // ============================================================================
    // Environment Updates
    // ============================================================================

    /**
     * Rebuild environments at boundaries before merge
     *
     * After local sweeps, environments at boundaries need to be rebuilt
     * to ensure correct two-site optimization during merge.
     */
    void rebuild_boundary_environments();

    /**
     * Update V matrices at boundaries after sweeps
     *
     * V = Lambda^-1 must be recomputed after canonization sweeps
     * to maintain exact boundary reconciliation.
     */
    void update_boundary_v();

    // ============================================================================
    // Site Distribution
    // ============================================================================

    /**
     * Distribute sites among segments
     *
     * For n_streams segments and chain_length sites:
     * - Segment i gets sites [start_i, end_i] (inclusive)
     * - Load balanced: each segment gets ~chain_length/n_streams sites
     * - Boundary overlaps handled by merge operations
     */
    void distribute_sites();
};
