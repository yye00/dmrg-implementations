#ifndef HEFF_OPTIMIZED_GPU_H
#define HEFF_OPTIMIZED_GPU_H

#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.hpp>
#include <map>
#include <string>
#include <vector>

/**
 * @brief Optimized H_eff application using hipTensor with workspace caching
 *
 * The effective Hamiltonian application H_eff * theta is the dominant
 * computational bottleneck in DMRG, called 15,000-30,000 times per run.
 *
 * This class implements an optimized version with:
 *   1. Workspace caching (allocate once, reuse across calls)
 *   2. Optimal contraction path selection (HIPTENSOR_ALGO_GREEDY)
 *   3. Fully GPU-resident (no CPU transfers)
 *   4. Pre-created contraction descriptors and plans
 *
 * The H_eff contraction sequence for two-site DMRG:
 *   Input: theta[a, s1, s2, b]  (two-site wavefunction)
 *          L[w, ap, a]          (left environment)
 *          W1[w, s1, s1p, x]    (left MPO tensor)
 *          W2[x, s2, s2p, y]    (right MPO tensor)
 *          R[y, b, bp]          (right environment)
 *
 *   Step 1: T1[w, ap, s1, s2, b] = L[w, ap, a] × theta[a, s1, s2, b]
 *   Step 2: T2[ap, s1p, s2, b, x] = W1[w, s1, s1p, x] × T1[w, ap, s1, s2, b]
 *   Step 3: T3[ap, s1p, s2p, b, y] = W2[x, s2, s2p, y] × T2[ap, s1p, s2, b, x]
 *   Step 4: result[a, s1p, s2p, bp] = T3[a, s1p, s2p, b, y] × R[y, b, bp]
 *
 * Expected performance: 70-90% of MI300X peak FP64 (~480M FLOPs per call)
 */
class OptimizedHeff {
private:
    // Dimensions (fixed at construction)
    int chi_L;      // Left bond dimension
    int chi_R;      // Right bond dimension
    int d;          // Physical dimension
    int D_mpo;      // MPO bond dimension

    // hipTensor handle
    hiptensorHandle_t* handle;

    // Tensor descriptors (reusable)
    hiptensorTensorDescriptor_t desc_L;
    hiptensorTensorDescriptor_t desc_R;
    hiptensorTensorDescriptor_t desc_W1;
    hiptensorTensorDescriptor_t desc_W2;
    hiptensorTensorDescriptor_t desc_theta;
    hiptensorTensorDescriptor_t desc_T1;
    hiptensorTensorDescriptor_t desc_T2;
    hiptensorTensorDescriptor_t desc_T3;
    hiptensorTensorDescriptor_t desc_result;

    // Contraction descriptors
    hiptensorContractionDescriptor_t contraction_1;  // L × theta
    hiptensorContractionDescriptor_t contraction_2;  // W1 × T1
    hiptensorContractionDescriptor_t contraction_3;  // W2 × T2
    hiptensorContractionDescriptor_t contraction_4;  // T3 × R

    // Contraction plans (pre-optimized)
    hiptensorPlan_t plan_1;
    hiptensorPlan_t plan_2;
    hiptensorPlan_t plan_3;
    hiptensorPlan_t plan_4;

    // Workspace management
    struct WorkspaceCache {
        void* d_workspace;
        size_t size;
        bool is_allocated;

        WorkspaceCache() : d_workspace(nullptr), size(0), is_allocated(false) {}
    };

    std::map<std::string, WorkspaceCache> workspace_cache;

    // Intermediate tensor storage (allocated once, reused)
    double* d_T1;
    double* d_T2;
    double* d_T3;

    bool is_initialized;

    /**
     * @brief Initialize tensor descriptors
     */
    void initialize_descriptors();

    /**
     * @brief Create contraction descriptors
     */
    void create_contractions();

    /**
     * @brief Create optimized contraction plans with workspace caching
     */
    void create_plans();

    /**
     * @brief Allocate intermediate tensors
     */
    void allocate_intermediates();

public:
    /**
     * @brief Constructor
     *
     * @param chi_L_in Left bond dimension
     * @param chi_R_in Right bond dimension
     * @param d_in Physical dimension
     * @param D_mpo_in MPO bond dimension
     * @param handle_in hipTensor handle (externally managed)
     */
    OptimizedHeff(
        int chi_L_in,
        int chi_R_in,
        int d_in,
        int D_mpo_in,
        hiptensorHandle_t* handle_in
    );

    /**
     * @brief Destructor - cleans up workspaces and intermediate tensors
     */
    ~OptimizedHeff();

    /**
     * @brief Apply H_eff to a two-site wavefunction
     *
     * Computes: result = H_eff * theta
     *
     * All tensors are assumed to be in column-major (Fortran) order for
     * compatibility with rocBLAS/hipTensor.
     *
     * @param d_theta Input wavefunction [chi_L, d, d, chi_R] on device
     * @param d_result Output H_eff * theta [chi_L, d, d, chi_R] on device
     * @param d_L Left environment [D_mpo, chi_L, chi_L] on device
     * @param d_R Right environment [D_mpo, chi_R, chi_R] on device
     * @param d_W1 Left MPO tensor [D_mpo, d, d, D_mpo] on device
     * @param d_W2 Right MPO tensor [D_mpo, d, d, D_mpo] on device
     * @param stream HIP stream for asynchronous execution (default 0)
     */
    void apply(
        const double* d_theta,
        double* d_result,
        const double* d_L,
        const double* d_R,
        const double* d_W1,
        const double* d_W2,
        hipStream_t stream = 0
    );

    /**
     * @brief Get workspace memory usage in bytes
     */
    size_t get_workspace_size() const;

    /**
     * @brief Get total memory footprint (workspace + intermediates) in bytes
     */
    size_t get_total_memory() const;

    /**
     * @brief Check if properly initialized
     */
    bool initialized() const { return is_initialized; }
};

/**
 * @brief Wrapper class for managing multiple OptimizedHeff instances
 *
 * In multi-stream PDMRG, each stream may have different bond dimensions
 * at different sites. This class manages a pool of OptimizedHeff instances
 * and returns the appropriate one for given dimensions.
 */
class HeffManager {
private:
    hiptensorHandle_t* handle;
    std::map<std::tuple<int, int, int, int>, OptimizedHeff*> heff_pool;

public:
    HeffManager(hiptensorHandle_t* handle_in) : handle(handle_in) {}

    ~HeffManager() {
        for (auto& [key, heff] : heff_pool) {
            delete heff;
        }
    }

    /**
     * @brief Get or create an OptimizedHeff for given dimensions
     *
     * @param chi_L Left bond dimension
     * @param chi_R Right bond dimension
     * @param d Physical dimension
     * @param D_mpo MPO bond dimension
     * @return Pointer to OptimizedHeff instance (internally managed)
     */
    OptimizedHeff* get(int chi_L, int chi_R, int d, int D_mpo) {
        auto key = std::make_tuple(chi_L, chi_R, d, D_mpo);

        auto it = heff_pool.find(key);
        if (it != heff_pool.end()) {
            return it->second;
        }

        // Create new instance
        OptimizedHeff* heff = new OptimizedHeff(chi_L, chi_R, d, D_mpo, handle);
        heff_pool[key] = heff;
        return heff;
    }

    /**
     * @brief Get total memory used by all cached instances
     */
    size_t get_total_memory() const {
        size_t total = 0;
        for (const auto& [key, heff] : heff_pool) {
            total += heff->get_total_memory();
        }
        return total;
    }

    /**
     * @brief Clear all cached instances (frees memory)
     */
    void clear() {
        for (auto& [key, heff] : heff_pool) {
            delete heff;
        }
        heff_pool.clear();
    }
};

#endif // HEFF_OPTIMIZED_GPU_H
