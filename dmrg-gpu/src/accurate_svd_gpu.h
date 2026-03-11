#ifndef ACCURATE_SVD_GPU_H
#define ACCURATE_SVD_GPU_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <vector>
#include <complex>

/**
 * @brief Result of accurate SVD decomposition
 */
struct AccurateSVDResult {
    double* d_U;           // Left singular vectors [m x k] on device
    double* d_S;           // Singular values [k] on device
    double* d_Vh;          // Right singular vectors (conjugate transposed) [k x n] on device
    int rank;              // Number of singular values
    int m;                 // Number of rows
    int n;                 // Number of columns

    // Memory ownership flag
    bool owns_memory;

    AccurateSVDResult()
        : d_U(nullptr), d_S(nullptr), d_Vh(nullptr),
          rank(0), m(0), n(0), owns_memory(false) {}

    ~AccurateSVDResult() {
        if (owns_memory) {
            if (d_U) hipFree(d_U);
            if (d_S) hipFree(d_S);
            if (d_Vh) hipFree(d_Vh);
        }
    }

    // Disable copy, enable move
    AccurateSVDResult(const AccurateSVDResult&) = delete;
    AccurateSVDResult& operator=(const AccurateSVDResult&) = delete;
    AccurateSVDResult(AccurateSVDResult&& other) noexcept {
        d_U = other.d_U;
        d_S = other.d_S;
        d_Vh = other.d_Vh;
        rank = other.rank;
        m = other.m;
        n = other.n;
        owns_memory = other.owns_memory;
        other.d_U = nullptr;
        other.d_S = nullptr;
        other.d_Vh = nullptr;
        other.owns_memory = false;
    }
};

/**
 * @brief Accurate SVD with recursive refinement for small singular values
 *
 * Standard SVD (rocsolver_zgesvd) loses accuracy for singular values
 * smaller than epsilon * sigma_max. This class implements recursive
 * refinement by re-orthogonalizing the inaccurate subspace.
 *
 * Algorithm (from pdmrg/numerics/accurate_svd.py):
 *   1. Compute standard SVD: M = U * S * Vh
 *   2. Find degradation threshold p where S[p]/S[0] < epsilon
 *   3. If p exists:
 *      a. Project M onto inaccurate subspace: X = U[:,p:]^H @ M @ Vh[p:,:]^H
 *      b. Recursively compute accurate SVD of X
 *      c. Update U[:,p:], Vh[p:,:], and S[p:] with refined values
 *   4. Return refined U, S, Vh
 *
 * This prevents V = 1/S from blowing up at segment boundaries in PDMRG.
 */
class AccurateSVD_GPU {
private:
    rocblas_handle rocblas_h;  // ROCm 7.2.0: rocsolver uses rocblas_handle

    double epsilon;                // Degradation threshold (default 1e-4)
    int max_recursion_depth;       // Maximum recursion levels (default 5)

    // Workspace management
    struct Workspace {
        void* d_work;
        size_t size;
        int* d_info;
        double* d_rwork;

        Workspace() : d_work(nullptr), size(0), d_info(nullptr), d_rwork(nullptr) {}

        void allocate(size_t work_size, size_t rwork_size) {
            if (work_size > size) {
                if (d_work) hipFree(d_work);
                hipMalloc(&d_work, work_size);
                size = work_size;
            }
            if (!d_info) {
                hipMalloc(&d_info, sizeof(int));
            }
            if (rwork_size > 0 && !d_rwork) {
                hipMalloc(&d_rwork, rwork_size * sizeof(double));
            }
        }

        ~Workspace() {
            if (d_work) hipFree(d_work);
            if (d_info) hipFree(d_info);
            if (d_rwork) hipFree(d_rwork);
        }
    };

    Workspace workspace;

    /**
     * @brief Find index p where S[p]/S[0] < epsilon
     * @param d_S Singular values on device [k]
     * @param k Number of singular values
     * @return Degradation threshold index, or -1 if none found
     */
    int find_degradation_threshold(double* d_S, int k);

    /**
     * @brief Perform standard SVD using rocsolver
     * @param d_M Input matrix [m x n] on device (will be overwritten)
     * @param m Number of rows
     * @param n Number of columns
     * @return SVD result with U, S, Vh on device
     */
    AccurateSVDResult standard_svd(double* d_M, int m, int n);

    /**
     * @brief Recursively refine SVD for small singular values
     * @param d_M Input matrix [m x n] on device
     * @param m Number of rows
     * @param n Number of columns
     * @param depth Current recursion depth
     * @return Refined SVD result
     */
    AccurateSVDResult decompose_recursive(double* d_M, int m, int n, int depth);

public:
    /**
     * @brief Constructor
     * @param eps Degradation threshold (S[p]/S[0] < eps triggers refinement)
     * @param max_depth Maximum recursion depth (prevents infinite loops)
     */
    AccurateSVD_GPU(double eps = 1e-4, int max_depth = 5);

    /**
     * @brief Destructor - cleans up handles and workspace
     */
    ~AccurateSVD_GPU();

    /**
     * @brief Compute accurate SVD with recursive refinement
     *
     * @param d_M Input matrix [m x n] on device (will be copied, not modified)
     * @param m Number of rows
     * @param n Number of columns
     * @return AccurateSVDResult with refined U, S, Vh on device
     *
     * @note The returned result owns the device memory for U, S, Vh.
     *       The caller is responsible for keeping the result alive or
     *       copying the data before the result goes out of scope.
     */
    AccurateSVDResult decompose(double* d_M, int m, int n);

    /**
     * @brief In-place version that modifies input matrix
     *
     * @param d_M Input/output matrix [m x n] on device (will be overwritten)
     * @param m Number of rows
     * @param n Number of columns
     * @return AccurateSVDResult with refined U, S, Vh on device
     */
    AccurateSVDResult decompose_inplace(double* d_M, int m, int n);

    /**
     * @brief Get current epsilon threshold
     */
    double get_epsilon() const { return epsilon; }

    /**
     * @brief Set epsilon threshold
     */
    void set_epsilon(double eps) { epsilon = eps; }

    /**
     * @brief Get maximum recursion depth
     */
    int get_max_depth() const { return max_recursion_depth; }

    /**
     * @brief Set maximum recursion depth
     */
    void set_max_depth(int depth) { max_recursion_depth = depth; }
};

/**
 * @brief Utility function to compute truncation dimension
 *
 * @param d_S Singular values on device [k]
 * @param k Number of singular values
 * @param max_bond_dim Maximum bond dimension to keep
 * @param cutoff Truncation threshold (sum of discarded weights < cutoff)
 * @param h_trunc_error Output: truncation error (sum of discarded squared singular values)
 * @return Number of singular values to keep
 */
int compute_truncation_dim(
    double* d_S,
    int k,
    int max_bond_dim,
    double cutoff,
    double* h_trunc_error
);

/**
 * @brief GPU kernel to invert singular values with clipping: V[i] = 1 / max(S[i], clip_min)
 *
 * @param d_S Input singular values [k]
 * @param d_V Output inverted values [k]
 * @param k Number of values
 * @param clip_min Minimum value for clipping (prevents overflow)
 */
void launch_invert_with_clipping(
    double* d_S,
    double* d_V,
    int k,
    double clip_min,
    hipStream_t stream = 0
);

#endif // ACCURATE_SVD_GPU_H
