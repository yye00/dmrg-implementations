#ifndef DMRG2_GPU_OPT_IMPL_H
#define DMRG2_GPU_OPT_IMPL_H

#include <rocsolver/rocsolver.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>


#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << hipGetErrorString(err) << std::endl; \
            throw std::runtime_error("HIP error"); \
        } \
    } while(0)

#define ROCBLAS_CHECK(call) \
    do { \
        rocblas_status status = call; \
        if (status != rocblas_status_success) { \
            std::cerr << "rocBLAS error in " << __FILE__ << ":" << __LINE__ \
                      << " - status " << status << std::endl; \
            throw std::runtime_error("rocBLAS error"); \
        } \
    } while(0)

// Fused Lanczos update:  w := w + (-α)*v_i + [(-β_{im1})*v_{im1}]
// d_neg_alpha and d_neg_beta_im1 are Scalar* on device (device-pointer mode).
// v_im1 may be nullptr when has_prev is false (iter==0 path).
template<typename Scalar>
__global__ void lanczos_fused_sub_kernel(
    Scalar* w, const Scalar* v_i, const Scalar* v_im1,
    const Scalar* d_neg_alpha, const Scalar* d_neg_beta_im1,
    int n, int has_prev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    Scalar wi = w[idx];
    wi = scalar_add(wi, scalar_mul(*d_neg_alpha, v_i[idx]));
    if (has_prev) {
        wi = scalar_add(wi, scalar_mul(*d_neg_beta_im1, v_im1[idx]));
    }
    w[idx] = wi;
}

// Fused normalize-and-copy:  v_{i+1}[k] = w[k] * inv_beta
template<typename Scalar, typename RealType>
__global__ void lanczos_fused_norm_copy_kernel(
    Scalar* v_next, const Scalar* w, const RealType* d_inv_beta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    v_next[idx] = ScalarTraits<Scalar>::scale_by_real(*d_inv_beta, w[idx]);
}

// Profiling counters (reset per sweep pair)
static double prof_davidson_ms = 0, prof_svd_ms = 0, prof_env_ms = 0;
static int prof_davidson_iters = 0, prof_site_count = 0;
static int prof_heff_calls = 0;

// ============================================================================
// Constructor
// ============================================================================

template<typename Scalar>
DMRG2GPUOpt<Scalar>::DMRG2GPUOpt(int L, int d, int chi_max, int D_mpo, double tol)
    : L_(L), d_(d), chi_max_(pad_mfma16(chi_max)), chi_max_user_(chi_max),
      D_mpo_(D_mpo), D_mpo_actual_(D_mpo), tol_(tol), energy_(0.0) {

    opts_.load_from_env();

    // LANCZOS_GRAPH safeguard: this variant uses Block-Davidson, which calls
    // apply_heff_two_site(site, V + j*dim, AV + j*dim) with a variable output
    // pointer per subspace column. HIP graph capture burns the first-seen
    // output address into the graph; replays then write to the stale address
    // instead of AV + j*dim, Rayleigh-Ritz sees garbage, and the outer loop
    // hangs indefinitely. Force the flag off with a one-line warning; use
    // dmrg-gpu (Lanczos) or pdmrg-gpu for graph-captured apply_heff.
    if (opts_.lanczos_graph) {
        std::fprintf(stderr,
            "[dmrg2-gpu-opt] LANCZOS_GRAPH=1 is incompatible with Block-Davidson "
            "(variable output pointer per subspace column). Disabling.\n");
        opts_.lanczos_graph = false;
    }

    // D_PAD: round MPO bond dim up to a multiple of 8 for MFMA-friendly
    // GEMM shapes. All allocations and internal GEMMs use the padded D;
    // the padded rows/cols of the W matrices are zero-filled in set_mpo
    // so they contribute nothing numerically. The R boundary still uses
    // D_mpo_actual_-1 (see build_initial_environments).
    if (opts_.d_pad) {
        int padded = (D_mpo_ + 7) & ~7;
        if (padded != D_mpo_) {
            std::fprintf(stderr, "[D_PAD] D_mpo padded: %d -> %d\n", D_mpo_, padded);
            D_mpo_ = padded;
        }
    }

    opts_.print(stderr);
    init_timers();

    if (chi_max_ != chi_max_user_) {
        printf("[OPT] MFMA-16 padding: chi_max %d -> %d\n", chi_max_user_, chi_max_);
    }

    // Bond dimensions (same as single-site: min-cut formula capped at chi_max)
    bond_dims_.resize(L + 1);
    bond_dims_[0] = 1;
    bond_dims_[L] = 1;
    for (int i = 1; i < L; i++) {
        double exact_dim = pow((double)d_, std::min(i, L - i));
        bond_dims_[i] = (exact_dim > chi_max_user_) ? chi_max_user_ : (int)exact_dim;
    }

    // GPU handles
    HIP_CHECK(hipStreamCreate(&stream_));
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_h_));
    ROCBLAS_CHECK(rocblas_set_stream(rocblas_h_, stream_));

    int dd = d_ * d_;  // d^2 for two-site

    // Contraction intermediates: D*d^2*chi_max^2
    int t_max = D_mpo_ * dd * chi_max_ * chi_max_;
    HIP_CHECK(hipMalloc(&d_T1_, t_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_T2_, t_max * sizeof(Scalar)));

    // MPS tensors
    d_mps_tensors_.resize(L, nullptr);
    for (int i = 0; i < L; i++) {
        allocate_mps_tensor(i, chi_L(i), chi_R(i));
    }

    // MPO tensors
    d_mpo_tensors_.resize(L, nullptr);

    // W matrices for single-site env updates (allocated in set_mpo)
    d_W_left_.resize(L, nullptr);
    d_W_right_.resize(L, nullptr);

    // Fused two-site MPO (allocated in set_mpo)
    d_WW_.resize(L - 1, nullptr);

    // SPARSE_MPO nnz lists (populated in precompute_fused_mpo)
    d_WW_nnz_rows_.resize(L - 1, nullptr);
    d_WW_nnz_cols_.resize(L - 1, nullptr);
    ww_nnz_rows_count_.assign(L - 1, 0);
    ww_nnz_cols_count_.assign(L - 1, 0);
    h_WW_nnz_rows_.resize(L - 1);
    h_WW_nnz_cols_.resize(L - 1);

    // Environments
    d_L_envs_.resize(L + 1, nullptr);
    d_R_envs_.resize(L + 1, nullptr);
    L_env_alloc_chi_.resize(L + 1, 0);
    R_env_alloc_chi_.resize(L + 1, 0);

    for (int i = 0; i <= L; i++) {
        int chi_alloc = (i == 0 || i == L) ? 1 : chi_max_;
        int sz = chi_alloc * D_mpo_ * chi_alloc;
        HIP_CHECK(hipMalloc(&d_L_envs_[i], sz * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_R_envs_[i], sz * sizeof(Scalar)));
        HIP_CHECK(hipMemset(d_L_envs_[i], 0, sz * sizeof(Scalar)));
        HIP_CHECK(hipMemset(d_R_envs_[i], 0, sz * sizeof(Scalar)));
        L_env_alloc_chi_[i] = chi_alloc;
        R_env_alloc_chi_[i] = chi_alloc;
    }

    // Lanczos workspace (fallback): theta is d^2 times larger than single-site
    theta_size_max_ = chi_max_ * dd * chi_max_;
    max_lanczos_iter_ = std::min(100, theta_size_max_);
    HIP_CHECK(hipMalloc(&d_theta_, theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_heff_result_, theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_lanczos_v_, (size_t)max_lanczos_iter_ * theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_ritz_coeffs_, max_lanczos_iter_ * sizeof(Scalar)));

    // Batched GEMM pointer arrays: D*d^2 batches for two-site
    int batch_max = D_mpo_ * dd;
    HIP_CHECK(hipMalloc(&d_batch_A_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_B_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_C_, batch_max * sizeof(Scalar*)));

    // SVD workspace
    int svd_max_m = chi_max_ * d_;
    int svd_max_n = d_ * chi_max_;
    int svd_max_k = std::min(svd_max_m, svd_max_n);  // = chi_max_ * d_

    HIP_CHECK(hipMalloc(&d_svd_A_,    theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_U_,    (size_t)svd_max_m * svd_max_k * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_S_,    svd_max_k * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_Vh_,   (size_t)svd_max_k * svd_max_n * sizeof(Scalar)));

    // CPU SVD workspace (fallback)
    h_svd_A_.resize(theta_size_max_);
    h_svd_U_.resize((size_t)svd_max_m * svd_max_k);
    h_svd_S_.resize(svd_max_k);
    h_svd_Vh_.resize((size_t)svd_max_k * svd_max_n);
    h_svd_tmp_.resize(std::max((size_t)svd_max_m * svd_max_k, (size_t)svd_max_k * svd_max_n));
    h_svd_rwork_.resize(Traits::svd_rwork_size(svd_max_m, svd_max_n));

    // Query optimal LAPACK SVD workspace
    {
        int m = svd_max_m, n = svd_max_n;
        int lwork_query = -1;
        Scalar work_opt;
        int info;
        const char jobu = 'S', jobvt = 'S';
        Traits::lapack_gesvd(&jobu, &jobvt, &m, &n, nullptr, &m, nullptr,
                nullptr, &m, nullptr, &svd_max_k, &work_opt, &lwork_query,
                h_svd_rwork_.empty() ? nullptr : h_svd_rwork_.data(), &info);
        int opt_size;
        if constexpr (Traits::is_complex) {
            opt_size = (int)Traits::real_part(work_opt) + 1;
        } else {
            opt_size = (int)work_opt + 1;
        }
        h_svd_work_.resize(opt_size);
    }

    // Block-Davidson workspace
    davidson_b_ = 4;
    davidson_max_sub_ = std::min(davidson_b_ * 8, theta_size_max_);
    HIP_CHECK(hipMalloc(&d_dav_V_,     (size_t)theta_size_max_ * davidson_max_sub_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_dav_AV_,    (size_t)theta_size_max_ * davidson_max_sub_ * sizeof(Scalar)));
    {
        // d_dav_work_ must hold both (dim, b) orthogonalization scratch and (k, k) projected H
        size_t dav_work_sz = std::max((size_t)theta_size_max_ * davidson_b_,
                                       (size_t)davidson_max_sub_ * davidson_max_sub_);
        HIP_CHECK(hipMalloc(&d_dav_work_,  dav_work_sz * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_dav_work2_, dav_work_sz * sizeof(Scalar)));
    }
    h_dav_H_proj_.resize(davidson_max_sub_ * davidson_max_sub_);
    h_dav_eigvals_.resize(davidson_max_sub_);
    h_dav_eigvecs_.resize(davidson_max_sub_ * davidson_max_sub_);

    // LANCZOS_GRAPH: bounce buffer so captured apply_heff_two_site graphs read
    // from a fixed address regardless of which Lanczos v_i the caller passes in.
    if (opts_.lanczos_graph) {
        HIP_CHECK(hipMalloc(&d_heff_input_, (size_t)theta_size_max_ * sizeof(Scalar)));
    }
}

// ============================================================================
// Destructor
// ============================================================================

template<typename Scalar>
DMRG2GPUOpt<Scalar>::~DMRG2GPUOpt() {
    free_gpu_resources();
}

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::free_gpu_resources() {
    for (auto ptr : d_mps_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_mpo_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_left_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_right_) if (ptr) hipFree(ptr);
    for (auto ptr : d_WW_) if (ptr) hipFree(ptr);
    // SPARSE_MPO nnz lists
    for (auto ptr : d_WW_nnz_rows_) if (ptr) hipFree(ptr);
    for (auto ptr : d_WW_nnz_cols_) if (ptr) hipFree(ptr);
    for (auto ptr : d_L_envs_) if (ptr) hipFree(ptr);
    for (auto ptr : d_R_envs_) if (ptr) hipFree(ptr);

    if (d_theta_) hipFree(d_theta_);
    if (d_heff_result_) hipFree(d_heff_result_);
    if (d_lanczos_v_) hipFree(d_lanczos_v_);
    if (d_ritz_coeffs_) hipFree(d_ritz_coeffs_);
    if (d_T1_) hipFree(d_T1_);
    if (d_T2_) hipFree(d_T2_);
    if (d_batch_A_) hipFree(d_batch_A_);
    if (d_batch_B_) hipFree(d_batch_B_);
    if (d_batch_C_) hipFree(d_batch_C_);
    if (d_svd_A_) hipFree(d_svd_A_);
    if (d_svd_U_) hipFree(d_svd_U_);
    if (d_svd_S_) hipFree(d_svd_S_);
    if (d_svd_Vh_) hipFree(d_svd_Vh_);

    // Block-Davidson workspace
    if (d_dav_V_) hipFree(d_dav_V_);
    if (d_dav_AV_) hipFree(d_dav_AV_);
    if (d_dav_work_) hipFree(d_dav_work_);
    if (d_dav_work2_) hipFree(d_dav_work2_);

    // LANCZOS_GRAPH: destroy cached graph execs and bounce buffer
    for (auto& kv : apply_heff_graph_cache_) {
        hipGraphExecDestroy(kv.second);
    }
    apply_heff_graph_cache_.clear();
    if (d_heff_input_) hipFree(d_heff_input_);

    rocblas_destroy_handle(rocblas_h_);
    hipStreamDestroy(stream_);
}

// ============================================================================
// Memory management
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::allocate_mps_tensor(int site, int cL, int cR) {
    if (d_mps_tensors_[site]) HIP_CHECK(hipFree(d_mps_tensors_[site]));
    HIP_CHECK(hipMalloc(&d_mps_tensors_[site], cL * d_ * cR * sizeof(Scalar)));
}

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::ensure_L_env_alloc(int idx, int chi) {
    if (chi > L_env_alloc_chi_[idx]) {
        if (d_L_envs_[idx]) HIP_CHECK(hipFree(d_L_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_L_envs_[idx], sz * sizeof(Scalar)));
        L_env_alloc_chi_[idx] = chi;
    }
}

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::ensure_R_env_alloc(int idx, int chi) {
    if (chi > R_env_alloc_chi_[idx]) {
        if (d_R_envs_[idx]) HIP_CHECK(hipFree(d_R_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_R_envs_[idx], sz * sizeof(Scalar)));
        R_env_alloc_chi_[idx] = chi;
    }
}

// ============================================================================
// MPS initialization
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::initialize_mps_random(double scale) {
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        std::vector<Scalar> h_A(size);
        for (int j = 0; j < size; j++) {
            h_A[j] = Traits::scale_by_real(scale, Traits::random_val());
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::initialize_mps_product() {
    // Reset all interior bonds to 1 for product state
    for (int i = 1; i < L_; i++) bond_dims_[i] = 1;

    for (int i = 0; i < L_; i++) {
        int cL = chi_L(i), cR = chi_R(i);
        allocate_mps_tensor(i, cL, cR);
        int size = cL * d_ * cR;
        std::vector<Scalar> h_A(size, Traits::zero());
        // |0> state: only s=0, diagonal bonds
        int chi_min = std::min(cL, cR);
        for (int a = 0; a < chi_min; a++) {
            h_A[a + 0*cL + a*cL*d_] = Traits::one();
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::initialize_mps_neel() {
    for (int i = 0; i < L_; i++) {
        int cL = chi_L(i), cR = chi_R(i);
        int size = cL * d_ * cR;
        std::vector<Scalar> h_A(size, Traits::zero());
        int spin = (i % 2 == 0) ? 0 : 1;
        int chi_min = std::min(cL, cR);
        for (int a = 0; a < chi_min; a++) {
            h_A[a + spin*cL + a*cL*d_] = Traits::one();
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

// ============================================================================
// MPO setup and fused two-site MPO precomputation
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::set_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    // D_use: padded bond dim used for all internal buffers (D_PAD on).
    // D_act: user's MPO bond dim.
    int D_use = D_mpo_;
    int D_act = D_mpo_actual_;
    int d = d_;
    for (int i = 0; i < L_; i++) {
        int size_use = D_use * d * d * D_use;
        HIP_CHECK(hipMalloc(&d_mpo_tensors_[i], size_use * sizeof(Scalar)));
        HIP_CHECK(hipMemset(d_mpo_tensors_[i], 0, size_use * sizeof(Scalar)));

        std::vector<Scalar> h_WL(size_use, Traits::zero());
        std::vector<Scalar> h_WR(size_use, Traits::zero());

        if (D_use == D_act) {
            HIP_CHECK(hipMemcpy(d_mpo_tensors_[i], h_mpo_tensors[i],
                                size_use * sizeof(Scalar), hipMemcpyHostToDevice));
        } else {
            std::vector<Scalar> h_pad(size_use, Traits::zero());
            for (int wp = 0; wp < D_act; wp++)
                for (int sp = 0; sp < d; sp++)
                    for (int s = 0; s < d; s++)
                        for (int w = 0; w < D_act; w++)
                            h_pad[w + s*D_use + sp*D_use*d + wp*D_use*d*d] =
                                h_mpo_tensors[i][w + s*D_act + sp*D_act*d + wp*D_act*d*d];
            HIP_CHECK(hipMemcpy(d_mpo_tensors_[i], h_pad.data(),
                                size_use * sizeof(Scalar), hipMemcpyHostToDevice));
        }

        // Precompute W_left and W_right matrices (for single-site env updates)
        for (int w = 0; w < D_act; w++)
            for (int s = 0; s < d; s++)
                for (int sp = 0; sp < d; sp++)
                    for (int wp = 0; wp < D_act; wp++) {
                        Scalar val = h_mpo_tensors[i][w + s*D_act + sp*D_act*d + wp*D_act*d*d];
                        h_WL[(w*d+s) + (wp*d+sp) * D_use * d] = val;
                        h_WR[(wp*d+s) + (w*d+sp) * D_use * d] = val;
                    }
        HIP_CHECK(hipMalloc(&d_W_left_[i], size_use * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_W_left_[i], h_WL.data(),
                            size_use * sizeof(Scalar), hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&d_W_right_[i], size_use * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_W_right_[i], h_WR.data(),
                            size_use * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // Precompute fused two-site MPO
    precompute_fused_mpo(h_mpo_tensors);
}

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::precompute_fused_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    // D_use/D_act: see set_mpo. User's h_mpo_tensors are indexed with
    // stride D_act; WW is allocated at padded size D_use*dd x dd*D_use
    // with padded (w,n) rows/cols zero.
    int D_use = D_mpo_;
    int D_act = D_mpo_actual_;
    int d = d_;
    int dd = d * d;

    for (int bond = 0; bond < L_ - 1; bond++) {
        int ww_size = D_use * dd * dd * D_use;
        std::vector<Scalar> h_WW(ww_size, Traits::zero());

        const Scalar* WL = h_mpo_tensors[bond];
        const Scalar* WR = h_mpo_tensors[bond + 1];

        for (int w = 0; w < D_act; w++)
            for (int n = 0; n < D_act; n++)
                for (int s1 = 0; s1 < d; s1++)
                    for (int s2 = 0; s2 < d; s2++)
                        for (int s1p = 0; s1p < d; s1p++)
                            for (int s2p = 0; s2p < d; s2p++) {
                                Scalar val = Traits::zero();
                                for (int m = 0; m < D_act; m++) {
                                    Scalar wl = WL[w + s1*D_act + s1p*D_act*d + m*D_act*d*d];
                                    Scalar wr = WR[m + s2*D_act + s2p*D_act*d + n*D_act*d*d];
                                    if constexpr (Traits::is_complex) {
                                        val = hipCadd(val, hipCmul(wl, wr));
                                    } else {
                                        val += wl * wr;
                                    }
                                }
                                int row = w * dd + s1 * d + s2;
                                int col = n * dd + s1p * d + s2p;
                                h_WW[row + col * D_use * dd] = val;
                            }

        HIP_CHECK(hipMalloc(&d_WW_[bond], ww_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_WW_[bond], h_WW.data(),
                            ww_size * sizeof(Scalar), hipMemcpyHostToDevice));

        // SPARSE_MPO: build nnz row/col index lists for WW[bond].
        // WW is (D_use*dd, dd*D_use) column-major. Row r = w*dd + s1*d + s2
        // nonzero if any col has |val| > 0; col c = n*dd + s1p*d + s2p analogously.
        if (opts_.sparse_mpo) {
            int rows = D_use * dd;
            int cols = dd * D_use;
            std::vector<int> nnz_rows, nnz_cols;
            nnz_rows.reserve(rows);
            nnz_cols.reserve(cols);
            const double eps = 1e-14;
            for (int r = 0; r < rows; r++) {
                bool nz = false;
                for (int c = 0; c < cols && !nz; c++) {
                    if (scalar_abs(h_WW[r + (size_t)c * rows]) > eps) nz = true;
                }
                if (nz) nnz_rows.push_back(r);
            }
            for (int c = 0; c < cols; c++) {
                bool nz = false;
                for (int r = 0; r < rows && !nz; r++) {
                    if (scalar_abs(h_WW[r + (size_t)c * rows]) > eps) nz = true;
                }
                if (nz) nnz_cols.push_back(c);
            }
            ww_nnz_rows_count_[bond] = (int)nnz_rows.size();
            ww_nnz_cols_count_[bond] = (int)nnz_cols.size();
            if (!nnz_rows.empty()) {
                HIP_CHECK(hipMalloc(&d_WW_nnz_rows_[bond], nnz_rows.size() * sizeof(int)));
                HIP_CHECK(hipMemcpy(d_WW_nnz_rows_[bond], nnz_rows.data(),
                                    nnz_rows.size() * sizeof(int), hipMemcpyHostToDevice));
            }
            if (!nnz_cols.empty()) {
                HIP_CHECK(hipMalloc(&d_WW_nnz_cols_[bond], nnz_cols.size() * sizeof(int)));
                HIP_CHECK(hipMemcpy(d_WW_nnz_cols_[bond], nnz_cols.data(),
                                    nnz_cols.size() * sizeof(int), hipMemcpyHostToDevice));
            }
            if (bond == 0) {
                std::fprintf(stderr,
                    "[SPARSE_MPO] bond 0: WW shape (%d x %d), nnz rows=%d, nnz cols=%d (%.0f%% sparse)\n",
                    rows, cols, ww_nnz_rows_count_[bond], ww_nnz_cols_count_[bond],
                    100.0 * (1.0 - (double)(ww_nnz_rows_count_[bond] * ww_nnz_cols_count_[bond]) / (rows * cols)));
            }
            h_WW_nnz_rows_[bond] = std::move(nnz_rows);
            h_WW_nnz_cols_[bond] = std::move(nnz_cols);
        }
    }
}

// ============================================================================
// Two-site theta formation
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::form_theta_two_site(int site) {
    int cL = chi_L(site);
    int chi_mid = bond_dims_[site + 1];
    int cR = chi_R(site + 1);
    Scalar one = Traits::one(), zero_val = Traits::zero();

    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        cL * d_, d_ * cR, chi_mid,
        &one,
        d_mps_tensors_[site], cL * d_,
        d_mps_tensors_[site + 1], chi_mid,
        &zero_val,
        d_theta_, cL * d_));
}

// ============================================================================
// Two-site H_eff application (3-step with fused WW)
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::apply_heff_two_site(int site, const Scalar* d_theta_in, Scalar* d_result) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int D = D_mpo_, d = d_;
    int dd = d * d;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    // LANCZOS_GRAPH: stage caller's theta into a fixed-address bounce buffer
    // BEFORE any capture window, then either replay the cached graph or
    // capture a new one for this (site, cL, cR) shape. See dmrg_gpu_impl.h
    // for the full rationale. Two-site theta has shape (cL, d, d, cR).
    const Scalar* theta_src = d_theta_in;
    bool graph_capture_miss = false;
    if (opts_.lanczos_graph) {
        int n_theta = cL * dd * cR;
        HIP_CHECK(hipMemcpyAsync(d_heff_input_, d_theta_in,
                                 (size_t)n_theta * sizeof(Scalar),
                                 hipMemcpyDeviceToDevice, stream_));
        theta_src = d_heff_input_;

        uint64_t key = graph_key(site, cL, cR);
        auto it = apply_heff_graph_cache_.find(key);
        if (it != apply_heff_graph_cache_.end()) {
            HIP_CHECK(hipGraphLaunch(it->second, stream_));
            return;
        }
        graph_capture_miss = true;
        HIP_CHECK(hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal));
    }

    Scalar* L_env = d_L_envs_[site];
    Scalar* R_env = d_R_envs_[site + 2];
    Scalar* WW = d_WW_[site];
    Scalar* T1 = d_T1_;
    Scalar* T2 = d_T2_;

    // SPARSE_MPO: compact Step 1 / Step 3 batches to non-zero (w,s1,s2) /
    // (n,s1p,s2p) rows/cols of WW[site]. Step 2 reads T1 densely, so skipped
    // slots must be zero (hipMemsetAsync before Step 1). Step 3 accumulates
    // into d_result; for sparse mode we zero d_result first then use beta=1.
    const bool sparse_s1 = opts_.sparse_mpo
                         && ww_nnz_rows_count_[site] > 0
                         && ww_nnz_rows_count_[site] < D * dd;
    const bool sparse_s3 = opts_.sparse_mpo
                         && ww_nnz_cols_count_[site] > 0
                         && ww_nnz_cols_count_[site] < D * dd;

    // Step 1: Batched GEMM — contract L_env with theta
    {
        if (sparse_s1) {
            HIP_CHECK(hipMemsetAsync(T1, 0, (size_t)D * dd * cL * cR * sizeof(Scalar), stream_));
            int nnz = ww_nnz_rows_count_[site];
            const std::vector<int>& h_nnz = h_WW_nnz_rows_[site];
            std::vector<Scalar*> h_A(nnz), h_B(nnz), h_C(nnz);
            for (int idx = 0; idx < nnz; idx++) {
                int packed = h_nnz[idx];
                int w = packed / dd;
                int ss = packed % dd;
                int s1 = ss / d, s2 = ss % d;
                h_A[idx] = L_env + w * cL;
                h_B[idx] = const_cast<Scalar*>(theta_src) + s1 * cL + s2 * cL * d;
                h_C[idx] = T1 + packed * cL * cR;
            }
            HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A.data(), nnz*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
            HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B.data(), nnz*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
            HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C.data(), nnz*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
            ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
                Traits::op_t, rocblas_operation_none,
                cL, cR, cL,
                &one,
                (const Scalar**)d_batch_A_, cL * D,
                (const Scalar**)d_batch_B_, cL * dd,
                &zero_val,
                d_batch_C_, cL,
                nnz));
        } else {
            int batch_count = D * dd;
            std::vector<Scalar*> h_A(batch_count), h_B(batch_count), h_C(batch_count);
            for (int w = 0; w < D; w++)
                for (int s1 = 0; s1 < d; s1++)
                    for (int s2 = 0; s2 < d; s2++) {
                        int ws = w * dd + s1 * d + s2;
                        h_A[ws] = L_env + w * cL;
                        h_B[ws] = const_cast<Scalar*>(theta_src) + s1 * cL + s2 * cL * d;
                        h_C[ws] = T1 + ws * cL * cR;
                    }
            HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A.data(), batch_count*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
            HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B.data(), batch_count*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
            HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C.data(), batch_count*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
            ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
                Traits::op_t, rocblas_operation_none,
                cL, cR, cL,
                &one,
                (const Scalar**)d_batch_A_, cL * D,
                (const Scalar**)d_batch_B_, cL * dd,
                &zero_val,
                d_batch_C_, cL,
                batch_count));
        }
    }

    // Step 2: Dense GEMM — absorb fused WW
    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, dd * D, D * dd,
        &one,
        T1, cL * cR,
        WW, D * dd,
        &zero_val,
        T2, cL * cR));

    // Step 3: Loop of GEMMs — contract R_env
    if (sparse_s3) {
        HIP_CHECK(hipMemsetAsync(d_result, 0, (size_t)cL * dd * cR * sizeof(Scalar), stream_));
        int nnz = ww_nnz_cols_count_[site];
        const std::vector<int>& h_nnz = h_WW_nnz_cols_[site];
        Scalar beta_one = Traits::one();
        for (int idx = 0; idx < nnz; idx++) {
            int packed = h_nnz[idx];
            int n  = packed / dd;
            int ss = packed % dd;
            int s1p = ss / d, s2p = ss % d;
            int ws_out = n * dd + s1p * d + s2p;
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR,
                &one,
                T2 + ws_out * cL * cR, cL,
                R_env + n * cR, cR * D,
                &beta_one,
                d_result + s1p * cL + s2p * cL * d, cL * dd));
        }
    } else if (cL >= 16 && cR >= 16 && d <= 2) {
        // Strided batched over s1p when d<=2 and chi>=16 (avoids cache contention at batch_count>2)
        for (int s2p = 0; s2p < d; s2p++) {
            for (int n = 0; n < D; n++) {
                Scalar beta = (n == 0) ? Traits::zero() : Traits::one();
                // Batch over s1p: T2 columns at offsets n*dd+s1p*d+s2p are spaced by d in ws_out
                // A: T2 + (n*dd + 0*d + s2p)*cL*cR, strideA = d*cL*cR
                // B: R_env + n*cR, strideB = 0 (shared)
                // C: d_result + 0*cL + s2p*cL*d, strideC = cL (interleaved by s1p)
                ROCBLAS_CHECK(Traits::gemm_strided_batched(rocblas_h_,
                    rocblas_operation_none, rocblas_operation_none,
                    cL, cR, cR, &one,
                    T2 + (size_t)(n * dd + s2p) * cL * cR, cL, (rocblas_stride)(d * cL * cR),
                    R_env + n * cR, cR * D, (rocblas_stride)0,
                    &beta, d_result + s2p * cL * d, cL * dd, (rocblas_stride)cL, d));
            }
        }
    } else {
        for (int s1p = 0; s1p < d; s1p++) {
            for (int s2p = 0; s2p < d; s2p++) {
                for (int n = 0; n < D; n++) {
                    Scalar beta = (n == 0) ? Traits::zero() : Traits::one();
                    int ws_out = n * dd + s1p * d + s2p;
                    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                        rocblas_operation_none, rocblas_operation_none,
                        cL, cR, cR,
                        &one,
                        T2 + ws_out * cL * cR, cL,
                        R_env + n * cR, cR * D,
                        &beta,
                        d_result + s1p * cL + s2p * cL * d, cL * dd));
                }
            }
        }
    }

    if (graph_capture_miss) {
        hipGraph_t graph;
        HIP_CHECK(hipStreamEndCapture(stream_, &graph));
        hipGraphExec_t exec;
        HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        HIP_CHECK(hipGraphDestroy(graph));
        apply_heff_graph_cache_[graph_key(site, cL, cR)] = exec;
        HIP_CHECK(hipGraphLaunch(exec, stream_));
    }
}

// ============================================================================
// Left environment update
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::update_left_env(int site) {
    int chi_in = bond_dims_[site];
    int chi_out = bond_dims_[site + 1];
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    ensure_L_env_alloc(site + 1, chi_out);

    Scalar* L_env = d_L_envs_[site];
    Scalar* A = d_mps_tensors_[site];
    Scalar* W_mat = d_W_left_[site];
    Scalar* L_new = d_L_envs_[site + 1];
    Scalar* V = d_T1_;
    Scalar* U = d_T2_;

    // Step 1: V_ws[a',b] = L_w^T[a',a] * A_s[a,b]  (batched GEMM)
    {
        std::vector<Scalar*> h_A(D * d), h_B(D * d), h_C(D * d);
        for (int w = 0; w < D; w++)
            for (int s = 0; s < d; s++) {
                int ws = w * d + s;
                h_A[ws] = L_env + w * chi_in;
                h_B[ws] = A + s * chi_in;
                h_C[ws] = V + ws * chi_in * chi_out;
            }
        HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
            Traits::op_t, rocblas_operation_none,
            chi_in, chi_out, chi_in,
            &one,
            (const Scalar**)d_batch_A_, chi_in * D,
            (const Scalar**)d_batch_B_, chi_in * d,
            &zero_val,
            d_batch_C_, chi_in,
            D * d));
    }

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        chi_in * chi_out, d * D, D * d,
        &one,
        V, chi_in * chi_out,
        W_mat, D * d,
        &zero_val,
        U, chi_in * chi_out));

    // Step 3: L_new_w'[b,b'] = sum_{a',s'} U[a',ws',b] * conj(A[a',s',b'])
    for (int wp = 0; wp < D; wp++) {
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            int ws_out = wp * d + sp;
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                Traits::op_h, rocblas_operation_none,
                chi_out, chi_out, chi_in,
                &one,
                U + ws_out * chi_in * chi_out, chi_in,
                A + sp * chi_in, chi_in * d,
                &beta,
                L_new + wp * chi_out, chi_out * D));
        }
    }

    if constexpr (Traits::is_complex) {
        conjugate_inplace(L_new, chi_out * D * chi_out, stream_);
    }
}

// ============================================================================
// Right environment update
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::update_right_env(int site) {
    int chi_in = bond_dims_[site + 1];
    int chi_out = bond_dims_[site];
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    ensure_R_env_alloc(site, chi_out);

    Scalar* A = d_mps_tensors_[site];
    Scalar* R_env = d_R_envs_[site + 1];
    Scalar* W_mat = d_W_right_[site];
    Scalar* R_new = d_R_envs_[site];
    Scalar* V = d_T1_;
    Scalar* U = d_T2_;

    // Step 1: V_ws[a,b'] = A_s[a,b] * R_w'[b,b']  (batched GEMM)
    {
        std::vector<Scalar*> h_A(D * d), h_B(D * d), h_C(D * d);
        for (int wp = 0; wp < D; wp++)
            for (int s = 0; s < d; s++) {
                int ws = wp * d + s;
                h_A[ws] = A + s * chi_out;
                h_B[ws] = R_env + wp * chi_in;
                h_C[ws] = V + ws * chi_out * chi_in;
            }
        HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
            rocblas_operation_none, rocblas_operation_none,
            chi_out, chi_in, chi_in,
            &one,
            (const Scalar**)d_batch_A_, chi_out * d,
            (const Scalar**)d_batch_B_, chi_in * D,
            &zero_val,
            d_batch_C_, chi_out,
            D * d));
    }

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        chi_out * chi_in, d * D, D * d,
        &one,
        V, chi_out * chi_in,
        W_mat, D * d,
        &zero_val,
        U, chi_out * chi_in));

    // Step 3: R_new_w[a,a'] = sum_s' U_ws'[a,b'] * A_s'^H[b',a']
    for (int w = 0; w < D; w++) {
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            int ws_out = w * d + sp;
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, Traits::op_h,
                chi_out, chi_out, chi_in,
                &one,
                U + ws_out * chi_out * chi_in, chi_out,
                A + sp * chi_out, chi_out * d,
                &beta,
                R_new + w * chi_out, chi_out * D));
        }
    }
}

// ============================================================================
// Environment building
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::build_initial_environments() {
    // L[0] = trivial left boundary
    {
        std::vector<Scalar> h_L(D_mpo_, Traits::zero());
        h_L[0] = Traits::one();
        HIP_CHECK(hipMemcpy(d_L_envs_[0], h_L.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // R[L] = trivial right boundary
    {
        // With D_PAD, D_mpo_ > D_mpo_actual_; the identity slot must be at
        // the unpadded index D_mpo_actual_ - 1 because the padded W rows past
        // that index are zero.
        std::vector<Scalar> h_R(D_mpo_, Traits::zero());
        h_R[D_mpo_actual_ - 1] = Traits::one();
        HIP_CHECK(hipMemcpy(d_R_envs_[L_], h_R.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // Build all R environments from right to left
    for (int i = L_ - 1; i >= 0; i--) {
        update_right_env(i);
    }
}

// ============================================================================
// Lanczos eigensolver (fallback for small systems / Davidson failure)
// ============================================================================

template<typename Scalar>
double DMRG2GPUOpt<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta, int theta_size) {
    int n = theta_size;
    int max_iter = std::min(max_lanczos_iter_, n);
    double tol_lanczos = 1e-12;
    double tol_eig_conv = 1e-12;

    Scalar* d_lanczos_v = d_lanczos_v_;

    std::vector<double> h_alpha(max_iter);
    std::vector<double> h_beta(max_iter);

    // FUSE_LANCZOS: small device scratches to stage host scalars
    // for the fused kernels (which expect device-pointer operands).
    Scalar*   d_neg_alpha_scr = nullptr;
    Scalar*   d_neg_beta_scr  = nullptr;
    RealType* d_inv_beta_scr  = nullptr;
    if (opts_.fuse_lanczos) {
        HIP_CHECK(hipMalloc(&d_neg_alpha_scr, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_neg_beta_scr,  sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_inv_beta_scr,  sizeof(RealType)));
    }

    // v[0] = theta / ||theta||
    double norm;
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_theta, 1, &norm));

    if (norm < 1e-14) {
        std::vector<Scalar> h_init(n);
        srand(42 + site);
        for (int i = 0; i < n; i++) h_init[i] = Traits::random_val();
        HIP_CHECK(hipMemcpy(d_theta, h_init.data(), n * sizeof(Scalar), hipMemcpyHostToDevice));
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_theta, 1, &norm));
    }

    double inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, &inv_norm, d_theta, 1));
    HIP_CHECK(hipMemcpy(d_lanczos_v, d_theta, n * sizeof(Scalar), hipMemcpyDeviceToDevice));

    double prev_energy = 1e30;
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        Scalar* d_vi = d_lanczos_v + (size_t)iter * n;

        // w = H|v_i>
        apply_heff_two_site(site, d_vi, d_heff_result_);

        // alpha_i = <v_i|w>
        Scalar alpha_result;
        ROCBLAS_CHECK(Traits::dot(rocblas_h_, n, d_vi, 1, d_heff_result_, 1, &alpha_result));
        double alpha_i = Traits::real_part(alpha_result);
        h_alpha[iter] = alpha_i;

        // w = w - alpha_i * v_i  [and - beta_{i-1} * v_{i-1}]
        Scalar neg_alpha = Traits::make_scalar(-alpha_i);
        if (opts_.fuse_lanczos) {
            // Fused: w += (-α)·v_i + (-β_{i-1})·v_{i-1} in one kernel pass.
            // Stage host scalars to tiny device scratches.
            HIP_CHECK(hipMemcpyAsync(d_neg_alpha_scr, &neg_alpha, sizeof(Scalar),
                                     hipMemcpyHostToDevice, stream_));
            const Scalar* v_im1 = nullptr;
            const Scalar* nb_im1 = d_neg_alpha_scr;  // unused when has_prev==0
            if (iter > 0) {
                Scalar neg_beta = Traits::make_scalar(-h_beta[iter - 1]);
                HIP_CHECK(hipMemcpyAsync(d_neg_beta_scr, &neg_beta, sizeof(Scalar),
                                         hipMemcpyHostToDevice, stream_));
                v_im1 = d_lanczos_v + (size_t)(iter - 1) * n;
                nb_im1 = d_neg_beta_scr;
            }
            int block = 256;
            int grid = (n + block - 1) / block;
            hipLaunchKernelGGL((lanczos_fused_sub_kernel<Scalar>),
                               dim3(grid), dim3(block), 0, stream_,
                               d_heff_result_, d_vi, v_im1,
                               d_neg_alpha_scr, nb_im1,
                               n, iter > 0 ? 1 : 0);
        } else {
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, &neg_alpha, d_vi, 1, d_heff_result_, 1));

            // w = w - beta_{i-1} * v_{i-1}
            if (iter > 0) {
                Scalar neg_beta = Traits::make_scalar(-h_beta[iter - 1]);
                Scalar* d_vim1 = d_lanczos_v + (size_t)(iter - 1) * n;
                ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, &neg_beta, d_vim1, 1, d_heff_result_, 1));
            }
        }

        // Full reorthogonalization via gemv
        if (iter > 0) {
            Scalar one_val = Traits::one(), zero_sc = Traits::zero(), neg_one = Traits::neg(Traits::one());
            ROCBLAS_CHECK(Traits::gemv(rocblas_h_, Traits::op_h,
                n, iter + 1, &one_val,
                d_lanczos_v, n,
                d_heff_result_, 1,
                &zero_sc, d_ritz_coeffs_, 1));
            ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
                n, iter + 1, &neg_one,
                d_lanczos_v, n,
                d_ritz_coeffs_, 1,
                &one_val, d_heff_result_, 1));
        } else {
            Scalar overlap;
            ROCBLAS_CHECK(Traits::dot(rocblas_h_, n, d_lanczos_v, 1, d_heff_result_, 1, &overlap));
            Scalar neg_overlap = Traits::neg(overlap);
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, &neg_overlap, d_lanczos_v, 1, d_heff_result_, 1));
        }

        // beta_i = ||w||
        double beta_i;
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_heff_result_, 1, &beta_i));
        h_beta[iter] = beta_i;

        if (beta_i < tol_lanczos) {
            iter++;
            break;
        }

        // Eigenvalue convergence check (every 3 iterations after iter >= 4)
        // LANCZOS_FIXED skips the check entirely — no mid-loop host syncs.
        if (!opts_.lanczos_fixed && iter >= 4 && iter % 3 == 0) {
            int ncheck = iter + 1;
            std::vector<double> h_D_chk(ncheck), h_E_chk(ncheck);
            std::copy(h_alpha.begin(), h_alpha.begin() + ncheck, h_D_chk.begin());
            for (int i = 0; i < ncheck - 1; i++) h_E_chk[i] = h_beta[i];
            h_E_chk[ncheck - 1] = 0.0;
            const char jobz_n = 'N';
            const int n_chk = ncheck;
            std::vector<double> h_work_chk(1);
            int info_chk = 0;
            dstev_(&jobz_n, &n_chk, h_D_chk.data(), h_E_chk.data(), nullptr, &n_chk, h_work_chk.data(), &info_chk);
            if (info_chk == 0) {
                double cur_energy = h_D_chk[0];
                if (std::abs(cur_energy - prev_energy) < tol_eig_conv) {
                    iter++;
                    break;
                }
                prev_energy = cur_energy;
            }
        }

        // v_{i+1} = w / beta_i
        if (iter + 1 < max_iter) {
            Scalar* d_vip1 = d_lanczos_v + (size_t)(iter + 1) * n;
            double scale = 1.0 / beta_i;
            if (opts_.fuse_lanczos) {
                // Fused normalize+copy: v_{i+1}[k] = w[k] * (1/β_i)
                RealType inv_beta_rt = (RealType)scale;
                HIP_CHECK(hipMemcpyAsync(d_inv_beta_scr, &inv_beta_rt, sizeof(RealType),
                                         hipMemcpyHostToDevice, stream_));
                int block = 256;
                int grid = (n + block - 1) / block;
                hipLaunchKernelGGL((lanczos_fused_norm_copy_kernel<Scalar, RealType>),
                                   dim3(grid), dim3(block), 0, stream_,
                                   d_vip1, d_heff_result_, d_inv_beta_scr, n);
            } else {
                HIP_CHECK(hipMemcpy(d_vip1, d_heff_result_, n * sizeof(Scalar), hipMemcpyDeviceToDevice));
                ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, &scale, d_vip1, 1));
            }
        }
    }

    // Free FUSE_LANCZOS scratches
    if (opts_.fuse_lanczos) {
        if (d_neg_alpha_scr) hipFree(d_neg_alpha_scr);
        if (d_neg_beta_scr)  hipFree(d_neg_beta_scr);
        if (d_inv_beta_scr)  hipFree(d_inv_beta_scr);
    }

    int niter = iter;
    prof_davidson_iters += niter;
    prof_heff_calls += niter;

    // Solve tridiagonal eigenvalue problem on CPU
    std::vector<double> h_D(niter), h_E(niter), h_Z(niter * niter);
    std::vector<double> h_work(std::max(1, 2*niter - 2));
    int lapack_info = 0;

    std::copy(h_alpha.begin(), h_alpha.begin() + niter, h_D.begin());
    for (int i = 0; i < niter - 1; i++) h_E[i] = h_beta[i];
    if (niter > 0) h_E[niter - 1] = 0.0;

    const char jobz = 'V';
    const int n_lapack = niter;
    const int ldz = niter;

    dstev_(&jobz, &n_lapack, h_D.data(), h_E.data(), h_Z.data(), &ldz, h_work.data(), &lapack_info);

    if (lapack_info != 0) {
        throw std::runtime_error("LAPACK dstev failed with info = " + std::to_string(lapack_info));
    }

    double energy = h_D[0];

    // Reconstruct ground state: |theta> = sum_i c[i] |v_i>
    std::vector<Scalar> h_ritz_scalar(niter);
    for (int i = 0; i < niter; i++) {
        h_ritz_scalar[i] = Traits::make_scalar(h_Z[i]);
    }
    HIP_CHECK(hipMemcpy(d_ritz_coeffs_, h_ritz_scalar.data(), niter * sizeof(Scalar), hipMemcpyHostToDevice));

    Scalar one_sc = Traits::one(), zero_sc = Traits::zero();
    ROCBLAS_CHECK(Traits::gemv(
        rocblas_h_, rocblas_operation_none,
        n, niter, &one_sc,
        d_lanczos_v, n,
        d_ritz_coeffs_, 1,
        &zero_sc, d_theta, 1
    ));

    // Normalize
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_theta, 1, &norm));
    inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, &inv_norm, d_theta, 1));

    return energy;
}

// ============================================================================
// SVD bond splitting (CPU LAPACK)
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::svd_split_fallback(int site, Scalar* d_theta, char direction) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);

    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_user_);

    // CPU SVD path
    HIP_CHECK(hipMemcpy(h_svd_A_.data(), d_theta, m * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost));

    int lwork = (int)h_svd_work_.size();
    int info;
    const char jobu = 'S', jobvt = 'S';
    Traits::lapack_gesvd(&jobu, &jobvt, &m, &n_svd, h_svd_A_.data(), &m, h_svd_S_.data(),
            h_svd_U_.data(), &m, h_svd_Vh_.data(), &full_k,
            h_svd_work_.data(), &lwork,
            h_svd_rwork_.empty() ? nullptr : h_svd_rwork_.data(), &info);

    if (info != 0) {
        throw std::runtime_error("svd_split_fallback: LAPACK gesvd failed, info=" + std::to_string(info));
    }

    Scalar* h_U_data = h_svd_U_.data();
    RealType* h_S_data = h_svd_S_.data();
    Scalar* h_Vh_data = h_svd_Vh_.data();

    // Truncation
    int new_k = k;
    for (int i = 0; i < new_k; i++) {
        if (h_S_data[i] < 1e-14) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;

    if (direction == 'R') {
        // U -> MPS[site] (left-canonical), S*Vh -> MPS[site+1]
        allocate_mps_tensor(site, cL, new_k);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_U_data,
                            m * new_k * sizeof(Scalar), hipMemcpyHostToDevice));

        // Compute S*Vh on CPU: (new_k, n_svd)
        for (int j = 0; j < n_svd; j++)
            for (int i = 0; i < new_k; i++)
                h_svd_tmp_[i + j * new_k] = Traits::scale_by_real(h_S_data[i], h_Vh_data[i + j * full_k]);

        allocate_mps_tensor(site + 1, new_k, cR);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site + 1], h_svd_tmp_.data(),
                            new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));

    } else {  // direction == 'L'
        // U*S -> MPS[site], Vh -> MPS[site+1] (right-canonical)
        for (int j = 0; j < new_k; j++)
            for (int i = 0; i < m; i++)
                h_svd_tmp_[i + j * m] = Traits::scale_by_real(h_S_data[j], h_U_data[i + j * m]);

        allocate_mps_tensor(site, cL, new_k);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_svd_tmp_.data(),
                            m * new_k * sizeof(Scalar), hipMemcpyHostToDevice));

        allocate_mps_tensor(site + 1, new_k, cR);
        if (new_k == full_k) {
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site + 1], h_Vh_data,
                                full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));
        } else {
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    h_svd_tmp_[i + j * new_k] = h_Vh_data[i + j * full_k];
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site + 1], h_svd_tmp_.data(),
                                new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));
        }
    }

    bond_dims_[site + 1] = new_k;
}

// ============================================================================
// Block-Davidson Eigensolver
// ============================================================================

template<typename Scalar>
double DMRG2GPUOpt<Scalar>::block_davidson_eigensolver(int site, Scalar* d_theta, int theta_size) {
    int dim = theta_size;
    int b = std::min(davidson_b_, dim);
    int max_sub = std::min(davidson_max_sub_, dim);
    int max_iter = 30;
    double tol_dav = 1e-10;

    Scalar one = Traits::one(), zero_val = Traits::zero();
    Scalar neg_one = Traits::neg(Traits::one());

    // For tiny systems, use Lanczos fallback
    if (dim <= 2 * b) {
        return lanczos_eigensolver(site, d_theta, theta_size);
    }

    // Initialize V: first column = theta/||theta||
    RealType norm;
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, d_theta, 1, &norm));
    if (norm < 1e-14) {
        srand(42 + site);
        std::vector<Scalar> h_init(dim);
        for (int i = 0; i < dim; i++) h_init[i] = Traits::random_val();
        HIP_CHECK(hipMemcpy(d_theta, h_init.data(), dim * sizeof(Scalar), hipMemcpyHostToDevice));
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, d_theta, 1, &norm));
    }

    Scalar* V = d_dav_V_;   // (dim, max_sub)
    Scalar* AV = d_dav_AV_; // (dim, max_sub)

    // V[:, 0] = theta / norm
    HIP_CHECK(hipMemcpyAsync(V, d_theta, dim * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));
    RealType inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_norm, V, 1));

    // Fill remaining b-1 columns with random orthogonalized vectors
    srand(42 + site);
    for (int i = 1; i < b; i++) {
        std::vector<Scalar> h_v(dim);
        for (int j = 0; j < dim; j++) h_v[j] = Traits::random_val();
        HIP_CHECK(hipMemcpyAsync(V + (size_t)i * dim, h_v.data(), dim * sizeof(Scalar),
                                  hipMemcpyHostToDevice, stream_));

        // Orthogonalize against previous columns using CGS
        ROCBLAS_CHECK(Traits::gemv(rocblas_h_, Traits::op_h,
            dim, i, &one, V, dim, V + (size_t)i * dim, 1,
            &zero_val, d_dav_work_, 1));
        ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
            dim, i, &neg_one, V, dim, d_dav_work_, 1,
            &one, V + (size_t)i * dim, 1));

        // Normalize
        RealType nrm_v;
        HIP_CHECK(hipStreamSynchronize(stream_));
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, V + (size_t)i * dim, 1, &nrm_v));
        if (nrm_v < 1e-14) {
            for (int j = 0; j < dim; j++) h_v[j] = Traits::random_val();
            HIP_CHECK(hipMemcpy(V + (size_t)i * dim, h_v.data(), dim * sizeof(Scalar), hipMemcpyHostToDevice));
            ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, V + (size_t)i * dim, 1, &nrm_v));
        }
        RealType inv_nrm_v = 1.0 / nrm_v;
        ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_nrm_v, V + (size_t)i * dim, 1));
    }

    // Compute AV[:, j] = H @ V[:, j] for j = 0..b-1
    for (int j = 0; j < b; j++) {
        apply_heff_two_site(site, V + (size_t)j * dim, AV + (size_t)j * dim);
    }

    double best_energy = 1e30;
    double energy_prev = 1e30;
    int k = b;  // current subspace size

    for (int iteration = 0; iteration < max_iter; iteration++) {
        HIP_CHECK(hipStreamSynchronize(stream_));

        // Rayleigh-Ritz: H_proj = V^H @ AV  -> (k, k)
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            Traits::op_h, rocblas_operation_none,
            k, k, dim, &one, V, dim, AV, dim, &zero_val, d_dav_work_, k));

        // Copy H_proj to host
        HIP_CHECK(hipMemcpy(h_dav_H_proj_.data(), d_dav_work_,
                            k * k * sizeof(Scalar), hipMemcpyDeviceToHost));

        // Symmetrize on host: H_proj = 0.5 * (H_proj + H_proj^H)
        for (int i = 0; i < k; i++) {
            for (int j = i; j < k; j++) {
                Scalar hij = h_dav_H_proj_[i + j * k];
                Scalar hji = h_dav_H_proj_[j + i * k];
                Scalar sym;
                if constexpr (Traits::is_complex) {
                    sym = make_hipDoubleComplex(
                        0.5 * (hipCreal(hij) + hipCreal(hji)),
                        0.5 * (hipCimag(hij) - hipCimag(hji)));
                    h_dav_H_proj_[i + j * k] = sym;
                    h_dav_H_proj_[j + i * k] = make_hipDoubleComplex(hipCreal(sym), -hipCimag(sym));
                } else {
                    sym = 0.5 * (hij + hji);
                    h_dav_H_proj_[i + j * k] = sym;
                    h_dav_H_proj_[j + i * k] = sym;
                }
            }
        }

        // Eigendecompose H_proj on CPU
        std::copy(h_dav_H_proj_.begin(), h_dav_H_proj_.begin() + k * k,
                  h_dav_eigvecs_.begin());
        int info;
        const char jobz = 'V', uplo = 'U';
        int lwork = -1;
        Scalar work_opt;
        // Query workspace (need valid rwork for zheev_)
        std::vector<RealType> syev_rwork_q(std::max(1, Traits::syev_rwork_size(k)));
        Traits::lapack_syev(&jobz, &uplo, &k,
                h_dav_eigvecs_.data(), &k, h_dav_eigvals_.data(),
                &work_opt, &lwork,
                syev_rwork_q.empty() ? nullptr : syev_rwork_q.data(), &info);
        if constexpr (Traits::is_complex) {
            lwork = (int)Traits::real_part(work_opt) + 1;
        } else {
            lwork = (int)work_opt + 1;
        }
        std::vector<Scalar> syev_work(lwork);
        std::vector<RealType> syev_rwork(Traits::syev_rwork_size(k));
        Traits::lapack_syev(&jobz, &uplo, &k,
                h_dav_eigvecs_.data(), &k, h_dav_eigvals_.data(),
                syev_work.data(), &lwork,
                syev_rwork.empty() ? nullptr : syev_rwork.data(), &info);

        if (info != 0) {
            // Eigendecomp failed -- fall back to Lanczos
            return lanczos_eigensolver(site, d_theta, theta_size);
        }

        double energy = h_dav_eigvals_[0];  // lowest eigenvalue

        if (energy < best_energy) {
            best_energy = energy;
        }

        // Upload eigenvectors to GPU for Ritz vector computation
        HIP_CHECK(hipMemcpy(d_dav_work2_, h_dav_eigvecs_.data(),
                            k * k * sizeof(Scalar), hipMemcpyHostToDevice));

        // x0 = V @ eigvecs[:, 0] -> (dim, 1)
        ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
            dim, k, &one, V, dim, d_dav_work2_, 1, &zero_val, d_theta, 1));

        // ax0 = AV @ eigvecs[:, 0] -> (dim, 1)
        ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
            dim, k, &one, AV, dim, d_dav_work2_, 1, &zero_val, d_heff_result_, 1));

        // Residual: r = ax0 - energy * x0
        Scalar neg_energy = Traits::make_scalar(-energy);
        ROCBLAS_CHECK(Traits::axpy(rocblas_h_, dim, &neg_energy, d_theta, 1, d_heff_result_, 1));

        RealType res_norm;
        HIP_CHECK(hipStreamSynchronize(stream_));
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, d_heff_result_, 1, &res_norm));

        if (res_norm < tol_dav && std::abs(energy - energy_prev) < tol_dav) {
            // Converged: d_theta already has the ground state vector
            ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, d_theta, 1, &norm));
            inv_norm = 1.0 / norm;
            ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_norm, d_theta, 1));
            return energy;
        }
        energy_prev = energy;

        // Expand subspace with residual corrections
        int n_new = 0;
        for (int i = 0; i < std::min(b, k); i++) {
            Scalar* r_i = d_dav_work_ + (size_t)n_new * dim;

            // r_i = AV @ eigvecs[:, i] - eigvals[i] * V @ eigvecs[:, i]
            ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
                dim, k, &one, V, dim, d_dav_work2_ + i * k, 1, &zero_val, r_i, 1));
            ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
                dim, k, &one, AV, dim, d_dav_work2_ + i * k, 1, &zero_val, d_heff_result_, 1));
            Scalar neg_ei = Traits::make_scalar(-h_dav_eigvals_[i]);
            ROCBLAS_CHECK(Traits::scal(rocblas_h_, dim, &neg_ei, r_i, 1));
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, dim, &one, d_heff_result_, 1, r_i, 1));

            RealType ri_norm;
            HIP_CHECK(hipStreamSynchronize(stream_));
            ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, r_i, 1, &ri_norm));

            if (ri_norm > tol_dav * 0.01) {
                RealType inv_ri = 1.0 / ri_norm;
                ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_ri, r_i, 1));
                n_new++;
            }
        }

        if (n_new == 0) {
            ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, d_theta, 1, &norm));
            inv_norm = 1.0 / norm;
            ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_norm, d_theta, 1));
            return energy;
        }

        // Orthogonalize new vectors against V
        Scalar* W = d_dav_work_;

        // overlap = V^H @ W -> (k, n_new)
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            Traits::op_h, rocblas_operation_none,
            k, n_new, dim, &one, V, dim, W, dim, &zero_val, d_dav_work2_, k));

        // W -= V @ overlap
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            rocblas_operation_none, rocblas_operation_none,
            dim, n_new, k, &neg_one, V, dim, d_dav_work2_, k, &one, W, dim));

        // Orthogonalize new vectors among themselves (CGS within the block)
        int n_good = 0;
        for (int i = 0; i < n_new; i++) {
            Scalar* wi = W + (size_t)i * dim;

            for (int j = 0; j < n_good; j++) {
                Scalar* wj = W + (size_t)j * dim;
                Scalar overlap_val;
                ROCBLAS_CHECK(Traits::dot(rocblas_h_, dim, wj, 1, wi, 1, &overlap_val));
                Scalar neg_ov = Traits::neg(overlap_val);
                ROCBLAS_CHECK(Traits::axpy(rocblas_h_, dim, &neg_ov, wj, 1, wi, 1));
            }

            RealType wi_norm;
            HIP_CHECK(hipStreamSynchronize(stream_));
            ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, wi, 1, &wi_norm));

            if (wi_norm > 1e-14) {
                RealType inv_wi = 1.0 / wi_norm;
                ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_wi, wi, 1));
                if (n_good != i) {
                    HIP_CHECK(hipMemcpyAsync(W + (size_t)n_good * dim, wi, dim * sizeof(Scalar),
                                              hipMemcpyDeviceToDevice, stream_));
                }
                n_good++;
            }
        }

        if (n_good == 0) {
            ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, d_theta, 1, &norm));
            inv_norm = 1.0 / norm;
            ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_norm, d_theta, 1));
            return energy;
        }

        // Check if subspace would be too large -> restart
        if (k + n_good > max_sub) {
            int keep = std::min(b, k);

            // X_keep = V @ eigvecs[:, :keep] -> (dim, keep)
            HIP_CHECK(hipMemcpy(d_dav_work2_, h_dav_eigvecs_.data(),
                                k * k * sizeof(Scalar), hipMemcpyHostToDevice));
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                dim, keep, k, &one, V, dim, d_dav_work2_, k,
                &zero_val, d_dav_work_, dim));

            // Copy X_keep to V[:, :keep]
            HIP_CHECK(hipMemcpyAsync(V, d_dav_work_, (size_t)dim * keep * sizeof(Scalar),
                                      hipMemcpyDeviceToDevice, stream_));

            // Re-orthogonalize V columns (MGS on GPU)
            for (int i = 0; i < keep; i++) {
                for (int j = 0; j < i; j++) {
                    Scalar ov;
                    ROCBLAS_CHECK(Traits::dot(rocblas_h_, dim,
                        V + (size_t)j * dim, 1, V + (size_t)i * dim, 1, &ov));
                    Scalar neg_ov = Traits::neg(ov);
                    ROCBLAS_CHECK(Traits::axpy(rocblas_h_, dim, &neg_ov,
                        V + (size_t)j * dim, 1, V + (size_t)i * dim, 1));
                }
                RealType vi_norm;
                HIP_CHECK(hipStreamSynchronize(stream_));
                ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, V + (size_t)i * dim, 1, &vi_norm));
                RealType inv_vi = 1.0 / vi_norm;
                ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_vi, V + (size_t)i * dim, 1));
            }

            // Recompute AV for kept vectors
            for (int j = 0; j < keep; j++) {
                apply_heff_two_site(site, V + (size_t)j * dim, AV + (size_t)j * dim);
            }

            k = keep;
            continue;
        }

        // Expand: append new vectors and compute their H-images
        for (int j = 0; j < n_good; j++) {
            HIP_CHECK(hipMemcpyAsync(V + (size_t)(k + j) * dim, W + (size_t)j * dim,
                                      dim * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));
            apply_heff_two_site(site, V + (size_t)(k + j) * dim, AV + (size_t)(k + j) * dim);
        }
        k += n_good;
    }

    // Didn't converge -- use best result
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, d_theta, 1, &norm));
    inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_norm, d_theta, 1));
    return best_energy;
}

// ============================================================================
// Bond optimization (two-site)
// ============================================================================

template<typename Scalar>
double DMRG2GPUOpt<Scalar>::optimize_bond(int site, char direction) {
    form_theta_two_site(site);
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int theta_size = cL * d_ * d_ * cR;

    auto t0 = std::chrono::high_resolution_clock::now();
    double energy = block_davidson_eigensolver(site, d_theta_, theta_size);
    auto t1 = std::chrono::high_resolution_clock::now();
    prof_davidson_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipStreamSynchronize(stream_));
    svd_split_fallback(site, d_theta_, direction);
    t1 = std::chrono::high_resolution_clock::now();
    prof_svd_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    prof_site_count++;
    return energy;
}

// ============================================================================
// Sweep methods
// ============================================================================

template<typename Scalar>
double DMRG2GPUOpt<Scalar>::sweep_left_to_right() {
    double energy = 0.0;

    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_bond(site, 'R');
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto te0 = std::chrono::high_resolution_clock::now();
        update_left_env(site);
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto te1 = std::chrono::high_resolution_clock::now();
        prof_env_ms += std::chrono::duration<double, std::milli>(te1 - te0).count();
    }

    return energy;
}

template<typename Scalar>
double DMRG2GPUOpt<Scalar>::sweep_right_to_left() {
    double energy = 0.0;

    for (int site = L_ - 2; site >= 0; site--) {
        energy = optimize_bond(site, 'L');
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto te0 = std::chrono::high_resolution_clock::now();
        update_right_env(site + 1);
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto te1 = std::chrono::high_resolution_clock::now();
        prof_env_ms += std::chrono::duration<double, std::milli>(te1 - te0).count();
    }

    return energy;
}

// ============================================================================
// Main algorithm
// ============================================================================

template<typename Scalar>
double DMRG2GPUOpt<Scalar>::run(int n_sweeps) {
    const char* type_name = Traits::is_complex ? "complex128" : "float64";
    printf("=== GPU-Native Two-Site DMRG-OPT (Davidson + MFMA-16 pad + batched, %s) ===\n", type_name);
    if (chi_max_ != chi_max_user_)
        printf("L = %d, d = %d, chi_max = %d (padded from %d), D_mpo = %d\n",
               L_, d_, chi_max_, chi_max_user_, D_mpo_);
    else
        printf("L = %d, d = %d, chi_max = %d, D_mpo = %d\n", L_, d_, chi_max_, D_mpo_);
    printf("Running %d sweeps...\n\n", n_sweeps);

    auto t_start = std::chrono::high_resolution_clock::now();

    printf("Building initial environments...\n");
    build_initial_environments();

    auto t_envs = std::chrono::high_resolution_clock::now();
    double env_time = std::chrono::duration<double>(t_envs - t_start).count();
    printf("  Environment build: %.3f s\n\n", env_time);

    double energy_prev = 0.0;

    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        prof_davidson_ms = prof_svd_ms = prof_env_ms = 0;
        prof_davidson_iters = prof_site_count = prof_heff_calls = 0;

        auto t_sweep = std::chrono::high_resolution_clock::now();

        double energy_LR = sweep_left_to_right();
        double energy_RL = sweep_right_to_left();

        auto t_sweep_end = std::chrono::high_resolution_clock::now();
        double sweep_time = std::chrono::duration<double>(t_sweep_end - t_sweep).count();

        energy_ = energy_RL;
        double dE = std::abs(energy_ - energy_prev);

        // Print bond dimensions
        printf("Sweep %3d: E = %.12f, dE = %.2e, time = %.3f s  chi=[",
               sweep, energy_, dE, sweep_time);
        for (int i = 1; i < L_; i++) {
            printf("%d", bond_dims_[i]);
            if (i < L_ - 1) printf(",");
        }
        printf("]\n");

        double other_ms = sweep_time*1000.0 - prof_davidson_ms - prof_svd_ms - prof_env_ms;
        printf("  Profile: davidson=%.0fms (%d iters, %d heff) svd=%.0fms env=%.0fms other=%.0fms\n",
               prof_davidson_ms, prof_davidson_iters, prof_heff_calls,
               prof_svd_ms, prof_env_ms, other_ms);

        if (dE < tol_ && sweep > 0) {
            printf("Converged after %d sweeps!\n", sweep + 1);
            break;
        }

        energy_prev = energy_;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();
    printf("\nTotal wall time: %.3f s\n", total_time);
    report_timers();

    return energy_;
}

// ============================================================================
// Phase timers
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::init_timers() {
    t_lanczos_.init("lanczos", opts_.profile);
    t_apply_heff_.init("apply_heff", opts_.profile);
    t_svd_.init("svd", opts_.profile);
    t_absorb_.init("absorb", opts_.profile);
    t_env_update_.init("env_update", opts_.profile);
}

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::report_timers() {
    if (!opts_.profile) return;
    auto row = [](PhaseTimer& t) {
        double ms = t.total_ms();
        int c = t.calls();
        double per = c > 0 ? ms / c : 0.0;
        std::fprintf(stderr, "  %-12s: %10.2f ms   (%6d calls, %8.3f ms/call)\n",
                     t.name, ms, c, per);
    };
    std::fprintf(stderr, "== Phase timings ==\n");
    row(t_lanczos_);
    row(t_apply_heff_);
    row(t_svd_);
    row(t_absorb_);
    row(t_env_update_);
}

// ============================================================================
// Utility methods
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::get_mps(std::vector<std::vector<Scalar>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        h_mps[i].resize(size);
        HIP_CHECK(hipMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(Scalar), hipMemcpyDeviceToHost));
    }
}

#endif // DMRG2_GPU_OPT_IMPL_H
