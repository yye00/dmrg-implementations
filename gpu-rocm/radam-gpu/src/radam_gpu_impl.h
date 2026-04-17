#ifndef RADAM_GPU_IMPL_H
#define RADAM_GPU_IMPL_H

#include <rocsolver/rocsolver.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <stdexcept>

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

// ============================================================================
// Device kernels
// ============================================================================

// Real-scalar helpers: multiply by double-precision real from host/device
__device__ inline double radam_as_real(double x) { return x; }
__device__ inline double radam_as_real(hipDoubleComplex x) { return hipCreal(x); }

__device__ inline double radam_imag_part(double) { return 0.0; }
__device__ inline double radam_imag_part(hipDoubleComplex x) { return hipCimag(x); }

__device__ inline double  radam_scale_r(double  v, double s) { return v * s; }
__device__ inline hipDoubleComplex radam_scale_r(hipDoubleComplex v, double s) {
    return make_hipDoubleComplex(hipCreal(v)*s, hipCimag(v)*s);
}

__device__ inline double  radam_add(double a, double b) { return a + b; }
__device__ inline hipDoubleComplex radam_add(hipDoubleComplex a, hipDoubleComplex b) {
    return make_hipDoubleComplex(hipCreal(a)+hipCreal(b), hipCimag(a)+hipCimag(b));
}

// Elementwise: y += x
template<typename Scalar>
__global__ void elementwise_add_kernel(Scalar* __restrict__ y, const Scalar* __restrict__ x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = radam_add(y[idx], x[idx]);
}

// Elementwise: y = a*y + b*x  (real scalars a, b)
template<typename Scalar>
__global__ void axpby_real_kernel(Scalar* y, double a, const Scalar* x, double b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Scalar yi = y[idx];
        Scalar xi = x[idx];
        y[idx] = radam_add(radam_scale_r(yi, a), radam_scale_r(xi, b));
    }
}

// Adam moment update + step composition (single per-site kernel).
// Preconditions:
//   m[]:   previous first-moment (updated in-place)
//   g[]:   current projected Riemannian gradient; overwritten with step direction Delta
// Postconditions:
//   m[j]  <- b1*m[j] + (1-b1)*g_in[j]
//   g[j]  <- step_scale * m[j] / bc1     where step_scale is precomputed on host as
//                                          -lr / (sqrt(v_hat) + eps)
// bc1 = 1 - b1^k  (bias correction on the first moment)
template<typename Scalar>
__global__ void adam_update_kernel(Scalar* __restrict__ m, Scalar* __restrict__ g,
                                    double b1, double one_minus_b1,
                                    double step_scale, double bc1,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Update first moment.
        Scalar mi = m[idx];
        Scalar gi = g[idx];
        Scalar new_m = radam_add(radam_scale_r(mi, b1), radam_scale_r(gi, one_minus_b1));
        m[idx] = new_m;
        // Bias-corrected step direction written into g.
        double coeff = step_scale / bc1;
        g[idx] = radam_scale_r(new_m, coeff);
    }
}

// Inverse-sqrt helper run on device
__global__ inline void radam_inv_real_kernel(const double* in, double* out) {
    out[0] = 1.0 / in[0];
}

// ============================================================================
// Constructor
// ============================================================================

template<typename Scalar>
RAdamGPU<Scalar>::RAdamGPU(int L, int d, int chi_max, int D_mpo, double tol)
    : L_(L), d_(d), chi_max_(chi_max), D_mpo_(D_mpo), tol_(tol),
      energy_(0.0), n_epochs_done_(0), use_cpu_svd_(true) {

    bond_dims_.resize(L + 1);
    bond_dims_[0] = 1;
    bond_dims_[L] = 1;
    for (int i = 1; i < L; i++) {
        double exact = std::pow((double)d, std::min(i, L - i));
        bond_dims_[i] = (exact > chi_max) ? chi_max : (int)exact;
    }

    HIP_CHECK(hipStreamCreate(&stream_));
    ROCBLAS_CHECK(rocblas_create_handle(&handle_));
    ROCBLAS_CHECK(rocblas_set_stream(handle_, stream_));

    bond_size_max_ = chi_max_ * d_ * chi_max_;
    theta_size_max_ = bond_size_max_;
    max_lanczos_iter_ = std::min(100, theta_size_max_);

    allocate_gpu_memory();
}

template<typename Scalar>
RAdamGPU<Scalar>::~RAdamGPU() {
    free_gpu_memory();
    rocblas_destroy_handle(handle_);
    hipStreamDestroy(stream_);
}

// ============================================================================
// Memory allocation
// ============================================================================

template<typename Scalar>
void RAdamGPU<Scalar>::allocate_gpu_memory() {
    size_t mps_sz = max_site_size();

    d_mps_.assign(L_, nullptr);
    d_M_.assign(L_, nullptr);
    d_grad_.assign(L_, nullptr);
    d_mpo_.assign(L_, nullptr);
    d_W_left_.assign(L_, nullptr);
    d_W_right_.assign(L_, nullptr);
    d_L_H_.assign(L_ + 1, nullptr);
    d_R_H_.assign(L_ + 1, nullptr);
    d_L_N_.assign(L_ + 1, nullptr);

    for (int i = 0; i < L_; i++) {
        HIP_CHECK(hipMalloc(&d_mps_[i], mps_sz * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_M_[i],   mps_sz * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_grad_[i], mps_sz * sizeof(Scalar)));
        HIP_CHECK(hipMemsetAsync(d_M_[i], 0, mps_sz * sizeof(Scalar), stream_));
    }

    int env_sz = chi_max_ * D_mpo_ * chi_max_;
    for (int i = 0; i <= L_; i++) {
        int chi = (i == 0 || i == L_) ? 1 : chi_max_;
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_L_H_[i], sz * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_R_H_[i], sz * sizeof(Scalar)));
        HIP_CHECK(hipMemsetAsync(d_L_H_[i], 0, sz * sizeof(Scalar), stream_));
        HIP_CHECK(hipMemsetAsync(d_R_H_[i], 0, sz * sizeof(Scalar), stream_));

        int n_sz = chi * chi;
        HIP_CHECK(hipMalloc(&d_L_N_[i], n_sz * sizeof(Scalar)));
        HIP_CHECK(hipMemsetAsync(d_L_N_[i], 0, n_sz * sizeof(Scalar), stream_));
    }
    (void)env_sz;

    // Scratch
    int t_max = D_mpo_ * d_ * d_ * chi_max_ * chi_max_;
    HIP_CHECK(hipMalloc(&d_T1_, (size_t)t_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_T2_, (size_t)t_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_heff_result_, (size_t)theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_theta_,       (size_t)theta_size_max_ * sizeof(Scalar)));
    int batch_max = D_mpo_ * d_ * d_;
    HIP_CHECK(hipMalloc(&d_batch_A_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_B_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_C_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_dot_result_,  sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_nrm2_result_, sizeof(RealType)));

    // Lanczos
    HIP_CHECK(hipMalloc(&d_lanczos_v_,        (size_t)max_lanczos_iter_ * theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_ritz_coeffs_,      max_lanczos_iter_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_neg_alpha_,        sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_neg_overlap_,      sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_inv_nrm_,          sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_alpha_dev_,        max_lanczos_iter_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_beta_dev_,         max_lanczos_iter_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_neg_beta_scalars_, max_lanczos_iter_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_const_one_,        sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_const_zero_,       sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_const_neg_one_,    sizeof(Scalar)));
    Scalar one = Traits::one(), zero_v = Traits::zero(), neg_one = Traits::neg(one);
    HIP_CHECK(hipMemcpyAsync(d_const_one_,     &one,     sizeof(Scalar), hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_const_zero_,    &zero_v,  sizeof(Scalar), hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_const_neg_one_, &neg_one, sizeof(Scalar), hipMemcpyHostToDevice, stream_));

    // CPU SVD workspace
    int svd_max_m = chi_max_ * d_;
    int svd_max_n = d_ * chi_max_;
    int svd_max_k = std::min(svd_max_m, svd_max_n);
    int lwork_svd = std::max(3 * svd_max_k + std::max(svd_max_m, svd_max_n),
                              5 * svd_max_k);
    h_svd_A_.resize((size_t)svd_max_m * svd_max_n);
    h_svd_U_.resize((size_t)svd_max_m * svd_max_k);
    h_svd_Vh_.resize((size_t)svd_max_k * svd_max_n);
    h_svd_S_.resize(svd_max_k);
    h_svd_work_.resize(lwork_svd);
    h_svd_tmp_.resize(std::max((size_t)svd_max_m * svd_max_n,
                                (size_t)svd_max_m * chi_max_ * d_));
    h_svd_rwork_.resize(Traits::svd_rwork_size(svd_max_m, svd_max_n));

    HIP_CHECK(hipMalloc(&d_svd_work_, (size_t)svd_max_m * svd_max_n * sizeof(Scalar)));

    HIP_CHECK(hipStreamSynchronize(stream_));
}

template<typename Scalar>
void RAdamGPU<Scalar>::free_gpu_memory() {
    auto safe_free = [](void* p){ if (p) hipFree(p); };
    for (auto p : d_mps_)     safe_free(p);
    for (auto p : d_M_)       safe_free(p);
    for (auto p : d_grad_)    safe_free(p);
    for (auto p : d_mpo_)     safe_free(p);
    for (auto p : d_W_left_)  safe_free(p);
    for (auto p : d_W_right_) safe_free(p);
    for (auto p : d_L_H_)     safe_free(p);
    for (auto p : d_R_H_)     safe_free(p);
    for (auto p : d_L_N_)     safe_free(p);
    safe_free(d_T1_); safe_free(d_T2_);
    safe_free(d_heff_result_); safe_free(d_theta_);
    safe_free(d_batch_A_); safe_free(d_batch_B_); safe_free(d_batch_C_);
    safe_free(d_dot_result_); safe_free(d_nrm2_result_);
    safe_free(d_lanczos_v_); safe_free(d_ritz_coeffs_);
    safe_free(d_neg_alpha_); safe_free(d_neg_overlap_); safe_free(d_inv_nrm_);
    safe_free(d_alpha_dev_); safe_free(d_beta_dev_); safe_free(d_neg_beta_scalars_);
    safe_free(d_const_one_); safe_free(d_const_zero_); safe_free(d_const_neg_one_);
    safe_free(d_svd_work_);
}

// ============================================================================
// MPO upload and fusion (W_left, W_right used by apply_heff / update_*_env)
// ============================================================================

template<typename Scalar>
void RAdamGPU<Scalar>::set_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    upload_and_fuse_mpo(h_mpo_tensors);
}

template<typename Scalar>
void RAdamGPU<Scalar>::upload_and_fuse_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    int D = D_mpo_, d = d_;
    for (int i = 0; i < L_; i++) {
        int sz = D * d * d * D;
        HIP_CHECK(hipMalloc(&d_mpo_[i], sz * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_mpo_[i], h_mpo_tensors[i], sz * sizeof(Scalar),
                            hipMemcpyHostToDevice));

        std::vector<Scalar> h_WL(sz, Traits::zero());
        std::vector<Scalar> h_WR(sz, Traits::zero());
        for (int w = 0; w < D; w++)
            for (int s = 0; s < d; s++)
                for (int sp = 0; sp < d; sp++)
                    for (int wp = 0; wp < D; wp++) {
                        Scalar val = h_mpo_tensors[i][w + s*D + sp*D*d + wp*D*d*d];
                        h_WL[(w*d+s) + (wp*d+sp) * D * d] = val;
                        h_WR[(wp*d+s) + (w*d+sp) * D * d] = val;
                    }
        HIP_CHECK(hipMalloc(&d_W_left_[i],  sz * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_W_right_[i], sz * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_W_left_[i],  h_WL.data(), sz * sizeof(Scalar), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_W_right_[i], h_WR.data(), sz * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

// ============================================================================
// MPS initialization + right-canonicalization (host QR sweep)
// ============================================================================

template<typename Scalar>
void RAdamGPU<Scalar>::initialize_mps_random(double scale, int seed) {
    srand(seed);
    for (int i = 0; i < L_; i++) {
        int sz = chi_L(i) * d_ * chi_R(i);
        std::vector<Scalar> h_A(sz);
        for (int j = 0; j < sz; j++) {
            h_A[j] = Traits::scale_by_real(scale, Traits::random_val());
        }
        HIP_CHECK(hipMemcpy(d_mps_[i], h_A.data(), sz * sizeof(Scalar),
                            hipMemcpyHostToDevice));
    }
    right_canonicalize_mps();
    normalize_site0();
}

// Host-side QR sweep using LAPACK. Copies each site to host, QR, copies back.
// bond dims are preserved (thin QR).
template<typename Scalar>
void RAdamGPU<Scalar>::right_canonicalize_mps() {
    // Right-canonicalize from site L-1 down to site 1.
    // For site i (i >= 1), flatten (chi_L, d*chi_R) and QR the transposed matrix so
    //   Q has orthonormal rows in the (chi_L, d*chi_R) flattening.
    for (int i = L_ - 1; i >= 1; i--) {
        int cL = chi_L(i), cR = chi_R(i);
        int rows = cL;
        int cols = d_ * cR;
        std::vector<Scalar> h_A((size_t)rows * cols);
        HIP_CHECK(hipMemcpy(h_A.data(), d_mps_[i],
                            (size_t)rows * cols * sizeof(Scalar), hipMemcpyDeviceToHost));

        // A is stored col-major (rows × cols) = (chi_L × d*chi_R):
        //   h_A[a + p*rows] = A[a, p]     for a ∈ [0, rows), p ∈ [0, cols).
        // Build M = A^H, col-major (cols × rows):
        //   h_M[p + a*cols] = conj(A[a, p]) = conj(h_A[a + p*rows]).
        std::vector<Scalar> h_M((size_t)cols * rows);
        for (int a = 0; a < rows; a++) {
            for (int p = 0; p < cols; p++) {
                Scalar v = h_A[a + (size_t)p * rows];
                if constexpr (Traits::is_complex) v = hipConj(v);
                h_M[p + (size_t)a * cols] = v;
            }
        }

        // geqrf expects column-major. M is (cols × rows), col-major lda=cols.
        int m_qr = cols, n_qr = rows;
        int min_mn = std::min(m_qr, n_qr);
        std::vector<Scalar> tau(min_mn);
        std::vector<Scalar> work(1);
        int lwork = -1, info = 0;

        // Workspace query
        if constexpr (Traits::is_complex) {
            zgeqrf_(&m_qr, &n_qr, h_M.data(), &m_qr, tau.data(), work.data(), &lwork, &info);
            lwork = (int)hipCreal(work[0]);
        } else {
            dgeqrf_(&m_qr, &n_qr, (double*)h_M.data(), &m_qr, (double*)tau.data(),
                    (double*)work.data(), &lwork, &info);
            lwork = (int)*((double*)work.data());
        }
        work.resize(lwork);

        // QR factorize
        if constexpr (Traits::is_complex) {
            zgeqrf_(&m_qr, &n_qr, h_M.data(), &m_qr, tau.data(), work.data(), &lwork, &info);
        } else {
            dgeqrf_(&m_qr, &n_qr, (double*)h_M.data(), &m_qr, (double*)tau.data(),
                    (double*)work.data(), &lwork, &info);
        }
        if (info != 0) throw std::runtime_error("geqrf failed");

        // Extract R (upper triangle of M[0..min_mn, 0..n_qr])
        std::vector<Scalar> h_R((size_t)min_mn * n_qr, Traits::zero());
        for (int j = 0; j < n_qr; j++)
            for (int k = 0; k <= std::min(j, min_mn - 1); k++)
                h_R[k + j * min_mn] = h_M[k + j * m_qr];

        // Form Q explicitly. Q is m_qr × min_mn.
        if constexpr (Traits::is_complex) {
            zungqr_(&m_qr, &min_mn, &min_mn, h_M.data(), &m_qr, tau.data(),
                    work.data(), &lwork, &info);
        } else {
            dorgqr_(&m_qr, &min_mn, &min_mn, (double*)h_M.data(), &m_qr, (double*)tau.data(),
                    (double*)work.data(), &lwork, &info);
        }
        if (info != 0) throw std::runtime_error("orgqr failed");

        // Q is m_qr × min_mn, representing (d*chi_R × chi_L) with orthonormal columns.
        // The right-canonical core is A_new = Q^H reshaped to (chi_L_new, d, chi_R), where
        //   Q^H is (chi_L × d*chi_R) and new chi_L = min_mn = min(d*chi_R, chi_L).
        int new_chi_L = min_mn;
        std::vector<Scalar> h_A_new((size_t)new_chi_L * d_ * cR);
        // Col-major (new_chi_L × d*chi_R): A_new[i + j*new_chi_L] = conj(Q[j + i*m_qr])
        if constexpr (Traits::is_complex) {
            for (int j = 0; j < cols; j++)
                for (int i = 0; i < new_chi_L; i++)
                    h_A_new[i + j * new_chi_L] = hipConj(h_M[j + i * m_qr]);
        } else {
            for (int j = 0; j < cols; j++)
                for (int i = 0; i < new_chi_L; i++)
                    h_A_new[i + j * new_chi_L] = h_M[j + i * m_qr];
        }

        // Upload site i back.
        HIP_CHECK(hipMemcpy(d_mps_[i], h_A_new.data(),
                            (size_t)new_chi_L * d_ * cR * sizeof(Scalar),
                            hipMemcpyHostToDevice));
        bond_dims_[i] = new_chi_L;

        // Form R^H (conj-transpose of R) and absorb into left neighbour's right bond.
        // R is (min_mn × n_qr), we need R^H of shape (n_qr × min_mn) = (chi_L_old × new_chi_L).
        std::vector<Scalar> h_RH((size_t)n_qr * min_mn);
        for (int i_row = 0; i_row < min_mn; i_row++) {
            for (int k = 0; k < n_qr; k++) {
                Scalar v = h_R[i_row + (size_t)k * min_mn];
                if constexpr (Traits::is_complex) v = hipConj(v);
                h_RH[k + (size_t)i_row * n_qr] = v;
            }
        }

        // Contract R^H into left-neighbour's right bond:
        //   B_new[l, p, k] = sum_s B_old[l, p, s] * RH[s, k],  where B_old shape (cL_prev, d, chi_L_old)
        // Copy B_old to host, contract, copy back.
        int cL_prev = chi_L(i - 1);
        int cR_old  = n_qr;   // old chi_L of site i, before we overwrote bond_dims_[i] above
        int B_old_sz = cL_prev * d_ * cR_old;
        std::vector<Scalar> h_B_old(B_old_sz);
        HIP_CHECK(hipMemcpy(h_B_old.data(), d_mps_[i - 1], B_old_sz * sizeof(Scalar), hipMemcpyDeviceToHost));

        // B_new dims: (cL_prev, d, new_chi_L)
        int B_new_sz = cL_prev * d_ * new_chi_L;
        std::vector<Scalar> h_B_new(B_new_sz, Traits::zero());
        // Flatten B_old as (cL_prev*d, cR_old) col-major, then B_new = B_old * R^H
        //   B_new (cL_prev*d × new_chi_L) = B_old (cL_prev*d × cR_old) * R^H (cR_old × new_chi_L)
        int rows_b = cL_prev * d_;
        int cols_b = new_chi_L;
        int k_b = cR_old;
        for (int col = 0; col < cols_b; col++) {
            for (int row = 0; row < rows_b; row++) {
                Scalar acc = Traits::zero();
                for (int kk = 0; kk < k_b; kk++) {
                    Scalar a = h_B_old[row + kk * rows_b];
                    Scalar b = h_RH[kk + col * k_b];
                    if constexpr (Traits::is_complex) acc = hipCadd(acc, hipCmul(a, b));
                    else acc += a * b;
                }
                h_B_new[row + col * rows_b] = acc;
            }
        }
        HIP_CHECK(hipMemcpy(d_mps_[i - 1], h_B_new.data(), B_new_sz * sizeof(Scalar),
                            hipMemcpyHostToDevice));
        // bond_dims_[i] is already updated; neighbour's chi_R is bond_dims_[i].
    }
}

template<typename Scalar>
void RAdamGPU<Scalar>::normalize_site0() {
    int sz = chi_L(0) * d_ * chi_R(0);
    RealType nrm;
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host));
    ROCBLAS_CHECK(Traits::nrm2(handle_, sz, d_mps_[0], 1, &nrm));
    if (nrm > 1e-30) {
        RealType inv = 1.0 / nrm;
        ROCBLAS_CHECK(Traits::scal_real(handle_, sz, &inv, d_mps_[0], 1));
    }
    HIP_CHECK(hipStreamSynchronize(stream_));
}

// ============================================================================
// Left H-environment update: chains the (chi, D_mpo, chi) env through site i
// ============================================================================

template<typename Scalar>
void RAdamGPU<Scalar>::update_left_H_env(int site) {
    int chi_in  = chi_L(site);
    int chi_out = chi_R(site);
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    Scalar* L_env = d_L_H_[site];
    Scalar* A     = d_mps_[site];
    Scalar* W_mat = d_W_left_[site];
    Scalar* L_new = d_L_H_[site + 1];
    Scalar* V = d_T1_;
    Scalar* U = d_T2_;

    // Step 1: batched GEMM — V[w*d+s][a',b] = L[w][:,a']^T * A[s][:,b]
    {
        int batch_count = D * d;
        std::vector<Scalar*> h_A(batch_count), h_B(batch_count), h_C(batch_count);
        for (int w = 0; w < D; w++)
            for (int s = 0; s < d; s++) {
                int idx = w * d + s;
                h_A[idx] = L_env + (size_t)w * chi_in;
                h_B[idx] = A     + (size_t)s * chi_in;
                h_C[idx] = V     + (size_t)idx * chi_in * chi_out;
            }
        HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A.data(), batch_count * sizeof(Scalar*),
                                 hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B.data(), batch_count * sizeof(Scalar*),
                                 hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C.data(), batch_count * sizeof(Scalar*),
                                 hipMemcpyHostToDevice, stream_));
        ROCBLAS_CHECK(Traits::gemm_batched(handle_,
            Traits::op_t, rocblas_operation_none,
            chi_in, chi_out, chi_in,
            &one,
            (const Scalar**)d_batch_A_, chi_in * D,
            (const Scalar**)d_batch_B_, chi_in * d,
            &zero_val,
            d_batch_C_, chi_in,
            batch_count));
    }

    // Step 2: dense GEMM — U = V * W_left
    ROCBLAS_CHECK(Traits::gemm(handle_,
        rocblas_operation_none, rocblas_operation_none,
        chi_in * chi_out, d * D, D * d,
        &one,
        V, chi_in * chi_out,
        W_mat, D * d,
        &zero_val,
        U, chi_in * chi_out));

    // Step 3: loop over MPO index wp and physical index sp on the main stream.
    //   L_new[wp][a', b'] = sum_{sp} (U + (wp*d+sp) * chi_in * chi_out)^H * A[sp][:, b']
    for (int wp = 0; wp < D; wp++) {
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            ROCBLAS_CHECK(Traits::gemm(handle_,
                Traits::op_h, rocblas_operation_none,
                chi_out, chi_out, chi_in,
                &one,
                U + (size_t)(wp * d + sp) * chi_in * chi_out, chi_in,
                A + (size_t)sp * chi_in,                       chi_in * d,
                &beta,
                L_new + (size_t)wp * chi_out, chi_out * D));
        }
    }

    if constexpr (Traits::is_complex) {
        conjugate_inplace(L_new, chi_out * D * chi_out, stream_);
    }
}

// ============================================================================
// Right H-environment update
// ============================================================================

template<typename Scalar>
void RAdamGPU<Scalar>::update_right_H_env(int site) {
    int chi_in  = chi_R(site);
    int chi_out = chi_L(site);
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    Scalar* A     = d_mps_[site];
    Scalar* R_env = d_R_H_[site + 1];
    Scalar* W_mat = d_W_right_[site];
    Scalar* R_new = d_R_H_[site];
    Scalar* V = d_T1_;
    Scalar* U = d_T2_;

    // Step 1: batched GEMM — V[wp*d+s][a,b'] = A[s][a, :] * R[wp][:, b']
    {
        int batch_count = D * d;
        std::vector<Scalar*> h_A(batch_count), h_B(batch_count), h_C(batch_count);
        for (int wp = 0; wp < D; wp++)
            for (int s = 0; s < d; s++) {
                int idx = wp * d + s;
                h_A[idx] = A     + (size_t)s  * chi_out;
                h_B[idx] = R_env + (size_t)wp * chi_in;
                h_C[idx] = V     + (size_t)idx * chi_in * chi_out;
            }
        HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A.data(), batch_count * sizeof(Scalar*),
                                 hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B.data(), batch_count * sizeof(Scalar*),
                                 hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C.data(), batch_count * sizeof(Scalar*),
                                 hipMemcpyHostToDevice, stream_));
        ROCBLAS_CHECK(Traits::gemm_batched(handle_,
            rocblas_operation_none, rocblas_operation_none,
            chi_out, chi_in, chi_in,
            &one,
            (const Scalar**)d_batch_A_, chi_out * d,
            (const Scalar**)d_batch_B_, chi_in * D,
            &zero_val,
            d_batch_C_, chi_out,
            batch_count));
    }

    // Step 2: dense GEMM — U = V * W_right
    ROCBLAS_CHECK(Traits::gemm(handle_,
        rocblas_operation_none, rocblas_operation_none,
        chi_out * chi_in, d * D, D * d,
        &one,
        V, chi_out * chi_in,
        W_mat, D * d,
        &zero_val,
        U, chi_out * chi_in));

    // Step 3: loop over MPO index w and physical index sp
    for (int w = 0; w < D; w++) {
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            ROCBLAS_CHECK(Traits::gemm(handle_,
                rocblas_operation_none, Traits::op_h,
                chi_out, chi_out, chi_in,
                &one,
                U + (size_t)(w * d + sp) * chi_out * chi_in, chi_out,
                A + (size_t)sp * chi_out,                    chi_out * d,
                &beta,
                R_new + (size_t)w * chi_out, chi_out * D));
        }
    }
}

// ============================================================================
// Environment building
// ============================================================================

template<typename Scalar>
void RAdamGPU<Scalar>::build_initial_H_envs() {
    std::vector<Scalar> h_L(D_mpo_, Traits::zero());
    h_L[0] = Traits::one();
    HIP_CHECK(hipMemcpy(d_L_H_[0], h_L.data(), D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));

    std::vector<Scalar> h_R(D_mpo_, Traits::zero());
    h_R[D_mpo_ - 1] = Traits::one();
    HIP_CHECK(hipMemcpy(d_R_H_[L_], h_R.data(), D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));

    for (int i = 0; i < L_; i++) update_left_H_env(i);
    HIP_CHECK(hipStreamSynchronize(stream_));
    for (int i = L_ - 1; i >= 0; i--) update_right_H_env(i);
    HIP_CHECK(hipStreamSynchronize(stream_));
}

// Left norm envs: L_N[i+1](b_bra, b_ket) = sum_{s, a_bra, a_ket}
//     L_N[i](a_bra, a_ket) * conj(A[a_bra, s, b_bra]) * A[a_ket, s, b_ket]
// Stored as (chi × chi) column-major. For right-canonical MPS, L_N[i] = I for i<=0? no, L_N[0]=1×1 ones.
// Here we build L_N always; right-canonical gauge makes R_N = I which we simply skip.
template<typename Scalar>
void RAdamGPU<Scalar>::build_all_L_N_envs() {
    Scalar one = Traits::one();
    HIP_CHECK(hipMemcpy(d_L_N_[0], &one, sizeof(Scalar), hipMemcpyHostToDevice));

    for (int i = 0; i < L_; i++) {
        int cL = chi_L(i), cR = chi_R(i);
        Scalar* A = d_mps_[i];
        Scalar* Lprev = d_L_N_[i];
        Scalar* Lnew  = d_L_N_[i + 1];
        Scalar  one_s = Traits::one(), zero_v = Traits::zero();

        // Step 1: T[a_bra, s, b_ket] = sum_{a_ket} L[a_bra, a_ket] * A[a_ket, s, b_ket]
        //   flatten A as (cL × d*cR); T as (cL × d*cR).
        ROCBLAS_CHECK(Traits::gemm(handle_,
            rocblas_operation_none, rocblas_operation_none,
            cL, d_ * cR, cL,
            &one_s,
            Lprev, cL,
            A, cL,
            &zero_v,
            d_T1_, cL));

        // Step 2: Lnew[b_bra, b_ket] = sum_{a_bra, s} conj(A[a_bra, s, b_bra]) * T[a_bra, s, b_ket]
        //   flatten A as (cL*d × cR) for the bra, T as (cL*d × cR).
        //   Lnew (cR × cR) = A^H (cR × cL*d) * T (cL*d × cR)
        ROCBLAS_CHECK(Traits::gemm(handle_,
            Traits::op_h, rocblas_operation_none,
            cR, cR, cL * d_,
            &one_s,
            A, cL * d_,
            d_T1_, cL * d_,
            &zero_v,
            Lnew, cR));
    }
}

// ============================================================================
// Effective Hamiltonian: apply_heff(site, in, out) — port of apply_heff_single_site
// ============================================================================

template<typename Scalar>
void RAdamGPU<Scalar>::apply_heff(int site, const Scalar* d_in, Scalar* d_out) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    Scalar* L_env = d_L_H_[site];
    Scalar* R_env = d_R_H_[site + 1];
    Scalar* W_mat = d_W_left_[site];
    Scalar* V = d_T1_;
    Scalar* U = d_T2_;

    // Step 1: V[w*d+s][a',b] = L[w][:,a']^T * theta[s][:,b]  (D*d batched GEMMs)
    {
        int batch_count = D * d;
        std::vector<Scalar*> h_A(batch_count), h_B(batch_count), h_C(batch_count);
        for (int w = 0; w < D; w++)
            for (int s = 0; s < d; s++) {
                int idx = w * d + s;
                h_A[idx] = L_env + (size_t)w * cL;
                h_B[idx] = const_cast<Scalar*>(d_in) + (size_t)s * cL;
                h_C[idx] = V + (size_t)idx * cL * cR;
            }
        HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A.data(), batch_count * sizeof(Scalar*),
                                 hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B.data(), batch_count * sizeof(Scalar*),
                                 hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C.data(), batch_count * sizeof(Scalar*),
                                 hipMemcpyHostToDevice, stream_));
        ROCBLAS_CHECK(Traits::gemm_batched(handle_,
            Traits::op_t, rocblas_operation_none,
            cL, cR, cL,
            &one,
            (const Scalar**)d_batch_A_, cL * D,
            (const Scalar**)d_batch_B_, cL * d,
            &zero_val,
            d_batch_C_, cL,
            batch_count));
    }

    // Step 2: U = V * W_left
    ROCBLAS_CHECK(Traits::gemm(handle_,
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, d * D, D * d,
        &one,
        V, cL * cR,
        W_mat, D * d,
        &zero_val,
        U, cL * cR));

    // Step 3: out[sp][a',b'] = sum_{wp} U[wp*d+sp][:,b] * R[wp][b,:]
    for (int wp = 0; wp < D; wp++) {
        Scalar beta = (wp == 0) ? Traits::zero() : Traits::one();
        std::vector<Scalar*> h_A(d), h_B(d), h_C(d);
        for (int sp = 0; sp < d; sp++) {
            h_A[sp] = U + (size_t)(wp * d + sp) * cL * cR;
            h_B[sp] = R_env + (size_t)wp * cR;
            h_C[sp] = d_out + (size_t)sp * cL;
        }
        HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A.data(), d * sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B.data(), d * sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C.data(), d * sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        ROCBLAS_CHECK(Traits::gemm_batched(handle_,
            rocblas_operation_none, rocblas_operation_none,
            cL, cR, cR, &one,
            (const Scalar**)d_batch_A_, cL,
            (const Scalar**)d_batch_B_, cR * D,
            &beta,
            d_batch_C_, cL * d, d));
    }
}

// Apply N_eff at site: out = L_N[site] * A, using R_N = I (right-canonical gauge).
// A is (chi_L, d, chi_R); L_N is (chi_L × chi_L). Output has same shape as A.
template<typename Scalar>
void RAdamGPU<Scalar>::apply_norm(int site, const Scalar* d_in, Scalar* d_out) {
    int cL = chi_L(site), cR = chi_R(site);
    Scalar one = Traits::one(), zero_v = Traits::zero();
    // Flatten A as (cL × d*cR): out = L_N * A (GEMM).
    ROCBLAS_CHECK(Traits::gemm(handle_,
        rocblas_operation_none, rocblas_operation_none,
        cL, d_ * cR, cL,
        &one,
        d_L_N_[site], cL,
        d_in, cL,
        &zero_v,
        d_out, cL));
}

// ============================================================================
// Gradient: compute_all_gradients returns energy E
//   For each site i: d_grad_[i] = (1/xx) * (H_eff * A_i - E * N_eff * A_i)
// ============================================================================

template<typename Scalar>
double RAdamGPU<Scalar>::compute_energy_from_envs() {
    // L_H_[L] has layout (chi_L=1, D_mpo, chi_R=1) col-major → D_mpo entries at offsets 0..D_mpo-1.
    // The right boundary vector selects channel D_mpo-1 (the "final" MPO row of the chain),
    // yielding the scalar <X|H|X>.
    // L_N_[L] is 1×1 → single scalar <X|X>.
    Scalar h_xhx, h_xx;
    HIP_CHECK(hipMemcpy(&h_xhx, d_L_H_[L_] + (size_t)(D_mpo_ - 1), sizeof(Scalar),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_xx,  d_L_N_[L_], sizeof(Scalar), hipMemcpyDeviceToHost));
    double xhx = Traits::real_part(h_xhx);
    double xx  = Traits::real_part(h_xx);
    if (xx <= 0.0) throw std::runtime_error("compute_energy: <X|X> non-positive");
    return xhx / xx;
}

template<typename Scalar>
double RAdamGPU<Scalar>::compute_all_gradients() {
    build_initial_H_envs();
    build_all_L_N_envs();

    double E = compute_energy_from_envs();

    // Get <X|X> again for inv_xx.
    Scalar h_xx;
    HIP_CHECK(hipMemcpy(&h_xx, d_L_N_[L_], sizeof(Scalar), hipMemcpyDeviceToHost));
    double xx = Traits::real_part(h_xx);
    double inv_xx = 1.0 / xx;

    for (int i = 0; i < L_; i++) {
        int sz = chi_L(i) * d_ * chi_R(i);
        // 1) d_grad_[i] = H_eff * A_i
        apply_heff(i, d_mps_[i], d_grad_[i]);
        // 2) d_heff_result_ = N_eff * A_i
        apply_norm(i, d_mps_[i], d_heff_result_);
        // 3) d_grad_[i] = inv_xx * (d_grad_[i] - E * d_heff_result_)
        //    = inv_xx * d_grad_[i] - inv_xx*E * d_heff_result_
        int block = 256;
        int grid = (sz + block - 1) / block;
        hipLaunchKernelGGL(axpby_real_kernel<Scalar>, dim3(grid), dim3(block), 0, stream_,
                           d_grad_[i], inv_xx, d_heff_result_, -inv_xx * E, sz);
    }
    HIP_CHECK(hipStreamSynchronize(stream_));
    energy_ = E;
    return E;
}

// ============================================================================
// Tangent projection (right-canonical gauge, center at site 0)
//   For i >= 1: G_i <- G_i - (G_i @ A_i^H) @ A_i
//   Viewing G_i, A_i as (chi_L × d*chi_R) matrices.
// ============================================================================

template<typename Scalar>
void RAdamGPU<Scalar>::tangent_project_inplace() {
    Scalar one = Traits::one(), zero_v = Traits::zero(), neg_one = Traits::neg(one);

    for (int i = 1; i < L_; i++) {
        int cL = chi_L(i), cR = chi_R(i);
        int cols = d_ * cR;

        Scalar* A = d_mps_[i];      // (cL × cols), col-major
        Scalar* G = d_grad_[i];
        Scalar* T = d_T1_;          // (cL × cL)

        // T = G * A^H    ((cL × cols) * (cols × cL) → (cL × cL))
        ROCBLAS_CHECK(Traits::gemm(handle_,
            rocblas_operation_none, Traits::op_h,
            cL, cL, cols,
            &one,
            G, cL,
            A, cL,
            &zero_v,
            T, cL));

        // G = G - T * A   ((cL × cL) * (cL × cols) → (cL × cols))
        ROCBLAS_CHECK(Traits::gemm(handle_,
            rocblas_operation_none, rocblas_operation_none,
            cL, cols, cL,
            &neg_one,
            T, cL,
            A, cL,
            &one,
            G, cL));
    }
    HIP_CHECK(hipStreamSynchronize(stream_));
}

template<typename Scalar>
double RAdamGPU<Scalar>::grad_frobenius_norm_sq() {
    double total = 0.0;
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host));
    for (int i = 0; i < L_; i++) {
        int sz = chi_L(i) * d_ * chi_R(i);
        RealType n;
        ROCBLAS_CHECK(Traits::nrm2(handle_, sz, d_grad_[i], 1, &n));
        total += (double)n * (double)n;
    }
    HIP_CHECK(hipStreamSynchronize(stream_));
    return total;
}

// ============================================================================
// Adam update: reads d_grad_[i] (projected Riemannian gradient), updates d_M_[i]
//  and overwrites d_grad_[i] with the step direction Delta_i.
//  Host-side manages the scalar v and step_scale; per-site kernel does the heavy work.
// ============================================================================

template<typename Scalar>
void RAdamGPU<Scalar>::adam_update(int step, double lr, double beta1, double beta2, double eps,
                                    double& v_state, double& grad_norm_out) {
    double g_norm_sq = grad_frobenius_norm_sq();
    grad_norm_out = std::sqrt(g_norm_sq);

    // Second-moment (scalar) update + bias correction.
    v_state = beta2 * v_state + (1.0 - beta2) * g_norm_sq;
    double bc1 = 1.0 - std::pow(beta1, step);
    double bc2 = 1.0 - std::pow(beta2, step);
    double v_hat = v_state / bc2;
    double step_scale = -lr / (std::sqrt(v_hat) + eps);

    for (int i = 0; i < L_; i++) {
        int sz = chi_L(i) * d_ * chi_R(i);
        int block = 256;
        int grid = (sz + block - 1) / block;
        hipLaunchKernelGGL(adam_update_kernel<Scalar>, dim3(grid), dim3(block), 0, stream_,
                           d_M_[i], d_grad_[i],
                           beta1, 1.0 - beta1,
                           step_scale, bc1,
                           sz);
    }
    HIP_CHECK(hipStreamSynchronize(stream_));
}

// ============================================================================
// Retraction: MPS += Delta (held in d_grad_[i] after adam_update), re-canonicalize,
//  then re-project momentum to the new tangent space (vector transport = re-project).
// ============================================================================

template<typename Scalar>
void RAdamGPU<Scalar>::retract_and_recanonicalize() {
    for (int i = 0; i < L_; i++) {
        int sz = chi_L(i) * d_ * chi_R(i);
        int block = 256;
        int grid = (sz + block - 1) / block;
        hipLaunchKernelGGL(elementwise_add_kernel<Scalar>, dim3(grid), dim3(block), 0, stream_,
                           d_mps_[i], d_grad_[i], sz);
    }
    HIP_CHECK(hipStreamSynchronize(stream_));
    right_canonicalize_mps();
    normalize_site0();
}

// ============================================================================
// Lanczos eigensolver for DMRG1 warmstart (single-site)
// Simplified port of pdmrg-gpu-opt's lanczos_eigensolver, always 1-site.
// ============================================================================

template<typename Scalar>
double RAdamGPU<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta, int theta_size) {
    int n = theta_size;
    int max_iter = std::min(max_lanczos_iter_, n);
    double tol_lanczos = 1e-12;
    double tol_eig_conv = 1e-12;

    std::vector<double> h_alpha(max_iter);
    std::vector<double> h_beta(max_iter);

    double norm;
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host));
    ROCBLAS_CHECK(Traits::nrm2(handle_, n, d_theta, 1, &norm));

    if (norm < 1e-14) {
        std::vector<Scalar> h_init(n);
        srand(42 + site);
        for (int i = 0; i < n; i++) h_init[i] = Traits::random_val();
        HIP_CHECK(hipMemcpy(d_theta, h_init.data(), n * sizeof(Scalar), hipMemcpyHostToDevice));
        ROCBLAS_CHECK(Traits::nrm2(handle_, n, d_theta, 1, &norm));
    }
    RealType inv_n = (RealType)(1.0 / norm);
    ROCBLAS_CHECK(Traits::scal_real(handle_, n, &inv_n, d_theta, 1));
    HIP_CHECK(hipMemcpyAsync(d_lanczos_v_, d_theta, n * sizeof(Scalar),
                              hipMemcpyDeviceToDevice, stream_));

    double prev_energy = 1e30;
    int iter = 0;
    int niter_final = 0;

    for (iter = 0; iter < max_iter; iter++) {
        Scalar* d_vi = d_lanczos_v_ + (size_t)iter * n;
        apply_heff(site, d_vi, d_heff_result_);

        // alpha = <v_i | H v_i>
        Scalar dot_val;
        ROCBLAS_CHECK(Traits::dot(handle_, n, d_vi, 1, d_heff_result_, 1, &dot_val));
        double alpha = Traits::real_part(dot_val);
        h_alpha[iter] = alpha;

        // heff_result -= alpha * v_i
        Scalar neg_alpha = Traits::scale_by_real(-alpha, Traits::one());
        ROCBLAS_CHECK(Traits::axpy(handle_, n, &neg_alpha, d_vi, 1, d_heff_result_, 1));

        // heff_result -= beta_{i-1} * v_{i-1}
        if (iter > 0) {
            Scalar neg_beta_prev = Traits::scale_by_real(-h_beta[iter - 1], Traits::one());
            ROCBLAS_CHECK(Traits::axpy(handle_, n, &neg_beta_prev,
                d_lanczos_v_ + (size_t)(iter - 1) * n, 1,
                d_heff_result_, 1));
        }

        // Full reorthogonalization against all previous v_j
        for (int j = 0; j <= iter; j++) {
            Scalar coeff;
            ROCBLAS_CHECK(Traits::dot(handle_, n,
                d_lanczos_v_ + (size_t)j * n, 1, d_heff_result_, 1, &coeff));
            Scalar neg_coeff = Traits::neg(coeff);
            ROCBLAS_CHECK(Traits::axpy(handle_, n, &neg_coeff,
                d_lanczos_v_ + (size_t)j * n, 1, d_heff_result_, 1));
        }

        // beta_i = ||heff_result||
        RealType beta_r;
        ROCBLAS_CHECK(Traits::nrm2(handle_, n, d_heff_result_, 1, &beta_r));
        double beta = (double)beta_r;
        h_beta[iter] = beta;

        if (iter + 1 < max_iter) {
            Scalar* d_vip1 = d_lanczos_v_ + (size_t)(iter + 1) * n;
            HIP_CHECK(hipMemcpyAsync(d_vip1, d_heff_result_, n * sizeof(Scalar),
                                      hipMemcpyDeviceToDevice, stream_));
            if (beta > 1e-30) {
                RealType inv_b = (RealType)(1.0 / beta);
                ROCBLAS_CHECK(Traits::scal_real(handle_, n, &inv_b, d_vip1, 1));
            }
        }

        // Periodic tridiag check
        if ((iter >= 4 && iter % 3 == 0) || beta < tol_lanczos) {
            int ncheck = iter + 1;
            std::vector<double> h_D(ncheck), h_E(ncheck);
            std::copy(h_alpha.begin(), h_alpha.begin() + ncheck, h_D.begin());
            for (int j = 0; j < ncheck - 1; j++) h_E[j] = h_beta[j];
            h_E[ncheck - 1] = 0.0;
            const char jobz_n = 'N';
            const int n_chk = ncheck;
            std::vector<double> h_work_chk(1);
            int info_chk = 0;
            dstev_(&jobz_n, &n_chk, h_D.data(), h_E.data(), nullptr, &n_chk,
                   h_work_chk.data(), &info_chk);
            if (info_chk == 0) {
                double cur_energy = h_D[0];
                if (std::abs(cur_energy - prev_energy) < tol_eig_conv) {
                    niter_final = iter + 1;
                    break;
                }
                prev_energy = cur_energy;
            }
            if (beta < tol_lanczos) {
                niter_final = iter + 1;
                break;
            }
        }
        niter_final = iter + 1;
    }

    // Final tridiagonal eigensolve
    int niter = niter_final;
    std::vector<double> h_D(niter), h_E(niter), h_Z((size_t)niter * niter);
    std::vector<double> h_work(std::max(1, 2 * niter - 2));
    int lapack_info = 0;
    std::copy(h_alpha.begin(), h_alpha.begin() + niter, h_D.begin());
    for (int i = 0; i < niter - 1; i++) h_E[i] = h_beta[i];
    if (niter > 0) h_E[niter - 1] = 0.0;
    const char jobz = 'V';
    const int n_lapack = niter;
    const int ldz = niter;
    dstev_(&jobz, &n_lapack, h_D.data(), h_E.data(), h_Z.data(), &ldz, h_work.data(), &lapack_info);
    if (lapack_info != 0) throw std::runtime_error("LAPACK dstev failed");
    double energy = h_D[0];

    // Reconstruct ground-state Ritz vector: theta = sum_i Z[i,0] * v_i
    std::vector<Scalar> h_ritz(niter);
    for (int i = 0; i < niter; i++) h_ritz[i] = Traits::make_scalar(h_Z[i]);
    HIP_CHECK(hipMemcpyAsync(d_ritz_coeffs_, h_ritz.data(), niter * sizeof(Scalar),
                              hipMemcpyHostToDevice, stream_));
    Scalar one = Traits::one(), zero_v = Traits::zero();
    ROCBLAS_CHECK(Traits::gemv(handle_, rocblas_operation_none,
        n, niter, &one,
        d_lanczos_v_, n,
        d_ritz_coeffs_, 1,
        &zero_v, d_theta, 1));

    RealType th_norm;
    ROCBLAS_CHECK(Traits::nrm2(handle_, n, d_theta, 1, &th_norm));
    if (th_norm > 1e-30) {
        RealType inv_th = (RealType)(1.0 / th_norm);
        ROCBLAS_CHECK(Traits::scal_real(handle_, n, &inv_th, d_theta, 1));
    }
    HIP_CHECK(hipStreamSynchronize(stream_));

    return energy;
}

// ============================================================================
// SVD split (single-site, CPU path) — used by DMRG1 warmstart sweeps
// direction 'R': theta is (cL*d × cR) — site keeps U, absorb S*Vh into site+1
// direction 'L': theta is (cL × d*cR) — site keeps Vh, absorb U*S into site-1
// ============================================================================

template<typename Scalar>
void RAdamGPU<Scalar>::svd_split_single_site(int site, Scalar* d_theta, char direction) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    int m, n_svd;
    if (direction == 'R') { m = cL * d_; n_svd = cR; }
    else                  { m = cL;      n_svd = d_ * cR; }
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    // Copy theta to host
    HIP_CHECK(hipMemcpyAsync(h_svd_A_.data(), d_theta,
                              m * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));

    int lwork = (int)h_svd_work_.size();
    int info;
    const char jobu = 'S', jobvt = 'S';
    Traits::lapack_gesvd(&jobu, &jobvt, &m, &n_svd, h_svd_A_.data(), &m,
        h_svd_S_.data(), h_svd_U_.data(), &m,
        h_svd_Vh_.data(), &full_k,
        h_svd_work_.data(), &lwork,
        h_svd_rwork_.empty() ? nullptr : h_svd_rwork_.data(), &info);
    if (info != 0) throw std::runtime_error("LAPACK gesvd failed");

    // Truncate
    int new_k = k;
    for (int i = 0; i < new_k; i++) {
        if (h_svd_S_[i] < 1e-14) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;

    Scalar one = Traits::one(), zero_v = Traits::zero();

    if (direction == 'R') {
        // MPS[site] <- U (m × new_k)
        HIP_CHECK(hipMemcpyAsync(d_mps_[site], h_svd_U_.data(),
                                 (size_t)m * new_k * sizeof(Scalar),
                                 hipMemcpyHostToDevice, stream_));
        // Absorb S*Vh into MPS[site+1]
        // S*Vh has shape (new_k × n_svd)
        for (int j = 0; j < n_svd; j++)
            for (int i = 0; i < new_k; i++)
                h_svd_tmp_[i + (size_t)j * new_k] =
                    Traits::scale_by_real(h_svd_S_[i], h_svd_Vh_[i + (size_t)j * full_k]);

        HIP_CHECK(hipMemcpyAsync(d_svd_work_, h_svd_tmp_.data(),
                                 (size_t)new_k * n_svd * sizeof(Scalar),
                                 hipMemcpyHostToDevice, stream_));
        if (site + 1 < L_) {
            int next_cR = chi_R(site + 1);
            ROCBLAS_CHECK(Traits::gemm(handle_,
                rocblas_operation_none, rocblas_operation_none,
                new_k, d_ * next_cR, cR, &one,
                d_svd_work_, new_k,
                d_mps_[site + 1], cR, &zero_v,
                d_T1_, new_k));
            HIP_CHECK(hipMemcpyAsync(d_mps_[site + 1], d_T1_,
                                     (size_t)new_k * d_ * next_cR * sizeof(Scalar),
                                     hipMemcpyDeviceToDevice, stream_));
        }
        bond_dims_[site + 1] = new_k;
    } else {  // 'L'
        // MPS[site] <- Vh (new_k × n_svd) with compacted rows.
        for (int j = 0; j < n_svd; j++)
            for (int i = 0; i < new_k; i++)
                h_svd_tmp_[i + (size_t)j * new_k] = h_svd_Vh_[i + (size_t)j * full_k];
        HIP_CHECK(hipMemcpyAsync(d_mps_[site], h_svd_tmp_.data(),
                                 (size_t)new_k * n_svd * sizeof(Scalar),
                                 hipMemcpyHostToDevice, stream_));
        // U*S (m × new_k) into neighbor's right bond.
        for (int j = 0; j < new_k; j++)
            for (int i = 0; i < m; i++)
                h_svd_tmp_[i + (size_t)j * m] =
                    Traits::scale_by_real(h_svd_S_[j], h_svd_U_[i + (size_t)j * m]);
        HIP_CHECK(hipMemcpyAsync(d_svd_work_, h_svd_tmp_.data(),
                                 (size_t)m * new_k * sizeof(Scalar),
                                 hipMemcpyHostToDevice, stream_));
        if (site > 0) {
            int prev_cL = chi_L(site - 1);
            ROCBLAS_CHECK(Traits::gemm(handle_,
                rocblas_operation_none, rocblas_operation_none,
                prev_cL * d_, new_k, cL, &one,
                d_mps_[site - 1], prev_cL * d_,
                d_svd_work_, m, &zero_v,
                d_T1_, prev_cL * d_));
            HIP_CHECK(hipMemcpyAsync(d_mps_[site - 1], d_T1_,
                                     (size_t)prev_cL * d_ * new_k * sizeof(Scalar),
                                     hipMemcpyDeviceToDevice, stream_));
        }
        bond_dims_[site] = new_k;
    }
    HIP_CHECK(hipStreamSynchronize(stream_));
}

template<typename Scalar>
double RAdamGPU<Scalar>::optimize_site_single(int site, char direction) {
    int cL = chi_L(site), cR = chi_R(site);
    int theta_size = cL * d_ * cR;
    HIP_CHECK(hipMemcpyAsync(d_theta_, d_mps_[site], theta_size * sizeof(Scalar),
                              hipMemcpyDeviceToDevice, stream_));
    double energy = lanczos_eigensolver(site, d_theta_, theta_size);
    svd_split_single_site(site, d_theta_, direction);
    return energy;
}

template<typename Scalar>
double RAdamGPU<Scalar>::dmrg1_sweep_LR() {
    double energy = 0.0;
    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_site_single(site, 'R');
        update_left_H_env(site);
    }
    // Last site: optimize, no SVD split
    {
        int cL = chi_L(L_ - 1), cR = chi_R(L_ - 1);
        int theta_size = cL * d_ * cR;
        HIP_CHECK(hipMemcpyAsync(d_theta_, d_mps_[L_ - 1], theta_size * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, stream_));
        energy = lanczos_eigensolver(L_ - 1, d_theta_, theta_size);
        HIP_CHECK(hipMemcpyAsync(d_mps_[L_ - 1], d_theta_, theta_size * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, stream_));
    }
    return energy;
}

template<typename Scalar>
double RAdamGPU<Scalar>::dmrg1_sweep_RL() {
    double energy = 0.0;
    for (int site = L_ - 1; site >= 1; site--) {
        energy = optimize_site_single(site, 'L');
        update_right_H_env(site);
    }
    // First site: optimize, no SVD split
    {
        int cL = chi_L(0), cR = chi_R(0);
        int theta_size = cL * d_ * cR;
        HIP_CHECK(hipMemcpyAsync(d_theta_, d_mps_[0], theta_size * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, stream_));
        energy = lanczos_eigensolver(0, d_theta_, theta_size);
        HIP_CHECK(hipMemcpyAsync(d_mps_[0], d_theta_, theta_size * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, stream_));
    }
    return energy;
}

// ============================================================================
// Driver: optional DMRG1 warmstart, then R-Adam loop
// ============================================================================

template<typename Scalar>
double RAdamGPU<Scalar>::run(const Config& cfg) {
    // 1) Optional DMRG1 warmstart (single-site only, per project PDMRG rules)
    if (cfg.n_warmup > 0) {
        build_initial_H_envs();
        for (int w = 0; w < cfg.n_warmup; w++) {
            double e_lr = dmrg1_sweep_LR();
            double e_rl = dmrg1_sweep_RL();
            if (cfg.verbose) {
                std::cout << "[warmup " << (w + 1) << "/" << cfg.n_warmup
                          << "] E_LR=" << std::fixed << std::setprecision(12) << e_lr
                          << " E_RL=" << e_rl << std::endl;
            }
        }
        // DMRG1 sweeps leave MPS in left-canonical form; re-right-canonicalize to restore gauge.
        right_canonicalize_mps();
        normalize_site0();
    }

    // 2) R-Adam loop
    // Zero momentum at start.
    for (int i = 0; i < L_; i++) {
        int sz = chi_L(i) * d_ * chi_R(i);
        HIP_CHECK(hipMemsetAsync(d_M_[i], 0, sz * sizeof(Scalar), stream_));
    }
    double v_state = 0.0;
    double prev_energy = 1e30;

    for (int step = 1; step <= cfg.max_epochs; step++) {
        // LR schedule
        double lr = cfg.lr;
        if (cfg.cosine_lr) {
            double frac = (double)(step - 1) / std::max(1, cfg.max_epochs - 1);
            lr = 0.5 * cfg.lr * (1.0 + std::cos(M_PI * frac));
        }

        // 2a) Euclidean gradient + energy
        double E = compute_all_gradients();
        // 2b) Project to tangent space
        tangent_project_inplace();
        // 2c) Adam update: overwrites d_grad_ with Delta; updates d_M_
        double gnorm;
        adam_update(step, lr, cfg.beta1, cfg.beta2, cfg.eps, v_state, gnorm);
        // 2d) Retract + re-canonicalize + re-normalize
        retract_and_recanonicalize();

        // 2e) Vector transport of momentum: re-project onto new tangent space.
        // Temporarily view d_M_ as "grad", project, and swap back.
        std::swap(d_grad_, d_M_);
        tangent_project_inplace();
        std::swap(d_grad_, d_M_);

        n_epochs_done_ = step;

        if (cfg.verbose && (step % cfg.log_every == 0 || step == 1)) {
            std::cout << "[epoch " << step << "/" << cfg.max_epochs << "] "
                      << "E=" << std::fixed << std::setprecision(12) << E
                      << "  |g|=" << std::scientific << std::setprecision(3) << gnorm
                      << "  lr=" << std::setprecision(3) << lr
                      << std::endl;
        }

        // Convergence checks
        if (gnorm < cfg.grad_tol) {
            if (cfg.verbose) std::cout << "[converged at step " << step
                                        << "] |g| < " << cfg.grad_tol << std::endl;
            break;
        }
        if (std::abs(E - prev_energy) < cfg.energy_tol && step > 2) {
            if (cfg.verbose) std::cout << "[converged at step " << step
                                        << "] |ΔE| < " << cfg.energy_tol << std::endl;
            break;
        }
        prev_energy = E;
    }

    // Final energy re-measure (in case we exited mid-loop without refreshed envs).
    build_initial_H_envs();
    build_all_L_N_envs();
    energy_ = compute_energy_from_envs();
    return energy_;
}

template<typename Scalar>
void RAdamGPU<Scalar>::get_mps(std::vector<std::vector<Scalar>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int sz = bond_dims_[i] * d_ * bond_dims_[i + 1];
        h_mps[i].resize(sz);
        HIP_CHECK(hipMemcpy(h_mps[i].data(), d_mps_[i], sz * sizeof(Scalar),
                            hipMemcpyDeviceToHost));
    }
}

#endif // RADAM_GPU_IMPL_H
