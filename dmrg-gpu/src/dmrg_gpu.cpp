#include "dmrg_gpu.h"
#include "accurate_svd_gpu.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <fstream>

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
            std::cerr << "rocBLAS error in " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("rocBLAS error"); \
        } \
    } while(0)

// LAPACK dstev
extern "C" void dstev_(const char* jobz, const int* n, double* d, double* e,
                       double* z, const int* ldz, double* work, int* info);


DMRGGPU::DMRGGPU(int L, int d, int chi_max, int D_mpo, double tol)
    : L_(L), d_(d), chi_max_(chi_max), D_mpo_(D_mpo), tol_(tol), energy_(0.0) {

    // Initialize bond dimensions using a single array
    // bond_dims_[i] = dimension of bond between site i-1 and site i
    bond_dims_.resize(L + 1);
    bond_dims_[0] = 1;
    bond_dims_[L] = 1;
    for (int i = 1; i < L; i++) {
        bond_dims_[i] = std::min(chi_max_, (int)pow(d_, std::min(i, L - i)));
    }

    // Allocate GPU resources
    HIP_CHECK(hipStreamCreate(&stream_));
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_h_));
    ROCBLAS_CHECK(rocblas_set_stream(rocblas_h_, stream_));

    svd_ = new AccurateSVD_GPU(1e-14, 0);  // No recursion needed

    // Allocate MPS tensors
    d_mps_tensors_.resize(L, nullptr);
    for (int i = 0; i < L; i++) {
        allocate_mps_tensor(i, chi_L(i), chi_R(i));
    }

    // Allocate MPO tensors (will be set later)
    d_mpo_tensors_.resize(L, nullptr);

    // Allocate environments with chi_max for non-boundary sites
    // This avoids reallocation when bond dimensions change during sweeps
    d_L_envs_.resize(L + 1, nullptr);
    d_R_envs_.resize(L + 1, nullptr);
    L_env_alloc_chi_.resize(L + 1, 0);
    R_env_alloc_chi_.resize(L + 1, 0);

    for (int i = 0; i <= L; i++) {
        // L_env[i] has shape (chi_i, D_mpo, chi_i) where chi_i = bond_dims_[i]
        // R_env[i] has shape (chi_i, D_mpo, chi_i) where chi_i = bond_dims_[i]
        // Use chi_max for interior bonds to avoid reallocation
        int chi_alloc;
        if (i == 0 || i == L) {
            chi_alloc = 1;
        } else {
            chi_alloc = chi_max_;
        }
        int sz = chi_alloc * D_mpo_ * chi_alloc;
        HIP_CHECK(hipMalloc(&d_L_envs_[i], sz * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_R_envs_[i], sz * sizeof(double)));
        HIP_CHECK(hipMemset(d_L_envs_[i], 0, sz * sizeof(double)));
        HIP_CHECK(hipMemset(d_R_envs_[i], 0, sz * sizeof(double)));
        L_env_alloc_chi_[i] = chi_alloc;
        R_env_alloc_chi_[i] = chi_alloc;
    }

    // Allocate workspace
    theta_size_max_ = chi_max_ * d_ * chi_max_;
    HIP_CHECK(hipMalloc(&d_theta_, theta_size_max_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_heff_result_, theta_size_max_ * sizeof(double)));
}

DMRGGPU::~DMRGGPU() {
    free_gpu_resources();
}

void DMRGGPU::free_gpu_resources() {
    for (auto ptr : d_mps_tensors_) if (ptr) HIP_CHECK(hipFree(ptr));
    for (auto ptr : d_mpo_tensors_) if (ptr) HIP_CHECK(hipFree(ptr));
    for (auto ptr : d_L_envs_) if (ptr) HIP_CHECK(hipFree(ptr));
    for (auto ptr : d_R_envs_) if (ptr) HIP_CHECK(hipFree(ptr));
    if (d_theta_) HIP_CHECK(hipFree(d_theta_));
    if (d_heff_result_) HIP_CHECK(hipFree(d_heff_result_));
    if (svd_) delete svd_;
    ROCBLAS_CHECK(rocblas_destroy_handle(rocblas_h_));
    HIP_CHECK(hipStreamDestroy(stream_));
}

void DMRGGPU::allocate_mps_tensor(int site, int cL, int cR) {
    if (d_mps_tensors_[site]) HIP_CHECK(hipFree(d_mps_tensors_[site]));
    HIP_CHECK(hipMalloc(&d_mps_tensors_[site], cL * d_ * cR * sizeof(double)));
}

void DMRGGPU::ensure_L_env_alloc(int idx, int chi) {
    if (chi > L_env_alloc_chi_[idx]) {
        if (d_L_envs_[idx]) HIP_CHECK(hipFree(d_L_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_L_envs_[idx], sz * sizeof(double)));
        L_env_alloc_chi_[idx] = chi;
    }
}

void DMRGGPU::ensure_R_env_alloc(int idx, int chi) {
    if (chi > R_env_alloc_chi_[idx]) {
        if (d_R_envs_[idx]) HIP_CHECK(hipFree(d_R_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_R_envs_[idx], sz * sizeof(double)));
        R_env_alloc_chi_[idx] = chi;
    }
}

void DMRGGPU::initialize_mps_random(double scale) {
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        std::vector<double> h_A(size);
        for (int j = 0; j < size; j++) {
            h_A[j] = scale * (2.0 * rand() / RAND_MAX - 1.0);
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(double), hipMemcpyHostToDevice));
    }
}

void DMRGGPU::initialize_mps_product() {
    for (int i = 0; i < L_; i++) {
        int cL = chi_L(i), cR = chi_R(i);
        int size = cL * d_ * cR;
        std::vector<double> h_A(size, 0.0);
        // A[a, s=0, b] = delta(a,b) for spin up
        int chi_min = std::min(cL, cR);
        for (int a = 0; a < chi_min; a++) {
            h_A[a + 0*cL + a*cL*d_] = 1.0;
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(double), hipMemcpyHostToDevice));
    }
}

void DMRGGPU::initialize_mps_neel() {
    for (int i = 0; i < L_; i++) {
        int cL = chi_L(i), cR = chi_R(i);
        int size = cL * d_ * cR;
        std::vector<double> h_A(size, 0.0);
        int spin = (i % 2 == 0) ? 0 : 1;  // Alternating
        int chi_min = std::min(cL, cR);
        for (int a = 0; a < chi_min; a++) {
            h_A[a + spin*cL + a*cL*d_] = 1.0;
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(double), hipMemcpyHostToDevice));
    }
}

void DMRGGPU::set_mpo(const std::vector<double*>& h_mpo_tensors) {
    for (int i = 0; i < L_; i++) {
        int size = D_mpo_ * d_ * d_ * D_mpo_;
        HIP_CHECK(hipMalloc(&d_mpo_tensors_[i], size * sizeof(double)));
        HIP_CHECK(hipMemcpy(d_mpo_tensors_[i], h_mpo_tensors[i],
                            size * sizeof(double), hipMemcpyHostToDevice));
    }
}

// ============================================================================
// Environment building (CPU contractions - not performance critical)
// ============================================================================

void DMRGGPU::build_initial_environments() {
    // L[0] = trivial left boundary: shape (1, D_mpo, 1), L[0][0, 0, 0] = 1
    {
        std::vector<double> h_L(D_mpo_, 0.0);
        h_L[0] = 1.0;
        HIP_CHECK(hipMemcpy(d_L_envs_[0], h_L.data(),
                            D_mpo_ * sizeof(double), hipMemcpyHostToDevice));
    }

    // R[L] = trivial right boundary: shape (1, D_mpo, 1), R[L][0, D-1, 0] = 1
    {
        std::vector<double> h_R(D_mpo_, 0.0);
        h_R[D_mpo_ - 1] = 1.0;
        HIP_CHECK(hipMemcpy(d_R_envs_[L_], h_R.data(),
                            D_mpo_ * sizeof(double), hipMemcpyHostToDevice));
    }

    // Build all R environments from right to left
    for (int i = L_ - 1; i >= 0; i--) {
        update_right_env(i);
    }
}

void DMRGGPU::update_left_env(int site) {
    // L[site+1][b, w', b'] = sum_{a,w,s,s'} L[site][a,w,a'] * A[site][a,s,b] * W[site][w,s,s',w'] * A*[site][a',s',b']
    // where A* = A (real)
    int cL = chi_L(site);
    int cR = chi_R(site);

    // L_env[site] has chi = bond_dims_[site] = cL
    int chi_in = bond_dims_[site];
    // L_env[site+1] will have chi = bond_dims_[site+1] = cR
    int chi_out = bond_dims_[site + 1];

    ensure_L_env_alloc(site + 1, chi_out);

    int n_L = chi_in * D_mpo_ * chi_in;
    int n_A = cL * d_ * cR;
    int n_W = D_mpo_ * d_ * d_ * D_mpo_;
    int n_L_out = chi_out * D_mpo_ * chi_out;

    std::vector<double> h_L(n_L), h_A(n_A), h_W(n_W), h_L_out(n_L_out, 0.0);

    HIP_CHECK(hipMemcpy(h_L.data(), d_L_envs_[site], n_L * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_A.data(), d_mps_tensors_[site], n_A * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_W.data(), d_mpo_tensors_[site], n_W * sizeof(double), hipMemcpyDeviceToHost));

    // L_out[b, w', b'] = sum_{a,w,a',s,s'} L[a,w,a'] * A[a,s,b] * W[w,s,s',w'] * A*[a',s',b']
    // Layout: L[a + w*chi_in + ap*chi_in*D], A[a + s*cL + b*cL*d], W[w + s*D + sp*D*d + wp*D*d*d]

    // Step 1: T1[w, a', s, b] = sum_a L[a, w, a'] * A[a, s, b]
    std::vector<double> T1(D_mpo_ * chi_in * d_ * cR, 0.0);
    for (int w = 0; w < D_mpo_; w++) {
        for (int ap = 0; ap < chi_in; ap++) {
            for (int s = 0; s < d_; s++) {
                for (int b = 0; b < cR; b++) {
                    double sum = 0.0;
                    for (int a = 0; a < chi_in; a++) {
                        sum += h_L[a + w*chi_in + ap*chi_in*D_mpo_] *
                               h_A[a + s*cL + b*cL*d_];
                    }
                    T1[w + ap*D_mpo_ + s*D_mpo_*chi_in + b*D_mpo_*chi_in*d_] = sum;
                }
            }
        }
    }

    // Step 2: T2[a', sp, w', b] = sum_{w,s} W[w, s, sp, w'] * T1[w, a', s, b]
    std::vector<double> T2(chi_in * d_ * D_mpo_ * cR, 0.0);
    for (int ap = 0; ap < chi_in; ap++) {
        for (int sp = 0; sp < d_; sp++) {
            for (int wp = 0; wp < D_mpo_; wp++) {
                for (int b = 0; b < cR; b++) {
                    double sum = 0.0;
                    for (int w = 0; w < D_mpo_; w++) {
                        for (int s = 0; s < d_; s++) {
                            sum += h_W[w + s*D_mpo_ + sp*D_mpo_*d_ + wp*D_mpo_*d_*d_] *
                                   T1[w + ap*D_mpo_ + s*D_mpo_*chi_in + b*D_mpo_*chi_in*d_];
                        }
                    }
                    T2[ap + sp*chi_in + wp*chi_in*d_ + b*chi_in*d_*D_mpo_] = sum;
                }
            }
        }
    }

    // Step 3: L_out[b, w', b'] = sum_{a',s'} A*[a', s', b'] * T2[a', s', w', b]
    for (int b = 0; b < chi_out; b++) {
        for (int wp = 0; wp < D_mpo_; wp++) {
            for (int bp = 0; bp < chi_out; bp++) {
                double sum = 0.0;
                for (int ap = 0; ap < chi_in; ap++) {
                    for (int sp = 0; sp < d_; sp++) {
                        sum += h_A[ap + sp*cL + bp*cL*d_] *
                               T2[ap + sp*chi_in + wp*chi_in*d_ + b*chi_in*d_*D_mpo_];
                    }
                }
                h_L_out[b + wp*chi_out + bp*chi_out*D_mpo_] = sum;
            }
        }
    }

    HIP_CHECK(hipMemcpy(d_L_envs_[site + 1], h_L_out.data(), n_L_out * sizeof(double), hipMemcpyHostToDevice));
}

void DMRGGPU::update_right_env(int site) {
    // R[site][a, w, a'] = sum_{b,w',s,s'} A[site][a,s,b] * W[site][w,s,s',w'] * A*[site][a',s',b'] * R[site+1][b,w',b']
    int cL = chi_L(site);
    int cR = chi_R(site);

    // R_env[site+1] has chi = bond_dims_[site+1] = cR
    int chi_in = bond_dims_[site + 1];
    // R_env[site] will have chi = bond_dims_[site] = cL
    int chi_out = bond_dims_[site];

    ensure_R_env_alloc(site, chi_out);

    int n_R = chi_in * D_mpo_ * chi_in;
    int n_A = cL * d_ * cR;
    int n_W = D_mpo_ * d_ * d_ * D_mpo_;
    int n_R_out = chi_out * D_mpo_ * chi_out;

    std::vector<double> h_R(n_R), h_A(n_A), h_W(n_W), h_R_out(n_R_out, 0.0);

    HIP_CHECK(hipMemcpy(h_R.data(), d_R_envs_[site + 1], n_R * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_A.data(), d_mps_tensors_[site], n_A * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_W.data(), d_mpo_tensors_[site], n_W * sizeof(double), hipMemcpyDeviceToHost));

    // Step 1: T1[a, s, w', b'] = sum_b A[a,s,b] * R[b,w',b']
    // Note: We sum over bra index b' of R, but the A that connects is through index b
    // Wait - let me be more careful. R[site+1] has shape (chi_R, D_mpo, chi_R).
    // R[b, w', b'] where b is ket-right, b' is bra-right.
    // A[site] has shape (cL, d, cR) = A[a, s, b].
    // We need: sum_b A[a,s,b] * R[b, w', b'] -> T1[a, s, w', b']
    std::vector<double> T1(cL * d_ * D_mpo_ * chi_in, 0.0);
    for (int a = 0; a < cL; a++) {
        for (int s = 0; s < d_; s++) {
            for (int wp = 0; wp < D_mpo_; wp++) {
                for (int bp = 0; bp < chi_in; bp++) {
                    double sum = 0.0;
                    for (int b = 0; b < cR; b++) {
                        sum += h_A[a + s*cL + b*cL*d_] *
                               h_R[b + wp*chi_in + bp*chi_in*D_mpo_];
                    }
                    T1[a + s*cL + wp*cL*d_ + bp*cL*d_*D_mpo_] = sum;
                }
            }
        }
    }

    // Step 2: T2[a, sp, w, b'] = sum_{w',s} W[w, s, sp, w'] * T1[a, s, w', b']
    std::vector<double> T2(cL * d_ * D_mpo_ * chi_in, 0.0);
    for (int a = 0; a < cL; a++) {
        for (int sp = 0; sp < d_; sp++) {
            for (int w = 0; w < D_mpo_; w++) {
                for (int bp = 0; bp < chi_in; bp++) {
                    double sum = 0.0;
                    for (int wp = 0; wp < D_mpo_; wp++) {
                        for (int s = 0; s < d_; s++) {
                            sum += h_W[w + s*D_mpo_ + sp*D_mpo_*d_ + wp*D_mpo_*d_*d_] *
                                   T1[a + s*cL + wp*cL*d_ + bp*cL*d_*D_mpo_];
                        }
                    }
                    T2[a + sp*cL + w*cL*d_ + bp*cL*d_*D_mpo_] = sum;
                }
            }
        }
    }

    // Step 3: R_out[a, w, a'] = sum_{sp,b'} A*[a', sp, b'] * T2[a, sp, w, b']
    // A*[a', sp, b'] = A[a' + sp*cL + b'*cL*d] (real case)
    for (int a = 0; a < chi_out; a++) {
        for (int w = 0; w < D_mpo_; w++) {
            for (int ap = 0; ap < chi_out; ap++) {
                double sum = 0.0;
                for (int sp = 0; sp < d_; sp++) {
                    for (int bp = 0; bp < chi_in; bp++) {
                        sum += h_A[ap + sp*cL + bp*cL*d_] *
                               T2[a + sp*cL + w*cL*d_ + bp*cL*d_*D_mpo_];
                    }
                }
                h_R_out[a + w*chi_out + ap*chi_out*D_mpo_] = sum;
            }
        }
    }

    HIP_CHECK(hipMemcpy(d_R_envs_[site], h_R_out.data(), n_R_out * sizeof(double), hipMemcpyHostToDevice));
}

// ============================================================================
// H_eff application (CPU contraction for correctness)
// ============================================================================

void DMRGGPU::form_theta(int site, double* d_theta) {
    int size = chi_L(site) * d_ * chi_R(site);
    HIP_CHECK(hipMemcpy(d_theta, d_mps_tensors_[site],
                        size * sizeof(double), hipMemcpyDeviceToDevice));
}

void DMRGGPU::apply_heff(int site, const double* d_theta_in, double* d_result) {
    // H_eff|theta> = sum_{a,w,a',s,w',b,b'} L[a,w,a'] * theta[a,s,b] * W[w,s,s',w'] * R[b,w',b'] -> result[a',s',b']
    int cL = chi_L(site);
    int cR = chi_R(site);

    int n = cL * d_ * cR;
    std::vector<double> h_theta(n), h_L(cL * D_mpo_ * cL),
                        h_W(D_mpo_ * d_ * d_ * D_mpo_), h_R(cR * D_mpo_ * cR),
                        h_result(n, 0.0);

    HIP_CHECK(hipMemcpy(h_theta.data(), d_theta_in, n * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_L.data(), d_L_envs_[site], cL * D_mpo_ * cL * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_W.data(), d_mpo_tensors_[site], D_mpo_ * d_ * d_ * D_mpo_ * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_R.data(), d_R_envs_[site + 1], cR * D_mpo_ * cR * sizeof(double), hipMemcpyDeviceToHost));

    // Step 1: T1[w, a', s, b] = sum_a L[a, w, a'] * theta[a, s, b]
    std::vector<double> T1(D_mpo_ * cL * d_ * cR, 0.0);
    for (int w = 0; w < D_mpo_; w++) {
        for (int ap = 0; ap < cL; ap++) {
            for (int s = 0; s < d_; s++) {
                for (int b = 0; b < cR; b++) {
                    double sum = 0.0;
                    for (int a = 0; a < cL; a++) {
                        sum += h_L[a + w*cL + ap*cL*D_mpo_] *
                               h_theta[a + s*cL + b*cL*d_];
                    }
                    T1[w + ap*D_mpo_ + s*D_mpo_*cL + b*D_mpo_*cL*d_] = sum;
                }
            }
        }
    }

    // Step 2: T2[a', sp, w', b] = sum_{w,s} W[w, s, sp, w'] * T1[w, a', s, b]
    std::vector<double> T2(cL * d_ * D_mpo_ * cR, 0.0);
    for (int ap = 0; ap < cL; ap++) {
        for (int sp = 0; sp < d_; sp++) {
            for (int wp = 0; wp < D_mpo_; wp++) {
                for (int b = 0; b < cR; b++) {
                    double sum = 0.0;
                    for (int w = 0; w < D_mpo_; w++) {
                        for (int s = 0; s < d_; s++) {
                            sum += h_W[w + s*D_mpo_ + sp*D_mpo_*d_ + wp*D_mpo_*d_*d_] *
                                   T1[w + ap*D_mpo_ + s*D_mpo_*cL + b*D_mpo_*cL*d_];
                        }
                    }
                    T2[ap + sp*cL + wp*cL*d_ + b*cL*d_*D_mpo_] = sum;
                }
            }
        }
    }

    // Step 3: result[a', sp, b'] = sum_{w',b} R[b, w', b'] * T2[a', sp, w', b]
    for (int ap = 0; ap < cL; ap++) {
        for (int sp = 0; sp < d_; sp++) {
            for (int bp = 0; bp < cR; bp++) {
                double sum = 0.0;
                for (int wp = 0; wp < D_mpo_; wp++) {
                    for (int b = 0; b < cR; b++) {
                        sum += h_R[b + wp*cR + bp*cR*D_mpo_] *
                               T2[ap + sp*cL + wp*cL*d_ + b*cL*d_*D_mpo_];
                    }
                }
                h_result[ap + sp*cL + bp*cL*d_] = sum;
            }
        }
    }

    HIP_CHECK(hipMemcpy(d_result, h_result.data(), n * sizeof(double), hipMemcpyHostToDevice));
}

// ============================================================================
// Lanczos eigensolver with FULL reorthogonalization
// ============================================================================

double DMRGGPU::lanczos_eigensolver(int site, double* d_theta) {
    int n = chi_L(site) * d_ * chi_R(site);
    int max_iter = std::min(100, n);
    double tol_lanczos = 1e-12;

    // Allocate Lanczos vectors on GPU
    double* d_lanczos_v;
    HIP_CHECK(hipMalloc(&d_lanczos_v, max_iter * n * sizeof(double)));

    std::vector<double> h_alpha(max_iter);
    std::vector<double> h_beta(max_iter);

    // v[0] = theta / ||theta||
    double norm;
    ROCBLAS_CHECK(rocblas_dnrm2(rocblas_h_, n, d_theta, 1, &norm));

    if (norm < 1e-14) {
        std::vector<double> h_init(n);
        srand(42 + site);
        for (int i = 0; i < n; i++) h_init[i] = 2.0 * rand() / RAND_MAX - 1.0;
        HIP_CHECK(hipMemcpy(d_theta, h_init.data(), n * sizeof(double), hipMemcpyHostToDevice));
        ROCBLAS_CHECK(rocblas_dnrm2(rocblas_h_, n, d_theta, 1, &norm));
    }

    double inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(rocblas_dscal(rocblas_h_, n, &inv_norm, d_theta, 1));
    HIP_CHECK(hipMemcpy(d_lanczos_v, d_theta, n * sizeof(double), hipMemcpyDeviceToDevice));

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        double* d_vi = d_lanczos_v + iter * n;

        // w = H|v_i>
        apply_heff(site, d_vi, d_heff_result_);

        // alpha_i = <v_i|w>
        double alpha_i;
        ROCBLAS_CHECK(rocblas_ddot(rocblas_h_, n, d_vi, 1, d_heff_result_, 1, &alpha_i));
        h_alpha[iter] = alpha_i;

        // w = w - alpha_i * v_i
        double neg_alpha = -alpha_i;
        ROCBLAS_CHECK(rocblas_daxpy(rocblas_h_, n, &neg_alpha, d_vi, 1, d_heff_result_, 1));

        // w = w - beta_{i-1} * v_{i-1}
        if (iter > 0) {
            double neg_beta = -h_beta[iter - 1];
            double* d_vim1 = d_lanczos_v + (iter - 1) * n;
            ROCBLAS_CHECK(rocblas_daxpy(rocblas_h_, n, &neg_beta, d_vim1, 1, d_heff_result_, 1));
        }

        // FULL REORTHOGONALIZATION: w = w - sum_j <v_j|w> v_j
        for (int j = 0; j <= iter; j++) {
            double* d_vj = d_lanczos_v + j * n;
            double overlap;
            ROCBLAS_CHECK(rocblas_ddot(rocblas_h_, n, d_vj, 1, d_heff_result_, 1, &overlap));
            double neg_overlap = -overlap;
            ROCBLAS_CHECK(rocblas_daxpy(rocblas_h_, n, &neg_overlap, d_vj, 1, d_heff_result_, 1));
        }

        // beta_i = ||w||
        double beta_i;
        ROCBLAS_CHECK(rocblas_dnrm2(rocblas_h_, n, d_heff_result_, 1, &beta_i));
        h_beta[iter] = beta_i;

        if (beta_i < tol_lanczos) {
            iter++;
            break;
        }

        // v_{i+1} = w / beta_i
        if (iter + 1 < max_iter) {
            double* d_vip1 = d_lanczos_v + (iter + 1) * n;
            double scale = 1.0 / beta_i;
            HIP_CHECK(hipMemcpy(d_vip1, d_heff_result_, n * sizeof(double), hipMemcpyDeviceToDevice));
            ROCBLAS_CHECK(rocblas_dscal(rocblas_h_, n, &scale, d_vip1, 1));
        }
    }

    int niter = iter;

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
    double* d_ritz_coeffs;
    HIP_CHECK(hipMalloc(&d_ritz_coeffs, niter * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_ritz_coeffs, h_Z.data(), niter * sizeof(double), hipMemcpyHostToDevice));

    const double one = 1.0, zero = 0.0;
    ROCBLAS_CHECK(rocblas_dgemv(
        rocblas_h_,
        rocblas_operation_none,
        n, niter,
        &one,
        d_lanczos_v, n,
        d_ritz_coeffs, 1,
        &zero,
        d_theta, 1
    ));

    // Normalize
    ROCBLAS_CHECK(rocblas_dnrm2(rocblas_h_, n, d_theta, 1, &norm));
    inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(rocblas_dscal(rocblas_h_, n, &inv_norm, d_theta, 1));

    HIP_CHECK(hipFree(d_ritz_coeffs));
    HIP_CHECK(hipFree(d_lanczos_v));

    return energy;
}

// ============================================================================
// SVD and MPS update WITH absorption into neighbor
// ============================================================================

void DMRGGPU::svd_and_update_mps(int site, double* d_theta, char direction) {
    int cL = chi_L(site);
    int cR = chi_R(site);

    if (direction == 'R') {
        // Moving right: theta[a,s,b] reshaped to M(cL*d, cR) -> U S Vh
        // Store U -> A[site], absorb S*Vh -> A[site+1]
        int m = cL * d_;
        int n_svd = cR;
        int k = std::min(m, n_svd);
        k = std::min(k, chi_max_);

        auto result = svd_->decompose(d_theta, m, n_svd);

        // Read S values to host
        std::vector<double> h_S(result.rank);
        HIP_CHECK(hipMemcpy(h_S.data(), result.d_S, result.rank * sizeof(double), hipMemcpyDeviceToHost));


        // Determine how many singular values to keep
        int new_k = std::min(k, result.rank);
        // Truncate small singular values
        for (int i = 0; i < new_k; i++) {
            if (h_S[i] < 1e-14) {
                new_k = i;
                break;
            }
        }
        if (new_k == 0) new_k = 1;

        int new_chi_R = new_k;

        // Update A[site] = U[:, :new_k] -> shape (cL, d, new_k)
        // U is column-major (m, result.rank), we need first new_k columns
        allocate_mps_tensor(site, cL, new_chi_R);
        // U in column-major: column j starts at j*m. We need m*new_k elements.
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], result.d_U,
                            m * new_chi_R * sizeof(double), hipMemcpyDeviceToDevice));

        // Compute S * Vh on CPU: S_Vh[i, j] = S[i] * Vh[i, j]
        // Vh is column-major (result.rank, n_svd), lda=result.rank
        // We need first new_k rows -> S_Vh(new_k, n_svd)
        std::vector<double> h_Vh(result.rank * n_svd);
        HIP_CHECK(hipMemcpy(h_Vh.data(), result.d_Vh,
                            result.rank * n_svd * sizeof(double), hipMemcpyDeviceToHost));

        std::vector<double> h_SVh(new_chi_R * n_svd);
        for (int j = 0; j < n_svd; j++) {
            for (int i = 0; i < new_chi_R; i++) {
                h_SVh[i + j * new_chi_R] = h_S[i] * h_Vh[i + j * result.rank];
            }
        }

        // Absorb S*Vh into A[site+1]
        // A_next has shape (chi_L_next=cR, d, chi_R_next)
        // New A_next = (S*Vh) @ A_next reshaped:
        //   (S*Vh)(new_k, cR) @ A_next(cR, d*chi_R_next) -> new_A_next(new_k, d*chi_R_next)
        if (site + 1 < L_) {
            int next_cR = chi_R(site + 1);
            int next_size = cR * d_ * next_cR;
            std::vector<double> h_A_next(next_size);
            HIP_CHECK(hipMemcpy(h_A_next.data(), d_mps_tensors_[site + 1],
                                next_size * sizeof(double), hipMemcpyDeviceToHost));

            // Matrix multiply: new_A_next = S_Vh @ A_next
            // S_Vh is (new_chi_R, cR) column-major
            // A_next is (cR, d*next_cR) column-major
            // Result is (new_chi_R, d*next_cR) column-major
            int M = new_chi_R;
            int N = d_ * next_cR;
            int K = cR;

            std::vector<double> h_new_A_next(M * N, 0.0);
            for (int jj = 0; jj < N; jj++) {
                for (int ii = 0; ii < M; ii++) {
                    double sum = 0.0;
                    for (int kk = 0; kk < K; kk++) {
                        sum += h_SVh[ii + kk * M] * h_A_next[kk + jj * K];
                    }
                    h_new_A_next[ii + jj * M] = sum;
                }
            }

            // Reallocate and write new A[site+1]
            allocate_mps_tensor(site + 1, new_chi_R, next_cR);
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site + 1], h_new_A_next.data(),
                                M * N * sizeof(double), hipMemcpyHostToDevice));
        }

        // Update bond dimensions
        bond_dims_[site + 1] = new_chi_R;

    } else {  // direction == 'L'
        // Moving left: theta[a,s,b] reshaped to M(cL, d*cR) -> U S Vh
        // Absorb U*S -> A[site-1], store Vh -> A[site]
        int m = cL;
        int n_svd = d_ * cR;
        int k = std::min(m, n_svd);
        k = std::min(k, chi_max_);

        auto result = svd_->decompose(d_theta, m, n_svd);

        std::vector<double> h_S(result.rank);
        HIP_CHECK(hipMemcpy(h_S.data(), result.d_S, result.rank * sizeof(double), hipMemcpyDeviceToHost));


        int new_k = std::min(k, result.rank);
        for (int i = 0; i < new_k; i++) {
            if (h_S[i] < 1e-14) {
                new_k = i;
                break;
            }
        }
        if (new_k == 0) new_k = 1;

        int new_chi_L = new_k;

        // Update A[site] = Vh[:new_k, :] -> shape (new_k, d, cR)
        // Vh is column-major (result.rank, n_svd), lda=result.rank
        // We need first new_k rows of Vh
        // In column-major storage, row i of column j is at Vh[i + j*lda]
        // We need to extract a submatrix: rows 0..new_k-1 of Vh(result.rank, n_svd)
        if (new_chi_L == result.rank) {
            // No truncation needed, direct copy
            allocate_mps_tensor(site, new_chi_L, cR);
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site], result.d_Vh,
                                new_chi_L * n_svd * sizeof(double), hipMemcpyDeviceToDevice));
        } else {
            // Need to extract first new_k rows from column-major Vh
            std::vector<double> h_Vh(result.rank * n_svd);
            HIP_CHECK(hipMemcpy(h_Vh.data(), result.d_Vh,
                                result.rank * n_svd * sizeof(double), hipMemcpyDeviceToHost));

            std::vector<double> h_Vh_trunc(new_chi_L * n_svd);
            for (int j = 0; j < n_svd; j++) {
                for (int i = 0; i < new_chi_L; i++) {
                    h_Vh_trunc[i + j * new_chi_L] = h_Vh[i + j * result.rank];
                }
            }
            allocate_mps_tensor(site, new_chi_L, cR);
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_Vh_trunc.data(),
                                new_chi_L * n_svd * sizeof(double), hipMemcpyHostToDevice));
        }

        // Compute U * S: U_S[i, j] = U[i, j] * S[j]
        // U is column-major (m, result.rank), lda=m
        // We need first new_k columns
        std::vector<double> h_U(m * result.rank);
        HIP_CHECK(hipMemcpy(h_U.data(), result.d_U,
                            m * result.rank * sizeof(double), hipMemcpyDeviceToHost));

        std::vector<double> h_US(m * new_chi_L);
        for (int j = 0; j < new_chi_L; j++) {
            for (int i = 0; i < m; i++) {
                h_US[i + j * m] = h_U[i + j * m] * h_S[j];
            }
        }

        // Absorb U*S into A[site-1]
        // A_prev has shape (chi_L_prev, d, cL) = (chi_L_prev, d, old_cL)
        // New A_prev = A_prev @ (U*S):
        //   A_prev reshaped (chi_L_prev*d, cL) @ U_S(cL, new_k) -> new_A_prev(chi_L_prev*d, new_k)
        if (site > 0) {
            int prev_cL = chi_L(site - 1);
            int prev_size = prev_cL * d_ * cL;
            std::vector<double> h_A_prev(prev_size);
            HIP_CHECK(hipMemcpy(h_A_prev.data(), d_mps_tensors_[site - 1],
                                prev_size * sizeof(double), hipMemcpyDeviceToHost));

            // A_prev is (prev_cL, d, cL) column-major = (prev_cL*d, cL) matrix
            int M = prev_cL * d_;
            int N = new_chi_L;
            int K = cL;  // = m

            std::vector<double> h_new_A_prev(M * N, 0.0);
            for (int jj = 0; jj < N; jj++) {
                for (int ii = 0; ii < M; ii++) {
                    double sum = 0.0;
                    for (int kk = 0; kk < K; kk++) {
                        sum += h_A_prev[ii + kk * M] * h_US[kk + jj * K];
                    }
                    h_new_A_prev[ii + jj * M] = sum;
                }
            }

            allocate_mps_tensor(site - 1, prev_cL, new_chi_L);
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site - 1], h_new_A_prev.data(),
                                M * N * sizeof(double), hipMemcpyHostToDevice));
        }

        // Update bond dimensions
        bond_dims_[site] = new_chi_L;
    }
}

// ============================================================================
// Site optimization
// ============================================================================

double DMRGGPU::optimize_site(int site, char direction) {
    form_theta(site, d_theta_);
    double energy = lanczos_eigensolver(site, d_theta_);
    svd_and_update_mps(site, d_theta_, direction);
    return energy;
}

// ============================================================================
// Sweep methods
// ============================================================================

double DMRGGPU::sweep_left_to_right() {
    double energy = 0.0;

    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_site(site, 'R');
        update_left_env(site);
    }
    // Optimize last site without SVD (just eigensolve)
    {
        int site = L_ - 1;
        form_theta(site, d_theta_);
        energy = lanczos_eigensolver(site, d_theta_);
        // Copy optimized theta back to MPS
        int sz = chi_L(site) * d_ * chi_R(site);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], d_theta_, sz * sizeof(double), hipMemcpyDeviceToDevice));
    }

    return energy;
}

double DMRGGPU::sweep_right_to_left() {
    double energy = 0.0;

    for (int site = L_ - 1; site >= 1; site--) {
        energy = optimize_site(site, 'L');
        update_right_env(site);
    }
    // Optimize first site without SVD
    {
        int site = 0;
        form_theta(site, d_theta_);
        energy = lanczos_eigensolver(site, d_theta_);
        int sz = chi_L(site) * d_ * chi_R(site);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], d_theta_, sz * sizeof(double), hipMemcpyDeviceToDevice));
    }

    return energy;
}

// ============================================================================
// Main algorithm
// ============================================================================

double DMRGGPU::run(int n_sweeps) {
    printf("=== Reference DMRG-GPU ===\n");
    printf("L = %d, d = %d, chi_max = %d, D_mpo = %d\n", L_, d_, chi_max_, D_mpo_);
    printf("Running %d sweeps...\n\n", n_sweeps);

    printf("Building initial environments...\n");
    build_initial_environments();

    double energy_prev = 0.0;

    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        printf("Sweep %3d (L->R):\n", sweep);
        double energy_LR = sweep_left_to_right();

        printf("  E(L->R) = %.12f\n", energy_LR);

        printf("Sweep %3d (R->L):\n", sweep);
        double energy_RL = sweep_right_to_left();

        energy_ = energy_RL;
        double dE = std::abs(energy_ - energy_prev);

        printf("  E(R->L) = %.12f, dE = %.2e\n\n", energy_, dE);

        if (dE < tol_ && sweep > 0) {
            printf("Converged after %d sweeps!\n", sweep + 1);
            break;
        }

        energy_prev = energy_;
    }

    return energy_;
}

// ============================================================================
// Utility methods
// ============================================================================

void DMRGGPU::load_mps_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open MPS file: " + filename);

    int L_file, d_file;
    file.read(reinterpret_cast<char*>(&L_file), sizeof(int));
    file.read(reinterpret_cast<char*>(&d_file), sizeof(int));
    if (L_file != L_ || d_file != d_)
        throw std::runtime_error("MPS file dimensions don't match");

    std::vector<int> bond_dims_file(L_ + 1);
    file.read(reinterpret_cast<char*>(bond_dims_file.data()), (L_ + 1) * sizeof(int));
    bond_dims_ = bond_dims_file;

    for (int i = 0; i < L_; i++) {
        allocate_mps_tensor(i, chi_L(i), chi_R(i));
        int size = chi_L(i) * d_ * chi_R(i);
        std::vector<double> h_A(size);
        file.read(reinterpret_cast<char*>(h_A.data()), size * sizeof(double));
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(double), hipMemcpyHostToDevice));
    }
    file.close();
}

void DMRGGPU::get_mps(std::vector<std::vector<double>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        h_mps[i].resize(size);
        HIP_CHECK(hipMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(double), hipMemcpyDeviceToHost));
    }
}
