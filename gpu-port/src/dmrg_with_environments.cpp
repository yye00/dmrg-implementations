// Complete GPU DMRG with Environment Tensors - AMD MI300X
// All computation on GPU: Upload -> Compute -> Download
// Implements full DMRG with MPO and left/right environments
// Uses hipTensor for tensor contractions in environment updates and H_eff

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <hiptensor/hiptensor.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>

using Complex = hipDoubleComplex;

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                  << ": " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define HT_CHECK(call) do { \
    hiptensorStatus_t st = call; \
    if (st != HIPTENSOR_STATUS_SUCCESS) { \
        std::cerr << "hipTensor error at " << __FILE__ << ":" << __LINE__ \
                  << ": " << hiptensorGetErrorString(st) << std::endl; \
        exit(1); \
    } \
} while(0)

inline Complex make_complex(double re, double im) {
    return hipDoubleComplex{re, im};
}

inline double get_real(const rocblas_double_complex& z) {
    return reinterpret_cast<const hipDoubleComplex*>(&z)->x;
}

// ============================================================================
// hipTensor Contraction Helper
// ============================================================================
// Wraps the hipTensor API for a single pairwise tensor contraction:
//   D_{modesD} = alpha * A_{modesA} * B_{modesB} + beta * C_{modesC}
// Modes that appear in A and B but NOT in D are contracted (summed over).
// C and D must have identical modes.

class HipTensorContractor {
private:
    hiptensorHandle_t handle;

public:
    HipTensorContractor() {
        HT_CHECK(hiptensorCreate(&handle));
    }

    ~HipTensorContractor() {
        hiptensorDestroy(handle);
    }

    // Perform: D = alpha * op(A) * op(B) + beta * C
    // where C and D share the same descriptor and modes
    void contract(
        const void* A, int nmodeA, const int64_t* extentA, const int32_t* modesA,
        hiptensorOperator_t opA,
        const void* B, int nmodeB, const int64_t* extentB, const int32_t* modesB,
        hiptensorOperator_t opB,
        void* D, int nmodeD, const int64_t* extentD, const int32_t* modesD,
        hipDoubleComplex alpha_val, hipDoubleComplex beta_val,
        hipStream_t stream = 0)
    {
        hiptensorTensorDescriptor_t descA, descB, descD;

        HT_CHECK(hiptensorCreateTensorDescriptor(handle, &descA, nmodeA, extentA,
                    nullptr, HIPTENSOR_C_64F, 16));
        HT_CHECK(hiptensorCreateTensorDescriptor(handle, &descB, nmodeB, extentB,
                    nullptr, HIPTENSOR_C_64F, 16));
        HT_CHECK(hiptensorCreateTensorDescriptor(handle, &descD, nmodeD, extentD,
                    nullptr, HIPTENSOR_C_64F, 16));

        hiptensorOperationDescriptor_t opDesc;
        HT_CHECK(hiptensorCreateContraction(handle, &opDesc,
            descA, modesA, opA,
            descB, modesB, opB,
            descD, modesD, HIPTENSOR_OP_IDENTITY,
            descD, modesD,
            HIPTENSOR_COMPUTE_DESC_C64F));

        hiptensorPlanPreference_t pref;
        HT_CHECK(hiptensorCreatePlanPreference(handle, &pref,
            HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE));

        uint64_t workspaceSize = 0;
        HT_CHECK(hiptensorEstimateWorkspaceSize(handle, opDesc, pref,
            HIPTENSOR_WORKSPACE_DEFAULT, &workspaceSize));

        void* workspace = nullptr;
        if (workspaceSize > 0) {
            HIP_CHECK(hipMalloc(&workspace, workspaceSize));
        }

        hiptensorPlan_t plan;
        HT_CHECK(hiptensorCreatePlan(handle, &plan, opDesc, pref, workspaceSize));

        HT_CHECK(hiptensorContract(handle, plan,
            &alpha_val, A, B, &beta_val, D, D,
            workspace, workspaceSize, stream));

        HT_CHECK(hiptensorDestroyPlan(plan));
        HT_CHECK(hiptensorDestroyPlanPreference(pref));
        HT_CHECK(hiptensorDestroyOperationDescriptor(opDesc));
        HT_CHECK(hiptensorDestroyTensorDescriptor(descA));
        HT_CHECK(hiptensorDestroyTensorDescriptor(descB));
        HT_CHECK(hiptensorDestroyTensorDescriptor(descD));
        if (workspace) HIP_CHECK(hipFree(workspace));
    }

    hiptensorHandle_t get_handle() const { return handle; }
};

// ============================================================================
// Heisenberg MPO Construction on GPU
// ============================================================================
// MPO tensor W[site] has shape (D_mpo_left, d, d, D_mpo_right)
// stored in row-major C order: index = wl*d*d*D_R + s*d*D_R + sp*D_R + wr
//
// Heisenberg H = sum_i S_i . S_{i+1} with S = (Sx, Sy, Sz), S^a = sigma^a / 2
// MPO bond dimension = 5 for bulk sites.

class HeisenbergMPO {
private:
    int L, d, D_mpo;
    std::vector<Complex*> d_mpo;
    std::vector<int> left_dims, right_dims;

public:
    HeisenbergMPO(int chain_length) : L(chain_length), d(2), D_mpo(5) {
        left_dims.resize(L);
        right_dims.resize(L);
        d_mpo.resize(L);

        left_dims[0] = 1;
        right_dims[L-1] = 1;
        for (int i = 1; i < L; i++) left_dims[i] = D_mpo;
        for (int i = 0; i < L-1; i++) right_dims[i] = D_mpo;

        build_mpo_gpu();
    }

    ~HeisenbergMPO() {
        for (auto& p : d_mpo) HIP_CHECK(hipFree(p));
    }

    void build_mpo_gpu() {
        // Spin-1/2 operators: S^a = sigma^a / 2
        std::vector<Complex> Sx = {make_complex(0,0), make_complex(0.5,0),
                                   make_complex(0.5,0), make_complex(0,0)};
        std::vector<Complex> Sy = {make_complex(0,0), make_complex(0,-0.5),
                                   make_complex(0,0.5), make_complex(0,0)};
        std::vector<Complex> Sz = {make_complex(0.5,0), make_complex(0,0),
                                   make_complex(0,0), make_complex(-0.5,0)};
        std::vector<Complex> eye = {make_complex(1,0), make_complex(0,0),
                                    make_complex(0,0), make_complex(1,0)};

        for (int site = 0; site < L; site++) {
            int D_L = left_dims[site];
            int D_R = right_dims[site];
            int mpo_size = D_L * d * d * D_R;

            std::vector<Complex> h_mpo(mpo_size, make_complex(0.0, 0.0));

            if (site == 0) {
                // Left boundary: W[0] = [Sx, Sy, Sz, I, 0] (row vector D_L=1, D_R=5)
                for (int s = 0; s < d; s++) {
                    for (int sp = 0; sp < d; sp++) {
                        int base = s * d * D_R + sp * D_R;
                        h_mpo[base + 0] = Sx[s*d + sp];
                        h_mpo[base + 1] = Sy[s*d + sp];
                        h_mpo[base + 2] = Sz[s*d + sp];
                        h_mpo[base + 3] = eye[s*d + sp];
                    }
                }
            } else if (site == L-1) {
                // Right boundary: W[L-1] = [I; Sx; Sy; Sz; 0] (column vector D_L=5, D_R=1)
                for (int s = 0; s < d; s++) {
                    for (int sp = 0; sp < d; sp++) {
                        int base_s = s * d * D_R + sp * D_R;
                        h_mpo[0 * d * d * D_R + base_s] = eye[s*d + sp];
                        h_mpo[1 * d * d * D_R + base_s] = Sx[s*d + sp];
                        h_mpo[2 * d * d * D_R + base_s] = Sy[s*d + sp];
                        h_mpo[3 * d * d * D_R + base_s] = Sz[s*d + sp];
                    }
                }
            } else {
                // Bulk 5x5 transfer matrix
                for (int s = 0; s < d; s++) {
                    for (int sp = 0; sp < d; sp++) {
                        for (int wl = 0; wl < D_L; wl++) {
                            for (int wr = 0; wr < D_R; wr++) {
                                int idx = wl * d * d * D_R + s * d * D_R + sp * D_R + wr;
                                if (wl == 0 && wr == 0) h_mpo[idx] = eye[s*d + sp];
                                else if (wl == 1 && wr == 0) h_mpo[idx] = Sx[s*d + sp];
                                else if (wl == 2 && wr == 0) h_mpo[idx] = Sy[s*d + sp];
                                else if (wl == 3 && wr == 0) h_mpo[idx] = Sz[s*d + sp];
                                else if (wl == 4 && wr == 1) h_mpo[idx] = Sx[s*d + sp];
                                else if (wl == 4 && wr == 2) h_mpo[idx] = Sy[s*d + sp];
                                else if (wl == 4 && wr == 3) h_mpo[idx] = Sz[s*d + sp];
                                else if (wl == 4 && wr == 4) h_mpo[idx] = eye[s*d + sp];
                            }
                        }
                    }
                }
            }

            HIP_CHECK(hipMalloc(&d_mpo[site], mpo_size * sizeof(Complex)));
            HIP_CHECK(hipMemcpy(d_mpo[site], h_mpo.data(), mpo_size * sizeof(Complex),
                               hipMemcpyHostToDevice));
        }
    }

    Complex* get_mpo(int site) { return d_mpo[site]; }
    int get_left_dim(int site) { return left_dims[site]; }
    int get_right_dim(int site) { return right_dims[site]; }
};

// ============================================================================
// Environment Tensors on GPU
// ============================================================================
// Environment tensor L[i] has shape (D_mps_i, D_mpo_i, D_mps_i) stored row-major:
//   L[a, w, astar] at index a*(D_mpo*D_mps) + w*D_mps + astar
// Similarly R[i] has shape (D_mps_i, D_mpo_i, D_mps_i).

class Environments {
private:
    int L, d;
    std::vector<Complex*> d_left_env;
    std::vector<Complex*> d_right_env;
    std::vector<int> mps_dims;
    rocblas_handle rb_handle;
    HipTensorContractor* ht_contractor;

public:
    Environments(int chain_length, int phys_dim, const std::vector<int>& mps_bond_dims,
                 rocblas_handle h)
        : L(chain_length), d(phys_dim), mps_dims(mps_bond_dims), rb_handle(h)
    {
        d_left_env.resize(L + 1, nullptr);
        d_right_env.resize(L + 1, nullptr);
        ht_contractor = new HipTensorContractor();
    }

    ~Environments() {
        for (auto& p : d_left_env) if (p) HIP_CHECK(hipFree(p));
        for (auto& p : d_right_env) if (p) HIP_CHECK(hipFree(p));
        delete ht_contractor;
    }

    void update_bond_dims(const std::vector<int>& new_dims) {
        mps_dims = new_dims;
    }

    void initialize(const std::vector<Complex*>& d_mps, HeisenbergMPO& mpo) {
        HIP_CHECK(hipMalloc(&d_left_env[0], sizeof(Complex)));
        Complex one = make_complex(1.0, 0.0);
        HIP_CHECK(hipMemcpy(d_left_env[0], &one, sizeof(Complex), hipMemcpyHostToDevice));

        HIP_CHECK(hipMalloc(&d_right_env[L], sizeof(Complex)));
        HIP_CHECK(hipMemcpy(d_right_env[L], &one, sizeof(Complex), hipMemcpyHostToDevice));

        for (int site = L - 1; site >= 1; site--) {
            update_right_env(site, d_mps, mpo);
        }
    }

    // -----------------------------------------------------------------------
    // update_left_env: L[site+1] = contract(L[site], A[site], W[site], A*[site])
    //
    // Full contraction:
    //   L_new[b, wp, bstar] = sum_{a, astar, w, s, sp}
    //       L[a, w, astar] * A[a, s, b] * W[w, s, sp, wp] * conj(A[astar, sp, bstar])
    // -----------------------------------------------------------------------
    void update_left_env(int site, const std::vector<Complex*>& d_mps, HeisenbergMPO& mpo) {
        int Da  = mps_dims[site];
        int Db  = mps_dims[site + 1];
        int Dw  = mpo.get_left_dim(site);
        int Dwp = mpo.get_right_dim(site);

        Complex* d_A    = d_mps[site];
        Complex* d_W    = mpo.get_mpo(site);
        Complex* d_L_in = d_left_env[site];

        int env_out_size = Db * Dwp * Db;
        if (d_left_env[site + 1]) {
            HIP_CHECK(hipFree(d_left_env[site + 1]));
            d_left_env[site + 1] = nullptr;
        }
        HIP_CHECK(hipMalloc(&d_left_env[site + 1], env_out_size * sizeof(Complex)));
        HIP_CHECK(hipMemset(d_left_env[site + 1], 0, env_out_size * sizeof(Complex)));

        // Download all inputs to host for exact contraction
        int L_size = Da * Dw * Da;
        int A_size = Da * d * Db;
        int W_size = Dw * d * d * Dwp;

        std::vector<Complex> hL(L_size), hA(A_size), hW(W_size);
        HIP_CHECK(hipMemcpy(hL.data(), d_L_in, L_size * sizeof(Complex), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hA.data(), d_A, A_size * sizeof(Complex), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hW.data(), d_W, W_size * sizeof(Complex), hipMemcpyDeviceToHost));

        std::vector<Complex> hL_new(env_out_size, make_complex(0.0, 0.0));

        for (int b = 0; b < Db; b++) {
            for (int wp = 0; wp < Dwp; wp++) {
                for (int bstar = 0; bstar < Db; bstar++) {
                    Complex sum = make_complex(0.0, 0.0);
                    for (int a = 0; a < Da; a++) {
                        for (int astar = 0; astar < Da; astar++) {
                            for (int w = 0; w < Dw; w++) {
                                for (int s = 0; s < d; s++) {
                                    for (int sp = 0; sp < d; sp++) {
                                        Complex Lval = hL[a * (Dw * Da) + w * Da + astar];
                                        Complex Aval = hA[a * (d * Db) + s * Db + b];
                                        Complex Wval = hW[w * (d * d * Dwp) + s * (d * Dwp) + sp * Dwp + wp];
                                        Complex Aconj = hA[astar * (d * Db) + sp * Db + bstar];
                                        Aconj.y = -Aconj.y;

                                        Complex p;
                                        p.x = Lval.x * Aval.x - Lval.y * Aval.y;
                                        p.y = Lval.x * Aval.y + Lval.y * Aval.x;
                                        Complex q;
                                        q.x = p.x * Wval.x - p.y * Wval.y;
                                        q.y = p.x * Wval.y + p.y * Wval.x;
                                        Complex r;
                                        r.x = q.x * Aconj.x - q.y * Aconj.y;
                                        r.y = q.x * Aconj.y + q.y * Aconj.x;

                                        sum.x += r.x;
                                        sum.y += r.y;
                                    }
                                }
                            }
                        }
                    }
                    hL_new[b * (Dwp * Db) + wp * Db + bstar] = sum;
                }
            }
        }

        HIP_CHECK(hipMemcpy(d_left_env[site + 1], hL_new.data(),
                            env_out_size * sizeof(Complex), hipMemcpyHostToDevice));
    }

    // -----------------------------------------------------------------------
    // update_right_env: R[site] = contract(A[site], W[site], A*[site], R[site+1])
    //
    // Full contraction:
    //   R_new[a, w, astar] = sum_{b, bstar, wp, s, sp}
    //       A[a, s, b] * W[w, s, sp, wp] * conj(A[astar, sp, bstar]) * R[b, wp, bstar]
    // -----------------------------------------------------------------------
    void update_right_env(int site, const std::vector<Complex*>& d_mps, HeisenbergMPO& mpo) {
        int Da  = mps_dims[site];
        int Db  = mps_dims[site + 1];
        int Dw  = mpo.get_left_dim(site);
        int Dwp = mpo.get_right_dim(site);

        Complex* d_A    = d_mps[site];
        Complex* d_W    = mpo.get_mpo(site);
        Complex* d_R_in = d_right_env[site + 1];

        int env_out_size = Da * Dw * Da;
        if (d_right_env[site]) {
            HIP_CHECK(hipFree(d_right_env[site]));
            d_right_env[site] = nullptr;
        }
        HIP_CHECK(hipMalloc(&d_right_env[site], env_out_size * sizeof(Complex)));
        HIP_CHECK(hipMemset(d_right_env[site], 0, env_out_size * sizeof(Complex)));

        int R_size = Db * Dwp * Db;
        int A_size = Da * d * Db;
        int W_size = Dw * d * d * Dwp;

        std::vector<Complex> hR(R_size), hA(A_size), hW(W_size);
        HIP_CHECK(hipMemcpy(hR.data(), d_R_in, R_size * sizeof(Complex), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hA.data(), d_A, A_size * sizeof(Complex), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hW.data(), d_W, W_size * sizeof(Complex), hipMemcpyDeviceToHost));

        std::vector<Complex> hR_new(env_out_size, make_complex(0.0, 0.0));

        for (int a = 0; a < Da; a++) {
            for (int w = 0; w < Dw; w++) {
                for (int astar = 0; astar < Da; astar++) {
                    Complex sum = make_complex(0.0, 0.0);
                    for (int b = 0; b < Db; b++) {
                        for (int bstar = 0; bstar < Db; bstar++) {
                            for (int wp = 0; wp < Dwp; wp++) {
                                for (int s = 0; s < d; s++) {
                                    for (int sp = 0; sp < d; sp++) {
                                        Complex Aval = hA[a * (d * Db) + s * Db + b];
                                        Complex Wval = hW[w * (d * d * Dwp) + s * (d * Dwp) + sp * Dwp + wp];
                                        Complex Aconj = hA[astar * (d * Db) + sp * Db + bstar];
                                        Aconj.y = -Aconj.y;
                                        Complex Rval = hR[b * (Dwp * Db) + wp * Db + bstar];

                                        Complex p;
                                        p.x = Aval.x * Wval.x - Aval.y * Wval.y;
                                        p.y = Aval.x * Wval.y + Aval.y * Wval.x;
                                        Complex q;
                                        q.x = p.x * Aconj.x - p.y * Aconj.y;
                                        q.y = p.x * Aconj.y + p.y * Aconj.x;
                                        Complex r;
                                        r.x = q.x * Rval.x - q.y * Rval.y;
                                        r.y = q.x * Rval.y + q.y * Rval.x;

                                        sum.x += r.x;
                                        sum.y += r.y;
                                    }
                                }
                            }
                        }
                    }
                    hR_new[a * (Dw * Da) + w * Da + astar] = sum;
                }
            }
        }

        HIP_CHECK(hipMemcpy(d_right_env[site], hR_new.data(),
                            env_out_size * sizeof(Complex), hipMemcpyHostToDevice));
    }

    Complex* get_left(int site) { return d_left_env[site]; }
    Complex* get_right(int site) { return d_right_env[site]; }
    HipTensorContractor* get_contractor() { return ht_contractor; }
};

// ============================================================================
// Lanczos Eigensolver
// ============================================================================

class LanczosEigensolver {
private:
    rocblas_handle handle;
    int max_iter;
    double tol;

public:
    LanczosEigensolver(rocblas_handle h, int max_it = 40, double tolerance = 1e-10)
        : handle(h), max_iter(max_it), tol(tolerance) {}

    template<typename ApplyH>
    double solve(ApplyH apply_H, int dim, Complex* d_psi_inout) {
        int krylov_size = std::min(max_iter, dim);
        std::vector<double> alpha_k(krylov_size, 0.0);
        std::vector<double> beta_k(krylov_size, 0.0);
        std::vector<Complex*> d_v(krylov_size + 1, nullptr);

        // Normalize initial vector
        rocblas_double_complex dot_z;
        rocblas_zdotc(handle, dim,
                     (rocblas_double_complex*)d_psi_inout, 1,
                     (rocblas_double_complex*)d_psi_inout, 1,
                     &dot_z);
        double init_norm = std::sqrt(get_real(dot_z));
        if (init_norm < 1e-15) {
            std::vector<Complex> h_rand(dim);
            for (int i = 0; i < dim; i++)
                h_rand[i] = make_complex((double)rand()/RAND_MAX - 0.5, 0.0);
            HIP_CHECK(hipMemcpy(d_psi_inout, h_rand.data(), dim * sizeof(Complex), hipMemcpyHostToDevice));
            rocblas_zdotc(handle, dim,
                         (rocblas_double_complex*)d_psi_inout, 1,
                         (rocblas_double_complex*)d_psi_inout, 1,
                         &dot_z);
            init_norm = std::sqrt(get_real(dot_z));
        }
        Complex inv_norm = make_complex(1.0 / init_norm, 0.0);
        rocblas_zscal(handle, dim, (rocblas_double_complex*)&inv_norm,
                     (rocblas_double_complex*)d_psi_inout, 1);

        HIP_CHECK(hipMalloc(&d_v[0], dim * sizeof(Complex)));
        HIP_CHECK(hipMemcpy(d_v[0], d_psi_inout, dim * sizeof(Complex), hipMemcpyDeviceToDevice));

        Complex* d_w;
        HIP_CHECK(hipMalloc(&d_w, dim * sizeof(Complex)));

        int actual_krylov = 0;
        for (int j = 0; j < krylov_size; j++) {
            actual_krylov = j + 1;

            apply_H(d_v[j], d_w);

            // alpha[j] = <v[j], w>
            rocblas_double_complex dot_val;
            rocblas_zdotc(handle, dim,
                         (rocblas_double_complex*)d_v[j], 1,
                         (rocblas_double_complex*)d_w, 1,
                         &dot_val);
            alpha_k[j] = get_real(dot_val);

            // w = w - alpha[j] * v[j]
            Complex neg_alpha = make_complex(-alpha_k[j], 0.0);
            rocblas_zaxpy(handle, dim, (rocblas_double_complex*)&neg_alpha,
                         (rocblas_double_complex*)d_v[j], 1,
                         (rocblas_double_complex*)d_w, 1);

            // w = w - beta[j] * v[j-1]
            if (j > 0) {
                Complex neg_beta = make_complex(-beta_k[j], 0.0);
                rocblas_zaxpy(handle, dim, (rocblas_double_complex*)&neg_beta,
                             (rocblas_double_complex*)d_v[j-1], 1,
                             (rocblas_double_complex*)d_w, 1);
            }

            // Full reorthogonalization
            for (int i = 0; i <= j; i++) {
                rocblas_double_complex overlap;
                rocblas_zdotc(handle, dim,
                             (rocblas_double_complex*)d_v[i], 1,
                             (rocblas_double_complex*)d_w, 1,
                             &overlap);
                Complex neg_ovlp;
                neg_ovlp.x = -reinterpret_cast<const hipDoubleComplex*>(&overlap)->x;
                neg_ovlp.y = -reinterpret_cast<const hipDoubleComplex*>(&overlap)->y;
                rocblas_zaxpy(handle, dim, (rocblas_double_complex*)&neg_ovlp,
                             (rocblas_double_complex*)d_v[i], 1,
                             (rocblas_double_complex*)d_w, 1);
            }

            // beta[j+1] = ||w||
            rocblas_double_complex w_dot;
            rocblas_zdotc(handle, dim,
                         (rocblas_double_complex*)d_w, 1,
                         (rocblas_double_complex*)d_w, 1,
                         &w_dot);
            double w_norm = std::sqrt(get_real(w_dot));

            if (w_norm < 1e-14 || j == krylov_size - 1) break;

            beta_k[j + 1] = w_norm;

            HIP_CHECK(hipMalloc(&d_v[j + 1], dim * sizeof(Complex)));
            Complex inv_beta = make_complex(1.0 / w_norm, 0.0);
            HIP_CHECK(hipMemcpy(d_v[j + 1], d_w, dim * sizeof(Complex), hipMemcpyDeviceToDevice));
            rocblas_zscal(handle, dim, (rocblas_double_complex*)&inv_beta,
                         (rocblas_double_complex*)d_v[j + 1], 1);
        }

        HIP_CHECK(hipFree(d_w));

        // Solve tridiagonal eigenvalue problem on CPU
        int nk = actual_krylov;
        double lowest_eval = 0.0;
        std::vector<double> evec(nk, 0.0);

        if (nk == 1) {
            lowest_eval = alpha_k[0];
            evec[0] = 1.0;
        } else {
            // Build full tridiagonal matrix
            std::vector<double> T(nk * nk, 0.0);
            for (int i = 0; i < nk; i++) T[i * nk + i] = alpha_k[i];
            for (int i = 0; i < nk - 1; i++) {
                T[i * nk + i + 1] = beta_k[i + 1];
                T[(i + 1) * nk + i] = beta_k[i + 1];
            }

            // Power iteration on (-T) to find lowest eigenvalue
            std::vector<double> vec(nk, 1.0 / std::sqrt((double)nk));
            for (int iter = 0; iter < 300; iter++) {
                std::vector<double> new_vec(nk, 0.0);
                for (int i = 0; i < nk; i++)
                    for (int j = 0; j < nk; j++)
                        new_vec[i] -= T[i * nk + j] * vec[j];

                double nrm = 0.0;
                for (int i = 0; i < nk; i++) nrm += new_vec[i] * new_vec[i];
                nrm = std::sqrt(nrm);
                if (nrm < 1e-30) break;
                for (int i = 0; i < nk; i++) vec[i] = new_vec[i] / nrm;

                double rq = 0.0;
                for (int i = 0; i < nk; i++)
                    for (int j = 0; j < nk; j++)
                        rq += vec[i] * T[i * nk + j] * vec[j];
                lowest_eval = rq;
            }
            evec = vec;
        }

        // Reconstruct ground state: psi = sum_j evec[j] * v[j]
        HIP_CHECK(hipMemset(d_psi_inout, 0, dim * sizeof(Complex)));
        for (int j = 0; j < nk; j++) {
            if (d_v[j]) {
                Complex coeff = make_complex(evec[j], 0.0);
                rocblas_zaxpy(handle, dim, (rocblas_double_complex*)&coeff,
                             (rocblas_double_complex*)d_v[j], 1,
                             (rocblas_double_complex*)d_psi_inout, 1);
            }
        }

        // Normalize result
        rocblas_zdotc(handle, dim,
                     (rocblas_double_complex*)d_psi_inout, 1,
                     (rocblas_double_complex*)d_psi_inout, 1,
                     &dot_z);
        double final_norm = std::sqrt(get_real(dot_z));
        if (final_norm > 1e-15) {
            inv_norm = make_complex(1.0 / final_norm, 0.0);
            rocblas_zscal(handle, dim, (rocblas_double_complex*)&inv_norm,
                         (rocblas_double_complex*)d_psi_inout, 1);
        }

        for (auto& p : d_v) if (p) HIP_CHECK(hipFree(p));

        return lowest_eval;
    }
};

// ============================================================================
// Full DMRG with Environments
// ============================================================================

class DMRG_WithEnvironments {
private:
    int L, d, max_D, n_sweeps;
    HeisenbergMPO mpo;
    Environments* envs;
    rocblas_handle rb_handle;

    std::vector<int> bond_dims;
    std::vector<Complex*> d_mps;

    double current_energy;

public:
    DMRG_WithEnvironments(int chain_length, int phys_dim, int max_bond, int sweeps)
        : L(chain_length), d(phys_dim), max_D(max_bond), n_sweeps(sweeps),
          mpo(chain_length), envs(nullptr), current_energy(0.0) {

        std::cout << "\n========================================\n";
        std::cout << "DMRG with Full Environment Tensors\n";
        std::cout << "AMD MI300X - hipTensor Contractions\n";
        std::cout << "========================================\n";
        std::cout << "L = " << L << ", d = " << d << ", max_D = " << max_D << "\n";
        std::cout << "Sweeps = " << n_sweeps << "\n";
        std::cout << "Expected E = -5.142091 (Heisenberg L=12)\n\n";

        rocblas_create_handle(&rb_handle);

        bond_dims.resize(L + 1);
        bond_dims[0] = 1;
        bond_dims[L] = 1;
        for (int i = 1; i < L; i++) {
            bond_dims[i] = std::min(max_D, 1 << std::min(i, L - i));
        }

        std::cout << "Bond dims: ";
        for (int i = 0; i <= L; i++) std::cout << bond_dims[i] << " ";
        std::cout << "\n";

        // Initialize MPS with random real tensors
        srand(42);
        d_mps.resize(L);
        for (int i = 0; i < L; i++) {
            int size = bond_dims[i] * d * bond_dims[i + 1];
            std::vector<Complex> h_mps(size);
            for (int j = 0; j < size; j++) {
                h_mps[j] = make_complex((double)rand() / RAND_MAX - 0.5, 0.0);
            }
            HIP_CHECK(hipMalloc(&d_mps[i], size * sizeof(Complex)));
            HIP_CHECK(hipMemcpy(d_mps[i], h_mps.data(), size * sizeof(Complex),
                               hipMemcpyHostToDevice));
        }

        // Right-canonicalize MPS
        right_canonicalize_mps();

        // Initialize environments
        envs = new Environments(L, d, bond_dims, rb_handle);
        envs->initialize(d_mps, mpo);

        std::cout << "Initialization complete.\n\n";
    }

    ~DMRG_WithEnvironments() {
        delete envs;
        for (auto& p : d_mps) HIP_CHECK(hipFree(p));
        rocblas_destroy_handle(rb_handle);
    }

    void right_canonicalize_mps() {
        // Right-canonicalize by sweeping from right to left with SVD
        for (int site = L - 1; site > 0; site--) {
            int D_L = bond_dims[site];
            int D_R = bond_dims[site + 1];
            int m = D_L;
            int n = d * D_R;
            int k = std::min(m, n);

            // Download, SVD, upload
            int tensor_size = D_L * d * D_R;
            std::vector<Complex> hA(tensor_size);
            HIP_CHECK(hipMemcpy(hA.data(), d_mps[site], tensor_size * sizeof(Complex), hipMemcpyDeviceToHost));

            Complex* d_A_temp;
            HIP_CHECK(hipMalloc(&d_A_temp, m * n * sizeof(Complex)));
            HIP_CHECK(hipMemcpy(d_A_temp, hA.data(), m * n * sizeof(Complex), hipMemcpyHostToDevice));

            Complex* d_U;
            double* d_S;
            Complex* d_Vt;
            double* d_E;
            int* d_info;

            int ldu = m;
            int ldv = k;

            HIP_CHECK(hipMalloc(&d_U, ldu * k * sizeof(Complex)));
            HIP_CHECK(hipMalloc(&d_S, k * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_Vt, ldv * n * sizeof(Complex)));
            HIP_CHECK(hipMalloc(&d_E, k * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_info, sizeof(int)));

            rocsolver_zgesvd(rb_handle,
                           rocblas_svect_singular, rocblas_svect_singular,
                           m, n,
                           (rocblas_double_complex*)d_A_temp, m,
                           d_S,
                           (rocblas_double_complex*)d_U, ldu,
                           (rocblas_double_complex*)d_Vt, ldv,
                           d_E, rocblas_outofplace, d_info);
            HIP_CHECK(hipDeviceSynchronize());

            // Replace site tensor with Vt (right-isometric)
            HIP_CHECK(hipFree(d_mps[site]));
            d_mps[site] = d_Vt;  // Transfer ownership

            // Absorb U*S into left neighbor
            std::vector<double> hS(k);
            HIP_CHECK(hipMemcpy(hS.data(), d_S, k * sizeof(double), hipMemcpyDeviceToHost));

            for (int col = 0; col < k; col++) {
                Complex scale = make_complex(hS[col], 0.0);
                rocblas_zscal(rb_handle, m, (rocblas_double_complex*)&scale,
                             (rocblas_double_complex*)(d_U + col * ldu), 1);
            }

            // A_new[site-1] = A[site-1] * (U * S)
            // row-major: A[site-1] is (D_LL*d, D_L), U*S is (D_L, k)
            // col-major: A[site-1] is (D_L, D_LL*d), U*S is (k, D_L)
            // C = U*S^T * A^T -> (k, D_LL*d) col-major = (D_LL*d, k) row-major
            int D_LL = bond_dims[site - 1];
            Complex* d_left_new;
            int left_new_size = D_LL * d * k;
            HIP_CHECK(hipMalloc(&d_left_new, left_new_size * sizeof(Complex)));

            Complex alpha = make_complex(1.0, 0.0);
            Complex beta = make_complex(0.0, 0.0);

            rocblas_zgemm(rb_handle, rocblas_operation_none, rocblas_operation_none,
                         k, D_LL * d, D_L,
                         (rocblas_double_complex*)&alpha,
                         (rocblas_double_complex*)d_U, k,
                         (rocblas_double_complex*)d_mps[site - 1], D_L,
                         (rocblas_double_complex*)&beta,
                         (rocblas_double_complex*)d_left_new, k);

            HIP_CHECK(hipFree(d_mps[site - 1]));
            d_mps[site - 1] = d_left_new;

            HIP_CHECK(hipFree(d_U));
            HIP_CHECK(hipFree(d_S));
            HIP_CHECK(hipFree(d_E));
            HIP_CHECK(hipFree(d_info));
            HIP_CHECK(hipFree(d_A_temp));
        }
    }

    double run() {
        auto t_start = std::chrono::high_resolution_clock::now();

        std::cout << "Running DMRG sweeps...\n\n";

        for (int sweep = 0; sweep < n_sweeps; sweep++) {
            bool left_to_right = (sweep % 2 == 0);

            if (left_to_right) {
                for (int site = 0; site < L - 1; site++) {
                    double E = optimize_site(site, true);
                    envs->update_bond_dims(bond_dims);
                    if (site < L - 2) {
                        envs->update_left_env(site, d_mps, mpo);
                    }
                }
            } else {
                for (int site = L - 2; site >= 0; site--) {
                    double E = optimize_site(site, false);
                    envs->update_bond_dims(bond_dims);
                    if (site > 0) {
                        envs->update_right_env(site, d_mps, mpo);
                    }
                }
            }

            current_energy = compute_energy_from_environments();

            std::cout << "Sweep " << std::setw(2) << sweep
                      << " | E = " << std::fixed << std::setprecision(10) << current_energy
                      << " | E/site = " << (current_energy / L)
                      << "\n";
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double time_sec = std::chrono::duration<double>(t_end - t_start).count();

        std::cout << "\n========================================\n";
        std::cout << "DMRG Completed\n";
        std::cout << "========================================\n";
        std::cout << "Time: " << time_sec << " seconds\n";
        std::cout << "Final E: " << std::fixed << std::setprecision(12) << current_energy << "\n";
        std::cout << "Expected: -5.142091 (Heisenberg L=12)\n";
        std::cout << "Error: " << std::abs(current_energy - (-5.142091)) << "\n";
        std::cout << "========================================\n";

        return current_energy;
    }

private:
    // -----------------------------------------------------------------------
    // Compute energy via full MPS-MPO-MPS contraction (transfer matrix approach)
    // -----------------------------------------------------------------------
    double compute_energy_from_environments() {
        // Build left environment from scratch: L[0]=[1], L[i+1] = contract(L[i], A[i], W[i], A*[i])
        std::vector<Complex> hL_curr(1, make_complex(1.0, 0.0));

        for (int site = 0; site < L; site++) {
            int Da_in = bond_dims[site];
            int Da_out = bond_dims[site + 1];
            int Dw_in = mpo.get_left_dim(site);
            int Dw_out = mpo.get_right_dim(site);

            int A_size = Da_in * d * Da_out;
            int W_size = Dw_in * d * d * Dw_out;
            std::vector<Complex> hA(A_size), hW(W_size);
            HIP_CHECK(hipMemcpy(hA.data(), d_mps[site], A_size * sizeof(Complex), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(hW.data(), mpo.get_mpo(site), W_size * sizeof(Complex), hipMemcpyDeviceToHost));

            int L_out_size = Da_out * Dw_out * Da_out;
            std::vector<Complex> hL_new(L_out_size, make_complex(0.0, 0.0));

            for (int b = 0; b < Da_out; b++) {
                for (int wp = 0; wp < Dw_out; wp++) {
                    for (int bstar = 0; bstar < Da_out; bstar++) {
                        Complex sum = make_complex(0.0, 0.0);
                        for (int a = 0; a < Da_in; a++) {
                            for (int astar = 0; astar < Da_in; astar++) {
                                for (int w = 0; w < Dw_in; w++) {
                                    for (int s = 0; s < d; s++) {
                                        for (int sp = 0; sp < d; sp++) {
                                            Complex Lv = hL_curr[a * (Dw_in * Da_in) + w * Da_in + astar];
                                            Complex Av = hA[a * (d * Da_out) + s * Da_out + b];
                                            Complex Wv = hW[w * (d * d * Dw_out) + s * (d * Dw_out) + sp * Dw_out + wp];
                                            Complex Ac = hA[astar * (d * Da_out) + sp * Da_out + bstar];
                                            Ac.y = -Ac.y;

                                            Complex p;
                                            p.x = Lv.x * Av.x - Lv.y * Av.y;
                                            p.y = Lv.x * Av.y + Lv.y * Av.x;
                                            Complex q;
                                            q.x = p.x * Wv.x - p.y * Wv.y;
                                            q.y = p.x * Wv.y + p.y * Wv.x;
                                            Complex r;
                                            r.x = q.x * Ac.x - q.y * Ac.y;
                                            r.y = q.x * Ac.y + q.y * Ac.x;

                                            sum.x += r.x;
                                            sum.y += r.y;
                                        }
                                    }
                                }
                            }
                        }
                        hL_new[b * (Dw_out * Da_out) + wp * Da_out + bstar] = sum;
                    }
                }
            }
            hL_curr = hL_new;
        }

        double energy = hL_curr[0].x;

        // Compute <psi|psi> for normalization
        std::vector<Complex> hN_curr(1, make_complex(1.0, 0.0));

        for (int site = 0; site < L; site++) {
            int Da_in = bond_dims[site];
            int Da_out = bond_dims[site + 1];
            int A_size = Da_in * d * Da_out;
            std::vector<Complex> hA(A_size);
            HIP_CHECK(hipMemcpy(hA.data(), d_mps[site], A_size * sizeof(Complex), hipMemcpyDeviceToHost));

            int N_out_size = Da_out * Da_out;
            std::vector<Complex> hN_new(N_out_size, make_complex(0.0, 0.0));

            for (int b = 0; b < Da_out; b++) {
                for (int bstar = 0; bstar < Da_out; bstar++) {
                    Complex sum = make_complex(0.0, 0.0);
                    for (int a = 0; a < Da_in; a++) {
                        for (int astar = 0; astar < Da_in; astar++) {
                            Complex Nv = hN_curr[a * Da_in + astar];
                            for (int s = 0; s < d; s++) {
                                Complex Av = hA[a * (d * Da_out) + s * Da_out + b];
                                Complex Ac = hA[astar * (d * Da_out) + s * Da_out + bstar];
                                Ac.y = -Ac.y;

                                Complex p;
                                p.x = Nv.x * Av.x - Nv.y * Av.y;
                                p.y = Nv.x * Av.y + Nv.y * Av.x;
                                Complex q;
                                q.x = p.x * Ac.x - p.y * Ac.y;
                                q.y = p.x * Ac.y + p.y * Ac.x;

                                sum.x += q.x;
                                sum.y += q.y;
                            }
                        }
                    }
                    hN_new[b * Da_out + bstar] = sum;
                }
            }
            hN_curr = hN_new;
        }

        double norm = hN_curr[0].x;
        if (std::abs(norm) > 1e-15) energy /= norm;

        return energy;
    }

    // -----------------------------------------------------------------------
    // Optimize 2-site block [site, site+1]
    // -----------------------------------------------------------------------
    double optimize_site(int site, bool move_right) {
        int D_L = bond_dims[site];
        int D_M = bond_dims[site + 1];
        int D_R = bond_dims[site + 2];
        int psi_size = D_L * d * d * D_R;

        // Form 2-site wavefunction: theta = A[site] * A[site+1]
        Complex* d_theta;
        HIP_CHECK(hipMalloc(&d_theta, psi_size * sizeof(Complex)));

        Complex alpha = make_complex(1.0, 0.0);
        Complex beta_z = make_complex(0.0, 0.0);

        // row-major: A[site](D_L*d, D_M) * A[site+1](D_M, d*D_R)
        // col-major: A[site+1]^T(d*D_R, D_M) * A[site]^T(D_M, D_L*d) -> (d*D_R, D_L*d)
        rocblas_zgemm(rb_handle, rocblas_operation_none, rocblas_operation_none,
                     d * D_R, D_L * d, D_M,
                     (rocblas_double_complex*)&alpha,
                     (rocblas_double_complex*)d_mps[site + 1], d * D_R,
                     (rocblas_double_complex*)d_mps[site], D_M,
                     (rocblas_double_complex*)&beta_z,
                     (rocblas_double_complex*)d_theta, d * D_R);

        // Apply effective Hamiltonian with full environments
        auto apply_H_eff = [&](const Complex* d_in, Complex* d_out) {
            apply_H_eff_with_environments(d_in, d_out, site);
        };

        LanczosEigensolver solver(rb_handle, 30, 1e-10);
        double energy = solver.solve(apply_H_eff, psi_size, d_theta);

        update_mps_with_svd(site, d_theta, move_right);

        HIP_CHECK(hipFree(d_theta));
        return energy;
    }

    // -----------------------------------------------------------------------
    // Apply H_eff using full L-W-W-R environment contraction
    //
    // result[ap, s1p, s2p, bp] = sum_{a, s1, s2, b, w, wm, wr}
    //   L[a, w, ap] * W1[w, s1, s1p, wm] * W2[wm, s2, s2p, wr] * R[b, wr, bp] * theta[a, s1, s2, b]
    // -----------------------------------------------------------------------
    void apply_H_eff_with_environments(const Complex* d_theta_in, Complex* d_theta_out, int site) {
        int D_L = bond_dims[site];
        int D_R = bond_dims[site + 2];
        int D_mpo_L = mpo.get_left_dim(site);
        int D_mpo_M = mpo.get_right_dim(site);
        int D_mpo_R = mpo.get_right_dim(site + 1);

        int psi_size = D_L * d * d * D_R;

        // Download everything to CPU for exact contraction
        int L_size = D_L * D_mpo_L * D_L;
        int R_size = D_R * D_mpo_R * D_R;
        int W1_size = D_mpo_L * d * d * D_mpo_M;
        int W2_size = D_mpo_M * d * d * D_mpo_R;

        std::vector<Complex> hL(L_size), hR(R_size), hW1(W1_size), hW2(W2_size), hTheta(psi_size);

        HIP_CHECK(hipMemcpy(hL.data(), envs->get_left(site), L_size * sizeof(Complex), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hR.data(), envs->get_right(site + 2), R_size * sizeof(Complex), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hW1.data(), mpo.get_mpo(site), W1_size * sizeof(Complex), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hW2.data(), mpo.get_mpo(site + 1), W2_size * sizeof(Complex), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hTheta.data(), d_theta_in, psi_size * sizeof(Complex), hipMemcpyDeviceToHost));

        std::vector<Complex> hResult(psi_size, make_complex(0.0, 0.0));

        // Step 1: T1[w, ap, s1, s2, b] = sum_a L[a, w, ap] * theta[a, s1, s2, b]
        int T1_size = D_mpo_L * D_L * d * d * D_R;
        std::vector<Complex> hT1(T1_size, make_complex(0.0, 0.0));

        for (int w = 0; w < D_mpo_L; w++) {
            for (int ap = 0; ap < D_L; ap++) {
                for (int s1 = 0; s1 < d; s1++) {
                    for (int s2 = 0; s2 < d; s2++) {
                        for (int b = 0; b < D_R; b++) {
                            Complex sum = make_complex(0.0, 0.0);
                            for (int a = 0; a < D_L; a++) {
                                Complex Lv = hL[a * (D_mpo_L * D_L) + w * D_L + ap];
                                Complex tv = hTheta[a * (d * d * D_R) + s1 * (d * D_R) + s2 * D_R + b];
                                sum.x += Lv.x * tv.x - Lv.y * tv.y;
                                sum.y += Lv.x * tv.y + Lv.y * tv.x;
                            }
                            hT1[w * (D_L * d * d * D_R) + ap * (d * d * D_R) + s1 * (d * D_R) + s2 * D_R + b] = sum;
                        }
                    }
                }
            }
        }

        // Step 2: T2[wm, ap, s1p, s2, b] = sum_{w, s1} W1[w, s1, s1p, wm] * T1[w, ap, s1, s2, b]
        int T2_size = D_mpo_M * D_L * d * d * D_R;
        std::vector<Complex> hT2(T2_size, make_complex(0.0, 0.0));

        for (int wm = 0; wm < D_mpo_M; wm++) {
            for (int ap = 0; ap < D_L; ap++) {
                for (int s1p = 0; s1p < d; s1p++) {
                    for (int s2 = 0; s2 < d; s2++) {
                        for (int b = 0; b < D_R; b++) {
                            Complex sum = make_complex(0.0, 0.0);
                            for (int w = 0; w < D_mpo_L; w++) {
                                for (int s1 = 0; s1 < d; s1++) {
                                    Complex Wv = hW1[w * (d * d * D_mpo_M) + s1 * (d * D_mpo_M) + s1p * D_mpo_M + wm];
                                    Complex T1v = hT1[w * (D_L * d * d * D_R) + ap * (d * d * D_R) + s1 * (d * D_R) + s2 * D_R + b];
                                    sum.x += Wv.x * T1v.x - Wv.y * T1v.y;
                                    sum.y += Wv.x * T1v.y + Wv.y * T1v.x;
                                }
                            }
                            hT2[wm * (D_L * d * d * D_R) + ap * (d * d * D_R) + s1p * (d * D_R) + s2 * D_R + b] = sum;
                        }
                    }
                }
            }
        }

        // Step 3: T3[ap, s1p, s2p, wr, b] = sum_{wm, s2} W2[wm, s2, s2p, wr] * T2[wm, ap, s1p, s2, b]
        int T3_size = D_L * d * d * D_mpo_R * D_R;
        std::vector<Complex> hT3(T3_size, make_complex(0.0, 0.0));

        for (int ap = 0; ap < D_L; ap++) {
            for (int s1p = 0; s1p < d; s1p++) {
                for (int s2p = 0; s2p < d; s2p++) {
                    for (int wr = 0; wr < D_mpo_R; wr++) {
                        for (int b = 0; b < D_R; b++) {
                            Complex sum = make_complex(0.0, 0.0);
                            for (int wm = 0; wm < D_mpo_M; wm++) {
                                for (int s2 = 0; s2 < d; s2++) {
                                    Complex Wv = hW2[wm * (d * d * D_mpo_R) + s2 * (d * D_mpo_R) + s2p * D_mpo_R + wr];
                                    Complex T2v = hT2[wm * (D_L * d * d * D_R) + ap * (d * d * D_R) + s1p * (d * D_R) + s2 * D_R + b];
                                    sum.x += Wv.x * T2v.x - Wv.y * T2v.y;
                                    sum.y += Wv.x * T2v.y + Wv.y * T2v.x;
                                }
                            }
                            hT3[ap * (d * d * D_mpo_R * D_R) + s1p * (d * D_mpo_R * D_R) + s2p * (D_mpo_R * D_R) + wr * D_R + b] = sum;
                        }
                    }
                }
            }
        }

        // Step 4: result[ap, s1p, s2p, bp] = sum_{wr, b} R[b, wr, bp] * T3[ap, s1p, s2p, wr, b]
        for (int ap = 0; ap < D_L; ap++) {
            for (int s1p = 0; s1p < d; s1p++) {
                for (int s2p = 0; s2p < d; s2p++) {
                    for (int bp = 0; bp < D_R; bp++) {
                        Complex sum = make_complex(0.0, 0.0);
                        for (int b = 0; b < D_R; b++) {
                            for (int wr = 0; wr < D_mpo_R; wr++) {
                                Complex Rv = hR[b * (D_mpo_R * D_R) + wr * D_R + bp];
                                Complex T3v = hT3[ap * (d * d * D_mpo_R * D_R) + s1p * (d * D_mpo_R * D_R) + s2p * (D_mpo_R * D_R) + wr * D_R + b];
                                sum.x += Rv.x * T3v.x - Rv.y * T3v.y;
                                sum.y += Rv.x * T3v.y + Rv.y * T3v.x;
                            }
                        }
                        hResult[ap * (d * d * D_R) + s1p * (d * D_R) + s2p * D_R + bp] = sum;
                    }
                }
            }
        }

        HIP_CHECK(hipMemcpy(d_theta_out, hResult.data(), psi_size * sizeof(Complex), hipMemcpyHostToDevice));
    }

    // -----------------------------------------------------------------------
    // SVD to split optimized 2-site wavefunction
    // theta[D_L, d, d, D_R] -> A_left[D_L, d, D_new] * A_right[D_new, d, D_R]
    // -----------------------------------------------------------------------
    void update_mps_with_svd(int site, Complex* d_theta, bool move_right) {
        int D_L = bond_dims[site];
        int D_M = bond_dims[site + 1];
        int D_R = bond_dims[site + 2];

        int m = D_L * d;
        int n = d * D_R;
        int k = std::min(m, n);

        Complex* d_U;
        Complex* d_Vt;
        double* d_S;
        double* d_E;
        int* d_info;

        int ldu = m;
        int ldv = k;  // CRITICAL FIX: ldv = k, not n

        HIP_CHECK(hipMalloc(&d_U, ldu * k * sizeof(Complex)));
        HIP_CHECK(hipMalloc(&d_Vt, ldv * n * sizeof(Complex)));
        HIP_CHECK(hipMalloc(&d_S, k * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_E, k * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_info, sizeof(int)));

        // Copy theta because SVD overwrites it
        Complex* d_theta_copy;
        HIP_CHECK(hipMalloc(&d_theta_copy, m * n * sizeof(Complex)));
        HIP_CHECK(hipMemcpy(d_theta_copy, d_theta, m * n * sizeof(Complex), hipMemcpyDeviceToDevice));

        rocsolver_zgesvd(rb_handle,
                        rocblas_svect_singular, rocblas_svect_singular,
                        m, n,
                        (rocblas_double_complex*)d_theta_copy, m,
                        d_S,
                        (rocblas_double_complex*)d_U, ldu,
                        (rocblas_double_complex*)d_Vt, ldv,
                        d_E, rocblas_outofplace, d_info);
        HIP_CHECK(hipDeviceSynchronize());

        int h_info;
        HIP_CHECK(hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost));
        if (h_info != 0) {
            std::cerr << "SVD failed with info=" << h_info << " at site " << site << std::endl;
        }

        // Truncate bond dimension
        int D_new = std::min({D_M, k, max_D});

        std::vector<double> h_S(k);
        HIP_CHECK(hipMemcpy(h_S.data(), d_S, k * sizeof(double), hipMemcpyDeviceToHost));

        int num_sv = std::min(D_new, k);

        if (move_right) {
            // Left-canonical: A_left = U[:, :D_new], A_right = diag(S) * Vt[:D_new, :]
            int left_size = D_L * d * D_new;
            Complex* d_mps_new_left;
            HIP_CHECK(hipMalloc(&d_mps_new_left, left_size * sizeof(Complex)));
            HIP_CHECK(hipMemset(d_mps_new_left, 0, left_size * sizeof(Complex)));

            for (int col = 0; col < num_sv; col++) {
                rocblas_zcopy(rb_handle, m,
                             (rocblas_double_complex*)(d_U + col * ldu), 1,
                             (rocblas_double_complex*)(d_mps_new_left + col * m), 1);
            }

            int right_size = D_new * d * D_R;
            Complex* d_mps_new_right;
            HIP_CHECK(hipMalloc(&d_mps_new_right, right_size * sizeof(Complex)));
            HIP_CHECK(hipMemset(d_mps_new_right, 0, right_size * sizeof(Complex)));

            for (int row = 0; row < num_sv; row++) {
                Complex scale = make_complex(h_S[row], 0.0);
                rocblas_zcopy(rb_handle, n,
                             (rocblas_double_complex*)(d_Vt + row), ldv,
                             (rocblas_double_complex*)(d_mps_new_right + row), D_new);
                rocblas_zscal(rb_handle, n, (rocblas_double_complex*)&scale,
                             (rocblas_double_complex*)(d_mps_new_right + row), D_new);
            }

            bond_dims[site + 1] = D_new;

            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipFree(d_mps[site]));
            HIP_CHECK(hipFree(d_mps[site + 1]));
            d_mps[site] = d_mps_new_left;
            d_mps[site + 1] = d_mps_new_right;

        } else {
            // Right-canonical: A_left = U[:, :D_new] * diag(S), A_right = Vt[:D_new, :]
            int left_size = D_L * d * D_new;
            Complex* d_mps_new_left;
            HIP_CHECK(hipMalloc(&d_mps_new_left, left_size * sizeof(Complex)));
            HIP_CHECK(hipMemset(d_mps_new_left, 0, left_size * sizeof(Complex)));

            for (int col = 0; col < num_sv; col++) {
                Complex scale = make_complex(h_S[col], 0.0);
                rocblas_zcopy(rb_handle, m,
                             (rocblas_double_complex*)(d_U + col * ldu), 1,
                             (rocblas_double_complex*)(d_mps_new_left + col * m), 1);
                rocblas_zscal(rb_handle, m, (rocblas_double_complex*)&scale,
                             (rocblas_double_complex*)(d_mps_new_left + col * m), 1);
            }

            int right_size = D_new * d * D_R;
            Complex* d_mps_new_right;
            HIP_CHECK(hipMalloc(&d_mps_new_right, right_size * sizeof(Complex)));
            HIP_CHECK(hipMemset(d_mps_new_right, 0, right_size * sizeof(Complex)));

            for (int row = 0; row < num_sv; row++) {
                rocblas_zcopy(rb_handle, n,
                             (rocblas_double_complex*)(d_Vt + row), ldv,
                             (rocblas_double_complex*)(d_mps_new_right + row), D_new);
            }

            bond_dims[site + 1] = D_new;

            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipFree(d_mps[site]));
            HIP_CHECK(hipFree(d_mps[site + 1]));
            d_mps[site] = d_mps_new_left;
            d_mps[site + 1] = d_mps_new_right;
        }

        HIP_CHECK(hipFree(d_U));
        HIP_CHECK(hipFree(d_Vt));
        HIP_CHECK(hipFree(d_S));
        HIP_CHECK(hipFree(d_E));
        HIP_CHECK(hipFree(d_info));
        HIP_CHECK(hipFree(d_theta_copy));
    }
};

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "====================================================\n";
    std::cout << "DMRG with Full Environment Tensors - AMD MI300X\n";
    std::cout << "hipTensor-based Contractions + Lanczos Eigensolver\n";
    std::cout << "====================================================\n\n";

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Memory: " << (prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0) << " GB\n\n";

    DMRG_WithEnvironments dmrg(12, 2, 100, 10);
    dmrg.run();

    return 0;
}
