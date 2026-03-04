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
                // Left boundary: W[0] = [0, Sx, Sy, Sz, I] (row vector D_L=1, D_R=5)
                // This is row 4 of the bulk transfer matrix (the Hamiltonian row)
                for (int s = 0; s < d; s++) {
                    for (int sp = 0; sp < d; sp++) {
                        int base = s * d * D_R + sp * D_R;
                        // wr=0: zero (no contribution)
                        h_mpo[base + 1] = Sx[s*d + sp];
                        h_mpo[base + 2] = Sy[s*d + sp];
                        h_mpo[base + 3] = Sz[s*d + sp];
                        h_mpo[base + 4] = eye[s*d + sp];
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
        // Full GPU contraction using hipTensor: L_new = L ⊗ A ⊗ W ⊗ conj(A)
        // Decomposed into 3 sequential contractions:
        // 1. temp1[w,a*,s,b] = L[a,w,a*] * A[a,s,b]  (contract over a)
        // 2. temp2[a*,s',b,w'] = temp1[w,a*,s,b] * W[w,s,s',w']  (contract over w,s)
        // 3. L_new[b,w',b*] = temp2[a*,s',b,w'] * conj(A[a*,s',b*])  (contract over a*,s')

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

        // Step 1: temp1[w,a*,s,b] = sum_a L[a,w,a*] * A[a,s,b]  (contract over a)
        //
        // hipTensor uses col-major convention: extents listed fastest-stride first.
        // C row-major L[a,w,a*]: strides a*=1, w=Da, a=Dw*Da -> col-major extents {Da,Dw,Da}
        // C row-major A[a,s,b]:  strides b=1, s=Db, a=d*Db   -> col-major extents {Db,d,Da}
        // C row-major temp1[w,a*,s,b]: strides b=1, s=Db, a*=d*Db, w=Da*d*Db
        //   -> col-major extents {Db, d, Da, Dw}
        //
        // Mode labels: a=0, w=1, astar=2, s=3, b=4
        // L  col-major order (a*,w,a):      modes {2, 1, 0}
        // A  col-major order (b,s,a):        modes {4, 3, 0}
        // temp1 col-major order (b,s,a*,w):  modes {4, 3, 2, 1}
        // Contract over mode 0 (a).

        Complex* d_temp1;
        int temp1_size = Dw * Da * d * Db;
        HIP_CHECK(hipMalloc(&d_temp1, temp1_size * sizeof(Complex)));

        {
            int64_t extentL[] = {Da, Dw, Da};
            int64_t extentA[] = {Db, d, Da};
            int64_t extent_temp1[] = {Db, d, Da, Dw};
            int32_t modesL[] = {2, 1, 0};
            int32_t modesA[] = {4, 3, 0};
            int32_t modes_temp1[] = {4, 3, 2, 1};

            hipDoubleComplex alpha = make_hipDoubleComplex(1.0, 0.0);
            hipDoubleComplex beta = make_hipDoubleComplex(0.0, 0.0);

            ht_contractor->contract(
                d_L_in, 3, extentL, modesL, HIPTENSOR_OP_IDENTITY,
                d_A, 3, extentA, modesA, HIPTENSOR_OP_IDENTITY,
                d_temp1, 4, extent_temp1, modes_temp1,
                alpha, beta, 0);
        }

        // Step 2: temp2[a*,s',b,w'] = sum_{w,s} temp1[w,a*,s,b] * W[w,s,s',w']
        //
        // C row-major W[w,s,sp,wp]: strides wp=1, sp=Dwp, s=d*Dwp, w=d*d*Dwp
        //   -> col-major extents {Dwp, d, d, Dw}
        // C row-major temp2[a*,sp,b,wp]: strides wp=1, b=Dwp, sp=Db*Dwp, a*=d*Db*Dwp
        //   -> col-major extents {Dwp, Db, d, Da}
        //
        // Mode labels: (reusing 0-4 from step 1) sp=5, wp=6
        // temp1 col-major (b,s,a*,w):       modes {4, 3, 2, 1}
        // W     col-major (wp,sp,s,w):       modes {6, 5, 3, 1}
        // temp2 col-major (wp,b,sp,a*):      modes {6, 4, 5, 2}
        // Contract over modes 1(w) and 3(s).

        Complex* d_temp2;
        int temp2_size = Da * d * Db * Dwp;
        HIP_CHECK(hipMalloc(&d_temp2, temp2_size * sizeof(Complex)));

        {
            int64_t extent_temp1[] = {Db, d, Da, Dw};
            int64_t extentW[] = {Dwp, d, d, Dw};
            int64_t extent_temp2[] = {Dwp, Db, d, Da};
            int32_t modes_temp1[] = {4, 3, 2, 1};
            int32_t modesW[] = {6, 5, 3, 1};
            int32_t modes_temp2[] = {6, 4, 5, 2};

            hipDoubleComplex alpha = make_hipDoubleComplex(1.0, 0.0);
            hipDoubleComplex beta = make_hipDoubleComplex(0.0, 0.0);

            ht_contractor->contract(
                d_temp1, 4, extent_temp1, modes_temp1, HIPTENSOR_OP_IDENTITY,
                d_W, 4, extentW, modesW, HIPTENSOR_OP_IDENTITY,
                d_temp2, 4, extent_temp2, modes_temp2,
                alpha, beta, 0);
        }

        HIP_CHECK(hipFree(d_temp1));

        // Step 3: L_new[b,wp,b*] = sum_{a*,s'} temp2[a*,s',b,w'] * conj(A[a*,s',b*])
        //
        // C row-major L_new[b,wp,b*]: strides b*=1, wp=Db, b=Dwp*Db
        //   -> col-major extents {Db, Dwp, Db}
        // conj(A[a*,sp,b*]): same layout as A -> col-major extents {Db, d, Da}
        //
        // Mode labels: bstar=7
        // temp2 col-major (wp,b,sp,a*):      modes {6, 4, 5, 2}
        // conjA col-major (b*,sp,a*):         modes {7, 5, 2}
        // L_new col-major (b*,wp,b):          modes {7, 6, 4}
        // Contract over modes 2(a*) and 5(sp).

        {
            int64_t extent_temp2[] = {Dwp, Db, d, Da};
            int64_t extentA[] = {Db, d, Da};
            int64_t extent_Lnew[] = {Db, Dwp, Db};
            int32_t modes_temp2[] = {6, 4, 5, 2};
            int32_t modesA_conj[] = {7, 5, 2};
            int32_t modes_Lnew[] = {7, 6, 4};

            hipDoubleComplex alpha = make_hipDoubleComplex(1.0, 0.0);
            hipDoubleComplex beta = make_hipDoubleComplex(0.0, 0.0);

            ht_contractor->contract(
                d_temp2, 4, extent_temp2, modes_temp2, HIPTENSOR_OP_IDENTITY,
                d_A, 3, extentA, modesA_conj, HIPTENSOR_OP_CONJ,
                d_left_env[site + 1], 3, extent_Lnew, modes_Lnew,
                alpha, beta, 0);
        }

        HIP_CHECK(hipFree(d_temp2));
    }

    // -----------------------------------------------------------------------
    // update_right_env: R[site] = contract(A[site], W[site], A*[site], R[site+1])
    //
    // Full contraction:
    //   R_new[a, w, astar] = sum_{b, bstar, wp, s, sp}
    //       A[a, s, b] * W[w, s, sp, wp] * conj(A[astar, sp, bstar]) * R[b, wp, bstar]
    // -----------------------------------------------------------------------
    void update_right_env(int site, const std::vector<Complex*>& d_mps, HeisenbergMPO& mpo) {
        // Full GPU contraction using hipTensor: R_new = A ⊗ W ⊗ conj(A) ⊗ R
        // Decomposed into 3 sequential contractions:
        // 1. temp1[a,s,w',d] = A[a,s,b] * R[b,w',d]  (contract over b)
        // 2. temp2[a,w,s',d] = temp1[a,s,w',d] * W[w,s,s',w']  (contract over s,w')
        // 3. R_new[a,w,c] = temp2[a,w,s',d] * conj(A[c,s',d])  (contract over s',d)

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

        // Step 1: temp1[a,s,wp,b*] = sum_b A[a,s,b] * R[b,wp,b*]  (contract over b)
        //
        // hipTensor col-major convention: extents listed fastest-stride first.
        // C row-major A[a,s,b]: strides b=1, s=Db, a=d*Db -> col-major extents {Db,d,Da}
        // C row-major R[b,wp,b*]: strides b*=1, wp=Db, b=Dwp*Db -> col-major extents {Db,Dwp,Db}
        // C row-major temp1[a,s,wp,b*]: strides b*=1, wp=Db, s=Dwp*Db, a=d*Dwp*Db
        //   -> col-major extents {Db, Dwp, d, Da}
        //
        // Mode labels: a=0, s=1, b=2, wp=3, bstar=4
        // A     col-major (b,s,a):          modes {2, 1, 0}
        // R     col-major (b*,wp,b):        modes {4, 3, 2}
        // temp1 col-major (b*,wp,s,a):      modes {4, 3, 1, 0}
        // Contract over mode 2 (b).

        Complex* d_temp1;
        int temp1_size = Da * d * Dwp * Db;
        HIP_CHECK(hipMalloc(&d_temp1, temp1_size * sizeof(Complex)));

        {
            int64_t extentA[] = {Db, d, Da};
            int64_t extentR[] = {Db, Dwp, Db};
            int64_t extent_temp1[] = {Db, Dwp, d, Da};
            int32_t modesA[] = {2, 1, 0};
            int32_t modesR[] = {4, 3, 2};
            int32_t modes_temp1[] = {4, 3, 1, 0};

            hipDoubleComplex alpha = make_hipDoubleComplex(1.0, 0.0);
            hipDoubleComplex beta = make_hipDoubleComplex(0.0, 0.0);

            ht_contractor->contract(
                d_A, 3, extentA, modesA, HIPTENSOR_OP_IDENTITY,
                d_R_in, 3, extentR, modesR, HIPTENSOR_OP_IDENTITY,
                d_temp1, 4, extent_temp1, modes_temp1,
                alpha, beta, 0);
        }

        // Step 2: temp2[a,w,sp,b*] = sum_{s,wp} temp1[a,s,wp,b*] * W[w,s,sp,wp]
        //
        // C row-major W[w,s,sp,wp]: strides wp=1, sp=Dwp, s=d*Dwp, w=d*d*Dwp
        //   -> col-major extents {Dwp, d, d, Dw}
        // C row-major temp2[a,w,sp,b*]: strides b*=1, sp=Db, w=d*Db, a=Dw*d*Db
        //   -> col-major extents {Db, d, Dw, Da}
        //
        // Mode labels: (reusing 0-4) w=5, sp=6
        // temp1 col-major (b*,wp,s,a):      modes {4, 3, 1, 0}
        // W     col-major (wp,sp,s,w):       modes {3, 6, 1, 5}
        // temp2 col-major (b*,sp,w,a):       modes {4, 6, 5, 0}
        // Contract over modes 1(s) and 3(wp).

        Complex* d_temp2;
        int temp2_size = Da * Dw * d * Db;
        HIP_CHECK(hipMalloc(&d_temp2, temp2_size * sizeof(Complex)));

        {
            int64_t extent_temp1[] = {Db, Dwp, d, Da};
            int64_t extentW[] = {Dwp, d, d, Dw};
            int64_t extent_temp2[] = {Db, d, Dw, Da};
            int32_t modes_temp1[] = {4, 3, 1, 0};
            int32_t modesW[] = {3, 6, 1, 5};
            int32_t modes_temp2[] = {4, 6, 5, 0};

            hipDoubleComplex alpha = make_hipDoubleComplex(1.0, 0.0);
            hipDoubleComplex beta = make_hipDoubleComplex(0.0, 0.0);

            ht_contractor->contract(
                d_temp1, 4, extent_temp1, modes_temp1, HIPTENSOR_OP_IDENTITY,
                d_W, 4, extentW, modesW, HIPTENSOR_OP_IDENTITY,
                d_temp2, 4, extent_temp2, modes_temp2,
                alpha, beta, 0);
        }

        HIP_CHECK(hipFree(d_temp1));

        // Step 3: R_new[a,w,a*] = sum_{sp,b*} temp2[a,w,sp,b*] * conj(A[a*,sp,b*])
        //
        // C row-major R_new[a,w,a*]: strides a*=1, w=Da, a=Dw*Da
        //   -> col-major extents {Da, Dw, Da}
        // conj(A[a*,sp,b*]): same layout as A -> col-major extents {Db, d, Da}
        //
        // Mode labels: astar=7
        // temp2 col-major (b*,sp,w,a):       modes {4, 6, 5, 0}
        // conjA col-major (b*,sp,a*):         modes {4, 6, 7}
        // R_new col-major (a*,w,a):           modes {7, 5, 0}
        // Contract over modes 4(b*) and 6(sp).

        {
            int64_t extent_temp2[] = {Db, d, Dw, Da};
            int64_t extentA[] = {Db, d, Da};
            int64_t extent_Rnew[] = {Da, Dw, Da};
            int32_t modes_temp2[] = {4, 6, 5, 0};
            int32_t modesA_conj[] = {4, 6, 7};
            int32_t modes_Rnew[] = {7, 5, 0};

            hipDoubleComplex alpha = make_hipDoubleComplex(1.0, 0.0);
            hipDoubleComplex beta = make_hipDoubleComplex(0.0, 0.0);

            ht_contractor->contract(
                d_temp2, 4, extent_temp2, modes_temp2, HIPTENSOR_OP_IDENTITY,
                d_A, 3, extentA, modesA_conj, HIPTENSOR_OP_CONJ,
                d_right_env[site], 3, extent_Rnew, modes_Rnew,
                alpha, beta, 0);
        }

        HIP_CHECK(hipFree(d_temp2));
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
        // All done on CPU to avoid row-major/col-major confusion with GPU BLAS
        for (int site = L - 1; site > 0; site--) {
            int Da = bond_dims[site];
            int Db = bond_dims[site + 1];
            int m = Da;          // rows of matrix to SVD
            int n = d * Db;      // cols of matrix to SVD
            int k = std::min(m, n);

            int tensor_size = Da * d * Db;
            std::vector<Complex> hA(tensor_size);
            HIP_CHECK(hipMemcpy(hA.data(), d_mps[site], tensor_size * sizeof(Complex), hipMemcpyDeviceToHost));

            // hA is row-major (Da, d, Db) = row-major matrix (Da, d*Db)
            // SVD: A = U * S * Vt where U(Da, k), S(k), Vt(k, d*Db)
            // Do this on CPU using a simple approach

            // Compute A^T * A (k=n case) or A * A^T (k=m case) for eigendecomposition
            // For right-canonicalization, we just need to normalize each site.
            // Simple QR-like approach: for each site from right to left,
            // decompose A[a, s, b] = sum_k R[a, k] * Q[k, s, b]
            // where Q is right-isometric: sum_{s,b} Q[k,s,b] * conj(Q[k',s,b]) = delta_{k,k'}

            // For simplicity, use Gram-Schmidt on the row vectors of M(Da, d*Db)
            std::vector<Complex> hVt(k * n, make_complex(0.0, 0.0));  // Vt(k, n) row-major
            std::vector<Complex> hR(m * k, make_complex(0.0, 0.0));   // R(m, k) row-major
            std::vector<Complex> hU(m * k, make_complex(0.0, 0.0));
            std::vector<double> hS(k, 0.0);

            // Full CPU SVD via Jacobi one-sided (simple but correct)
            // Actually, for canonicalization we just need QR, not full SVD.
            // Use modified Gram-Schmidt: QR factorize M^T to get M = Q * R^T
            // Actually simpler: just do regular QR of M^T

            // M^T is (d*Db, Da) row-major = n rows, m cols
            // QR of M^T: M^T = Q * R where Q(n, k), R(k, m)
            // Then M = R^T * Q^T and A[site] = Q^T (right-isometric), absorbed R^T into left

            // Gram-Schmidt on columns of M^T (= rows of M as column vectors)
            std::vector<std::vector<Complex>> cols(m, std::vector<Complex>(n));
            for (int a = 0; a < m; a++) {
                for (int j = 0; j < n; j++) {
                    cols[a][j] = hA[a * n + j];
                }
            }

            std::vector<std::vector<Complex>> Q_cols(k, std::vector<Complex>(n, make_complex(0.0, 0.0)));
            std::vector<std::vector<Complex>> R_mat(k, std::vector<Complex>(m, make_complex(0.0, 0.0)));

            int actual_k = 0;
            for (int a = 0; a < m && actual_k < k; a++) {
                std::vector<Complex> v = cols[a];

                // Orthogonalize against previous Q vectors
                for (int j = 0; j < actual_k; j++) {
                    // r = <Q_j, v>
                    Complex r = make_complex(0.0, 0.0);
                    for (int i = 0; i < n; i++) {
                        r.x += Q_cols[j][i].x * v[i].x + Q_cols[j][i].y * v[i].y;
                        r.y += Q_cols[j][i].x * v[i].y - Q_cols[j][i].y * v[i].x;
                    }
                    R_mat[j][a] = r;
                    for (int i = 0; i < n; i++) {
                        v[i].x -= r.x * Q_cols[j][i].x - r.y * Q_cols[j][i].y;
                        v[i].y -= r.x * Q_cols[j][i].y + r.y * Q_cols[j][i].x;
                    }
                }

                // Compute norm
                double nrm = 0.0;
                for (int i = 0; i < n; i++) {
                    nrm += v[i].x * v[i].x + v[i].y * v[i].y;
                }
                nrm = std::sqrt(nrm);

                if (nrm > 1e-14) {
                    R_mat[actual_k][a] = make_complex(nrm, 0.0);
                    for (int i = 0; i < n; i++) {
                        Q_cols[actual_k][i] = make_complex(v[i].x / nrm, v[i].y / nrm);
                    }
                    actual_k++;
                }
            }

            // Now M = R^T * Q where Q is (actual_k, n) and R is (actual_k, m)
            // M[a, j] = sum_q R[q, a] * Q[q, j]
            // A_site[site] = Q (shape: actual_k, d*Db -> actual_k, d, Db)
            // A_site[site-1] = A_old[site-1] * R^T (absorb into left)

            // Store Q as new site tensor (right-isometric)
            // Note: actual_k should equal Da in normal cases
            std::vector<Complex> hA_new(actual_k * n, make_complex(0.0, 0.0));
            for (int q = 0; q < actual_k; q++) {
                for (int j = 0; j < n; j++) {
                    hA_new[q * n + j] = Q_cols[q][j];
                }
            }

            HIP_CHECK(hipFree(d_mps[site]));
            HIP_CHECK(hipMalloc(&d_mps[site], actual_k * n * sizeof(Complex)));
            HIP_CHECK(hipMemcpy(d_mps[site], hA_new.data(), actual_k * n * sizeof(Complex), hipMemcpyHostToDevice));

            // Absorb R^T into left neighbor
            // A_new[site-1][a_LL, s, q] = sum_a A_old[site-1][a_LL, s, a] * R[q, a]^T
            //                            = sum_a A_old[site-1][a_LL, s, a] * R_mat[q][a]
            // Wait, R^T[a, q] = R[q, a], so:
            // A_new[a_LL, s, q] = sum_a A_old[a_LL, s, a] * R^T[a, q]
            //                   = sum_a A_old[a_LL, s, a] * R_mat[q][a]

            int D_LL = bond_dims[site - 1];
            int left_old_size = D_LL * d * Da;
            std::vector<Complex> hA_left(left_old_size);
            HIP_CHECK(hipMemcpy(hA_left.data(), d_mps[site - 1], left_old_size * sizeof(Complex), hipMemcpyDeviceToHost));

            int left_new_size = D_LL * d * actual_k;
            std::vector<Complex> hA_left_new(left_new_size, make_complex(0.0, 0.0));

            for (int aLL = 0; aLL < D_LL; aLL++) {
                for (int s = 0; s < d; s++) {
                    for (int q = 0; q < actual_k; q++) {
                        Complex sum = make_complex(0.0, 0.0);
                        for (int a = 0; a < Da; a++) {
                            Complex Av = hA_left[aLL * (d * Da) + s * Da + a];
                            Complex Rv = R_mat[q][a];
                            sum.x += Av.x * Rv.x - Av.y * Rv.y;
                            sum.y += Av.x * Rv.y + Av.y * Rv.x;
                        }
                        hA_left_new[aLL * (d * actual_k) + s * actual_k + q] = sum;
                    }
                }
            }

            HIP_CHECK(hipFree(d_mps[site - 1]));
            HIP_CHECK(hipMalloc(&d_mps[site - 1], left_new_size * sizeof(Complex)));
            HIP_CHECK(hipMemcpy(d_mps[site - 1], hA_left_new.data(), left_new_size * sizeof(Complex), hipMemcpyHostToDevice));

            // Bond dimension stays the same (actual_k should equal Da)
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
                        // After optimizing [site, site+1] with move_right=false,
                        // A[site+1] is right-isometric. Update R[site+1] for next step.
                        envs->update_right_env(site + 1, d_mps, mpo);
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
    //
    // Uses GPU rocsolver_zgesvd with correct col-major layout handling.
    // Row-major theta (D_L*d, d*D_R) = col-major (d*D_R, D_L*d).
    // We SVD the col-major version: M = U * S * Vt where M is (d*D_R, D_L*d).
    // Then in row-major: M^T = Vt^H * S * U^H, so:
    //   U_rowmaj = Vt^H (first D_new columns = conjugate of first D_new rows of Vt)
    //   Vt_rowmaj = U^H (first D_new rows = conjugate of first D_new columns of U)
    // -----------------------------------------------------------------------
    void update_mps_with_svd(int site, Complex* d_theta, bool move_right) {
        int D_L = bond_dims[site];
        int D_M = bond_dims[site + 1];
        int D_R = bond_dims[site + 2];

        int m_row = D_L * d;   // row-major rows
        int n_row = d * D_R;   // row-major cols

        // For col-major SVD: m_col = n_row = d*D_R, n_col = m_row = D_L*d
        int m_col = n_row;
        int n_col = m_row;
        int k = std::min(m_col, n_col);

        int ldu = m_col;
        int ldv = k;

        Complex* d_U_col;
        Complex* d_Vt_col;
        double* d_S;
        double* d_E;
        int* d_info;

        HIP_CHECK(hipMalloc(&d_U_col, ldu * k * sizeof(Complex)));
        HIP_CHECK(hipMalloc(&d_Vt_col, ldv * n_col * sizeof(Complex)));
        HIP_CHECK(hipMalloc(&d_S, k * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_E, k * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_info, sizeof(int)));

        // Copy theta (SVD overwrites input)
        Complex* d_theta_copy;
        HIP_CHECK(hipMalloc(&d_theta_copy, m_col * n_col * sizeof(Complex)));
        HIP_CHECK(hipMemcpy(d_theta_copy, d_theta, m_col * n_col * sizeof(Complex), hipMemcpyDeviceToDevice));

        // SVD of col-major (m_col, n_col) matrix with lda = m_col
        rocsolver_zgesvd(rb_handle,
                        rocblas_svect_singular, rocblas_svect_singular,
                        m_col, n_col,
                        (rocblas_double_complex*)d_theta_copy, m_col,
                        d_S,
                        (rocblas_double_complex*)d_U_col, ldu,
                        (rocblas_double_complex*)d_Vt_col, ldv,
                        d_E, rocblas_outofplace, d_info);
        HIP_CHECK(hipDeviceSynchronize());

        int h_info;
        HIP_CHECK(hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost));
        if (h_info != 0) {
            std::cerr << "SVD failed with info=" << h_info << " at site " << site << std::endl;
        }

        // Get singular values
        std::vector<double> h_S(k);
        HIP_CHECK(hipMemcpy(h_S.data(), d_S, k * sizeof(double), hipMemcpyDeviceToHost));

        // Truncate
        int D_new = std::min({D_M, k, max_D});
        int num_sv = std::min(D_new, k);

        // Download SVD results to CPU for correct reconstruction
        std::vector<Complex> hU_col(ldu * k), hVt_col(ldv * n_col);
        HIP_CHECK(hipMemcpy(hU_col.data(), d_U_col, ldu * k * sizeof(Complex), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hVt_col.data(), d_Vt_col, ldv * n_col * sizeof(Complex), hipMemcpyDeviceToHost));

        // Col-major SVD: M_col = U_col * S * Vt_col
        // M_col is (m_col, n_col) = (d*D_R, D_L*d)
        // U_col is (m_col, k) = (d*D_R, k), element [i,j] at hU_col[i + j*ldu]
        // Vt_col is (k, n_col) = (k, D_L*d), element [i,j] at hVt_col[i + j*ldv]
        //
        // Our row-major matrix is M_row = M_col^T, so:
        // M_row = (Vt_col^H * S * U_col^H)^T... no, simpler:
        // M_row[a, b] = M_col[b, a] = sum_j U_col[b, j] * S[j] * Vt_col[j, a]
        //
        // So M_row[a, b] = sum_j conj(U_col[b, j])^* nope...
        // Actually M_col[b, a] = sum_j U_col[b,j] * S[j] * Vt_col[j, a]
        // This gives: M_row = Vt_col^T * S * U_col^T
        // i.e., in row-major SVD: U_row = conj(Vt_col)^T and Vt_row = conj(U_col)^T
        //
        // Simpler: M_row = Vt_col^T * S * U_col^T
        // where Vt_col^T is (n_col, k) = (D_L*d, k) <- this is U_row
        // and U_col^T is (k, m_col) = (k, d*D_R) <- this is Vt_row
        //
        // A_left[a, s, j] = U_row[a*d+s, j] = Vt_col^T[a*d+s, j] = Vt_col[j, a*d+s]
        // A_right[j, s, b] = S[j] * Vt_row[j, s*D_R+b] = S[j] * U_col^T[j, s*D_R+b] = S[j] * U_col[s*D_R+b, j]

        // Build A_left: shape (D_L, d, D_new) row-major
        int left_size = D_L * d * D_new;
        std::vector<Complex> hA_left(left_size, make_complex(0.0, 0.0));

        for (int a = 0; a < D_L; a++) {
            for (int s = 0; s < d; s++) {
                for (int j = 0; j < num_sv; j++) {
                    // U_row[a*d+s, j] = Vt_col[j, a*d+s]
                    // Vt_col element [j, a*d+s] is at hVt_col[j + (a*d+s)*ldv]
                    Complex val = hVt_col[j + (a * d + s) * ldv];
                    if (move_right) {
                        // Left-canonical: A_left = U_row (no S)
                        hA_left[a * (d * D_new) + s * D_new + j] = val;
                    } else {
                        // Right-canonical: A_left = U_row * S
                        hA_left[a * (d * D_new) + s * D_new + j] = make_complex(
                            val.x * h_S[j], val.y * h_S[j]);
                    }
                }
            }
        }

        // Build A_right: shape (D_new, d, D_R) row-major
        int right_size = D_new * d * D_R;
        std::vector<Complex> hA_right(right_size, make_complex(0.0, 0.0));

        for (int j = 0; j < num_sv; j++) {
            for (int s = 0; s < d; s++) {
                for (int b = 0; b < D_R; b++) {
                    // Vt_row[j, s*D_R+b] = U_col[s*D_R+b, j]
                    // U_col element [s*D_R+b, j] is at hU_col[(s*D_R+b) + j*ldu]
                    Complex val = hU_col[(s * D_R + b) + j * ldu];
                    if (move_right) {
                        // Left-canonical: A_right = S * Vt_row
                        hA_right[j * (d * D_R) + s * D_R + b] = make_complex(
                            val.x * h_S[j], val.y * h_S[j]);
                    } else {
                        // Right-canonical: A_right = Vt_row (no S)
                        hA_right[j * (d * D_R) + s * D_R + b] = val;
                    }
                }
            }
        }

        bond_dims[site + 1] = D_new;

        // Upload new tensors
        HIP_CHECK(hipFree(d_mps[site]));
        HIP_CHECK(hipFree(d_mps[site + 1]));
        HIP_CHECK(hipMalloc(&d_mps[site], left_size * sizeof(Complex)));
        HIP_CHECK(hipMalloc(&d_mps[site + 1], right_size * sizeof(Complex)));
        HIP_CHECK(hipMemcpy(d_mps[site], hA_left.data(), left_size * sizeof(Complex), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_mps[site + 1], hA_right.data(), right_size * sizeof(Complex), hipMemcpyHostToDevice));

        // Cleanup
        HIP_CHECK(hipFree(d_U_col));
        HIP_CHECK(hipFree(d_Vt_col));
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
