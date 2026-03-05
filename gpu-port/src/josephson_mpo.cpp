// Josephson Junction Array MPO Construction for GPU
//
// Hamiltonian:
//   H = -E_J/2 * sum_<ij> (e^{i*phi_ext} * exp_iphi_i * exp_miphi_j + h.c.)
//       + E_C * sum_i n_i^2  - mu * sum_i n_i
//
// Charge basis: states |n> for n in {-n_max, ..., +n_max}, d = 2*n_max+1
// Operators:
//   n_op|n>  = n|n>       (diagonal, integer charges)
//   e^{iphi}|n> = |n+1>   (raises charge by 1)
//   e^{-iphi}|n> = |n-1>  (lowers charge by 1)
//
// MPO bond dimension = 4 for nearest-neighbor hopping:
//   row 0: I (pass through)
//   row 1: e^{i*phi_ext} * exp_iphi  (left half of hopping)
//   row 2: e^{-i*phi_ext} * exp_miphi (left half of conjugate hopping)
//   row 3: H_local + completed terms (Hamiltonian accumulator)
//
// Transfer matrix W (bulk):
//   W[0,0] = I          W[0,1] = 0          W[0,2] = 0          W[0,3] = 0
//   W[1,0] = exp_miphi  W[1,1] = 0          W[1,2] = 0          W[1,3] = 0
//   W[2,0] = exp_iphi   W[2,1] = 0          W[2,2] = 0          W[2,3] = 0
//   W[3,0] = H_onsite   W[3,1] = a*exp_iphi W[3,2] = a'*exp_mi  W[3,3] = I
//
// where a = -E_J/2 * e^{i*phi_ext}, a' = -E_J/2 * e^{-i*phi_ext}

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

using Complex = hipDoubleComplex;

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                  << ": " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

inline Complex make_complex(double re, double im) {
    return hipDoubleComplex{re, im};
}

// ============================================================================
// Josephson Junction MPO
// ============================================================================

class JosephsonMPO {
private:
    int L, d, D_mpo, n_max;
    double E_J, E_C, mu, phi_ext;
    std::vector<Complex*> d_mpo;
    std::vector<int> left_dims, right_dims;

public:
    JosephsonMPO(int chain_length, int max_charge = 2,
                 double josephson_energy = 1.0, double charging_energy = 0.5,
                 double chemical_potential = 0.0, double external_flux = M_PI/4.0)
        : L(chain_length), n_max(max_charge),
          E_J(josephson_energy), E_C(charging_energy),
          mu(chemical_potential), phi_ext(external_flux)
    {
        d = 2 * n_max + 1;  // local Hilbert space dimension
        D_mpo = 4;          // MPO bond dimension for nearest-neighbor

        left_dims.resize(L);
        right_dims.resize(L);
        d_mpo.resize(L);

        left_dims[0] = 1;
        right_dims[L-1] = 1;
        for (int i = 1; i < L; i++) left_dims[i] = D_mpo;
        for (int i = 0; i < L-1; i++) right_dims[i] = D_mpo;

        build_mpo_gpu();
    }

    ~JosephsonMPO() {
        for (auto& p : d_mpo) if (p) HIP_CHECK(hipFree(p));
    }

    void build_mpo_gpu() {
        // Build charge basis operators
        // n_op: diagonal matrix with entries {-n_max, ..., +n_max}
        std::vector<Complex> n_op(d * d, make_complex(0.0, 0.0));
        for (int i = 0; i < d; i++) {
            double charge = (double)(i - n_max);
            n_op[i * d + i] = make_complex(charge, 0.0);
        }

        // n^2 operator
        std::vector<Complex> n2_op(d * d, make_complex(0.0, 0.0));
        for (int i = 0; i < d; i++) {
            double charge = (double)(i - n_max);
            n2_op[i * d + i] = make_complex(charge * charge, 0.0);
        }

        // e^{iphi}: raises charge by 1 -> exp_iphi[i+1, i] = 1
        std::vector<Complex> exp_iphi(d * d, make_complex(0.0, 0.0));
        for (int i = 0; i < d - 1; i++) {
            exp_iphi[(i + 1) * d + i] = make_complex(1.0, 0.0);
        }

        // e^{-iphi}: lowers charge by 1 -> exp_miphi[i, i+1] = 1
        std::vector<Complex> exp_miphi(d * d, make_complex(0.0, 0.0));
        for (int i = 0; i < d - 1; i++) {
            exp_miphi[i * d + (i + 1)] = make_complex(1.0, 0.0);
        }

        // Identity
        std::vector<Complex> eye(d * d, make_complex(0.0, 0.0));
        for (int i = 0; i < d; i++) {
            eye[i * d + i] = make_complex(1.0, 0.0);
        }

        // On-site Hamiltonian: E_C * n^2 - mu * n
        std::vector<Complex> H_onsite(d * d, make_complex(0.0, 0.0));
        for (int i = 0; i < d; i++) {
            double charge = (double)(i - n_max);
            H_onsite[i * d + i] = make_complex(
                E_C * charge * charge - mu * charge, 0.0);
        }

        // Flux phase factors
        double cos_phi = cos(phi_ext);
        double sin_phi = sin(phi_ext);
        Complex flux_phase = make_complex(cos_phi, sin_phi);     // e^{i*phi_ext}
        Complex flux_conj  = make_complex(cos_phi, -sin_phi);    // e^{-i*phi_ext}

        // Coupling coefficients: -E_J/2 * e^{i*phi_ext} and conjugate
        Complex alpha_coup = make_complex(-E_J/2.0 * cos_phi, -E_J/2.0 * sin_phi);
        Complex alpha_conj = make_complex(-E_J/2.0 * cos_phi,  E_J/2.0 * sin_phi);

        // Build W tensors for each site
        // Storage: row-major W[wl, s, sp, wr]
        // index = wl * d * d * D_R + s * d * D_R + sp * D_R + wr

        for (int site = 0; site < L; site++) {
            int D_L = left_dims[site];
            int D_R = right_dims[site];
            int mpo_size = D_L * d * d * D_R;

            std::vector<Complex> h_mpo(mpo_size, make_complex(0.0, 0.0));

            auto set_op = [&](int wl, int wr, const std::vector<Complex>& op,
                              Complex coeff = make_complex(1.0, 0.0)) {
                // MPO stores W[wl, s_ket, s_bra, wr] where operator acts as
                // <s_bra|op|s_ket>. The operator matrix op[row][col] = <row|op|col>,
                // so we need: W[wl, s_ket, s_bra, wr] = op[s_bra * d + s_ket].
                for (int s = 0; s < d; s++) {
                    for (int sp = 0; sp < d; sp++) {
                        int idx = wl * d * d * D_R + s * d * D_R + sp * D_R + wr;
                        Complex val = op[sp * d + s];  // transposed: <sp|op|s>
                        // h_mpo[idx] += coeff * val
                        h_mpo[idx].x += coeff.x * val.x - coeff.y * val.y;
                        h_mpo[idx].y += coeff.x * val.y + coeff.y * val.x;
                    }
                }
            };

            if (site == 0) {
                // Left boundary: row vector (D_L=1, D_R=4)
                // This is the bottom row of the bulk transfer matrix
                // W[0,3] = H_onsite (on-site terms)
                // W[0,1] = alpha * exp_iphi (start left hopping)
                // W[0,2] = alpha_conj * exp_miphi (start conjugate hopping)
                // W[0,0] is not needed at boundary -- but we need to match
                // Actually for left boundary we take row 3 of the bulk matrix:
                //   [H_onsite, alpha*exp_iphi, alpha'*exp_miphi, I]
                set_op(0, 0, H_onsite);
                set_op(0, 1, exp_iphi, alpha_coup);
                set_op(0, 2, exp_miphi, alpha_conj);
                set_op(0, 3, eye);
            } else if (site == L - 1) {
                // Right boundary: column vector (D_L=4, D_R=1)
                // This is the top column of the bulk matrix:
                //   [I; exp_miphi; exp_iphi; 0]
                set_op(0, 0, eye);
                set_op(1, 0, exp_miphi);
                set_op(2, 0, exp_iphi);
                // row 3 col 0 = 0 (no on-site at right boundary)
                // Actually we need H_onsite at the right boundary too
                // The standard convention is row 3 has H_onsite (for the last site)
                // but in the column-vector convention, it goes unused.
                // For correct energy, we add H_onsite to row 3:
                set_op(3, 0, H_onsite);
            } else {
                // Bulk: 4x4 transfer matrix
                // Row 0 (pass through): W[0,0] = I
                set_op(0, 0, eye);
                // Row 1 (complete hopping left): W[1,0] = exp_miphi
                set_op(1, 0, exp_miphi);
                // Row 2 (complete conjugate hopping): W[2,0] = exp_iphi
                set_op(2, 0, exp_iphi);
                // Row 3 (Hamiltonian row):
                //   W[3,0] = H_onsite
                //   W[3,1] = alpha * exp_iphi
                //   W[3,2] = alpha_conj * exp_miphi
                //   W[3,3] = I
                set_op(3, 0, H_onsite);
                set_op(3, 1, exp_iphi, alpha_coup);
                set_op(3, 2, exp_miphi, alpha_conj);
                set_op(3, 3, eye);
            }

            HIP_CHECK(hipMalloc(&d_mpo[site], mpo_size * sizeof(Complex)));
            HIP_CHECK(hipMemcpy(d_mpo[site], h_mpo.data(), mpo_size * sizeof(Complex),
                               hipMemcpyHostToDevice));
        }
    }

    Complex* get_mpo(int site) { return d_mpo[site]; }
    int get_left_dim(int site) { return left_dims[site]; }
    int get_right_dim(int site) { return right_dims[site]; }
    int get_phys_dim() const { return d; }
    int get_length() const { return L; }
    int get_mpo_dim() const { return D_mpo; }
};
