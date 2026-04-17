// Shared challenge MPO builders (header-only).
// Built for the three GPU binaries (dmrg-gpu, dmrg2-gpu, pdmrg-gpu).
//
// Storage convention (matches existing Heisenberg / TFIM / Josephson builders):
//   W[w, s, sp, wp] stored as W[w + s*D_mpo + sp*D_mpo*d + wp*D_mpo*d*d]
// Left boundary L[0,0,0] = 1 picks row 0 of W[0].
// Right boundary R[0,D_mpo-1,0] = 1 picks col D_mpo-1 of W[L-1].

#ifndef CHALLENGE_MPOS_H
#define CHALLENGE_MPOS_H

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace challenge_mpos {

// ============================================================================
// J1-J2 Heisenberg chain (real, d=2, D_mpo=11)
// ============================================================================
// H = J1 sum_i S_i . S_{i+1} + J2 sum_i S_i . S_{i+2}
//
// Automaton states (columns on each bond):
//   0  : "waiting" (identity channel, can start a pair)
//   1  : NN S+ started, closes at NEXT site with (J1/2) S-
//   2  : NN S-
//   3  : NN Sz
//   4  : NNN S+ started, needs 1 identity pass, then closes with (J2/2) S-
//   5  : NNN S-
//   6  : NNN Sz
//   7  : NNN S+, identity passed, closes at THIS site with (J2/2) S-
//   8  : NNN S-, identity passed
//   9  : NNN Sz, identity passed
//   10 : closed (accumulator, identity channel)
//
// Bulk transitions (row -> col):
//   0 -> 0 : I     | 0 -> 1..3 : S+, S-, Sz (start NN) | 0 -> 4..6 : S+, S-, Sz (start NNN)
//   1 -> 10 : (J1/2) S-   (close NN S+-)
//   2 -> 10 : (J1/2) S+   (close NN -S+)
//   3 -> 10 : J1 Sz       (close NN zz)
//   4 -> 7  : I           (NNN S+ identity pass)
//   5 -> 8  : I
//   6 -> 9  : I
//   7 -> 10 : (J2/2) S-   (close NNN S+-)
//   8 -> 10 : (J2/2) S+
//   9 -> 10 : J2 Sz
//   10 -> 10: I
//
// Frustration maximum: J2/J1 ~ 0.5 (Majumdar-Ghosh point near 0.5).
static inline void build_j1j2_mpo(int L, double J1, double J2,
                                  std::vector<double*>& h_mpo_tensors) {
    const int d = 2;
    const int D = 11;

    auto set_op = [&](double* W, int w, int wp, const double (&op)[4]) {
        for (int s = 0; s < d; s++)
            for (int sp = 0; sp < d; sp++)
                W[w + s*D + sp*D*d + wp*D*d*d] = op[sp*d + s];
    };

    // Operator matrices (stored as op[sp*d + s], i.e. column-major 2x2).
    const double Sp[4] = {0, 1, 0, 0};   // S+ : |0><1|
    const double Sm[4] = {0, 0, 1, 0};   // S- : |1><0|
    const double Sz[4] = {0.5, 0, 0, -0.5};
    const double Id[4] = {1, 0, 0, 1};
    const double J1_half_Sm[4] = {0, 0.5*J1, 0, 0};
    const double J1_half_Sp[4] = {0, 0, 0.5*J1, 0};
    const double J1_Sz[4]     = {0.5*J1, 0, 0, -0.5*J1};
    const double J2_half_Sm[4] = {0, 0.5*J2, 0, 0};
    const double J2_half_Sp[4] = {0, 0, 0.5*J2, 0};
    const double J2_Sz[4]     = {0.5*J2, 0, 0, -0.5*J2};

    for (int site = 0; site < L; site++) {
        int size = D * d * d * D;
        h_mpo_tensors[site] = new double[size]();
        double* W = h_mpo_tensors[site];

        if (site == 0) {
            // Only row 0 reachable (L picks row 0).
            set_op(W, 0, 0,  Id);
            set_op(W, 0, 1,  Sp);
            set_op(W, 0, 2,  Sm);
            set_op(W, 0, 3,  Sz);
            set_op(W, 0, 4,  Sp);
            set_op(W, 0, 5,  Sm);
            set_op(W, 0, 6,  Sz);
        } else if (site == L - 1) {
            // Only col D-1 reachable (R picks col D-1).
            set_op(W, 1, 10, J1_half_Sm);
            set_op(W, 2, 10, J1_half_Sp);
            set_op(W, 3, 10, J1_Sz);
            set_op(W, 7, 10, J2_half_Sm);
            set_op(W, 8, 10, J2_half_Sp);
            set_op(W, 9, 10, J2_Sz);
            set_op(W, 10, 10, Id);
        } else {
            // Bulk: full automaton.
            set_op(W, 0, 0,  Id);
            set_op(W, 0, 1,  Sp);
            set_op(W, 0, 2,  Sm);
            set_op(W, 0, 3,  Sz);
            set_op(W, 0, 4,  Sp);
            set_op(W, 0, 5,  Sm);
            set_op(W, 0, 6,  Sz);
            set_op(W, 1, 10, J1_half_Sm);
            set_op(W, 2, 10, J1_half_Sp);
            set_op(W, 3, 10, J1_Sz);
            set_op(W, 4, 7,  Id);
            set_op(W, 5, 8,  Id);
            set_op(W, 6, 9,  Id);
            set_op(W, 7, 10, J2_half_Sm);
            set_op(W, 8, 10, J2_half_Sp);
            set_op(W, 9, 10, J2_Sz);
            set_op(W, 10, 10, Id);
        }
    }
}

// ============================================================================
// Heisenberg 2-leg ladder via d=4 "supersite" encoding (real, d=4, D_mpo=8)
// ============================================================================
// Each MPS site carries a rung (two spin-1/2s).  Rung coupling is on-site;
// leg couplings are NN in MPS order.
//
// Physical basis (d=4):  |up,up>=0, |up,dn>=1, |dn,up>=2, |dn,dn>=3
// Tensor product: s = s1*2 + s2.
// Operators: S1^a = S^a (x) I_2,  S2^a = I_2 (x) S^a.
//
// Automaton (columns):
//   0 : waiting (I)
//   1..3 : leg-1 S+, S-, Sz started (close next site)
//   4..6 : leg-2 S+, S-, Sz started
//   7 : closed (I)
//
// On-site rung contribution goes directly to (0, 7) via H_rung operator:
//   H_rung = J_r * (S1x S2x + S1y S2y + S1z S2z)
//          = J_r * (0.5*(S1+ S2- + S1- S2+) + S1z S2z)
//
// Leg closes at next site with:
//   J_leg/2 * S1-   (closes leg-1 S+-)
//   J_leg/2 * S1+   (closes leg-1 -S+)
//   J_leg   * S1z   (closes leg-1 zz)
//   similarly for leg-2
static inline void build_ladder_mpo(int L_rungs, double J_leg, double J_rung,
                                    std::vector<double*>& h_mpo_tensors) {
    const int d = 4;
    const int D = 8;
    const int L = L_rungs;

    // Build d=4 operators row-major (op[sp*d + s]).
    auto op4 = [&](double (&out)[16], double v00, double v01, double v02, double v03,
                   double v10, double v11, double v12, double v13,
                   double v20, double v21, double v22, double v23,
                   double v30, double v31, double v32, double v33) {
        // out[sp*d + s] = M(s, sp)  (i.e. matrix element <s|M|sp>)
        // So out[sp*d + s] = value at row s, col sp.
        // I'll pass arguments as row-major M[s][sp] for readability.
        double m[16] = {v00,v01,v02,v03, v10,v11,v12,v13, v20,v21,v22,v23, v30,v31,v32,v33};
        for (int s = 0; s < d; s++)
            for (int sp = 0; sp < d; sp++)
                out[sp*d + s] = m[s*d + sp];
    };

    double Id4[16];  op4(Id4, 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);

    // Basis: 0=uu, 1=ud, 2=du, 3=dd
    // S1+ flips spin-1 from dn->up: |dn,?> -> |up,?>  i.e. 2->0, 3->1
    //   Matrix M: M[s,sp]=1 if S1+|sp>=|s>. S1+|du>=|uu>: M[0,2]=1. S1+|dd>=|ud>: M[1,3]=1.
    // S1- : 0->2, 1->3 : M[2,0]=1, M[3,1]=1
    // S1z : 0=+0.5, 1=+0.5, 2=-0.5, 3=-0.5
    // S2+ flips spin-2 dn->up: S2+|ud>=|uu>: M[0,1]=1. S2+|dd>=|du>: M[2,3]=1.
    // S2- : 0->1, 2->3 : M[1,0]=1, M[3,2]=1
    // S2z : 0=+0.5, 1=-0.5, 2=+0.5, 3=-0.5
    double S1p[16]; op4(S1p, 0,0,1,0, 0,0,0,1, 0,0,0,0, 0,0,0,0);
    double S1m[16]; op4(S1m, 0,0,0,0, 0,0,0,0, 1,0,0,0, 0,1,0,0);
    double S1z[16]; op4(S1z, 0.5,0,0,0, 0,0.5,0,0, 0,0,-0.5,0, 0,0,0,-0.5);
    double S2p[16]; op4(S2p, 0,1,0,0, 0,0,0,0, 0,0,0,1, 0,0,0,0);
    double S2m[16]; op4(S2m, 0,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,1,0);
    double S2z[16]; op4(S2z, 0.5,0,0,0, 0,-0.5,0,0, 0,0,0.5,0, 0,0,0,-0.5);

    // Scale helpers.
    auto scale = [&](double* dst, const double* src, double a) {
        for (int i = 0; i < d*d; i++) dst[i] = a * src[i];
    };
    auto add = [&](double* dst, const double* a, const double* b) {
        for (int i = 0; i < d*d; i++) dst[i] = a[i] + b[i];
    };
    auto matmul = [&](double* dst, const double* A, const double* B) {
        // Both stored as O[sp*d + s]; matrix mul as M_dst[s,sp] = sum_k M_A[s,k]*M_B[k,sp]
        // => dst[sp*d+s] = sum_k A[k*d+s] * B[sp*d+k]
        for (int s = 0; s < d; s++)
            for (int sp = 0; sp < d; sp++) {
                double acc = 0;
                for (int k = 0; k < d; k++) acc += A[k*d + s] * B[sp*d + k];
                dst[sp*d + s] = acc;
            }
    };

    // Rung H_rung = J_r * (0.5 (S1+ S2- + S1- S2+) + S1z S2z)
    double H_rung[16] = {0};
    {
        double tmp1[16], tmp2[16], acc[16];
        // S1+ S2-
        matmul(tmp1, S1p, S2m);
        // S1- S2+
        matmul(tmp2, S1m, S2p);
        add(acc, tmp1, tmp2);
        // acc = S1+ S2- + S1- S2+, scale by 0.5
        double half_acc[16];
        scale(half_acc, acc, 0.5);
        // S1z S2z
        double sz_sz[16];
        matmul(sz_sz, S1z, S2z);
        double sum[16];
        add(sum, half_acc, sz_sz);
        scale(H_rung, sum, J_rung);
    }

    double J_leg_half_S1m[16]; scale(J_leg_half_S1m, S1m, 0.5*J_leg);
    double J_leg_half_S1p[16]; scale(J_leg_half_S1p, S1p, 0.5*J_leg);
    double J_leg_S1z[16];      scale(J_leg_S1z,      S1z, J_leg);
    double J_leg_half_S2m[16]; scale(J_leg_half_S2m, S2m, 0.5*J_leg);
    double J_leg_half_S2p[16]; scale(J_leg_half_S2p, S2p, 0.5*J_leg);
    double J_leg_S2z[16];      scale(J_leg_S2z,      S2z, J_leg);

    auto set_op16 = [&](double* W, int w, int wp, const double* op) {
        for (int s = 0; s < d; s++)
            for (int sp = 0; sp < d; sp++)
                W[w + s*D + sp*D*d + wp*D*d*d] = op[sp*d + s];
    };

    for (int site = 0; site < L; site++) {
        int size = D * d * d * D;
        h_mpo_tensors[site] = new double[size]();
        double* W = h_mpo_tensors[site];

        if (site == 0) {
            set_op16(W, 0, 0, Id4);
            set_op16(W, 0, 1, S1p);
            set_op16(W, 0, 2, S1m);
            set_op16(W, 0, 3, S1z);
            set_op16(W, 0, 4, S2p);
            set_op16(W, 0, 5, S2m);
            set_op16(W, 0, 6, S2z);
            set_op16(W, 0, 7, H_rung);
        } else if (site == L - 1) {
            set_op16(W, 0, 7, H_rung);
            set_op16(W, 1, 7, J_leg_half_S1m);
            set_op16(W, 2, 7, J_leg_half_S1p);
            set_op16(W, 3, 7, J_leg_S1z);
            set_op16(W, 4, 7, J_leg_half_S2m);
            set_op16(W, 5, 7, J_leg_half_S2p);
            set_op16(W, 6, 7, J_leg_S2z);
            set_op16(W, 7, 7, Id4);
        } else {
            set_op16(W, 0, 0, Id4);
            set_op16(W, 0, 1, S1p);
            set_op16(W, 0, 2, S1m);
            set_op16(W, 0, 3, S1z);
            set_op16(W, 0, 4, S2p);
            set_op16(W, 0, 5, S2m);
            set_op16(W, 0, 6, S2z);
            set_op16(W, 0, 7, H_rung);
            set_op16(W, 1, 7, J_leg_half_S1m);
            set_op16(W, 2, 7, J_leg_half_S1p);
            set_op16(W, 3, 7, J_leg_S1z);
            set_op16(W, 4, 7, J_leg_half_S2m);
            set_op16(W, 5, 7, J_leg_half_S2p);
            set_op16(W, 6, 7, J_leg_S2z);
            set_op16(W, 7, 7, Id4);
        }
    }
}

}  // namespace challenge_mpos

#endif  // CHALLENGE_MPOS_H
