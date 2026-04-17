"""Validate challenge_mpos.h MPO builders against direct-Hamiltonian ED.

Builds the same MPO layout as challenge_mpos.h in numpy, contracts it into
the full H for small L, and compares against H built directly from spin operators.

Usage:
    python3 verify_challenge_mpos.py

Exits 0 on success, 1 on mismatch.
"""
from __future__ import annotations

import numpy as np

# Spin-1/2 operators.
Sp = np.array([[0, 1], [0, 0]], dtype=float)
Sm = np.array([[0, 0], [1, 0]], dtype=float)
Sz = np.array([[0.5, 0], [0, -0.5]], dtype=float)
Id2 = np.eye(2)


# ------------------------------------------------------------------
# J1-J2 Heisenberg
# ------------------------------------------------------------------
def build_j1j2_mpo_numpy(L: int, J1: float, J2: float) -> list[np.ndarray]:
    """Builds MPO matching challenge_mpos.h build_j1j2_mpo.

    Returns list of W[site], each shape (D, d, d, D) in (w, s, sp, wp) order.
    """
    d = 2
    D = 11
    Ws: list[np.ndarray] = []

    Sp_op, Sm_op, Sz_op, Id_op = Sp, Sm, Sz, Id2

    def mk_bulk() -> np.ndarray:
        W = np.zeros((D, d, d, D))
        # row 0
        W[0, :, :, 0] = Id_op
        W[0, :, :, 1] = Sp_op
        W[0, :, :, 2] = Sm_op
        W[0, :, :, 3] = Sz_op
        W[0, :, :, 4] = Sp_op
        W[0, :, :, 5] = Sm_op
        W[0, :, :, 6] = Sz_op
        # NN close at col 10
        W[1, :, :, 10] = 0.5 * J1 * Sm_op
        W[2, :, :, 10] = 0.5 * J1 * Sp_op
        W[3, :, :, 10] = J1 * Sz_op
        # NNN identity pass 4->7, 5->8, 6->9
        W[4, :, :, 7] = Id_op
        W[5, :, :, 8] = Id_op
        W[6, :, :, 9] = Id_op
        # NNN close at col 10
        W[7, :, :, 10] = 0.5 * J2 * Sm_op
        W[8, :, :, 10] = 0.5 * J2 * Sp_op
        W[9, :, :, 10] = J2 * Sz_op
        # row 10 identity accumulator
        W[10, :, :, 10] = Id_op
        return W

    for site in range(L):
        Wbulk = mk_bulk()
        if site == 0:
            W = np.zeros_like(Wbulk)
            W[0, :, :, :] = Wbulk[0, :, :, :]
        elif site == L - 1:
            W = np.zeros_like(Wbulk)
            W[:, :, :, D - 1] = Wbulk[:, :, :, D - 1]
        else:
            W = Wbulk
        Ws.append(W)
    return Ws


def build_j1j2_hamiltonian_direct(L: int, J1: float, J2: float) -> np.ndarray:
    """Direct H via Kronecker products for spin-1/2 Heisenberg J1-J2 chain."""
    dim = 2 ** L

    def single_site(op: np.ndarray, site: int) -> np.ndarray:
        mats = [Id2] * L
        mats[site] = op
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    def pair(op_a: np.ndarray, op_b: np.ndarray, i: int, j: int) -> np.ndarray:
        return single_site(op_a, i) @ single_site(op_b, j)

    H = np.zeros((dim, dim))
    for i in range(L):
        # J1 NN
        j = i + 1
        if j < L:
            H += 0.5 * J1 * pair(Sp, Sm, i, j)
            H += 0.5 * J1 * pair(Sm, Sp, i, j)
            H += J1 * pair(Sz, Sz, i, j)
        # J2 NNN
        j = i + 2
        if j < L:
            H += 0.5 * J2 * pair(Sp, Sm, i, j)
            H += 0.5 * J2 * pair(Sm, Sp, i, j)
            H += J2 * pair(Sz, Sz, i, j)
    return H


def mpo_to_hamiltonian(Ws: list[np.ndarray]) -> np.ndarray:
    """Contract MPO list into full H.

    L[0]=1 picks row 0, R[D-1]=1 picks col D-1.
    """
    L = len(Ws)
    D = Ws[0].shape[0]
    d = Ws[0].shape[1]
    # Start with left boundary: shape (D,)
    Lvec = np.zeros(D); Lvec[0] = 1
    Rvec = np.zeros(D); Rvec[D - 1] = 1

    # Build total H by tracing the MPO.
    # H(s1..sL, sp1..spL) = L_w0 * W[0](w0,s1,sp1,w1) * W[1](w1,s2,sp2,w2) * ... * R_wL
    # Contract from left to right.
    # State: tensor T[w, s1..sk, sp1..spk], starts as T[w] = Lvec[w].
    T = Lvec.copy()  # shape (D,)
    for site, W in enumerate(Ws):
        # T has shape (D, d^site, d^site)  — physical indices already contracted in
        # For simplicity, build gradually keeping explicit per-site axes then reshape.
        if site == 0:
            # T shape (D,)
            # Combine with W[w, s, sp, wp] -> T'[wp, s, sp] = sum_w T[w] * W[w, s, sp, wp]
            T = np.einsum("w,wsSp->Sps", T.reshape(-1) if T.ndim == 0 else T, W,
                          optimize=True)
            # Hmm, cleaner: use explicit contractions.
            T = np.einsum("w,wsap->psa", Lvec, W)
            # T shape (wp, sp, s) ... easier to keep as (wp, {s_bra multi-index}, {sp_ket multi-index})
            # Let me restart with simpler shape tracking.
            break
    # Restart: do everything in a single einsum with reshaping.
    ket_shape = [d] * L
    bra_shape = [d] * L
    # Final tensor: (s1..sL, sp1..spL) → reshape to (d^L, d^L)
    # Build via left-to-right contraction with explicit indexing on D.
    # Use flat approach: keep intermediate T[w, bra_multi, ket_multi].
    bra_dim = 1
    ket_dim = 1
    T = Lvec.reshape(D, 1, 1)  # (D, 1, 1)
    for site, W in enumerate(Ws):
        # W: (D, d, d, D)  indices (w, s, sp, wp)
        # T: (D, bra_dim, ket_dim) -> (D, a, b)
        # We want T_new[q, (a, s), (b, p)] = sum_w T[w, a, b] * W[w, s, p, q]
        # Unique labels to avoid einsum collision.
        T_new = np.einsum("wab,wspq->qasbp",
                          T, W.reshape(D, d, d, D), optimize=True)
        T_new = T_new.reshape(D, bra_dim * d, ket_dim * d)
        T = T_new
        bra_dim *= d
        ket_dim *= d
    # Final: contract with Rvec over leftmost axis.
    H_mpo = np.einsum("qbk,q->bk", T, Rvec)
    return H_mpo


def verify_j1j2(L: int, J1: float, J2: float, tol: float = 1e-10) -> None:
    H_direct = build_j1j2_hamiltonian_direct(L, J1, J2)
    Ws = build_j1j2_mpo_numpy(L, J1, J2)
    H_mpo = mpo_to_hamiltonian(Ws)
    err = np.max(np.abs(H_direct - H_mpo))
    e0_direct = float(np.linalg.eigvalsh(H_direct)[0])
    e0_mpo = float(np.linalg.eigvalsh(H_mpo)[0])
    print(f"  J1-J2 L={L:2d} J1={J1:.2f} J2={J2:.2f}: "
          f"|H_mpo - H_direct|_max = {err:.2e}  E0_direct = {e0_direct:.10f}  "
          f"E0_mpo = {e0_mpo:.10f}")
    assert err < tol, f"H mismatch {err}"
    assert abs(e0_direct - e0_mpo) < tol


# ------------------------------------------------------------------
# 2-leg Heisenberg ladder (d=4 supersite)
# ------------------------------------------------------------------
# Basis: s = 2*s1 + s2 where s1,s2 in {0=up, 1=down}.
def make_ladder_ops() -> dict[str, np.ndarray]:
    S1p = np.kron(Sp, Id2)
    S1m = np.kron(Sm, Id2)
    S1z = np.kron(Sz, Id2)
    S2p = np.kron(Id2, Sp)
    S2m = np.kron(Id2, Sm)
    S2z = np.kron(Id2, Sz)
    Id4 = np.eye(4)
    return dict(S1p=S1p, S1m=S1m, S1z=S1z, S2p=S2p, S2m=S2m, S2z=S2z, Id4=Id4)


def build_ladder_mpo_numpy(L: int, J_leg: float, J_rung: float) -> list[np.ndarray]:
    ops = make_ladder_ops()
    S1p, S1m, S1z = ops["S1p"], ops["S1m"], ops["S1z"]
    S2p, S2m, S2z = ops["S2p"], ops["S2m"], ops["S2z"]
    Id4 = ops["Id4"]
    d = 4
    D = 8

    # H_rung = J_r * (0.5*(S1p S2m + S1m S2p) + S1z S2z)
    H_rung = J_rung * (0.5 * (S1p @ S2m + S1m @ S2p) + S1z @ S2z)

    def mk_bulk() -> np.ndarray:
        W = np.zeros((D, d, d, D))
        W[0, :, :, 0] = Id4
        W[0, :, :, 1] = S1p
        W[0, :, :, 2] = S1m
        W[0, :, :, 3] = S1z
        W[0, :, :, 4] = S2p
        W[0, :, :, 5] = S2m
        W[0, :, :, 6] = S2z
        W[0, :, :, 7] = H_rung
        W[1, :, :, 7] = 0.5 * J_leg * S1m
        W[2, :, :, 7] = 0.5 * J_leg * S1p
        W[3, :, :, 7] = J_leg * S1z
        W[4, :, :, 7] = 0.5 * J_leg * S2m
        W[5, :, :, 7] = 0.5 * J_leg * S2p
        W[6, :, :, 7] = J_leg * S2z
        W[7, :, :, 7] = Id4
        return W

    Ws: list[np.ndarray] = []
    for site in range(L):
        Wbulk = mk_bulk()
        if site == 0:
            W = np.zeros_like(Wbulk)
            W[0, :, :, :] = Wbulk[0, :, :, :]
        elif site == L - 1:
            W = np.zeros_like(Wbulk)
            W[:, :, :, D - 1] = Wbulk[:, :, :, D - 1]
        else:
            W = Wbulk
        Ws.append(W)
    return Ws


def build_ladder_hamiltonian_direct(L_rungs: int, J_leg: float, J_rung: float) -> np.ndarray:
    """Direct H for 2-leg ladder: spins on (tau, i), tau in {0,1}, i in 0..L-1.

    Flattens site-major (tau slowest): supersite i has spins (tau=0,i) tensor (tau=1,i).
    Basis index for supersite i = 2*s1 + s2, which matches kron(S_leg0, S_leg1) ordering.

    Full-chain basis: kron_{i=0..L-1} supersite_i (each d=4).
    """
    L = L_rungs
    dim = 4 ** L

    def super_op(op4: np.ndarray, site: int) -> np.ndarray:
        mats = [np.eye(4)] * L
        mats[site] = op4
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    ops = make_ladder_ops()
    S1p, S1m, S1z = ops["S1p"], ops["S1m"], ops["S1z"]
    S2p, S2m, S2z = ops["S2p"], ops["S2m"], ops["S2z"]

    H = np.zeros((dim, dim))
    # Rungs: on-site J_r (S1.S2)
    H_rung_site = J_rung * (0.5 * (S1p @ S2m + S1m @ S2p) + S1z @ S2z)
    for i in range(L):
        H += super_op(H_rung_site, i)
    # Legs: J_leg * S1.S1 between supersites i and i+1, and S2.S2 similarly
    for i in range(L - 1):
        for (A, B) in [(S1p, S1m), (S1m, S1p), (S1z, S1z),
                       (S2p, S2m), (S2m, S2p), (S2z, S2z)]:
            coeff = 0.5 * J_leg if (A is S1p or A is S1m or A is S2p or A is S2m) else J_leg
            H += coeff * super_op(A, i) @ super_op(B, i + 1)
    return H


def verify_ladder(L_rungs: int, J_leg: float, J_rung: float, tol: float = 1e-10) -> None:
    H_direct = build_ladder_hamiltonian_direct(L_rungs, J_leg, J_rung)
    Ws = build_ladder_mpo_numpy(L_rungs, J_leg, J_rung)
    H_mpo = mpo_to_hamiltonian(Ws)
    err = np.max(np.abs(H_direct - H_mpo))
    e0_direct = float(np.linalg.eigvalsh(H_direct)[0])
    e0_mpo = float(np.linalg.eigvalsh(H_mpo)[0])
    print(f"  ladder L_rungs={L_rungs:2d} J_leg={J_leg:.2f} J_rung={J_rung:.2f}: "
          f"|H_mpo - H_direct|_max = {err:.2e}  E0_direct = {e0_direct:.10f}  "
          f"E0_mpo = {e0_mpo:.10f}")
    assert err < tol, f"H mismatch {err}"
    assert abs(e0_direct - e0_mpo) < tol


# ------------------------------------------------------------------
# J1-J2-J3 Heisenberg
# ------------------------------------------------------------------
def build_j1j2j3_mpo_numpy(L: int, J1: float, J2: float, J3: float) -> list[np.ndarray]:
    """Builds MPO matching challenge_mpos.h build_j1j2j3_mpo (D=20)."""
    d = 2
    D = 20
    Sp_op, Sm_op, Sz_op, Id_op = Sp, Sm, Sz, Id2

    def mk_bulk() -> np.ndarray:
        W = np.zeros((D, d, d, D))
        # row 0 starts everything
        W[0, :, :, 0] = Id_op
        # NN
        W[0, :, :, 1] = Sp_op
        W[0, :, :, 2] = Sm_op
        W[0, :, :, 3] = Sz_op
        # NNN
        W[0, :, :, 4] = Sp_op
        W[0, :, :, 5] = Sm_op
        W[0, :, :, 6] = Sz_op
        # NNNN
        W[0, :, :, 10] = Sp_op
        W[0, :, :, 11] = Sm_op
        W[0, :, :, 12] = Sz_op
        # NN close
        W[1, :, :, 19] = 0.5 * J1 * Sm_op
        W[2, :, :, 19] = 0.5 * J1 * Sp_op
        W[3, :, :, 19] = J1 * Sz_op
        # NNN pass + close
        W[4, :, :, 7] = Id_op
        W[5, :, :, 8] = Id_op
        W[6, :, :, 9] = Id_op
        W[7, :, :, 19] = 0.5 * J2 * Sm_op
        W[8, :, :, 19] = 0.5 * J2 * Sp_op
        W[9, :, :, 19] = J2 * Sz_op
        # NNNN pass1 + pass2 + close
        W[10, :, :, 13] = Id_op
        W[11, :, :, 14] = Id_op
        W[12, :, :, 15] = Id_op
        W[13, :, :, 16] = Id_op
        W[14, :, :, 17] = Id_op
        W[15, :, :, 18] = Id_op
        W[16, :, :, 19] = 0.5 * J3 * Sm_op
        W[17, :, :, 19] = 0.5 * J3 * Sp_op
        W[18, :, :, 19] = J3 * Sz_op
        # accumulator
        W[19, :, :, 19] = Id_op
        return W

    Ws: list[np.ndarray] = []
    for site in range(L):
        Wbulk = mk_bulk()
        if site == 0:
            W = np.zeros_like(Wbulk)
            W[0, :, :, :] = Wbulk[0, :, :, :]
        elif site == L - 1:
            W = np.zeros_like(Wbulk)
            W[:, :, :, D - 1] = Wbulk[:, :, :, D - 1]
        else:
            W = Wbulk
        Ws.append(W)
    return Ws


def build_j1j2j3_hamiltonian_direct(L: int, J1: float, J2: float, J3: float) -> np.ndarray:
    dim = 2 ** L

    def single_site(op: np.ndarray, site: int) -> np.ndarray:
        mats = [Id2] * L
        mats[site] = op
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    def pair(op_a: np.ndarray, op_b: np.ndarray, i: int, j: int) -> np.ndarray:
        return single_site(op_a, i) @ single_site(op_b, j)

    H = np.zeros((dim, dim))
    for i in range(L):
        for (dr, J) in [(1, J1), (2, J2), (3, J3)]:
            j = i + dr
            if j < L:
                H += 0.5 * J * pair(Sp, Sm, i, j)
                H += 0.5 * J * pair(Sm, Sp, i, j)
                H += J * pair(Sz, Sz, i, j)
    return H


def verify_j1j2j3(L: int, J1: float, J2: float, J3: float, tol: float = 1e-10) -> None:
    H_direct = build_j1j2j3_hamiltonian_direct(L, J1, J2, J3)
    Ws = build_j1j2j3_mpo_numpy(L, J1, J2, J3)
    H_mpo = mpo_to_hamiltonian(Ws)
    err = np.max(np.abs(H_direct - H_mpo))
    e0_direct = float(np.linalg.eigvalsh(H_direct)[0])
    e0_mpo = float(np.linalg.eigvalsh(H_mpo)[0])
    print(f"  J1-J2-J3 L={L:2d} J1={J1:.2f} J2={J2:.2f} J3={J3:.2f}: "
          f"|H_mpo - H_direct|_max = {err:.2e}  E0_direct = {e0_direct:.10f}  "
          f"E0_mpo = {e0_mpo:.10f}")
    assert err < tol, f"H mismatch {err}"
    assert abs(e0_direct - e0_mpo) < tol


if __name__ == "__main__":
    print("Verifying J1-J2 Heisenberg MPO ...")
    verify_j1j2(L=4, J1=1.0, J2=0.0)   # degenerates to Heisenberg
    verify_j1j2(L=4, J1=1.0, J2=0.5)   # frustrated (Majumdar-Ghosh nearby)
    verify_j1j2(L=6, J1=1.0, J2=0.5)
    verify_j1j2(L=8, J1=1.0, J2=0.5)
    verify_j1j2(L=6, J1=1.0, J2=0.4)   # non-MG frustrated
    verify_j1j2(L=8, J1=1.0, J2=0.4)

    print("Verifying 2-leg ladder MPO ...")
    verify_ladder(L_rungs=2, J_leg=1.0, J_rung=1.0)   # 4 spins: 2x2 ladder
    verify_ladder(L_rungs=3, J_leg=1.0, J_rung=1.0)
    verify_ladder(L_rungs=4, J_leg=1.0, J_rung=1.0)
    verify_ladder(L_rungs=3, J_leg=1.0, J_rung=2.0)   # rung-dominated

    print("Verifying J1-J2-J3 Heisenberg MPO ...")
    verify_j1j2j3(L=4, J1=1.0, J2=0.5, J3=0.0)  # degenerates to J1-J2
    verify_j1j2j3(L=6, J1=1.0, J2=0.5, J3=0.0)
    verify_j1j2j3(L=6, J1=1.0, J2=0.4, J3=0.2)
    verify_j1j2j3(L=8, J1=1.0, J2=0.4, J3=0.2)
    verify_j1j2j3(L=8, J1=1.0, J2=0.5, J3=0.25)

    print("All MPO checks passed.")
