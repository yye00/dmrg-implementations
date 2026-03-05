#!/usr/bin/env python3
"""Quick check: does swapping phys indices affect Heisenberg (real ops)?"""
import numpy as np

# Heisenberg operators
Sx = np.array([[0, 0.5], [0.5, 0]], dtype='complex128')
Sy = np.array([[0, -0.5j], [0.5j, 0]], dtype='complex128')
Sz = np.array([[0.5, 0], [0, -0.5]], dtype='complex128')
I = np.eye(2, dtype='complex128')

print("Operator symmetry check:")
print(f"  Sx symmetric: {np.allclose(Sx, Sx.T)}")
print(f"  Sy symmetric: {np.allclose(Sy, Sy.T)}")
print(f"  Sz symmetric: {np.allclose(Sz, Sz.T)}")

# Sy is NOT symmetric! Sy^T = -Sy
print(f"  Sy: {Sy}")
print(f"  Sy^T: {Sy.T}")
print(f"  Sy is anti-symmetric: {np.allclose(Sy, -Sy.T)}")

# But Sy*Sy is symmetric
SySy = Sy @ Sy
print(f"  Sy*Sy symmetric: {np.allclose(SySy, SySy.T)}")
print(f"  Sy*Sy = {SySy}")

# In Heisenberg MPO, the coupling is Sx*Sx + Sy*Sy + Sz*Sz
# Each term has the product structure: op_L * op_R
# For Sy: term is Sy_L * Sy_R = (-Sy_L^T) * (-Sy_R^T) = Sy_L^T * Sy_R^T
# So transposing BOTH operators gives the same result.
# The Heisenberg H is: sum_<ij> S_i . S_j = sum_<ij> Sx_i*Sx_j + Sy_i*Sy_j + Sz_i*Sz_j

# Since S.S = Sx*Sx + Sy*Sy + Sz*Sz, and
# Sx_i^T * Sx_j^T = Sx_i*Sx_j (Sx symmetric)
# Sy_i^T * Sy_j^T = (-Sy_i)*(-Sy_j) = Sy_i*Sy_j (both anti-symmetric, product unchanged)
# Sz_i^T * Sz_j^T = Sz_i*Sz_j (Sz symmetric)
# So transposing all operators doesn't change the Hamiltonian!

print("\nHeisenberg should be unaffected by the swap.")
print("The bug only manifests for operators where op != op^T")
print("which is the case for exp_iphi and exp_miphi with flux phase.")
print("(Without flux, exp_iphi^T = exp_miphi, so the h.c. term compensates.)")

# Let's verify: with flux, are the coupling terms symmetric under transpose?
phi_ext = np.pi / 4
flux_phase = np.exp(1j * phi_ext)
d = 3
n_max = 1
exp_iphi = np.zeros((d, d), dtype='complex128')
exp_miphi = np.zeros((d, d), dtype='complex128')
for i in range(d - 1):
    exp_iphi[i + 1, i] = 1.0 + 0j
    exp_miphi[i, i + 1] = 1.0 + 0j

# Term 1: -E_J/2 * flux_phase * exp_iphi_L * exp_miphi_R
# Term 2: -E_J/2 * conj(flux_phase) * exp_miphi_L * exp_iphi_R
# After transposing all operators:
# Term 1 becomes: -E_J/2 * flux_phase * exp_iphi_L^T * exp_miphi_R^T
#               = -E_J/2 * flux_phase * exp_miphi_L * exp_iphi_R
#               = flux_phase/conj(flux_phase) * Term 2
#               = e^{2i*phi_ext} * Term 2
# This is NOT the same as Term 1 unless phi_ext = 0.

print(f"\nWith flux phi_ext = {phi_ext}:")
print(f"  exp_iphi^T = exp_miphi: {np.allclose(exp_iphi.T, exp_miphi)}")
print(f"  Transposing swaps Term1 <-> Term2 * e^{{2i*phi_ext}}")
print(f"  This changes the Hamiltonian for phi_ext != 0!")
print(f"  e^{{2i*phi_ext}} = {np.exp(2j * phi_ext):.6f}")
