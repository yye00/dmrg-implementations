#pragma once

#include <vector>

/**
 * Build real-valued Heisenberg MPO for Phase 2 multi-stream DMRG
 *
 * Hamiltonian: H = Σ_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + S^z_i S^z_{i+1})
 *
 * Physical dimension: d = 2 (spin-1/2)
 * MPO bond dimension: D_mpo = 5
 * Storage: column-major real doubles
 */

/**
 * Build Heisenberg MPO tensor for a single site
 *
 * @param site Site index (0 to L-1)
 * @param L Total chain length
 * @param h_mpo Output buffer (host memory) to store MPO tensor
 *              Size must be D_left * d * d * D_right
 *              where D_left = (site==0 ? 1 : 5), D_right = (site==L-1 ? 1 : 5)
 */
void build_heisenberg_mpo_real_site(int site, int L, double* h_mpo);

/**
 * Build Heisenberg MPO on GPU for all sites
 *
 * @param L Chain length
 * @return Vector of device pointers to MPO tensors (one per site)
 *
 * User is responsible for freeing the device memory:
 *   for (auto* ptr : mpo) hipFree(ptr);
 */
std::vector<double*> build_heisenberg_mpo_real_gpu(int L);

/**
 * Get exact ground state energy for Heisenberg chain
 *
 * @param L Chain length
 * @return Ground state energy E_0
 *
 * For L=8: E_0 ≈ -3.374931816815
 */
double heisenberg_exact_energy_real(int L);
