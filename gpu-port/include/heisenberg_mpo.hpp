#pragma once

#include "dmrg_types.hpp"

// Build Heisenberg spin-1/2 MPO for chain of length L
Tensor4D<Complex> build_heisenberg_mpo(int L);

// Get exact ground state energy for comparison
double heisenberg_exact_energy(int L);
