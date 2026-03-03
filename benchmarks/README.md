# DMRG Benchmarks

This directory contains publication-quality benchmarks for comparing PDMRG and A2DMRG implementations against reference quimb DMRG.

## Benchmarks Overview

| Benchmark | Model | Physics | Parameters | Tests |
|-----------|-------|---------|------------|-------|
| `heisenberg_benchmark.py` | Heisenberg spin chain | Quantum magnetism | L=12, D=20 | PDMRG/A2DMRG (1,2,4,8 procs) |
| `josephson_benchmark.py` | Josephson junction array | Superconducting circuits | L=20, D=50, complex128 | PDMRG/A2DMRG (1,2,4,8 procs) |

---

## Heisenberg Model Benchmark

### Model Description

The spin-1/2 Heisenberg XXZ chain with open boundary conditions:

$$H = J \sum_{i=1}^{L-1} \left( S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + \Delta S_i^z S_{i+1}^z \right)$$

This is a standard benchmark for DMRG algorithms with well-known exact solutions for small systems and excellent convergence properties.

### Parameters
- **L = 12**: System size (sites)
- **J = 1.0**: Exchange coupling
- **Δ = 1.0**: Anisotropy (isotropic Heisenberg)
- **Bond dimension = 20**: Sufficient for exact ground state at this size
- **Tolerance = 10⁻¹²**: Convergence criterion

### Running
```bash
cd dmrg-implementations
mpirun -np 4 python benchmarks/heisenberg_benchmark.py
```

---

## Josephson Junction Array Benchmark

### Physical Motivation

Josephson junction arrays are fundamental building blocks of superconducting quantum computing platforms. Understanding their ground state properties is essential for:

- **Transmon qubit design**: The transmon operates in the regime E_J >> E_C where charge noise is suppressed [1,2]
- **Fluxonium qubits**: Arrays of junctions create the superinductance needed for fluxonium [3,4]
- **Protected qubits**: Understanding many-body effects in junction arrays enables fault-tolerant qubit designs [5]
- **Quantum phase transitions**: The array exhibits a superconductor-insulator transition controlled by E_J/E_C [6,7]

### Model Hamiltonian

The 1D Josephson junction array in the charge basis [8,9]:

$$H = E_C \sum_i (n_i - n_g)^2 - E_J \sum_{\langle i,j \rangle} \cos(\phi_i - \phi_j + \Phi_{ext})$$

where:
- **n_i**: Cooper pair number operator on island i (integer eigenvalues)
- **φ_i**: Superconducting phase operator (conjugate to n_i: [φ_i, n_j] = iδ_ij)
- **E_C = e²/2C**: Charging energy (capacitive cost of adding a Cooper pair)
- **E_J**: Josephson coupling energy (tunneling amplitude)
- **n_g**: Offset charge (gate voltage induced)
- **Φ_ext**: External magnetic flux threading the array

In the charge basis, the phase operators become raising/lowering operators:
$$e^{i\phi}|n\rangle = |n+1\rangle, \quad e^{-i\phi}|n\rangle = |n-1\rangle$$

This naturally requires **complex128** arithmetic when Φ_ext ≠ 0.

### Computational Challenge

The Josephson junction array presents several computational challenges that make it an ideal DMRG benchmark:

1. **Complex amplitudes**: External flux breaks time-reversal symmetry, requiring complex128
2. **Large local Hilbert space**: Each site has d = 2n_max + 1 states (we use n_max=2, so d=5)
3. **Non-integrable**: No exact solution for general parameters
4. **Strong correlations**: Near the phase transition, entanglement grows significantly
5. **Physical relevance**: Direct application to superconducting qubit simulation

### Benchmark Parameters

We use parameters representative of realistic superconducting circuits:

| Parameter | Value | Physical Interpretation |
|-----------|-------|------------------------|
| L | 20 | Number of junctions (scalable) |
| E_C | 1.0 | Charging energy (energy unit) |
| E_J | 2.0 | Josephson energy (E_J/E_C = 2, intermediate regime) |
| n_max | 2 | Charge truncation (d=5 local states) |
| Φ_ext | π/4 | External flux (ensures complex128) |
| Bond dim | 50 | MPS bond dimension |
| n_g | 0 | Gate charge offset |

The ratio **E_J/E_C = 2** places the system in the intermediate coupling regime where neither charge nor phase fluctuations dominate, maximizing entanglement and computational difficulty [6].

### Running
```bash
# Single benchmark
cd dmrg-implementations
mpirun -np 4 python benchmarks/josephson_benchmark.py

# Full scaling test (1, 2, 4, 8 cores)
python benchmarks/josephson_benchmark.py --scaling
```

### Expected Output

```
Josephson Junction Array Benchmark
===================================
Parameters: L=20, E_J/E_C=2.0, n_max=2, d=5, bond_dim=50
External flux: Φ_ext = π/4 (complex128)

Reference (quimb DMRG2): E = -XX.XXXXXXXXXX

Scaling Results:
  np=1: E = -XX.XXXXXXXXXX, ΔE = X.XXe-XX, time = XX.XXs
  np=2: E = -XX.XXXXXXXXXX, ΔE = X.XXe-XX, time = XX.XXs
  np=4: E = -XX.XXXXXXXXXX, ΔE = X.XXe-XX, time = XX.XXs
  np=8: E = -XX.XXXXXXXXXX, ΔE = X.XXe-XX, time = XX.XXs
```

---

## Literature References

### Josephson Junction Physics

[1] J. Koch et al., "Charge-insensitive qubit design derived from the Cooper pair box," 
    Phys. Rev. A 76, 042319 (2007). 
    https://doi.org/10.1103/PhysRevA.76.042319
    *Foundational transmon paper establishing E_J >> E_C regime.*

[2] A. Blais et al., "Circuit quantum electrodynamics," 
    Rev. Mod. Phys. 93, 025005 (2021).
    https://doi.org/10.1103/RevModPhys.93.025005
    *Comprehensive review of superconducting circuit QED.*

[3] V. E. Manucharyan et al., "Fluxonium: Single Cooper-Pair Circuit Free of Charge Offsets," 
    Science 326, 113-116 (2009).
    https://doi.org/10.1126/science.1175552
    *Introduction of fluxonium qubit using junction arrays.*

[4] L. B. Nguyen et al., "High-Coherence Fluxonium Qubit," 
    Phys. Rev. X 9, 041041 (2019).
    https://doi.org/10.1103/PhysRevX.9.041041
    *State-of-the-art fluxonium with junction array superinductance.*

[5] A. Gyenis et al., "Experimental Realization of a Protected Superconducting Circuit Derived from the 0-π Qubit,"
    PRX Quantum 2, 010339 (2021).
    https://doi.org/10.1103/PRXQuantum.2.010339
    *Protected qubits using junction arrays.*

### Quantum Phase Transitions in Junction Arrays

[6] R. Fazio and H. van der Zant, "Quantum phase transitions and vortex dynamics in superconducting networks,"
    Phys. Rep. 355, 235-334 (2001).
    https://doi.org/10.1016/S0370-1573(01)00022-9
    *Comprehensive review of quantum phase transitions in junction arrays.*

[7] M.-S. Choi, J. Yi, M. Y. Choi, J. Choi, and S.-I. Lee,
    "Quantum phase transitions in Josephson-junction chains,"
    Phys. Rev. B 57, R716 (1998).
    https://doi.org/10.1103/PhysRevB.57.R716
    *DMRG study of superconductor-insulator transition.*

### DMRG for Superconducting Circuits

[8] E. Jeckelmann, "Dynamical density-matrix renormalization-group method,"
    Phys. Rev. B 66, 045114 (2002).
    https://doi.org/10.1103/PhysRevB.66.045114
    *DMRG methodology applicable to bosonic/charge systems.*

[9] U. Schollwöck, "The density-matrix renormalization group in the age of matrix product states,"
    Ann. Phys. 326, 96-192 (2011).
    https://doi.org/10.1016/j.aop.2010.09.012
    *Definitive MPS/DMRG review.*

### Parallel DMRG

[10] E. M. Stoudenmire and S. R. White, "Real-space parallel density matrix renormalization group,"
     Phys. Rev. B 87, 155137 (2013).
     https://doi.org/10.1103/PhysRevB.87.155137
     *The PDMRG algorithm implemented here.*

---

## Validation Criteria

All benchmarks must satisfy:

1. **Accuracy**: |ΔE| < 10⁻¹⁰ compared to quimb DMRG2 reference
2. **Reproducibility**: Results consistent across multiple runs
3. **Scaling**: Energy unchanged (within tolerance) regardless of processor count
4. **Dtype preservation**: Complex128 maintained throughout for Josephson benchmark

---

## Directory Structure

```
benchmarks/
├── README.md                    # This file
├── heisenberg_benchmark.py      # Heisenberg spin chain benchmark
├── josephson_benchmark.py       # Josephson junction array benchmark  
└── results/                     # Benchmark output files
    ├── heisenberg_results.json
    └── josephson_results.json
```

---

## Citation

If you use these benchmarks in your research, please cite:

```bibtex
@article{pdmrg_a2dmrg_2026,
  title={Parallel Density Matrix Renormalization Group Algorithms for Superconducting Circuit Simulation},
  author={...},
  journal={...},
  year={2026}
}
```
