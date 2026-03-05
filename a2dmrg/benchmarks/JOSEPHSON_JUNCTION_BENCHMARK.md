# Josephson Junction Array Benchmark

## Overview

This benchmark evaluates parallel DMRG implementations on a **Josephson Junction Array** (JJA) model, a physically realistic system from superconducting quantum computing with intrinsically **complex coefficients**. The model exhibits rich quantum phase transitions and has been extensively studied both experimentally and theoretically.

## Physical Background

### What is a Josephson Junction?

A Josephson junction consists of two superconducting electrodes separated by a thin insulating barrier. The junction exhibits the **Josephson effect** [1,2]:
- Cooper pairs can tunnel coherently across the barrier
- The supercurrent depends on the phase difference: I = I_c sin(φ)
- Energy is stored as E_J cos(φ)

When multiple junctions are coupled in an array, the system exhibits collective quantum behavior including:
- Superconductor-insulator quantum phase transitions
- Charge-vortex duality
- Quantum interference effects

### The Bose-Hubbard Model for JJAs

Josephson junction arrays map onto the **Bose-Hubbard model** [3,4], where:
- **Bosons** = Cooper pairs (charge 2e)
- **Lattice sites** = superconducting islands
- **Hopping** = Josephson tunneling between islands
- **On-site interaction** = charging energy of each island

This mapping was established in seminal work by Fisher et al. [5] and has been validated extensively in experiments [6,7].

## The Hamiltonian

We implement the **1D Bose-Hubbard model with external flux**:

```
H = -E_J Σ_i (b†_i b_{i+1} e^{iφ} + h.c.) + E_C Σ_i (n_i - n̄)²
```

Where:
- **E_J**: Josephson coupling energy (controls tunneling strength)
- **E_C**: Charging energy (Coulomb cost of adding a Cooper pair)
- **b†_i, b_i**: Boson creation/annihilation operators
- **n_i = b†_i b_i**: Number operator
- **φ = φ_ext**: External magnetic flux (in units where flux quantum = 2π)
- **n̄**: Background charge (offset, typically 0)

### Why External Flux Creates Complex Coefficients

The **Peierls substitution** [8] incorporates magnetic flux by multiplying hopping terms by a phase factor:

```
-E_J (b†_i b_{i+1}) → -E_J (b†_i b_{i+1} e^{iφ})
```

For non-zero φ (e.g., φ = π/4), this creates **complex hopping amplitudes**, making the Hamiltonian Hermitian but not real-symmetric. This is critical for testing numerical implementations that must handle complex arithmetic correctly.

### Physical Interpretation

- **φ = 0**: Standard Bose-Hubbard model (real coefficients)
- **φ = π**: Fully frustrated array (still real due to gauge choice)
- **φ = π/4, π/3, etc.**: Partially frustrated, genuinely complex

The external flux can be realized experimentally by:
1. Threading magnetic flux through a ring geometry
2. Applying a transverse magnetic field to a ladder geometry
3. Using synthetic gauge fields in cold atom simulators [9]

## Quantum Phase Diagram

The ratio **E_J/E_C** controls the quantum phase transition [5,10]:

```
E_J/E_C >> 1: Superfluid phase (delocalized Cooper pairs)
E_J/E_C << 1: Mott insulator phase (localized charge)
E_J/E_C ~ 1:  Quantum critical region
```

Our benchmark uses **E_J/E_C = 2**, placing the system in the interesting crossover regime where quantum fluctuations are significant but superfluidity still dominates.

## Why This is a Good DMRG Benchmark

### 1. Complex Arithmetic Requirement
The external flux creates genuinely complex matrix elements. This tests whether implementations correctly handle:
- Complex tensor contractions
- Hermitian (not symmetric) eigenvalue problems
- Complex conjugation in bra-ket contractions

### 2. Non-trivial Local Hilbert Space
With n_max = 2, each site has d = 5 states: |0⟩, |1⟩, |2⟩, |3⟩, |4⟩
This is larger than spin-1/2 (d=2) or spin-1 (d=3), increasing computational cost.

### 3. Competing Energy Scales
The interplay between E_J (delocalizing) and E_C (localizing) creates non-trivial ground state entanglement that requires significant bond dimension to capture.

### 4. Physical Relevance
Josephson junction arrays are directly relevant to:
- Superconducting qubit architectures [11,12]
- Quantum annealing devices
- Topological quantum computing proposals

## Benchmark Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| L | 20 | Chain length (number of junctions) |
| D | 50 | Bond dimension |
| n_max | 2 | Maximum boson number per site |
| d | 5 | Local Hilbert space dimension (2×n_max+1) |
| E_J | 1.0 | Josephson energy (sets energy scale) |
| E_C | 0.5 | Charging energy |
| E_J/E_C | 2.0 | In superfluid-dominated regime |
| φ_ext | π/4 | External flux (creates complex coefficients) |
| dtype | complex128 | Required for correct physics |

### Convergence Criteria
- Energy tolerance: 1e-10 (relative to reference)
- Maximum sweeps: 20
- Reference: Serial quimb DMRG2

## Expected Behavior

1. **All parallel runs must match serial reference** to ~1e-10
2. **Speedup scaling** should be observed with increasing np
3. **Complex arithmetic overhead** is ~2× compared to real

## Key References

1. **Josephson, B.D.** (1962). "Possible new effects in superconductive tunnelling." *Physics Letters* 1(7): 251-253. [Original Josephson effect prediction]

2. **Anderson, P.W. & Rowell, J.M.** (1963). "Probable Observation of the Josephson Superconducting Tunneling Effect." *Physical Review Letters* 10(6): 230.

3. **Fazio, R. & van der Zant, H.** (2001). "Quantum phase transitions and vortex dynamics in superconducting networks." *Physics Reports* 355(4): 235-334. [Comprehensive review of JJA physics]

4. **Bruder, C., Fazio, R., & Schön, G.** (2005). "The Bose-Hubbard model: from Josephson junction arrays to optical lattices." *Annalen der Physik* 14(9-10): 566-577.

5. **Fisher, M.P.A., Weichman, P.B., Grinstein, G., & Fisher, D.S.** (1989). "Boson localization and the superfluid-insulator transition." *Physical Review B* 40(1): 546. [Seminal paper on superfluid-insulator transition]

6. **van der Zant, H.S.J., Fritschy, F.C., Elion, W.J., Geerligs, L.J., & Mooij, J.E.** (1992). "Field-induced superconductor-to-insulator transitions in Josephson-junction arrays." *Physical Review Letters* 69(20): 2971.

7. **Chow, E., Delsing, P., & Haviland, D.B.** (1998). "Length-scale dependence of the superconductor-to-insulator quantum phase transition in one dimension." *Physical Review Letters* 81(1): 204.

8. **Peierls, R.** (1933). "Zur Theorie des Diamagnetismus von Leitungselektronen." *Zeitschrift für Physik* 80(11-12): 763-791. [Peierls substitution for magnetic fields]

9. **Dalibard, J., Gerbier, F., Juzeliūnas, G., & Öhberg, P.** (2011). "Colloquium: Artificial gauge potentials for neutral atoms." *Reviews of Modern Physics* 83(4): 1523.

10. **Kühner, T.D. & Monien, H.** (1998). "Phases of the one-dimensional Bose-Hubbard model." *Physical Review B* 58(22): R14741. [DMRG study of 1D Bose-Hubbard]

11. **Koch, J., Yu, T.M., Gambetta, J., Houck, A.A., Schuster, D.I., Majer, J., Blais, A., Devoret, M.H., Girvin, S.M., & Schoelkopf, R.J.** (2007). "Charge-insensitive qubit design derived from the Cooper pair box." *Physical Review A* 76(4): 042319. [Transmon qubit]

12. **Kjaergaard, M., Schwartz, M.E., Braumüller, J., et al.** (2020). "Superconducting Qubits: Current State of Play." *Annual Review of Condensed Matter Physics* 11: 369-395.

## DMRG Studies of Bose-Hubbard Models

The Bose-Hubbard model has been extensively studied with DMRG:

- **White, S.R.** (1992). "Density matrix formulation for quantum renormalization groups." *Physical Review Letters* 69: 2863. [Original DMRG paper]

- **Kühner, T.D., White, S.R., & Monien, H.** (2000). "One-dimensional Bose-Hubbard model with nearest-neighbor interaction." *Physical Review B* 61(18): 12474. [Extended Bose-Hubbard with DMRG]

- **Ejima, S., Fehske, H., & Gebhard, F.** (2011). "Dynamic properties of the one-dimensional Bose-Hubbard model." *EPL (Europhysics Letters)* 93(3): 30002.

## Implementation Notes

### MPO Construction

The Hamiltonian is constructed as a Matrix Product Operator (MPO) with bond dimension D_MPO = 4:

```
W[i] = | I          0      0    0   |
       | b†e^{iφ}   0      0    0   |
       | b e^{-iφ}  0      0    0   |
       | E_C(n-n̄)² -E_J b -E_J b†  I |
```

The complex phase factors e^{±iφ} appear in the off-diagonal hopping terms.

### Verification

Before running benchmarks, verify that:
1. MPO dtype is complex128
2. MPO matrices have non-zero imaginary parts
3. Energy is real (imaginary part < 1e-14)

---

*Document created for A2DMRG paper benchmarks*
*Last updated: 2026-02-17*
