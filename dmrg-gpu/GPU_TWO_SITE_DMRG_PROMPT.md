# GPU-Native Two-Site DMRG (DMRG2): Extend Single-Site to Two-Site Optimization

## Objective

Extend the existing **working, verified** single-site GPU DMRG (`dmrg-gpu`) to support **two-site DMRG optimization** in a new directory `dmrg2-gpu`. Two-site DMRG optimizes a pair of adjacent MPS tensors simultaneously, which allows the bond dimension to grow adaptively and avoids local minima that trap single-site DMRG.

The existing single-site code provides the GPU infrastructure: ScalarTraits dispatch, rocBLAS GEMM contractions, Lanczos eigensolver, SVD (CPU + GPU paths), and environment updates. The two-site variant reuses all of this with targeted modifications to handle the larger optimization window.

**Correctness targets:**
- Heisenberg L=4 d=2: E = -1.616025403784 (real, matches single-site)
- Heisenberg L=8 d=2: E = -3.374932598687 (real, matches single-site)
- Josephson L=4 d=3: E = -1.053346829927 (complex128, matches single-site)
- Josephson L=6 d=3: E = -1.748843818181 (complex128, matches single-site)

**Performance expectation:** Two-site will be slower than single-site at the same chi_max (theta is d× larger), but should still be GPU-bound, not CPU-bound.

## Remote Machine — ALL Work Happens Here

**All compilation, editing, and testing MUST happen on the remote GPU machine.**
The local machine has no GPU and cannot compile HIP code.

See `CLAUDE.md` in the repository root for full remote access instructions. Quick reference:

- **SSH:** `ssh hotaisle@23.183.40.79` (passwordless, no password prompt)
- **Remote repo path:** `/home/hotaisle/dmrg-implementations/`
- **GPU code path:** `/home/hotaisle/dmrg-implementations/dmrg2-gpu/`

### Workflow

1. SSH into the remote machine
2. Edit files on the remote machine
3. Build: `cd /home/hotaisle/dmrg-implementations/dmrg2-gpu/build && cmake .. -DGPU_TARGETS=gfx942 && make -j8`
4. Test: `./dmrg2_gpu 4 16 10` (L=4, chi_max=16, 10 sweeps)
5. Commit and push from the remote machine

## Starting Point: Copy from dmrg-gpu

```bash
cp -r dmrg-gpu dmrg2-gpu
cd dmrg2-gpu
# Rename target in CMakeLists.txt from dmrg_gpu to dmrg2_gpu
# Rename test file from test_dmrg_gpu.cpp to test_dmrg2_gpu.cpp
```

Files to modify:
- `src/dmrg_gpu.h` → class definition (add two-site members, change method signatures)
- `src/dmrg_gpu_impl.h` → implementation (two-site apply_heff, SVD split, sweep logic)
- `src/test_dmrg2_gpu.cpp` → test harness (same MPO builders, adapted for two-site)
- `CMakeLists.txt` → rename target
- `src/scalar_traits.h` → **no changes needed** (reuse as-is)
- `src/dmrg_gpu.cpp` → **no changes needed** (template instantiations)

## What Changes from Single-Site to Two-Site

### 1. Theta Tensor Shape

**Single-site:** `theta[a, s, b]` with shape `(chi_L, d, chi_R)`, size = `chi_L * d * chi_R`

**Two-site:** `theta[a, s1, s2, b]` with shape `(chi_L, d, d, chi_R)`, size = `chi_L * d * d * chi_R`

Memory layout (flat C-order): `theta[a + s1*chi_L + s2*chi_L*d + b*chi_L*d*d]`

### 2. Form Theta (Combining Two Adjacent Sites)

Single-site just copies `MPS[site]` to `d_theta_`.

Two-site must contract `MPS[site]` and `MPS[site+1]` across their shared bond:

```
theta[a, s1, s2, b] = sum_c A[site][a, s1, c] * A[site+1][c, s2, b]
```

Where `c` runs over `chi_mid = bond_dims_[site+1]`.

**GPU implementation:** This is a GEMM:
```
// Reshape A[site] as (chi_L * d, chi_mid) and A[site+1] as (chi_mid, d * chi_R)
// theta = A[site] @ A[site+1]
// Result: (chi_L * d, d * chi_R) which IS (chi_L, d, d, chi_R) in memory
Traits::gemm(handle, N, N,
    chi_L * d, d * chi_R, chi_mid,    // m, n, k
    &one,
    d_mps_tensors_[site], chi_L * d,  // A, lda
    d_mps_tensors_[site+1], chi_mid,  // B, ldb
    &zero,
    d_theta_, chi_L * d);             // C, ldc
```

### 3. Apply H_eff (Two-Site) — THE CORE CHANGE

The single-site H_eff contracts `L_env × theta × W × R_env` (4 tensors).
The two-site H_eff contracts `L_env × theta × W_left × W_right × R_env` (5 tensors).

**Full contraction:**
```
result[a', s1', s2', b'] = sum_{a,s1,s2,b,w,m,n}
    L[a', w, a] * theta[a, s1, s2, b] * W_L[w, s1, s1', m] * W_R[m, s2, s2', n] * R[b', n, b]
```

Where:
- `L_env` at bond `site`: shape `(chi_L, D_mpo, chi_L)`
- `theta`: shape `(chi_L, d, d, chi_R)`
- `W_L = MPO[site]`: shape `(D_mpo, d, d, D_mpo)` — indices `(w, s1, s1', m)`
- `W_R = MPO[site+1]`: shape `(D_mpo, d, d, D_mpo)` — indices `(m, s2, s2', n)`
- `R_env` at bond `site+2`: shape `(chi_R, D_mpo, chi_R)`
- Result: shape `(chi_L, d, d, chi_R)` — same as theta

**Four-step GPU contraction (left-to-right order):**

#### Step 1: Contract L_env with theta
```
V[w, s1, a', s2, b] = sum_a L[a', w, a] * theta[a, s1, s2, b]
```

Treat theta as having shape `(chi_L, d*d*chi_R)` — i.e., contract on the `a` (chi_L) index.

This is a batched GEMM over the D_mpo `w` values:
```
For each w in [0, D):
    V[w] = L_env[w]^T @ theta_reshaped
    // L_env[w] is (chi_L, chi_L) at offset w*chi_L in env
    // theta_reshaped is (chi_L, d*d*chi_R)
    // V[w] is (chi_L, d*d*chi_R)

    m = chi_L, n = d*d*chi_R, k = chi_L
```

Use `gemm_batched` with D batches:
```
Traits::gemm_batched(handle,
    op_t, N,                          // L^T @ theta
    chi_L, d*d*chi_R, chi_L,          // m, n, k
    &one,
    batch_A (L_env pointers), chi_L*D, // lda = chi_L * D (stride between w blocks)
    batch_B (theta repeated D times), chi_L, // ldb = chi_L (theta is same for all w)
    &zero,
    batch_C (V block pointers), chi_L, // ldc = chi_L
    D);                                // batch_count
```

Result V has total size `D * chi_L * d * d * chi_R`.

**IMPORTANT:** The batch_B pointers all point to the same theta (same data, D identical pointers). The ldb stride of chi_L is the leading dimension of the (chi_L, d*d*chi_R) matrix.

#### Step 2: Contract V with W_left (absorb left MPO)
```
U[m, s1', a', s2, b] = sum_{w,s1} V[w, s1, a', s2, b] * W_L[w, s1, s1', m]
```

Reshape V from `(D*d, chi_L*d*chi_R)` by treating `(w*d+s1)` as one index and `(a', s2, b)` as the other.

This is a single large GEMM using the precomputed `W_left` matrix:
```
// V is (D*d, chi_L * d * chi_R) — rows indexed by (w*d+s1)
// W_left is (D*d, d*D) — rows (w*d+s1), cols (m*d+s1')
// U = V^T @ W_left  ... or equivalently:
// U is (chi_L*d*chi_R, d*D) — rows indexed by (a'*d*chi_R + s2*chi_R + b)

Traits::gemm(handle, N, N,
    chi_L * d * chi_R, d * D, D * d,  // m, n, k
    &one,
    V, chi_L * d * chi_R,             // lda (V stored as (D*d) × (chi_L*d*chi_R) column-major)
    W_left, D * d,                    // ldb
    &zero,
    U, chi_L * d * chi_R);            // ldc
```

Wait — V's memory layout from step 1 needs care. Each batch w produces a (chi_L, d*d*chi_R) block. These D blocks are contiguous, so V is `(D, chi_L, d, d, chi_R)` stored as D blocks of size `chi_L*d*d*chi_R`.

To use V as a `(D*d, chi_L*d*chi_R)` matrix, we need V indexed as `V[(w*d+s1), (a'*d*chi_R + s2*chi_R + b)]`. But from step 1, V is stored as `V[a' + w*(chi_L*d*d*chi_R) + s1*chi_L + s2*chi_L*d + b*chi_L*d*d]`.

This doesn't align — the `w` and `s1` indices aren't adjacent in memory. We need to **rearrange** the intermediate or adjust the contraction.

**Better approach — fuse steps 1+2 like single-site does:**

The single-site code does steps 1+2 as:
1. Batched GEMM: `V[(w*d+s), a', b] = L[w, a', a]^T * theta[s, a, b]` — D*d batches of (chi_L × chi_R) matrices
2. Dense GEMM: `U = V @ W_left` where V is `(chi_L*chi_R, D*d)` and W_left is `(D*d, d*D)`

For two-site, do the same with theta having shape `(chi_L, d*d*chi_R)` treated as d*d*chi_R "columns":

**Step 1 (batched, D*d batches):**
```
For each (w, s1) pair:
    V[w*d+s1, a', :] = L_env[w, a', :]^T @ theta[s1, :, :]

    // Here theta[s1, :, :] means: fix s1, treat remaining (chi_L × d*chi_R)
    // But theta is (chi_L, d, d, chi_R), and fixing s1 gives (chi_L, d*chi_R)
    // V[ws1] is (chi_L, d*chi_R) per batch element
```

Batched GEMM:
```
For each (w, s1):
    h_batch_A[w*d+s1] = L_env + w * chi_L          // L_env[w] block
    h_batch_B[w*d+s1] = theta + s1 * chi_L          // theta[s1] slice
    h_batch_C[w*d+s1] = V + (w*d+s1) * chi_L * d_R_block  // V block
    where d_R_block = d * chi_R

Traits::gemm_batched(handle,
    op_t, N,
    chi_L, d*chi_R, chi_L,           // m, n, k
    &one,
    batch_A, chi_L * D,              // lda (stride between L_env[w] blocks)
    batch_B, chi_L * d,              // ldb (stride between theta[s1] slices)
    &zero,
    batch_C, chi_L,                  // ldc
    D * d);                          // batch_count = D*d

// Result: V is (D*d) blocks of (chi_L × d*chi_R), total (D*d, chi_L, d, chi_R)
// As a matrix: V is (chi_L * d * chi_R, D*d) in column-major
// i.e., column ws1 of V is the (chi_L * d*chi_R) vector for that (w,s1) pair
```

**Step 2 (dense GEMM, absorb W_left):**
```
U = V @ W_left
// V: (chi_L * d * chi_R) × (D*d)  — treat as matrix
// W_left: (D*d) × (d*D)
// U: (chi_L * d * chi_R) × (d*D)

Traits::gemm(handle, N, N,
    chi_L * d * chi_R, d * D, D * d,
    &one,
    V, chi_L * d * chi_R,
    W_left[site], D * d,
    &zero,
    U, chi_L * d * chi_R);
```

**Step 3 (dense GEMM, absorb W_right):**

Now U has shape `(chi_L * d * chi_R, d * D)` where the column index is `(m*d + s1')` from step 2.

We need to contract W_right on indices `(m, s2)` and produce index `(s2', n)`. But the column index of U is `(m*d + s1')`, not `(m*d + s2)`. The index `s1'` is an *output* physical index from W_left, while `s2` is the *input* physical index consumed by W_right.

This means U's columns are indexed by `(s1', m)` — the output of W_left. Now we need to contract on `m` with W_right. The `s1'` index is a free/output index that goes straight to the result.

**Reindex:** U is `(chi_L * d * chi_R) × (d_L * D_mid)` where d_L-columns are the s1' output and D_mid-columns are the m MPO index.

We need: `Z[a', s2, b, s1', s2', n] = U[a'*d*chi_R + s2*chi_R + b, s1'*D + m] * WR[m, s2, s2', n]`

This is tricky because `m` and `s2` appear in different positions. Let me reconsider.

**Actually, let's use a different 4-step approach that maps more cleanly to GEMMs:**

### Recommended 4-Step Contraction for Two-Site H_eff

Rather than trying to share the single-site approach identically, use this order which keeps GEMM-friendly index groupings:

#### Step 1: Contract theta with R_env (batched GEMM)
```
V[n, s2, a, s1, b'] = sum_b theta[a, s1, s2, b] * R[b', n, b]
```

For each (n, s2): `V[n*d+s2] = theta[s2]^T @ R[n]`

But we need to be more careful about memory layout.

theta is `(chi_L, d, d, chi_R)` stored as `theta[a + s1*cL + s2*cL*d + b*cL*d*d]`.
R_env at bond site+2 is `(chi_R, D, chi_R)` stored as `R[b' + n*cR + b*cR*D]`.

For each (n, s2):
- theta[s2] selects a `(chi_L*d, chi_R)` submatrix: row = `a + s1*cL`, col = `b`, at offset `s2*cL*d` (stride cL*d*d between s2 slices)

Hmm, theta[s2] isn't contiguous as a `(cL*d, cR)` matrix because s1 varies faster than s2 in memory. The layout is `a + s1*cL + s2*cL*d + b*cL*d*d`. Fixing s2, the (a,s1) indices span `[s2*cL*d .. s2*cL*d + cL*d - 1]` for each b. The stride between b values is `cL*d*d`, not `cL*d`. So theta[s2] as a matrix has rows (a,s1) contiguous within a block but blocks separated by stride cL*d*d.

This is NOT a simple (cL*d × cR) matrix — it has stride cL*d*d between columns.

**This is the core difficulty of the two-site contraction on GPU.** Let me choose the approach that avoids stride issues.

### Clean 4-Step Approach (No Stride Issues)

#### Step 1: Batched GEMM — contract L_env with theta on left bond

For each (w, s1): `V[w*d+s1, :, :] = L[w]^T @ theta[s1]`

- `L[w]` is at L_env + w*cL, shape conceptually (cL, cL) with lda = cL*D
- `theta[s1]` is theta + s1*cL, shape (cL, d*cR) with ldb = cL*d

Wait — `theta[s1]` fixing s1: elements at positions `a + s1*cL + s2*cL*d + b*cL*d*d` for varying (a, s2, b). This spans `a ∈ [0,cL)`, and s2*cL*d + b*cL*d*d varies. The stride between consecutive `a` values is 1 (contiguous). The stride between consecutive `(s2,b)` "columns" needs to be cL*d (from s2 to s2+1) and cL*d*d (from b to b+1).

Treating this as a (cL, d*cR) matrix: the "column" index combines (s2, b) with total d*cR values. The stride between columns is cL*d (the s2 stride). But b's stride is cL*d*d = cL*d * d, which means b jumps d columns at a time. So the (s2, b) column layout IS contiguous: column j = s2 + b*d maps to offset s2*cL*d + b*cL*d*d = (s2 + b*d)*cL*d. NO — that's not right.

Let me re-derive. theta index formula: `theta[a + s1*cL + s2*cL*d + b*cL*d*d]`.

Fix s1. Varying (a, s2, b):
- Column index = s2*d_stride + b*b_stride where d_stride = cL*d, b_stride = cL*d*d
- Offset from theta base = s1*cL + a + s2*cL*d + b*cL*d*d

Treat as a matrix M[a, (s2,b)] where (s2,b) → s2 + b*d (d*cR total columns):
- Element M[a, s2+b*d] = theta[a + s1*cL + s2*cL*d + b*cL*d*d]
- M[a, col] at col=s2+b*d is at offset: s1*cL + a + (s2*cL*d + b*cL*d*d) = s1*cL + a + cL*d*(s2 + b*d)
- Stride between rows (a→a+1): 1
- Stride between cols (col→col+1): cL*d

So `ldb = cL*d` with the matrix starting at `theta + s1*cL`. This IS a valid column-major matrix `(cL, d*cR)` with leading dimension `cL*d`.

Similarly, L[w] starts at `L_env + w*cL`, treating L_env as (cL, D, cL) → L[w] is (cL, cL) submatrix with stride cL*D between columns → `lda = cL*D`.

So the batched GEMM is:
```
For ws = w*d + s1:
    h_A[ws] = L_env + w * cL
    h_B[ws] = theta + s1 * cL
    h_C[ws] = T1 + ws * cL * d * cR

Traits::gemm_batched(handle, op_t, N,
    cL, d*cR, cL,                   // m, n, k
    &one,
    batch_A, cL*D,                   // lda (L_env column stride)
    batch_B, cL*d,                   // ldb (theta column stride fixing s1)
    &zero,
    batch_C, cL,                     // ldc
    D*d);                            // batch_count
```

Result T1 size: `D*d * cL * d*cR`. T1 has shape `(D*d, cL, d, cR)` stored as D*d blocks of `(cL, d*cR)`.

As a matrix for step 2: T1 is `(cL*d*cR, D*d)` in column-major — column ws of T1 is the (cL, d*cR) block flattened.

#### Step 2: Dense GEMM — absorb W_left

```
T2 = T1 @ W_left[site]
// T1: (cL*d*cR, D*d), W_left: (D*d, d*D)
// T2: (cL*d*cR, d*D)

Traits::gemm(handle, N, N,
    cL * d * cR, d * D, D * d,
    &one, T1, cL * d * cR,
    W_left[site], D * d,
    &zero, T2, cL * d * cR);
```

T2 columns are indexed by `(m*d + s1')` where m is the middle MPO bond and s1' is the left output physical index. So T2 has shape `(cL*d*cR, d*D)` — rows indexed by `(a', s2, b)`, columns by `(s1', m)`.

#### Step 3: Absorb W_right via another dense GEMM

Now we need to contract on the `m` (middle MPO bond) and `s2` (right ket physical) indices with W_right, and produce `s2'` (right bra physical) and `n` (right MPO bond) indices.

T2 rows are `(a', s2, b)` = `a' + s2*cL + b*cL*d`. T2 columns are `(s1', m)` = `s1' + m*d`.

We want: `T3[a', s2', b, s1', n] = sum_{s2, m} T2[a'+ s2*cL + b*cL*d, s1' + m*d] * WR[m, s2, s2', n]`

The indices `s2` and `m` appear in different positions in T2 (s2 in row, m in column). This means we can't do a single GEMM — we'd need to transpose/reshape.

**Alternative: reshape T2 so m and s2 are together, then contract with W_right.**

T2 has shape `(cL, d_R, cR, d_L, D_mid)` where d_R=d (s2), d_L=d (s1'). We need to group `(m, s2)` together.

This gets complicated. Let's use a **loop approach** for step 3 like the single-site code does for step 3.

#### Step 3 (loop): Contract T2 columns with R_env

Actually, let's reconsider. After step 2, T2[row, col] where:
- row = `a' + s2*cL + b*cL*d` (spanning cL*d*cR values, grouped as (a', s2, b))
- col = `s1' + m*d` (spanning d*D values, grouped as (s1', m))

For each `(s1', m)` column: the rows of T2 contain `(a', s2, b)`. We still need to contract `(m, s2)` with W_right and `(b)` with R_env.

**Step 3+4 combined: Loop over (s1', s2', n) — accumulate with R_env contracted.**

For each output physical pair `(s1', s2')` and right MPO index `n`:
```
result[a', s1', s2', b'] += sum_{m, s2, b} T2[a'+s2*cL+b*cL*d, s1'+m*d] * WR[m,s2,s2',n] * R[b',n,b]
```

This is O(cL * cR² * d² * D²) per (s1', s2', n) triple — that's d²*D individual GEMMs, too many.

### Better Approach: 5-Step with Explicit W_right Absorption

Let's insert a **reshape + GEMM** to absorb W_right before contracting with R_env.

Actually, the cleanest approach is:

#### Revised Strategy: Two Batched GEMMs + Two Dense GEMMs

**Step 1: L_env × theta** (batched, D*d batches) → T1 of shape (cL*d*cR, D*d)

Same as above.

**Step 2: T1 × W_left** (dense GEMM) → T2 of shape (cL*d*cR, d*D_mid)

Same as above. Columns of T2 indexed by (s1', m).

**Step 3: Reshape T2 and contract with W_right** (dense GEMM)

We need to build a combined `W_LR` matrix that fuses `W_left` and `W_right` in a single step.

No — that changes the operator. Instead:

**Step 3: Contract T2 with W_right on (m, s2) indices**

Rewrite T2 as T2[a', s2, b, s1', m]. We need to gather the (m, s2) pair.

Reshape T2 from (cL*d*cR, d*D_mid) to (cL*cR, d, d*D_mid):
- Original row = a' + s2*cL + b*cL*d
- New indexing: outer row = a' + b*cL (cL*cR values), inner index = s2 (d values)

But s2 is in the middle of (a', s2, b), not at the end. So `a' + s2*cL + b*cL*d` means s2 varies in the "middle" — not contiguous with m.

**The fundamental issue**: in column-major layout, the indices (m, s2) that need to contract with W_right sit in different axes of T2. One is in the row index, the other in the column index.

### Final Recommended Approach

Use the **same 3-step pattern as single-site, but applied twice** — once for the left half (L_env + W_left) and once for the right half (W_right + R_env):

#### Step 1: Batched GEMM — L_env × theta (D*d batches)

Same as above. For each (w, s1):
```
T1[w*d+s1, a', (s2,b)] = L[w, a', a]^T * theta[s1, a, (s2,b)]
```

Batch setup:
```
h_A[ws] = L_env + w * cL            // L[w] submatrix
h_B[ws] = theta + s1 * cL           // theta[s1] slice
h_C[ws] = T1 + ws * cL * d * cR     // output block

gemm_batched(op_t, N, cL, d*cR, cL, ..., cL*D, cL*d, cL, D*d)
```

T1: (D*d) blocks of size (cL × d*cR). As matrix: (cL*d*cR, D*d).

#### Step 2: Dense GEMM — absorb W_left

```
T2 = T1 @ W_left[site]
// (cL*d*cR) × (D*d) @ (D*d) × (d*D) → (cL*d*cR) × (d*D)
gemm(N, N, cL*d*cR, d*D, D*d, ...)
```

T2: (cL*d*cR, d*D). Row = (a', s2, b), Col = (s1', m). Stored column-major.

#### Step 3: Batched GEMM — contract R_env on right bond (D*d batches)

Now we need to contract R_env with theta on the right bond `b`. After step 2, T2 contains the contraction of L × theta × W_L. We still need to contract with W_R and R_env on the (m, s2, b) indices.

**Key insight: extract each column of T2 (fixing s1' and m), and within that column, we have (a', s2, b). Contract on b with R_env[n] to get (a', s2, b').**

For each `(s1', m, n)` triple:
```
T3[(s1',m,n), a', s2, b'] = sum_b T2[a' + s2*cL + b*cL*d, s1' + m*d] * R[b', n, b]
```

T2 column `(s1'+m*d)` is a vector of length cL*d*cR. Reshape it as `(cL, d, cR)` indexed by (a', s2, b):
- This is a matrix `(cL*d, cR)` with lda = cL*d  ... wait, check strides.
- T2 column at col = s1'+m*d: elements at offset `(a' + s2*cL + b*cL*d) + (s1'+m*d)*cL*d*cR`
- Fixing col, row varies as a' + s2*cL + b*cL*d
- As matrix (cL, d*cR) → lda = cL ... no.
- As matrix ((cL*d), cR): row = a'+s2*cL ∈ [0, cL*d), col = b ∈ [0, cR)
  - Element offset: (a' + s2*cL) + b * cL*d = a' + s2*cL + b*cL*d ✓
  - Leading dimension = cL*d ✓

So each T2 column can be viewed as a `(cL*d, cR)` matrix with lda = cL*d.

R_env[n] is at R_env + n*cR, shape `(cR, cR)` with lda = cR*D.

```
T3 = T2_col @ R_env[n]
// (cL*d, cR) @ (cR, cR) → (cL*d, cR)

gemm(N, N, cL*d, cR, cR, ...)
```

But this is d*D*D = d*D² separate GEMMs over (s1', m, n). That's potentially large. With D=5, d=2 that's 2*25 = 50 GEMMs, each of modest size. Similar to single-site's D*d = 10 GEMMs in step 3.

Actually we can be smarter. We should **sum over m with W_right simultaneously**. Instead of looping over (s1', m, n), loop over (s1', s2', n) and accumulate:

```
For each (s1', s2', n):
    result[a', s1', s2', b'] = sum_{m} WR[m, s2, s2', n] *
        sum_b T2[a'+s2*cL+b*cL*d, s1'+m*d] * R[b', n, b]
```

This requires first doing T2_col @ R[n] for each m, then weighting by WR and summing over (m, s2). This is D*d GEMMs followed by accumulation.

**Simplest correct approach — loop over (s1', s2', n) with beta accumulation:**

```
For s1' in [0, d):
    For n in [0, D):
        For s2' in [0, d):
            // Compute sum over m, s2 of WR[m,s2,s2',n] * T2_col(s1',m)[a',s2,b] * R[n,b,b']
            // = sum_m WR_coeff(m,s2,s2',n) * sum_b T2[...,s1'+m*d] * R[n,:,:]

            beta = (first iteration) ? 0 : 1
            For m in [0, D):
                For s2 in [0, d):
                    w_coeff = WR[m, s2, s2', n]  // scalar from MPO
                    if w_coeff == 0: continue     // sparse MPO optimization

                    // T2 column (s1' + m*d): treat as (cL*d, cR), but we only want row-slice s2
                    // Row s2 of the (cL, d, cR) tensor = rows [s2*cL..(s2+1)*cL-1] of (cL*d, cR) matrix
                    // That's a (cL, cR) submatrix

                    src = T2 + (s1' + m*d) * cL*d*cR + s2*cL  // (cL, cR) with lda=cL*d

                    // result[s1', s2'] is a (cL, cR) submatrix of result
                    // result[a', s1', s2', b'] at offset: s1'*cL + s2'*cL*d in the (cL, d, d, cR) result
                    // As (cL, cR) matrix: lda = cL*d*d

                    // GEMM: result[s1',s2'][:,:]  += w_coeff * src[:,:] @ R[n][:,:]
                    gemm(N, N, cL, cR, cR,
                         &w_coeff,
                         src, cL*d,              // lda=cL*d (stride between s2 slices in T2 column)
                         R_env + n*cR, cR*D,     // R[n] with lda=cR*D
                         &beta,
                         result + s1'*cL + s2'*cL*d, cL*d*d);  // result[s1',s2'] with lda=cL*d*d

                    beta = 1;  // accumulate after first contribution
```

**This is the inner loop.** The total number of GEMMs is at most `d * D * d * D * d = d³ * D²`. For Heisenberg (d=2, D=5): 8*25 = 200 GEMMs. For Josephson (d=3, D=4): 27*16 = 432 GEMMs. Each GEMM is (cL, cR, cR) — small but GPU-accelerated.

**Sparse MPO optimization:** Most MPO entries are zero. The upper-triangular structure means roughly half the (m, s2, s2', n) combinations have WR=0. Skip them with a `w_coeff == 0` check. This can be precomputed as a sparse list of (m, s2, s2', n, coeff) tuples.

**THIS IS THE KEY COMPLEXITY DIFFERENCE FROM SINGLE-SITE.** Single-site step 3 does D*d GEMMs. Two-site step 3 does up to d²*D² GEMMs (with sparsity reducing this). If this is too slow, the alternative is to precompute a fused W_left⊗W_right tensor, but that increases memory by d².

### Alternative: Precompute Fused Two-Site MPO (Recommended for Simplicity)

Precompute `WW[site]` that fuses W_left and W_right:
```
WW[(w*d*d+s1*d+s2), (n*d*d+s1'*d+s2')] = sum_m W_L[w,s1,s1',m] * W_R[m,s2,s2',n]
```

Shape: `(D*d², d²*D)` — a matrix of size `(D*d*d, d*d*D)`.

Then the apply_heff becomes **exactly the single-site 3-step pattern** with `d` replaced by `d²`:

**Step 1 (batched, D*d² batches):**
```
For each (w, s1, s2):
    V[w*d²+s1*d+s2, a', b] = L[w, a', a]^T * theta[(s1,s2), a, b]

gemm_batched(op_t, N, cL, cR, cL, ..., cL*D, cL*d*d, cL, D*d*d)
```

Where theta[(s1,s2)] starts at `theta + (s1*d+s2)*cL` (wait: check memory layout).

theta[a, s1, s2, b] stored as `a + s1*cL + s2*cL*d + b*cL*d*d`.
Fixing (s1, s2): elements at `a + s1*cL + s2*cL*d + b*cL*d*d` for varying (a, b).
Matrix M[a, b] at base offset `s1*cL + s2*cL*d`, stride 1 between a, stride `cL*d*d` between b.
This is a `(cL, cR)` matrix with lda = cL*d*d ... NOT cL. ❌

The stride between columns (b→b+1) is `cL*d*d = cL*d²`, not `cL`. So this is NOT a standard column-major `(cL, cR)` matrix.

**We need ldb = cL*d² for theta when fixing (s1, s2).** rocBLAS supports arbitrary leading dimensions, so:

```
For each (w, s1, s2):
    ws = w*d*d + s1*d + s2
    h_A[ws] = L_env + w * cL
    h_B[ws] = theta + s1*cL + s2*cL*d  // base for (s1,s2) slice
    h_C[ws] = T1 + ws * cL * cR

gemm_batched(op_t, N,
    cL, cR, cL,
    &one,
    batch_A, cL*D,     // lda (L block spacing)
    batch_B, cL*d*d,   // ldb = cL*d² (stride between b columns in theta)
    &zero,
    batch_C, cL,       // ldc
    D*d*d);            // batch_count = D*d²
```

**CRITICAL:** The `ldb` parameter in `gemm_batched` is the leading dimension of EACH B matrix in the batch, NOT the stride between batch elements. Each B matrix is `(cL, cR)` but stored with stride `cL*d²` between columns. So `ldb = cL*d*d` is correct — it tells rocBLAS that consecutive columns of B are `cL*d²` elements apart.

Wait — but that's the leading dimension of the full theta tensor, not of the individual `(cL, cR)` slice. For `gemm_batched`, each pointer in batch_B points to a different `(k, n) = (cL, cR)` matrix. The ldb tells rocBLAS the stride between columns of that matrix. Since theta columns (varying b) are spaced `cL*d*d` apart, `ldb = cL*d*d` is correct. ✓

**Step 2 (dense GEMM, absorb fused WW):**
```
T2 = T1 @ WW[site]
// T1: (cL*cR, D*d²), WW: (D*d², d²*D)
// T2: (cL*cR, d²*D)

gemm(N, N, cL*cR, d*d*D, D*d*d, ...)
```

**Step 3 (loop of GEMMs, contract R_env):**
```
For each (s1', s2', n):
    ws_out = n*d*d + s1'*d + s2'
    beta = (first iteration) ? 0 : 1

    gemm(N, N, cL, cR, cR,
         &one,
         T2 + ws_out * cL * cR, cL,    // T2 column ws_out, (cL, cR) matrix
         R_env + n * cR, cR * D,        // R[n] submatrix
         &beta,
         result + s1'*cL + s2'*cL*d, cL*d*d);  // result[s1',s2'] submatrix, ldc=cL*d²
```

Wait, the result layout needs checking.

result[a', s1', s2', b'] stored as `a' + s1'*cL + s2'*cL*d + b'*cL*d*d`.
Fixing (s1', s2'): base at `s1'*cL + s2'*cL*d`, varying (a', b'):
- a' stride = 1, b' stride = cL*d*d = cL*d²
- As (cL, cR) matrix with ldc = cL*d² ✓

And T2 column ws_out: base at `ws_out * cL * cR`, shape (cL*cR) = (cL, cR) matrix with ldc = cL. ✓

**But the R_env contraction is over b (ket_R), not n.** Let me re-derive.

T2[row, col] where row ∈ [0, cL*cR), col ∈ [0, d²*D).
Row = a' + b*cL (if T1 blocks are (cL, cR) → a' varies fastest, then b).

Wait, T1 from step 1: each batch element produces a (cL, cR) matrix with ldc=cL. So T1 column ws = block of cL*cR elements, stored as (cL, cR) column-major: `T1[a' + b*cL + ws*cL*cR]`. Row = a' + b*cL.

After T2 = T1 @ WW: T2[row, col] where row = a' + b*cL.

T2 column ws_out: a (cL*cR) vector indexed by (a', b) with a' varying fastest. As a (cL, cR) matrix: T2_col[a', b] with lda = cL. ✓

Now the contraction with R_env on index b:
```
result[a', (s1',s2'), b'] = sum_{b, n} T2[a'+b*cL, (s1',s2')+n*d²] * R[b', n, b]
```

For each n, fixing T2 column at col = (s1'*d+s2') + n*d²:
- T2_col is (cL, cR) matrix M[a', b] with lda = cL
- R[n] is R_env + n*cR, shape (cR, cR) with lda = cR*D

```
result[a', (s1',s2'), b'] += M[a', b] * R[b', n, b] = M @ R[n]^T
```

Wait: R[b', n, b] — we're summing over b (=ket) and the result is indexed by b' (=bra).

`result[a', b'] += sum_b M[a', b] * R[b', n, b]`

R stored as `R[b' + n*cR + b*cR*D]`. R[n] at offset n*cR is `R_n[b', b]` with b' stride 1, b stride cR*D. As (cR, cR) matrix: R_n[b', b] with lda = cR*D.

We need `result = M @ R_n^T`, but R_n^T means transposing on (b', b).

Actually: `sum_b M[a', b] * R_n[b', b]` — the sum is over b. M has b as columns. R_n has b as columns too (stride cR*D). So:

`result[a', b'] = sum_b M[a', b] * R_n[b', b]`

This is `result = M @ R_n^T`. But R_n^T with the op_t flag.

**Wait, I need to reconsider.** In the single-site code, step 3 uses:
```
gemm(N, N, cL, cR, cR, ..., U, cL, R_env+wp*cR, cR*D, ...)
```

This computes `C = U @ R` where `R` is (cR, cR) at stride cR*D. The GEMM does:
- C[i,j] = sum_k U[i,k] * R[k,j]
- With R[k,j] at memory R_base + k + j*(cR*D)
- This gives C[a', b'] = sum_b U[a', b] * R_n[b, b'] which is sum over b with R indexed as (b, b').

R_n at offset n*cR: `R_n[x, y]` = R[n*cR + x + y*cR*D]. With x=b, y=b':
`R_n[b, b'] = R[n*cR + b + b'*cR*D]`

Compare to R stored as R[b' + n*cR + b*cR*D]:
`R[b', n, b]` at `b' + n*cR + b*cR*D`

Setting n fixed: R_n_raw[b', b] = R[b' + n*cR + b*cR*D]
Compare: R_n_gemm[b, b'] = R[n*cR + b + b'*cR*D]

These are NOT the same. R_n_raw[b', b] = b' + n*cR + b*cR*D vs R_n_gemm[b, b'] = n*cR + b + b'*cR*D.

R_n_raw[b', b] = R_n_gemm[b', b] + n*cR - n*cR ... no.

R_n_raw[b', b] at position b' + n*cR + b*cR*D
R_n_gemm[b, b'] at position n*cR + b + b'*cR*D

If b' = b_bra, b = b_ket:
- R_n_raw: b_bra + n*cR + b_ket*cR*D → R_n_raw[row=b_bra, col=b_ket] at lda=cR*D? No, stride between b_bra values is 1, stride between b_ket values is cR*D. So R_n_raw is (cR, cR) column-major with lda = cR*D ← wait, lda should be >= m, and m = cR, so lda = cR*D works but means the matrix has "gaps" between columns.

Actually in the GEMM call, R is passed with ldb = cR*D. The B matrix in GEMM is (k, n) = (cR, cR). Element B[k_idx, n_idx] is at B_ptr + k_idx + n_idx * ldb = R_base + n*cR + k_idx + n_idx * cR*D.

So B[k_idx, n_idx] = R[k_idx + n*cR + n_idx*cR*D]. With k_idx=b (ket, being summed) and n_idx=b' (bra, output):
B[b, b'] = R[b + n*cR + b'*cR*D]

From the stored format: R[b' + n*cR + b*cR*D]. So B[b, b'] = R[b + n*cR + b'*cR*D] ≠ R[b' + n*cR + b*cR*D] unless b=b'.

These are TRANSPOSED. The GEMM uses R as (b, b') but storage is (b', b).

So `result = U @ R_gemm` computes:
`result[a', b'] = sum_b U[a', b] * R_gemm[b, b']`

where R_gemm[b, b'] = R[b + n*cR + b'*cR*D]. But R stored as R[b', n, b] = R[b' + n*cR + b*cR*D].

So R_gemm[b, b'] = R[b, n, b'] (transposed bra↔ket). The GEMM **implicitly transposes** R_env!

This is consistent: the env convention stores (bra, mpo, ket) and the GEMM naturally reads it as (ket, bra) which gives the transpose. The physical contraction `sum_b theta[...,b] * R[b', n, b]` becomes `sum_b theta[...,b] * R_gemm[b, b']` = theta @ R_gemm. ✓

OK. So for the two-site case, the same R_env contraction works. The step 3 loop is:

```
For each (s1', s2') output pair and n (right MPO bond):
    ws_out = n*d*d + s1'*d + s2'   // column index into T2
    beta = ... // 0 for first (n=0), 1 otherwise

    gemm(N, N, cL, cR, cR,
         &one,
         T2 + ws_out * cL * cR, cL,
         R_env + n * cR, cR * D,
         &beta,
         result + (s1'*d + s2') * cL, cL * d * d);
```

Wait — we need to accumulate over n (sum over right MPO bond). The result for a given (s1', s2') output should accumulate contributions from all n values. Use beta=0 for n=0, beta=1 for n>0:

```
For s1' in [0, d):
    For s2' in [0, d):
        For n in [0, D):
            ws_out = n*d*d + s1'*d + s2'
            beta = (n == 0) ? zero : one

            gemm(N, N, cL, cR, cR,
                 &one,
                 T2 + ws_out * cL * cR, cL,
                 R_env + n * cR, cR * D,
                 &beta,
                 result + s1'*cL + s2'*cL*d, cL*d*d);
```

**Total GEMMs in step 3: d² * D.** For Heisenberg (d=2, D=5): 20. For Josephson (d=3, D=4): 36. Very manageable.

**THIS IS THE CLEAN SOLUTION.** The fused WW approach reduces the two-site apply_heff to:
- Step 1: D*d² batched GEMMs of size (cL, cR, cL)
- Step 2: 1 dense GEMM of size (cL*cR, d²*D, D*d²)
- Step 3: d²*D individual GEMMs of size (cL, cR, cR)

Compare to single-site:
- Step 1: D*d batched GEMMs of size (cL, cR, cL)
- Step 2: 1 dense GEMM of size (cL*cR, d*D, D*d)
- Step 3: D*d individual GEMMs of size (cL, cR, cR)

The only change is d → d² in the batch counts and intermediate sizes. **The structure is identical.**

### 4. SVD Split After Two-Site Optimization

After Lanczos produces optimized theta of shape `(chi_L, d, d, chi_R)`:

**Right-moving sweep (site, site+1):**
```
// Reshape theta to (chi_L * d, d * chi_R) for SVD
m_svd = chi_L * d
n_svd = d * chi_R
// theta is already in this layout: a + s1*cL in rows, s2 + b*d in columns
// ... wait, check:
// theta[a + s1*cL + s2*cL*d + b*cL*d*d]
// Row index = a + s1*cL ∈ [0, cL*d)
// Col index = s2 + b*d ∈ [0, d*cR)
// Element at (row, col) = row + col * cL*d = a + s1*cL + (s2 + b*d)*cL*d
//                        = a + s1*cL + s2*cL*d + b*cL*d*d ✓
// Leading dimension = cL*d ✓

SVD: theta(m_svd × n_svd) = U(m_svd × k) * S(k) * Vh(k × n_svd)
Truncate: k_new = min(k, chi_max, number of S[i] > 1e-14)

// U reshaped to (chi_L, d, k_new) → becomes MPS[site]
MPS[site] = U[:, :k_new]   // (chi_L * d, k_new) = (chi_L, d, k_new) ✓

// S * Vh reshaped to (k_new, d, chi_R) → becomes MPS[site+1]
MPS[site+1] = diag(S[:k_new]) @ Vh[:k_new, :]   // (k_new, d * chi_R) = (k_new, d, chi_R) ✓

bond_dims_[site+1] = k_new
```

**Left-moving sweep (site, site+1):**
```
// Same reshape: (chi_L * d, d * chi_R)
SVD: theta = U * S * Vh
Truncate to k_new

// U*S reshaped to (chi_L, d, k_new) → becomes MPS[site]
MPS[site] = U[:, :k_new] @ diag(S[:k_new])   // (chi_L * d, k_new) ✓

// Vh reshaped to (k_new, d, chi_R) → becomes MPS[site+1]
MPS[site+1] = Vh[:k_new, :]   // (k_new, d * chi_R) = (k_new, d, chi_R) ✓

bond_dims_[site+1] = k_new
```

**Key difference from single-site:** In single-site, SVD produces one tensor that stays at the current site and absorbs S into the neighbor. In two-site, SVD splits theta into two tensors that replace BOTH sites. No absorption into a third tensor is needed.

### 5. Sweep Structure

**Single-site sweeps:** Visit every site, SVD at each.

**Two-site sweeps:** Visit every PAIR of sites.

```
sweep_right():
    for site = 0 to L-2:
        form_theta(site, site+1)           // Contract MPS[site] @ MPS[site+1]
        lanczos_eigensolver(site, d_theta)  // Optimize (uses L[site], R[site+2])
        svd_split_right(site, d_theta)      // U → MPS[site], S*Vh → MPS[site+1]
        update_left_env(site)               // Build L[site+1] from L[site] and MPS[site]

sweep_left():
    for site = L-2 to 0:
        form_theta(site, site+1)
        lanczos_eigensolver(site, d_theta)
        svd_split_left(site, d_theta)       // U*S → MPS[site], Vh → MPS[site+1]
        update_right_env(site+1)            // Build R[site+1] from R[site+2] and MPS[site+1]
```

**Environment indexing:**
- `apply_heff` at pair (site, site+1) uses `L_envs_[site]` and `R_envs_[site+2]`
- After right SVD at pair (site, site+1): update `L_envs_[site+1]` using MPS[site] (now left-canonical)
- After left SVD at pair (site, site+1): update `R_envs_[site+1]` using MPS[site+1] (now right-canonical)

### 6. Precomputed Fused Two-Site MPO

For each bond (site, site+1), precompute:
```
WW[w*d²+s1*d+s2, n*d²+s1'*d+s2'] = sum_m W_L[w, s1, s1', m] * W_R[m, s2, s2', n]
```

Shape: `(D*d², d²*D)`, same as single-site `W_left` but with d→d².

**Memory:** `D² * d⁴` elements per bond. For D=5, d=2: 400 elements (3.2 KB). For D=5, d=3: 2025 elements (~16 KB). Negligible.

**Build on host, upload to GPU.** Store as `d_WW_[L-1]` (one per bond, L-1 bonds total).

Construction from the existing MPO tensors:
```cpp
// W_L = MPO[site]:   W_L[w + s1*D + s1p*D*d + m*D*d*d]
// W_R = MPO[site+1]: W_R[m + s2*D + s2p*D*d + n*D*d*d]

for (int w = 0; w < D; w++)
    for (int s1 = 0; s1 < d; s1++)
        for (int s2 = 0; s2 < d; s2++)
            for (int s1p = 0; s1p < d; s1p++)
                for (int s2p = 0; s2p < d; s2p++)
                    for (int n = 0; n < D; n++) {
                        Scalar val = zero;
                        for (int m = 0; m < D; m++)
                            val += W_L[w + s1*D + s1p*D*d + m*D*d*d]
                                 * W_R[m + s2*D + s2p*D*d + n*D*d*d];
                        int row = w*d*d + s1*d + s2;
                        int col = n*d*d + s1p*d + s2p;
                        WW[row + col * D*d*d] = val;  // column-major
                    }
```

This is a CPU loop at initialization time — O(D³ * d⁴) per bond, completely negligible.

### 7. Workspace Sizing

**Two-site theta:** `chi_max² * d²` (vs chi_max² * d for single-site)

**Lanczos vectors:** `max_iter * chi_max² * d²`

**Intermediates T1, T2:** `D * d² * chi_max² * d²`... wait, that's too big.

T1 from step 1: `D*d² * cL * cR` elements. At chi_max: `D * d² * chi_max²`.
T2 from step 2: `cL*cR * d²*D` = `chi_max² * d² * D` elements. Same size as T1.

For D=5, d=2, chi_max=64: `5 * 4 * 4096 = 81,920` doubles = 640 KB. Fine.
For D=5, d=2, chi_max=256: `5 * 4 * 65536 = 1,310,720` doubles = 10 MB. Fine on MI300X.

**Batch pointer arrays:** `D * d²` pointers (vs D*d for single-site). Still small.

**SVD workspace:** `m_svd = chi_max * d, n_svd = d * chi_max`. Same max as single-site theta_size but with d²: `chi_max² * d²`. U is `(chi_max*d, chi_max*d)` max, S is `(chi_max*d)`, Vh is `(chi_max*d, d*chi_max)`.

### 8. Class Changes Summary

```cpp
template<typename Scalar>
class DMRG2GPU {  // or keep as DMRGGPU with a mode flag
    // NEW members:
    Scalar** d_WW_;              // Fused two-site MPO, L-1 entries, each (D*d², d²*D)

    // CHANGED sizing (d → d²):
    // theta_size_max_ = chi_max_ * d_ * d_ * chi_max_
    // T1, T2 sized for D * d² * chi_max²
    // Lanczos vectors sized for theta_size_max_
    // Batch pointer arrays sized for D * d²
    // SVD workspace: m = chi_max*d, n = d*chi_max

    // NEW methods:
    void form_theta_two_site(int site);     // Contract MPS[site] @ MPS[site+1]
    void apply_heff_two_site(int site, ...);// 3-step with fused WW
    void svd_split_right(int site, ...);    // SVD and assign to both sites
    void svd_split_left(int site, ...);     // SVD and assign to both sites
    void precompute_fused_mpo();            // Build WW from MPO pairs

    // CHANGED methods:
    void sweep_left_to_right();  // Loop over pairs, not individual sites
    void sweep_right_to_left();
    void optimize_bond(int site, char direction);  // Replaces optimize_site

    // UNCHANGED methods:
    void build_initial_environments();  // Same (build all R from right)
    void update_left_env(int site);     // Same (single-site env update)
    void update_right_env(int site);    // Same (single-site env update)
    void lanczos_eigensolver(...);      // Same (just operates on larger theta)
    // All ScalarTraits, memory management, initialization
};
```

## Implementation Plan (Step by Step)

### Phase 1: Scaffolding
1. Copy `dmrg-gpu/` to `dmrg2-gpu/`
2. Rename CMake target, test file
3. Rename class to `DMRG2GPU` (or add `_two_site` suffix to methods)
4. Update workspace sizing: `theta_size_max_ = chi_max_ * d_ * d_ * chi_max_`
5. Update T1/T2 sizes, Lanczos vector sizes, batch pointer array sizes
6. Add `d_WW_` member and allocation
7. **Build and verify compilation**

### Phase 2: Fused MPO
1. Implement `precompute_fused_mpo()` on CPU, upload to GPU
2. Call it at the end of `set_mpo()`
3. **Build and verify**

### Phase 3: form_theta_two_site
1. Implement as GEMM: `MPS[site] @ MPS[site+1]`
2. **Test:** form theta, copy back to host, verify against CPU computation

### Phase 4: apply_heff_two_site
1. Implement 3-step contraction using fused WW (mirror single-site structure)
2. Step 1: batched GEMM with D*d² batches, ldb = cL*d*d
3. Step 2: dense GEMM with WW matrix
4. Step 3: loop of d²*D GEMMs with R_env
5. **Test:** build small system, compute H_eff*theta on CPU, compare GPU result

### Phase 5: SVD Split
1. Implement `svd_split_right()`: SVD of `(cL*d, d*cR)` matrix, U→MPS[site], S*Vh→MPS[site+1]
2. Implement `svd_split_left()`: U*S→MPS[site], Vh→MPS[site+1]
3. **Test:** verify MPS normalization after SVD

### Phase 6: Sweep Logic
1. Implement `sweep_left_to_right()`: loop site=0..L-2, form_theta, lanczos, svd_split_right, update_left_env
2. Implement `sweep_right_to_left()`: loop site=L-2..0, form_theta, lanczos, svd_split_left, update_right_env
3. Update `run()` to use new sweep functions
4. **Test with Heisenberg L=4:** should converge to -1.616025403784

### Phase 7: Full Validation
1. Test Heisenberg L=4, L=8 (real)
2. Test Josephson L=4, L=6 (complex128)
3. Compare energies to exact values and single-site DMRG results
4. Profile: compare timing to single-site at same chi_max

## Key Pitfalls to Avoid

1. **theta memory layout:** `theta[a + s1*cL + s2*cL*d + b*cL*d*d]`. The `ldb` in step 1 batched GEMM must be `cL*d*d` (not `cL*d`), because fixing (s1,s2) leaves b-columns spaced `cL*d²` apart.

2. **SVD reshape:** theta IS already `(cL*d, d*cR)` in memory — no data movement needed. Just pass `m = cL*d, n = d*cR` to the SVD routine.

3. **Environment indexing:** `apply_heff` uses `L_envs_[site]` and `R_envs_[site+2]` (not site+1). The right environment is TWO bonds to the right of the left site.

4. **No absorption into third tensor:** Unlike single-site where S gets absorbed into a neighbor, two-site SVD produces both output tensors directly. No GEMM with a third MPS tensor.

5. **Boundary handling:** The last pair in a right sweep is (L-2, L-1). The first pair in a left sweep is also (L-2, L-1). Make sure env indices don't go out of bounds.

6. **Fused MPO is bond-dependent:** `WW[bond]` depends on `MPO[site]` and `MPO[site+1]`. For uniform MPOs (bulk sites identical), all interior WW are the same, but boundary WW differ.

7. **Complex conjugation:** The left environment conjugation fix from single-site (`conjugate_inplace` after step 3 of `update_left_env`) carries over unchanged — env updates are per-site, not per-pair.

8. **Bond dimension growth:** Two-site DMRG naturally grows bond dimensions (single-site cannot). Initial MPS can start at chi=1 everywhere and the SVD will grow bonds up to chi_max. This is a feature, not a bug.

## Files Reference (Single-Site, to Copy From)

- `dmrg-gpu/src/dmrg_gpu.h` — class definition, member variables, method signatures
- `dmrg-gpu/src/dmrg_gpu_impl.h` — full implementation (~700 lines)
- `dmrg-gpu/src/scalar_traits.h` — ScalarTraits<double>, ScalarTraits<hipDoubleComplex> (no changes)
- `dmrg-gpu/src/dmrg_gpu.cpp` — template instantiations (no changes)
- `dmrg-gpu/src/test_dmrg_gpu.cpp` — Heisenberg + Josephson MPO builders, test harness
- `dmrg-gpu/CMakeLists.txt` — build configuration

## Verification Energies

| System | L | d | D_mpo | chi_max | Exact Energy |
|--------|---|---|-------|---------|-------------|
| Heisenberg | 4 | 2 | 5 | 16+ | -1.616025403784 |
| Heisenberg | 8 | 2 | 5 | 32+ | -3.374932598687 |
| Josephson (E_J=1, E_C=0.5, φ=π/4) | 4 | 3 | 4 | 16+ | -1.053346829927 |
| Josephson (E_J=1, E_C=0.5, φ=π/4) | 6 | 3 | 4 | 32+ | -1.748843818181 |
