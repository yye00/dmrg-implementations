# Critical Bug: Environment Building After MPS Load

## Problem
Currently, after loading MPS from binary, we call `initialize_environments()` which sets ALL environments to identity. This gives completely wrong energy (-0.089 vs -3.375).

## Root Cause  
Identity environments do NOT encode the actual MPS state. To compute correct energy, environments must be built FROM the MPS tensors.

## Solution
Implement `build_environments_from_mps()` that:

### 1. Initialize Boundary Environments
- L_envs[0] = identity (chi_L=1, no sites to left)
- R_envs[num_sites] = identity (chi_R=1, no sites to right)

### 2. Build Left Environments (left-to-right sweep)
For i = 0 to num_sites-2:
  L_envs[i+1] = update_left_env(L_envs[i], A[i], W[i])
  
Where update_left_env does 3-step contraction (same as rebuild_right_boundary_env):
  L[a,w,a'] * A[a,s,b] * W[w,s,s',wp] * A*[a',s',b'] -> L_new[b,wp,b']

### 3. Build Right Environments (right-to-left sweep)
For i = num_sites-1 down to 1:
  R_envs[i-1] = update_right_env(R_envs[i], A[i], W[i])

Where update_right_env does similar 3-step contraction.

## Implementation
Add function to stream_segment.cpp:
```cpp
void StreamSegment::build_environments_from_mps() {
    // 1. Init boundaries  
    // 2. Build L_envs left-to-right
    // 3. Build R_envs right-to-left
}
```

Call this INSTEAD of initialize_environments() in load_mps_from_binary().

## Expected Result
With proper environment building:
- Energy should be ~-3.375 Ha immediately after loading
- No need for optimization iterations to fix it

## Reference
CPU code: pdmrg/pdmrg/environments/environment.py::build_initial_envs()
