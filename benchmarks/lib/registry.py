"""
Implementation registry for DMRG benchmark suite.

Each implementation is described by its type (cpu/gpu, serial/parallel),
what parallelism knobs it supports, and how to invoke it.

To add a new implementation:
  1. Add an entry to IMPLEMENTATIONS below
  2. Create or extend a runner in lib/runners/
"""

IMPLEMENTATIONS = {
    # --- CPU serial implementations (quimb reference) ---
    "quimb-dmrg1": {
        "type": "cpu-serial",
        "description": "Quimb DMRG single-site (CPU reference)",
        "supports_threads": True,
        "supports_np": False,
        "runner": "quimb",
        "algorithm": "dmrg1",
    },
    "quimb-dmrg2": {
        "type": "cpu-serial",
        "description": "Quimb DMRG two-site (CPU reference)",
        "supports_threads": True,
        "supports_np": False,
        "runner": "quimb",
        "algorithm": "dmrg2",
    },

    # --- CPU parallel implementations (MPI-based) ---
    "pdmrg": {
        "type": "cpu-parallel",
        "description": "Parallel DMRG (numpy tensordot, MPI)",
        "supports_threads": True,
        "supports_np": True,
        "runner": "pdmrg",
        "package": "pdmrg",
        "entry": "pdmrg.dmrg",
        "function": "pdmrg_main",
    },
    "pdmrg2": {
        "type": "cpu-parallel",
        "description": "Parallel DMRG two-site (numpy, MPI)",
        "supports_threads": True,
        "supports_np": True,
        "runner": "pdmrg",
        "package": "pdmrg2",
        "entry": "pdmrg.dmrg",
        "function": "pdmrg_main",
    },
    "pdmrg-cotengra": {
        "type": "cpu-parallel",
        "description": "Parallel DMRG with cotengra contraction paths (MPI)",
        "supports_threads": True,
        "supports_np": True,
        "runner": "pdmrg",
        "package": "pdmrg-cotengra",
        "entry": "pdmrg.dmrg",
        "function": "pdmrg_main",
    },
    "a2dmrg": {
        "type": "cpu-parallel",
        "description": "Additive two-level DMRG (MPI)",
        "supports_threads": True,
        "supports_np": True,
        "runner": "a2dmrg",
        "package": "a2dmrg",
        "entry": "a2dmrg.dmrg",
        "function": "a2dmrg_main",
    },

    # --- GPU serial implementations ---
    "dmrg-gpu": {
        "type": "gpu-serial",
        "description": "GPU DMRG single-site (HIP/rocBLAS)",
        "supports_threads": False,
        "supports_np": False,
        "runner": "gpu",
        "executable": "dmrg-gpu/build/test_dmrg_gpu",
    },
    "dmrg2-gpu": {
        "type": "gpu-serial",
        "description": "GPU DMRG two-site (HIP/rocBLAS)",
        "supports_threads": False,
        "supports_np": False,
        "runner": "gpu",
        "executable": "dmrg2-gpu/build/test_dmrg2_gpu",
    },

    # --- GPU parallel implementations ---
    "pdmrg-gpu": {
        "type": "gpu-parallel",
        "description": "GPU parallel DMRG (HIP streams)",
        "supports_threads": False,
        "supports_np": True,
        "runner": "gpu",
        "executable": "pdmrg-gpu/build/pdmrg_gpu",
    },
    "pdmrg2-gpu": {
        "type": "gpu-parallel",
        "description": "GPU parallel DMRG two-site (HIP streams)",
        "supports_threads": False,
        "supports_np": True,
        "runner": "gpu",
        "executable": "pdmrg2-gpu/build/pdmrg2_gpu",
    },
}


# --- Problem size definitions ---

SIZES = {
    "heisenberg": {
        "small":  {"L": 12, "chi": 20,  "max_sweeps": 30},
        "medium": {"L": 20, "chi": 50,  "max_sweeps": 30},
        "large":  {"L": 40, "chi": 100, "max_sweeps": 50},
    },
    "josephson": {
        "small":  {"L": 8,  "chi": 20,  "n_max": 2, "max_sweeps": 30},
        "medium": {"L": 12, "chi": 50,  "n_max": 2, "max_sweeps": 40},
        "large":  {"L": 16, "chi": 100, "n_max": 2, "max_sweeps": 50},
    },
}


# --- Validation tolerances ---

VALIDATION_TOL = 1e-10       # energy must match gold standard within this
CONVERGENCE_TOL = 1e-11      # eigensolver convergence tolerance


def get_impl(name):
    """Get implementation config by name, raising KeyError if not found."""
    if name not in IMPLEMENTATIONS:
        available = ", ".join(sorted(IMPLEMENTATIONS.keys()))
        raise KeyError(f"Unknown implementation '{name}'. Available: {available}")
    return IMPLEMENTATIONS[name]


def list_impls(type_filter=None):
    """List implementation names, optionally filtered by type prefix."""
    if type_filter is None:
        return list(IMPLEMENTATIONS.keys())
    return [k for k, v in IMPLEMENTATIONS.items() if v["type"].startswith(type_filter)]


def get_size(model, size_name):
    """Get problem size parameters for a model and size category."""
    if model not in SIZES:
        raise KeyError(f"Unknown model '{model}'. Available: {', '.join(SIZES.keys())}")
    if size_name not in SIZES[model]:
        raise KeyError(f"Unknown size '{size_name}'. Available: {', '.join(SIZES[model].keys())}")
    return SIZES[model][size_name]
