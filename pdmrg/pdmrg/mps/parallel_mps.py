"""ParallelMPS: MPS fragment with V matrices at shared bonds.

Each MPI rank holds a contiguous block of MPS sites plus the
V = Λ⁻¹ matrices at its boundary bonds with neighboring ranks.
"""

import numpy as np


class ParallelMPS:
    """MPS fragment owned by one MPI rank with V boundary matrices.

    Attributes
    ----------
    arrays : list of ndarray
        MPS tensors in our convention (left_bond, phys, right_bond).
        Indexed locally: arrays[0] is the leftmost site of this rank.
    my_sites : range
        Global site indices owned by this rank.
    rank : int
        MPI rank.
    n_procs : int
        Total number of MPI processes.
    V_left : ndarray or None
        V = Λ⁻¹ at left boundary bond (with rank-1). None if rank == 0.
    V_right : ndarray or None
        V = Λ⁻¹ at right boundary bond (with rank+1). None if last rank.
    """

    def __init__(self, arrays, my_sites, rank, n_procs,
                 V_left=None, V_right=None):
        self.arrays = arrays
        self.my_sites = my_sites
        self.rank = rank
        self.n_procs = n_procs
        self.V_left = V_left
        self.V_right = V_right

    @property
    def n_local(self):
        return len(self.arrays)

    @property
    def is_leftmost(self):
        return self.rank == 0

    @property
    def is_rightmost(self):
        return self.rank == self.n_procs - 1

    def get_left_boundary_tensor(self):
        """Return the leftmost tensor of this rank's block."""
        return self.arrays[0]

    def get_right_boundary_tensor(self):
        """Return the rightmost tensor of this rank's block."""
        return self.arrays[-1]

    def set_left_boundary_tensor(self, tensor):
        self.arrays[0] = tensor

    def set_right_boundary_tensor(self, tensor):
        self.arrays[-1] = tensor
