import quimb.tensor as qtn

from a2dmrg.parallel.local_steps import prepare_candidate_mps_list
from a2dmrg.mps.mps_utils import create_random_mps


def test_prepare_candidate_mps_list_reduced_sites():
    L = 6
    mps0 = create_random_mps(L, 4, phys_dim=2, dtype="float64", canonical="left")

    # Fake "all_results" with energies so selection logic can keep a subset.
    all_results = {}
    for site in range(L):
        upd = create_random_mps(L, 4, phys_dim=2, dtype="float64", canonical="left")
        all_results[site] = (upd, float(site))  # monotone energies

    # Keep only two sites
    keep_sites = [0, 1]
    reduced = {s: all_results[s] for s in keep_sites}

    cand = prepare_candidate_mps_list(mps0, reduced)
    assert len(cand) == 1 + len(keep_sites)
    assert isinstance(cand[0], qtn.MatrixProductState)
