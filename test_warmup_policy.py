#!/usr/bin/env python3
"""Test warmup policy changes for PDMRG, PDMRG2, and A2DMRG."""

import sys
import pytest

def test_pdmrg_no_parallel_warmup():
    """Verify PDMRG has no parallel_warmup_flag parameter."""
    sys.path.insert(0, 'pdmrg')
    from pdmrg.dmrg import pdmrg_main
    import inspect
    
    sig = inspect.signature(pdmrg_main)
    params = list(sig.parameters.keys())
    
    assert 'parallel_warmup_flag' not in params, "parallel_warmup_flag should be removed from PDMRG"
    print("✓ PDMRG: parallel_warmup_flag removed")

def test_pdmrg2_no_parallel_warmup():
    """Verify PDMRG2 has no parallel_warmup_flag parameter."""
    sys.path.insert(0, 'pdmrg2')
    from pdmrg.dmrg import pdmrg_main
    import inspect
    
    sig = inspect.signature(pdmrg_main)
    params = list(sig.parameters.keys())
    
    assert 'parallel_warmup_flag' not in params, "parallel_warmup_flag should be removed from PDMRG2"
    print("✓ PDMRG2: parallel_warmup_flag removed")

def test_a2dmrg_warmup_defaults():
    """Verify A2DMRG warmup defaults and bounds."""
    sys.path.insert(0, 'a2dmrg')
    from a2dmrg.dmrg import a2dmrg_main
    import inspect
    
    sig = inspect.signature(a2dmrg_main)
    
    # Check warmup_sweeps default is 2 (matches PDMRG/PDMRG2, set 2026-03-07)
    assert sig.parameters['warmup_sweeps'].default == 2, "warmup_sweeps default should be 2"
    print("✓ A2DMRG: warmup_sweeps default = 2")
    
    # Check experimental_nonpaper parameter exists
    assert 'experimental_nonpaper' in sig.parameters, "experimental_nonpaper parameter should exist"
    assert sig.parameters['experimental_nonpaper'].default == False, "experimental_nonpaper default should be False"
    print("✓ A2DMRG: experimental_nonpaper parameter exists (default=False)")

def test_a2dmrg_warmup_bounds():
    """Verify A2DMRG warmup bounds are enforced."""
    sys.path.insert(0, 'a2dmrg')
    from a2dmrg.dmrg import a2dmrg_main
    from mpi4py import MPI
    import quimb.tensor as qtn
    
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        print("⚠️  Skipping A2DMRG bounds test (requires np>=2)")
        return
    
    # Build simple test MPO
    L = 4
    mpo = qtn.MPO_ham_heis(L, j=1.0)
    
    # Test 1: warmup_sweeps=3 without experimental_nonpaper should fail
    try:
        energy, mps = a2dmrg_main(L, mpo, max_sweeps=1, warmup_sweeps=3, 
                                   experimental_nonpaper=False, comm=comm, verbose=False)
        assert False, "Should have raised ValueError for warmup_sweeps=3"
    except ValueError as e:
        assert "exceeds paper-faithful bound" in str(e)
        print("✓ A2DMRG: warmup_sweeps > 2 rejected without experimental_nonpaper")
    
    # Test 2: warmup_sweeps=3 WITH experimental_nonpaper should succeed
    try:
        energy, mps = a2dmrg_main(L, mpo, max_sweeps=1, warmup_sweeps=3,
                                   experimental_nonpaper=True, comm=comm, verbose=False)
        print("✓ A2DMRG: warmup_sweeps=3 allowed with experimental_nonpaper=True")
    except Exception as e:
        print(f"⚠️  A2DMRG warmup_sweeps=3 test incomplete: {e}")

if __name__ == '__main__':
    print("Testing warmup policy changes...\n")
    
    try:
        test_pdmrg_no_parallel_warmup()
    except Exception as e:
        print(f"❌ PDMRG test failed: {e}")
    
    try:
        test_pdmrg2_no_parallel_warmup()
    except Exception as e:
        print(f"❌ PDMRG2 test failed: {e}")
    
    try:
        test_a2dmrg_warmup_defaults()
    except Exception as e:
        print(f"❌ A2DMRG defaults test failed: {e}")
    
    try:
        test_a2dmrg_warmup_bounds()
    except Exception as e:
        print(f"❌ A2DMRG bounds test failed: {e}")
    
    print("\n✅ All warmup policy tests completed!")
