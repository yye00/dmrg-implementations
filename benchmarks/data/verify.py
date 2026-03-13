"""
Verify MPS/MPO data file integrity.

Loads all binary files in data/ and checks that they parse correctly.
"""

import sys
from pathlib import Path

from benchmarks.lib.data_loader import load_mps_from_binary, load_mpo_from_binary


def verify(data_dir=None):
    """Verify all binary data files in the data directory."""
    if data_dir is None:
        data_dir = Path(__file__).parent

    data_dir = Path(data_dir)
    mps_files = sorted(data_dir.glob("*_mps.bin"))
    mpo_files = sorted(data_dir.glob("*_mpo.bin"))

    print(f"Verifying data in: {data_dir}")
    print(f"  MPS files: {len(mps_files)}")
    print(f"  MPO files: {len(mpo_files)}")

    errors = 0

    for f in mps_files:
        try:
            tensors, meta = load_mps_from_binary(f, quiet=True)
            print(f"  OK: {f.name} ({len(tensors)} sites)")
        except Exception as e:
            print(f"  FAIL: {f.name}: {e}")
            errors += 1

    for f in mpo_files:
        try:
            tensors, meta = load_mpo_from_binary(f, quiet=True)
            print(f"  OK: {f.name} ({len(tensors)} sites)")
        except Exception as e:
            print(f"  FAIL: {f.name}: {e}")
            errors += 1

    if errors:
        print(f"\n{errors} file(s) failed verification!")
        return False

    print(f"\nAll {len(mps_files) + len(mpo_files)} files OK.")
    return True


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    ok = verify(data_dir)
    sys.exit(0 if ok else 1)
