"""
Workaround for quimb import issue on Python 3.13+

The issue: quimb uses numba caching, but numba's caching mechanism
fails on Python 3.13+ with "no locator available" error.

Solution: Monkeypatch numba's enable_caching methods to be no-ops
BEFORE importing quimb.

Usage:
    import fix_quimb_python313
    import quimb  # Will now work
"""

import numba
from numba.core.dispatcher import Dispatcher
from numba.np.ufunc import ufuncbuilder

# Monkeypatch both regular dispatcher and ufunc dispatcher
Dispatcher.enable_caching = lambda self: None
ufuncbuilder.UFuncDispatcher.enable_caching = lambda self: None
