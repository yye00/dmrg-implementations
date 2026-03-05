"""
MPI compatibility layer.

Provides MPI-like interface for serial execution when mpi4py is unavailable.
"""

try:
    from mpi4py import MPI
    HAS_MPI = True
except (ImportError, RuntimeError, OSError):
    # ImportError: mpi4py not installed
    # RuntimeError: mpi4py installed but MPI library not found
    # OSError: library loading issues
    HAS_MPI = False

    class _FakeCommunicator:
        """Minimal MPI communicator mock for serial execution."""
        
        @property
        def rank(self):
            return 0
        
        @property
        def size(self):
            return 1
        
        def Get_rank(self):
            return 0
        
        def Get_size(self):
            return 1
        
        def Barrier(self):
            pass
        
        def bcast(self, obj, root=0):
            return obj
        
        def Bcast(self, buf, root=0):
            pass
        
        def gather(self, sendobj, root=0):
            return [sendobj]
        
        def Gather(self, sendbuf, recvbuf, root=0):
            if recvbuf is not None:
                recvbuf[:] = sendbuf
        
        def allgather(self, sendobj):
            return [sendobj]
        
        def Allgather(self, sendbuf, recvbuf):
            if recvbuf is not None:
                recvbuf[:] = sendbuf
        
        def allreduce(self, sendobj, op=None):
            return sendobj
        
        def Allreduce(self, sendbuf, recvbuf, op=None):
            recvbuf[:] = sendbuf
        
        def reduce(self, sendobj, op=None, root=0):
            return sendobj
        
        def Reduce(self, sendbuf, recvbuf, op=None, root=0):
            if recvbuf is not None:
                recvbuf[:] = sendbuf

    class _FakeMPI:
        """Minimal MPI module mock for serial execution."""
        COMM_WORLD = _FakeCommunicator()
        COMM_SELF = _FakeCommunicator()
        SUM = "SUM"
        MAX = "MAX"
        MIN = "MIN"
        Comm = type(_FakeCommunicator())  # Type for type hints
        Intracomm = type(_FakeCommunicator())
        
        @staticmethod
        def Finalize():
            pass
    
    MPI = _FakeMPI()

# Export
__all__ = ['MPI', 'HAS_MPI']
