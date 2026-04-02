// Explicit template instantiations for DMRGGPU
#include "dmrg_gpu.h"

template class DMRGGPU<double>;
template class DMRGGPU<cuDoubleComplex>;
