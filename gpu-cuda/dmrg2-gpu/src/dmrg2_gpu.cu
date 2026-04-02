// Explicit template instantiations for DMRG2GPU
#include "dmrg2_gpu.h"

template class DMRG2GPU<double>;
template class DMRG2GPU<cuDoubleComplex>;
