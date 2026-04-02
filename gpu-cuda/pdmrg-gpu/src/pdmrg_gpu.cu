// Explicit template instantiations for PDMRGGPU
#include "pdmrg_gpu.h"

template class PDMRGGPU<double>;
template class PDMRGGPU<cuDoubleComplex>;
