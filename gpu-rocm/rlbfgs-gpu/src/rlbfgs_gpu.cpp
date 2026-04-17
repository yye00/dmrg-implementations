// Explicit template instantiations for RLBFGSGPU
#include "rlbfgs_gpu.h"

template class RLBFGSGPU<double>;
template class RLBFGSGPU<hipDoubleComplex>;
