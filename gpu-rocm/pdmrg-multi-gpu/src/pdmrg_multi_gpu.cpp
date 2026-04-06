// Explicit template instantiations for PDMRGMultiGPU
#include "pdmrg_multi_gpu.h"

template class PDMRGMultiGPU<double>;
template class PDMRGMultiGPU<hipDoubleComplex>;
