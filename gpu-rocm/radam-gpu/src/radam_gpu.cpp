// Explicit template instantiations for RAdamGPU
#include "radam_gpu.h"

template class RAdamGPU<double>;
template class RAdamGPU<hipDoubleComplex>;
