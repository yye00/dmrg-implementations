// Explicit template instantiations for PDMRG2GPU
#include "pdmrg2_gpu.h"

template class PDMRG2GPU<double>;
template class PDMRG2GPU<hipDoubleComplex>;
