#ifndef FPX_h
#define FPX_h
#include <limits>

namespace cms {

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) or defined(ALPAKA_ACC_GPU_HIP_ENABLED)
  // on NVIDIA or AMD GPUs
  using FPX = ::__half;
  __host__ __device__ inline FPX makeNaN() { return __float2half(std::numeric_limits<float>::quiet_NaN()); }

#else
  // on CPU
  using FPX = ::_Float16;
  inline FPX makeNaN() { return std::numeric_limits<FPX>::quiet_NaN(); }

#endif

}  // namespace cms

#endif
