#ifndef HeterogeneousCore_AlpakaMath_float16_t_h
#define HeterogeneousCore_AlpakaMath_float16_t_h

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <cuda_fp16.h>
#endif
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <hip/hip_fp16.h>
#endif

namespace cms {

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) or defined(ALPAKA_ACC_GPU_HIP_ENABLED)
  // on NVIDIA or AMD GPUs
  using float16_t = __half;
#else
  // on CPU
  using float16_t = _Float16;
#endif

}  // namespace cms
#endif
