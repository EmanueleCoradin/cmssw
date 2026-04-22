#ifndef PhysicsTools_PyTorch_interface_PyTorchFP16Bridge_h
#define PhysicsTools_PyTorch_interface_PyTorchFP16Bridge_h

#include <c10/core/ScalarType.h>
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <cuda_fp16.h>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <hip/hip_fp16.h>
#endif

namespace c10 {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) or defined(ALPAKA_ACC_GPU_HIP_ENABLED)
  template <>
  struct CppTypeToScalarType<__half> : std::integral_constant<ScalarType, ScalarType::Half> {};

#else
  template <>
  struct CppTypeToScalarType<_Float16> : std::integral_constant<ScalarType, ScalarType::Half> {};
#endif

}  // namespace c10

#endif  // PhysicsTools_PyTorch_interface_PyTorchFPXBridge_h
