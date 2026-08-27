#ifndef PhysicsTools_PyTorchAlpaka_interface_alpaka_AlpakaModelAOT_h
#define PhysicsTools_PyTorchAlpaka_interface_alpaka_AlpakaModelAOT_h

#include "alpaka/alpaka.hpp"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/ModelAOT.h"
#include "PhysicsTools/PyTorchAlpaka/interface/GetDevice.h"
#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/SoAConversion.h"
#include "PhysicsTools/PyTorchAlpaka/interface/QueueGuard.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torch {

  class AlpakaModelAOT : public cms::torch::ModelAOT {

  public:
    // inherit generic pytorch interface methods
    using cms::torch::ModelAOT::forward;
    explicit AlpakaModelAOT(const std::string &model_path, const Queue &queue)
      : cms::torch::ModelAOT(model_path, cms::torch::alpakatools::getDevice(queue)){}

    // Forward pass (inference) of model with SoA metadata input/output.
    // Allows to run inference directly using SoA portable objects/collections without excessive copies and conversions.
    // Refer: PhysicsTools/PyTorch/interface/SoAConversion.h for details about wrapping memory layouts.
    void forward(Queue &queue,
                 cms::torch::alpakatools::TensorCollection<Queue> &inputs,
                 cms::torch::alpakatools::TensorCollection<Queue> &outputs,
                 std::optional<::torch::Dtype> dtype = std::nullopt) {
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      inputs.copy(queue, cms::torch::alpakatools::detail::MemcpyKind::DeviceToHost);
      outputs.copy(queue, cms::torch::alpakatools::detail::MemcpyKind::DeviceToHost);
#else
      inputs.copy(queue, cms::torch::alpakatools::detail::MemcpyKind::DeviceToDevice);
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
      auto input_tensor = cms::torch::alpakatools::detail::convertInput<Queue, at::Tensor>(inputs, device_, dtype);
      
      void* stream_handle = nullptr;
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) // || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
      stream_handle = static_cast<void*>(queue.getNativeHandle());
#endif  

      auto output_tensors = cms::torch::ModelAOT::forward(input_tensor, stream_handle);
      cms::torch::alpakatools::detail::convertOutput(output_tensors, outputs, device_);

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      outputs.copy(queue, cms::torch::alpakatools::detail::MemcpyKind::HostToDevice);
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torch

#endif  // PhysicsTools_PyTorchAlpaka_interface_alpaka_AlpakaModelAOT_h
