#ifndef PhysicsTools_PyTorchAlpaka_interface_alpaka_PyTorchAllocatorBridge_h
#define PhysicsTools_PyTorchAlpaka_interface_alpaka_PyTorchAllocatorBridge_h

#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <utility>

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/FixedQueueRegistry.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/getDeviceCachingAllocator.h"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <ATen/cuda/CUDABlas.h>
#elif ALPAKA_ACC_GPU_HIP_ENABLED

#ifndef USE_ROCM
#define USE_ROCM 1
#endif

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <ATen/hip/HIPBlas.h>
#endif

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // PyTorch provides the device and native stream for each allocation.
  // The PyTorchAllocatorBridge makes use of the FixedQueueRegistry to retrieve the associated Alpaka queue
  // and call the CMSSW caching allocator functions.
  class PyTorchAllocatorBridge {
  public:
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    using QueueHandle = decltype(alpaka::getNativeHandle(std::declval<Queue>()));

    static void* allocate(size_t size, int deviceId, QueueHandle stream) {
      auto queue = cms::alpakatools::getFixedQueueRegistry<Queue>().findQueue(deviceId, stream);

      if (!queue) {
        throw cms::Exception("PyTorchAllocatorBridge")
            << "Could not find an Alpaka Queue associated to CUDA device " << deviceId << " and stream " << stream;
      }

      auto const device = alpaka::getDev(*queue);
      auto& allocator = cms::alpakatools::getDeviceCachingAllocator<Device, Queue>(device);

      return allocator.allocate(size, std::move(*queue));
    }

    static void free(void* ptr, size_t size, int deviceId, QueueHandle /*stream*/) {
      if (ptr == nullptr) {
        return;
      }

      auto const& deviceList = cms::alpakatools::devices<Platform>();

      if (deviceId < 0 || static_cast<size_t>(deviceId) >= deviceList.size()) {
        throw cms::Exception("PyTorchAllocatorBridge") << "Invalid CUDA device ID " << deviceId;
      }

      auto const& device = deviceList[deviceId];
      auto& allocator = cms::alpakatools::getDeviceCachingAllocator<Device, Queue>(device);
      allocator.free(ptr);
    }
#endif
    static void install() {
      if (active_)
        return;
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
      namespace PA = ::torch::cuda::CUDAPluggableAllocator;
      auto allocator = PA::createCustomAllocator(&PyTorchAllocatorBridge::allocate, &PyTorchAllocatorBridge::free);
      PA::changeCurrentAllocator(allocator);
      active_ = true;
#endif
    }

    static void clearBlasWorkspace(Queue queue) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
      auto stream = alpaka::getNativeHandle(queue);
      at::cuda::clearCublasWorkspacesForStream(stream);
#endif
    }

    static bool isActive() { return active_; }

  private:
    inline static bool active_ = false;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif
