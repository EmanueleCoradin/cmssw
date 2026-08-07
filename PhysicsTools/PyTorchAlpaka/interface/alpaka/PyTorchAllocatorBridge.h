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
#endif

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TQueue>
  class PyTorchAllocatorBridge {
  public:
    using Queue = TQueue;
    using Device = alpaka::Dev<Queue>;
    using Platform = alpaka::Platform<Device>;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    static void* allocate(size_t size, int deviceId, cudaStream_t stream) {
      auto queue = cms::alpakatools::getFixedQueueRegistry<Queue>().findQueue(deviceId, stream);

      if (!queue) {
        throw cms::Exception("PyTorchAllocatorBridge")
            << "Could not find an Alpaka Queue associated to CUDA device " << deviceId << " and stream " << stream;
      }

      auto const device = alpaka::getDev(*queue);
      auto& allocator = cms::alpakatools::getDeviceCachingAllocator<Device, Queue>(device);

      return allocator.allocate(size, std::move(*queue));
    }

    static void free(void* ptr, size_t size, int deviceId, cudaStream_t /*stream*/) {
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
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

    static void reset_fn() {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      auto const& deviceList = cms::alpakatools::devices<Platform>();
      for (auto const& device : deviceList) {
        cms::alpakatools::getDeviceCachingAllocator<Device, Queue>(device).freeAllCached();
      }
#endif
    }

    static void install() {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      namespace PA = ::torch::cuda::CUDAPluggableAllocator;
      auto allocator = PA::createCustomAllocator(&PyTorchAllocatorBridge::allocate, &PyTorchAllocatorBridge::free);
      PA::changeCurrentAllocator(allocator);
#endif
    }

    static void resetCUBlas(Queue queue) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      auto stream = alpaka::getNativeHandle(queue);
      at::cuda::clearCublasWorkspacesForStream(stream);
#endif
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif
