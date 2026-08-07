#ifndef PhysicsTools_PyTorchAlpaka_interface_alpaka_PyTorchEDProducer_h
#define PhysicsTools_PyTorchAlpaka_interface_alpaka_PyTorchEDProducer_h

#include <exception>

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/FixedQueueRegistry.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/PyTorchAllocatorBridge.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::stream {

  template <typename... Args>
  class PyTorchEDProducer : public FixedQueueEDProducer<Args...> {
    using Base = FixedQueueEDProducer<Args...>;

  protected:
    explicit PyTorchEDProducer(edm::ParameterSet const& config) : Base(config) {}

    // Optional hooks for derived producers.
    //
    // beginStreamHook() is called after the queue has been registered
    // with the PyTorch allocator bridge.
    virtual void beginStreamHook(edm::StreamID, Queue) {}

    // endStreamHook() is called before the BLAS workspace is cleared
    // and before the queue is unregistered.
    virtual void endStreamHook(Queue) {}

  public:
    void beginStream(edm::StreamID sid, Queue queue) final {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
      if (PyTorchAllocatorBridge::isActive()) {
        auto& registry = cms::alpakatools::getFixedQueueRegistry<Queue>();
        registry.registerQueue(queue);

        try {
          beginStreamHook(sid, queue);
        } catch (...) {
          auto exception = std::current_exception();
          cleanupQueue(queue, registry, exception);
          std::rethrow_exception(exception);
        }

        return;
      } else
        beginStreamHook(sid, queue);
#else
      beginStreamHook(sid, queue);
#endif
    }

    void endStream(Queue queue) final {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
      if (PyTorchAllocatorBridge::isActive()) {
        auto& registry = cms::alpakatools::getFixedQueueRegistry<Queue>();

        std::exception_ptr exception;
        try {
          endStreamHook(queue);
        } catch (...) {
          exception = std::current_exception();
        }

        cleanupQueue(queue, registry, exception);
        if (exception) {
          std::rethrow_exception(exception);
        }
        return;
      }
#endif
      endStreamHook(queue);
    }

  private:
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    static void cleanupQueue(Queue const& queue,
                             cms::alpakatools::FixedQueueRegistry<Queue>& registry,
                             std::exception_ptr& exception) noexcept {
      try {
        PyTorchAllocatorBridge::clearBlasWorkspace(queue);
      } catch (...) {
        if (!exception) {
          exception = std::current_exception();
        }
      }

      try {
        registry.unregisterQueue(queue);
      } catch (...) {
        if (!exception) {
          exception = std::current_exception();
        }
      }
    }
#endif
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::stream

#endif  // PhysicsTools_PyTorchAlpaka_interface_alpaka_PyTorchEDProducer_h
