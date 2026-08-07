#ifndef PhysicsTools_PyTorchAlpaka_interface_alpaka_PyTorchEDProducer_h
#define PhysicsTools_PyTorchAlpaka_interface_alpaka_PyTorchEDProducer_h

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/FixedQueueRegistry.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/PyTorchAllocatorBridge.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::stream {

  template <typename... Args>
  class PyTorchEDProducer : public FixedQueueEDProducer<Args...> {
    using Base = FixedQueueEDProducer<Args...>;

  protected:
    PyTorchEDProducer(edm::ParameterSet const iConfig) : Base(iConfig) {}

    virtual void beginStreamImpl(edm::StreamID sid, Queue queue) {}

    virtual void endStreamImpl(Queue queue) {}

  public:
    void beginStream(edm::StreamID sid, Queue queue) final {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      // Register the queue before calling user code.
      cms::alpakatools::getFixedQueueRegistry<Queue>().registerQueue(queue);
#endif

      beginStreamImpl(sid, queue);
    }

    void endStream(Queue queue) final {
      // Give user code access to the registered queue.
      PyTorchAllocatorBridge<Queue>::resetCUBlas(queue);
      endStreamImpl(queue);

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      // Unregister after user code has finished.
      cms::alpakatools::getFixedQueueRegistry<Queue>().unregisterQueue(queue);
#endif
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::stream

#endif  // PhysicsTools_PyTorchAlpaka_interface_alpaka_PyTorchEDProducer_h
