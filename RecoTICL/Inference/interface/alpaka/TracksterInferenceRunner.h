#ifndef RecoTICL_Inference_alpaka_TracksterInferenceRunner_h
#define RecoTICL_Inference_alpaka_TracksterInferenceRunner_h

#include <cstdint>
#include <string>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceTensorAdapters.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // TICL-specific orchestration of PyTorchAlpaka inference. This class is
  // independent of the EDM Event and of the framework module lifecycle.
  // Each stream::FixedQueueEDProducer owns its own runner, as before.
  template <typename Adapter>
  class TracksterInferenceRunner {
  public:
    using InputDevice = typename Adapter::InputDevice;
    using PublishedSourceDevice = typename Adapter::PublishedSourceDevice;
    using PublishedOutputDevice = typename Adapter::PublishedOutputDevice;

    TracksterInferenceRunner(std::string const& modelPath, uint32_t batchSize, bool convertToFP16, int warmupIterations);

    void warmup(Queue& queue);

    // Enqueues copies, model inference and result scattering on queue.
    // As in the original producer, completion is coordinated by Alpaka/EDM.
    [[nodiscard]] PublishedOutputDevice run(Queue& queue,
                                            InputDevice const& features,
                                            PublishedSourceDevice const& tracksters);

  private:
    void forward(Queue& queue, InferenceTensors& inputs, InferenceTensors& outputs);

    torch::AlpakaModel model_;
    uint32_t const batchSize_;
    bool const convertToFP16_;
    int const warmupIterations_;
  };

  // Explicitly instantiated in RecoTICL/Inference/src/alpaka for each backend.
  // Add corresponding instantiations when introducing another adapter.
  extern template class TracksterInferenceRunner<CNNPIDAdapter>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Inference_alpaka_TracksterInferenceRunner_h
