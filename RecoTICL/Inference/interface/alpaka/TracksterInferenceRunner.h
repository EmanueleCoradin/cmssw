#ifndef RecoTICL_Inference_alpaka_TracksterInferenceRunner_h
#define RecoTICL_Inference_alpaka_TracksterInferenceRunner_h

#include <cstdint>
#include <string>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTICL/Inference/interface/alpaka/BatchedInferenceRunner.h"
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

    // Update an already copied output, allowing PID and regression to share it.
    void runInto(Queue& queue, InputDevice const& features, PublishedOutputDevice& output, float minEnergy = 1.f);

    // Enqueues copies, model inference and result scattering on queue.
    // As in the original producer, completion is coordinated by Alpaka/EDM.
    [[nodiscard]] PublishedOutputDevice run(Queue& queue,
                                            InputDevice const& features,
                                            PublishedSourceDevice const& tracksters);

  private:
    void forward(Queue& queue, InferenceTensors& inputs, InferenceTensors& outputs);

    BatchedInferenceRunner runner_;
    uint32_t const batchSize_;
    int const warmupIterations_;
  };

  // Explicitly instantiated in RecoTICL/Inference/src/alpaka for each backend.
  // Add corresponding instantiations when introducing another adapter.
  extern template class TracksterInferenceRunner<CNNPIDAdapter>;
  extern template class TracksterInferenceRunner<DNNPIDAdapter>;
  extern template class TracksterInferenceRunner<PFNPIDAdapter>;
  extern template class TracksterInferenceRunner<DNNEnergyAdapter>;
  extern template class TracksterInferenceRunner<PFNEnergyAdapter>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Inference_alpaka_TracksterInferenceRunner_h
