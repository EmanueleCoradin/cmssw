#include <cstdint>
#include <string>
#include <Eigen/Core>

#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceRunner.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename Adapter>
  TracksterInferenceRunner<Adapter>::TracksterInferenceRunner(std::string const& modelPath,
                                                               uint32_t batchSize,
                                                               bool convertToFP16,
                                                               int warmupIterations)
      : model_{modelPath},
        batchSize_{batchSize},
        convertToFP16_{convertToFP16},
        warmupIterations_{warmupIterations} {
    if (batchSize_ == 0u) {
      throw cms::Exception("TracksterInferenceRunner") << "batchSize must be positive";
    }
    if (warmupIterations_ < 0) {
      throw cms::Exception("TracksterInferenceRunner") << "warmupIterations must be non-negative";
    }
  }

  template <typename Adapter>
  void TracksterInferenceRunner<Adapter>::warmup(Queue& queue) {
    if (warmupIterations_ == 0) {
      return;
    }

    auto warmupInput = Adapter::makeWarmupInput(queue, batchSize_);
    auto warmupOutput = Adapter::makeModelOutput(queue, batchSize_);
    auto inputRecords = warmupInput.view().records();
    auto outputRecords = warmupOutput.view().records();

    for (int iteration = 0; iteration < warmupIterations_; ++iteration) {
      InferenceTensors inputs, outputs;
      Adapter::registerInputs(inputs, inputRecords, batchSize_);
      Adapter::registerOutputs(outputs, outputRecords, batchSize_);
      forward(queue, inputs, outputs);
    }
  }

  template <typename Adapter>
  typename TracksterInferenceRunner<Adapter>::PublishedOutputDevice TracksterInferenceRunner<Adapter>::run(
      Queue& queue, InputDevice const& features, PublishedSourceDevice const& tracksters) {
    const int total = detail::checkedSoASize(features.const_view().metadata().size(), "inference samples");
    Adapter::validateSampleCount(tracksters, total);

    auto publishedOutput = Adapter::makePublishedOutput(queue, tracksters);
    if (total == 0) {
      return publishedOutput;
    }

    auto predictions = Adapter::makeModelOutput(queue, total);
    auto featureRecords = features.const_view().records();
    auto predictionRecords = predictions.view().records();

    InferenceTensors inputs, outputs;
    Adapter::registerInputs(inputs, featureRecords, batchSize_);
    Adapter::registerOutputs(outputs, predictionRecords, batchSize_);
    forward(queue, inputs, outputs);

    Adapter::applyResults(queue, featureRecords, predictions, publishedOutput, static_cast<uint32_t>(total));
    return publishedOutput;
  }

  template <typename Adapter>
  void TracksterInferenceRunner<Adapter>::forward(Queue& queue, InferenceTensors& inputs, InferenceTensors& outputs) {
    if (convertToFP16_) {
      model_.forward(queue, inputs, outputs, ::torch::kHalf);
    } else {
      model_.forward(queue, inputs, outputs);
    }
  }

  template class TracksterInferenceRunner<CNNPIDAdapter>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
