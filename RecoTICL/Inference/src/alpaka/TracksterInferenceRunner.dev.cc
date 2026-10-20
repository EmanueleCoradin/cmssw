#include <cstdint>
#include <limits>
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
      : runner_{modelPath, convertToFP16}, batchSize_{batchSize}, warmupIterations_{warmupIterations} {
    if (batchSize_ == 0u || batchSize_ > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      throw cms::Exception("TracksterInferenceRunner") << "batchSize must be positive and fit the SoA int32_t size";
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
  void TracksterInferenceRunner<Adapter>::runInto(Queue& queue,
                                                  InputDevice const& features,
                                                  PublishedOutputDevice& output,
                                                  float minEnergy) {
    const auto total = static_cast<int32_t>(features.const_view().metadata().size());

    auto predictions = Adapter::makeModelOutput(queue, total);
    auto featureRecords = features.const_view().records();
    auto predictionRecords = predictions.view().records();
    InferenceTensors inputs, outputs;
    Adapter::registerInputs(inputs, featureRecords, batchSize_);
    Adapter::registerOutputs(outputs, predictionRecords, batchSize_);
    forward(queue, inputs, outputs);
    Adapter::applyResults(queue, featureRecords, predictions, output, static_cast<uint32_t>(total), minEnergy);
  }

  template <typename Adapter>
  void TracksterInferenceRunner<Adapter>::forward(Queue& queue, InferenceTensors& inputs, InferenceTensors& outputs) {
    runner_.run(
        queue, inputs, outputs, Adapter::channelsLast ? InputTransform::ImageChannelsLast : InputTransform::Identity);
  }

  template class TracksterInferenceRunner<CNNPIDAdapter>;
  template class TracksterInferenceRunner<DNNPIDAdapter>;
  template class TracksterInferenceRunner<PFNPIDAdapter>;
  template class TracksterInferenceRunner<DNNEnergyAdapter>;
  template class TracksterInferenceRunner<PFNEnergyAdapter>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
