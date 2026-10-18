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
      : model_{modelPath}, batchSize_{batchSize}, convertToFP16_{convertToFP16}, warmupIterations_{warmupIterations} {
    model_.to(convertToFP16_ ? ::torch::kHalf : ::torch::kFloat);
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
  typename TracksterInferenceRunner<Adapter>::PublishedOutputDevice TracksterInferenceRunner<Adapter>::run(
      Queue& queue, InputDevice const& features, PublishedSourceDevice const& tracksters) {
    const int total = detail::checkedSoASize(features.const_view().metadata().size(), "inference samples");
    Adapter::validateSampleCount(tracksters, total);

    auto publishedOutput = Adapter::makePublishedOutput(queue, tracksters);
    runInto(queue, features, publishedOutput);
    return publishedOutput;
  }

  template <typename Adapter>
  void TracksterInferenceRunner<Adapter>::runInto(Queue& queue, InputDevice const& features,
                                                 PublishedOutputDevice& output, float minEnergy) {
    const int total = detail::checkedSoASize(features.const_view().metadata().size(), "inference samples");
    Adapter::validateSampleCount(output, total);
    if (total == 0)
      return;
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
    if (inputs.batchCount() != outputs.batchCount())
      throw cms::Exception("TracksterInference") << "Input/output batch counts differ";
    torch::QueueGuard guard(queue);
    ::torch::NoGradGuard noGrad;
    auto dtype = convertToFP16_ ? ::torch::kHalf : ::torch::kFloat;
    model_.to(queue, dtype);
    for (uint32_t batch = 0; batch < inputs.batchCount(); ++batch) {
      cms::torch::alpakatools::TensorCollection<Queue> input, output;
      inputs.materializeBatch(batch, input);
      outputs.materializeBatch(batch, output);
      input.copy(queue);
      auto arguments = cms::torch::alpakatools::detail::convertInput(input, model_.device(), dtype);
      if constexpr (Adapter::channelsLast)
        arguments[0] = arguments[0].toTensor().permute({0, 2, 3, 1});
      auto result = model_.forward(arguments);
      if (!result.isTensor())
        throw cms::Exception("TracksterInference") << "TorchScript forward must return one Tensor";
      auto prediction = result.toTensor();
      auto destination = cms::torch::alpakatools::detail::convertOutput(output, model_.device());
      // Permit the conventional [B] regression output as well as [B, 1].
      if (prediction.dim() == 1 && destination.dim() == 2 && destination.size(1) == 1)
        prediction = prediction.unsqueeze(1);
      if (prediction.sizes() != destination.sizes())
        throw cms::Exception("TracksterInference") << "Unexpected TorchScript output shape: " << prediction.sizes()
                                                   << "; expected " << destination.sizes();
      if (prediction.device() != destination.device() || !prediction.is_floating_point())
        throw cms::Exception("TracksterInference") << "Model output must be floating point on the Alpaka device";
      destination.copy_(prediction);
    }
  }

  template class TracksterInferenceRunner<CNNPIDAdapter>;
  template class TracksterInferenceRunner<DNNPIDAdapter>;
  template class TracksterInferenceRunner<PFNPIDAdapter>;
  template class TracksterInferenceRunner<DNNEnergyAdapter>;
  template class TracksterInferenceRunner<PFNEnergyAdapter>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
