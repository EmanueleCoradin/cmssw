#include "RecoTICL/Inference/interface/alpaka/BatchedInferenceRunner.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  void adaptInferenceInputs(std::vector<::torch::IValue>& arguments,
                            InputTransform transform) {
    if (transform == InputTransform::ImageChannelsLast)
      arguments[0] = arguments[0].toTensor().permute({0, 2, 3, 1});
  }

  BatchedInferenceRunner::BatchedInferenceRunner(std::string const& modelPath, bool convertToFP16)
      : model_{modelPath}, convertToFP16_{convertToFP16} {
    model_.to(convertToFP16_ ? ::torch::kHalf : ::torch::kFloat);
  }

  void BatchedInferenceRunner::run(Queue& queue,
                                   InferenceTensors& inputs,
                                   InferenceTensors& outputs,
                                   InputTransform transform) {
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
      // input.copy(queue); TODO: to be changed
      auto arguments = cms::torch::alpakatools::detail::convertInput(input, model_.device(), dtype);
      adaptInferenceInputs(arguments, transform);
      auto result = model_.forward(arguments);
      if (!result.isTensor())
        throw cms::Exception("TracksterInference") << "TorchScript forward must return one Tensor";
      auto prediction = result.toTensor();
      auto destination = cms::torch::alpakatools::detail::convertOutput(output, model_.device());
      // Permit the conventional [B] regression output as well as [B, 1].
      if (prediction.dim() == 1 && destination.dim() == 2 && destination.size(1) == 1)
        prediction = prediction.unsqueeze(1);
      if (prediction.sizes() != destination.sizes())
        throw cms::Exception("TracksterInference")
            << "Unexpected TorchScript output shape: " << prediction.sizes() << "; expected " << destination.sizes();
      if (prediction.device() != destination.device() || !prediction.is_floating_point())
        throw cms::Exception("TracksterInference") << "Model output must be floating point on the Alpaka device";
      // destination.copy_(prediction); TODO: to be changed
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
