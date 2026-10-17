#ifndef RecoTICL_Inference_alpaka_BatchedInferenceRunner_h
#define RecoTICL_Inference_alpaka_BatchedInferenceRunner_h

#include <string>
#include <vector>
#include "PhysicsTools/PyTorchAlpaka/interface/BatchedTensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using InferenceTensors = cms::torch::alpakatools::BatchedTensorCollection<Queue>;
  enum class InputTransform { Identity, ImageChannelsLast, FlatRows };

  // Exposed independently so tensor contracts can be tested without a model.
  void adaptInferenceInputs(std::vector<::torch::IValue>& arguments, InputTransform transform,
                            unsigned int featuresPerRow = 1);

  // One instance per EDM stream. Tensor storage and result application belong to
  // the caller; no Trackster collection or reconstruction policy is assumed here.
  class BatchedInferenceRunner {
  public:
    BatchedInferenceRunner(std::string const& modelPath, bool convertToFP16);
    void run(Queue& queue, InferenceTensors& inputs, InferenceTensors& outputs,
             InputTransform transform = InputTransform::Identity, unsigned int featuresPerRow = 1);

  private:
    torch::AlpakaModel model_;
    bool const convertToFP16_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif
