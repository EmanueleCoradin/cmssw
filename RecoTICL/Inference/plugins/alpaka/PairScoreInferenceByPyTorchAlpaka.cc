#include <algorithm>
#include <limits>
#include <memory>
#include "RecoTICL/Inference/interface/PairScoreInference.h"
#include "RecoTICL/Inference/interface/TracksterInferenceHost.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceDevice.h"
#include "RecoTICL/Inference/interface/alpaka/PairScoreTensorAdapters.h"
#include "RecoTICL/Inference/interface/alpaka/InferenceBridgeUtilities.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class PairScoreInferenceByPyTorchAlpaka : public ::ticl::PairScoreInference {
    // Reuse single-float storage for arbitrary flat tensors. The column name
    // "energy" is historical; here it carries features or compatibility scores.
    using ScalarHost = ::ticl::TracksterInferenceEnergyHost;
    using ScalarDevice = ticl::TracksterInferenceEnergyDevice;
    struct Workspace {
      Workspace(Queue& queue, unsigned int rows, unsigned int features)
          : rows(rows), features(features), inputHost(queue, rows * features), inputDevice(queue, rows * features),
            outputHost(queue, rows), outputDevice(queue, rows) {}
      unsigned int rows, features;
      ScalarHost inputHost;
      ScalarDevice inputDevice;
      ScalarHost outputHost;
      ScalarDevice outputDevice;
    };
  public:
    explicit PairScoreInferenceByPyTorchAlpaka(edm::ParameterSet const& config)
        : queue_(inferenceDevice(config)), batchSize_(inferenceBatchSize(config)),
          runner_(inferenceModelPath(config, "modelPath"), config.getParameter<bool>("convertToFP16")),
          warmupIterations_(config.getParameter<int>("warmupIterations")) {}

    ~PairScoreInferenceByPyTorchAlpaka() override { alpaka::wait(queue_); }

    std::vector<float> evaluate(std::span<float const> input, unsigned int rows, unsigned int features) override {
      if (features != 9 && features != 17 && features != 18)
        throw cms::Exception("Configuration") << "Superclustering requires 9, 17 or 18 features";
      auto values = static_cast<std::size_t>(rows) * features;
      if (input.size() != values || values > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()) ||
          batchSize_ > static_cast<unsigned int>(std::numeric_limits<int32_t>::max()) / features)
        throw cms::Exception("InvalidTensor") << "Invalid superclustering input dimensions or minibatch size";
      if (rows == 0) return {};
      if (!workspace_ || workspace_->rows != rows || workspace_->features != features)
        workspace_ = std::make_unique<Workspace>(queue_, rows, features);
      auto& ws = *workspace_;
      for (std::size_t i = 0; i < values; ++i) ws.inputHost.view()[i].energy() = input[i];
      alpaka::memcpy(queue_, ws.inputDevice.buffer(), ws.inputHost.buffer());
      InferenceTensors inputs, outputs;
      // Mutable private scratch avoids the const-input COW copy. No other
      // consumer observes these buffers, and models must not mutate inputs.
      PairScoreTensorAdapter::registerInputs(inputs, ws.inputDevice.view().records(), batchSize_, features);
      PairScoreTensorAdapter::registerOutputs(outputs, ws.outputDevice.view().records(), batchSize_);
      try {
        // Warm up using the first nonempty batch shape, once per stream.
        for (int i = 0; i < warmupIterations_; ++i)
          runner_.run(queue_, inputs, outputs, InputTransform::FlatRows, features);
        warmupIterations_ = 0;
        runner_.run(queue_, inputs, outputs, InputTransform::FlatRows, features);
        alpaka::memcpy(queue_, ws.outputHost.buffer(), ws.outputDevice.buffer());
        alpaka::wait(queue_);
      } catch (...) {
        alpaka::wait(queue_);
        throw;
      }
      std::vector<float> scores(rows);
      for (unsigned int i = 0; i < rows; ++i) scores[i] = ws.outputHost.const_view()[i].energy();
      return scores;
    }
  private:
    Queue queue_;
    unsigned int const batchSize_;
    BatchedInferenceRunner runner_;
    int warmupIterations_;
    std::unique_ptr<Workspace> workspace_;
  };
}
#define TICL_STRINGIFY_IMPL(x) #x
#define TICL_STRINGIFY(x) TICL_STRINGIFY_IMPL(x)
DEFINE_EDM_PLUGIN(PairScoreInferenceFactory, ALPAKA_ACCELERATOR_NAMESPACE::PairScoreInferenceByPyTorchAlpaka,
                  TICL_STRINGIFY(ALPAKA_ACCELERATOR_NAMESPACE::PairScoreInferenceByPyTorchAlpaka));
#undef TICL_STRINGIFY
#undef TICL_STRINGIFY_IMPL
