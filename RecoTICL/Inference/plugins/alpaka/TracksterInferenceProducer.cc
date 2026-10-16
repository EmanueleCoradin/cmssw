#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/BatchedTensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceTensorAdapters.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The producer only knows the input, intermediate output and published
  // product types through the adapter. No model-specific SoA registration or
  // physics postprocessing is performed here.
  template <typename Adapter>
  class TracksterInferenceProducerT : public stream::FixedQueueEDProducer<> {
  public:
    using InputDevice = typename Adapter::InputDevice;
    using ModelOutputDevice = typename Adapter::ModelOutputDevice;
    using PublishedSourceDevice = typename Adapter::PublishedSourceDevice;
    using PublishedOutputDevice = typename Adapter::PublishedOutputDevice;

    explicit TracksterInferenceProducerT(edm::ParameterSet const& config)
        : FixedQueueEDProducer<>(config),
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("tracksters"))},
          featuresToken_{consumes(config.getParameter<edm::InputTag>("features"))},
          model_{config.getParameter<edm::FileInPath>("model").fullPath()},
          batchSize_{config.getParameter<uint32_t>("batchSize")},
          convertToFP16_{config.getParameter<bool>("convertToFP16")},
          warmupIterations_{config.getParameter<int>("warmupIterations")},
          outputToken_{produces()} {
      if (batchSize_ <= 0) {
        throw cms::Exception("TracksterInferenceProducer") << "batchSize must be positive";
      }
      if (warmupIterations_ < 0) {
        throw cms::Exception("TracksterInferenceProducer") << "warmupIterations must be non-negative";
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription description;
      description.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersToSoAProducer"));
      description.add<edm::InputTag>("features", edm::InputTag("tracksterInferenceFeaturesProducer", "features"));
      description.add<edm::FileInPath>("model");
      description.add<uint32_t>("batchSize", 64);
      description.add<bool>("convertToFP16", false);
      description.add<int>("warmupIterations", 3);
      descriptions.addWithDefaultLabel(description);
    }

    void beginStream(edm::StreamID, Queue queue) override {
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

    void produce(device::Event& event, device::EventSetup const&) override {
      // Retrieve both products before launching work on this module's queue.
      edm::LogError("TrackstersInference") << "producing";

      const auto& inputFeatures = event.get(featuresToken_);
      const auto& inputTracksters = event.get(trackstersToken_);
      auto& queue = event.queue();

      const int total = detail::checkedSoASize(inputFeatures.const_view().metadata().size(), "inference samples");
      Adapter::validateSampleCount(inputTracksters, total);

      // Preserve the original output format for direct comparison with the
      // previous TracksterInferenceByCNNTorch implementation.
      auto publishedOutput = Adapter::makePublishedOutput(queue, inputTracksters);
      if (total == 0) {
        event.emplace(outputToken_, std::move(publishedOutput));
        return;
      }

      auto predictions = Adapter::makeModelOutput(queue, total);
      auto featureRecords = inputFeatures.const_view().records();
      auto predictionRecords = predictions.view().records();

      InferenceTensors inputs, outputs;

      Adapter::registerInputs(inputs, featureRecords, batchSize_);
      Adapter::registerOutputs(outputs, predictionRecords, batchSize_);
      forward(queue, inputs, outputs);

      // Model-specific application is implemented by the adapter, not here.
      Adapter::applyResults(queue, featureRecords, predictions, publishedOutput, static_cast<uint32_t>(total));
      event.emplace(outputToken_, std::move(publishedOutput));
    }

  private:
    void forward(Queue& queue, InferenceTensors& inputs, InferenceTensors& outputs) {
      if (convertToFP16_) {
        model_.forward(queue, inputs, outputs, ::torch::kHalf);
      } else {
        model_.forward(queue, inputs, outputs);
      }
    }

    device::EDGetToken<PublishedSourceDevice> const trackstersToken_;
    device::EDGetToken<InputDevice> const featuresToken_;
    torch::AlpakaModel model_;
    uint32_t const batchSize_;
    bool const convertToFP16_;
    int const warmupIterations_;
    device::EDPutToken<PublishedOutputDevice> const outputToken_;
  };

  // Keep the existing module type for regression testing. Additional typed
  // specializations can be registered here without duplicating the producer.
  using TracksterInferenceProducer = TracksterInferenceProducerT<CNNPIDAdapter>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterInferenceProducer);
