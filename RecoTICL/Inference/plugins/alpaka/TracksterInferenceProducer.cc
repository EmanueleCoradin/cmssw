#include <cstdint>
#include <utility>
#include <Eigen/Core>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceRunner.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The EDM module manages only framework inputs/outputs and lifecycle.
  // Trackster-specific batching, inference and result scattering live in
  // TracksterInferenceRunner, implemented in the reusable Inference library.
  template <typename Adapter>
  class TracksterInferenceProducerT : public stream::FixedQueueEDProducer<> {
  public:
    using InputDevice = typename Adapter::InputDevice;
    using PublishedSourceDevice = typename Adapter::PublishedSourceDevice;
    using PublishedOutputDevice = typename Adapter::PublishedOutputDevice;

    explicit TracksterInferenceProducerT(edm::ParameterSet const& config)
        : FixedQueueEDProducer<>(config),
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("tracksters"))},
          featuresToken_{consumes(config.getParameter<edm::InputTag>("features"))},
          runner_{config.getParameter<edm::FileInPath>("model").fullPath(),
                  config.getParameter<uint32_t>("batchSize"),
                  config.getParameter<bool>("convertToFP16"),
                  config.getParameter<int>("warmupIterations")},
          outputToken_{produces()} {}

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

    void beginStream(edm::StreamID, Queue queue) override { runner_.warmup(queue); }

    void produce(device::Event& event, device::EventSetup const&) override {
      // Retrieve input products first so Alpaka handles queue dependencies.
      auto const& features = event.get(featuresToken_);
      auto const& tracksters = event.get(trackstersToken_);
      auto output = runner_.run(event.queue(), features, tracksters);
      event.emplace(outputToken_, std::move(output));
    }

  private:
    device::EDGetToken<PublishedSourceDevice> const trackstersToken_;
    device::EDGetToken<InputDevice> const featuresToken_;
    TracksterInferenceRunner<Adapter> runner_;
    device::EDPutToken<PublishedOutputDevice> const outputToken_;
  };

  using TracksterInferenceProducer = TracksterInferenceProducerT<CNNPIDAdapter>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterInferenceProducer);
