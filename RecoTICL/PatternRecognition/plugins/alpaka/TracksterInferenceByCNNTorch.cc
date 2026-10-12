#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <string_view>
#include <tuple>
#include <Eigen/Core>
#include <alpaka/alpaka.hpp>
#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDevice.h"
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

#include "RecoTICL/Inference/interface/TracksterInferenceSoA.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceDevice.h"
#include "RecoTICL/PatternRecognition/plugins/alpaka/TracksterInferenceKernels.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace {
    int32_t checkedSize(std::size_t size, std::string_view name) {
      if (size > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw cms::Exception("TracksterInferenceByCNNTorch")
            << name << " contains " << size << " entries, which does not fit in an int32_t.";
      }
      return static_cast<int32_t>(size);
    }
    template <typename TracksterView>
    std::array<int32_t, ::ticl::TracksterSoA::blocksNumber> tracksterSizes(TracksterView const& view) {
      return {checkedSize(view.tracksters().metadata().size(), "tracksters"),
              checkedSize(view.vertices().keys(), "vertices keys"),
              checkedSize(view.vertices().content().metadata().size(), "vertices content"),
              checkedSize(view.multiplicity().keys(), "multiplicity keys"),
              checkedSize(view.multiplicity().content().metadata().size(), "multiplicity content"),
              checkedSize(view.edges().keys(), "edges keys"),
              checkedSize(view.edges().content().metadata().size(), "edges content"),
              checkedSize(view.tracks().keys(), "tracks keys"),
              checkedSize(view.tracks().content().metadata().size(), "tracks content"),
              checkedSize(view.tracksterGsfTrack().keys(), "GSF-track keys"),
              checkedSize(view.tracksterGsfTrack().content().metadata().size(), "GSF-track content"),
              checkedSize(view.globalSeedingTracks().keys(), "global-seeding-track keys"),
              checkedSize(view.globalSeedingTracks().content().metadata().size(), "global-seeding-track content")};
    }
  }  // namespace
  class TracksterInferenceByCNNTorch : public stream::FixedQueueEDProducer<> {
  public:
    explicit TracksterInferenceByCNNTorch(edm::ParameterSet const& config)
        : FixedQueueEDProducer<>(config),
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("tracksters"))},
          featuresToken_{consumes(config.getParameter<edm::InputTag>("features"))},
          model_{config.getParameter<edm::FileInPath>("model").fullPath()},
          batchSize_{config.getParameter<uint32_t>("batchSize")},
          convertToFP16_{config.getParameter<bool>("convertToFP16")},
          warmupIterations_{config.getParameter<int>("warmupIterations")},
          trackstersOutToken_{produces()} {
      if (batchSize_ == 0) {
        throw cms::Exception("TracksterInferenceByCNNTorch") << "batchSize must be positive";
      }
      if (warmupIterations_ < 0) {
        throw cms::Exception("TracksterInferenceByCNNTorch") << "warmupIterations must be non-negative";
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
      ticl::TracksterInferenceDevice features(queue, batchSize_);
      ticl::TracksterInferencePIDScoresDevice scores(queue, batchSize_);
      features.zeroInitialise(queue);
      scores.zeroInitialise(queue);
      auto featureRecords = features.view().records();
      auto scoreRecords = scores.view().records();
      for (int iteration = 0; iteration < warmupIterations_; ++iteration) {
        cms::torch::alpakatools::BatchedTensorCollection<Queue> inputs, outputs;
        inputs.add<::ticl::TracksterInferenceSoA>(
            "input", featureRecords.energy(), featureRecords.absEta(), featureRecords.phi());
        outputs.add<::ticl::TracksterInferencePIDScoresSoA>("pid_output",
                                                            scoreRecords.id_probabilities0(),
                                                            scoreRecords.id_probabilities1(),
                                                            scoreRecords.id_probabilities2(),
                                                            scoreRecords.id_probabilities3(),
                                                            scoreRecords.id_probabilities4(),
                                                            scoreRecords.id_probabilities5(),
                                                            scoreRecords.id_probabilities6(),
                                                            scoreRecords.id_probabilities7());
        forward(queue, inputs, outputs);
      }
    }
    void produce(device::Event& event, device::EventSetup const&) override {
      // Consume first to allow Alpaka to synchronize queue dependencies.
      const auto& features = event.get(featuresToken_);
      const auto& inputTracksters = event.get(trackstersToken_);
      auto& queue = event.queue();
      const auto inputView = inputTracksters.const_view();
      const auto nTracksters = checkedSize(inputView.tracksters().metadata().size(), "Tracksters");
      const auto total = checkedSize(features.const_view().metadata().size(), "selected Tracksters");
      if (total > nTracksters) {
        throw cms::Exception("TracksterInferenceByCNNTorch")
            << "Number of selected Tracksters exceeds the input Trackster count";
      }
      ticl::TracksterDevice outputTracksters(queue, tracksterSizes(inputView));
      alpaka::memcpy(queue, outputTracksters.buffer(), inputTracksters.buffer());
      if (total == 0) {
        event.emplace(trackstersOutToken_, std::move(outputTracksters));
        return;
      }
      ticl::TracksterInferencePIDScoresDevice scoresDevice(queue, total);
      scoresDevice.zeroInitialise(queue);
      auto featureRecords = features.const_view().records();
      auto scoreRecords = scoresDevice.view().records();

      cms::torch::alpakatools::BatchedTensorCollection<Queue> inputs, outputs;

      inputs.addBatched<::ticl::TracksterInferenceSoA>(
          "input", batchSize_, featureRecords.energy(), featureRecords.absEta(), featureRecords.phi());
      outputs.addBatched<::ticl::TracksterInferencePIDScoresSoA>("pid_output",
                                                                 batchSize_,
                                                                 scoreRecords.id_probabilities0(),
                                                                 scoreRecords.id_probabilities1(),
                                                                 scoreRecords.id_probabilities2(),
                                                                 scoreRecords.id_probabilities3(),
                                                                 scoreRecords.id_probabilities4(),
                                                                 scoreRecords.id_probabilities5(),
                                                                 scoreRecords.id_probabilities6(),
                                                                 scoreRecords.id_probabilities7());

      forward(queue, inputs, outputs);

      fillPIDProbabilities(queue,
                           std::get<0>(featureRecords.selectedTracksterIndex()).data(),
                           scoresDevice.const_view(),
                           outputTracksters.view(),
                           total);
      event.emplace(trackstersOutToken_, std::move(outputTracksters));
    }

  private:
    void forward(Queue& queue,
                 cms::torch::alpakatools::BatchedTensorCollection<Queue>& inputs,
                 cms::torch::alpakatools::BatchedTensorCollection<Queue>& outputs) {
      if (convertToFP16_) {
        model_.forward(queue, inputs, outputs, ::torch::kHalf);
      } else {
        model_.forward(queue, inputs, outputs);
      }
    }
    device::EDGetToken<ticl::TracksterDevice> const trackstersToken_;
    device::EDGetToken<ticl::TracksterInferenceDevice> const featuresToken_;
    torch::AlpakaModel model_;
    uint32_t const batchSize_;
    bool const convertToFP16_;
    int const warmupIterations_;
    device::EDPutToken<ticl::TracksterDevice> const trackstersOutToken_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterInferenceByCNNTorch);
