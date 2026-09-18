#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <alpaka/alpaka.hpp>

#include "CondFormats/HGCalObjects/interface/TICLGeomHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLayersHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLookupHost.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/TracksterHost.h"
#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDevice.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceHost.h"
#include "RecoHGCal/TICL/interface/alpaka/TracksterInferenceDevice.h"
#include "RecoHGCal/TICL/plugins/alpaka/TracksterInferenceKernels.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {

    struct BatchIO {
      cms::torch::alpakatools::TensorCollection<Queue> inputs;
      cms::torch::alpakatools::TensorCollection<Queue> outputs;
    };

    int32_t checkedSize(std::size_t size, std::string_view name) {
      if (size > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw cms::Exception("TracksterInferenceByCNNTorch")
            << name << " contains " << size << " entries, which does not fit in an int32_t.";
      }
      return static_cast<int32_t>(size);
    }

    template <typename TracksterView>
    std::array<int32_t, ::ticl::TracksterSoA::blocksNumber> tracksterSizes(TracksterView const& view) {
      return {
          checkedSize(view.tracksters().metadata().size(), "tracksters"),
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

    struct ScatterPIDProbabilities {
      template <typename TAcc, typename ScoreView, typename OutputView>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    uint32_t const* tracksterIndices,
                                    ScoreView scores,
                                    OutputView output,
                                    uint32_t size) const {
        for (auto row : cms::alpakatools::uniform_elements(acc, size)) {
          auto const source = scores[row];
          auto destination = output[tracksterIndices[row]];

          destination.id_probabilities0() = source.id_probabilities0();
          destination.id_probabilities1() = source.id_probabilities1();
          destination.id_probabilities2() = source.id_probabilities2();
          destination.id_probabilities3() = source.id_probabilities3();
          destination.id_probabilities4() = source.id_probabilities4();
          destination.id_probabilities5() = source.id_probabilities5();
          destination.id_probabilities6() = source.id_probabilities6();
          destination.id_probabilities7() = source.id_probabilities7();
        }
      }
    };

  }  // namespace

  class TracksterInferenceByCNNTorch : public stream::FixedQueueEDProducer<edm::stream::WatchRuns> {
  public:
    explicit TracksterInferenceByCNNTorch(edm::ParameterSet const& config)
        : FixedQueueEDProducer<edm::stream::WatchRuns>(config),
          detector_{config.getParameter<std::string>("detector")},
          doBarrel_{detector_ == "Barrel"},
          ticlGeomToken_{esConsumes<TICLGeomHost, CaloGeometryRecord, edm::Transition::BeginRun>(
              edm::ESInputTag("", doBarrel_ ? "withBarrel" : ""))},
          ticlGeomLookupToken_{esConsumes<TICLGeomLookupHost, CaloGeometryRecord, edm::Transition::BeginRun>(
              edm::ESInputTag("", doBarrel_ ? "withBarrel" : ""))},
          ticlGeomLayersToken_{esConsumes<TICLGeomLayersHost, CaloGeometryRecord, edm::Transition::BeginRun>(
              edm::ESInputTag("", ""))},
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("tracksters"))},
          layerClustersToken_{consumes(config.getParameter<edm::InputTag>("layerClusters"))},
          model_{config.getParameter<edm::FileInPath>("model").fullPath()},
          minClusterEnergy_{static_cast<float>(config.getParameter<double>("minClusterEnergy"))},
          batchSize_{config.getParameter<int>("batchSize")},
          convertToFP16_{config.getParameter<bool>("convertToFP16")},
          warmupIterations_{config.getParameter<int>("warmupIterations")},
          trackstersOutToken_{produces()} {
      if (batchSize_ <= 0) {
        throw cms::Exception("TracksterInferenceByCNNTorch") << "batchSize must be positive, got " << batchSize_;
      }

      if (warmupIterations_ < 0) {
        throw cms::Exception("TracksterInferenceByCNNTorch")
            << "warmupIterations must be non-negative, got " << warmupIterations_;
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription description;
      description.add<std::string>("detector", "HGCAL");
      description.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersToSoAProducer"));
      description.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
      description.add<edm::FileInPath>("model");
      description.add<double>("minClusterEnergy", 1.0);
      description.add<int>("batchSize", 64);
      description.add<bool>("convertToFP16", false);
      description.add<int>("warmupIterations", 3);
      descriptions.addWithDefaultLabel(description);
    }

    void beginRun(edm::Run const&, edm::EventSetup const& eventSetup) override {
      rhtools_.setGeometry(eventSetup.getData(ticlGeomToken_),
                           eventSetup.getData(ticlGeomLookupToken_),
                           eventSetup.getData(ticlGeomLayersToken_));
    }

    void beginStream(edm::StreamID, Queue queue) override {
      ticl::TracksterInferenceDevice features(queue, batchSize_);
      ticl::TracksterInferencePIDScoresDevice scores(queue, batchSize_);

      features.zeroInitialise(queue);
      scores.zeroInitialise(queue);

      auto featureRecords = features.view().records();
      auto scoreRecords = scores.view().records();

      for (int iteration = 0; iteration < warmupIterations_; ++iteration) {
        cms::torch::alpakatools::TensorCollection<Queue> inputs(batchSize_);
        cms::torch::alpakatools::TensorCollection<Queue> outputs(batchSize_);

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

        // Keep the warm-up tensors and their backing buffers alive until the
        // asynchronous model execution has completed.
        alpaka::wait(queue);
      }
    }

    void produce(device::Event& event, device::EventSetup const&) override {
      auto& queue = event.queue();
      auto const& inputTracksters = event.get(trackstersToken_);
      auto const inputView = inputTracksters.const_view();
      auto const& layerClusters = event.get(layerClustersToken_);

      auto const nTracksters = static_cast<std::size_t>(inputView.tracksters().metadata().size());
      if (static_cast<std::size_t>(inputView.vertices().keys()) != nTracksters ||
          static_cast<std::size_t>(inputView.multiplicity().keys()) != nTracksters) {
        throw cms::Exception("TracksterInferenceByCNNTorch")
            << "Inconsistent TracksterSoA key counts: tracksters=" << nTracksters
            << ", vertices=" << inputView.vertices().keys()
            << ", multiplicities=" << inputView.multiplicity().keys() << '.';
      }

      // Preserve all scalar fields and all association blocks. Only the PID
      // probability columns of selected Tracksters are replaced below.
      ticl::TracksterDevice outputTracksters(queue, tracksterSizes(inputView));
      alpaka::memcpy(queue, outputTracksters.buffer(), inputTracksters.buffer());

      std::vector<uint32_t> selectedTracksters;
      selectedTracksters.reserve(nTracksters);

      for (std::size_t tracksterIndex = 0; tracksterIndex < nTracksters; ++tracksterIndex) {
        float clusterEnergy = 0.f;
        for (auto vertex : inputView.vertices()[tracksterIndex]) {
          if (vertex >= layerClusters.size()) {
            throw cms::Exception("TracksterInferenceByCNNTorch")
                << "Trackster " << tracksterIndex << " references layer cluster " << vertex
                << ", but the collection contains " << layerClusters.size() << " entries.";
          }

          clusterEnergy += static_cast<float>(layerClusters[vertex].energy());
          if (clusterEnergy >= minClusterEnergy_) {
            selectedTracksters.push_back(static_cast<uint32_t>(tracksterIndex));
            break;
          }
        }
      }

      auto const total = checkedSize(selectedTracksters.size(), "selected Tracksters");
      if (total == 0) {
        event.emplace(trackstersOutToken_, std::move(outputTracksters));
        return;
      }

      ::ticl::TracksterInferenceHost featuresHost(queue, total);
      featuresHost.zeroInitialise();
      auto featuresView = featuresHost.view();

      auto selectedTrackstersHost = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, total);
      std::copy(selectedTracksters.begin(), selectedTracksters.end(), alpaka::getPtrNative(selectedTrackstersHost));

      std::array<int, ::ticl::kTracksterCNNLayers> seenClusters;
      std::vector<int> clusterIndices;

      for (int row = 0; row < total; ++row) {
        auto const tracksterIndex = selectedTracksters[row];
        auto const vertices = inputView.vertices()[tracksterIndex];
        auto const multiplicities = inputView.multiplicity()[tracksterIndex];

        if (vertices.size() != multiplicities.size()) {
          throw cms::Exception("TracksterInferenceByCNNTorch")
              << "Trackster " << tracksterIndex << " has " << vertices.size() << " vertices but "
              << multiplicities.size() << " multiplicities.";
        }

        for (auto vertex : vertices) {
          if (vertex >= layerClusters.size()) {
            throw cms::Exception("TracksterInferenceByCNNTorch")
                << "Trackster " << tracksterIndex << " references layer cluster " << vertex
                << ", but the collection contains " << layerClusters.size() << " entries.";
          }
        }

        clusterIndices.resize(vertices.size());
        std::iota(clusterIndices.begin(), clusterIndices.end(), 0);
        std::sort(clusterIndices.begin(), clusterIndices.end(), [&](int a, int b) {
          return layerClusters[vertices[a]].energy() > layerClusters[vertices[b]].energy();
        });
        seenClusters.fill(0);

        auto feature = featuresView[row];
        for (auto k : clusterIndices) {
          auto const vertex = vertices[k];
          auto const& cluster = layerClusters[vertex];

          if (cluster.hitsAndFractions().empty()) {
            throw cms::Exception("TracksterInferenceByCNNTorch")
                << "Layer cluster " << vertex << " has no hits, so its layer cannot be determined.";
          }

          auto const layer =
              static_cast<int>(rhtools_.getLayerWithOffset(cluster.hitsAndFractions()[0].first)) - 1;
          if (layer < 0 || layer >= ::ticl::kTracksterCNNLayers ||
              seenClusters[layer] >= ::ticl::kTracksterCNNClusters) {
            continue;
          }

          auto const slot = seenClusters[layer]++;
          if (multiplicities[k] == 0.f) {
            throw cms::Exception("TracksterInferenceByCNNTorch")
                << "Trackster " << tracksterIndex << " has zero multiplicity for layer cluster " << vertex << '.';
          }
          feature.energy()(layer, slot) =
              static_cast<float>(cluster.energy() / static_cast<float>(multiplicities[k]));
          feature.absEta()(layer, slot) = static_cast<float>(std::abs(cluster.eta()));
          feature.phi()(layer, slot) = static_cast<float>(cluster.phi());
        }
      }

      ticl::TracksterInferenceDevice featuresDevice(queue, total);
      ticl::TracksterInferencePIDScoresDevice scoresDevice(queue, total);
      auto selectedTrackstersDevice = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, total);

      scoresDevice.zeroInitialise(queue);
      alpaka::memcpy(queue, featuresDevice.buffer(), featuresHost.buffer());
      alpaka::memcpy(queue, selectedTrackstersDevice, selectedTrackstersHost);

      // The two host staging buffers are local to produce(). Keep them alive
      // until their asynchronous H2D copies are complete.
      alpaka::wait(queue);

      auto featureRecords = featuresDevice.view().records();
      auto scoreRecords = scoresDevice.view().records();
      auto const nBatches = (total + batchSize_ - 1) / batchSize_;
      std::deque<BatchIO> batches;

      for (int batchIndex = 0; batchIndex < nBatches; ++batchIndex) {
        batches.emplace_back(BatchIO{cms::torch::alpakatools::TensorCollection<Queue>(batchSize_, total),
                                     cms::torch::alpakatools::TensorCollection<Queue>(batchSize_, total)});
        auto& batch = batches.back();

        batch.inputs.add<::ticl::TracksterInferenceSoA>("input",
                                                                batchIndex,
                                                                featureRecords.energy(),
                                                                featureRecords.absEta(),
                                                                featureRecords.phi());
        batch.outputs.add<::ticl::TracksterInferencePIDScoresSoA>("pid_output",
                                                                  batchIndex,
                                                                  scoreRecords.id_probabilities0(),
                                                                  scoreRecords.id_probabilities1(),
                                                                  scoreRecords.id_probabilities2(),
                                                                  scoreRecords.id_probabilities3(),
                                                                  scoreRecords.id_probabilities4(),
                                                                  scoreRecords.id_probabilities5(),
                                                                  scoreRecords.id_probabilities6(),
                                                                  scoreRecords.id_probabilities7());
        forward(queue, batch.inputs, batch.outputs);
      }

      fillPIDProbabilities(queue,
                     alpaka::getPtrNative(selectedTrackstersDevice),
                     scoresDevice.const_view(),
                     outputTracksters.view(),
                     total);
      // The inference output is compact. Scatter each row to the corresponding
      // Trackster in the cloned output collection.
      /*
      constexpr uint32_t elementsPerBlock = 256;
      auto const blocks = (static_cast<uint32_t>(total) + elementsPerBlock - 1) / elementsPerBlock;
      auto const workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, 1);
      alpaka::exec<Acc1D>(queue,
                          workDiv,
                          ScatterPIDProbabilities{},
                          alpaka::getPtrNative(selectedTrackstersDevice),
                          scoresDevice.const_view(),
                          outputTracksters.view().tracksters(),
                          static_cast<uint32_t>(total));

      // scoresDevice, the index buffer, and the TensorCollections are local
      // temporaries used by queued work, so they must remain alive until the
      // scatter has completed.*/
      alpaka::wait(queue);

      event.emplace(trackstersOutToken_, std::move(outputTracksters));
    }

  private:
    void forward(Queue& queue,
                 cms::torch::alpakatools::TensorCollection<Queue>& inputs,
                 cms::torch::alpakatools::TensorCollection<Queue>& outputs) {
      if (convertToFP16_) {
        model_.forward(queue, inputs, outputs, ::torch::kHalf);
      } else {
        model_.forward(queue, inputs, outputs);
      }
    }

    std::string const detector_;
    bool const doBarrel_;
    edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> const ticlGeomToken_;
    edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> const ticlGeomLookupToken_;
    edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> const ticlGeomLayersToken_;
    ticlgeom::Tools rhtools_;

    edm::EDGetTokenT<::ticl::TracksterHost> const trackstersToken_;
    edm::EDGetTokenT<std::vector<reco::CaloCluster>> const layerClustersToken_;
    torch::AlpakaModel model_;
    float const minClusterEnergy_;
    int const batchSize_;
    bool const convertToFP16_;
    int const warmupIterations_;
    device::EDPutToken<ticl::TracksterDevice> const trackstersOutToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterInferenceByCNNTorch);
