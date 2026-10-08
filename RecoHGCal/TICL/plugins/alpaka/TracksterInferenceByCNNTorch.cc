#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>
#include <alpaka/alpaka.hpp>

#include "CondFormats/HGCalObjects/interface/TICLGeomHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLayersHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLookupHost.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDevice.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
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
#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceSoA.h"
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
              checkedSize(view.globalSeedingTracks().content().metadata().size(),
                          "global-seeding-track content")};
    }

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
          model_{config.getParameter<edm::FileInPath>("model").fullPath()},
          minClusterEnergy_{static_cast<float>(config.getParameter<double>("minClusterEnergy"))},
          batchSize_{config.getParameter<int>("batchSize")},
          convertToFP16_{config.getParameter<bool>("convertToFP16")},
          warmupIterations_{config.getParameter<int>("warmupIterations")},
          trackstersOutToken_{produces()} {
      auto const layerClusterTags = config.getParameter<std::vector<edm::InputTag>>("layerClusters");
      if (layerClusterTags.size() != 3) {
        throw cms::Exception("TracksterInferenceByCNNTorch")
            << "layerClusters must contain exactly three device products in merged order: "
               "EE, HSi, HSci; got "
            << layerClusterTags.size() << ".";
      }
      for (auto const& tag : layerClusterTags) {
        layerClustersTokens_.emplace_back(consumes(tag));
      }
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
      description.add<std::vector<edm::InputTag>>(
          "layerClusters",
          {edm::InputTag("hgcalSoALayerClustersEE"),
           edm::InputTag("hgcalSoALayerClustersHSi"),
           edm::InputTag("hgcalSoALayerClustersHSci")});
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
      layersPerEndcap_ = checkedSize(rhtools_.lastLayer(false), "layers per HGCAL endcap");
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
        alpaka::wait(queue);
      }
    }

    void produce(device::Event& event, device::EventSetup const&) override {
      auto& queue = event.queue();
      auto const& inputTracksters = event.get(trackstersToken_);
      auto const inputView = inputTracksters.const_view();
      auto const& layerClustersEE = event.get(layerClustersTokens_[0]);
      auto const& layerClustersHSi = event.get(layerClustersTokens_[1]);
      auto const& layerClustersHSci = event.get(layerClustersTokens_[2]);
      auto const layerClustersEEView = layerClustersEE.view();
      auto const layerClustersHSiView = layerClustersHSi.view();
      auto const layerClustersHSciView = layerClustersHSci.view();

      auto const nTracksters = checkedSize(inputView.tracksters().metadata().size(), "Tracksters");
      auto const nLayerClustersEE = checkedSize(layerClustersEEView.metadata().size()[0], "EE LayerClusters");
      auto const nLayerClustersHSi = checkedSize(layerClustersHSiView.metadata().size()[0], "HSi LayerClusters");
      auto const nLayerClustersHSci = checkedSize(layerClustersHSciView.metadata().size()[0], "HSci LayerClusters");
      // Preserve all Trackster fields and association blocks. The final kernel
      // overwrites only the PID columns of selected Tracksters.
      ticl::TracksterDevice outputTracksters(queue, tracksterSizes(inputView));
      alpaka::memcpy(queue, outputTracksters.buffer(), inputTracksters.buffer());

      if (nTracksters == 0) {
        event.emplace(trackstersOutToken_, std::move(outputTracksters));
        return;
      }

      // Allocate to the worst-case number of selected Tracksters. Compact rows
      // occupy [0, selectedCount) after fillInputFeatures().
      ticl::TracksterInferenceDevice featuresDevice(queue, nTracksters);
      featuresDevice.zeroInitialise(queue);
      auto selectedTrackstersDevice = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracksters);
      auto selectedCountDevice = cms::alpakatools::make_device_buffer<uint32_t>(queue);
      auto selectedCountHost = cms::alpakatools::make_host_buffer<uint32_t>(queue);

      alpaka::memset(queue, selectedCountHost, 0);
      alpaka::memcpy(queue, selectedCountDevice, selectedCountHost);

      LayerClusterEnergyMultiView clusterEnergies;
      LayerClusterPositionMultiView clusterPositions;
      clusterEnergies.addView(layerClustersEE.const_view().energy(), nLayerClustersEE);
      clusterEnergies.addView(layerClustersHSi.const_view().energy(), nLayerClustersHSi);
      clusterEnergies.addView(layerClustersHSci.const_view().energy(), nLayerClustersHSci);
      clusterPositions.addView(layerClustersEE.const_view().position(), nLayerClustersEE);
      clusterPositions.addView(layerClustersHSi.const_view().position(), nLayerClustersHSi);
      clusterPositions.addView(layerClustersHSci.const_view().position(), nLayerClustersHSci);

      fillInputFeatures(queue,
                        inputView,
                        clusterEnergies,
                        clusterPositions,
                        static_cast<uint32_t>(layersPerEndcap_),
                        minClusterEnergy_,
                        alpaka::getPtrNative(selectedTrackstersDevice),
                        alpaka::getPtrNative(selectedCountDevice),
                        featuresDevice.view(),
                        static_cast<uint32_t>(nTracksters));

      // Read back only the compact selection count to construct batches.
      alpaka::memcpy(queue, selectedCountHost, selectedCountDevice);
      alpaka::wait(queue);

      auto const total = checkedSize(*alpaka::getPtrNative(selectedCountHost), "selected Tracksters");
      if (total == 0) {
        event.emplace(trackstersOutToken_, std::move(outputTracksters));
        return;
      }
      if (total > nTracksters) {
        throw cms::Exception("TracksterInferenceByCNNTorch")
            << "Device preprocessing selected " << total << " rows from only " << nTracksters << " Tracksters.";
      }

      ticl::TracksterInferencePIDScoresDevice scoresDevice(queue, total);
      scoresDevice.zeroInitialise(queue);

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

      // Keep temporary TensorCollections and all backing device allocations
      // alive until inference and scatter complete.
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

    device::EDGetToken<ticl::TracksterDevice> const trackstersToken_;
    std::vector<device::EDGetToken<reco::CaloClusterDeviceCollection>> layerClustersTokens_;
    torch::AlpakaModel model_;
    float const minClusterEnergy_;
    int const batchSize_;
    bool const convertToFP16_;
    int const warmupIterations_;
    device::EDPutToken<ticl::TracksterDevice> const trackstersOutToken_;
    int32_t layersPerEndcap_ = 0;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterInferenceByCNNTorch);
