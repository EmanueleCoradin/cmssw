#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <alpaka/alpaka.hpp>

#include "CondFormats/HGCalObjects/interface/TICLGeomHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLayersHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLookupHost.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDevice.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterFeatureExtractionRunner.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {
    int32_t checkedSize(std::size_t size, std::string_view name) {
      if (size > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw cms::Exception("TracksterInferenceFeaturesProducer")
            << name << " contains " << size << " entries, which does not fit in an int32_t.";
      }
      return static_cast<int32_t>(size);
    }
  }  // namespace

  // The EDM module owns only tokens, geometry setup and the acquire/produce
  // lifecycle. Device selection, count transfer and feature filling live in
  // the reusable TracksterFeatureExtractionRunner.
  template <typename Extractor>
  class TracksterInferenceFeaturesProducerT : public stream::SynchronizingEDProducer<edm::stream::WatchRuns> {
  public:
    using FeaturesDevice = typename Extractor::FeaturesDevice;

    explicit TracksterInferenceFeaturesProducerT(edm::ParameterSet const& config)
        : SynchronizingEDProducer<edm::stream::WatchRuns>(config),
          doBarrel_{config.getParameter<std::string>("detector") == "Barrel"},
          ticlGeomToken_{esConsumes<TICLGeomHost, CaloGeometryRecord, edm::Transition::BeginRun>(
              edm::ESInputTag("", doBarrel_ ? "withBarrel" : ""))},
          ticlGeomLookupToken_{esConsumes<TICLGeomLookupHost, CaloGeometryRecord, edm::Transition::BeginRun>(
              edm::ESInputTag("", doBarrel_ ? "withBarrel" : ""))},
          ticlGeomLayersToken_{
              esConsumes<TICLGeomLayersHost, CaloGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag("", ""))},
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("tracksters"))},
          runner_{config},
          featuresOutToken_{produces("features")} {
      auto const tags = config.getParameter<std::vector<edm::InputTag>>("layerClusters");
      if (tags.size() != 3) {
        throw cms::Exception("TracksterInferenceFeaturesProducer")
            << "Expected exactly three LayerCluster collections (EE, HSi, HSci)";
      }
      for (auto const& tag : tags) {
        layerClustersTokens_.emplace_back(consumes(tag));
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription description;
      description.add<std::string>("detector", "HGCAL");
      description.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersToSoAProducer"));
      description.add<std::vector<edm::InputTag>>("layerClusters",
                                                  {edm::InputTag("hgcalSoALayerClustersEE"),
                                                   edm::InputTag("hgcalSoALayerClustersHSi"),
                                                   edm::InputTag("hgcalSoALayerClustersHSci")});
      Extractor::fillDescription(description);
      descriptions.addWithDefaultLabel(description);
    }

    void beginRun(edm::Run const&, edm::EventSetup const& setup) override {
      rhtools_.setGeometry(
          setup.getData(ticlGeomToken_), setup.getData(ticlGeomLookupToken_), setup.getData(ticlGeomLayersToken_));

      layersPerEndcap_ = checkedSize(rhtools_.lastLayer(false), "layers per HGCAL endcap");
      if (layersPerEndcap_ <= 0) {
        throw cms::Exception("TracksterInferenceFeaturesProducer") << "Invalid number of layers per endcap";
      }
    }

    void acquire(device::Event const& event, device::EventSetup const&) override {
      auto const& input = event.get(trackstersToken_);
      auto const tracksters = input.const_view();
      auto& queue = event.queue();

      // Preserve the original empty-event behavior: no need to retrieve
      // the layer-cluster products if no Tracksters were produced.
      if (tracksters.tracksters().metadata().size() == 0u) {
        runner_.prepare(queue, tracksters, LayerClusterEnergyMultiView{});
        return;
      }

      auto const& ee = event.get(layerClustersTokens_[0]);
      auto const& hsi = event.get(layerClustersTokens_[1]);
      auto const& hsci = event.get(layerClustersTokens_[2]);
      auto const nEE = checkedSize(ee.const_view().metadata().size()[0], "EE LayerClusters");
      auto const nHSi = checkedSize(hsi.const_view().metadata().size()[0], "HSi LayerClusters");
      auto const nHSci = checkedSize(hsci.const_view().metadata().size()[0], "HSci LayerClusters");

      LayerClusterIndexMultiView indexes;
      indexes.addView(ee.const_view().indexes(), nEE);
      indexes.addView(hsi.const_view().indexes(), nHSi);
      indexes.addView(hsci.const_view().indexes(), nHSci);
      LayerClusterEnergyMultiView energies;
      energies.addView(ee.const_view().energy(), nEE);
      energies.addView(hsi.const_view().energy(), nHSi);
      energies.addView(hsci.const_view().energy(), nHSci);
      runner_.prepare(queue, tracksters, energies, indexes);
    }

    void produce(device::Event& event, device::EventSetup const&) override {
      auto& queue = event.queue();
      auto const& input = event.get(trackstersToken_);
      auto const tracksters = input.const_view();

      // The count has been transferred by acquire() and synchronized by
      // SynchronizingEDProducer before this method is called.
      if (runner_.selectedCount() == 0u) {
        auto output = runner_.run(
            queue, tracksters, LayerClusterEnergyMultiView{}, LayerClusterPositionMultiView{}, layersPerEndcap_);
        event.emplace(featuresOutToken_, std::move(output));
        return;
      }

      auto const& ee = event.get(layerClustersTokens_[0]);
      auto const& hsi = event.get(layerClustersTokens_[1]);
      auto const& hsci = event.get(layerClustersTokens_[2]);
      auto const nEE = checkedSize(ee.const_view().metadata().size()[0], "EE LayerClusters");
      auto const nHSi = checkedSize(hsi.const_view().metadata().size()[0], "HSi LayerClusters");
      auto const nHSci = checkedSize(hsci.const_view().metadata().size()[0], "HSci LayerClusters");

      LayerClusterIndexMultiView indexes;
      indexes.addView(ee.const_view().indexes(), nEE);
      indexes.addView(hsi.const_view().indexes(), nHSi);
      indexes.addView(hsci.const_view().indexes(), nHSci);
      LayerClusterEnergyMultiView energies;
      LayerClusterPositionMultiView positions;
      energies.addView(ee.const_view().energy(), nEE);
      energies.addView(hsi.const_view().energy(), nHSi);
      energies.addView(hsci.const_view().energy(), nHSci);
      positions.addView(ee.const_view().position(), nEE);
      positions.addView(hsi.const_view().position(), nHSi);
      positions.addView(hsci.const_view().position(), nHSci);

      auto output =
          runner_.run(queue, tracksters, energies, positions, static_cast<uint32_t>(layersPerEndcap_), indexes);
      event.emplace(featuresOutToken_, std::move(output));
    }

  private:
    bool const doBarrel_;
    edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> const ticlGeomToken_;
    edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> const ticlGeomLookupToken_;
    edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> const ticlGeomLayersToken_;

    ticlgeom::Tools rhtools_;

    device::EDGetToken<ticl::TracksterDevice> const trackstersToken_;
    std::vector<device::EDGetToken<reco::CaloClusterDeviceCollection>> layerClustersTokens_;

    TracksterFeatureExtractionRunner<Extractor> runner_;
    device::EDPutToken<FeaturesDevice> const featuresOutToken_;

    int32_t layersPerEndcap_ = 0;
  };

  // Preserve the existing plugin name and product type.
  using TracksterDNNFeaturesProducer = TracksterInferenceFeaturesProducerT<DNNFeatureExtractor>;
  using TracksterPFNFeaturesProducer = TracksterInferenceFeaturesProducerT<PFNFeatureExtractor>;
  using TracksterInferenceFeaturesProducer = TracksterInferenceFeaturesProducerT<CNNFeatureExtractor>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterInferenceFeaturesProducer);

DEFINE_FWK_ALPAKA_MODULE(TracksterDNNFeaturesProducer);
DEFINE_FWK_ALPAKA_MODULE(TracksterPFNFeaturesProducer);
