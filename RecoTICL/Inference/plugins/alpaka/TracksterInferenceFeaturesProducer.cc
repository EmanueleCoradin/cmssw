#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
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
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterFeatureExtractors.h"

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

  // Common Trackster-to-device-feature lifecycle. The extractor policy
  // chooses how Tracksters are selected and how their feature SoA is filled.
  // It is required to expose:
  //   FeaturesDevice; constructor(config); fillDescription(description);
  //   select(...); makeFeatures(queue, selectedCount); fill(...).
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
          ticlGeomLayersToken_{esConsumes<TICLGeomLayersHost, CaloGeometryRecord, edm::Transition::BeginRun>(
              edm::ESInputTag("", ""))},
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("tracksters"))},
          extractor_{config},
          featuresOutToken_{produces("features")} {
      const auto tags = config.getParameter<std::vector<edm::InputTag>>("layerClusters");
      if (tags.size() != 3) {
        throw cms::Exception("TracksterInferenceFeaturesProducer")
            << "Expected exactly three LayerCluster collections (EE, HSi, HSci)";
      }
      for (const auto& tag : tags) {
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
      const auto& input = event.get(trackstersToken_);
      const auto tracksters = input.const_view();
      const auto nTracksters = checkedSize(tracksters.tracksters().metadata().size(), "Tracksters");
      auto& queue = event.queue();

      selectedCountHost_.emplace(cms::alpakatools::make_host_buffer<uint32_t>(queue));
      *alpaka::getPtrNative(*selectedCountHost_) = 0u;
      selectionMaskDevice_.reset();
      selectionOffsetsDevice_.reset();

      if (nTracksters == 0) {
        return;
      }

      const auto& ee = event.get(layerClustersTokens_[0]);
      const auto& hsi = event.get(layerClustersTokens_[1]);
      const auto& hsci = event.get(layerClustersTokens_[2]);
      const auto nEE = checkedSize(ee.const_view().metadata().size()[0], "EE LayerClusters");
      const auto nHSi = checkedSize(hsi.const_view().metadata().size()[0], "HSi LayerClusters");
      const auto nHSci = checkedSize(hsci.const_view().metadata().size()[0], "HSci LayerClusters");

      LayerClusterEnergyMultiView clusterEnergies;
      clusterEnergies.addView(ee.const_view().energy(), nEE);
      clusterEnergies.addView(hsi.const_view().energy(), nHSi);
      clusterEnergies.addView(hsci.const_view().energy(), nHSci);

      selectionMaskDevice_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracksters));
      selectionOffsetsDevice_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracksters));

      extractor_.select(queue,
                        tracksters,
                        clusterEnergies,
                        selectionMaskDevice_->data(),
                        selectionOffsetsDevice_->data(),
                        static_cast<uint32_t>(nTracksters));

      // select() must produce an inclusive prefix sum; the last offset
      // gives the exact output size. SynchronizingEDProducer waits before produce().
      auto lastOffset = cms::alpakatools::make_device_view(queue, selectionOffsetsDevice_->data() + nTracksters - 1, 1u);
      auto hostCount = cms::alpakatools::make_host_view(alpaka::getPtrNative(*selectedCountHost_), 1u);
      alpaka::memcpy(queue, hostCount, lastOffset);
    }

    void produce(device::Event& event, device::EventSetup const&) override {
      auto& queue = event.queue();
      const auto selectedCount = *alpaka::getPtrNative(*selectedCountHost_);
      const auto& input = event.get(trackstersToken_);
      const auto tracksters = input.const_view();
      const auto nTracksters = checkedSize(tracksters.tracksters().metadata().size(), "Tracksters");

      if (selectedCount > static_cast<uint32_t>(nTracksters)) {
        throw cms::Exception("TracksterInferenceFeaturesProducer")
            << "Selected " << selectedCount << " Tracksters out of " << nTracksters;
      }

      auto features = extractor_.makeFeatures(queue, selectedCount);
      if (selectedCount == 0u) {
        event.emplace(featuresOutToken_, std::move(features));
        selectionMaskDevice_.reset();
        selectionOffsetsDevice_.reset();
        return;
      }

      const auto& ee = event.get(layerClustersTokens_[0]);
      const auto& hsi = event.get(layerClustersTokens_[1]);
      const auto& hsci = event.get(layerClustersTokens_[2]);
      const auto nEE = checkedSize(ee.const_view().metadata().size()[0], "EE LayerClusters");
      const auto nHSi = checkedSize(hsi.const_view().metadata().size()[0], "HSi LayerClusters");
      const auto nHSci = checkedSize(hsci.const_view().metadata().size()[0], "HSci LayerClusters");

      LayerClusterEnergyMultiView clusterEnergies;
      LayerClusterPositionMultiView clusterPositions;
      clusterEnergies.addView(ee.const_view().energy(), nEE);
      clusterEnergies.addView(hsi.const_view().energy(), nHSi);
      clusterEnergies.addView(hsci.const_view().energy(), nHSci);
      clusterPositions.addView(ee.const_view().position(), nEE);
      clusterPositions.addView(hsi.const_view().position(), nHSi);
      clusterPositions.addView(hsci.const_view().position(), nHSci);

      extractor_.fill(queue,
                      tracksters,
                      clusterEnergies,
                      clusterPositions,
                      static_cast<uint32_t>(layersPerEndcap_),
                      selectionMaskDevice_->data(),
                      selectionOffsetsDevice_->data(),
                      features,
                      static_cast<uint32_t>(nTracksters));

      event.emplace(featuresOutToken_, std::move(features));
      selectionMaskDevice_.reset();
      selectionOffsetsDevice_.reset();
    }

  private:
    bool const doBarrel_;
    edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> const ticlGeomToken_;
    edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> const ticlGeomLookupToken_;
    edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> const ticlGeomLayersToken_;

    ticlgeom::Tools rhtools_;

    device::EDGetToken<ticl::TracksterDevice> const trackstersToken_;
    std::vector<device::EDGetToken<reco::CaloClusterDeviceCollection>> layerClustersTokens_;

    Extractor extractor_;
    device::EDPutToken<FeaturesDevice> const featuresOutToken_;

    int32_t layersPerEndcap_ = 0;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> selectionMaskDevice_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> selectionOffsetsDevice_;
    std::optional<cms::alpakatools::host_buffer<uint32_t>> selectedCountHost_;
  };

  // Preserve the module name and product type expected by existing Python
  // configurations and downstream inference modules.
  using TracksterInferenceFeaturesProducer = TracksterInferenceFeaturesProducerT<CNNFeatureExtractor>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterInferenceFeaturesProducer);
