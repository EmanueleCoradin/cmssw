#include <cstddef>
#include <cstdint>
#include <limits>

#include <Eigen/Core>
#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterFeatureExtractionRunner.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace {
    uint32_t checkedSize(std::size_t size) {
      if (size > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw cms::Exception("TracksterFeatureExtractionRunner")
            << "Trackster collection contains " << size << " entries (exceeds int32_t).";
      }
      return static_cast<uint32_t>(size);
    }
  }  // namespace

  template <typename Extractor>
  TracksterFeatureExtractionRunner<Extractor>::TracksterFeatureExtractionRunner(edm::ParameterSet const& config)
      : extractor_{config} {}

  template <typename Extractor>
  void TracksterFeatureExtractionRunner<Extractor>::prepare(Queue& queue,
                                                            ::ticl::TracksterSoA::ConstView tracksters,
                                                            LayerClusterEnergyMultiView clusterEnergies,
                                                            LayerClusterIndexMultiView clusterIndexes) {
    auto const nTracksters = checkedSize(tracksters.tracksters().metadata().size());

    selectedCountHost_.emplace(cms::alpakatools::make_host_buffer<uint32_t>(queue));
    *alpaka::getPtrNative(*selectedCountHost_) = 0u;
    selectionMaskDevice_.reset();
    selectionOffsetsDevice_.reset();

    if (nTracksters == 0u) {
      return;
    }

    selectionMaskDevice_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracksters));
    selectionOffsetsDevice_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracksters));

    extractor_.select(queue,
                      tracksters,
                      clusterEnergies,
                      clusterIndexes,
                      selectionMaskDevice_->data(),
                      selectionOffsetsDevice_->data(),
                      nTracksters);

    // An inclusive prefix sum's final element is the selected count.
    // SynchronizingEDProducer waits for this transfer before produce().
    auto lastOffset = cms::alpakatools::make_device_view(queue, selectionOffsetsDevice_->data() + nTracksters - 1, 1u);
    auto hostCount = cms::alpakatools::make_host_view(alpaka::getPtrNative(*selectedCountHost_), 1u);
    alpaka::memcpy(queue, hostCount, lastOffset);
  }

  template <typename Extractor>
  uint32_t TracksterFeatureExtractionRunner<Extractor>::selectedCount() const {
    if (!selectedCountHost_) {
      throw cms::Exception("TracksterFeatureExtractionRunner") << "prepare() must be called before selectedCount().";
    }
    return *alpaka::getPtrNative(*selectedCountHost_);
  }

  template <typename Extractor>
  typename TracksterFeatureExtractionRunner<Extractor>::FeaturesDevice TracksterFeatureExtractionRunner<Extractor>::run(
      Queue& queue,
      ::ticl::TracksterSoA::ConstView tracksters,
      LayerClusterEnergyMultiView clusterEnergies,
      LayerClusterPositionMultiView clusterPositions,
      uint32_t layersPerEndcap,
      LayerClusterIndexMultiView clusterIndexes) {
    auto const nTracksters = checkedSize(tracksters.tracksters().metadata().size());
    auto const total = selectedCount();
    if (layersPerEndcap == 0u)
      throw cms::Exception("TracksterFeatureExtractionRunner") << "layersPerEndcap must be positive";
    if (total > nTracksters) {
      throw cms::Exception("TracksterFeatureExtractionRunner")
          << "Selected " << total << " Tracksters out of " << nTracksters << ".";
    }

    auto features = extractor_.makeFeatures(queue, total);
    if (total == 0u) {
      selectionMaskDevice_.reset();
      selectionOffsetsDevice_.reset();
      return features;
    }

    if (!selectionMaskDevice_ || !selectionOffsetsDevice_) {
      throw cms::Exception("TracksterFeatureExtractionRunner") << "Selection buffers are missing.";
    }

    extractor_.fill(queue,
                    tracksters,
                    clusterEnergies,
                    clusterIndexes,
                    clusterPositions,
                    layersPerEndcap,
                    selectionMaskDevice_->data(),
                    selectionOffsetsDevice_->data(),
                    features,
                    nTracksters);

    selectionMaskDevice_.reset();
    selectionOffsetsDevice_.reset();
    return features;
  }

  template class TracksterFeatureExtractionRunner<CNNFeatureExtractor>;
  template class TracksterFeatureExtractionRunner<DNNFeatureExtractor>;
  template class TracksterFeatureExtractionRunner<PFNFeatureExtractor>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
