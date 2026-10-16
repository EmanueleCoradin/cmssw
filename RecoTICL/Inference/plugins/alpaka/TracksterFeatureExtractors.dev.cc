#include <cstdint>
#include <tuple>

#include "RecoTICL/Inference/interface/alpaka/TracksterFeatureExtractors.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  CNNFeatureExtractor::CNNFeatureExtractor(edm::ParameterSet const& config)
      : minClusterEnergy_{static_cast<float>(config.getParameter<double>("minClusterEnergy"))} {}

  void CNNFeatureExtractor::fillDescription(edm::ParameterSetDescription& description) {
    description.add<double>("minClusterEnergy", 1.0);
  }

  void CNNFeatureExtractor::select(Queue& queue,
                                   ::ticl::TracksterSoA::ConstView tracksters,
                                   LayerClusterEnergyMultiView clusterEnergies,
                                   uint32_t* selectionMask,
                                   uint32_t* selectionOffsets,
                                   uint32_t nTracksters) const {
    selectTracksters(
        queue, tracksters, clusterEnergies, minClusterEnergy_, selectionMask, selectionOffsets, nTracksters);
  }

  CNNFeatureExtractor::FeaturesDevice CNNFeatureExtractor::makeFeatures(Queue& queue, uint32_t selectedCount) const {
    FeaturesDevice features(queue, selectedCount);
    // The CNN features use a partially filled image, so zero-initialization
    // is required for missing layer clusters.
    features.zeroInitialise(queue);
    return features;
  }

  void CNNFeatureExtractor::fill(Queue& queue,
                                 ::ticl::TracksterSoA::ConstView tracksters,
                                 LayerClusterEnergyMultiView clusterEnergies,
                                 LayerClusterPositionMultiView clusterPositions,
                                 uint32_t layersPerEndcap,
                                 uint32_t const* selectionMask,
                                 uint32_t const* selectionOffsets,
                                 FeaturesDevice& features,
                                 uint32_t nTracksters) const {
    auto featureRecords = features.view().records();
    fillInputFeatures(queue,
                      tracksters,
                      clusterEnergies,
                      clusterPositions,
                      layersPerEndcap,
                      selectionMask,
                      selectionOffsets,
                      std::get<0>(featureRecords.selectedTracksterIndex()).data(),
                      features.view(),
                      nTracksters);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
