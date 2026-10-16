#ifndef RecoTICL_Inference_alpaka_TracksterFeatureExtractors_h
#define RecoTICL_Inference_alpaka_TracksterFeatureExtractors_h

#include <cstdint>

#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceDevice.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Extractor policy for the legacy Trackster CNN features.
  // The policy owns model-specific configuration and delegates GPU work to
  // the existing TracksterInferenceKernels implementations.
  //
  // A different model can supply its own FeaturesDevice, selection policy,
  // and feature-filling implementation without changing the producer.
  struct CNNFeatureExtractor {
    using FeaturesDevice = ticl::TracksterInferenceDevice;

    explicit CNNFeatureExtractor(edm::ParameterSet const& config);

    static void fillDescription(edm::ParameterSetDescription& description);

    // Must produce an inclusive selection prefix sum in selectionOffsets.
    // The last offset is the number of selected Tracksters.
    void select(Queue& queue,
                ::ticl::TracksterSoA::ConstView tracksters,
                LayerClusterEnergyMultiView clusterEnergies,
                uint32_t* selectionMask,
                uint32_t* selectionOffsets,
                uint32_t nTracksters) const;

    FeaturesDevice makeFeatures(Queue& queue, uint32_t selectedCount) const;

    // Fills the model-specific SoA and its selectedTracksterIndex mapping.
    void fill(Queue& queue,
              ::ticl::TracksterSoA::ConstView tracksters,
              LayerClusterEnergyMultiView clusterEnergies,
              LayerClusterPositionMultiView clusterPositions,
              uint32_t layersPerEndcap,
              uint32_t const* selectionMask,
              uint32_t const* selectionOffsets,
              FeaturesDevice& features,
              uint32_t nTracksters) const;

  private:
    float minClusterEnergy_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Inference_alpaka_TracksterFeatureExtractors_h
