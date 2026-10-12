#ifndef RecoHGCal_TICL_plugins_alpaka_TracksterInferenceKernels_h
#define RecoHGCal_TICL_plugins_alpaka_TracksterInferenceKernels_h

#include <cstdint>

#include "DataFormats/CaloRecHit/interface/CaloClusterSoA.h"
#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "DataFormats/SoATemplate/interface/SoAConstMultiView.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTICL/Inference/interface/TracksterInferenceSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using LayerClusterEnergyMultiView = SoAConstMultiView<::reco::CaloClusterSoAEnergy<>::ConstView, 3>;
  using LayerClusterPositionMultiView = SoAConstMultiView<::reco::CaloClusterSoAPosition<>::ConstView, 3>;

  void selectTracksters(Queue& queue,
                        ::ticl::TracksterSoA::ConstView inputTracksters,
                        LayerClusterEnergyMultiView clusterEnergies,
                        float minClusterEnergy,
                        uint32_t* selectionMask,
                        uint32_t* selectionOffsets,
                        uint32_t nTracksters);

  void fillInputFeatures(Queue& queue,
                         ::ticl::TracksterSoA::ConstView inputTracksters,
                         LayerClusterEnergyMultiView clusterEnergies,
                         LayerClusterPositionMultiView clusterPositions,
                         uint32_t layersPerEndcap,
                         uint32_t const* selectionMask,
                         uint32_t const* selectionOffsets,
                         uint32_t* selectedTracksters,
                         ::ticl::TracksterInferenceSoA::View features,
                         uint32_t nTracksters);

  void fillPIDProbabilities(Queue& queue,
                            uint32_t const* selectedTracksters,
                            ::ticl::TracksterInferencePIDScoresSoA::ConstView scores,
                            ::ticl::TracksterSoA::View outputTracksters,
                            int32_t total);
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
