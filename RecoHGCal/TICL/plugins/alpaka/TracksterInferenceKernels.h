#ifndef RecoHGCal_TICL_plugins_alpaka_TracksterInferenceKernels_h
#define RecoHGCal_TICL_plugins_alpaka_TracksterInferenceKernels_h

#include <cstdint>

#include "DataFormats/CaloRecHit/interface/CaloClusterSoA.h"
#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "DataFormats/SoATemplate/interface/SoAConstMultiView.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using LayerClusterEnergyMultiView = SoAConstMultiView<::reco::CaloClusterSoAEnergy<>::ConstView, 3>;
  using LayerClusterPositionMultiView = SoAConstMultiView<::reco::CaloClusterSoAPosition<>::ConstView, 3>;

  // Access the EE, HSi and HSci collections through their merged index space.
  // The input producer must provide valid Trackster association indices.
  // Invalid indices and nonpositive multiplicities are skipped on the device.
  void fillInputFeatures(Queue& queue,
                         ::ticl::TracksterSoA::ConstView inputTracksters,
                         LayerClusterEnergyMultiView clusterEnergies,
                         LayerClusterPositionMultiView clusterPositions,
                         uint32_t layersPerEndcap,
                         float minClusterEnergy,
                         uint32_t* selectedTracksters,
                         uint32_t* selectedCount,
                         ::ticl::TracksterInferenceSoA::View features,
                         uint32_t nTracksters);

  void fillPIDProbabilities(Queue& queue,
                            uint32_t const* selectedTracksters,
                            ::ticl::TracksterInferencePIDScoresSoA::ConstView scores,
                            ::ticl::TracksterSoA::View outputTracksters,
                            int32_t total);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
