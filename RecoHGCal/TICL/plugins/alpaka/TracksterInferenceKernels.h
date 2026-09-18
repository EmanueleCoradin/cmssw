#ifndef RecoHGCal_TICL_plugins_alpaka_TracksterInferenceKernels_h
#define RecoHGCal_TICL_plugins_alpaka_TracksterInferenceKernels_h

#include <cstdint>

#include "DataFormats/CaloRecHit/interface/CaloClusterSoA.h"
#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  enum TracksterPreprocessingError : uint32_t {
    kNoPreprocessingError = 0,
    kInvalidLayerClusterIndex = 1u << 0,
    kMismatchedVerticesAndMultiplicities = 1u << 1,
    kZeroMultiplicity = 1u << 2,
    kInvalidLayer = 1u << 3,
  };

  // Contract for the kernel(s) implemented in TracksterInferenceKernels.dev.cc:
  //
  //  * examine each Trackster from the device TracksterSoA;
  //  * translate its merged LayerCluster index through the concatenated
  //    EE/HSi/HSci ranges into one of the three device CaloCluster SoAs;
  //  * select it when the sum of the energies of its LayerClusters reaches
  //    minClusterEnergy;
  //  * atomically reserve a compact output row in selectedCount;
  //  * write the original Trackster index to selectedTracksters[row];
  //  * zero/fill features[row];
  //  * retain the kTracksterCNNClusters highest-energy LayerClusters per layer;
  //  * write energy / multiplicity, abs(eta), and phi to the CNN feature row;
  //  * atomically OR TracksterPreprocessingError bits into errorMask instead of
  //    attempting to throw from device code.
  //
  // selectedTracksters and features have capacity nTracksters. selectedCount
  // and errorMask point to one device-side uint32_t each and are initialized to
  // zero before this function is called.
  void fillInputFeatures(
      Queue& queue,
      ::ticl::TracksterSoA::ConstView inputTracksters,
      ::reco::CaloClusterSoAConstView layerClustersEE,
      ::reco::CaloClusterSoAConstView layerClustersHSi,
      ::reco::CaloClusterSoAConstView layerClustersHSci,
      uint32_t nLayerClustersEE,
      uint32_t nLayerClustersHSi,
      uint32_t nLayerClustersHSci,
      uint32_t layersPerEndcap,
      float minClusterEnergy,
      uint32_t* selectedTracksters,
      uint32_t* selectedCount,
      uint32_t* errorMask,
      ::ticl::TracksterInferenceSoA::View features,
      uint32_t nTracksters,
      uint32_t nLayerClusters);

  void fillPIDProbabilities(
      Queue& queue,
      uint32_t const* selectedTracksters,
      ::ticl::TracksterInferencePIDScoresSoA::ConstView scores,
      ::ticl::TracksterSoA::View outputTracksters,
      int32_t total);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
