#include <Eigen/Core>
#include "RecoHGCal/TICL/plugins/alpaka/TracksterInferenceKernels.h"

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {

    struct LayerClusterData {
      float energy;
      float x;
      float y;
      float z;
      int32_t layer;
    };

    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void setError(TAcc const& acc,
                                                  uint32_t* errorMask,
                                                  TracksterPreprocessingError error) {
      alpaka::atomicOr(acc, errorMask, static_cast<uint32_t>(error));
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE void readLayerCluster(
        ::reco::CaloClusterSoAConstView const& collection, uint32_t localIndex, LayerClusterData& result) {
      result.energy = collection.energy()[localIndex].energy();
      result.x = collection.position()[localIndex].x();
      result.y = collection.position()[localIndex].y();
      result.z = collection.position()[localIndex].z();
      result.layer = collection.position()[localIndex].layer();
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool loadLayerCluster(
        uint32_t mergedIndex,
        ::reco::CaloClusterSoAConstView const& layerClustersEE,
        ::reco::CaloClusterSoAConstView const& layerClustersHSi,
        ::reco::CaloClusterSoAConstView const& layerClustersHSci,
        uint32_t nLayerClustersEE,
        uint32_t nLayerClustersHSi,
        uint32_t nLayerClustersHSci,
        LayerClusterData& result) {
      if (mergedIndex < nLayerClustersEE) {
        readLayerCluster(layerClustersEE, mergedIndex, result);
      } else if (mergedIndex < nLayerClustersEE + nLayerClustersHSi) {
        readLayerCluster(layerClustersHSi, mergedIndex - nLayerClustersEE, result);
      } else if (mergedIndex < nLayerClustersEE + nLayerClustersHSi + nLayerClustersHSci) {
        readLayerCluster(layerClustersHSci, mergedIndex - nLayerClustersEE - nLayerClustersHSi, result);
      } else {
        return false;
      }
      return true;
    }

    // One thread handles one Trackster. The output feature row itself is used
    // as top-k scratch storage, avoiding a 50x10 local array per thread:
    //   energy -> raw LayerCluster energy (sorting key)
    //   absEta -> exactly represented merged LayerCluster index
    //   phi    -> vertex multiplicity
    // A final pass converts those temporary values into the CNN features.
    struct FillInputFeaturesKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(
          TAcc const& acc,
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
          uint32_t nLayerClusters) const {
        auto const verticesAssociations = inputTracksters.vertices();
        auto const multiplicityAssociations = inputTracksters.multiplicity();

        for (auto tracksterIndex : cms::alpakatools::uniform_elements(acc, nTracksters)) {
          if (tracksterIndex >= static_cast<uint32_t>(verticesAssociations.keys()) ||
              tracksterIndex >= static_cast<uint32_t>(multiplicityAssociations.keys())) {
            setError(acc, errorMask, kMismatchedVerticesAndMultiplicities);
            continue;
          }

          auto const vertices = verticesAssociations[tracksterIndex];
          auto const multiplicities = multiplicityAssociations[tracksterIndex];
          if (vertices.size() != multiplicities.size()) {
            setError(acc, errorMask, kMismatchedVerticesAndMultiplicities);
            continue;
          }

          // Validate every merged index while computing the selection energy.
          // LayerCluster energies are non-negative, so checking the final sum
          // is equivalent to the CPU implementation's early threshold exit.
          float clusterEnergy = 0.f;
          bool valid = true;
          for (uint32_t k = 0; k < vertices.size(); ++k) {
            auto const mergedIndex = vertices[k];
            LayerClusterData cluster;
            if (mergedIndex >= nLayerClusters ||
                !loadLayerCluster(mergedIndex,
                                  layerClustersEE,
                                  layerClustersHSi,
                                  layerClustersHSci,
                                  nLayerClustersEE,
                                  nLayerClustersHSi,
                                  nLayerClustersHSci,
                                  cluster)) {
              setError(acc, errorMask, kInvalidLayerClusterIndex);
              valid = false;
              break;
            }
            clusterEnergy += cluster.energy;
          }
          if (!valid || clusterEnergy < minClusterEnergy) {
            continue;
          }

          auto const row = alpaka::atomicAdd(acc, selectedCount, uint32_t{1});
          selectedTracksters[row] = static_cast<uint32_t>(tracksterIndex);
          auto feature = features[row];

          uint8_t seenClusters[::ticl::kTracksterCNNLayers] = {};

          // Maintain a descending-energy insertion list independently in each
          // layer. This is equivalent to the CPU global sort followed by the
          // per-layer cap, without sorting unrelated layers together.
          for (uint32_t k = 0; k < vertices.size(); ++k) {
            auto const mergedIndex = vertices[k];
            auto const multiplicity = multiplicities[k];
            if (multiplicity == 0.f) {
              setError(acc, errorMask, kZeroMultiplicity);
              continue;
            }

            LayerClusterData cluster;
            if (!loadLayerCluster(mergedIndex,
                                  layerClustersEE,
                                  layerClustersHSi,
                                  layerClustersHSci,
                                  nLayerClustersEE,
                                  nLayerClustersHSi,
                                  nLayerClustersHSci,
                                  cluster)) {
              // This was already checked above, but keep the second pass safe.
              setError(acc, errorMask, kInvalidLayerClusterIndex);
              continue;
            }

            if (layersPerEndcap == 0 || cluster.layer < 0 ||
                static_cast<uint32_t>(cluster.layer) >= 2u * layersPerEndcap) {
              setError(acc, errorMask, kInvalidLayer);
              continue;
            }
            auto const layer = static_cast<uint32_t>(cluster.layer) % layersPerEndcap;
            if (layer >= static_cast<uint32_t>(::ticl::kTracksterCNNLayers)) {
              // Preserve the legacy inference behaviour: geometrically valid
              // layers outside the configured CNN image are ignored.
              continue;
            }

            auto const count = static_cast<uint32_t>(seenClusters[layer]);
            auto const stored =
                count < static_cast<uint32_t>(::ticl::kTracksterCNNClusters)
                    ? count
                    : static_cast<uint32_t>(::ticl::kTracksterCNNClusters);
            uint32_t insertion = stored;
            for (uint32_t slot = 0; slot < stored; ++slot) {
              if (cluster.energy > feature.energy()(layer, slot)) {
                insertion = slot;
                break;
              }
            }
            if (insertion >= static_cast<uint32_t>(::ticl::kTracksterCNNClusters)) {
              continue;
            }

            auto const last =
                stored < static_cast<uint32_t>(::ticl::kTracksterCNNClusters)
                    ? stored
                    : static_cast<uint32_t>(::ticl::kTracksterCNNClusters - 1);
            for (uint32_t slot = last; slot > insertion; --slot) {
              feature.energy()(layer, slot) = feature.energy()(layer, slot - 1);
              feature.absEta()(layer, slot) = feature.absEta()(layer, slot - 1);
              feature.phi()(layer, slot) = feature.phi()(layer, slot - 1);
            }
            feature.energy()(layer, insertion) = cluster.energy;
            feature.absEta()(layer, insertion) = static_cast<float>(mergedIndex);
            feature.phi()(layer, insertion) = multiplicity;
            if (count < static_cast<uint32_t>(::ticl::kTracksterCNNClusters)) {
              ++seenClusters[layer];
            }
          }

          // Replace the scratch representation with the three CNN channels.
          for (uint32_t layer = 0; layer < static_cast<uint32_t>(::ticl::kTracksterCNNLayers); ++layer) {
            for (uint32_t slot = 0; slot < static_cast<uint32_t>(seenClusters[layer]); ++slot) {
              auto const rawEnergy = feature.energy()(layer, slot);
              auto const mergedIndex = static_cast<uint32_t>(feature.absEta()(layer, slot));
              auto const multiplicity = feature.phi()(layer, slot);

              LayerClusterData cluster;
              if (!loadLayerCluster(mergedIndex,
                                    layerClustersEE,
                                    layerClustersHSi,
                                    layerClustersHSci,
                                    nLayerClustersEE,
                                    nLayerClustersHSi,
                                    nLayerClustersHSci,
                                    cluster)) {
                setError(acc, errorMask, kInvalidLayerClusterIndex);
                continue;
              }

              auto const transverse2 = cluster.x * cluster.x + cluster.y * cluster.y;
              if (!(transverse2 > 0.f)) {
                setError(acc, errorMask, kInvalidLayerClusterIndex);
                feature.energy()(layer, slot) = 0.f;
                feature.absEta()(layer, slot) = 0.f;
                feature.phi()(layer, slot) = 0.f;
                continue;
              }
              auto const transverse = xtd::sqrt(transverse2);
              auto const absZ = cluster.z < 0.f ? -cluster.z : cluster.z;
              auto const magnitude = xtd::sqrt(transverse2 + cluster.z * cluster.z);

              feature.energy()(layer, slot) = rawEnergy / multiplicity;
              feature.absEta()(layer, slot) = xtd::log((magnitude + absZ) / transverse);
              feature.phi()(layer, slot) = xtd::atan2(cluster.y, cluster.x);
            }
          }
        }
      }
    };

    struct FillPIDProbabilitiesKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(
          TAcc const& acc,
          uint32_t const* selectedTracksters,
          ticl::TracksterInferencePIDScoresSoA::ConstView scores,
          ticl::TracksterSoA::View outputTracksters,
          int32_t total) const {
        for (auto row : cms::alpakatools::uniform_elements(acc, total)) {
          auto const source = scores[row];
          auto destination = outputTracksters.tracksters()[selectedTracksters[row]];

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
      uint32_t nLayerClusters) {
    if (nTracksters == 0) {
      return;
    }
    constexpr uint32_t elementsPerBlock = 256;
    auto const blocks = (nTracksters + elementsPerBlock - 1) / elementsPerBlock;
    auto const workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, elementsPerBlock);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        FillInputFeaturesKernel{},
                        inputTracksters,
                        layerClustersEE,
                        layerClustersHSi,
                        layerClustersHSci,
                        nLayerClustersEE,
                        nLayerClustersHSi,
                        nLayerClustersHSci,
                        layersPerEndcap,
                        minClusterEnergy,
                        selectedTracksters,
                        selectedCount,
                        errorMask,
                        features,
                        nTracksters,
                        nLayerClusters);
  }

  void fillPIDProbabilities(
      Queue& queue,
      uint32_t const* selectedTracksters,
      ticl::TracksterInferencePIDScoresSoA::ConstView scores,
      ticl::TracksterSoA::View outputTracksters,
      int32_t total) {
    constexpr uint32_t elementsPerBlock = 256;
    auto const blocks =
        (static_cast<uint32_t>(total) + elementsPerBlock - 1) / elementsPerBlock;
    auto const workDiv =
        cms::alpakatools::make_workdiv<Acc1D>(blocks, elementsPerBlock);

    alpaka::exec<Acc1D>(
        queue,
        workDiv,
        FillPIDProbabilitiesKernel{},
        selectedTracksters,
        scores,
        outputTracksters,
        total);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE