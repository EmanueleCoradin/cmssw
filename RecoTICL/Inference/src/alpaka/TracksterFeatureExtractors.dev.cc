#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>

#include <Eigen/Core>
#include <alpaka/alpaka.hpp>
#include <xtd/math.h>

#include "RecoTICL/Inference/interface/alpaka/TracksterFeatureExtractors.h"
#include "RecoTICL/Inference/interface/TracksterInferenceSoA.h"
#include "RecoTICL/Inference/interface/TracksterClusterOrder.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {

    ALPAKA_FN_HOST_ACC float absoluteEta(float x, float y, float z) {
      auto rho = xtd::sqrt(x * x + y * y);
      auto absZ = xtd::abs(z);
      // ROOT::Math::Impl::Eta_FromRhoZ uses |z| + etaMax at rho == 0.
      return rho > 0.f ? xtd::asinh(absZ / rho) : (absZ == 0.f ? 0.f : absZ + 22756.f);
    }

    // Same seed-ID predicate as ticlgeom::Tools::isBarrel().
    ALPAKA_FN_HOST_ACC bool isBarrel(DetId id) {
      return (id.det() == DetId::Ecal && id.subdetId() == 1) ||
             (id.det() == DetId::Hcal && (id.subdetId() == 1 || id.subdetId() == 3));
    }

    template <TracksterModelFamily Family>
    struct SelectTrackstersKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::ticl::TracksterSoA::ConstView inputTracksters,
                                    LayerClusterEnergyMultiView clusterEnergies,
                                    LayerClusterIndexMultiView clusterIndexes,
                                    float minClusterEnergy,
                                    uint32_t* selectionMask,
                                    uint32_t nTracksters) const {
        const auto associations = inputTracksters.vertices();

        for (auto tracksterIndex : cms::alpakatools::uniform_elements(acc, nTracksters)) {
          const auto vertices = associations[tracksterIndex];

          float clusterEnergy = 0.f;
          bool selected = Family == TracksterModelFamily::PFN;
          for (auto k = 0u; k < vertices.size(); ++k) {
            if constexpr (Family != TracksterModelFamily::CNN) {
              if (isBarrel(clusterIndexes[vertices[k]].seedID())) {
                if constexpr (Family == TracksterModelFamily::PFN) {
                  selected = false;
                  break;
                }
                continue;
              }
            }
            if constexpr (Family != TracksterModelFamily::PFN) {
              clusterEnergy += clusterEnergies[vertices[k]].energy();
              if (clusterEnergy >= minClusterEnergy) {
                selected = true;
                break;
              }
            }
          }
          selectionMask[tracksterIndex] = selected ? 1u : 0u;
        }
      }
    };

    struct ClusterBefore {
      LayerClusterEnergyMultiView energies;
      uint32_t const* vertices;
      ALPAKA_FN_HOST_ACC bool operator()(uint32_t a, uint32_t b) const {
        return energies[vertices[a]].energy() > energies[vertices[b]].energy();
      }
    };

    template <TracksterModelFamily Family, typename FeatureView>
    struct FillInputFeaturesKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::ticl::TracksterSoA::ConstView inputTracksters,
                                    LayerClusterEnergyMultiView clusterEnergies,
                                    LayerClusterIndexMultiView clusterIndexes,
                                    LayerClusterPositionMultiView clusterPositions,
                                    uint32_t layersPerEndcap,
                                    uint32_t const* selectionMask,
                                    uint32_t const* selectionOffsets,
                                    uint32_t* selectedTracksters,
                                    FeatureView features,
                                    uint32_t* vertexOrder,
                                    uint32_t nTracksters) const {
        const auto verticesAssociations = inputTracksters.vertices();
        const auto multiplicityAssociations = inputTracksters.multiplicity();

        constexpr auto maxClusters = ::ticl::kTracksterCNNClusters;
        constexpr auto nLayers = ::ticl::kTracksterCNNLayers;

        // One thread processes one Trackster.
        for (auto tracksterIndex : cms::alpakatools::uniform_elements(acc, nTracksters)) {
          // Discard low energetic tracksters
          if (selectionMask[tracksterIndex] == 0u) {
            continue;
          }

          const auto vertices = verticesAssociations[tracksterIndex];
          const auto multiplicities = multiplicityAssociations[tracksterIndex];

          const auto row = selectionOffsets[tracksterIndex] - 1u;
          selectedTracksters[row] = tracksterIndex;
          // Allocate features
          auto feature = features[row];

          if constexpr (Family == TracksterModelFamily::PFN) {
            auto ts = inputTracksters.tracksters()[tracksterIndex];
            auto x = ts.barycenterX();
            auto y = ts.barycenterY();
            auto z = xtd::abs(ts.barycenterZ());
            feature.rawEnergy() = ts.raw_energy();
            feature.rawEmEnergy() = ts.raw_em_energy();
            feature.barycenterX() = x;
            feature.barycenterY() = y;
            feature.barycenterAbsZ() = z;
            feature.barycenterAbsEta() = absoluteEta(x, y, z);
            feature.barycenterPhi() = xtd::atan2(y, x);
          }

          // Sort the complete vertex list, matching the CPU energy-only std::sort
          // including equal energies; top-k alone does not preserve that order.
          auto* order = vertexOrder + verticesAssociations.offsets()[tracksterIndex].keys_offsets();
          for (uint32_t k = 0; k < vertices.size(); ++k)
            order[k] = k;
          ::ticl::detail::TracksterClusterOrder<ClusterBefore>{ClusterBefore{clusterEnergies, vertices.data()}}(
              order, static_cast<int>(vertices.size()));
          uint8_t seenClusters[nLayers] = {};
          for (uint32_t index = 0; index < vertices.size(); ++index) {
            auto k = order[index];
            const auto vertexIndex = vertices[k];
            const auto multiplicity = multiplicities[k];
            if constexpr (Family == TracksterModelFamily::DNN) {
              if (isBarrel(clusterIndexes[vertexIndex].seedID()))
                continue;
            }
            const auto position = clusterPositions[vertexIndex];
            if (position.layer() < 0)
              continue;
            const auto layer = static_cast<uint32_t>(position.layer()) % layersPerEndcap;
            if (layer >= nLayers || seenClusters[layer] >= maxClusters)
              continue;
            const auto slot = seenClusters[layer]++;
            feature.energy()(layer, slot) = clusterEnergies[vertexIndex].energy();

            const auto x = position.x();
            const auto y = position.y();
            const auto z = position.z();

            const auto absZ = xtd::abs(z);

            feature.energy()(layer, slot) /= multiplicity;
            feature.absEta()(layer, slot) = absoluteEta(x, y, z);
            feature.phi()(layer, slot) = xtd::atan2(y, x);
            if constexpr (Family == TracksterModelFamily::PFN) {
              feature.x()(layer, slot) = x;
              feature.y()(layer, slot) = y;
              feature.absZ()(layer, slot) = absZ;
              feature.cells()(layer, slot) = static_cast<float>(position.cells());
            }
          }
        }
      }
    };

  }  // namespace

  template <TracksterModelFamily Family>
  TracksterFeatureExtractor<Family>::TracksterFeatureExtractor(edm::ParameterSet const& config)
      : minClusterEnergy_{static_cast<float>(config.getParameter<double>("minClusterEnergy"))} {
    if (!std::isfinite(minClusterEnergy_))
      throw cms::Exception("Configuration") << "minClusterEnergy must be finite";
  }

  template <TracksterModelFamily Family>
  void TracksterFeatureExtractor<Family>::fillDescription(edm::ParameterSetDescription& description) {
    description.add<double>("minClusterEnergy", 1.0);
  }

  template <TracksterModelFamily Family>
  void TracksterFeatureExtractor<Family>::select(Queue& queue,
                                                 ::ticl::TracksterSoA::ConstView inputTracksters,
                                                 LayerClusterEnergyMultiView clusterEnergies,
                                                 LayerClusterIndexMultiView clusterIndexes,
                                                 uint32_t* selectionMask,
                                                 uint32_t* selectionOffsets,
                                                 uint32_t nTracksters) const {
    if (nTracksters == 0) {
      return;
    }

    constexpr auto threadsPerBlock = 256u;
    const auto blocks = cms::alpakatools::divide_up_by(nTracksters, threadsPerBlock);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SelectTrackstersKernel<Family>{},
                        inputTracksters,
                        clusterEnergies,
                        clusterIndexes,
                        minClusterEnergy_,
                        selectionMask,
                        nTracksters);

    auto blockCounter = cms::alpakatools::make_device_buffer<int32_t>(queue);
    alpaka::memset(queue, blockCounter, 0);

    cms::alpakatools::checkSharedMemoryPrefixScan<Acc1D>(nTracksters, blocks, alpaka::getDev(queue));

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        cms::alpakatools::multiBlockPrefixScan<uint32_t>{},
                        selectionMask,
                        selectionOffsets,
                        nTracksters,
                        static_cast<int32_t>(blocks),
                        blockCounter.data(),
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));
  }

  template <TracksterModelFamily Family>
  typename TracksterFeatureExtractor<Family>::FeaturesDevice TracksterFeatureExtractor<Family>::makeFeatures(
      Queue& queue, uint32_t selectedCount) const {
    FeaturesDevice features(queue, selectedCount);
    // The CNN features use a partially filled image, so zero-initialization
    // is required for missing layer clusters.
    features.zeroInitialise(queue);
    return features;
  }

  template <TracksterModelFamily Family>
  void TracksterFeatureExtractor<Family>::fill(Queue& queue,
                                               ::ticl::TracksterSoA::ConstView inputTracksters,
                                               LayerClusterEnergyMultiView clusterEnergies,
                                               LayerClusterIndexMultiView clusterIndexes,
                                               LayerClusterPositionMultiView clusterPositions,
                                               uint32_t layersPerEndcap,
                                               uint32_t const* selectionMask,
                                               uint32_t const* selectionOffsets,
                                               FeaturesDevice& features,
                                               uint32_t nTracksters) const {
    if (nTracksters == 0) {
      return;
    }
    auto featureRecords = features.view().records();
    auto vertexOrder = cms::alpakatools::make_device_buffer<uint32_t[]>(
        queue, std::max<int32_t>(1, inputTracksters.vertices().content().metadata().size()));
    constexpr auto elementsPerBlock = 256u;
    const auto blocks = (nTracksters + elementsPerBlock - 1) / elementsPerBlock;
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, elementsPerBlock);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        FillInputFeaturesKernel<Family, typename FeaturesDevice::View>{},
                        inputTracksters,
                        clusterEnergies,
                        clusterIndexes,
                        clusterPositions,
                        layersPerEndcap,
                        selectionMask,
                        selectionOffsets,
                        std::get<0>(featureRecords.selectedTracksterIndex()).data(),
                        features.view(),
                        vertexOrder.data(),
                        nTracksters);
  }

  template struct TracksterFeatureExtractor<TracksterModelFamily::CNN>;
  template struct TracksterFeatureExtractor<TracksterModelFamily::DNN>;
  template struct TracksterFeatureExtractor<TracksterModelFamily::PFN>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
