#include <Eigen/Core>
#include <xtd/math.h>

#include "RecoTICL/PatternRecognition/plugins/alpaka/TracksterInferenceKernels.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {

    struct SelectTrackstersKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::ticl::TracksterSoA::ConstView inputTracksters,
                                    LayerClusterEnergyMultiView clusterEnergies,
                                    float minClusterEnergy,
                                    uint32_t* selectionMask,
                                    uint32_t nTracksters) const {
        const auto associations = inputTracksters.vertices();

        for (auto tracksterIndex : cms::alpakatools::uniform_elements(acc, nTracksters)) {
          const auto vertices = associations[tracksterIndex];

          float clusterEnergy = 0.f;
          for (auto k = 0u; k < vertices.size(); ++k) {
            clusterEnergy += clusterEnergies[vertices[k]].energy();
          }

          selectionMask[tracksterIndex] = clusterEnergy >= minClusterEnergy ? 1u : 0u;
        }
      }
    };

    struct FillInputFeaturesKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::ticl::TracksterSoA::ConstView inputTracksters,
                                    LayerClusterEnergyMultiView clusterEnergies,
                                    LayerClusterPositionMultiView clusterPositions,
                                    uint32_t layersPerEndcap,
                                    uint32_t const* selectionMask,
                                    uint32_t const* selectionOffsets,
                                    uint32_t* selectedTracksters,
                                    ::ticl::TracksterInferenceSoA::View features,
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

          // Temporary buffers for top-k selection.
          uint8_t seenClusters[nLayers] = {};
          uint32_t minSlot[nLayers] = {};
          float minEnergy[nLayers] = {};

          // Store positions in the Trackster's vertex association array.
          // Both the cluster index and multiplicity can be recovered from k.
          uint32_t selectedVertices[nLayers][maxClusters];

          // Phase 1: retain the most energetic clusters in each layer.
          for (auto k = 0u; k < vertices.size(); ++k) {
            const auto vertexIndex = vertices[k];
            const auto position = clusterPositions[vertexIndex];

            const auto layer = static_cast<uint32_t>(position.layer()) % layersPerEndcap;

            if (layer >= nLayers) {
              continue;
            }

            const auto energy = clusterEnergies[vertexIndex].energy();
            auto& count = seenClusters[layer];

            uint32_t slot;
            bool replacedMinimum = false;

            if (count < maxClusters) {
              // Append while there is room.
              slot = count++;

              if (count == 1 || energy < minEnergy[layer]) {
                minEnergy[layer] = energy;
                minSlot[layer] = slot;
              }
            } else {
              // Replace the least energetic cluster only if necessary.
              if (energy <= minEnergy[layer]) {
                continue;
              }

              slot = minSlot[layer];
              replacedMinimum = true;
            }

            // Store the raw energy and its associated vertex position.
            feature.energy()(layer, slot) = energy;
            selectedVertices[layer][slot] = k;

            if (replacedMinimum) {
              // The previous minimum was replaced; find the new one.
              minSlot[layer] = 0u;

              for (auto j = 1u; j < maxClusters; ++j) {
                if (feature.energy()(layer, j) < feature.energy()(layer, minSlot[layer])) {
                  minSlot[layer] = j;
                }
              }

              minEnergy[layer] = feature.energy()(layer, minSlot[layer]);
            }
          }

          // Phase 2: sort retained clusters by descending raw energy.
          for (auto layer = 0u; layer < nLayers; ++layer) {
            const auto count = static_cast<uint32_t>(seenClusters[layer]);

            for (auto end = count; end > 1u; --end) {
              // The minimum is already known on the first iteration.
              auto minIndex = end == count ? minSlot[layer] : 0u;

              if (end != count) {
                for (auto j = 1u; j < end; ++j) {
                  if (feature.energy()(layer, j) < feature.energy()(layer, minIndex)) {
                    minIndex = j;
                  }
                }
              }

              const auto last = end - 1u;

              if (minIndex == last) {
                continue;
              }

              // Swap energy and vertex position together.
              const auto energy = feature.energy()(layer, minIndex);
              const auto vertex = selectedVertices[layer][minIndex];

              feature.energy()(layer, minIndex) = feature.energy()(layer, last);
              selectedVertices[layer][minIndex] = selectedVertices[layer][last];

              feature.energy()(layer, last) = energy;
              selectedVertices[layer][last] = vertex;
            }
          }

          // Phase 3: convert retained clusters into CNN input features.
          for (auto layer = 0u; layer < nLayers; ++layer) {
            for (auto slot = 0u; slot < seenClusters[layer]; ++slot) {
              const auto k = selectedVertices[layer][slot];

              const auto vertexIndex = vertices[k];
              const auto multiplicity = multiplicities[k];
              const auto position = clusterPositions[vertexIndex];

              const auto x = position.x();
              const auto y = position.y();
              const auto z = position.z();

              const auto transverse2 = x * x + y * y;
              // Check present in the original cpu implementation
              if (!(transverse2 > 0.f)) {
                feature.energy()(layer, slot) = 0.f;
                feature.absEta()(layer, slot) = 0.f;
                feature.phi()(layer, slot) = 0.f;
                continue;
              }

              const auto transverse = xtd::sqrt(transverse2);
              const auto absZ = z < 0.f ? -z : z;
              const auto magnitude = xtd::sqrt(transverse2 + z * z);

              feature.energy()(layer, slot) /= multiplicity;
              feature.absEta()(layer, slot) = xtd::log((magnitude + absZ) / transverse);
              feature.phi()(layer, slot) = xtd::atan2(y, x);
            }
          }
        }
      }
    };

    struct FillPIDProbabilitiesKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    uint32_t const* selectedTracksters,
                                    ticl::TracksterInferencePIDScoresSoA::ConstView scores,
                                    ticl::TracksterSoA::View outputTracksters,
                                    int32_t total) const {
        for (auto row : cms::alpakatools::uniform_elements(acc, total)) {
          const auto source = scores[row];
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

  void selectTracksters(Queue& queue,
                        ::ticl::TracksterSoA::ConstView inputTracksters,
                        LayerClusterEnergyMultiView clusterEnergies,
                        float minClusterEnergy,
                        uint32_t* selectionMask,
                        uint32_t* selectionOffsets,
                        uint32_t nTracksters) {
    if (nTracksters == 0) {
      return;
    }

    constexpr auto threadsPerBlock = 256u;
    const auto blocks = cms::alpakatools::divide_up_by(nTracksters, threadsPerBlock);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SelectTrackstersKernel{},
                        inputTracksters,
                        clusterEnergies,
                        minClusterEnergy,
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

  void fillInputFeatures(Queue& queue,
                         ::ticl::TracksterSoA::ConstView inputTracksters,
                         LayerClusterEnergyMultiView clusterEnergies,
                         LayerClusterPositionMultiView clusterPositions,
                         uint32_t layersPerEndcap,
                         uint32_t const* selectionMask,
                         uint32_t const* selectionOffsets,
                         uint32_t* selectedTracksters,
                         ::ticl::TracksterInferenceSoA::View features,
                         uint32_t nTracksters) {
    if (nTracksters == 0) {
      return;
    }
    constexpr auto elementsPerBlock = 256u;
    const auto blocks = (nTracksters + elementsPerBlock - 1) / elementsPerBlock;
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, elementsPerBlock);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        FillInputFeaturesKernel{},
                        inputTracksters,
                        clusterEnergies,
                        clusterPositions,
                        layersPerEndcap,
                        selectionMask,
                        selectionOffsets,
                        selectedTracksters,
                        features,
                        nTracksters);
  }

  void fillPIDProbabilities(Queue& queue,
                            uint32_t const* selectedTracksters,
                            ticl::TracksterInferencePIDScoresSoA::ConstView scores,
                            ticl::TracksterSoA::View outputTracksters,
                            int32_t total) {
    constexpr auto elementsPerBlock = 256u;
    const auto blocks = (static_cast<uint32_t>(total) + elementsPerBlock - 1) / elementsPerBlock;
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, elementsPerBlock);

    alpaka::exec<Acc1D>(
        queue, workDiv, FillPIDProbabilitiesKernel{}, selectedTracksters, scores, outputTracksters, total);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
