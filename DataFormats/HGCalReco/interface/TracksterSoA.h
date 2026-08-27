#ifndef DataFormats_HGCalReco_TracksterSoA_h
#define DataFormats_HGCalReco_TracksterSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include <Eigen/Core>
#include <xtd/xtd.h>

namespace ticl {

  GENERATE_SOA_LAYOUT(TracksterLayout,
                      SOA_COLUMN(float, regressed_energy),
                      SOA_COLUMN(float, raw_energy),
                      SOA_COLUMN(float, boundTime),
                      SOA_COLUMN(float, time),
                      SOA_COLUMN(float, raw_pt),
                      SOA_COLUMN(float, raw_em_pt),
                      SOA_COLUMN(float, raw_em_energy),
                      SOA_COLUMN(float, timeError),
                      SOA_COLUMN(int, seedIndex),
                      // seedID (edm::ProductID) — deferred

                      SOA_COLUMN(float, eigenvalues0),
                      SOA_COLUMN(float, eigenvalues1),
                      SOA_COLUMN(float, eigenvalues2),

                      SOA_EIGEN_COLUMN(Eigen::Vector3f, eigenvectors0),
                      SOA_EIGEN_COLUMN(Eigen::Vector3f, eigenvectors1),
                      SOA_EIGEN_COLUMN(Eigen::Vector3f, eigenvectors2),

                      SOA_COLUMN(float, sigmas0),
                      SOA_COLUMN(float, sigmas1),
                      SOA_COLUMN(float, sigmas2),

                      SOA_COLUMN(float, sigmasPCA0),
                      SOA_COLUMN(float, sigmasPCA1),
                      SOA_COLUMN(float, sigmasPCA2),

                      SOA_COLUMN(float, barycenterX),
                      SOA_COLUMN(float, barycenterY),
                      SOA_COLUMN(float, barycenterZ),

                      SOA_COLUMN(float, id_probabilities0),
                      SOA_COLUMN(float, id_probabilities1),
                      SOA_COLUMN(float, id_probabilities2),
                      SOA_COLUMN(float, id_probabilities3),
                      SOA_COLUMN(float, id_probabilities4),
                      SOA_COLUMN(float, id_probabilities5),
                      SOA_COLUMN(float, id_probabilities6),
                      SOA_COLUMN(float, id_probabilities7),

                      SOA_COLUMN(uint8_t, iterationIndex))

  GENERATE_SOA_LAYOUT(TracksterVerticesLayout,
                      SOA_COLUMN(uint32_t, vertices),
                      SOA_COLUMN(uint32_t, tracksterId),
                      SOA_COLUMN(float, multiplicity))

  GENERATE_SOA_LAYOUT(TracksterEdgesLayout,
                      SOA_COLUMN(uint32_t, edges_inner),
                      SOA_COLUMN(uint32_t, edges_outer),
                      SOA_COLUMN(uint32_t, tracksterId))

  GENERATE_SOA_LAYOUT(TracksterTrackIdxsLayout, SOA_COLUMN(int, track_idxs), SOA_COLUMN(uint32_t, tracksterId))

  GENERATE_SOA_LAYOUT(TracksterGsfTrackIdxsLayout, SOA_COLUMN(int, gsftrack_idxs), SOA_COLUMN(uint32_t, tracksterId))

  // clang-format off
  GENERATE_SOA_BLOCKS(TracksterBlocksLayout,
                      SOA_BLOCK(trackster, TracksterLayout),
                      SOA_BLOCK(tracksterVertices, TracksterVerticesLayout),
                      SOA_BLOCK(tracksterEdge, TracksterEdgesLayout),
                      SOA_BLOCK(tracksterTrackIdxs, TracksterTrackIdxsLayout),
                      SOA_BLOCK(tracksterGsfTrackIdxs, TracksterGsfTrackIdxsLayout),
                      SOA_CONST_VIEW_METHODS(
                        constexpr SOA_HOST_DEVICE inline float barycenterEta(std::integral auto idx) {
                          const auto x = t[idx].barycenterX(), y = t[idx].barycenterY(), z = t[idx].barycenterZ();
                          return -xtd::log(xtd::tan(0.5f * xtd::acos(z / xtd::sqrt(x * x + y * y + z * z))));
                        }
                      )
  )
  // clang-format on

  using TracksterSoA = TracksterBlocksLayout<>;
  using TracksterSoAView = TracksterBlocks::View;
  using TracksterSoAConstView = TracksterBlocks::ConstView;

  /* ALPAKA_FN_HOST_ACC inline void calculateRawPt(TracksterSoAView &t, int32_t i) { */
  /*   t[i].raw_pt() = t[i].raw_energy() / std::cosh(barycenterEta(t, i)); */
  /* } */

  /* ALPAKA_FN_HOST_ACC inline void calculateRawEmPt(TracksterSoAView &t, int32_t i) { */
  /*   t[i].raw_em_pt() = t[i].raw_em_energy() / std::cosh(barycenterEta(t, i)); */
  /* } */

}  // namespace ticl

#endif
