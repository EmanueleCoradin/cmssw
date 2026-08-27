#ifndef DataFormats_HGCalReco_TracksterSoA_h
#define DataFormats_HGCalReco_TracksterSoA_h

#include <cstdint>
#include <Eigen/Core>
#include <alpaka/alpaka.hpp>
#include <xtd/xtd.h>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/TICL/interface/AssociationMap.h"

namespace ticl {

  GENERATE_SOA_LAYOUT(TracksterLayout,
                      SOA_COLUMN(float, regressed_energy),
                      SOA_COLUMN(float, raw_energy),
                      SOA_COLUMN(float, boundTime),
                      SOA_COLUMN(float, time),
                      /* SOA_COLUMN(float, raw_pt), */     // keep?
                      /* SOA_COLUMN(float, raw_em_pt), */  // keep?
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

  struct EdgePair {
    uint32_t inner;
    uint32_t outer;
  };

  template <std::size_t Size, bool Boolean>
  using VerticesLayout = typename ticl::AssociationMapLayout<uint32_t, uint32_t>::template Layout<Size, Boolean>;
  template <std::size_t Size, bool Boolean>
  using MultiplicityLayout = typename ticl::AssociationMapLayout<uint32_t, float>::template Layout<Size, Boolean>;
  template <std::size_t Size, bool Boolean>
  using EdgesLayout = typename ticl::AssociationMapLayout<uint32_t, EdgePair>::template Layout<Size, Boolean>;
  template <std::size_t Size, bool Boolean>
  using TracksAssocLayout = typename ticl::AssociationMapLayout<uint32_t, int>::template Layout<Size, Boolean>;
  template <std::size_t Size, bool Boolean>
  using GlobalSeedingTracksAssocLayout =
      typename ticl::AssociationMapLayout<uint32_t, int>::template Layout<Size, Boolean>;

  // clang-format off
  GENERATE_SOA_BLOCKS(TracksterBlocksLayout,
                      SOA_BLOCK(tracksters, TracksterLayout),
                      // TODO: can these two maps be one map containing {vertex, multiplicity} pairs?
                      SOA_BLOCK(vertices, VerticesLayout),
                      SOA_BLOCK(multiplicity, MultiplicityLayout),
                      SOA_BLOCK(edges, EdgesLayout),
                      SOA_BLOCK(tracks, TracksAssocLayout),
                      SOA_BLOCK(globalSeedingTracks, GlobalSeedingTracksAssocLayout),
                      SOA_CONST_VIEW_METHODS(
                        inline constexpr SOA_HOST_DEVICE auto barycenterEta(std::integral auto idx) {
                          const auto x = this->tracksters()[idx].barycenterX();
                          const auto y = this->tracksters()[idx].barycenterY();
                          const auto z = this->tracksters()[idx].barycenterZ();
                          return -xtd::log(xtd::tan(0.5f * xtd::acos(z / xtd::sqrt(x * x + y * y + z * z))));
                        }
                        inline constexpr SOA_HOST_DEVICE auto rawPt(std::integral auto idx) {
                            const auto barycenter = this->barycenterEta(idx);
                            return this->tracksters()[idx].raw_energy() / xtd::cosh(barycenter);
                        }
                        inline constexpr SOA_HOST_DEVICE auto rawEmPt(std::integral auto idx) {
                            const auto barycenter = this->barycenterEta(idx);
                            return this->tracksters()[idx].raw_em_energy() / xtd::cosh(barycenter);
                        }
                      )
  )
  // clang-format on

  using TracksterSoA = TracksterBlocksLayout<>;
  using TracksterSoAView = TracksterSoA::View;
  using TracksterSoAConstView = TracksterSoA::ConstView;

}  // namespace ticl

#endif
