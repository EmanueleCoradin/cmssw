#include <cassert>
#include <cstdio>
#include <alpaka/alpaka.hpp>
#include "DataFormats/HGCalReco/interface/TracksterHost.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

int main() {
  constexpr int32_t nTracksters = 4;
  constexpr int32_t nVertices   = 10;
  constexpr int32_t nEdges      = 6;
  constexpr int32_t nTrackIdxs  = 3;
  constexpr int32_t nGsfIdxs    = 2;
  constexpr int32_t nGlobalSeedingTrackIdxs = 2;

  constexpr std::array<int32_t, ::ticl::TracksterSoA::blocksNumber> sizes{
    nTracksters,                          // tracksters
    nTracksters, nVertices,               // vertices
    nTracksters, nVertices,               // multiplicity
    nTracksters, nEdges,                  // edges
    nTracksters, nTrackIdxs,              // tracks
    nTracksters, nGsfIdxs,                // tracksterGsfTrack
    nTracksters, nGlobalSeedingTrackIdxs  // globalSeedingTracks
  };

  auto const& device = cms::alpakatools::devices<Platform>()[0];
  Queue queue(device);

  ::ticl::TracksterHost src(queue, sizes);
  auto sv = src.view();

  sv.tracksters().raw_energy()[0] = 42.5f;
  sv.tracksters().raw_energy()[nTracksters - 1] = 99.25f;
  sv.tracksters().iterationIndex()[nTracksters - 1] = 3;
  sv.tracksters().eigenvectors0(nTracksters - 1) = Eigen::Vector3f(4.f, 5.f, 6.f);
  
  // Associate all vertices with trackster 0.
  auto vertices = sv.vertices();
  vertices.offsets().keys_offsets()[0] = 0;
  for (int32_t i = 1; i <= nTracksters; ++i) {
    vertices.offsets().keys_offsets()[i] = nVertices;
  }
  vertices.content().values()[nVertices - 1] = 7u;

  // Associate all GSF tracks with trackster 0.
  auto gsfTracks = sv.tracksterGsfTrack();
  gsfTracks.offsets().keys_offsets()[0] = 0;
  for (int32_t i = 1; i <= nTracksters; ++i) {
    gsfTracks.offsets().keys_offsets()[i] = nGsfIdxs;
  }
  gsfTracks.content().values()[nGsfIdxs - 1] = -11;

  ALPAKA_ACCELERATOR_NAMESPACE::ticl::TracksterDevice dev(queue, sizes);
  ::ticl::TracksterHost dst(queue, sizes);

  alpaka::memcpy(queue, dev.buffer(), src.buffer());
  alpaka::memcpy(queue, dst.buffer(), dev.buffer());
  alpaka::wait(queue);

  auto dv = dst.view();
  Eigen::Vector3f e = dv.tracksters().eigenvectors0(nTracksters - 1);

  printf("index 0        : %f\n", dv.tracksters().raw_energy()[0]);
  printf("last trackster : %f\n", dv.tracksters().raw_energy()[nTracksters - 1]);
  printf("last iterIndex : %u\n", (unsigned)dv.tracksters().iterationIndex()[nTracksters - 1]);
  printf("last eigenvec  : %f %f %f\n", e(0), e(1), e(2));
  printf("last vertex    : %u\n", dv.vertices()[0][nVertices - 1]);
  printf("last gsfIdx    : %d\n", dv.tracksterGsfTrack()[0][nGsfIdxs - 1]);

  assert(dv.tracksters().raw_energy()[0] == 42.5f);
  assert(dv.tracksters().raw_energy()[nTracksters - 1] == 99.25f);
  assert(dv.tracksters().iterationIndex()[nTracksters - 1] == 3);
  assert(e(0) == 4.f && e(1) == 5.f && e(2) == 6.f);
  
  assert(dv.vertices().count(0) == nVertices);
  assert(dv.vertices().count(1) == 0);
  assert(dv.vertices()[0][nVertices - 1] == 7u);

  assert(dv.tracksterGsfTrack().count(0) == nGsfIdxs);
  assert(dv.tracksterGsfTrack().count(1) == 0);
  assert(dv.tracksterGsfTrack()[0][nGsfIdxs - 1] == -11);

  printf("host->device->host round-trip OK\n");
  return 0;
}