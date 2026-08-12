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

  auto const& device = cms::alpakatools::devices<Platform>()[0];
  Queue queue(device);

  ::ticl::TracksterHost src(queue, nTracksters, nVertices, nEdges, nTrackIdxs, nGsfIdxs);
  auto sv = src.view();

  sv.trackster().raw_energy()[0] = 42.5f;
  sv.trackster().raw_energy()[nTracksters - 1] = 99.25f;
  sv.trackster().iterationIndex()[nTracksters - 1] = 3;
  sv.trackster().eigenvectors0(nTracksters - 1) = Eigen::Vector3f(4.f, 5.f, 6.f);
  sv.tracksterVertices().tracksterId()[nVertices - 1] = 7u;
  sv.tracksterGsfTrackIdxs().gsftrack_idxs()[nGsfIdxs - 1] = -11;

  ALPAKA_ACCELERATOR_NAMESPACE::ticl::TracksterDevice dev(queue, nTracksters, nVertices, nEdges, nTrackIdxs, nGsfIdxs);
  ::ticl::TracksterHost dst(queue, nTracksters, nVertices, nEdges, nTrackIdxs, nGsfIdxs);

  alpaka::memcpy(queue, dev.buffer(), src.buffer());
  alpaka::memcpy(queue, dst.buffer(), dev.buffer());
  alpaka::wait(queue);

  auto dv = dst.view();
  printf("index 0        : %f\n", dv.trackster().raw_energy()[0]);
  printf("last trackster : %f\n", dv.trackster().raw_energy()[nTracksters - 1]);
  printf("last iterIndex : %u\n", (unsigned)dv.trackster().iterationIndex()[nTracksters - 1]);
  Eigen::Vector3f e = dv.trackster().eigenvectors0(nTracksters - 1);
  printf("last eigenvec  : %f %f %f\n", e(0), e(1), e(2));
  printf("last vertexId  : %u\n", dv.tracksterVertices().tracksterId()[nVertices - 1]);
  printf("last gsfIdx    : %d\n", dv.tracksterGsfTrackIdxs().gsftrack_idxs()[nGsfIdxs - 1]);

  assert(dv.trackster().raw_energy()[0] == 42.5f);
  assert(dv.trackster().raw_energy()[nTracksters - 1] == 99.25f);
  assert(dv.trackster().iterationIndex()[nTracksters - 1] == 3);
  assert(e(0) == 4.f && e(1) == 5.f && e(2) == 6.f);
  assert(dv.tracksterVertices().tracksterId()[nVertices - 1] == 7u);
  assert(dv.tracksterGsfTrackIdxs().gsftrack_idxs()[nGsfIdxs - 1] == -11);

  printf("host->device->host round-trip OK\n");
  return 0;
}
