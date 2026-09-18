#include <cassert>
#include <cstdio>

#include "DataFormats/HGCalReco/interface/TracksterHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

int main() {
  constexpr int32_t nTracksters = 4;
  constexpr int32_t nVertices = 10;
  constexpr int32_t nEdges = 6;
  constexpr int32_t nTrackIdxs = 3;
  constexpr int32_t nGsfIdxs = 2;
  constexpr int32_t nGlobalSeedingTrackIdxs = 2;

  constexpr std::array<int32_t, ticl::TracksterSoA::blocksNumber> sizes{
      nTracksters,  // tracksters
      nTracksters,
      nVertices,  // vertices
      nTracksters,
      nVertices,  // multiplicity
      nTracksters,
      nEdges,  // edges
      nTracksters,
      nTrackIdxs,  // tracks
      nTracksters,
      nGsfIdxs,  // tracksterGsfTrack
      nTracksters,
      nGlobalSeedingTrackIdxs  // globalSeedingTracks
  };

  auto const& host = cms::alpakatools::host();
  ticl::TracksterHost collection(host, sizes);

  // Test trackster
  auto view = collection.view();

  view.tracksters().raw_energy()[0] = 42.5f;

  float readBack = view.tracksters().raw_energy()[0];
  printf("wrote 42.5, read %f\n", readBack);
  assert(readBack == 42.5f);

  view.tracksters().eigenvectors0(0) = Eigen::Vector3f(1.f, 2.f, 3.f);
  Eigen::Vector3f v = view.tracksters().eigenvectors0(0);
  printf("eigenvectors0 = %f %f %f\n", v(0), v(1), v(2));
  assert(v(0) == 1.f && v(1) == 2.f && v(2) == 3.f);
  printf("eigen round-trip OK\n");

  auto tv = view.tracksters();
  tv[0].barycenterX() = 1.f;
  tv[0].barycenterY() = 0.f;
  tv[0].barycenterZ() = 0.f;
  tv[0].raw_energy() = 10.f;
  tv[0].raw_em_energy() = 4.f;

  const float eta = view.barycenterEta(0);
  const float rawPt = view.rawPt(0);
  const float rawEmPt = view.rawEmPt(0);

  printf("eta=%f raw_pt=%f raw_em_pt=%f\n", eta, rawPt, rawEmPt);

  // Associate all the vertices to the first trackster
  auto vertices = view.vertices();
  vertices.offsets().keys_offsets()[0] = 0;
  for (int32_t i = 1; i <= nTracksters; ++i) {
    vertices.offsets().keys_offsets()[i] = nVertices;
  }

  vertices.content().values()[nVertices - 1] = 7u;

  assert(vertices.count(0) == nVertices);
  assert(vertices[0][nVertices - 1] == 7u);
  assert(vertices.count(1) == 0);

  std::printf("vertices association round-trip OK\n");

  // Test GSF-tracks
  auto gsfTracks = view.tracksterGsfTrack();
  gsfTracks.offsets().keys_offsets()[0] = 0;
  for (int32_t i = 1; i <= nTracksters; ++i) {
    gsfTracks.offsets().keys_offsets()[i] = nGsfIdxs;
  }

  gsfTracks.content().values()[nGsfIdxs - 1] = -11;

  assert(gsfTracks.count(0) == nGsfIdxs);
  assert(gsfTracks[0][nGsfIdxs - 1] == -11);
  assert(gsfTracks.count(1) == 0);
  std::printf("GSF-track association round-trip OK\n");

  printf("all round-trips OK\n");
  return 0;
}
