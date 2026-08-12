#include <cassert>
#include <cstdio>

#include "DataFormats/HGCalReco/interface/TracksterHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

int main() {
  constexpr int32_t nTracksters = 4;
  constexpr int32_t nVertices   = 10;
  constexpr int32_t nEdges      = 6;
  constexpr int32_t nTrackIdxs  = 3;
  constexpr int32_t nGsfIdxs    = 2;

  auto const& host = cms::alpakatools::host();
  ticl::TracksterHost collection(host, nTracksters, nVertices, nEdges, nTrackIdxs, nGsfIdxs);

  auto view = collection.view();

  view.trackster().raw_energy()[0] = 42.5f;

  float readBack = view.trackster().raw_energy()[0];
  printf("wrote 42.5, read %f\n", readBack);
  assert(readBack == 42.5f);

  view.tracksterVertices().tracksterId()[0] = 7u;
  assert(view.tracksterVertices().tracksterId()[0] == 7u);
  printf("vertices block round-trip OK\n");

  view.trackster().eigenvectors0(0) = Eigen::Vector3f(1.f, 2.f, 3.f);
  Eigen::Vector3f v = view.trackster().eigenvectors0(0);
  printf("eigenvectors0 = %f %f %f\n", v(0), v(1), v(2));
  assert(v(0) == 1.f && v(1) == 2.f && v(2) == 3.f);
  printf("eigen round-trip OK\n");

  auto tv = view.trackster();
  tv[0].barycenterX() = 1.f;
  tv[0].barycenterY() = 0.f;
  tv[0].barycenterZ() = 0.f;
  tv[0].raw_energy() = 10.f;
  tv[0].raw_em_energy() = 4.f;
  ticl::calculateRawPt(tv, 0);
  ticl::calculateRawEmPt(tv, 0);
  printf("eta=%f raw_pt=%f raw_em_pt=%f\n", ticl::barycenterEta(tv, 0), tv[0].raw_pt(), tv[0].raw_em_pt());
  printf("all round-trips OK\n");
  return 0;
}