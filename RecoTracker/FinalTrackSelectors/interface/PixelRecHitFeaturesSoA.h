#ifndef RecoTracker_FinalTrackSelectors_PixelRecHitFeaturesSoA_h
#define RecoTracker_FinalTrackSelectors_PixelRecHitFeaturesSoA_h

#include <Eigen/Core>
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace RecHitFeatures {

  constexpr int MaxHitsPerTrack = 16;
  constexpr int HitFeatures = 8;

  struct HitFeature {
    static constexpr int x    = 0;
    static constexpr int y    = 1;
    static constexpr int z    = 2;
    static constexpr int xErr = 3;
    static constexpr int yErr = 4;
    static constexpr int r    = 5;
    static constexpr int eta  = 6;
    static constexpr int phi  = 7;
  };

  using HitMatrixRM = Eigen::Matrix<float, MaxHitsPerTrack, HitFeatures, Eigen::RowMajor>;

  GENERATE_SOA_LAYOUT(PixelRecHitFeaturesSoALayout,
                      SOA_EIGEN_COLUMN(HitMatrixRM, hits)
  );

  using PixelRecHitFeaturesSoA = PixelRecHitFeaturesSoALayout<>;

}  // namespace RecHitFeatures

#endif
