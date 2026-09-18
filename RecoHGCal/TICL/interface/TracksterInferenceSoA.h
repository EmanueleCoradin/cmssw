#ifndef RecoHGCal_TICL_TracksterInferenceSoA_h
#define RecoHGCal_TICL_TracksterInferenceSoA_h

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace ticl {

  inline constexpr int kTracksterCNNLayers = 50;
  inline constexpr int kTracksterCNNClusters = 10;

  using TracksterCNNImage = Eigen::Matrix<float, kTracksterCNNLayers, kTracksterCNNClusters, Eigen::RowMajor>;

  GENERATE_SOA_LAYOUT(TracksterInferenceFeaturesLayout,
      SOA_EIGEN_COLUMN(TracksterCNNImage, energy),
      SOA_EIGEN_COLUMN(TracksterCNNImage, absEta),
      SOA_EIGEN_COLUMN(TracksterCNNImage, phi))
  
  GENERATE_SOA_LAYOUT(TracksterInferencePIDScoresLayout,
      SOA_COLUMN(float, id_probabilities0),
      SOA_COLUMN(float, id_probabilities1),
      SOA_COLUMN(float, id_probabilities2),
      SOA_COLUMN(float, id_probabilities3),
      SOA_COLUMN(float, id_probabilities4),
      SOA_COLUMN(float, id_probabilities5),
      SOA_COLUMN(float, id_probabilities6),
      SOA_COLUMN(float, id_probabilities7))

  using TracksterInferenceSoA = TracksterInferenceFeaturesLayout<>;
  using TracksterInferencePIDScoresSoA = TracksterInferencePIDScoresLayout<>;

}  // namespace ticl

#endif
