#ifndef RecoHGCal_TICL_TracksterInferenceSoA_h
#define RecoHGCal_TICL_TracksterInferenceSoA_h

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace ticl {

  inline constexpr int kTracksterCNNLayers = 50;
  inline constexpr int kTracksterCNNClusters = 10;

  using TracksterCNNImage = Eigen::Matrix<float, kTracksterCNNLayers, kTracksterCNNClusters, Eigen::ColMajor>;

  GENERATE_SOA_LAYOUT(TracksterInferenceFeaturesLayout,
                      SOA_EIGEN_COLUMN(TracksterCNNImage, energy),
                      SOA_EIGEN_COLUMN(TracksterCNNImage, absEta),
                      SOA_EIGEN_COLUMN(TracksterCNNImage, phi),
                      SOA_COLUMN(uint32_t, selectedTracksterIndex))

  GENERATE_SOA_LAYOUT(TracksterInferencePIDScoresLayout,
                      SOA_COLUMN(float, id_probabilities0),
                      SOA_COLUMN(float, id_probabilities1),
                      SOA_COLUMN(float, id_probabilities2),
                      SOA_COLUMN(float, id_probabilities3),
                      SOA_COLUMN(float, id_probabilities4),
                      SOA_COLUMN(float, id_probabilities5),
                      SOA_COLUMN(float, id_probabilities6),
                      SOA_COLUMN(float, id_probabilities7))

  GENERATE_SOA_LAYOUT(TracksterPFNFeaturesLayout,
                      SOA_EIGEN_COLUMN(TracksterCNNImage, energy),
                      SOA_EIGEN_COLUMN(TracksterCNNImage, absEta),
                      SOA_EIGEN_COLUMN(TracksterCNNImage, phi),
                      SOA_EIGEN_COLUMN(TracksterCNNImage, x),
                      SOA_EIGEN_COLUMN(TracksterCNNImage, y),
                      SOA_EIGEN_COLUMN(TracksterCNNImage, absZ),
                      SOA_EIGEN_COLUMN(TracksterCNNImage, cells),
                      SOA_COLUMN(float, rawEnergy),
                      SOA_COLUMN(float, rawEmEnergy),
                      SOA_COLUMN(float, barycenterX),
                      SOA_COLUMN(float, barycenterY),
                      SOA_COLUMN(float, barycenterAbsZ),
                      SOA_COLUMN(float, barycenterAbsEta),
                      SOA_COLUMN(float, barycenterPhi),
                      SOA_COLUMN(uint32_t, selectedTracksterIndex))

  GENERATE_SOA_LAYOUT(TracksterInferenceEnergyLayout, SOA_COLUMN(float, energy))
  using TracksterPFNFeaturesSoA = TracksterPFNFeaturesLayout<>;
  using TracksterInferenceEnergySoA = TracksterInferenceEnergyLayout<>;

  using TracksterInferenceSoA = TracksterInferenceFeaturesLayout<>;
  using TracksterInferencePIDScoresSoA = TracksterInferencePIDScoresLayout<>;

}  // namespace ticl

#endif
