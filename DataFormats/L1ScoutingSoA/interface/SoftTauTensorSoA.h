#ifndef DataFormats_L1ScoutingSoA_interface_SoftTauTensorSoA_h
#define DataFormats_L1ScoutingSoA_interface_SoftTauTensorSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace l1sc {

  using JetFeatures = Eigen::Matrix<float, 16, 10>;
  using PaddingMask = Eigen::Vector<float, 16>;
  GENERATE_SOA_LAYOUT(SoftTauInputTensorLayout,
                      SOA_EIGEN_COLUMN(JetFeatures, features),
                      SOA_EIGEN_COLUMN(PaddingMask, pad_mask))
  using SoftTauInputTensorSoA = SoftTauInputTensorLayout<>;

  GENERATE_SOA_LAYOUT(SoftTauOutputTensorLayout,
                      SOA_COLUMN(float, cls),
                      SOA_COLUMN(float, vz),
                      SOA_COLUMN(float, pt),
                      SOA_COLUMN(float, charge))
  using SoftTauOutputTensorSoA = SoftTauOutputTensorLayout<>;

}  // namespace l1sc

#endif  // DataFormats_L1ScoutingSoA_interface_SoftTauTensorSoA_h