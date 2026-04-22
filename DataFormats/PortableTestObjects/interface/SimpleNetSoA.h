#ifndef DataFormats_PortableTestObjects_interface_SimpleNetSoA_h
#define DataFormats_PortableTestObjects_interface_SimpleNetSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaMath/interface/float16_t.h"

namespace portabletest {

  GENERATE_SOA_LAYOUT(SimpleNetLayout, SOA_COLUMN(float, reco_pt))
  GENERATE_SOA_LAYOUT(SimpleNetLayoutFP16, SOA_COLUMN(cms::float16_t, reco_pt))

  using SimpleNetSoA = SimpleNetLayout<>;
  using SimpleNetFP16SoA = SimpleNetLayoutFP16<>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_SimpleNetSoA_h
