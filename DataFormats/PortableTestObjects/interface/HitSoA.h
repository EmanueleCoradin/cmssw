#ifndef DataFormats_PortableTestObjects_interface_HitSoA_h
#define DataFormats_PortableTestObjects_interface_HitSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace portabletest {

  GENERATE_SOA_LAYOUT(HitLayout, SOA_COLUMN(float, x), SOA_COLUMN(float, y), SOA_COLUMN(float, z))
  GENERATE_SOA_LAYOUT(HitOffsetsLayout, SOA_COLUMN(int32_t, offset))

  using HitSoA = HitLayout<>;
  using HitOffsetsSoA = HitOffsetsLayout<>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_HitSoA_h
