#ifndef DataFormats_PortableTestObjects_interface_HitHostCollection_h
#define DataFormats_PortableTestObjects_interface_HitHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/HitSoA.h"

namespace portabletest {

  using HitHostCollection = PortableHostCollection<HitSoA>;
  using HitOffsetsHostCollection = PortableHostCollection<HitOffsetsSoA>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_HitHostCollection_h
