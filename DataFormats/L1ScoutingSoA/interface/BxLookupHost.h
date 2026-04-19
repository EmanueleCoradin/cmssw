#ifndef DataFormats_L1ScoutingSoA_interface_BxLookupHost_h
#define DataFormats_L1ScoutingSoA_interface_BxLookupHost_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/L1ScoutingSoA/interface/BxLookupSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"


namespace l1sc {

  using BxLookupHost = PortableHostCollection<BxLookupSoA>;

}  // namespace l1sc

#endif  // DataFormats_L1ScoutingSoA_interface_BxLookupHost_h