#ifndef DataFormats_L1ScoutingSoA_interface_AssociationMapHost_h
#define DataFormats_L1ScoutingSoA_interface_AssociationMapHost_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/AssociationMapSoA.h"

namespace l1sc {

  using AssociationMapHost = PortableHostCollection<AssociationMapSoA>;

}  // namespace l1sc

#endif  // DataFormats_L1ScoutingSoA_interface_AssociationMapHost_h