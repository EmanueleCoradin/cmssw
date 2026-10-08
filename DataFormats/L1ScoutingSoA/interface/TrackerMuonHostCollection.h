#ifndef DataFormats_L1ScoutingSoA_interface_TrackerMuonHostCollection_h
#define DataFormats_L1ScoutingSoA_interface_TrackerMuonHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/TrackerMuonSoA.h"

namespace l1sc {

  using TrackerMuonHostCollection = PortableHostCollection<TrackerMuonSoA>;

}  // namespace l1sc

#endif  // DataFormats_L1ScoutingSoA_interface_TrackerMuonHostCollection_h
