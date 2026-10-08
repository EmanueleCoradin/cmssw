#ifndef DataFormats_L1ScoutingSoA_interface_alpaka_TrackerMuonDeviceCollection_h
#define DataFormats_L1ScoutingSoA_interface_alpaka_TrackerMuonDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/TrackerMuonHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  using namespace ::l1sc;
  using TrackerMuonDeviceCollection = PortableCollection<TrackerMuonSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(l1sc::TrackerMuonDeviceCollection, l1sc::TrackerMuonHostCollection);

#endif  // DataFormats_L1ScoutingSoA_interface_alpaka_TrackerMuonDeviceCollection_h
