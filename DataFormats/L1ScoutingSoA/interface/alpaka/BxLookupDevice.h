#ifndef DataFormats_L1ScoutingSoA_interface_alpaka_BxLookupCollection_h
#define DataFormats_L1ScoutingSoA_interface_alpaka_BxLookupCollection_h

#include "DataFormats/L1ScoutingSoA/interface/BxLookupSoA.h"
#include "DataFormats/L1ScoutingSoA/interface/BxLookupHost.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AssertDeviceMatchesHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/concepts.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  // make the names from the top-level `l1sc` namespace visible for unqualified lookup
  // inside the `ALPAKA_ACCELERATOR_NAMESPACE::l1sc` namespace
  using namespace ::l1sc;

  using BxLookupDevice = PortableCollection<BxLookupSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(l1sc::BxLookupDevice, l1sc::BxLookupHost);

#endif  // DataFormats_L1ScoutingSoA_interface_alpaka_BxLookupCollection_h