#ifndef DataFormats_HGCalReco_interface_alpaka_TracksterDevice_h
#define DataFormats_HGCalReco_interface_alpaka_TracksterDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterHost.h"
#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace ticl {
    using namespace ::ticl;

    using ::ticl::TracksterHost;

    using TracksterDevice = PortableCollection<TracksterBlocks>;
  } 
}  

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(ticl::TracksterDevice, ticl::TracksterHost);

#endif