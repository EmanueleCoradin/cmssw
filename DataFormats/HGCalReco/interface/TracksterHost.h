#ifndef DataFormats_HGCalReco_interface_TracksterHost_H
#define DataFormats_HGCalReco_interface_TracksterHost_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ticl {
  using TracksterHost = PortableHostCollection<ticl::TracksterBlocks>;
} 

#endif 