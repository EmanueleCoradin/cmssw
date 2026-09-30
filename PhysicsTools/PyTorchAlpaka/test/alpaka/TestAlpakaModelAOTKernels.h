#ifndef PhysicsTools_PyTorchAlpaka_alpaka_TestAlpakaModelAOTKernels_h
#define PhysicsTools_PyTorchAlpaka_alpaka_TestAlpakaModelAOTKernels_h

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  GENERATE_SOA_LAYOUT(SoAInputTemplate,
                      SOA_COLUMN(float, x),
                      SOA_COLUMN(float, y),
                      SOA_COLUMN(float, z),
                      SOA_COLUMN(float, a),
                      SOA_COLUMN(float, b),
                      SOA_COLUMN(float, c),
                      SOA_COLUMN(float, d),
                      SOA_COLUMN(float, e),
                      SOA_COLUMN(float, f),
                      SOA_COLUMN(float, g))

  using SoAInput = SoAInputTemplate<>;
  using InputDeviceCollection = PortableCollection<Device, SoAInput>;
  using InputDeviceCollectionView = InputDeviceCollection::View;

  void fill(Queue& queue, InputDeviceCollection& collection);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

#endif
