#ifndef DataFormats_L1ScoutingSoA_interface_alpaka_SoftTauDeviceTensor_h
#define DataFormats_L1ScoutingSoA_interface_alpaka_SoftTauDeviceTensor_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/SoftTauHostTensor.h"
#include "DataFormats/L1ScoutingSoA/interface/SoftTauTensorSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace l1sc {

    // make the names from the top-level portabletest namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::portabletest namespace
    using namespace ::l1sc;

    using SoftTauInputDeviceTensor = PortableCollection<SoftTauInputTensorSoA>;
    using SoftTauOutputDeviceTensor = PortableCollection<SoftTauOutputTensorSoA>;

  }  // namespace l1sc

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// heterogeneous ml data checks
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(l1sc::SoftTauInputDeviceTensor,
                                      l1sc::SoftTauInputHostTensor);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(l1sc::SoftTauOutputDeviceTensor,
                                      l1sc::SoftTauOutputHostTensor);

#endif  // DataFormats_L1ScoutingSoA_interface_alpaka_SoftTauDeviceTensor_h