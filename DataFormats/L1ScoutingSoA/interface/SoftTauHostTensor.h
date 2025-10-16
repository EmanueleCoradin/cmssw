#ifndef DataFormats_L1ScoutingSoA_interface_SoftTauHostTensor_h
#define DataFormats_L1ScoutingSoA_interface_SoftTauHostTensor_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/SoftTauTensorSoA.h"

namespace l1sc {

  using SoftTauInputHostTensor = PortableHostCollection<SoftTauInputTensorSoA>;
  using SoftTauOutputHostTensor = PortableHostCollection<SoftTauOutputTensorSoA>;

}  // namespace l1sc

#endif  // DataFormats_L1ScoutingSoA_interface_SoftTauHostTensor_h