#include "L1TriggerScouting/TauTagging/plugins/alpaka/TransformKernel.h"

#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  using namespace cms::alpakatools;

  SoftTauInputDeviceTensor transform(Queue& queue, 
                 const PFCandidateDeviceCollection& pf, 
                 const BxLookupDeviceCollection& bx_lookup, 
                 const ClustersDeviceCollection& clusters) {
    auto input_tensor = SoftTauInputDeviceTensor(1, queue);
    input_tensor.zeroInitialise(queue);
    return input_tensor; 
  }

  SoftTauInputDeviceTensor transform(Queue& queue, 
                 const PFCandidateDeviceCollection& pf, 
                 const ClustersDeviceCollection& clusters) {
    auto input_tensor = SoftTauInputDeviceTensor(1, queue);
    input_tensor.zeroInitialise(queue);
    return input_tensor;     
  }

  SoftTauInputDeviceTensor transform(Queue& queue, 
                 const PFCandidateDeviceCollection& pf, 
                 const AssociationMapDevice& association_map) {
    auto input_tensor = SoftTauInputDeviceTensor(1, queue);
    input_tensor.zeroInitialise(queue);
    return input_tensor;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels
