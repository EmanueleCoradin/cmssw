#ifndef L1TriggerScouting_Phase2_plugins_alpaka_L1TScPhase2TkEmRawToDigiKernels_h
#define L1TriggerScouting_Phase2_plugins_alpaka_L1TScPhase2TkEmRawToDigiKernels_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/L1ScoutingSoA/interface/alpaka/TkEmDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/TkEleDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "L1TriggerScouting/Phase2/interface/L1TScPhase2Common.h"
#include "L1TriggerScouting/Phase2/plugins/alpaka/L1TScPhase2BitsEncoding.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  // Device constant memory constructs.
  ALPAKA_STATIC_ACC_MEM_CONSTANT alpaka::DevGlobal<Acc1D, const float> kPi4096;

  class L1TScPhase2TkEmRawToDigiKernels {
  public:
    L1TScPhase2TkEmRawToDigiKernels() = default;
    explicit L1TScPhase2TkEmRawToDigiKernels(Queue& queue);

    void initialize(Queue& queue);

  private:
    inline static std::once_flag init_flag_;
  };

  void decode_tkem(Queue& queue, uint128_t* p_data, TkEmDeviceCollection& tkem);
  void decode_tkele(Queue& queue, uint128_t* p_data, TkEleDeviceCollection& tkele);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels

#endif  // L1TriggerScouting_Phase2_plugins_alpaka_L1TScPhase2TkEmRawToDigiKernels_h