#ifndef L1TriggerScouting_Phase2_plugins_alpaka_L1TScPhase2TrackerMuonRawToDigiKernels_h
#define L1TriggerScouting_Phase2_plugins_alpaka_L1TScPhase2TrackerMuonRawToDigiKernels_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/L1ScoutingSoA/interface/alpaka/BxLookupDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/TrackerMuonDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "L1TriggerScouting/Phase2/plugins/alpaka/L1TScPhase2BitsEncoding.h"
// shares kernels::RawSlice and kernels::prefix_scan_u32 with the TkEm unpacker (same namespace and plugin library)
#include "L1TriggerScouting/Phase2/plugins/alpaka/L1TScPhase2TkEmRawToDigiKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  // Device constant memory constructs.
  ALPAKA_STATIC_ACC_MEM_CONSTANT alpaka::DevGlobal<Acc1D, const float> kTrackerMuonPi4096;

  class L1TScPhase2TrackerMuonRawToDigiKernels {
  public:
    L1TScPhase2TrackerMuonRawToDigiKernels() = default;
    explicit L1TScPhase2TrackerMuonRawToDigiKernels(Queue& queue);

    void initialize(Queue& queue);

  private:
    inline static std::once_flag init_flag_;
  };

  // Fill the bx column and the leading offset cell of the bx lookup table.
  // The offsets themselves (cells 1..nbx) are written beforehand by prefix_scan_u32.
  void fill_muon_lookup(Queue& queue, const uint32_t* bx_arr, uint32_t nbx, BxLookupDevice& lookup);

  // Decode all tracker muon payload slices in a single kernel launch, reading directly
  // the packed 64-bit payload words (2 objects per 3 words) and skipping all-zero padding
  // objects, exactly like the legacy unpacker. One thread per slice; object positions come
  // from the (already computed) device bx lookup offset column.
  void decode_tracker_muons(Queue& queue,
                            const data_t* words,
                            const RawSlice* slices,
                            uint32_t nslices,
                            TrackerMuonDeviceCollection& muons,
                            const BxLookupDevice& lookup);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels

#endif  // L1TriggerScouting_Phase2_plugins_alpaka_L1TScPhase2TrackerMuonRawToDigiKernels_h
