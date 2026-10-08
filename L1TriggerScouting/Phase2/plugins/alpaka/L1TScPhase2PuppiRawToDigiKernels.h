#ifndef L1TriggerScouting_Phase2_plugins_alpaka_L1TScPhase2PuppiRawToDigiKernels_h
#define L1TriggerScouting_Phase2_plugins_alpaka_L1TScPhase2PuppiRawToDigiKernels_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/L1ScoutingSoA/interface/alpaka/BxLookupDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "L1TriggerScouting/Phase2/interface/L1TScPhase2Common.h"
#include "L1TriggerScouting/Phase2/plugins/alpaka/L1TScPhase2BitsEncoding.h"
// shares kernels::RawSlice and kernels::prefix_scan_u32 with the TkEm unpacker (same namespace and plugin library)
#include "L1TriggerScouting/Phase2/plugins/alpaka/L1TScPhase2TkEmRawToDigiKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  // Device constant memory constructs.
  ALPAKA_STATIC_ACC_MEM_CONSTANT alpaka::DevGlobal<Acc1D, const int16_t[8]> kPdgid;
  ALPAKA_STATIC_ACC_MEM_CONSTANT alpaka::DevGlobal<Acc1D, const float> kPi720;

  class L1TScPhase2PuppiRawToDigiKernels {
  public:
    L1TScPhase2PuppiRawToDigiKernels() = default;
    explicit L1TScPhase2PuppiRawToDigiKernels(Queue& queue);

    void initialize(Queue& queue);

  private:
    inline static std::once_flag init_flag_;
  };

  void decode_candidates(Queue& queue, data_t* p_data, PuppiDeviceCollection& puppi);
  void decode_headers(Queue& queue, data_t* h_data, BxLookupDevice& bx_lookup, BxLookupDevice& bx_sizes);
  void fillBxLookupPadded(Queue& queue, BxLookupDevice& bx_lookup_padded, unsigned int nele);
  void fillCandsPadded(Queue& queue, BxLookupDevice& bx_lookup, 
                      PuppiDeviceCollection& puppi_padded, 
                      PuppiDeviceCollection& puppi, 
                      unsigned int nele);

  // ---------------- V2 additions (see L1TScPhase2PuppiRawToDigiV2.cc) ---------------------
  // RawSlice and prefix_scan_u32 are shared with the TkEm unpacker (declared in
  // L1TScPhase2TkEmRawToDigiKernels.h, same namespace and plugin library).

  // Fill both bx lookup tables: bx columns, the per-BX sizes column, and offset[0] = 0.
  // The offsets themselves (cells 1..nbx) are written beforehand by prefix_scan_u32.
  void fill_lookups_puppi(Queue& queue,
                          const uint32_t* bx_arr,
                          const uint32_t* cnt_arr,
                          uint32_t nbx,
                          BxLookupDevice& bx_lookup,
                          BxLookupDevice& bx_sizes);

  // Decode all puppi payload slices in a single kernel launch, reading directly the packed
  // 64-bit payload words (one object per word). One thread per slice; object positions come
  // from the (already computed) device bx lookup offset column.
  void decode_candidates_v2(Queue& queue,
                            const data_t* words,
                            const RawSlice* slices,
                            uint32_t nslices,
                            PuppiDeviceCollection& puppi,
                            const BxLookupDevice& bx_lookup);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels

#endif  // L1TriggerScouting_Phase2_plugins_alpaka_L1TScPhase2PuppiRawToDigiKernels_h