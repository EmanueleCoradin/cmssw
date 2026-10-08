#ifndef L1TriggerScouting_Phase2_plugins_alpaka_L1TScPhase2TkEmRawToDigiKernels_h
#define L1TriggerScouting_Phase2_plugins_alpaka_L1TScPhase2TkEmRawToDigiKernels_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/L1ScoutingSoA/interface/alpaka/BxLookupDevice.h"
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

  // V1 unpacker entry points ---------------------------------------------------

  void decode_tkem(Queue& queue, uint128_t* p_data, TkEmDeviceCollection& tkem);
  void decode_tkele(Queue& queue, uint128_t* p_data, TkEleDeviceCollection& tkele);

  // V2 unpacker entry points ---------------------------------------------------

  // Descriptor of one contiguous payload slice: a (bx, stream) fragment of one class.
  // word_offset      : index, in 64-bit words, into the concatenated payload word buffer
  // bx_row           : row index of this slice's bx in the bx lookup tables
  // within_bx_offset : position (in objects) of this slice within its bx's object range
  // count            : number of objects in the slice
  struct RawSlice {
    uint32_t word_offset;
    uint32_t bx_row;
    uint32_t within_bx_offset;
    uint32_t count;
  };

  // Device-side inclusive prefix scan (GPU launches live in the .dev.cc translation unit)
  void prefix_scan_u32(Queue& queue, uint32_t* input, uint32_t* output, uint32_t size);

  // Fill the bx column and the leading offset cell of both bx lookup tables.
  // The offsets themselves (cells 1..nbx) are written beforehand by cms::alpakatools::iterativePrefixScan.
  void fill_lookups(Queue& queue,
                    const uint32_t* bx_arr,
                    uint32_t nbx,
                    BxLookupDevice& tkem_lookup,
                    BxLookupDevice& tkele_lookup);

  // Decode all tkEm and tkEle payload slices in a single kernel launch, reading directly
  // the packed 64-bit payload words and scattering objects into the two device collections.
  // Object positions come from the (already computed) device bx lookup offset columns.
  void decode_v2(Queue& queue,
                 const data_t* words,
                 const RawSlice* slices_tkem,
                 uint32_t nslices_tkem,
                 TkEmDeviceCollection& tkem,
                 const RawSlice* slices_tkele,
                 uint32_t nslices_tkele,
                 TkEleDeviceCollection& tkele,
                 const BxLookupDevice& tkem_lookup,
                 const BxLookupDevice& tkele_lookup);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels

#endif  // L1TriggerScouting_Phase2_plugins_alpaka_L1TScPhase2TkEmRawToDigiKernels_h