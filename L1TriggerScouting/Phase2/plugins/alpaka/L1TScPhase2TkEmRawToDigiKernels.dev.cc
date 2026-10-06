#include "L1TriggerScouting/Phase2/plugins/alpaka/L1TScPhase2TkEmRawToDigiKernels.h"

#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  L1TScPhase2TkEmRawToDigiKernels::L1TScPhase2TkEmRawToDigiKernels(Queue& queue) { initialize(queue); }

  // Initialize device constant memory for the kernels.
  // Called only once (thread-safe)
  void L1TScPhase2TkEmRawToDigiKernels::initialize(Queue& queue) {
    std::call_once(init_flag_, [&]() {
      // hw to float conversion: 3.14 / 2^12
      constexpr float host_pi_4096 = alpaka::math::constants::pi / 4096.0f;
      auto view_var = createView(cms::alpakatools::host(), &host_pi_4096, Vec1D{1});
      alpaka::memcpy(queue, kPi4096<Acc1D>, view_var);
    });
  }

  // Convert raw data to TkEmDeviceCollection
  // Takes 128b words and decodes them into the corresponding floating-point-based device collection
  class TkEmRawToDigiKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, uint128_t* payload_vec, TkEmDeviceCollection::View tkem) const {
      for (int32_t idx : cms::alpakatools::uniform_elements(acc, tkem.metadata().size())) {
        uint128_t data = payload_vec[idx];

        // hardware values
        auto hwPt = decodeBits<uint16_t, 1, 16>(data);
        auto hwEta = decodeBitsSigned<int16_t, 30, 14>(data);
        auto hwPhi = decodeBitsSigned<int16_t, 17, 13>(data);
        auto hwIso = decodeBits<uint16_t, 48, 11>(data);
        auto quality = decodeBits<uint8_t, 44, 4>(data);

        // convert to real values
        tkem.pt()[idx] = hwPt * 0.03125f;
        tkem.eta()[idx] = hwEta * kPi4096<Acc1D>.get();
        tkem.phi()[idx] = hwPhi * kPi4096<Acc1D>.get();
        tkem.isolation()[idx] = hwIso * 0.25f;

        // assign quality
        tkem.quality()[idx] = quality;
      }
    }
  };

  // Convert raw data to TkEleDeviceCollection
  // Takes 128b words and decodes them into the corresponding floating-point-based device collection
  class TkEleRawToDigiKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, uint128_t* payload_vec, TkEleDeviceCollection::View tkele) const {
      for (int32_t idx : cms::alpakatools::uniform_elements(acc, tkele.metadata().size())) {
        uint128_t data = payload_vec[idx];

        // hardware values
        auto hwPt = decodeBits<uint16_t, 1, 16>(data);
        auto hwEta = decodeBitsSigned<int16_t, 30, 14>(data);
        auto hwPhi = decodeBitsSigned<int16_t, 17, 13>(data);
        auto hwIso = decodeBits<uint16_t, 48, 11>(data);
        auto hwZ0 = decodeBitsSigned<int16_t, 60, 10>(data);
        auto quality = decodeBits<uint8_t, 44, 4>(data);

        // convert to real values
        tkele.pt()[idx] = hwPt * 0.03125f;
        tkele.eta()[idx] = hwEta * kPi4096<Acc1D>.get();
        tkele.phi()[idx] = hwPhi * kPi4096<Acc1D>.get();
        tkele.isolation()[idx] = hwIso * 0.25f;
        tkele.z0()[idx] = hwZ0 * 0.05f;

        // assign quality
        tkele.quality()[idx] = quality;

        // charge
        tkele.charge()[idx] = testBit<59>(data);
      }
    }
  };

  void decode_tkem(Queue& queue, uint128_t* payload_vec, TkEmDeviceCollection& tkem) {
    // move host residing data to device memory space
    auto extent = Vec1D{tkem.const_view().metadata().size()};
    auto payload_vec_device = alpaka::allocAsyncBuf<uint128_t, Idx>(queue, extent);
    alpaka::memcpy(queue, payload_vec_device, createView(cms::alpakatools::host(), payload_vec, extent));

    // grid dims can be tuned for performance
    uint32_t threads_per_block = 1024;
    uint32_t blocks_per_grid = cms::alpakatools::divide_up_by(tkem.const_view().metadata().size(), threads_per_block);
    auto grid = cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

    // decode tkem features
    alpaka::exec<Acc1D>(queue, grid, TkEmRawToDigiKernel{}, payload_vec_device.data(), tkem.view());
  }

  void decode_tkele(Queue& queue, uint128_t* payload_vec, TkEleDeviceCollection& tkele) {
    // move host residing data to device memory space
    auto extent = Vec1D{tkele.const_view().metadata().size()};
    auto payload_vec_device = alpaka::allocAsyncBuf<uint128_t, Idx>(queue, extent);
    alpaka::memcpy(queue, payload_vec_device, createView(cms::alpakatools::host(), payload_vec, extent));

    // grid dims can be tuned for performance
    uint32_t threads_per_block = 1024;
    uint32_t blocks_per_grid = cms::alpakatools::divide_up_by(tkele.const_view().metadata().size(), threads_per_block);
    auto grid = cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

    // decode tkele features
    alpaka::exec<Acc1D>(queue, grid, TkEleRawToDigiKernel{}, payload_vec_device.data(), tkele.view());
  }

  // V2 unpacker kernels ----------------------------------------------------------

  namespace {
    // Reconstruct the 96-bit object payload, zero-extended to 128 bits, directly from the
    // packed 64-bit words (2 objects per 3 words). Bit-identical to the V1 flattenBuffer output.
    ALPAKA_FN_ACC inline uint128_t reconstruct_object(const data_t* w, uint32_t i) {
      const uint32_t k = 3 * (i >> 1);
      const data_t w0 = w[k];
      const data_t w1 = w[k + 1];
      const data_t w2 = w[k + 2];
      return ((i & 1) == 0) ? static_cast<uint128_t>(w0) | (static_cast<uint128_t>(w1 & 0xffffffffULL) << 64)
                            : static_cast<uint128_t>(w2) | (static_cast<uint128_t>(w1 >> 32) << 64);
    }

    ALPAKA_FN_ACC inline void store_tkem(TkEmDeviceCollection::View tkem, int32_t idx, uint128_t data) {
      tkem.pt()[idx] = decodeBits<uint16_t, 1, 16>(data) * 0.03125f;
      tkem.eta()[idx] = decodeBitsSigned<int16_t, 30, 14>(data) * kPi4096<Acc1D>.get();
      tkem.phi()[idx] = decodeBitsSigned<int16_t, 17, 13>(data) * kPi4096<Acc1D>.get();
      tkem.isolation()[idx] = decodeBits<uint16_t, 48, 11>(data) * 0.25f;
      tkem.quality()[idx] = decodeBits<uint8_t, 44, 4>(data);
    }

    ALPAKA_FN_ACC inline void store_tkele(TkEleDeviceCollection::View tkele, int32_t idx, uint128_t data) {
      tkele.pt()[idx] = decodeBits<uint16_t, 1, 16>(data) * 0.03125f;
      tkele.eta()[idx] = decodeBitsSigned<int16_t, 30, 14>(data) * kPi4096<Acc1D>.get();
      tkele.phi()[idx] = decodeBitsSigned<int16_t, 17, 13>(data) * kPi4096<Acc1D>.get();
      tkele.isolation()[idx] = decodeBits<uint16_t, 48, 11>(data) * 0.25f;
      tkele.z0()[idx] = decodeBitsSigned<int16_t, 60, 10>(data) * 0.05f;
      tkele.quality()[idx] = decodeBits<uint8_t, 44, 4>(data);
      tkele.charge()[idx] = testBit<59>(data);
    }
  }  // namespace

  // Fill the bx columns and the leading offset cell of the bx lookup tables
  class FillLookupsKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const uint32_t* bx_arr,
                                  uint32_t nbx,
                                  BxLookupDevice::View tkem_lookup,
                                  BxLookupDevice::View tkele_lookup) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u] == 0u) {
        tkem_lookup.offset().offset()[0] = 0;
        tkele_lookup.offset().offset()[0] = 0;
      }
      for (int32_t idx : cms::alpakatools::uniform_elements(acc, static_cast<int32_t>(nbx))) {
        const auto bx = static_cast<uint16_t>(bx_arr[idx]);
        tkem_lookup.bx().bx()[idx] = bx;
        tkele_lookup.bx().bx()[idx] = bx;
      }
    }
  };

  // Decode all payload slices of both classes (one thread per slice)
  class DecodeSlicesKernelV2 {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const data_t* words,
                                  const RawSlice* slices_tkem,
                                  uint32_t nslices_tkem,
                                  TkEmDeviceCollection::View tkem,
                                  const RawSlice* slices_tkele,
                                  uint32_t nslices_tkele,
                                  TkEleDeviceCollection::View tkele,
                                  BxLookupDevice::ConstView tkem_lookup,
                                  BxLookupDevice::ConstView tkele_lookup) const {
      for (int32_t idx : cms::alpakatools::uniform_elements(acc, static_cast<int32_t>(nslices_tkem))) {
        const RawSlice slice = slices_tkem[idx];
        const uint32_t base = tkem_lookup.offset().offset()[slice.bx_row] + slice.within_bx_offset;
        const data_t* w = words + slice.word_offset;
        for (uint32_t i = 0; i < slice.count; ++i) {
          store_tkem(tkem, base + i, reconstruct_object(w, i));
        }
      }
      for (int32_t idx : cms::alpakatools::uniform_elements(acc, static_cast<int32_t>(nslices_tkele))) {
        const RawSlice slice = slices_tkele[idx];
        const uint32_t base = tkele_lookup.offset().offset()[slice.bx_row] + slice.within_bx_offset;
        const data_t* w = words + slice.word_offset;
        for (uint32_t i = 0; i < slice.count; ++i) {
          store_tkele(tkele, base + i, reconstruct_object(w, i));
        }
      }
    }
  };

  void prefix_scan_u32(Queue& queue, uint32_t* input, uint32_t* output, uint32_t size) {
    cms::alpakatools::iterativePrefixScan<Acc1D>(input, output, size, queue);
  }

  void fill_lookups(Queue& queue,
                    const uint32_t* bx_arr,
                    uint32_t nbx,
                    BxLookupDevice& tkem_lookup,
                    BxLookupDevice& tkele_lookup) {
    const uint32_t threads_per_block = 512;
    const uint32_t blocks_per_grid = cms::alpakatools::divide_up_by(nbx, threads_per_block);
    const auto grid = cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
    alpaka::exec<Acc1D>(queue, grid, FillLookupsKernel{}, bx_arr, nbx, tkem_lookup.view(), tkele_lookup.view());
  }

  void decode_v2(Queue& queue,
                 const data_t* words,
                 const RawSlice* slices_tkem,
                 uint32_t nslices_tkem,
                 TkEmDeviceCollection& tkem,
                 const RawSlice* slices_tkele,
                 uint32_t nslices_tkele,
                 TkEleDeviceCollection& tkele,
                 const BxLookupDevice& tkem_lookup,
                 const BxLookupDevice& tkele_lookup) {
    // one thread per payload slice; slices are short (12 objects for tkEm), so per-thread loops stay cheap
    const uint32_t nslices = nslices_tkem + nslices_tkele;
    const uint32_t threads_per_block = 512;
    const uint32_t blocks_per_grid = cms::alpakatools::divide_up_by(nslices, threads_per_block);
    const auto grid = cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
    alpaka::exec<Acc1D>(queue,
                        grid,
                        DecodeSlicesKernelV2{},
                        words,
                        slices_tkem,
                        nslices_tkem,
                        tkem.view(),
                        slices_tkele,
                        nslices_tkele,
                        tkele.view(),
                        tkem_lookup.const_view(),
                        tkele_lookup.const_view());
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels