#include "L1TriggerScouting/Phase2/plugins/alpaka/L1TScPhase2TrackerMuonRawToDigiKernels.h"

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  L1TScPhase2TrackerMuonRawToDigiKernels::L1TScPhase2TrackerMuonRawToDigiKernels(Queue& queue) { initialize(queue); }

  // Initialize device constant memory for the kernels.
  // Called only once (thread-safe)
  void L1TScPhase2TrackerMuonRawToDigiKernels::initialize(Queue& queue) {
    std::call_once(init_flag_, [&]() {
      // hw to float conversion: 3.14 / 2^12
      constexpr float host_pi_4096 = alpaka::math::constants::pi / 4096.0f;
      auto view_var = alpaka::createView(cms::alpakatools::host(), &host_pi_4096, Vec1D{1});
      alpaka::memcpy(queue, kTrackerMuonPi4096<Acc1D>, view_var);
    });
  }

  namespace {
    // Extract one 96-bit object ({low 64, high 32}) from the packed 64-bit payload words.
    // Two objects share each triplet of 64-bit words; identical to the legacy unpacker
    // (and to the TkEm/TkEle wire format): the even object takes w0 and the low half of w1,
    // the odd object takes w2 and the high half of w1.
    ALPAKA_FN_ACC inline void unpack_object(const data_t* w, uint32_t i, data_t& wlo, uint32_t& whi) {
      const uint32_t k = 3 * (i >> 1);
      if ((i & 1) == 0) {
        wlo = w[k];
        whi = static_cast<uint32_t>(w[k + 1] & 0xffffffffULL);
      } else {
        wlo = w[k + 2];
        whi = static_cast<uint32_t>(w[k + 1] >> 32);
      }
    }

    ALPAKA_FN_ACC inline bool is_zero_object(data_t wlo, uint32_t whi) { return (wlo == 0) && (whi == 0); }

    ALPAKA_FN_ACC inline void store_muon(TrackerMuonDeviceCollection::View muons,
                                         uint32_t idx,
                                         data_t wlo,
                                         uint32_t whi) {
      const uint64_t high = whi;
      muons.pt()[idx] = decodeBits<uint16_t, 1, 16>(wlo) * 0.03125f;
      muons.phi()[idx] = decodeBitsSigned<int16_t, 17, 13>(wlo) * kTrackerMuonPi4096<Acc1D>.get();
      muons.eta()[idx] = decodeBitsSigned<int16_t, 30, 14>(wlo) * kTrackerMuonPi4096<Acc1D>.get();
      muons.z0()[idx] = decodeBitsSigned<int16_t, 44, 10>(wlo) * 0.05f;
      muons.d0()[idx] = decodeBitsSigned<int16_t, 54, 10>(wlo) * 0.03f;
      muons.quality()[idx] = decodeBits<uint8_t, 1, 8>(high);
      muons.isolation()[idx] = decodeBits<uint8_t, 9, 4>(high);
      muons.beta()[idx] = decodeBits<uint8_t, 13, 4>(high) * 0.06f;
      muons.charge()[idx] = testBit<0>(high) ? -1 : 1;
    }
  }  // namespace

  // Fill the bx column and the leading offset cell of the bx lookup table
  class FillMuonLookupKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const uint32_t* bx_arr,
                                  uint32_t nbx,
                                  BxLookupDevice::View lookup) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u] == 0u)
        lookup.offset().offset()[0] = 0;
      for (int32_t idx : cms::alpakatools::uniform_elements(acc, static_cast<int32_t>(nbx)))
        lookup.bx().bx()[idx] = static_cast<uint16_t>(bx_arr[idx]);
    }
  };

  // Decode all tracker muon payload slices (one thread per slice)
  class DecodeTrackerMuonKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const data_t* words,
                                  const RawSlice* slices,
                                  uint32_t nslices,
                                  TrackerMuonDeviceCollection::View muons,
                                  BxLookupDevice::ConstView lookup) const {
      for (int32_t idx : cms::alpakatools::uniform_elements(acc, static_cast<int32_t>(nslices))) {
        const RawSlice slice = slices[idx];
        const data_t* w = words + slice.word_offset;
        uint32_t out = lookup.offset().offset()[slice.bx_row] + slice.within_bx_offset;
        for (uint32_t i = 0; i < slice.count; ++i) {
          data_t wlo;
          uint32_t whi;
          unpack_object(w, i, wlo, whi);
          if (is_zero_object(wlo, whi))
            continue;  // skip padding objects, as in the legacy unpacker
          store_muon(muons, out++, wlo, whi);
        }
      }
    }
  };

  void fill_muon_lookup(Queue& queue, const uint32_t* bx_arr, uint32_t nbx, BxLookupDevice& lookup) {
    const uint32_t threads_per_block = 512;
    const auto grid = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nbx, threads_per_block),
                                                            threads_per_block);
    alpaka::exec<Acc1D>(queue, grid, FillMuonLookupKernel{}, bx_arr, nbx, lookup.view());
  }

  void decode_tracker_muons(Queue& queue,
                            const data_t* words,
                            const RawSlice* slices,
                            uint32_t nslices,
                            TrackerMuonDeviceCollection& muons,
                            const BxLookupDevice& lookup) {
    // one thread per payload slice; slices are short, so per-thread loops stay cheap
    const uint32_t threads_per_block = 512;
    const auto grid = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nslices, threads_per_block),
                                                            threads_per_block);
    alpaka::exec<Acc1D>(queue, grid, DecodeTrackerMuonKernel{}, words, slices, nslices, muons.view(), lookup.const_view());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels
