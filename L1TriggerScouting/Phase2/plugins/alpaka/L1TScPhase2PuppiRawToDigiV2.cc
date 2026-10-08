#include <array>
#include <cstdint>
#include <cstring>

#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/CounterHost.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/BxLookupDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "L1TriggerScouting/Phase2/interface/L1TScPhase2Common.h"
#include "L1TriggerScouting/Phase2/plugins/alpaka/L1TScPhase2PuppiRawToDigiKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  // Optimized puppi/PF-candidate unpacker, following the TkEm V2 design:
  //  - single-pass parse with flat bx buckets (no priority queue, no per-BX allocations)
  //  - verbatim per-stream payload copy into pinned staging buffers
  //  - bx lookup offsets computed on device with a prefix scan
  //  - one fused decode kernel over all slices
  // The raw format is identical for PUPPI and PF candidates: one 64-bit word per object.
  class L1TScPhase2PuppiRawToDigiV2 : public stream::EDProducer<> {
  public:
    explicit L1TScPhase2PuppiRawToDigiV2(const edm::ParameterSet& params)
        : EDProducer<>(params),
          raw_data_token_{consumes(params.getParameter<edm::InputTag>("src"))},
          puppi_token_{produces("candidates")},
          bx_lookup_token_{produces("bxLookup")},
          bx_sizes_token_{produces("bxSizes")},
          nbx_token_{produces("nbx")},
          streams_(params.getParameter<std::vector<uint32_t>>("streams")),
          splitFactor_(params.getParameter<unsigned int>("splitFactor")) {
      if (splitFactor_ == 0)
        throw cms::Exception("Configuration") << "splitFactor must be positive\n";
    }

    void produce(device::Event& event, const device::EventSetup& event_setup) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src");
      desc.add<std::vector<uint32_t>>("streams");
      desc.add<unsigned int>("splitFactor", 1)->setComment("Number of streams per BX");
      desc.addUntracked<int>("environment", static_cast<int>(Environment::kProduction));
      descriptions.addWithDefaultLabel(desc);
    };

  private:
    // maximum number of BX in an orbit
    static constexpr uint32_t kMaxNBX = 3564;
    // RawSlice is 4 uint32_t
    static constexpr uint32_t kSliceU32 = sizeof(kernels::RawSlice) / sizeof(uint32_t);

    void reset() {
      slot_of_bx_.fill(-1);
      nslots_ = 0;
      nbx_ = 0;
      ngoodbx_ = 0;
      nslices_ = 0;
      tot_cands_ = 0;
      words_used_ = 0;
    }

    const edm::EDGetTokenT<SDSRawDataCollection> raw_data_token_;

    // device-side products
    const device::EDPutToken<PuppiDeviceCollection> puppi_token_;
    const device::EDPutToken<BxLookupDevice> bx_lookup_token_, bx_sizes_token_;

    // host-side product
    const edm::EDPutTokenT<CounterHost> nbx_token_;

    const std::vector<uint32_t> streams_;
    const uint32_t splitFactor_;

    // per-event bookkeeping, reused across events
    std::array<int32_t, kMaxNBX> slot_of_bx_;    // bx number -> slot of the event (-1 if absent)
    std::array<uint32_t, kMaxNBX> fill_slot_;    // candidates appended so far per slot (== total at the end)
    std::array<uint32_t, kMaxNBX> slices_slot_;  // number of slices merged so far per slot
    std::array<uint32_t, kMaxNBX> row_of_slot_;  // slot -> bx lookup row

    uint32_t nslots_, nbx_, ngoodbx_, nslices_;
    count_t tot_cands_;
    size_t words_used_;

    // kernel
    kernels::L1TScPhase2PuppiRawToDigiKernels rtd_kernels_;
  };

  void L1TScPhase2PuppiRawToDigiV2::produce(device::Event& event, const device::EventSetup& event_setup) {
    auto& queue = event.queue();
    const auto& raw_data = event.get(raw_data_token_);

    // ---- upper bounds for this event ------------------------------------------------
    size_t words_ub = 0;
    for (auto stream_id : streams_)
      words_ub += raw_data.FEDData(stream_id).size() / sizeof(data_t);
    const uint32_t slices_ub = static_cast<uint32_t>(words_ub / 2) + 1;  // a block holds at least 1 payload word

    // ---- pinned staging buffers ----------------------------------------------------------
    // words: full streams verbatim (headers included), concatenated per stream.
    // meta:  slice descriptors | bx column | per-bx candidate counts.
    auto h_words = cms::alpakatools::make_host_buffer<data_t[]>(queue, std::max<Idx>(words_ub, 1u));
    auto h_meta = cms::alpakatools::make_host_buffer<uint32_t[]>(
        queue, kSliceU32 * slices_ub + 2 * kMaxNBX);
    const uint32_t rows_offset = kSliceU32 * slices_ub;
    auto* slices = reinterpret_cast<kernels::RawSlice*>(h_meta.data());
    uint32_t* bx_col = h_meta.data() + rows_offset;
    uint32_t* cnt_col = bx_col + kMaxNBX;

    // ---- single-pass parse (slices + counts; word offsets into the verbatim copy) ---------
    reset();
    size_t words_base = 0;
    for (auto stream_id : streams_) {
      const auto& stream = raw_data.FEDData(stream_id);
      if (stream.size() % sizeof(data_t) != 0)
        throw cms::Exception("CorruptData") << "Puppi stream size is not a multiple of 64 bits\n";
      const auto* chunk_begin = reinterpret_cast<const data_t*>(stream.data());
      const auto* ptr = chunk_begin;
      const auto* end = ptr + stream.size() / sizeof(data_t);

      while (ptr < end) {
        const data_t* block_header = ptr;
        const data_t header = *ptr++;
        if (header == 0)
          continue;  // skip empty words
        const uint32_t bx = (header >> 12) & 0xfff; // pay attention that the encoded value has [0, 3563] range, not [1, 3564]!
        const uint32_t nwords = header & 0xfff;
        const uint64_t orbit = (header >> 24) & 0xfffffffffULL;
        if (orbit != event.id().event() || bx >= kMaxNBX || static_cast<size_t>(end - ptr) < nwords)
          throw cms::Exception("CorruptData")
              << "Invalid puppi block in stream " << stream_id << ", BX " << bx << ", orbit " << orbit << "\n";

        int32_t slot = slot_of_bx_[bx];
        if (slot < 0) { // the slot is a redundant index that allows to take into account also situations in which the order in which bxs are packed inside the raw data stream is scattered (e.g., BX 12 is before BX 0)
          slot = nslots_++;
          slot_of_bx_[bx] = slot;
          fill_slot_[slot] = 0;
          slices_slot_[slot] = 0;
        }
        
        slices[nslices_++] = kernels::RawSlice{static_cast<uint32_t>(words_base + (block_header - chunk_begin)) + 1, // one candidate per payload word, word_offset points at the payload start inside the verbatim stream copy
                                               static_cast<uint32_t>(slot), // bx redundant index
                                               fill_slot_[slot], // here we use the non-updated value beacause this field is a within-bx offset
                                               nwords}; // number of words belonging to this slice
        fill_slot_[slot] += nwords;
        ++slices_slot_[slot];
        tot_cands_ += nwords;

        ptr += nwords;
      }
      words_base += stream.size() / sizeof(data_t);
    }

    // ---- one bulk copy per stream into the pinned staging buffer --------------------------
    {
      size_t words_base_copy = 0;
      for (auto stream_id : streams_) {
        const auto& stream = raw_data.FEDData(stream_id);
        const size_t stream_words = stream.size() / sizeof(data_t);
        std::memcpy(h_words.data() + words_base_copy, stream.data(), stream_words * sizeof(data_t));
        words_base_copy += stream_words;
      }
      words_used_ = words_base_copy;
    }

    // ---- finalize bx rows (ascending bx for free, buckets are indexed by bx) ----------
    // here the bx ordering is recovered
    for (bx_t bx = 0; bx < kMaxNBX; ++bx) {
      const int32_t slot = slot_of_bx_[bx];
      if (slot < 0)
        continue;
      bx_col[nbx_] = bx;
      cnt_col[nbx_] = fill_slot_[slot];
      row_of_slot_[slot] = nbx_;
      if (slices_slot_[slot] == splitFactor_)
        ++ngoodbx_;
      ++nbx_;
    }

    // map slice bucket index -> bx lookup row
    for (uint32_t s = 0; s < nslices_; ++s)
      slices[s].bx_row = row_of_slot_[slices[s].bx_row /* convert the slot into the corresponding bx */];

    // ---- allocate products --------------------------------------------------------------
    auto puppi = PuppiDeviceCollection(queue, tot_cands_);
    auto bx_lookup = BxLookupDevice(queue, nbx_, nbx_ + 1);
    auto bx_sizes = BxLookupDevice(queue, nbx_, nbx_);

    // initialize device constant memory (called once)
    rtd_kernels_.initialize(queue);

    if (nbx_ > 0 && nslices_ > 0) {
      // ---- host-to-device transfers -------------------------------------------------------
      auto d_words = cms::alpakatools::make_device_buffer<data_t[]>(queue, static_cast<Idx>(words_used_));
      auto d_meta = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, rows_offset + 2 * nbx_);
      alpaka::memcpy(queue,
                     alpaka::createView(alpaka::getDev(queue), d_words.data(), Vec1D{static_cast<Idx>(words_used_)}),
                     alpaka::createView(cms::alpakatools::host(), h_words.data(), Vec1D{static_cast<Idx>(words_used_)}));
      const uint32_t slices_u32 = kSliceU32 * nslices_;
      alpaka::memcpy(queue,
                     alpaka::createView(alpaka::getDev(queue), d_meta.data(), Vec1D{slices_u32}),
                     alpaka::createView(cms::alpakatools::host(), h_meta.data(), Vec1D{slices_u32}));
      alpaka::memcpy(queue,
                     alpaka::createView(alpaka::getDev(queue), d_meta.data() + rows_offset, Vec1D{2 * nbx_}),
                     alpaka::createView(cms::alpakatools::host(), h_meta.data() + rows_offset, Vec1D{2 * nbx_}));

      uint32_t* bx_col_d = d_meta.data() + rows_offset;
      uint32_t* cnt_col_d = bx_col_d + nbx_;
      const auto* slices_d = reinterpret_cast<const kernels::RawSlice*>(d_meta.data());

      // ---- bx lookups computed on device ---------------------------------------------------
      // inclusive prefix sum of the per-bx candidate counts directly into the offset column
      kernels::prefix_scan_u32(queue, cnt_col_d, bx_lookup.view().offset().offset().data() + 1, nbx_);
      // bx columns, per-bx sizes column, and the leading 0 of the offset column
      kernels::fill_lookups_puppi(queue, bx_col_d, cnt_col_d, nbx_, bx_lookup, bx_sizes);

      // ---- one fused decode kernel ---------------------------------------------------------
      kernels::decode_candidates_v2(queue, d_words.data(), slices_d, nslices_, puppi, bx_lookup);
    }

    // ---- store data in the event ------------------------------------------------------------
    event.emplace(puppi_token_, std::move(puppi));
    event.emplace(bx_lookup_token_, std::move(bx_lookup));
    event.emplace(bx_sizes_token_, std::move(bx_sizes));

    auto nbx_portable = CounterHost(queue, static_cast<unsigned int>(ngoodbx_));
    event.emplace(nbx_token_, std::move(nbx_portable));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

DEFINE_FWK_ALPAKA_MODULE(l1sc::L1TScPhase2PuppiRawToDigiV2);
