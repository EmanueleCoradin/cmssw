#include <algorithm>
#include <array>
#include <cstring>

#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/CounterHost.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/BxLookupDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/TkEleDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/TkEmDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "L1TriggerScouting/Phase2/interface/L1TScPhase2Common.h"
#include "L1TriggerScouting/Phase2/plugins/alpaka/L1TScPhase2TkEmRawToDigiKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  // V2 unpacker:
  //  - single-pass parse with flat bx buckets instead of priority queues
  //  - payload words and metadata copied directly into pinned host staging buffers
  //  - one fused host-to-device payload transfer and one fused decode kernel for both classes
  //  - bx columns and per-bx offsets computed on device (iterative prefix scan)
  //  - bookkeeping storage reused across events (no per-event allocations on the host path)
  class L1TScPhase2TkEmRawToDigiV2 : public stream::EDProducer<> {
  public:
    explicit L1TScPhase2TkEmRawToDigiV2(const edm::ParameterSet& params)
        : EDProducer<>(params),
          raw_data_token_{consumes(params.getParameter<edm::InputTag>("src"))},
          tkem_collection_token_{produces("tkem")},
          tkele_collection_token_{produces("tkele")},
          tkem_bx_lookup_token_{produces("tkEmBxLookup")},
          tkele_bx_lookup_token_{produces("tkEleBxLookup")},
          nbx_token_{produces("nbx")},
          streams_(params.getParameter<std::vector<uint32_t>>("streams")),
          splitFactor_(params.getParameter<unsigned int>("splitFactor")) {}

    void produce(device::Event& event, const device::EventSetup& event_setup) override {
      // get raw data input
      const auto& raw_data = event.get(raw_data_token_);
      produceImpl(event, raw_data);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::vector<uint32_t>>("streams");
      desc.add<unsigned int>("splitFactor", 1)->setComment("Number of streams per BX");
      desc.add<edm::InputTag>("src");
      desc.addUntracked<int>("environment", static_cast<int>(Environment::kProduction));
      descriptions.addWithDefaultLabel(desc);
    };

  private:
    // maximum number of BX in an orbit
    static constexpr uint32_t kMaxNBX = 3564;
    // RawSlice is 4 uint32_t
    static constexpr uint32_t kSliceU32 = sizeof(kernels::RawSlice) / sizeof(uint32_t);

    void produceImpl(device::Event& event, const SDSRawDataCollection& raw_data) {
      auto& queue = event.queue();

      // ---- upper bounds for this event ------------------------------------------------
      // every payload-carrying block occupies at least numTkEmWords + 1 words
      size_t words_ub = 0;
      for (auto stream_id : streams_)
        words_ub += raw_data.FEDData(stream_id).size() / sizeof(data_t);
      const uint32_t slices_ub = static_cast<uint32_t>(words_ub / (numTkEmWords + 1)) + 1;

      // ---- pinned host staging buffers (asynchronous copies source) -------------------
      // words: concatenated tkEm + tkEle payload words (64-bit words, densely packed)
      // meta:  slices_tkem | slices_tkele | bx column | tkem counts | tkele counts
      auto h_words = cms::alpakatools::make_host_buffer<data_t[]>(queue, std::max<Idx>(words_ub, 1u));
      const uint32_t rows_offset = 2 * kSliceU32 * slices_ub;  // slices region size in u32 units
      auto h_meta = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, rows_offset + 3 * kMaxNBX);
      auto* slices_tkem = reinterpret_cast<kernels::RawSlice*>(h_meta.data());
      auto* slices_tkele = slices_tkem + slices_ub;
      uint32_t* bx_col = h_meta.data() + rows_offset;
      uint32_t* cnt_tkem = bx_col + kMaxNBX;
      uint32_t* cnt_tkele = bx_col + 2 * kMaxNBX;

      // ---- single-pass parse --------------------------------------------------
      reset();
      for (auto stream_id : streams_) {
        const auto& stream = raw_data.FEDData(stream_id);
        const auto chunk_begin = reinterpret_cast<const data_t*>(stream.data());
        const auto chunk_end = reinterpret_cast<const data_t*>(stream.data() + stream.size());

        for (auto ptr = chunk_begin; ptr < chunk_end;) {
          if (*ptr == 0) {
            ++ptr;
            continue;
          }  // skip empty words

          const bx_t bx = ((*ptr) >> 12) & 0xFFF;
          const size_t nwords = (*ptr) & 0xFFF;
          assert(bx < kMaxNBX);
          assert(nwords >= numTkEmWords);
          const count_t negamma = (nwords * 2) / 3;
          const count_t ntkem = numTkEm;
          const count_t ntkele = ((nwords - numTkEmWords) * 2) / 3;
          assert(negamma == (ntkem + ntkele));  // closure test

          int32_t slot = slot_of_bx_[bx];
          if (slot < 0) {
            slot = nslots_++;
            slot_of_bx_[bx] = slot;
            fill_tkem_slot_[slot] = 0;
            fill_tkele_slot_[slot] = 0;
            slices_slot_[slot] = 0;
          }

          assert(words_used_ + nwords <= words_ub);
          // tkem slice: numTkEm fixed objects = numTkEmWords words
          std::memcpy(h_words.data() + words_used_, ptr + 1, numTkEmWords * sizeof(data_t));
          slices_tkem[nslices_tkem_++] = kernels::RawSlice{
              static_cast<uint32_t>(words_used_), static_cast<uint32_t>(slot), fill_tkem_slot_[slot], numTkEm};
          words_used_ += numTkEmWords;
          fill_tkem_slot_[slot] += ntkem;
          // tkele slice: remaining payload words
          std::memcpy(h_words.data() + words_used_, ptr + 1 + numTkEmWords, (nwords - numTkEmWords) * sizeof(data_t));
          slices_tkele[nslices_tkele_] = kernels::RawSlice{
              static_cast<uint32_t>(words_used_), static_cast<uint32_t>(slot), fill_tkele_slot_[slot], ntkele};
          ++nslices_tkele_;
          words_used_ += nwords - numTkEmWords;
          fill_tkele_slot_[slot] += ntkele;

          ++slices_slot_[slot];
          tot_tkem_ += ntkem;
          tot_tkele_ += ntkele;

          ptr += (nwords + 1);  // header + payload words
        }
      }

      // ---- finalize bx rows (ascending bx for free, buckets are indexed by bx) ---------
      nbx_ = 0;
      ngoodbx_ = 0;
      for (bx_t bx = 0; bx < kMaxNBX; ++bx) {
        const int32_t slot = slot_of_bx_[bx];
        if (slot < 0)
          continue;
        bx_col[nbx_] = bx;
        cnt_tkem[nbx_] = fill_tkem_slot_[slot];
        cnt_tkele[nbx_] = fill_tkele_slot_[slot];
        row_of_slot_[slot] = nbx_;
        if (slices_slot_[slot] == splitFactor_)
          ++ngoodbx_;
        ++nbx_;
      }

      // map slice bucket index -> bx row
      for (uint32_t s = 0; s < nslices_tkem_; ++s)
        slices_tkem[s].bx_row = row_of_slot_[slices_tkem[s].bx_row];
      for (uint32_t s = 0; s < nslices_tkele_; ++s)
        slices_tkele[s].bx_row = row_of_slot_[slices_tkele[s].bx_row];

      // ---- allocate products ----------------------------------------------------------
      auto tkem_lookup = BxLookupDevice(queue, nbx_, nbx_ + 1);
      auto tkele_lookup = BxLookupDevice(queue, nbx_, nbx_ + 1);
      auto tkem = TkEmDeviceCollection(queue, tot_tkem_);
      auto tkele = TkEleDeviceCollection(queue, tot_tkele_);

      // initialize device constant memory (called once)
      rtd_kernels_.initialize(queue);

      if (nbx_ > 0) {
        // ---- one fused H2D transfer for the payload words ------------------------------
        auto d_words = cms::alpakatools::make_device_buffer<data_t[]>(queue, static_cast<Idx>(words_used_));
        alpaka::memcpy(queue,
                       d_words,
                       alpaka::createView(cms::alpakatools::host(), h_words.data(), Vec1D{static_cast<Idx>(words_used_)}));

        // ---- H2D of the slice and bx-row metadata --------------------------------------
        const uint32_t slices_u32 = kSliceU32 * (nslices_tkem_ + nslices_tkele_);
        auto d_meta = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, rows_offset + 3 * nbx_);
        alpaka::memcpy(queue,
                       alpaka::createView(alpaka::getDev(queue), d_meta.data(), Vec1D{slices_u32}),
                       alpaka::createView(cms::alpakatools::host(), h_meta.data(), Vec1D{slices_u32}));
        alpaka::memcpy(queue,
                       alpaka::createView(alpaka::getDev(queue), d_meta.data() + rows_offset, Vec1D{3 * nbx_}),
                       alpaka::createView(cms::alpakatools::host(), h_meta.data() + rows_offset, Vec1D{3 * nbx_}));

        uint32_t* bx_col_d = d_meta.data() + rows_offset;
        uint32_t* cnt_tkem_d = bx_col_d + nbx_;
        uint32_t* cnt_tkele_d = cnt_tkem_d + nbx_;
        const auto* slices_tkem_d = reinterpret_cast<const kernels::RawSlice*>(d_meta.data());
        const auto* slices_tkele_d = slices_tkem_d + nslices_tkem_;

        // ---- bx lookup computed on device ----------------------------------------------
        // inclusive prefix sum of the per-bx object counts directly into the offset columns
        kernels::prefix_scan_u32(queue, cnt_tkem_d, tkem_lookup.view().offset().offset().data() + 1, nbx_);
        kernels::prefix_scan_u32(queue, cnt_tkele_d, tkele_lookup.view().offset().offset().data() + 1, nbx_);
        // bx columns and the leading 0 of the offset columns
        kernels::fill_lookups(queue, bx_col_d, nbx_, tkem_lookup, tkele_lookup);

        // ---- one fused decode kernel for both classes ----------------------------------
        kernels::decode_v2(queue,
                           d_words.data(),
                           slices_tkem_d,
                           nslices_tkem_,
                           tkem,
                           slices_tkele_d,
                           nslices_tkele_,
                           tkele,
                           tkem_lookup,
                           tkele_lookup);
      }

      // ---- store data in the event ------------------------------------------------------
      event.emplace(tkem_collection_token_, std::move(tkem));
      event.emplace(tkele_collection_token_, std::move(tkele));
      event.emplace(tkem_bx_lookup_token_, std::move(tkem_lookup));
      event.emplace(tkele_bx_lookup_token_, std::move(tkele_lookup));

      auto nbx_portable = CounterHost(event.queue(), static_cast<unsigned int>(ngoodbx_));
      event.emplace(nbx_token_, std::move(nbx_portable));
    }

    void reset() {
      slot_of_bx_.fill(-1);
      nslots_ = 0;
      nbx_ = 0;
      ngoodbx_ = 0;
      nslices_tkem_ = 0;
      nslices_tkele_ = 0;
      tot_tkem_ = 0;
      tot_tkele_ = 0;
      words_used_ = 0;
    }

    // consume host side input data
    const edm::EDGetTokenT<SDSRawDataCollection> raw_data_token_;

    // produce device-side products
    const device::EDPutToken<TkEmDeviceCollection> tkem_collection_token_;
    const device::EDPutToken<TkEleDeviceCollection> tkele_collection_token_;
    const device::EDPutToken<BxLookupDevice> tkem_bx_lookup_token_, tkele_bx_lookup_token_;

    // produce host-side products
    const edm::EDPutTokenT<CounterHost> nbx_token_;

    // utility members
    const std::vector<uint32_t> streams_;
    const unsigned int splitFactor_;  // number of streams per BX

    // per-event bookkeeping, reused across events
    std::array<int32_t, kMaxNBX> slot_of_bx_;     // bx number -> slot of the event (-1 if absent)
    std::array<uint32_t, kMaxNBX> fill_tkem_slot_;   // objects placed so far (and total at the end) per slot
    std::array<uint32_t, kMaxNBX> fill_tkele_slot_;
    std::array<uint32_t, kMaxNBX> slices_slot_;   // number of slices merged so far per slot
    std::array<uint32_t, kMaxNBX> row_of_slot_;   // slot -> bx lookup row

    uint32_t nslots_, nbx_, ngoodbx_;
    uint32_t nslices_tkem_, nslices_tkele_;
    count_t tot_tkem_, tot_tkele_;
    size_t words_used_;

    // kernel
    kernels::L1TScPhase2TkEmRawToDigiKernels rtd_kernels_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

DEFINE_FWK_ALPAKA_MODULE(l1sc::L1TScPhase2TkEmRawToDigiV2);
