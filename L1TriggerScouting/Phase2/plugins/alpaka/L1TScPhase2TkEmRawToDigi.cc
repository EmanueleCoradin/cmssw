#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/CounterHost.h"
#include "DataFormats/L1ScoutingSoA/interface/BxLookupHost.h"
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
#include "L1TriggerScouting/Phase2/interface/L1TScPhase2Common.h"
#include "L1TriggerScouting/Phase2/plugins/alpaka/L1TScPhase2TkEmRawToDigiKernels.h" // modify !

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  struct TkEmBxData {
    bx_t bx;
    count_t bx_size; // size in terms of number of objects
    const data_t *data_ptr;
    size_t data_size; // size in terms of data_t words

    // comparison operator for priority queue
    bool operator>(const TkEmBxData &other) const { return bx > other.bx; }
  };

  using TkEmHeap = std::priority_queue<TkEmBxData, std::vector<TkEmBxData>, std::greater<>>;
  
  using namespace ::l1sc;

  class L1TScPhase2TkEmRawToDigi : public stream::EDProducer<> {
  public:
    L1TScPhase2TkEmRawToDigi(const edm::ParameterSet &params)
        : EDProducer<>(params),
          raw_data_token_{consumes(params.getParameter<edm::InputTag>("src"))},
          tkem_collection_token_{produces("tkem")},
          tkele_collection_token_{produces("tkele")},
          tkem_bx_lookup_token_{produces("tkEmBxLookup")},
          tkele_bx_lookup_token_{produces("tkEleBxLookup")},
          nbx_token_{produces("nbx")},
          streams_(params.getParameter<std::vector<uint32_t>>("streams")),
          splitFactor_(params.getParameter<unsigned int>("splitFactor")) {}

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      // get raw data input
      const auto &raw_data = event.get(raw_data_token_);

      // normalize header & payload
      auto ngoodbx = normalize(raw_data);
      assert(tkem_bx_vec_.size() == tkele_bx_vec_.size() && "[L1TScPhase2TkEmRawToDigi] TkEm and TkEle bx numbers differ.\n");
      const auto nbx = static_cast<int32_t>(tkem_bx_vec_.size());

      // create bx lookup host (associaton maps are created on the host)
      auto bx_lookup_tkem = BxLookupHost(event.queue(), nbx, nbx + 1);
      auto bx_lookup_tkele = BxLookupHost(event.queue(), nbx, nbx + 1);

      // copy bx indexes to the host collection bx column
      memcpy(bx_lookup_tkem.view().bx().bx().data(), tkem_bx_vec_.data(), sizeof(tkem_bx_vec_.size() * sizeof(bx_t)));
      memcpy(bx_lookup_tkele.view().bx().bx().data(), tkele_bx_vec_.data(), sizeof(tkele_bx_vec_.size() * sizeof(bx_t)));

      // calculate offsets using inclusive scan and put them into the host collection offset column
      std::inclusive_scan(tkem_count_vec_.begin(), tkem_count_vec_.end(), bx_lookup_tkem.view().offset().offset().data() + 1);
      std::inclusive_scan(tkele_count_vec_.begin(), tkele_count_vec_.end(), bx_lookup_tkele.view().offset().offset().data() + 1);

      // create TkEm/TkEle device collections (TkEm/TkEle collections are created on the device)
      auto tkem = TkEmDeviceCollection(event.queue(), tkem_payload_vec_.size());
      auto tkele = TkEleDeviceCollection(event.queue(), tkele_payload_vec_.size());

      // initialize device constant memory (called once)
      rtd_kernels_.initialize(event.queue());

      // decode raw data
      kernels::decode_tkem(event.queue(), tkem_payload_vec_.data(), tkem);
      kernels::decode_tkele(event.queue(), tkele_payload_vec_.data(), tkele);

      // store data in the event (device-side products)
      event.emplace(tkem_collection_token_, std::move(tkem));
      event.emplace(tkele_collection_token_, std::move(tkele));
      event.emplace(tkem_bx_lookup_token_, std::move(bx_lookup_tkem));
      event.emplace(tkele_bx_lookup_token_, std::move(bx_lookup_tkele));

      // store nbx
      auto nbx_portable = CounterHost(event.queue(), static_cast<unsigned int>(ngoodbx));
      event.emplace(nbx_token_, std::move(nbx_portable));
    };

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::vector<uint32_t>>("streams");
      desc.add<unsigned int>("splitFactor", 1)->setComment("Number of streams per BX");
      desc.add<edm::InputTag>("src");
      desc.addUntracked<int>("environment", static_cast<int>(Environment::kProduction));
      descriptions.addWithDefaultLabel(desc);
    };

    unsigned int normalize(const SDSRawDataCollection &raw_data) {
      tkem_bx_vec_.clear();
      tkele_bx_vec_.clear();
      tkem_count_vec_.clear();
      tkele_count_vec_.clear();
      tkem_payload_vec_.clear();
      tkele_payload_vec_.clear();

      TkEmHeap tkem_heap, tkele_heap;
      // readout data from links breadth-first (order not guaranteed)
      for (auto stream_id : streams_) {
        const auto &stream = raw_data.FEDData(stream_id);
        const auto chunk_begin = reinterpret_cast<const data_t *>(stream.data());
        const auto chunk_end = reinterpret_cast<const data_t *>(stream.data() + stream.size());

        for (auto ptr = chunk_begin; ptr < chunk_end;) { 
          if (*ptr == 0) {
            ++ptr;
            continue;
          }  // skip empty words

          bx_t          bx            = ((*ptr) >> 12) & 0xFFF;
          size_t        nwords        = (*ptr) & 0xFFF;              // unpack chunk size (no. of 64 bit word of Native64 header)
          const count_t negamma       = (nwords * 2) / 3;

          auto          first_tkem    = ptr + 1;                     // move past header
          auto          last_tkem     = ptr + numTkEmWords;          // move the pointer to the last tkEm of the 12 tkEm block
          const count_t ntkem         = numTkEm;                     // the number of tkem words is fixed to 12 a-priori

          auto          first_tkele   = last_tkem + 1;               // move the pointer to the first tkEle of the remaining tkEle block
          size_t        numTkEleWords = nwords - numTkEmWords;       // number of 64b words
          const count_t ntkele        = (numTkEleWords * 2) / 3;     // convert number of 64b words difference in number of tkEle

          assert(negamma == (ntkem + ntkele));                       // closure test
          tkem_heap.push({bx, ntkem, first_tkem, numTkEmWords});
          tkele_heap.push({bx, ntkele, first_tkele, numTkEleWords});

          ptr += (nwords + 1);                                      // move to the next bx, the + 1 is needed to move past the header
        }
      }

      // restore bx order (size of heap at max 3564 not that heavy to track)
      // pop the first element outside the loop
      if (tkem_heap.empty() | tkele_heap.empty())
        return 0;

      count_t ngoodbx_tkem = consume_heap(tkem_bx_vec_, tkem_count_vec_, tkem_payload_vec_, tkem_heap);
      count_t ngoodbx_tkele = consume_heap(tkele_bx_vec_, tkele_count_vec_, tkele_payload_vec_, tkele_heap);

      assert(ngoodbx_tkem == ngoodbx_tkele && "[L1TScPhase2TkEmRawToDigi] Total good tkem BXs and total good tkele BX do not match.");
      return ngoodbx_tkem;
    }

    count_t consume_heap(BxVec& bx_vec, CountVec& count_vec, TkEmPayloadVec& payload_vec, TkEmHeap& pq) {
      count_t ngoodbx = 0;
      count_t nslices = 0;
      
      auto bx_data = pq.top();
      
      bx_vec.push_back(bx_data.bx);
      count_vec.push_back(bx_data.bx_size);
      auto buf_vec = flattenBuffer(bx_data);
      payload_vec.insert(payload_vec.end(), buf_vec.begin(), buf_vec.end());
      pq.pop();

      while (!pq.empty()) {
        const auto bx_data2 = pq.top();

        if (bx_data2.bx != bx_data.bx) {
          if (nslices == splitFactor_) {
            ngoodbx++;
          }
          nslices = 1;
          bx_vec.push_back(bx_data2.bx);
          count_vec.push_back(bx_data2.bx_size);
        } else {
          count_vec.back() += bx_data2.bx_size;
          nslices++;
        }
        
        bx_data = bx_data2;

        buf_vec = flattenBuffer(bx_data);
        payload_vec.insert(payload_vec.end(), buf_vec.begin(), buf_vec.end());
        pq.pop();
      }

      if (nslices == splitFactor_)
        ngoodbx++;

      return ngoodbx;
    }

    TkEmPayloadVec flattenBuffer(TkEmBxData& data)
    {
        TkEmPayloadVec output;
        output.reserve(data.bx_size);

        for (std::size_t i = 0; i + 2 < data.data_size; i += 3) {
          const uint64_t w0 = data.data_ptr[i];
          const uint64_t w1 = data.data_ptr[i + 1];
          const uint64_t w2 = data.data_ptr[i + 2];

          // Logical element 0:
          // [95:64] = lower 32 bits of w1
          // [63:0]  = w0
          const uint128_t x0 =
              static_cast<uint128_t>(w0) |
              (static_cast<uint128_t>(w1 & 0xffffffffULL) << 64);

          // Logical element 1:
          // [95:64] = upper 32 bits of w1
          // [63:0]  = w2
          const uint128_t x1 =
              static_cast<uint128_t>(w2) |
              (static_cast<uint128_t>(w1 >> 32) << 64);

          output.push_back(x0);
          output.push_back(x1);
        }

        return output;
    }

  private:
    // consume host side input data
    const edm::EDGetTokenT<SDSRawDataCollection> raw_data_token_;

    // produce device-side products
    const device::EDPutToken<TkEmDeviceCollection> tkem_collection_token_;
    const device::EDPutToken<TkEleDeviceCollection> tkele_collection_token_;
    const edm::EDPutTokenT<BxLookupHost> tkem_bx_lookup_token_, tkele_bx_lookup_token_;

    // produce host-side products
    const edm::EDPutTokenT<CounterHost> nbx_token_;

    // utility members
    const std::vector<uint32_t> streams_;
    const unsigned int splitFactor_;  // number of streams per BX

    // temporary storage
    BxVec tkem_bx_vec_, tkele_bx_vec_;
    CountVec tkem_count_vec_, tkele_count_vec_;
    TkEmPayloadVec tkem_payload_vec_, tkele_payload_vec_;

    // kernel
    kernels::L1TScPhase2TkEmRawToDigiKernels rtd_kernels_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

DEFINE_FWK_ALPAKA_MODULE(l1sc::L1TScPhase2TkEmRawToDigi);