#include "L1TriggerScouting/Phase2/plugins/alpaka/L1TScPhase2PuppiRawToDigiKernels.h"

#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  L1TScPhase2PuppiRawToDigiKernels::L1TScPhase2PuppiRawToDigiKernels(Queue& queue) { initialize(queue); }

  // Initialize device constant memory for the kernels.
  // Called only once (thread-safe)
  void L1TScPhase2PuppiRawToDigiKernels::initialize(Queue& queue) {
    std::call_once(init_flag_, [&]() {
      // pdgid mapping
      constexpr int16_t host_pdgid[8] = {130, 22, -211, 211, 11, -11, 13, -13};
      auto view = createView(cms::alpakatools::host(), host_pdgid, Vec1D{8});
      alpaka::memcpy(queue, kPdgid<Acc1D>, view);

      // hw to float conversion: 3.14 / 720.0
      constexpr float host_pi_720 = alpaka::math::constants::pi / 720.0f;
      auto view_var = createView(cms::alpakatools::host(), &host_pi_720, Vec1D{1});
      alpaka::memcpy(queue, kPi720<Acc1D>, view_var);
    });
  }

  // Convert raw data to PuppiDeviceCollection
  // Takes 64bit words and decodes them into real values for further analysis
  class RawToDigiKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, data_t* p_data, PuppiDeviceCollection::View puppi) const {
      for (int32_t idx : cms::alpakatools::uniform_elements(acc, puppi.metadata().size())) {
        uint64_t data = p_data[idx];

        // hardware values
        auto hwPt = decodeBits<uint16_t, 0, 14>(data);
        auto hwEta = decodeBitsSigned<int16_t, 14, 12>(data);
        auto hwPhi = decodeBitsSigned<int16_t, 26, 11>(data);
        auto pid = decodeBits<uint8_t, 37, 3>(data);

        // convert to real values
        puppi.pt()[idx] = hwPt * 0.25f;
        puppi.eta()[idx] = hwEta * kPi720<Acc1D>.get();
        puppi.phi()[idx] = hwPhi * kPi720<Acc1D>.get();
        puppi.pdgid()[idx] = kPdgid<Acc1D>.get()[pid];

        if (pid > 1) {
          auto hwZ0 = decodeBitsSigned<int16_t, 40, 10>(data);
          auto hwDxy = decodeBitsSigned<int16_t, 50, 8>(data);
          auto hwQual = decodeBits<uint8_t, 58, 3>(data);

          puppi.z0()[idx] = hwZ0 * 0.05f;
          puppi.dxy()[idx] = hwDxy * 0.05f;
          puppi.puppiw()[idx] = 1.0f;
          puppi.quality()[idx] = hwQual;
        } else {
          auto hwPuppiw = decodeBits<uint16_t, 40, 10>(data);
          auto hwQual = decodeBits<uint8_t, 50, 6>(data);

          puppi.z0()[idx] = 0.0f;
          puppi.dxy()[idx] = 0.0f;
          puppi.puppiw()[idx] = hwPuppiw * (1.0f / 256.0f);
          puppi.quality()[idx] = hwQual;
        }
      }
    }
  };

  class PadPuppiPerBxKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  PuppiDeviceCollection::ConstView inPuppi,
                                  OffsetsSoA::ConstView offsets,
                                  PuppiDeviceCollection::View outPuppi,
                                  unsigned int maxCandsPerBx) const {

      const auto nBx = offsets.metadata().size() - 1;

      // one independent group for each BX
      for (auto bx : cms::alpakatools::independent_groups(acc, nBx)) {
        const auto begin = offsets.offsets()[bx];
        const auto end = offsets.offsets()[bx + 1];
        const auto nCand = end - begin;

        const auto outBase = bx * maxCandsPerBx;
        for (auto tid : cms::alpakatools::independent_group_elements(acc, maxCandsPerBx)) {
          const auto outIdx = outBase + tid;

          if (tid < nCand && tid < maxCandsPerBx) {
            const auto srcIdx = begin + tid;

            outPuppi.pt()[outIdx]      = inPuppi.pt()[srcIdx];
            outPuppi.eta()[outIdx]     = inPuppi.eta()[srcIdx];
            outPuppi.phi()[outIdx]     = inPuppi.phi()[srcIdx];
            outPuppi.z0()[outIdx]      = inPuppi.z0()[srcIdx];
            outPuppi.dxy()[outIdx]     = inPuppi.dxy()[srcIdx];
            outPuppi.puppiw()[outIdx]  = inPuppi.puppiw()[srcIdx];
            outPuppi.quality()[outIdx] = inPuppi.quality()[srcIdx];
            outPuppi.pdgid()[outIdx]   = inPuppi.pdgid()[srcIdx];

          } else if (tid < maxCandsPerBx) {
            outPuppi.pt()[outIdx]      = std::numeric_limits<float>::max();
            outPuppi.eta()[outIdx]     = std::numeric_limits<float>::max();
            outPuppi.phi()[outIdx]     = std::numeric_limits<float>::max();
            outPuppi.z0()[outIdx]      = std::numeric_limits<float>::max();
            outPuppi.dxy()[outIdx]     = std::numeric_limits<float>::max();
            outPuppi.puppiw()[outIdx]  = std::numeric_limits<float>::max();
            outPuppi.quality()[outIdx] = std::numeric_limits<uint8_t>::max();
            outPuppi.pdgid()[outIdx]   = std::numeric_limits<int16_t>::max();
          }
        }
      }
    }
  };

  void decode_candidates(Queue& queue, data_t* p_data, PuppiDeviceCollection& puppi) {
    // move host residing data to device memory space
    auto extent = Vec1D{puppi.const_view().metadata().size()};
    auto p_data_device = alpaka::allocAsyncBuf<data_t, Idx>(queue, extent);
    alpaka::memcpy(queue, p_data_device, createView(cms::alpakatools::host(), p_data, extent));

    // grid dims can be tuned for performance
    uint32_t threads_per_block = 1024;
    uint32_t blocks_per_grid = cms::alpakatools::divide_up_by(puppi.const_view().metadata().size(), threads_per_block);
    auto grid = cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

    // decode particles features
    alpaka::exec<Acc1D>(queue, grid, RawToDigiKernel{}, p_data_device.data(), puppi.view());
  }

  void decode_headers_to_map(Queue& queue, data_t* h_data, BxLookupDeviceCollection& bx_lookup) {
    // move host residing data to device memory space
    auto extent = Vec1D(bx_lookup.const_view().metadata().size());
    auto h_data_device = alpaka::allocAsyncBuf<data_t, Idx>(queue, extent);
    alpaka::memcpy(queue, h_data_device, createView(cms::alpakatools::host(), h_data, extent));

    // grid dims can be tuned for performance
    uint32_t threads_per_block = 1024;
    uint32_t blocks_per_grid =
        cms::alpakatools::divide_up_by(bx_lookup.const_view().metadata().size(), threads_per_block);
    auto grid = cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

    // accumulate buffer with events sizes
    alpaka::exec<Acc1D>(
        queue,
        grid,
        [] ALPAKA_FN_ACC(Acc1D const& acc, data_t* data, BxIndexSoA::View bx_index, OffsetsSoA::View offsets) {
          if (cms::alpakatools::once_per_grid(acc))
            offsets.offsets()[0] = 0;

          for (int32_t idx : cms::alpakatools::uniform_elements(acc, offsets.metadata().size() - 1)) {
            auto range = decodeBits<uint32_t, 0, 12>(data[idx]);
            offsets.offsets()[idx + 1] = range;
            bx_index.bx()[idx] = decodeBits<uint32_t, 12, 12>(data[idx]);
          }
        },
        h_data_device.data(),
        bx_lookup.view<BxIndexSoA>(),
        bx_lookup.view<OffsetsSoA>());

    // prefix sum to build association map used for span extraction and batching
    auto pc = alpaka::allocAsyncBuf<int32_t, Idx>(queue, Vec1D{1});
    alpaka::memset(queue, pc, 0x00);
    alpaka::exec<Acc1D>(queue,
                        grid,
                        cms::alpakatools::multiBlockPrefixScan<uint32_t>{},
                        bx_lookup.view<OffsetsSoA>().offsets().data(),
                        bx_lookup.view<OffsetsSoA>().offsets().data(),
                        bx_lookup.view<OffsetsSoA>().metadata().size(),
                        blocks_per_grid,
                        pc.data(),
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));
  }

  void decode_headers_to_sizes(Queue& queue, data_t* h_data, BxLookupDeviceCollection& bx_sizes) {
    // move host residing data to device memory space
    auto extent = Vec1D(bx_sizes.const_view().metadata().size());
    auto h_data_device = alpaka::allocAsyncBuf<data_t, Idx>(queue, extent);
    alpaka::memcpy(queue, h_data_device, createView(cms::alpakatools::host(), h_data, extent));

    // grid dims can be tuned for performance
    uint32_t threads_per_block = 1024;
    uint32_t blocks_per_grid =
        cms::alpakatools::divide_up_by(bx_sizes.const_view().metadata().size(), threads_per_block);
    auto grid = cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

    // accumulate buffer with events sizes
    alpaka::exec<Acc1D>(
        queue,
        grid,
        [] ALPAKA_FN_ACC(Acc1D const& acc, data_t* data, BxIndexSoA::View bx_index, OffsetsSoA::View offsets) {
          for (int32_t idx : cms::alpakatools::uniform_elements(acc, offsets.metadata().size())) {
            auto range = decodeBits<uint32_t, 0, 12>(data[idx]);
            bx_index.bx()[idx] = decodeBits<uint32_t, 12, 12>(data[idx]);
            offsets.offsets()[idx] = range;
          }
        },
        h_data_device.data(),
        bx_sizes.view<BxIndexSoA>(),
        bx_sizes.view<OffsetsSoA>());
  }

  void fillBxLookupPadded(Queue& queue, BxLookupDeviceCollection& bx_lookup_padded, unsigned int nele) {
    // create host-side bx indexes
    auto nindexes = static_cast<unsigned int>(bx_lookup_padded.const_view<BxIndexSoA>().metadata().size());
    std::vector<uint32_t> indexes(nindexes);
    std::iota(indexes.begin(), indexes.end(), 0);

    // copy indexes from host to device
    auto dstBxIndex = alpaka::createView(alpaka::getDev(queue), 
                                    bx_lookup_padded.view<BxIndexSoA>().bx().data(), 
                                    Vec1D{nindexes});
    alpaka::memcpy(queue, dstBxIndex, indexes, Vec1D{indexes.size()});

    // create host-side fixed offsets
    auto noffsets = static_cast<unsigned int>(bx_lookup_padded.const_view<OffsetsSoA>().metadata().size());
    std::vector<uint32_t> offsets_padded(noffsets);
    for (unsigned int i = 0; i < noffsets; ++i) {
      offsets_padded[i] = i * nele;
    }

    // copy fixed offsets from host to device
    auto dstOffsets = alpaka::createView(alpaka::getDev(queue), 
                                    bx_lookup_padded.view<OffsetsSoA>().offsets().data(), 
                                    Vec1D{noffsets});
    alpaka::memcpy(queue, dstOffsets, offsets_padded, Vec1D{offsets_padded.size()});
  }

  void fillCandsPadded(Queue& queue, BxLookupDeviceCollection& bx_lookup, 
                      PuppiDeviceCollection& puppi_padded, 
                      PuppiDeviceCollection& puppi, 
                      unsigned int nele) {
    auto nbx = bx_lookup.const_view<BxIndexSoA>().metadata().size();
    auto grid = cms::alpakatools::make_workdiv<Acc1D>(nbx, nele);

    alpaka::exec<Acc1D>(queue,
                        grid,
                        PadPuppiPerBxKernel{},
                        puppi.const_view(),
                        bx_lookup.const_view<OffsetsSoA>(),
                        puppi_padded.view(),
                        nele);
  } 
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels