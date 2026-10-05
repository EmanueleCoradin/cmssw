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
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels