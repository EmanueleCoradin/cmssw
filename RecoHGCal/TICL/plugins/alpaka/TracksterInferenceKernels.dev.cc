#include "RecoHGCal/TICL/plugins/alpaka/TracksterInferenceKernels.h"

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {

    struct FillPIDProbabilitiesKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(
          TAcc const& acc,
          uint32_t const* selectedTracksters,
          ticl::TracksterInferencePIDScoresSoA::ConstView scores,
          ticl::TracksterSoA::View outputTracksters,
          int32_t total) const {
        for (auto row : cms::alpakatools::uniform_elements(acc, total)) {
          auto const source = scores[row];
          auto destination = outputTracksters.tracksters()[selectedTracksters[row]];

          destination.id_probabilities0() = source.id_probabilities0();
          destination.id_probabilities1() = source.id_probabilities1();
          destination.id_probabilities2() = source.id_probabilities2();
          destination.id_probabilities3() = source.id_probabilities3();
          destination.id_probabilities4() = source.id_probabilities4();
          destination.id_probabilities5() = source.id_probabilities5();
          destination.id_probabilities6() = source.id_probabilities6();
          destination.id_probabilities7() = source.id_probabilities7();
        }
      }
    };

  }  // namespace

  void fillPIDProbabilities(
      Queue& queue,
      uint32_t const* selectedTracksters,
      ticl::TracksterInferencePIDScoresSoA::ConstView scores,
      ticl::TracksterSoA::View outputTracksters,
      int32_t total) {
    constexpr uint32_t elementsPerBlock = 256;
    auto const blocks =
        (static_cast<uint32_t>(total) + elementsPerBlock - 1) / elementsPerBlock;
    auto const workDiv =
        cms::alpakatools::make_workdiv<Acc1D>(blocks, elementsPerBlock);

    alpaka::exec<Acc1D>(
        queue,
        workDiv,
        FillPIDProbabilitiesKernel{},
        selectedTracksters,
        scores,
        outputTracksters,
        total);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE