
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceTensorAdapters.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {

    struct ScatterPIDToTrackstersKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    uint32_t const* selectedTracksters,
                                    ::ticl::TracksterInferencePIDScoresSoA::ConstView scores,
                                    ::ticl::TracksterSoA::View outputTracksters,
                                    uint32_t total) const {
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

  void PIDOutputAdapter::applyResults(Queue& queue,
                                      uint32_t const* selectedTracksters,
                                      ModelOutputDevice const& predictions,
                                      PublishedOutputDevice& published,
                                      uint32_t total) {
    if (total == 0u) {
      return;
    }

    constexpr uint32_t elementsPerBlock = 256u;

    const auto blocks = cms::alpakatools::divide_up_by(total, elementsPerBlock);

    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, elementsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        ScatterPIDToTrackstersKernel{},
                        selectedTracksters,
                        predictions.const_view(),
                        published.view(),
                        total);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
