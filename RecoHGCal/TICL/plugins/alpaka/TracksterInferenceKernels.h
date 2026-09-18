#ifndef TracksterInferenceKernels_h
#define TracksterInferenceKernels_h

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    void fillPIDProbabilities(
        Queue& queue,
        uint32_t const* selectedTracksters,
        ticl::TracksterInferencePIDScoresSoA::ConstView scores,
        ticl::TracksterSoA::View outputTracksters,
        int32_t total);

}
#endif
