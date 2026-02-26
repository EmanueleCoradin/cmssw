#ifndef L1TriggerScouting_TauTagging_plugins_alpaka_CLUEJetsProducerAlgo_h
#define L1TriggerScouting_TauTagging_plugins_alpaka_CLUEJetsProducerAlgo_h

#include <alpaka/alpaka.hpp>
#include <fmt/core.h> 

#include "CLUEstering/CLUEstering.hpp"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/AssociationMapDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/ClustersDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PFCandidateHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include <fstream>

// #define __DEBUG_CLUE__
// #define __DEBUG_DUMP__

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

    using namespace ::l1sc;

    constexpr size_t kDims = 2;

    class CLUEJetsProducerAlgo {
    public:
        explicit CLUEJetsProducerAlgo(float dc, float rhoc, float dm, bool wrap_coords);

        AssociationMapDevice run(Queue& queue,
                                const PFCandidateDeviceCollection& pf,
                                ClustersDeviceCollection& clusters) const;

    private:
        float dc_, rhoc_, dm_;
        bool wrap_coords_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels

#endif  // L1TriggerScouting_TauTagging_plugins_alpaka_CLUEJetsProducerAlgo_h