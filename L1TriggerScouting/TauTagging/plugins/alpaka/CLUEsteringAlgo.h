#ifndef L1TriggerScouting_TauTagging_plugins_alpaka_CLUEsteringAlgo_h
#define L1TriggerScouting_TauTagging_plugins_alpaka_CLUEsteringAlgo_h

#include <alpaka/alpaka.hpp>
#include <fmt/core.h> 

#include "CLUEstering/CLUEstering.hpp"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/BxLookupDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/AssociationMapDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/ClustersDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PFCandidateHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include <fstream>

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  using namespace ::l1sc;

  constexpr size_t kDims = 2;

  class CLUEsteringAlgo {
  public:
    explicit CLUEsteringAlgo(float dc, float rhoc, float dm, bool wrap_coords);
    typedef std::tuple<BxLookupDevice, ClustersDeviceCollection, AssociationMapDevice> return_type;

    return_type run(Queue& queue,
                    const PFCandidateDeviceCollection& pf,
                    const BxLookupDevice& bx_sizes,
                    ClustersDeviceCollection& points_clusters) const;


  private:
    float dc_, rhoc_, dm_;
    bool wrap_coords_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels

#endif  // L1TriggerScouting_TauTagging_plugins_alpaka_CLUEsteringAlgo_h