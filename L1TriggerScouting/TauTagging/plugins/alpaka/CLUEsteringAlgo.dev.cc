#include "L1TriggerScouting/TauTagging/plugins/alpaka/CLUEsteringAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  CLUEsteringAlgo::CLUEsteringAlgo(float dc, float rhoc, float dm, bool wrap_coords)
      : dc_(dc), rhoc_(rhoc), dm_(dm), wrap_coords_(wrap_coords) {}

  void CLUEsteringAlgo::run(Queue& queue,
                            const PFCandidateDeviceCollection& pf,
                            ClustersDeviceCollection& clusters) const {
    // CLUEstering call internally reinterpret_cast<T*> to non-const ptr
    auto& pf_cast = const_cast<PFCandidateDeviceCollection&>(pf);
    const uint32_t n_points = pf_cast.view().metadata().size();

    // buffers
    auto* coords_ptr = pf_cast.view().eta();
    auto* weights_ptr = pf_cast.view().pt();
    auto* cluster_indices_ptr = clusters.view().cluster();
    auto* is_seed_ptr = clusters.view().is_seed();

    // wrap device buffers
    auto points_device =
        clue::PointsDevice<kDims, Device>(queue, n_points, coords_ptr, weights_ptr, cluster_indices_ptr, is_seed_ptr);
    
    // run (wrap coords if enabled)
    auto clue_algo = clue::Clusterer<kDims>(queue, dc_, rhoc_, dm_);    
    if (wrap_coords_)
      clue_algo.setWrappedCoordinates({{0, 1}});
    clue_algo.make_clusters(queue, points_device);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels