#include "L1TriggerScouting/TauTagging/plugins/alpaka/CLUEsteringAlgo.h"

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  template<typename TAcc, typename T>
  inline ALPAKA_FN_ACC void swap(TAcc const& acc, T &a, T &b) {
    T temp = a;
    a = b;
    b = temp;
  }

  using namespace cms::alpakatools;

  class SortClustersKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        const float* weights,
        IndexSoA::View indexes, 
        OffsetsSoA::View offsets) const {
      const uint8_t kSharedMemSize = 128;
      auto& indices_shared = alpaka::declareSharedVar<int[kSharedMemSize], __COUNTER__>(acc);
      auto& weights_shared = alpaka::declareSharedVar<float[kSharedMemSize], __COUNTER__>(acc); 

      // loop over clusters in parallel
      for (uint32_t block_idx: independent_groups(acc, offsets.metadata().size() - 1)) {
        // bind range to hw block
        uint32_t begin = offsets.offsets()[block_idx];
        uint32_t end = offsets.offsets()[block_idx + 1];
        // define block dimensions
        uint32_t block_dim = end - begin;
        if (block_dim == 0)
          continue;

        // load global to shared memory with EOF sentinels
        for (uint32_t tid : independent_group_elements(acc, kSharedMemSize)) {
          if (tid < block_dim) {
            uint32_t thread_idx = begin + tid;
            auto p_index = indexes.indexes()[thread_idx];
            indices_shared[tid] = p_index;
            weights_shared[tid] = weights[p_index];
          } else {
            // sentinel so unused slots never win
            indices_shared[tid] = -1;
            weights_shared[tid] = -1.0f;
          }
        }
        alpaka::syncBlockThreads(acc);

        // odd-even sort algorithm
        for (uint32_t i = 0; i < block_dim; i++) {
          for (uint32_t tid : independent_group_elements(acc, block_dim - 1)) {
            if (tid + 1 < block_dim) {
              if ((i % 2 == 0 && tid % 2 == 0) || (i % 2 == 1 && tid % 2 == 1)) {
                if (weights_shared[tid] < weights_shared[tid + 1]) {
                  swap(acc, weights_shared[tid], weights_shared[tid + 1]);
                  swap(acc, indices_shared[tid], indices_shared[tid + 1]);
                }
              }
            }
            // sync tree
            alpaka::syncBlockThreads(acc);
          }
        }

        // write back shared to global memory
        for (uint32_t tid : independent_group_elements(acc, block_dim)) {
          uint32_t thread_idx = tid + begin;
          indexes.indexes()[thread_idx] = indices_shared[tid];
        }
      }
    }
  };

  class CopyAssociatorKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc, 
        clue::AssociationMapView associator, 
        IndexSoA::View indexes, 
        OffsetsSoA::View offsets, 
        const float* weights) const {
      if (once_per_grid(acc)) {
        offsets.offsets()[0] = 0;
        for (int c_id = 0; c_id < offsets.metadata().size() - 1; c_id++) {
          auto span = associator[c_id];
          offsets.offsets()[c_id+1] = span.size() + offsets.offsets()[c_id];
          auto begin = offsets.offsets()[c_id];
          for (int i = 0; i < span.size(); i++) {
            indexes.indexes()[i+begin] = span[i];
          }
        }
      }
    }
  };

  CLUEsteringAlgo::CLUEsteringAlgo(float dc, float rhoc, float dm, bool wrap_coords)
      : dc_(dc), rhoc_(rhoc), dm_(dm), wrap_coords_(wrap_coords) {}

  AssociationMapDevice CLUEsteringAlgo::run(Queue& queue, const PFCandidateDeviceCollection& pf, ClustersDeviceCollection& clusters) const {
    const uint32_t n_points = pf.const_view().metadata().size();

    // buffers
    // CLUEstering call internally reinterpret_cast<T*> to non-const ptr
    auto* eta_coord_ptr = const_cast<float*>(pf.const_view().eta().data());
    auto* phi_coord_ptr = const_cast<float*>(pf.const_view().phi().data());
    auto* weights_ptr = const_cast<float*>(pf.const_view().pt().data());
    auto* clusters_ptr = clusters.view().cluster().data();

    // wrap device buffers
    auto points_device =
        clue::PointsDevice<kDims, Device>(queue, n_points, eta_coord_ptr, phi_coord_ptr, weights_ptr, clusters_ptr);
    // run (wrap coords if enabled)
    auto clue_algo = clue::Clusterer<kDims>(queue, dc_, rhoc_, dm_);
    if (wrap_coords_)
      clue_algo.setWrappedCoordinates({{0, 1}});
    clue_algo.make_clusters(queue, points_device);
    auto associator = clue_algo.getClusters(queue, points_device);

    auto association_soa = AssociationMapDevice({{static_cast<int>(n_points), static_cast<int>(associator.size()+1)}}, queue);
    association_soa.zeroInitialise(queue);

    alpaka::exec<Acc1D>(queue, 
      make_workdiv<Acc1D>(1, 1), 
      CopyAssociatorKernel{}, 
      associator.view(), 
      association_soa.view<IndexSoA>(), 
      association_soa.view<OffsetsSoA>(),
      pf.const_view().pt().data());

    alpaka::exec<Acc1D>(queue, 
      make_workdiv<Acc1D>(associator.size(), 128), 
      SortClustersKernel{}, 
      pf.const_view().pt().data(),
      association_soa.view<IndexSoA>(), 
      association_soa.view<OffsetsSoA>());

    return association_soa;
  }

  AssociationMapDevice CLUEsteringAlgo::run(Queue& queue,
                            const PFCandidateDeviceCollection& pf,
                            const BxLookupDeviceCollection& bx_lookup,
                            ClustersDeviceCollection& clusters) const {
    const auto nbx = static_cast<int32_t>(bx_lookup.const_view<BxIndexSoA>().metadata().size());
    auto bx_lookup_host = BxLookupHostCollection({{nbx, nbx + 1}}, queue);
    alpaka::memcpy(queue, bx_lookup_host.buffer(), bx_lookup.buffer());
    alpaka::wait(queue);

    auto n_clusters = 0;
    for (int32_t idx = 0; idx < bx_lookup_host.const_view<BxIndexSoA>().metadata().size(); idx++) {
      const auto begin = bx_lookup_host.const_view<OffsetsSoA>().offsets()[idx];
      const auto end = bx_lookup_host.const_view<OffsetsSoA>().offsets()[idx + 1];
      const uint32_t n_points = end - begin;

      if (n_points == 0)
        continue;

      // buffers
      // CLUEstering call internally reinterpret_cast<T*> to non-const ptr
      auto* eta_coord_ptr = const_cast<float*>(pf.const_view().eta().data() + begin);
      auto* phi_coord_ptr = const_cast<float*>(pf.const_view().phi().data() + begin);
      auto* weights_ptr = const_cast<float*>(pf.const_view().pt().data() + begin);
      auto* clusters_ptr = clusters.view().cluster().data() + begin;

      // wrap device buffers
      auto points_device =
          clue::PointsDevice<kDims, Device>(queue, n_points, eta_coord_ptr, phi_coord_ptr, weights_ptr, clusters_ptr);
      // run (wrap coords if enabled)
      auto clue_algo = clue::Clusterer<kDims>(queue, dc_, rhoc_, dm_);
      if (wrap_coords_)
        clue_algo.setWrappedCoordinates({{0, 1}});
      clue_algo.make_clusters(queue, points_device);
      auto associator = clue_algo.getClusters(queue, points_device);
      n_clusters += associator.size();
    }

    // tmp placeholder
    const auto n_pf = pf.const_view().metadata().size();
    auto association_soa = AssociationMapDevice({{n_pf, n_clusters}}, queue);
    association_soa.zeroInitialise(queue);

    return association_soa;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels
