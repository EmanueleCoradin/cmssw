#include "L1TriggerScouting/TauTagging/plugins/alpaka/TransformKernel.h"

#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  using namespace cms::alpakatools;

  SoftTauInputDeviceTensor transform(Queue& queue, 
                 const PFCandidateDeviceCollection& pf, 
                 const BxLookupDeviceCollection& bx_lookup, 
                 const ClustersDeviceCollection& clusters) {
    return SoftTauInputDeviceTensor(1, queue);
  }

  class TransformKernel {
  public:
    template <typename TAcc>
      requires alpaka::isAccelerator<TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  PFCandidateDeviceCollection::ConstView pf,
                                  ClustersDeviceCollection::ConstView clusters,
                                  SoftTauInputDeviceTensor::View input_tensor,
                                  PortableCounter* num_clusters) const {
      // uint32_t grid_dim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0];
      // uint32_t block_dim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0];
      // for (uint32_t cluster_idx : independent_groups(acc, grid_dim)) {
      //   for (uint32_t tid : independent_group_elements(acc, block_dim)) {
      //     printf("%d, %d", cluster_idx, tid);
      //   }
      // }
    }
  };

  SoftTauInputDeviceTensor transform(Queue& queue, 
                 const PFCandidateDeviceCollection& pf, 
                 const ClustersDeviceCollection& clusters) {
    CounterDevice n_clusters(queue);
    alpaka::exec<Acc1D>(
        queue,
        make_workdiv<Acc1D>(64, clusters.const_view().metadata().size()),
        [] ALPAKA_FN_ACC(Acc1D const& acc, ClustersDeviceCollection::ConstView clusters, PortableCounter* n_clusters) {
          if (once_per_grid(acc))
            n_clusters->value = -1;
          
          for (int32_t thread_idx : uniform_elements(acc, clusters.metadata().size())) {
            alpaka::atomicMax(acc, &n_clusters->value, clusters.cluster()[thread_idx]);
          }
        },
        clusters.const_view(),
        n_clusters.data());

    CounterHost n_clusters_host(queue);
    alpaka::memcpy(queue, n_clusters_host.buffer(), n_clusters.buffer());
    alpaka::wait(queue);

    const int num_clusters = n_clusters_host.data()->value + 1;
    SoftTauInputDeviceTensor input_tensor(num_clusters, queue);
    input_tensor.zeroInitialise(queue);

    // grid dims can be tuned for performance
    uint32_t threads_per_block = 64;
    uint32_t blocks_per_grid = num_clusters;
    auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

    alpaka::exec<Acc1D>(queue,
          grid,
          TransformKernel{},
          pf.const_view(),
          clusters.const_view(),
          input_tensor.view(),
          n_clusters.data());

    return input_tensor;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels
