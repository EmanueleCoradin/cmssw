#include "L1TriggerScouting/TauTagging/plugins/alpaka/TransformKernel.h"

#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  using namespace cms::alpakatools;

  template<typename TAcc, typename T>
  inline ALPAKA_FN_ACC void swap(TAcc const& acc, T &a, T &b) {
    T temp = a;
    a = b;
    b = temp;
  }

  class NotEfficientMaxKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, ClustersDeviceCollection::ConstView clusters, PortableCounter* n_clusters) const {
      if (once_per_grid(acc))
        n_clusters->value = 0;
      
      for (int32_t thread_idx : uniform_elements(acc, clusters.metadata().size())) {
        alpaka::atomicMax(acc, &n_clusters->value, clusters.cluster()[thread_idx]);
      }
    }
  };

  class NotEfficientHistKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, ClustersDeviceCollection::ConstView clusters, uint32_t* hist) const {
      for (uint32_t thread_idx : uniform_elements(acc, clusters.metadata().size())) {
        auto cluster_idx = clusters.cluster()[thread_idx];
        if (cluster_idx < 0)
          continue;
        alpaka::atomicAdd(acc, &hist[clusters.cluster()[thread_idx]], static_cast<uint32_t>(1));
      }
    }
  };

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
                                  PortableCounter* max_clusters,
                                  uint32_t* offsets,
                                  uint32_t* hist,
                                  uint16_t* indices) const {
      const uint8_t SHARED_MEM_BLOCK = 128;
      auto& sorted_indices = alpaka::declareSharedVar<int[SHARED_MEM_BLOCK], __COUNTER__>(acc);
      auto& shared_pt = alpaka::declareSharedVar<float[SHARED_MEM_BLOCK], __COUNTER__>(acc); 

      // define grid dimensions
      for (uint32_t block_idx: independent_groups(acc, max_clusters->value + 1)) {
        // fill shared mem (EOF flags)
        if (once_per_block(acc)) {
          for (auto idx = 0; idx < SHARED_MEM_BLOCK; idx++) {
            sorted_indices[idx] = -1;
            shared_pt[idx] = -1.0f;
          }
        }
        alpaka::syncBlockThreads(acc);

        // bind range to hw block
        uint32_t begin = offsets[block_idx];
        uint32_t end = offsets[block_idx + 1];
        // define block dimensions
        uint32_t block_dim = end - begin;
        if (block_dim == 0)
          continue;

        // fill shared mem
        for (uint32_t tid : independent_group_elements(acc, block_dim)) {
          uint32_t thread_idx = tid + begin; // global index
          sorted_indices[tid] = tid;
          shared_pt[tid] = pf.pt()[thread_idx];
        }
        alpaka::syncBlockThreads(acc);

        // odd-even sort algorithm
        for (uint32_t i = 0; i < block_dim; i++) {
          for (uint32_t tid : independent_group_elements(acc, block_dim - 1)) {
            if (tid + 1 < block_dim) {
              if ((i % 2 == 0 && tid % 2 == 0) || (i % 2 == 1 && tid % 2 == 1)) {
                if (shared_pt[tid] < shared_pt[tid + 1]) {
                  swap(acc, shared_pt[tid], shared_pt[tid + 1]);
                  swap(acc, sorted_indices[tid], sorted_indices[tid + 1]);
                }
              }
            }
            // sync tree
            alpaka::syncBlockThreads(acc);
          }
        }

        for (uint32_t tid : independent_group_elements(acc, block_dim)) {
          uint32_t thread_idx = tid + begin; // global index
          indices[thread_idx] = sorted_indices[tid];
        }
      }
    }
  };

  SoftTauInputDeviceTensor transform(Queue& queue, 
                 const PFCandidateDeviceCollection& pf, 
                 const ClustersDeviceCollection& clusters) {
    CounterDevice max_clusters(queue);
    alpaka::exec<Acc1D>(
        queue, make_workdiv<Acc1D>(64, clusters.const_view().metadata().size()),
        NotEfficientMaxKernel{}, clusters.const_view(), max_clusters.data());

    CounterHost max_clusters_host(queue);
    alpaka::memcpy(queue, max_clusters_host.buffer(), max_clusters.buffer());
    alpaka::wait(queue);

    const int num_clusters = max_clusters_host.data()->value + 1;
    SoftTauInputDeviceTensor input_tensor(num_clusters, queue);
    input_tensor.zeroInitialise(queue);

    auto hist_buf = make_device_buffer<uint32_t[]>(queue, num_clusters);
    alpaka::memset(queue, hist_buf, 0x00);
    auto offsets_buf = make_device_buffer<uint32_t[]>(queue, num_clusters+1);
    alpaka::memset(queue, offsets_buf, 0x00);

    // grid dims can be tuned for performance
    uint32_t threads_per_block = 256;
    uint32_t blocks_per_grid = divide_up_by(clusters.const_view().metadata().size(), threads_per_block);
    auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

    alpaka::exec<Acc1D>(queue, grid, NotEfficientHistKernel{}, clusters.const_view(), alpaka::getPtrNative(hist_buf));
    
    auto pc = alpaka::allocAsyncBuf<int32_t, Idx>(queue, Vec1D{blocks_per_grid});
    alpaka::memset(queue, pc, 0x00);

    alpaka::exec<Acc1D>(queue,
                        grid,
                        cms::alpakatools::multiBlockPrefixScan<uint32_t>{},
                        alpaka::getPtrNative(hist_buf),
                        alpaka::getPtrNative(offsets_buf) + 1,
                        alpaka::getExtents(offsets_buf)[0] - 1,
                        blocks_per_grid,
                        pc.data(),
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));

    // sort clusters by pt
    const auto max_part_per_cluster = 128;
    const auto num_pf = pf.const_view().metadata().size();
    auto index_buf = make_device_buffer<uint16_t[]>(queue, num_pf);

    alpaka::exec<Acc1D>(queue,
        make_workdiv<Acc1D>(num_clusters, max_part_per_cluster),
        TransformKernel{},
        pf.const_view(),
        clusters.const_view(),
        input_tensor.view(),
        max_clusters.data(),
        alpaka::getPtrNative(offsets_buf),
        alpaka::getPtrNative(hist_buf),
        alpaka::getPtrNative(index_buf));

    alpaka::exec<Acc1D>(queue,
        make_workdiv<Acc1D>(1, 1),
        [] ALPAKA_FN_ACC(Acc1D const& acc, PFCandidateDeviceCollection::ConstView pf,
                                  PortableCounter* max_clusters, uint32_t* offsets, uint16_t* indices) {
          if (once_per_grid(acc)) {
            for (int c = 0; c < max_clusters->value + 1; c++) {
              auto begin = offsets[c];
              auto end = offsets[c + 1];
              int cluster_size = end - begin;
              printf("Cluster %d [%d]", c, cluster_size);
              for (int i = 0; i < cluster_size; i++) {
                printf("%5d (%.2f)", indices[i+begin]+begin, pf.pt()[indices[i+begin]+begin]);
              }
              printf("\n");
            }
          }
        },
        pf.const_view(),
        max_clusters.data(),
        alpaka::getPtrNative(offsets_buf),
        alpaka::getPtrNative(index_buf));

    return input_tensor;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels
