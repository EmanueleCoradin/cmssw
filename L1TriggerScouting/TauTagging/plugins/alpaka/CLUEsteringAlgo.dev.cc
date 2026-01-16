#include "L1TriggerScouting/TauTagging/plugins/alpaka/CLUEsteringAlgo.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {
  CLUEsteringAlgo::CLUEsteringAlgo(float dc, float rhoc, float dm, bool wrap_coords)
      : dc_(dc), rhoc_(rhoc), dm_(dm), wrap_coords_(wrap_coords) {}

  std::tuple<BxLookupDeviceCollection, AssociationMapDevice>
  CLUEsteringAlgo::run(Queue& queue,
                      const PFCandidateDeviceCollection& pf,
                      const BxLookupDeviceCollection& bx_sizes,
                      ClustersDeviceCollection& clusters) const {
    // move bx_sizes from device to host because make_clusters (batched) requires bx_sizes to be on the host
    const auto nbx = static_cast<int32_t>(bx_sizes.const_view<BxIndexSoA>().metadata().size());
    auto bx_sizes_host = BxLookupHostCollection({{nbx, nbx}}, queue);
    alpaka::memcpy(queue, bx_sizes_host.buffer(), bx_sizes.buffer());
    alpaka::wait(queue);
    
    // buffers
    // CLUEstering call internally reinterpret_cast<T*> to non-const ptr
    auto* eta_coord_ptr = const_cast<float*>(pf.const_view().eta().data());
    auto* phi_coord_ptr = const_cast<float*>(pf.const_view().phi().data());
    auto* weights_ptr = const_cast<float*>(pf.const_view().pt().data());
    auto* clusters_ptr = clusters.view().cluster().data();
    
    // create points
    const auto n_points = pf.const_view().metadata().size();
    auto points_device =
        clue::PointsDevice<kDims, Device>(queue, n_points, eta_coord_ptr, phi_coord_ptr, weights_ptr, clusters_ptr);
    auto clue_algo = clue::Clusterer<kDims>(queue, dc_, rhoc_, dm_);
    
    // call the clustering function
    clue_algo.make_clusters(queue, points_device, bx_sizes_host.const_view<OffsetsSoA>().offsets()); // here give bx_sizes as inputs
    
    // get clusters -> candidates (clue) association map and copy the buffer to 
    // a portable collection
    auto clusters_cands_map_clue = clue_algo.getClusters(queue, points_device);
    AssociationMapDevice clusters_cands_map({{static_cast<int>(clusters_cands_map_clue.extents().values), 
                                            static_cast<int>(clusters_cands_map_clue.extents().keys + 1)}}, 
                                            queue); // the last value of the keys buffer is actually not considered as a key
    auto dstIndexesClustersCands = alpaka::createView(alpaka::getDev(queue), 
                                        clusters_cands_map.view<IndexSoA>().indexes().data(),
                                        Vec1D{clusters_cands_map.view<IndexSoA>().metadata().size()});
    auto dstOffsetsClustersCands = alpaka::createView(alpaka::getDev(queue), 
                                        clusters_cands_map.view<OffsetsSoA>().offsets().data(),
                                        Vec1D{clusters_cands_map.view<OffsetsSoA>().metadata().size()});
    auto srcIndexesClustersCands = alpaka::createView(alpaka::getDev(queue),
                                      reinterpret_cast<const uint32_t *>(clusters_cands_map_clue.extract().values.data()), 
                                      Vec1D{clusters_cands_map_clue.extents().values});
    auto srcOffsetsClustersCands = alpaka::createView(alpaka::getDev(queue),
                                      reinterpret_cast<const uint32_t *>(clusters_cands_map_clue.extract().keys.data()), 
                                      Vec1D{clusters_cands_map_clue.extents().keys + 1});
    alpaka::memcpy(queue, dstIndexesClustersCands, srcIndexesClustersCands);
    alpaka::memcpy(queue, dstOffsetsClustersCands, srcOffsetsClustersCands); // here the actual dimension of the buffer is extents + 1
    
    // get bx -> clusters association map and copy the buffer to 
    // a portable collection
    auto bx_clusters_map_clue = clue_algo.getSampleAssociations(queue, points_device);
    BxLookupDeviceCollection bx_clusters_map({{static_cast<int>(bx_clusters_map_clue.extents().values), 
                                              static_cast<int>(bx_clusters_map_clue.extents().keys + 1)}}, queue);
    auto dstIndexesBxClusters = alpaka::createView(alpaka::getDev(queue), 
                                        bx_clusters_map.view<BxIndexSoA>().bx().data(),
                                        Vec1D{bx_clusters_map.view<BxIndexSoA>().bx().size()});
    auto dstOffsetsBxClusters = alpaka::createView(alpaka::getDev(queue), 
                                        bx_clusters_map.view<OffsetsSoA>().offsets().data(),
                                        Vec1D{bx_clusters_map.view<OffsetsSoA>().metadata().size()});
    auto srcIndexesBxClusters = alpaka::createView(alpaka::getDev(queue),
                                      reinterpret_cast<const uint32_t *>(bx_clusters_map_clue.extract().values.data()), 
                                      Vec1D{bx_clusters_map_clue.extents().values});
    auto srcOffsetsBxClusters = alpaka::createView(alpaka::getDev(queue),
                                      reinterpret_cast<const uint32_t *>(bx_clusters_map_clue.extract().keys.data()), 
                                      Vec1D{bx_clusters_map_clue.extents().keys + 1});

    alpaka::memcpy(queue, dstIndexesBxClusters, srcIndexesBxClusters);
    alpaka::memcpy(queue, dstOffsetsBxClusters, srcOffsetsBxClusters); // here the actual dimension of the buffer is extents + 1

    return std::make_tuple(std::move(bx_clusters_map), std::move(clusters_cands_map));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels
