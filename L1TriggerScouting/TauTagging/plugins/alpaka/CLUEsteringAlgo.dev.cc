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

    #if defined(__DEBUG_CLUE__)
      auto h_points = clue::PointsHost<kDims>(queue, n_points);
      clue::copyToHost(queue, h_points, points_device);
      alpaka::wait(queue);

      auto clusters_ = clue::get_clusters(h_points);
      
      // dump the content to csv
      std::ofstream cc_map_host("cc_map_host.csv", std::ios::out);
      cc_map_host << "key,value\n";
      for (std::size_t clu_idx = 0; clu_idx < clusters_.size(); ++clu_idx) {
        for (auto it = clusters_.lower_bound(clu_idx); it != clusters_.upper_bound(clu_idx); ++it) {
          cc_map_host << fmt::format("{},{}\n", clu_idx, *it);
        }
      }
      cc_map_host.close();

      // create views of the clusters_ host container buffers
      auto srcIndexesClusters_ = alpaka::createView(cms::alpakatools::host(),
                                      reinterpret_cast<const uint32_t *>(clusters_.extract().values.data()), 
                                      Vec1D{clusters_.extents().values});
      auto srcOffsetsClusters_ = alpaka::createView(cms::alpakatools::host(),
                                      reinterpret_cast<const uint32_t *>(clusters_.extract().keys.data()), 
                                      Vec1D{clusters_.extents().keys + 1});
                                      
      // create vectors to copy the content of the host container buffers
      std::vector<uint32_t> cc_indexes_host(static_cast<size_t>(clusters_.extents().values));
      std::vector<uint32_t> cc_offsets_host(static_cast<size_t>(clusters_.extents().keys + 1));
      alpaka::memcpy(queue, cc_indexes_host, srcIndexesClusters_);
      alpaka::memcpy(queue, cc_offsets_host, srcOffsetsClusters_);
      alpaka::wait(queue);
      
      // resize vectors to the same length
      auto max_size = std::max({cc_indexes_host.size(), cc_offsets_host.size()});
      cc_indexes_host.resize(max_size, std::numeric_limits<uint32_t>::max());
      cc_offsets_host.resize(max_size, std::numeric_limits<uint32_t>::max());

      // dump the content to csv
      std::ofstream cc_buffers_host("cc_buffers_host.csv", std::ios::out);
      cc_buffers_host << "ClusterCandIdx,ClusterCandOff\n";
      for (int i = 0; i < max_size; ++i) 
        cc_buffers_host << fmt::format("{},{}\n", cc_indexes_host[i], cc_offsets_host[i]);
      cc_buffers_host.close();
    #endif

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

    #if defined(__DEBUG_DUMP__)
      std::vector<uint32_t> cc_indexes(static_cast<size_t>(clusters_cands_map_clue.extents().values));
      std::vector<uint32_t> cc_offsets(static_cast<size_t>(clusters_cands_map_clue.extents().keys + 1));
      alpaka::memcpy(queue, cc_indexes, srcIndexesClustersCands);
      alpaka::memcpy(queue, cc_offsets, srcOffsetsClustersCands);
      alpaka::wait(queue);

      std::vector<uint32_t> bxc_indexes(static_cast<size_t>(bx_clusters_map_clue.extents().values));
      std::vector<uint32_t> bxc_offsets(static_cast<size_t>(bx_clusters_map_clue.extents().keys + 1));
      alpaka::memcpy(queue, bxc_indexes, srcIndexesBxClusters);
      alpaka::memcpy(queue, bxc_offsets, srcOffsetsBxClusters);
      alpaka::wait(queue);

      // resize vectors to same shape
      max_size = std::max({cc_indexes.size(), cc_offsets.size(), bxc_indexes.size(), bxc_offsets.size()});
      cc_indexes.resize(max_size, std::numeric_limits<uint32_t>::max());
      cc_offsets.resize(max_size, std::numeric_limits<uint32_t>::max());
      bxc_indexes.resize(max_size, std::numeric_limits<uint32_t>::max());
      bxc_offsets.resize(max_size, std::numeric_limits<uint32_t>::max());

      std::ofstream bxc_cc_buffers_device("bxc_cc_buffers_device.csv", std::ios::out);
      bxc_cc_buffers_device << "BxClusterIdx,BxClusterOff,ClusterCandIdx,ClusterCandOff\n";
      for (int i = 0; i < max_size; ++i) 
        bxc_cc_buffers_device << fmt::format("{},{},{},{}\n", bxc_indexes[i], bxc_offsets[i], cc_indexes[i], cc_offsets[i]);
      bxc_cc_buffers_device.close();
    #endif

    return std::make_tuple(std::move(bx_clusters_map), std::move(clusters_cands_map));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels
