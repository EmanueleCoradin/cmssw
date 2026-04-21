#include "L1TriggerScouting/TauTagging/plugins/alpaka/CLUEsteringAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  CLUEsteringAlgo::CLUEsteringAlgo(float dc, float rhoc, float dm, bool wrap_coords)
      : dc_(dc), rhoc_(rhoc), dm_(dm), wrap_coords_(wrap_coords) {}

  std::tuple<BxLookupDevice, ClustersDeviceCollection, AssociationMapDevice>
  CLUEsteringAlgo::run(Queue& queue,
                      const PFCandidateDeviceCollection& pf,
                      const BxLookupDevice& bx_sizes,
                      ClustersDeviceCollection& points_clusters) const {
    // move bx_sizes from device to host because make_clusters (batched) requires bx_sizes to be on the host
    const auto nbx = static_cast<int32_t>(bx_sizes.const_view().bx().metadata().size());
    auto bx_sizes_host = BxLookupHost(queue, nbx, nbx);
    alpaka::memcpy(queue, bx_sizes_host.buffer(), bx_sizes.buffer());
    alpaka::wait(queue);

    // buffers
    // CLUEstering call internally reinterpret_cast<T*> to non-const ptr
    auto* eta_coord_ptr = const_cast<float*>(pf.const_view().eta().data());
    auto* phi_coord_ptr = const_cast<float*>(pf.const_view().phi().data());
    auto* weights_ptr = const_cast<float*>(pf.const_view().pt().data());
    auto* clusters_ptr = points_clusters.view().cluster().data();
    
    // create points
    const auto n_points = pf.const_view().metadata().size();
    auto points_device =
        clue::PointsDevice<kDims, float, Device>(queue, n_points, eta_coord_ptr, phi_coord_ptr, weights_ptr, clusters_ptr);
    auto clue_algo = clue::Clusterer<kDims>(queue, dc_, rhoc_, dm_);
    
    // call the clustering function
    clue_algo.make_clusters(queue, points_device, bx_sizes_host.const_view().offset().offset()); // here give bx_sizes as inputs
    alpaka::wait(queue); // this is very important
    // get clusters -> candidates (clue) association map and copy the buffer to 
    // a portable collection
    auto clusters_cands_map_clue = clue_algo.getClusters(queue, points_device);

    AssociationMapDevice clusters_cands_map(queue,
                                            static_cast<int>(clusters_cands_map_clue.extents().values), 
                                            static_cast<int>(clusters_cands_map_clue.extents().keys + 1)); // the last value of the keys buffer is actually not considered as a key
    auto dstIndexesClustersCands = alpaka::createView(alpaka::getDev(queue), 
                                                      clusters_cands_map.view().index().index().data(),
                                                      Vec1D{clusters_cands_map.view().index().metadata().size()});
    auto dstOffsetsClustersCands = alpaka::createView(alpaka::getDev(queue), 
                                                      clusters_cands_map.view().offset().offset().data(),
                                                      Vec1D{clusters_cands_map.view().offset().metadata().size()});
    auto srcIndexesClustersCands = alpaka::createView(alpaka::getDev(queue),
                                                      reinterpret_cast<const uint32_t *>(clusters_cands_map_clue.extract().values.data()), 
                                                      Vec1D{clusters_cands_map_clue.extents().values});
    auto srcOffsetsClustersCands = alpaka::createView(alpaka::getDev(queue),
                                                      reinterpret_cast<const uint32_t *>(clusters_cands_map_clue.extract().keys.data()), 
                                                      Vec1D{clusters_cands_map_clue.extents().keys + 1});
    alpaka::memcpy(queue, dstIndexesClustersCands, srcIndexesClustersCands);
    alpaka::memcpy(queue, dstOffsetsClustersCands, srcOffsetsClustersCands); // here the actual dimension of the buffer is extents + 1

    /* BEGIN DEBUG */
    // std::vector<uint32_t> cc_indexes(static_cast<size_t>(clusters_cands_map.const_view().index().metadata().size()));
    // std::vector<uint32_t> cc_offsets(static_cast<size_t>(clusters_cands_map.const_view().offset().metadata().size()));
    // alpaka::memcpy(queue, cc_indexes, dstIndexesClustersCands);
    // alpaka::memcpy(queue, cc_offsets, dstOffsetsClustersCands);
    // alpaka::wait(queue);

    // auto max_size = std::max({cc_indexes.size(), cc_offsets.size()});
    // cc_indexes.resize(max_size, std::numeric_limits<uint32_t>::max());
    // cc_offsets.resize(max_size, std::numeric_limits<uint32_t>::max());
    
    // std::ofstream cc_map_stream("cc_map_v2.csv", std::ios::out);
    // cc_map_stream << "cand_idx,offset\n";
    // for (int i = 0; i < max_size; ++i) 
    //   cc_map_stream << fmt::format("{},{}\n", cc_indexes[i], cc_offsets[i]);
    // cc_map_stream.close();
    /* END DEBUG */

    // get bx -> clusters association map
    auto bx_clusters_map_clue = clue_algo.getSampleAssociations(queue, points_device);
    
    // BxLookup in which the indexes are the bx indexes and the offsets divide cluster indexes into bx
    assert(nbx + 1 == bx_clusters_map_clue.extents().keys + 1 && "The number of offset of bx_clusters_map_clue is expected to be nbx + 1");
    BxLookupDevice bx_clusters_map(queue,
                                    nbx,
                                    static_cast<int>(bx_clusters_map_clue.extents().keys + 1));

    auto dstIndexesBxClusters = alpaka::createView(alpaka::getDev(queue), 
                                        bx_clusters_map.view().bx().bx().data(),
                                        Vec1D{bx_clusters_map.const_view().bx().metadata().size()});
    auto dstOffsetsBxClusters = alpaka::createView(alpaka::getDev(queue), 
                                        bx_clusters_map.view().offset().offset().data(),
                                        Vec1D{bx_clusters_map.const_view().offset().metadata().size()});
    auto srcIndexesBxClusters = alpaka::createView(alpaka::getDev(queue),
                                      bx_sizes.const_view().bx().bx().data(), 
                                      Vec1D{bx_sizes.const_view().bx().metadata().size()});
    auto srcOffsetsBxClusters = alpaka::createView(alpaka::getDev(queue),
                                      reinterpret_cast<const uint32_t *>(bx_clusters_map_clue.extract().keys.data()), 
                                      Vec1D{bx_clusters_map_clue.extents().keys + 1});

    alpaka::memcpy(queue, dstIndexesBxClusters, srcIndexesBxClusters);
    alpaka::memcpy(queue, dstOffsetsBxClusters, srcOffsetsBxClusters);

    /* BEGIN DEBUG */
    // std::vector<uint16_t> bxc_indexes(static_cast<size_t>(bx_clusters_map.const_view().bx().metadata().size()));
    // std::vector<uint32_t> bxc_offsets(static_cast<size_t>(bx_clusters_map.const_view().offset().metadata().size()));
    // alpaka::memcpy(queue, bxc_indexes, dstIndexesBxClusters);
    // alpaka::memcpy(queue, bxc_offsets, dstOffsetsBxClusters);
    // alpaka::wait(queue);

    // auto max_size = std::max({bxc_indexes.size(), bxc_offsets.size()});
    // bxc_indexes.resize(max_size, std::numeric_limits<uint16_t>::max());
    // bxc_offsets.resize(max_size, std::numeric_limits<uint32_t>::max());
    
    // std::ofstream bxc_map_stream("bxc_map_new.csv", std::ios::out);
    // bxc_map_stream << "bx_idx,offset\n";
    // for (int i = 0; i < max_size; ++i) 
    //   bxc_map_stream << fmt::format("{},{}\n", bxc_indexes[i], bxc_offsets[i]);
    // bxc_map_stream.close();
    /* END DEBUG */
    
    // ClustersDeviceCollection to store the indexes of the clusters, accesible via the BxLookup
    ClustersDeviceCollection cluster_indexes(queue, static_cast<int>(bx_clusters_map_clue.extents().values));

    auto srcClusterIndexes = alpaka::createView(alpaka::getDev(queue),
                                      reinterpret_cast<const int32_t *>(bx_clusters_map_clue.extract().values.data()), 
                                      Vec1D{bx_clusters_map_clue.extents().values});

    auto dstClusterIndexes = alpaka::createView(alpaka::getDev(queue), 
                                      cluster_indexes.view().cluster().data(), 
                                      Vec1D{cluster_indexes.const_view().metadata().size()});

    alpaka::memcpy(queue, dstClusterIndexes, srcClusterIndexes);

    /* BEGIN DEBUG */
    // std::vector<int32_t> cidxs(static_cast<size_t>(cluster_indexes.const_view().metadata().size()));
    // alpaka::memcpy(queue, cidxs, dstClusterIndexes);
    // alpaka::wait(queue);

    // std::ofstream cluster_indexes_stream("cluster_indexes_new.csv", std::ios::out);
    // cluster_indexes_stream << "cidx\n";
    // for (int i = 0; i < cidxs.size(); ++i) 
    //   cluster_indexes_stream << cidxs[i] << std::endl;
    // cluster_indexes_stream.close();
    /* END DEBUG */

    // return
    return std::make_tuple(std::move(bx_clusters_map), std::move(cluster_indexes), std::move(clusters_cands_map));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels
