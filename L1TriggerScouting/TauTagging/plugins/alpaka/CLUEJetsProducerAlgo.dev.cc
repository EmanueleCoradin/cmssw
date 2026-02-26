#include "L1TriggerScouting/TauTagging/plugins/alpaka/CLUEJetsProducerAlgo.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {
    CLUEJetsProducerAlgo::CLUEJetsProducerAlgo(float dc, float rhoc, float dm, bool wrap_coords) :
        dc_(dc), rhoc_(rhoc), dm_(dm), wrap_coords_(wrap_coords) {}

    AssociationMapDevice 
    CLUEJetsProducerAlgo::run(Queue& queue, 
                            const PFCandidateDeviceCollection& pf, 
                            ClustersDeviceCollection& clusters) const {
        
        // buffers
        // CLUEstering call internally reinterpret_cast<T*> to non-const ptr
        auto* eta_coord_ptr = const_cast<float*>(pf.const_view().eta().data());
        auto* phi_coord_ptr = const_cast<float*>(pf.const_view().phi().data());
        auto* weights_ptr = const_cast<float*>(pf.const_view().pt().data());
        auto* clusters_ptr = clusters.view().cluster().data();

        // create points
        const auto n_points = pf.const_view().metadata().size();

        auto x = cms::alpakatools::make_host_buffer<float[]>(queue, n_points);
        auto y = cms::alpakatools::make_host_buffer<float[]>(queue, n_points);
        auto w = cms::alpakatools::make_host_buffer<float[]>(queue, n_points);
        alpaka::memcpy(queue, x, cms::alpakatools::make_device_view(alpaka::getDev(queue), eta_coord_ptr, n_points));
        alpaka::memcpy(queue, y, cms::alpakatools::make_device_view(alpaka::getDev(queue), phi_coord_ptr, n_points));
        alpaka::memcpy(queue, w, cms::alpakatools::make_device_view(alpaka::getDev(queue), weights_ptr, n_points));

        for (auto i = 0; i < n_points; ++i) {
            std::cout << x[i] << ' ' << y[i] << ' ' << w[i] << std::endl;
        }

        auto points_device =
            clue::PointsDevice<kDims, float, Device>(queue, n_points, eta_coord_ptr, phi_coord_ptr, weights_ptr, clusters_ptr);
        auto clusterer = clue::Clusterer<kDims>(queue, dc_, rhoc_, dm_);
        std::cout << dc_ << ' ' << rhoc_ << ' ' << dm_ << std::endl;

        // call the clustering function
        clusterer.make_clusters(queue, points_device);

        // print number of clusters
        std::cout << "Number of clusters: " << points_device.n_clusters() << std::endl;

        clue::PointsHost<kDims> hp(queue, n_points);
        clue::copyToHost(queue, hp, points_device);
        alpaka::wait(queue);
        std::cout << "Number of clusters from host: " << hp.n_clusters() << std::endl;

        auto cluster_idx_host = hp.clusterIndexes();
        for (auto idx : cluster_idx_host) {
            std::cout << idx << ", ";
        }
        std::cout << std::endl;


        // get the associator
        auto associator = clusterer.getClusters(queue, points_device);

        // copy result to AssociationMapDevice
        auto clusters_cands_map = AssociationMapDevice({{static_cast<int32_t>(associator.extents().values), 
                                                    static_cast<int32_t>(associator.extents().keys + 1)}}, 
                                                    queue);

        auto dstIndexes = alpaka::createView(alpaka::getDev(queue), 
                                            clusters_cands_map.view<IndexSoA>().indexes().data(),
                                            Vec1D{clusters_cands_map.view<IndexSoA>().metadata().size()});
        auto dstOffsets = alpaka::createView(alpaka::getDev(queue), 
                                            clusters_cands_map.view<OffsetsSoA>().offsets().data(),
                                            Vec1D{clusters_cands_map.view<OffsetsSoA>().metadata().size()});
        auto srcIndexes = alpaka::createView(alpaka::getDev(queue), 
                                            reinterpret_cast<const uint32_t *>(associator.extract().values.data()),
                                            Vec1D{associator.extents().values});
        auto srcOffsets = alpaka::createView(alpaka::getDev(queue), 
                                            reinterpret_cast<const uint32_t *>(associator.extract().keys.data()),
                                            Vec1D{associator.extents().keys + 1});

        alpaka::memcpy(queue, dstIndexes, srcIndexes);
        alpaka::memcpy(queue, dstOffsets, srcOffsets);

        // return
        return clusters_cands_map;
    }
} // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels