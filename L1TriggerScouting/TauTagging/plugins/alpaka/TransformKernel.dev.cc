#include "L1TriggerScouting/TauTagging/plugins/alpaka/TransformKernel.h"

#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "DataFormats/L1ScoutingSoA/interface/AssociationMapHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  using namespace cms::alpakatools;

  template <typename TAcc, typename T>
  ALPAKA_FN_ACC T px(const TAcc& acc, T pt, T phi) {
    return pt * alpaka::math::cos(acc, phi);
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_ACC T py(const TAcc& acc, T pt, T phi) {
    return pt * alpaka::math::sin(acc, phi);
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_ACC T pz(const TAcc& acc, T pt, T eta) {
    return pt * alpaka::math::sinh(acc, eta);
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_ACC T energy(const TAcc& acc, T px, T py, T pz, T mass = 0.13957f) {
    return alpaka::math::sqrt(acc, 
        alpaka::math::pow(acc, px, 2) + 
        alpaka::math::pow(acc, py, 2) + 
        alpaka::math::pow(acc, pz, 2) + 
        alpaka::math::pow(acc, mass, 2));
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_ACC T jet_pt(const TAcc& acc, T Px, T Py) {
    return alpaka::math::sqrt(acc, alpaka::math::pow(acc, Px, 2) + alpaka::math::pow(acc, Py, 2));
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_ACC T jet_eta(const TAcc& acc, T Pt, T Pz) {
    return (Pt > 0.0) ? alpaka::math::asinh(acc, Pz / Pt) : 0.0; 
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_ACC T jet_phi(const TAcc& acc, T Px, T Py) {
    return alpaka::math::atan2(acc, Py, Px);
  }

  ALPAKA_FN_ACC float phi(Acc1D const& acc, float phi, float Phi) {
    auto kPi = alpaka::math::constants::pi;
    return alpaka::math::remainder(acc, phi - Phi + kPi, 2.0 * kPi) - kPi;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC float charge(const TAcc& acc, int pdgid) {
    auto charge = (alpaka::math::abs(acc, pdgid) == 211) ? ((pdgid > 0) ? +1 : -1) : ((pdgid > 0) ? -1 : +1);
    return charge;
  }

  struct BuildBxWiseRadixSortKeysOffsets {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator() (
          TAcc const &acc,
          BxLookupDevice::ConstView bx_clusters,
          AssociationMapDevice::ConstView clusters_cands,
          PFCandidateDeviceCollection::ConstView cands, // collection of candidates with features
          ClustersDeviceCollection::ConstView clusters, // cluster index assigned to each candidate (-1 if outlier)
          uint32_t* bx_clustered_cands_offset,
          uint32_t* keys
      ) const {
          const auto gridDim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0]; // this should correspond to the number of bx = 3564
          assert(gridDim == bx_clusters.bx().metadata().size() && "[BuildKeysBxWiseKernel] gridDim is not equal to the number of bxs");

          if (cms::alpakatools::once_per_grid(acc)) {
              bx_clustered_cands_offset[0] = 0;
          }

          for (auto bx_index : cms::alpakatools::independent_groups(acc, gridDim)) {
              const auto cluster_range_start = bx_clusters.offset()[bx_index].offset();
              const auto cluster_range_end = bx_clusters.offset()[bx_index + 1].offset();

              // here is the trick: identify the block of rows in the index column of clusters->cands that will be sorted
              const auto clusters_cands_row_start = clusters_cands.offset()[cluster_range_start].offset();
              const auto clusters_cands_row_end = clusters_cands.offset()[cluster_range_end].offset();
              bx_clustered_cands_offset[bx_index + 1] = clusters_cands_row_end;
              const auto rows_range_size = clusters_cands_row_end - clusters_cands_row_start;

              for (auto tid : cms::alpakatools::independent_group_elements(acc, rows_range_size)) {
                  const auto cand_local_idx = clusters_cands_row_start + tid;
                  const auto cand_idx = clusters_cands.index()[cand_local_idx].index(); // retrieve the global candidate index
                  const auto gcl = clusters.cluster()[cand_idx]; // get the global cluster ID of the current candidate
                  
                  assert(gcl > -1 && "[BuildKeysBxWiseKernel] Found a -1 global cluster index inside the clustered candidates");
                  assert(gcl >= cluster_range_start && "[BuildKeysBxWiseKernel] Global cluster index should be greater or equal than the base cluster index for the current bx");
                  assert(gcl < cluster_range_end && "[BuildKeysBxWiseKernel] Global cluster index should be smaller than the base cluster index for the following bx");
                  const auto lcl = gcl - cluster_range_start;
                  const auto pt_cand = cands.pt()[cand_idx];
                  const uint16_t pt_code = static_cast<uint16_t>(alpaka::math::max(acc, 65535.f - pt_cand * 32.f, 0.f));
                  keys[cand_local_idx] = (static_cast<uint32_t>(lcl) << 16) | pt_code;
              }
          }
      }
  }; // struct BuildBxWiseRadixSortKeysOffsets
  
  struct ApplyBxWisePermutationKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator() (
          TAcc const &acc,
          AssociationMapDevice::ConstView clusters_cands, 
          uint32_t* bx_clustered_cands_offset, 
          uint16_t* perm, 
          uint32_t* sorted_clusters_cands_indexes
      ) const {
          const auto gridDim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0];
          assert(gridDim == 3564 && "[ApplyBxWisePermutationKernel] gridDim is not equal to the number of bxs");

          for (auto bx_index : cms::alpakatools::independent_groups(acc, gridDim)) {
              const auto clusters_cands_row_start = bx_clustered_cands_offset[bx_index];
              const auto clusters_cands_row_end = bx_clustered_cands_offset[bx_index + 1];
              const auto rows_range_size = clusters_cands_row_end - clusters_cands_row_start;

              for (auto tid : cms::alpakatools::independent_group_elements(acc, rows_range_size)) {
                  sorted_clusters_cands_indexes[clusters_cands_row_start + tid] = clusters_cands.index()[clusters_cands_row_start + perm[clusters_cands_row_start + tid]].index();
              }
          }
      }
  }; // struct ApplyBxWisePermutationKernel

  class ComputeClueTauFeaturesKernel {
  public:
    ALPAKA_FN_ACC void operator()(
      Acc1D const& acc,
      PFCandidateDeviceCollection::ConstView pf,
      AssociationMapDevice::ConstView clusters_cands,
      SoftTauInputDeviceTensor::View input_tensors) const {
        for (auto block_idx: independent_groups(acc, clusters_cands.offset().metadata().size() - 1)) {
          auto begin = clusters_cands.offset()[block_idx].offset();
          auto end = clusters_cands.offset()[block_idx + 1].offset();
          auto block_size = end - begin;
          if (block_size == 0)
            continue;
        
          auto E = 0.0f;
          auto Px = 0.0f;
          auto Py = 0.0f;
          auto Pz = 0.0f;
          auto Pt = 0.0f;
          auto Eta = 0.0f;
          auto Phi = 0.0f;
          
          // build jet axis
          if (once_per_block(acc)) {
            for (auto ii = 0; ii < block_size; ii++) {
              auto p = clusters_cands.index()[ii + begin].index();
              auto px_v = px(acc, pf.pt()[p], pf.phi()[p]);
              auto py_v = py(acc, pf.pt()[p], pf.phi()[p]);
              auto pz_v = pz(acc, pf.pt()[p], pf.eta()[p]);
              auto e_v = energy(acc, px_v, py_v, pz_v);
              Px += px_v;
              Py += py_v;
              Pz += pz_v;
              E += e_v;
            }
            
            Pt = jet_pt(acc, Px, Py);
            Eta = jet_eta(acc, Pt, Pz);
            Phi = jet_phi(acc, Px, Py);
          }

          // fill input tensor corresponding to the current cluster
          auto input_tensor = input_tensors[block_idx];
          auto input_size = (block_size > JetFeatures::RowsAtCompileTime) ? JetFeatures::RowsAtCompileTime : block_size;
          for (auto tid : independent_group_elements(acc, input_size)) {
            auto p = clusters_cands.index()[begin + tid].index(); // global candidate index
            auto pdgid_abs = alpaka::math::abs(acc, static_cast<int>(pf.pdgid()[p]));
            // features
            input_tensor.features()(tid, 0) = pf.pt()[p];
            input_tensor.features()(tid, 1) = pf.eta()[p] - Eta;
            input_tensor.features()(tid, 2) = phi(acc, pf.phi()[p], Phi);
            input_tensor.features()(tid, 3) = charge(acc, pf.pdgid()[p]);
            input_tensor.features()(tid, 4) = pf.z0()[p];
            // one hot-encoding for pdgid
            input_tensor.features()(tid, 5) = (pdgid_abs == 211) ? 1.0f : 0.0f;
            input_tensor.features()(tid, 6) = (pdgid_abs == 130) ? 1.0f : 0.0f;
            input_tensor.features()(tid, 7) = (pdgid_abs == 11) ? 1.0f : 0.0f;
            input_tensor.features()(tid, 8) = (pdgid_abs == 13) ? 1.0f : 0.0f;
            input_tensor.features()(tid, 9) = (pdgid_abs == 22) ? 1.0f : 0.0f;
            // pad mask
            input_tensor.pad_mask()(tid) = 1.0f;
          }
        }
      }
  };

  class CopyInputChunkKernel {
  public:
    ALPAKA_FN_ACC void operator()(
      Acc1D const& acc,
      SoftTauInputDeviceTensor::ConstView full_input,
      SoftTauInputDeviceTensor::View chunk_input,
      uint32_t begin,
      uint32_t chunk_size) const {
        constexpr uint32_t nRows = JetFeatures::RowsAtCompileTime;
        constexpr uint32_t nCols = JetFeatures::ColsAtCompileTime;

        for (auto local_idx : cms::alpakatools::independent_groups(acc, chunk_size)) {
          const auto global_idx = begin + local_idx;

          auto src = full_input[global_idx];
          auto dst = chunk_input[local_idx];

          for (auto tid : cms::alpakatools::independent_group_elements(acc, nRows)) {
            dst.pad_mask()(tid) = src.pad_mask()(tid);
            for (uint32_t j = 0; j < nCols; ++j) {
              dst.features()(tid, j) = src.features()(tid, j);
            }
          }
        }
      }
  };


  class CopyOutputChunkKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        SoftTauOutputDeviceTensor::ConstView batch_output,
        SoftTauOutputDeviceTensor::View full_output,
        uint32_t begin,
        uint32_t chunk_size) const {
          for (auto local_idx : cms::alpakatools::independent_groups(acc, chunk_size)) {
            const auto global_idx = begin + local_idx;

            full_output.cls()[global_idx]    = batch_output.cls()[local_idx];
            full_output.vz()[global_idx]     = batch_output.vz()[local_idx];
            full_output.pt()[global_idx]     = batch_output.pt()[local_idx];
            full_output.charge()[global_idx] = batch_output.charge()[local_idx];
          }
        }
  };

  AssociationMapDevice sortClustersCandsMap(Queue& queue, 
                        const PFCandidateDeviceCollection& pf, 
                        const BxLookupDevice& bxClustersMap,
                        const AssociationMapDevice& clusterCandsMap, 
                        const ClustersDeviceCollection& clusters) {
    const auto nbx = bxClustersMap.const_view().bx().metadata().size();
    const auto num_clustered = clusterCandsMap.const_view().index().metadata().size();
    
    // device buffers needed for the radixSort to work
    auto perm = cms::alpakatools::make_device_buffer<uint16_t[]>(queue, num_clustered);
    auto permWork = cms::alpakatools::make_device_buffer<uint16_t[]>(queue, num_clustered);
    auto bxClusteredCandsOffsets = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nbx + 1);
    auto keys = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, num_clustered);

    alpaka::memset(queue, perm, 0u);
    alpaka::memset(queue, permWork, 0u);
    alpaka::memset(queue, bxClusteredCandsOffsets, 0u);
    alpaka::memset(queue, keys, 0u);

    // define workdiv
    auto threadsPerBlock = 256;
    auto workDiv = make_workdiv<Acc1D>(nbx, threadsPerBlock);
    
    // generate uint32_t keys that are accepted by radixSort
    alpaka::exec<Acc1D>(
      queue, 
      workDiv, 
      BuildBxWiseRadixSortKeysOffsets{}, 
      bxClustersMap.const_view(),
      clusterCandsMap.const_view(),
      pf.const_view(),
      clusters.const_view(),
      alpaka::getPtrNative(bxClusteredCandsOffsets), 
      alpaka::getPtrNative(keys) 
    );
    
    // run radix sort
    alpaka::exec<Acc1D>(
      queue,
      workDiv, 
      radixSortMultiWrapper2<uint32_t, 4>{}, 
      alpaka::getPtrNative(keys), 
      alpaka::getPtrNative(perm),
      alpaka::getPtrNative(bxClusteredCandsOffsets),
      alpaka::getPtrNative(permWork)
    );

    alpaka::wait(queue);
    
    // instantiate new sorted association map
    AssociationMapDevice clusterCandsMapSorted(queue,
                                                clusterCandsMap.const_view().index().metadata().size(),
                                                clusterCandsMap.const_view().offset().metadata().size()); 

    // apply permutation returned by radix sort
    alpaka::exec<Acc1D>(
      queue, 
      workDiv, 
      ApplyBxWisePermutationKernel{}, 
      clusterCandsMap.const_view(),
      alpaka::getPtrNative(bxClusteredCandsOffsets),
      alpaka::getPtrNative(perm), 
      clusterCandsMapSorted.view().index().index().data()
    );

    // copy also the offsets column to the new sorted map
    auto srcOffsets = alpaka::createView(alpaka::getDev(queue),
                                        clusterCandsMap.const_view().offset().offset().data(), 
                                        Vec1D{clusterCandsMap.const_view().offset().metadata().size()});
    auto dstOffsets = alpaka::createView(alpaka::getDev(queue),
                                        clusterCandsMapSorted.view().offset().offset().data(), 
                                        Vec1D{clusterCandsMapSorted.view().offset().metadata().size()});
    alpaka::memcpy(queue, dstOffsets, srcOffsets);
    alpaka::wait(queue);

    /* BEGIN DEBUG */
    // std::vector<uint32_t> cc_indexes(static_cast<size_t>(clusterCandsMapSorted.const_view().index().metadata().size()));
    // std::vector<uint32_t> cc_offsets(static_cast<size_t>(clusterCandsMapSorted.const_view().offset().metadata().size()));
    // auto dstIndexes = alpaka::createView(alpaka::getDev(queue),
    //                                     clusterCandsMapSorted.view().index().index().data(), 
    //                                     Vec1D{clusterCandsMapSorted.view().index().metadata().size()});
    // alpaka::memcpy(queue, cc_indexes, dstIndexes);
    // alpaka::memcpy(queue, cc_offsets, dstOffsets);
    // alpaka::wait(queue);

    // auto max_size = std::max({cc_indexes.size(), cc_offsets.size()});
    // cc_indexes.resize(max_size, std::numeric_limits<uint32_t>::max());
    // cc_offsets.resize(max_size, std::numeric_limits<uint32_t>::max());
    
    // std::ofstream cc_map_stream("cc_map_reordered_v2.csv", std::ios::out);
    // cc_map_stream << "cand_idx,offset\n";
    // for (int i = 0; i < max_size; ++i) 
    //   cc_map_stream << fmt::format("{},{}\n", cc_indexes[i], cc_offsets[i]);
    // cc_map_stream.close();
    /* END DEBUG */

    return clusterCandsMapSorted;
  }

  SoftTauInputDeviceTensor transform(Queue& queue, 
                                      const PFCandidateDeviceCollection& pf,
                                      const AssociationMapDevice& clusterCandsMap) {
    // initialize input tensor
    const auto num_clusters = clusterCandsMap.const_view().offset().metadata().size() - 1;
    auto input_tensors = SoftTauInputDeviceTensor(queue, num_clusters);
    input_tensors.zeroInitialise(queue);

    // work division
    auto threadsPerBlock = 256;
    auto workDiv = make_workdiv<Acc1D>(num_clusters, threadsPerBlock);

    alpaka::exec<Acc1D>(queue, 
      workDiv, 
      ComputeClueTauFeaturesKernel{}, 
      pf.const_view(),
      clusterCandsMap.const_view(),
      input_tensors.view());
      
    return input_tensors;
  }

  SoftTauInputDeviceTensor copyInputChunk(Queue& queue,
                                          const SoftTauInputDeviceTensor& full_input,
                                          uint32_t begin,
                                          uint32_t chunk_size) {
    auto chunk_input = SoftTauInputDeviceTensor(queue, chunk_size);
    chunk_input.zeroInitialise(queue);

    auto threadsPerBlock = 256;
    auto workDiv = make_workdiv<Acc1D>(chunk_size, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        CopyInputChunkKernel{},
                        full_input.const_view(),
                        chunk_input.view(),
                        begin,
                        chunk_size);

    return chunk_input;
  }

  void copyOutputChunk(Queue& queue,
                       const SoftTauOutputDeviceTensor& batch_output,
                       SoftTauOutputDeviceTensor& full_output,
                       uint32_t begin,
                       uint32_t chunk_size) {
    auto threadsPerBlock = 256;
    auto workDiv = make_workdiv<Acc1D>(chunk_size, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        CopyOutputChunkKernel{},
                        batch_output.const_view(),
                        full_output.view(),
                        begin,
                        chunk_size);
  }

  
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels
