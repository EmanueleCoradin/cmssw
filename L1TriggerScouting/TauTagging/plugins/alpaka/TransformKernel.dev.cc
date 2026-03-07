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

  struct BuildKeysKernel {
    ALPAKA_FN_ACC void operator() (
      Acc1D const& acc,
      PFCandidateDeviceCollection::ConstView pf,
      IndexSoA::ConstView indexes, // const indexes of the input map
      uint32_t* keys) const {
        for (auto ii : uniform_elements(acc, indexes.metadata().size())) {
          auto pidx = indexes.indexes()[ii];
          auto pt_rescaled = pf.pt()[pidx] * 4.0f; // divide by 0.25
          pt_rescaled = alpaka::math::max(acc, pt_rescaled, 0.0f);
          pt_rescaled = alpaka::math::min(acc, pt_rescaled, 4294967295.0f);
          auto key = static_cast<uint32_t>(pt_rescaled);
          keys[ii] = 0xFFFFFFFFu - key; // max(uint32_t) - key in order to sort by descending value
        }
      }
  };
  
  struct ApplyPermKernel {
    ALPAKA_FN_ACC void operator() (
      Acc1D const& acc, 
      IndexSoA::ConstView indexes, 
      OffsetsSoA::ConstView offsets,
      IndexSoA::View indexes_perm, 
      uint16_t* perm) const {
        for (auto ii : independent_groups(acc, offsets.metadata().size() - 1)) {
          auto begin = offsets.offsets()[ii];
          auto end = offsets.offsets()[ii + 1];
          auto block_size = end - begin;
          if (block_size == 0)
            continue;
          
          for (auto jj : independent_group_elements(acc, block_size)) {
            auto new_local_idx = static_cast<uint32_t>(perm[begin + jj]);
            indexes_perm.indexes()[begin + jj] = indexes.indexes()[begin + new_local_idx];
          }
        }
      }
  };

  class ComputeClueTauFeaturesKernel {
  public:
    ALPAKA_FN_ACC void operator()(
      Acc1D const& acc,
      PFCandidateDeviceCollection::ConstView pf,
      IndexSoA::ConstView indexes, 
      OffsetsSoA::ConstView offsets,
      SoftTauInputDeviceTensor::View input_tensors) const {
        for (auto block_idx: independent_groups(acc, offsets.metadata().size() - 1)) {
          auto begin = offsets.offsets()[block_idx];
          auto end = offsets.offsets()[block_idx + 1];
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
              auto p = indexes.indexes()[ii + begin];
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
            Phi = jet_phi(acc, Py, Px);
          }

          // fill input tensor corresponding to the current cluster
          auto input_tensor = input_tensors[block_idx];
          auto input_size = (block_size > JetFeatures::RowsAtCompileTime) ? JetFeatures::RowsAtCompileTime : block_size;
          for (auto tid : independent_group_elements(acc, input_size)) {
            auto p = indexes.indexes()[begin + tid]; // global candidate index
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

  AssociationMapDevice sortClustersCandsMap(Queue& queue, 
                        const PFCandidateDeviceCollection& pf, 
                        const AssociationMapDevice& clusterCandsMap) {
    const auto num_clusters = clusterCandsMap.const_view<OffsetsSoA>().metadata().size() - 1;
    const auto num_clustered = clusterCandsMap.const_view<IndexSoA>().metadata().size();
    
    // device buffers needed for the radixSort to work
    auto keys = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, num_clustered);
    auto perm = cms::alpakatools::make_device_buffer<uint16_t[]>(queue, num_clustered);
    auto permWork = cms::alpakatools::make_device_buffer<uint16_t[]>(queue, num_clustered);

    alpaka::memset(queue, keys, 0u);
    alpaka::memset(queue, perm, 0u);
    alpaka::memset(queue, permWork, 0u);

    // define workdiv
    auto threadsPerBlock = 256;
    auto workDiv = make_workdiv<Acc1D>(num_clusters, threadsPerBlock);
    
    // generate uint32_t keys that are accepted by radixSort
    alpaka::exec<Acc1D>(
      queue, 
      workDiv, 
      BuildKeysKernel{}, 
      pf.const_view(), // global pf collection
      clusterCandsMap.const_view<IndexSoA>(), // indexes to retrieve pt of clustered candidates 
      alpaka::getPtrNative(keys) // get pointers to keys device buffer
    );
    
    // run radix sort
    alpaka::exec<Acc1D>(
      queue,
      workDiv, 
      radixSortMultiWrapper2<uint32_t, 4>{}, 
      alpaka::getPtrNative(keys), 
      alpaka::getPtrNative(perm),
      clusterCandsMap.const_view<OffsetsSoA>().offsets().data(),
      alpaka::getPtrNative(permWork)
    );
    
    // instantiate new sorted association map
    AssociationMapDevice clusterCandsMapSorted({{clusterCandsMap.const_view<IndexSoA>().metadata().size(),
                                                clusterCandsMap.const_view<OffsetsSoA>().metadata().size()}}, 
                                                queue); 
    alpaka::wait(queue);
      
    // apply permutation returned by radix sort
    alpaka::exec<Acc1D>(
      queue, 
      workDiv, 
      ApplyPermKernel{}, 
      clusterCandsMap.const_view<IndexSoA>(), 
      clusterCandsMap.const_view<OffsetsSoA>(), 
      clusterCandsMapSorted.view<IndexSoA>(),
      alpaka::getPtrNative(perm)
    );
    alpaka::wait(queue);

    // copy also the offsets column to the new sorted map
    auto srcOffsets = alpaka::createView(alpaka::getDev(queue),
                                        clusterCandsMap.const_view<OffsetsSoA>().offsets().data(), 
                                        Vec1D{clusterCandsMap.const_view<OffsetsSoA>().metadata().size()});
    auto dstOffsets = alpaka::createView(alpaka::getDev(queue),
                                        clusterCandsMapSorted.view<OffsetsSoA>().offsets().data(), 
                                        Vec1D{clusterCandsMapSorted.view<OffsetsSoA>().metadata().size()});
    alpaka::memcpy(queue, dstOffsets, srcOffsets);
    
    #if defined(__DEBUG__)
      alpaka::wait(queue);
      AssociationMapHost debugMap({{clusterCandsMap.const_view<IndexSoA>().metadata().size(),
                                    clusterCandsMap.const_view<OffsetsSoA>().metadata().size()}}, 
                                    queue);
      auto srcOffsetsDevice = alpaka::createView(alpaka::getDev(queue),
                                                clusterCandsMapSorted.const_view<OffsetsSoA>().offsets().data(), 
                                                Vec1D{clusterCandsMapSorted.const_view<OffsetsSoA>().metadata().size()});
      auto srcIndexesDevice = alpaka::createView(alpaka::getDev(queue),
                                                clusterCandsMapSorted.const_view<IndexSoA>().indexes().data(), 
                                                Vec1D{clusterCandsMapSorted.const_view<IndexSoA>().metadata().size()});
      auto dstOffsetsHost = alpaka::createView(alpaka::getDev(queue),
                                                debugMap.view<OffsetsSoA>().offsets().data(), 
                                                Vec1D{debugMap.view<OffsetsSoA>().metadata().size()});
      auto dstIndexesHost = alpaka::createView(alpaka::getDev(queue),
                                                debugMap.view<IndexSoA>().indexes().data(), 
                                                Vec1D{debugMap.view<IndexSoA>().metadata().size()});
      alpaka::memcpy(queue, dstOffsetsHost, srcOffsetsDevice);
      alpaka::memcpy(queue, dstIndexesHost, srcIndexesDevice);
      alpaka::wait(queue);

      std::vector<float> pt_vec(pf.const_view().metadata().size());
      auto srcPtDevice = alpaka::createView(alpaka::getDev(queue),
                                            pf.const_view().pt().data(), 
                                            Vec1D{pf.const_view().metadata().size()});
      alpaka::memcpy(queue, pt_vec, srcPtDevice);
      alpaka::wait(queue);
      
      auto num_clusters_new = debugMap.const_view<OffsetsSoA>().metadata().size() - 1;
      auto num_clustered_new = debugMap.const_view<IndexSoA>().metadata().size();
      
      for (auto ii = 0; ii < debugMap.const_view<OffsetsSoA>().metadata().size() - 1; ++ii) {
        std::cout << "------------------------------" << std::endl;
        auto begin = debugMap.const_view<OffsetsSoA>().offsets()[ii];
        auto end = debugMap.const_view<OffsetsSoA>().offsets()[ii + 1];
        std::cout << "ii = " << ii << ", begin = " << begin << ", end = " << end << ", size = " << end - begin << std::endl;
        
        for (auto jj = begin; jj < end; ++jj) {
          auto p = debugMap.const_view<IndexSoA>().indexes()[jj];
          std::cout << "p = " << p << ", pt[p] = " << pt_vec[p] << std::endl;
        }
      }
      
      std::cout << "Original num clusters = " << num_clusters << ", New num clusters = " << num_clusters_new << std::endl;
      std::cout << "Original num clustered = " << num_clustered << ", New num clusters = " << num_clustered_new << std::endl;
    #endif

    return clusterCandsMapSorted;
  }

  SoftTauInputDeviceTensor transform(Queue& queue, 
                                      const PFCandidateDeviceCollection& pf,
                                      const AssociationMapDevice& clusterCandsMap) {
    // initialize input tensor
    const auto num_clusters = clusterCandsMap.const_view<OffsetsSoA>().metadata().size() - 1;
    auto input_tensors = SoftTauInputDeviceTensor(num_clusters, queue);
    input_tensors.zeroInitialise(queue);

    // work division
    auto threadsPerBlock = 256;
    auto workDiv = make_workdiv<Acc1D>(num_clusters, threadsPerBlock);

    alpaka::exec<Acc1D>(queue, 
      workDiv, 
      ComputeClueTauFeaturesKernel{}, 
      pf.const_view(),
      clusterCandsMap.const_view<IndexSoA>(),
      clusterCandsMap.const_view<OffsetsSoA>(),
      input_tensors.view());
      
    return input_tensors;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels
