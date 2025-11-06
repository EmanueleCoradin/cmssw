#include "L1TriggerScouting/Phase2/plugins/alpaka/L1TScPhase2SCJetsKernels.h"

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"
#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"

//#define L1TSC_VERBOSE_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  using namespace cms::alpakatools;

  class JetKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  PuppiDeviceCollection::ConstView puppi,
                                  OffsetsSoA::ConstView bx_lookup,
                                  BxIndexSoA::ConstView bx_index,
                                  float R,
                                  float R2,
                                  uint16_t* uieta,
                                  uint16_t* idx,
                                  ClusterObjDeviceCollection::View work,
                                  ClustersDeviceCollection::View clusters,
                                  ClusterObjDeviceCollection::View jets,
                                  OffsetsSoA::View jetBxLookup,
                                  BxIndexSoA::View jetBxIndex,
                                  unsigned int* nJetsTotal) const {
      if (cms::alpakatools::once_per_grid(acc))
        jetBxLookup.offsets()[0] = 0;
      uint32_t grid_dim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0];
      // step-0: sort by eta in blocks
      for (uint32_t block_idx : independent_groups(acc, grid_dim)) {
        // get event range
        uint32_t begin = bx_lookup.offsets()[block_idx];
        uint32_t end = bx_lookup.offsets()[block_idx + 1];
        if (end <= begin)
          continue;
        uint32_t block_dim = end - begin;
#ifdef L1TSC_VERBOSE_DEBUG
        if (block_idx <= 2)
          printf("\nBlock %u of size %u (from %u to %u):\n", block_idx, block_dim, begin, end);
#endif
        for (uint32_t tid : independent_group_elements(acc, block_dim)) {
          uieta[tid + begin] = (puppi.eta()[tid + begin] + 5.f) * (std::numeric_limits<uint16_t>::max() / 10.0f);
          idx[tid + begin] = tid;  // this is important for the GPU implementation of radixSort
#ifdef L1TSC_VERBOSE_DEBUG
          if (block_idx <= 2)
            printf(" -  pt %7.2f eta %+6.3f phi %+6.3f  index %d, id %u --> uieta %u\n",
                   puppi.pt()[tid + begin],
                   puppi.eta()[tid + begin],
                   puppi.phi()[tid + begin],
                   tid,
                   tid + begin,
                   uieta[tid + begin]);
#endif
        }
      }
      radixSortMulti(acc, uieta, idx, bx_lookup.offsets().data(), nullptr);

      // step-1: rearrange
      for (uint32_t block_idx : independent_groups(acc, grid_dim)) {
        // get event range
        uint32_t begin = bx_lookup.offsets()[block_idx];
        uint32_t end = bx_lookup.offsets()[block_idx + 1];
        if (end <= begin)
          continue;
        uint32_t block_dim = end - begin;
#ifdef L1TSC_VERBOSE_DEBUG
        if (block_idx <= 2)
          printf("\nRearranged:\n");
#endif
        for (uint32_t tid : independent_group_elements(acc, block_dim)) {
          auto ipart = tid + begin;  // global index
          auto isrc = idx[ipart] + begin;
          work.pt()[ipart] = puppi.pt()[isrc];
          work.eta()[ipart] = puppi.eta()[isrc];
          work.phi()[ipart] = puppi.phi()[isrc];
          work.cluster()[ipart] = isrc;
#ifdef L1TSC_VERBOSE_DEBUG
          if (block_idx <= 2)
            printf(" -  %3d pt %7.2f eta %+6.3f phi %+6.3f id %u from %u\n",
                   tid,
                   work.pt()[tid + begin],
                   work.eta()[tid + begin],
                   work.phi()[tid + begin],
                   isrc,
                   idx[ipart]);
#endif
        }
      }

      auto& nseeds = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      if (once_per_block(acc))
        nseeds = 0;
      alpaka::syncBlockThreads(acc);

      // step-2: seed finding
      for (uint32_t block_idx : independent_groups(acc, grid_dim)) {
        // get event range
        uint32_t begin = bx_lookup.offsets()[block_idx];
        uint32_t end = bx_lookup.offsets()[block_idx + 1];
        // skip if malformed or empty
        if (end <= begin)
          continue;

        uint32_t block_dim = end - begin;
        // pre-cluster
        for (uint32_t tid : independent_group_elements(acc, block_dim)) {
          // try this as a seed
          uint32_t iseed = tid + begin, icluster = work.cluster()[iseed];  // global index
          float seed_pt = work.pt()[iseed], seed_eta = work.eta()[iseed], seed_phi = work.phi()[iseed];
          float sum_pt = seed_pt, sum_eta = 0, sum_phi = 0;
          bool is_seed = true;
          // scan up

          for (uint32_t j = tid; j > 0; --j) {
            uint32_t ipart = j - 1 + begin;  // global index
            float deta = work.eta()[ipart] - seed_eta;
            if (deta < -R)  // sorted in eta, so we can stop here
              break;
            float dphi = cms::alpakatools::deltaPhi(acc, work.phi()[ipart], seed_phi);
            if (deta * deta + dphi * dphi < R2) {
              if (work.pt()[ipart] >= seed_pt) {  // here we use >=, since we're for indices above ipart
                is_seed = false;
                break;
              } else {
                sum_pt += work.pt()[ipart];
                sum_eta += work.pt()[ipart] * deta;
                sum_phi += work.pt()[ipart] * dphi;
              }
            }
          }
          for (uint32_t j = tid + 1; j < block_dim; ++j) {
            uint32_t ipart = j + begin;  // global index
            float deta = work.eta()[ipart] - seed_eta;
            if (deta > R)  // sorted in eta, so we can stop here
              break;
            float dphi = cms::alpakatools::deltaPhi(acc, work.phi()[ipart], seed_phi);
            if (deta * deta + dphi * dphi < R2) {
              if (work.pt()[ipart] > seed_pt) {  // note: here we use > instead of >=
                is_seed = false;
                break;
              } else {
                sum_pt += work.pt()[ipart];
                sum_eta += work.pt()[ipart] * deta;
                sum_phi += work.pt()[ipart] * dphi;
              }
            }
          }
#ifdef L1TSC_VERBOSE_DEBUG
          if (block_idx <= 2)
            printf("Cluster %u at index %u pt %7.2f eta %+6.3f phi %+6.3f, %s a seed\n",
                   icluster,
                   iseed,
                   seed_pt,
                   seed_eta,
                   seed_phi,
                   is_seed ? "is" : "is not");
#endif
          sum_eta = seed_eta + sum_eta / sum_pt;
          sum_phi = cms::alpakatools::reducePhiRange(acc, seed_phi + sum_phi / sum_pt);
          if (is_seed) {
            auto ijet = alpaka::atomicAdd(acc, &nseeds, 1u, alpaka::hierarchy::Threads{}) + begin;
            jets.pt()[ijet] = sum_pt;
            jets.eta()[ijet] = sum_eta;
            jets.phi()[ijet] = sum_phi;
            jets.cluster()[ijet] = ijet - begin;
            jets.numberOfDaughters()[ijet] = 0;
            clusters.is_seed()[icluster] = 1;
#ifdef L1TSC_VERBOSE_DEBUG
            if (block_idx <= 2)
              printf(
                  "Jet %3u pt %7.2f eta %+6.3f phi %+6.3f, from seed index %d, id %u pt %7.2f eta %+6.3f phi "
                  "%+6.3f\n\n",
                  ijet - begin,
                  jets.pt()[ijet],
                  jets.eta()[ijet],
                  jets.phi()[ijet],
                  iseed,
                  icluster,
                  seed_pt,
                  seed_eta,
                  seed_phi);
#endif
          }
        }
        alpaka::syncBlockThreads(acc);

        // reassociate
        for (uint32_t tid : independent_group_elements(acc, block_dim)) {
          auto ipart = tid + begin;                   // global index
          uint32_t icluster = work.cluster()[ipart];  // original index of the item
          float nearest = R2;
          int jcluster = -1;
          for (uint32_t j = 0; j < nseeds; ++j) {
            auto jseed = j + begin;  // global index
            float deta = work.eta()[ipart] - jets.eta()[jseed];
            float dphi = cms::alpakatools::deltaPhi(acc, work.phi()[ipart], jets.phi()[jseed]);
            float dr2 = deta * deta + dphi * dphi;
            if (dr2 < nearest) {
              jcluster = jets.cluster()[jseed];
              nearest = dr2;
            }
          }
          if (jcluster != -1) {
            alpaka::atomicAdd(acc, &jets.numberOfDaughters()[jcluster + begin], 1u, alpaka::hierarchy::Threads{});
#ifdef L1TSC_VERBOSE_DEBUG
            if (block_idx <= 2)
              printf("Particle %3u (globally %3u) assigned to jet %3u (globally %u) with deltaR %.5f\n",
                     tid,
                     ipart,
                     jcluster,
                     jcluster + begin,
                     std::sqrt(nearest));
#endif
          }
          clusters.cluster()[icluster] = jcluster;
        }

        if (once_per_block(acc)) {
          jetBxIndex.bx()[block_idx] = bx_index.bx()[block_idx];
          jetBxLookup.offsets()[block_idx + 1] = nseeds;
          alpaka::atomicAdd(acc, nJetsTotal, nseeds, alpaka::hierarchy::Blocks{});
        }
        alpaka::syncBlockThreads(acc);
      }  // block
    }  // operator()
  };  // class

  class JetZSKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  OffsetsSoA::ConstView bxLookup,
                                  ClusterObjDeviceCollection::ConstView jetsNonZS,
                                  OffsetsSoA::ConstView jetBxLookup,
                                  ClusterObjDeviceCollection::View jets,
                                  unsigned int* nClustered) const {
      uint32_t grid_dim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0];
      for (uint32_t block_idx : independent_groups(acc, grid_dim)) {
        // get event range
        uint32_t beginSrc = bxLookup.offsets()[block_idx];
        uint32_t beginDst = jetBxLookup.offsets()[block_idx];
        uint32_t endDst = jetBxLookup.offsets()[block_idx + 1];
        if (endDst <= beginDst)
          continue;
#ifdef L1TSC_VERBOSE_DEBUG
        if (once_per_block(acc) && block_idx <= 2)
          printf("In BX %u jet range %u - %u, ZS range %u - %u\n",
                 block_idx + 1,
                 beginSrc,
                 bxLookup.offsets()[block_idx + 1],
                 beginDst,
                 endDst);
#endif
        for (uint32_t tid : independent_group_elements(acc, endDst - beginDst)) {
          uint32_t ijetDst = tid + beginDst;
          uint32_t ijetSrc = tid + beginSrc;
          jets.pt()[ijetDst] = jetsNonZS.pt()[ijetSrc];
          jets.eta()[ijetDst] = jetsNonZS.eta()[ijetSrc];
          jets.phi()[ijetDst] = jetsNonZS.phi()[ijetSrc];
          jets.cluster()[ijetDst] = jetsNonZS.cluster()[ijetSrc];
          jets.numberOfDaughters()[ijetDst] = jetsNonZS.numberOfDaughters()[ijetSrc];
#ifdef L1TSC_VERBOSE_DEBUG
          if (block_idx <= 2)
            printf("In BX %u jet %u/%u daughters %u nonZS index %u --> ZS index %u\n",
                   block_idx + 1,
                   tid,
                   endDst - beginDst,
                   jets.numberOfDaughters()[ijetDst],
                   ijetSrc,
                   ijetDst);
#endif
        }
        alpaka::syncBlockThreads(acc);
        if (once_per_block(acc)) {
          // simpler to do it on a single thread
          unsigned int nClusteredBlock = 0;
          for (uint32_t iJet = beginDst; iJet < endDst; ++iJet) {
            nClusteredBlock += jets.numberOfDaughters()[iJet];
          }
          alpaka::atomicAdd(acc, nClustered, nClusteredBlock, alpaka::hierarchy::Blocks{});
        }  // once per block
      }  // block loop
    }  // operator()
  };  // class

  class JetToAssociationMapKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  OffsetsSoA::ConstView bxLookup,
                                  OffsetsSoA::ConstView jetBxLookup,
                                  OffsetsSoA::ConstView clusterdParticleOffsets,
                                  ClustersDeviceCollection::ConstView clusters,
                                  const unsigned int njets,
                                  const unsigned int nclustered,
                                  uint16_t* key,
                                  uint16_t* idx,
                                  IndexSoA::View map) const {
      uint32_t grid_dim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0];
      unsigned int nparticles = clusters.metadata().size();
      for (uint32_t block_idx : independent_groups(acc, grid_dim)) {
        // get event range
        uint32_t begin = bxLookup.offsets()[block_idx];
        uint32_t end = bxLookup.offsets()[block_idx + 1];
        if (end <= begin)
          continue;
        for (uint32_t tid : independent_group_elements(acc, end - begin)) {
          uint32_t i = tid + begin;
          assert(i < nparticles);
          assert(clusters.cluster()[i] == -1 || clusters.cluster()[i] < int(end - begin));
          key[i] = static_cast<uint16_t>(clusters.cluster()[i]);
          idx[i] = tid;
#ifdef L1TSC_VERBOSE_DEBUG
          if (block_idx <= 2)
            printf(
                "In BX %u cand %3u/%3u assigned to cluster key %u\n", block_idx + 1, tid, end - begin, unsigned(key[i]));
#endif
        }
      }
      alpaka::syncBlockThreads(acc);
      // sort
      radixSortMulti(acc, key, idx, bxLookup.offsets().data(), nullptr);
      // rearrange jet constituents
      for (uint32_t block_idx : independent_groups(acc, grid_dim)) {
        // get range of particles in this BX
        uint32_t begin = bxLookup.offsets()[block_idx];
        uint32_t end = bxLookup.offsets()[block_idx + 1];
        if (end <= begin)
          continue;
        // get the offset, i.e. global index, of the first and one-past-the-last jet in this BX
        uint32_t jetsBegin = jetBxLookup.offsets()[block_idx];
        uint32_t jetsEnd = jetBxLookup.offsets()[block_idx + 1];
        assert(jetsEnd >= jetsBegin);
        assert(jetsEnd <= njets);
        assert(jetBxLookup.offsets()[grid_dim] == njets);
        assert(clusterdParticleOffsets.metadata().size() == int(njets + 1));
        assert(clusterdParticleOffsets.offsets()[njets] == nclustered);
        // get the offset of the consitutents of the first jet in this BX
        uint32_t clusteredBegin = clusterdParticleOffsets.offsets()[jetsBegin];
        uint32_t clusteredEnd = clusterdParticleOffsets.offsets()[jetsEnd];
#ifdef L1TSC_VERBOSE_DEBUG
        if (once_per_block(acc) && block_idx <= 2000)
          printf("In BX %u, particle offsets %u - %u (%u), jets %u - %u (%u), clustered particles %u - %u (%u) \n",
                 block_idx + 1,
                 begin,
                 end,
                 end - begin,
                 jetsBegin,
                 jetsEnd,
                 jetsEnd - jetsBegin,
                 clusteredBegin,
                 clusteredEnd,
                 clusteredEnd - clusteredBegin);
#endif
        assert(clusteredEnd >= clusteredBegin);
        assert((clusteredEnd - clusteredBegin) <= (end - begin));
        for (uint32_t tid : independent_group_elements(acc, clusteredEnd - clusteredBegin)) {
          // tid is the index of the constituent within this BX, which is also the order of the constituent across the whole list of clustered particles in this BX
          uint32_t i = tid + begin;
          uint32_t icand = idx[i] + begin;
          map.indexes()[clusteredBegin + tid] = icand;
          assert(clusteredBegin + tid < nclustered);
          assert(map.metadata().size() == int(nclustered));
#ifdef L1TSC_VERBOSE_DEBUG
          uint32_t ijet = key[icand];
          if (block_idx <= 2)
            printf("In BX %u index %u/%u is cand %3u/%3u of index %u assigned to jet %u (local)\n",
                   block_idx + 1,
                   tid,
                   clusteredEnd - clusteredBegin,
                   idx[i],
                   end - begin,
                   icand,
                   ijet);
#endif
        }  // elements
      }  // blocks
    }  // operator()
  };  // class

  class JetIterKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ClusterObjDeviceCollection::View puppi,
                                  OffsetsSoA::ConstView bx_lookup,
                                  float R2,
                                  unsigned int nIters,
                                  ClustersDeviceCollection::View clusters,
                                  uint32_t* tag,
                                  ClusterObjDeviceCollection::View work2,
                                  ClusterObjDeviceCollection::View jets) const {
      // for prefix scan (only on GPU)
      uint32_t* ws = nullptr;
      [[maybe_unused]] constexpr bool single_thread = requires_single_thread_per_block<TAcc>::value;
      if constexpr (!requires_single_thread_per_block_v<TAcc>) {
        ws = alpaka::getDynSharedMem<uint32_t>(acc);
      }
      uint32_t grid_dim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0];
      for (uint32_t block_idx : independent_groups(acc, grid_dim)) {
        // get event range
        uint32_t begin = bx_lookup.offsets()[block_idx];
        uint32_t end = bx_lookup.offsets()[block_idx + 1];

        // skip if empty
        if (end <= begin)
          continue;

        auto& size = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
        size = end - begin;

#ifdef L1TSC_VERBOSE_DEBUG
        if (once_per_block(acc) && (block_idx <= 2))
          printf("In BX %u begin with %u PF candidates: \n", block_idx + 1, end - begin);
#endif
        // running sums (accumulating on multiple threads)
        auto& seed_pt = alpaka::declareSharedVar<float, __COUNTER__>(acc);
        auto& seed_eta = alpaka::declareSharedVar<float, __COUNTER__>(acc);
        auto& seed_phi = alpaka::declareSharedVar<float, __COUNTER__>(acc);
        auto& seed_i = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
        auto& sum_pt = alpaka::declareSharedVar<float, __COUNTER__>(acc);
        auto& sum_eta = alpaka::declareSharedVar<float, __COUNTER__>(acc);
        auto& sum_phi = alpaka::declareSharedVar<float, __COUNTER__>(acc);

        for (unsigned int iter = 0; iter < nIters; ++iter) {
          bool even = (iter % 2 == 0);
          auto pt = even ? puppi.pt() : work2.pt();
          auto eta = even ? puppi.eta() : work2.eta();
          auto phi = even ? puppi.phi() : work2.phi();
          auto cluster = even ? puppi.cluster() : work2.cluster();
          auto pt2 = !even ? puppi.pt() : work2.pt();
          auto eta2 = !even ? puppi.eta() : work2.eta();
          auto phi2 = !even ? puppi.phi() : work2.phi();
          auto cluster2 = !even ? puppi.cluster() : work2.cluster();

          // seeding (identical on all threads)
          if (once_per_block(acc)) {
            float spt = 0, seta = 0, sphi = 0;
            unsigned int iseed = end;
            for (unsigned int j = begin, myend = begin + size; j < myend; ++j) {
#ifdef L1TSC_VERBOSE_DEBUG
              if ((block_idx <= 2) && (iter == 0) && single_thread) {
                printf("  %4u: pt %7.2f eta %+6.3f phi %+6.3f  index %7d\n",
                       j,
                       pt[j],
                       eta[j],
                       puppi.phi()[j],
                       puppi.cluster()[j] - begin);
              }
#endif
              if (pt[j] > spt) {
                spt = pt[j];
                seta = eta[j];
                sphi = phi[j];
                iseed = j;
              }
            }
            seed_pt = spt;
            seed_eta = seta;
            seed_phi = sphi;
            seed_i = iseed;
            sum_pt = 0;
            sum_eta = 0;
            sum_phi = 0;
#ifdef L1TSC_VERBOSE_DEBUG
            if (block_idx <= 2)
              printf(
                  "In BX %u selected %u (pt %7.2f eta %+6.3f phi %+6.3f) as seed for iteration %u (%u/%u particles "
                  "left)\n",
                  block_idx + 1,
                  iseed - begin,
                  spt,
                  seed_eta,
                  seed_phi,
                  iter,
                  size,
                  end - begin);
#endif
          }

          alpaka::syncBlockThreads(acc);

          if (seed_pt == 0)
            break;

          for (uint32_t tid : independent_group_elements(acc, size)) {
            auto ipart = tid + begin;  // global index
            float deta = eta[ipart] - seed_eta;
            float dphi = cms::alpakatools::deltaPhi(acc, phi[ipart], seed_phi);
            float dr2 = deta * deta + dphi * dphi;
            if (dr2 < R2) {
              clusters.is_seed()[cluster[ipart]] = (ipart == seed_i ? 1 : 0);
              clusters.cluster()[cluster[ipart]] = iter;
              tag[ipart] = 0;
              alpaka::atomicAdd(acc, &sum_pt, pt[ipart], alpaka::hierarchy::Threads{});
              alpaka::atomicAdd(acc, &sum_eta, deta * pt[ipart], alpaka::hierarchy::Threads{});
              alpaka::atomicAdd(acc, &sum_phi, dphi * pt[ipart], alpaka::hierarchy::Threads{});
#ifdef L1TSC_VERBOSE_DEBUG
              if (block_idx <= 2 && single_thread)
                printf("  %4u: pt %7.2f eta %+6.3f phi %+6.3f  cluster %7d <<= selected (dr %7.4f)\n",
                       ipart,
                       pt[ipart],
                       eta[ipart],
                       phi[ipart],
                       cluster[ipart] - begin,
                       alpaka::math::sqrt(acc, dr2));
#endif
            } else {
              tag[ipart] = 1;
            }
          }  // elements

          alpaka::syncBlockThreads(acc);

          if (once_per_block(acc)) {
            jets.pt()[begin + iter] = sum_pt;
            jets.eta()[begin + iter] = seed_eta + sum_eta / sum_pt;
            jets.phi()[begin + iter] = cms::alpakatools::reducePhiRange(acc, seed_phi + sum_phi / sum_pt);
#ifdef L1TSC_VERBOSE_DEBUG
            if (block_idx <= 2)
              printf("In BX %u Jet pt %7.2f eta %+6.3f phi %+6.3f, seed %d\n\n",
                     block_idx + 1,
                     jets.pt()[begin + iter],
                     jets.eta()[begin + iter],
                     jets.phi()[begin + iter],
                     seed_i);
#endif
          }

          blockPrefixScan(acc, tag + begin, size, ws);

#ifdef L1TSC_VERBOSE_DEBUG
          if (once_per_block(acc) && (block_idx <= 2) && single_thread)
            printf("Reordering candidates\n");
#endif
          for (uint32_t tid : independent_group_elements(acc, size)) {
            auto ipart = tid + begin;  // global index
            if (tag[ipart] > (tid == 0 ? 0 : tag[ipart - 1])) {
              int dest = begin + tag[ipart] - 1;
              pt2[dest] = pt[ipart];
              eta2[dest] = eta[ipart];
              phi2[dest] = phi[ipart];
              cluster2[dest] = cluster[ipart];
#ifdef L1TSC_VERBOSE_DEBUG
              if (block_idx <= 2 && single_thread)
                printf("  %4u: pt %7.2f eta %+6.3f phi %+6.3f  cluster %7d tag %u --> %8u\n",
                       ipart,
                       pt[ipart],
                       eta[ipart],
                       phi[ipart],
                       cluster[ipart] - begin,
                       tag[ipart],
                       dest);
#endif
            }
          }
          if (once_per_block(acc)) {
            size = tag[begin + size - 1];
#ifdef L1TSC_VERBOSE_DEBUG
            if (block_idx <= 2)
              printf("In BX %u size updated to %u\n\n", block_idx + 1, size);
#endif
          }
          alpaka::syncBlockThreads(acc);

        }  // iter

      }  // block
    }  // operator()
  };  // class

  L1TScPhase2SCJetsKernels::L1TScPhase2SCJetsKernels() {}

  std::tuple<BxLookupDeviceCollection, ClusterObjDeviceCollection, AssociationMapDevice> L1TScPhase2SCJetsKernels::run(
      Queue& queue,
      const PuppiDeviceCollection& src,
      const BxLookupDeviceCollection& bx_lookup,
      float R2,
      ClustersDeviceCollection& clusters) const {
    unsigned int nbx = bx_lookup.const_view<OffsetsSoA>().metadata().size() - 1;

    uint32_t threads_per_block = 256;
    uint32_t blocks_per_grid = nbx;
    auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

    // index buffer for reordering
    unsigned int npart = src.const_view().metadata().size();
    auto h_key_device = alpaka::allocAsyncBuf<uint16_t, Idx>(queue, Vec1D(npart));
    auto h_idx_device = alpaka::allocAsyncBuf<uint16_t, Idx>(queue, Vec1D(npart));
    alpaka::memset(queue, h_idx_device, 0x00);

    // buffer for the number of jets per BX
    auto nJetsTotalDevice = CounterDevice(queue);
    nJetsTotalDevice.zeroInitialise(queue);

    auto work = ClusterObjDeviceCollection(npart, queue);
    auto jetsNonZS = ClusterObjDeviceCollection(npart, queue);
    clusters.zeroInitialise(queue);
    jetsNonZS.zeroInitialise(queue);

    auto jetBxLookup = BxLookupDeviceCollection({{int(nbx), int(nbx + 1)}}, queue);
    jetBxLookup.zeroInitialise(queue);

    alpaka::exec<Acc1D>(queue,
                        grid,
                        JetKernel{},
                        src.const_view(),
                        bx_lookup.const_view<OffsetsSoA>(),
                        bx_lookup.const_view<BxIndexSoA>(),
                        std::sqrt(R2),
                        R2,
                        h_key_device.data(),
                        h_idx_device.data(),
                        work.view(),
                        clusters.view(),
                        jetsNonZS.view(),
                        jetBxLookup.view<OffsetsSoA>(),
                        jetBxLookup.view<BxIndexSoA>(),
                        nJetsTotalDevice.data());

    auto nJetsTotalHost = CounterHost(queue);
    alpaka::memcpy(queue, nJetsTotalHost.buffer(), nJetsTotalDevice.buffer());
    alpaka::wait(queue);

    auto njets = nJetsTotalHost.value();

    auto jets = ClusterObjDeviceCollection(njets, queue);

    // prefix sum to build jet offsets
    auto pc = alpaka::allocAsyncBuf<int32_t, Idx>(queue, Vec1D{1});
    alpaka::memset(queue, pc, 0x00);
    uint32_t jet_threads_per_block = 1024;
    uint32_t jet_blocks_per_grid =
        cms::alpakatools::divide_up_by(jetBxLookup.view<OffsetsSoA>().metadata().size(), jet_threads_per_block);
    auto jet_grid = cms::alpakatools::make_workdiv<Acc1D>(jet_blocks_per_grid, jet_threads_per_block);
    alpaka::exec<Acc1D>(queue,
                        jet_grid,
                        cms::alpakatools::multiBlockPrefixScan<uint32_t>{},
                        jetBxLookup.view<OffsetsSoA>().offsets().data(),
                        jetBxLookup.view<OffsetsSoA>().offsets().data(),
                        jetBxLookup.view<OffsetsSoA>().metadata().size(),
                        jet_blocks_per_grid,
                        pc.data(),
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));

    // copy jets to final collection; compute running sum of total clustered particles
    auto nClusteredDevice = CounterDevice(queue);
    nClusteredDevice.zeroInitialise(queue);
    alpaka::exec<Acc1D>(queue,
                        grid,
                        JetZSKernel{},
                        bx_lookup.const_view<OffsetsSoA>(),
                        jetsNonZS.const_view(),
                        jetBxLookup.const_view<OffsetsSoA>(),
                        jets.view(),
                        nClusteredDevice.data());

    auto nClusteredHost = CounterHost(queue);
    alpaka::memcpy(queue, nClusteredHost.buffer(), nClusteredDevice.buffer());
    alpaka::wait(queue);
    unsigned int nclustered = nClusteredHost.value();

    // finally, build the association map
    auto map = AssociationMapDevice({{int(nclustered), int(njets + 1)}}, queue);
    map.zeroInitialise(queue);

    // let's first do the stupid thing and just copy the offsets
    uint32_t constit_threads_per_block = 1024;
    uint32_t constit_blocks_per_grid = cms::alpakatools::divide_up_by(njets, constit_threads_per_block);
    auto constit_grid = cms::alpakatools::make_workdiv<Acc1D>(constit_blocks_per_grid, constit_threads_per_block);
    uint32_t constit_blocks_per_grid1 = cms::alpakatools::divide_up_by(njets + 1, constit_threads_per_block);
    auto constit_grid1 = cms::alpakatools::make_workdiv<Acc1D>(constit_blocks_per_grid1, constit_threads_per_block);
    alpaka::exec<Acc1D>(
        queue,
        constit_grid,
        [] ALPAKA_FN_ACC(Acc1D const& acc, ClusterObjDeviceCollection::ConstView jets, OffsetsSoA::View offsets) {
          if (cms::alpakatools::once_per_grid(acc))
            offsets.offsets()[0] = 0;
          for (int32_t idx : cms::alpakatools::uniform_elements(acc, offsets.metadata().size() - 1)) {
            offsets.offsets()[idx + 1] = jets.numberOfDaughters()[idx];
          }
        },
        jets.const_view(),
        map.view<OffsetsSoA>());

    // prefix sum for jet constituent offsets
    alpaka::memset(queue, pc, 0x00);
    alpaka::exec<Acc1D>(queue,
                        constit_grid1,
                        cms::alpakatools::multiBlockPrefixScan<uint32_t>{},
                        map.view<OffsetsSoA>().offsets().data(),
                        map.view<OffsetsSoA>().offsets().data(),
                        njets + 1,
                        constit_blocks_per_grid1,
                        pc.data(),
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));

    // indices of the clustered particles within each jet
    alpaka::exec<Acc1D>(queue,
                        grid,
                        JetToAssociationMapKernel{},
                        bx_lookup.const_view<OffsetsSoA>(),
                        jetBxLookup.const_view<OffsetsSoA>(),
                        map.const_view<OffsetsSoA>(),
                        clusters.const_view(),
                        njets,
                        nclustered,
                        h_key_device.data(),
                        h_idx_device.data(),
                        map.view<IndexSoA>());
    return std::make_tuple(std::move(jetBxLookup), std::move(jets), std::move(map));
  }

  std::tuple<BxLookupDeviceCollection, ClusterObjDeviceCollection, AssociationMapDevice> L1TScPhase2SCJetsKernels::run(
      Queue& queue,
      const PuppiDeviceCollection& src,
      const BxLookupDeviceCollection& bx_lookup,
      float R2,
      unsigned int nJets,
      ClustersDeviceCollection& clusters) const {
    // one grid per particles in blocks per BX
    uint32_t threads_per_block = 256;
    uint32_t blocks_per_grid = bx_lookup.const_view<OffsetsSoA>().metadata().size() - 1;
    auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

    // space for output and for reordering inputs
    auto work = ClusterObjDeviceCollection(src.const_view().metadata().size(), queue);
    auto work2 = ClusterObjDeviceCollection(src.const_view().metadata().size(), queue);

    // a buffer space for counting items
    auto partExtent = Vec1D(src.const_view().metadata().size());
    auto h_tag_device = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, partExtent);
    alpaka::memset(queue, h_tag_device, 0x00);

    // one flat grid per particle, with arbitrary block size
    uint32_t threads_per_flatblock = 1024;
    uint32_t blocks_per_flatgrid =
        cms::alpakatools::divide_up_by(src.const_view().metadata().size(), threads_per_flatblock);
    auto flatgrid = cms::alpakatools::make_workdiv<Acc1D>(blocks_per_flatgrid, threads_per_flatblock);

    auto jets = ClusterObjDeviceCollection(src.const_view().metadata().size(), queue);
    jets.zeroInitialise(queue);

    alpaka::exec<Acc1D>(
        queue,
        flatgrid,
        [] ALPAKA_FN_ACC(Acc1D const& acc,
                         PuppiDeviceCollection::ConstView puppi,
                         ClusterObjDeviceCollection::View work,
                         ClustersDeviceCollection::View clusters) {
          for (int32_t idx : cms::alpakatools::uniform_elements(acc, clusters.metadata().size())) {
            work.pt()[idx] = puppi.pt()[idx];
            work.eta()[idx] = puppi.eta()[idx];
            work.phi()[idx] = puppi.phi()[idx];
            work.cluster()[idx] = idx;
            clusters.cluster()[idx] = -1;
          }
        },
        src.const_view(),
        work.view(),
        clusters.view());

    alpaka::exec<Acc1D>(queue,
                        grid,
                        JetIterKernel{},
                        work.view(),
                        bx_lookup.const_view<OffsetsSoA>(),
                        R2,
                        nJets,
                        clusters.view(),
                        h_tag_device.data(),
                        work2.view(),
                        jets.view());

    auto map = AssociationMapDevice({{1, 1}}, queue);
    auto jetBxLookup = BxLookupDeviceCollection({{0, 1}}, queue);
    return std::make_tuple(std::move(jetBxLookup), std::move(jets), std::move(map));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels