#include "DataFormats/L1ScoutingSoA/interface/alpaka/AssociationMapDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/BxLookupDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/ClustersDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "L1TriggerScouting/Phase2/interface/L1TScPhase2Common.h"
#include "L1TriggerScouting/Phase2/plugins/alpaka/L1TScPhase2CLUEJetsKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  class L1TScPhase2CLUEJets : public stream::EDProducer<> {
  public:
    explicit L1TScPhase2CLUEJets(const edm::ParameterSet &params)
        : EDProducer<>(params),
          pf_candidates_token_{consumes(params.getParameter<edm::InputTag>("candidates"))},
          bx_sizes_token_{consumes(params.getParameter<edm::InputTag>("bxSizes"))},
          bx_clusters_map_token_{produces("bxClustersMap")},
          clusters_cands_map_token_{produces("clustersCandsMap")},
          cluster_objects_token_{produces("jets")},
          clusters_token_{produces("clusters")},
          clustering_(static_cast<float>(params.getParameter<double>("density_radius")),
                      static_cast<float>(params.getParameter<double>("min_density")),
                      static_cast<float>(params.getParameter<double>("outlier_distance")),
                      params.getParameter<bool>("wrapCoords")) {}

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      // get collection from device memory space (implicit copy done by framework)
      const auto &pf = event.get(pf_candidates_token_);
      const auto &bx_sizes = event.get(bx_sizes_token_);
      const auto n_points = pf.const_view().metadata().size();

      // allocate buffer for the index of the cluster for each pf candidate
      auto points_clusters = ClustersDeviceCollection(event.queue(), n_points);

      // run CLUEstering algo
      auto [bx_clusters_map, cluster_objects, clusters_cands_map] = clustering_.run(event.queue(), pf, bx_sizes, points_clusters);

      // emplace clustering products into the orbit
      event.emplace(bx_clusters_map_token_, std::move(bx_clusters_map)); // bx -> clusters map
      event.emplace(clusters_cands_map_token_, std::move(clusters_cands_map)); // clusters -> candidates map
      event.emplace(cluster_objects_token_, std::move(cluster_objects)); // collection of cluster features (currently only cluster index is filled)
      event.emplace(clusters_token_, std::move(points_clusters)); // list of the cluster index each candidate belongs to (-1 if it is outlier)
    }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("candidates");
      desc.add<edm::InputTag>("bxSizes");
      desc.add<double>("density_radius");
      desc.add<double>("min_density");
      desc.add<double>("outlier_distance");
      desc.add<bool>("wrapCoords");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    // get device pf data
    const device::EDGetToken<PFCandidateDeviceCollection> pf_candidates_token_;
    const device::EDGetToken<BxLookupDevice> bx_sizes_token_;
    // put device clustering data
    const device::EDPutToken<BxLookupDevice> bx_clusters_map_token_;
    const device::EDPutToken<AssociationMapDevice> clusters_cands_map_token_;
    const device::EDPutToken<ClusterObjDeviceCollection> cluster_objects_token_;
    const device::EDPutToken<ClustersDeviceCollection> clusters_token_;
    // algorithm
    const kernels::CLUEsteringAlgo clustering_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

DEFINE_FWK_ALPAKA_MODULE(l1sc::L1TScPhase2CLUEJets);