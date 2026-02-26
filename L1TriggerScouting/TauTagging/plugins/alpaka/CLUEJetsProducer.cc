#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/ClustersDeviceCollection.h"
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
#include "L1TriggerScouting/TauTagging/plugins/alpaka/CLUEJetsProducerAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {
    class CLUEJetsProducer : public stream::EDProducer<> {
    public:
        explicit CLUEJetsProducer(const edm::ParameterSet &params) :
            EDProducer<>(params),
            pf_candidates_token_{consumes(params.getParameter<edm::InputTag>("candidates"))}, 
            cluster_cands_map_token_{produces("clustersCandsMap")}, 
            clustering_(static_cast<float>(params.getParameter<double>("dc")),
                        static_cast<float>(params.getParameter<double>("rhoc")),
                        static_cast<float>(params.getParameter<double>("dm")),
                        params.getParameter<bool>("wrapCoords")) {}
        
        void produce(device::Event &event, const device::EventSetup &event_setup) override {
            // get pf candidates collection
            const auto &pf = event.get(pf_candidates_token_);
            const auto n_points = pf.const_view().metadata().size();

            // allocate buffer for the index of the cluster for each pf candidate
            auto clusters = ClustersDeviceCollection(n_points, event.queue());

            // run CLUEstering
            auto cluster_cands_map = clustering_.run(event.queue(), pf, clusters);

            // emplace results
            event.emplace(cluster_cands_map_token_, std::move(cluster_cands_map));
        }

        static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
            edm::ParameterSetDescription desc;
            desc.add<edm::InputTag>("candidates");
            desc.add<double>("dc");
            desc.add<double>("rhoc");
            desc.add<double>("dm");
            desc.add<bool>("wrapCoords");
            descriptions.addWithDefaultLabel(desc);
        }

    private:
        // input data token
        const device::EDGetToken<PFCandidateDeviceCollection> pf_candidates_token_;
        // output data token
        const device::EDPutToken<AssociationMapDevice> cluster_cands_map_token_;
        // algorithm
        const kernels::CLUEJetsProducerAlgo clustering_;
    };

} // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

DEFINE_FWK_ALPAKA_MODULE(l1sc::CLUEJetsProducer);