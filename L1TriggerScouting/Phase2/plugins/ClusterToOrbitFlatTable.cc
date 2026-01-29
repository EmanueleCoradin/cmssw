#include "FWCore/Framework/interface/MakerMacros.h"

#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <cmath>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "L1TriggerScouting/Utilities/interface/BxOffsetsFiller.h"

#include "DataFormats/NanoAOD/interface/OrbitFlatTable.h"
#include "DataFormats/L1ScoutingSoA/interface/AssociationMapHost.h"
#include "DataFormats/L1ScoutingSoA/interface/BxLookupHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PFCandidateHostCollection.h"


class ClusterToOrbitFlatTable : public edm::global::EDProducer<> {
public:
  // constructor and destructor
  explicit ClusterToOrbitFlatTable(const edm::ParameterSet&);
  ~ClusterToOrbitFlatTable() override {};

  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // the tokens to access the data
  edm::EDGetTokenT<l1sc::BxLookupHostCollection> srcBxClustersMap_;
  edm::EDGetTokenT<l1sc::AssociationMapHost> srcClustersCandsMap_;
  edm::EDGetTokenT<l1sc::PFCandidateHostCollection> srcCandidates_;

  std::string name_, doc_;
};
// -----------------------------------------------------------------------------

// -------------------------------- constructor  -------------------------------

ClusterToOrbitFlatTable::ClusterToOrbitFlatTable(const edm::ParameterSet& iConfig) :
      srcBxClustersMap_(consumes<l1sc::BxLookupHostCollection>(iConfig.getParameter<edm::InputTag>("srcBxClustersMap"))),
      srcClustersCandsMap_(consumes<l1sc::AssociationMapHost>(iConfig.getParameter<edm::InputTag>("srcClustersCandsMap"))),
      srcCandidates_(consumes<l1sc::PFCandidateHostCollection>(iConfig.getParameter<edm::InputTag>("srcCandidates"))),
      name_(iConfig.getParameter<std::string>("name")),
      doc_(iConfig.getParameter<std::string>("doc")) {
  produces<l1ScoutingRun3::OrbitFlatTable>();
}
// -----------------------------------------------------------------------------

// ----------------------- method called for each orbit  -----------------------
void ClusterToOrbitFlatTable::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  edm::Handle<l1sc::BxLookupHostCollection> srcBxClustersMap;
  iEvent.getByToken(srcBxClustersMap_, srcBxClustersMap);
  edm::Handle<l1sc::AssociationMapHost> srcClustersCandsMap;
  iEvent.getByToken(srcClustersCandsMap_, srcClustersCandsMap);
  edm::Handle<l1sc::PFCandidateHostCollection> srcCandidates;
  iEvent.getByToken(srcCandidates_, srcCandidates);

  const unsigned int nbx = srcBxClustersMap->const_view<l1sc::OffsetsSoA>().metadata().size() - 1;

  const auto *bx_offsets = srcBxClustersMap->const_view<l1sc::OffsetsSoA>().offsets().data();
  const auto *cluster_idx = srcBxClustersMap->const_view<l1sc::BxIndexSoA>().bx().data();
  const auto *cluster_off = srcClustersCandsMap->const_view<l1sc::OffsetsSoA>().offsets().data();
  const auto *candidate_idx = srcClustersCandsMap->const_view<l1sc::IndexSoA>().indexes().data();

  const auto *pt = srcCandidates->const_view().pt().data();
  const auto *eta = srcCandidates->const_view().eta().data();
  const auto *phi = srcCandidates->const_view().phi().data();

  std::vector<float> pts, etas, phis;
  std::vector<uint32_t> clusters;

  // bxOffsets is needed in order to keep track of the offsets
  // for every bx with respect to the candidates that are within it.
  // But none of the offsets inside the CandsClusterBxHostCollection 
  // contain such information
  l1ScoutingRun3::BxOffsetsFillter bxOffsetsFiller;
  bxOffsetsFiller.start();

  // loop over the bxs in the orbit
  for (unsigned int bx_idx = 1; bx_idx <= nbx; ++bx_idx) { // pay attention that bx_idx has to start at 1
    auto bx_start = bx_offsets[bx_idx - 1];
    auto bx_end = bx_offsets[bx_idx];
    auto bx_size = 0;
    
    // loop over the clusters indexes of the clusters in the current bx
    for (unsigned int ii = bx_start; ii < bx_end; ++ii) {
      auto cl_idx = cluster_idx[ii];
      auto cluster_start = cluster_off[cl_idx];
      auto cluster_end = cluster_off[cl_idx + 1];
      bx_size += cluster_end - cluster_start;

      // loop over the candidates indexes of the candidats in the current cluster
      for (unsigned int jj = cluster_start; jj < cluster_end; ++jj) {
        auto cand_idx = candidate_idx[jj];
        pts.push_back(pt[cand_idx]);
        etas.push_back(eta[cand_idx]);
        phis.push_back(phi[cand_idx]);
        clusters.push_back(cl_idx);
      }
    }

    bxOffsetsFiller.addBx(bx_idx, bx_size);
  }

  auto bxOffsets = bxOffsetsFiller.done();
  auto out = std::make_unique<l1ScoutingRun3::OrbitFlatTable>(bxOffsets, name_);
  out->setDoc(doc_);
  out->addColumn<float>("pt", pts, "candidate pt (GeV)");
  out->addColumn<float>("eta", etas, "candidate eta (GeV)");
  out->addColumn<float>("phi", phis, "candidate phi (GeV)");
  out->addColumn<uint32_t>("cluster", clusters, "cluster index");
  iEvent.put(std::move(out));
}

void ClusterToOrbitFlatTable::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcBxClustersMap");
  desc.add<edm::InputTag>("srcClustersCandsMap");
  desc.add<edm::InputTag>("srcCandidates");
  desc.add<std::string>("name");
  desc.add<std::string>("doc");
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ClusterToOrbitFlatTable);
