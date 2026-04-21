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
#include "DataFormats/L1ScoutingSoA/interface/PFCandidateHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/BxLookupHost.h"
#include "DataFormats/L1ScoutingSoA/interface/AssociationMapHost.h"
#include "DataFormats/L1ScoutingSoA/interface/ClustersHostCollection.h"

#include "DataFormats/Math/interface/LorentzVector.h"

using LorentzVector = math::PtEtaPhiMLorentzVectorF;

class ScPhase2ClusterMapsToOrbitFlatTable : public edm::global::EDProducer<> {
public:
  // constructor and destructor
  explicit ScPhase2ClusterMapsToOrbitFlatTable(const edm::ParameterSet&);
  ~ScPhase2ClusterMapsToOrbitFlatTable() override {};

  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // the tokens to access the data
  edm::EDGetTokenT<l1sc::PFCandidateHostCollection> srcCandidates_;
  edm::EDGetTokenT<l1sc::BxLookupHost> srcBxClustersMap_;
  edm::EDGetTokenT<l1sc::AssociationMapHost> srcClustersCandsMap_;
  edm::EDGetTokenT<l1sc::ClustersHostCollection> srcClusterIndexes_;

  std::string name_clusters_, doc_;
};
// -----------------------------------------------------------------------------

// -------------------------------- constructor  -------------------------------

ScPhase2ClusterMapsToOrbitFlatTable::ScPhase2ClusterMapsToOrbitFlatTable(const edm::ParameterSet& iConfig) :
      srcCandidates_(consumes<l1sc::PFCandidateHostCollection>(iConfig.getParameter<edm::InputTag>("srcCandidates"))),
      srcBxClustersMap_(consumes<l1sc::BxLookupHost>(iConfig.getParameter<edm::InputTag>("srcBxClustersMap"))),
      srcClustersCandsMap_(consumes<l1sc::AssociationMapHost>(iConfig.getParameter<edm::InputTag>("srcClustersCandsMap"))),
      srcClusterIndexes_(consumes<l1sc::ClustersHostCollection>(iConfig.getParameter<edm::InputTag>("srcClusterIndexes"))),
      name_clusters_(iConfig.getParameter<std::string>("nameClusters")),
      doc_(iConfig.getParameter<std::string>("doc")) {
  produces<l1ScoutingRun3::OrbitFlatTable>("clusters");
}
// -----------------------------------------------------------------------------

// ----------------------- method called for each orbit  -----------------------
void ScPhase2ClusterMapsToOrbitFlatTable::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  edm::Handle<l1sc::PFCandidateHostCollection> srcCandidates;
  iEvent.getByToken(srcCandidates_, srcCandidates);
  edm::Handle<l1sc::BxLookupHost> srcBxClustersMap;
  iEvent.getByToken(srcBxClustersMap_, srcBxClustersMap);
  edm::Handle<l1sc::AssociationMapHost> srcClustersCandsMap;
  iEvent.getByToken(srcClustersCandsMap_, srcClustersCandsMap);
  edm::Handle<l1sc::ClustersHostCollection> srcClusterIndexes;
  iEvent.getByToken(srcClusterIndexes_, srcClusterIndexes);

  const auto npoints = static_cast<uint32_t>(srcCandidates->const_view().metadata().size());
  const auto nbx = srcBxClustersMap->const_view().bx().metadata().size();
  const auto nclusters = srcClusterIndexes->const_view().metadata().size();

  // bx->clusters map
  const auto *bxc_offsets = srcBxClustersMap->const_view().offset().offset().data();
  const auto *bxc_indexes = srcBxClustersMap->const_view().bx().bx().data();
  
  // clusters->candidates map
  const auto *cc_offsets = srcClustersCandsMap->const_view().offset().offset().data();
  const auto *cc_indexes = srcClustersCandsMap->const_view().index().index().data();

  // cluster indexes
  const auto *cluster_indexes = srcClusterIndexes->const_view().cluster().data();

  // candidates features
  const auto *pt = srcCandidates->const_view().pt().data();
  const auto *eta = srcCandidates->const_view().eta().data();
  const auto *phi = srcCandidates->const_view().phi().data();
  std::vector<float> pf_pt{pt, pt + npoints};
  std::vector<float> pf_eta{eta, eta + npoints};
  std::vector<float> pf_phi{phi, phi + npoints};

  // clusters features
  std::vector<float> pt_clu, eta_clu, phi_clu, mass_clu;

  l1ScoutingRun3::BxOffsetsFillter bxOffsetsFiller;
  bxOffsetsFiller.start();
  for (auto ii = 0; ii < nbx; ++ii) {
    auto bx_idx = bxc_indexes[ii];
    auto bx_start = bxc_offsets[bx_idx];
    auto bx_end = bxc_offsets[bx_idx + 1];

    // loop though the cluster indexes of the current bx
    for (auto jj = bx_start; jj < bx_end; ++jj) {
      auto clu_idx = cluster_indexes[jj]; // fix a cluster index
      auto clu_start = cc_offsets[clu_idx];
      auto clu_end = cc_offsets[clu_idx + 1];

      LorentzVector sum(0.0f, 0.0f, 0.0f, 0.0f);
      for (auto kk = clu_start; kk < clu_end; ++kk) {
        auto cand_idx = cc_indexes[kk]; // fix a candidate index
        LorentzVector cand(pf_pt[cand_idx], pf_eta[cand_idx], pf_phi[cand_idx], 0.13957f); // assume Pion mass
        sum += cand;
      }

      pt_clu.push_back(sum.Pt());
      eta_clu.push_back(sum.Eta());
      phi_clu.push_back(sum.Phi());
      mass_clu.push_back(sum.M());
    }

    auto bx = bx_idx + 1;
    auto bx_size = bx_end - bx_start;
    bxOffsetsFiller.addBx(bx, bx_size);
  }
  auto bxOffsets = bxOffsetsFiller.done();

  // table with cluster features
  auto clu_table = std::make_unique<l1ScoutingRun3::OrbitFlatTable>(bxOffsets, name_clusters_);
  clu_table->setDoc(doc_);
  clu_table->addColumn<float>("pt", pt_clu, "CLUECluster pt");
  clu_table->addColumn<float>("eta", eta_clu, "CLUECluster eta");
  clu_table->addColumn<float>("phi", phi_clu, "CLUECluster phi");
  clu_table->addColumn<float>("mass", mass_clu, "CLUECluster mass");
  iEvent.put(std::move(clu_table), "clusters");
}

void ScPhase2ClusterMapsToOrbitFlatTable::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcCandidates");
  desc.add<edm::InputTag>("srcBxClustersMap");
  desc.add<edm::InputTag>("srcClustersCandsMap");
  desc.add<edm::InputTag>("srcClusterIndexes");
  desc.add<std::string>("nameClusters");
  desc.add<std::string>("doc");
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ScPhase2ClusterMapsToOrbitFlatTable);
