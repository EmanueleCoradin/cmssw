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

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/L1ScoutingSoA/interface/AssociationMapHost.h"
#include "DataFormats/L1ScoutingSoA/interface/PFCandidateHostCollection.h"

#include "DataFormats/Math/interface/LorentzVector.h"

using LorentzVector = math::PtEtaPhiMLorentzVectorF;

class ClusterToFlatTable : public edm::global::EDProducer<> {
public:
    // constructor and destructor
    explicit ClusterToFlatTable(const edm::ParameterSet&);
    ~ClusterToFlatTable() override {};

    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    // the tokens to access the data
    edm::EDGetTokenT<l1sc::PFCandidateHostCollection> srcCandidates_;
    edm::EDGetTokenT<l1sc::AssociationMapHost> srcClustersCandsMap_;

    std::string name_candidates_, name_clusters_, doc_;
};
// -----------------------------------------------------------------------------

// -------------------------------- constructor  -------------------------------

ClusterToFlatTable::ClusterToFlatTable(const edm::ParameterSet& iConfig) :
    srcCandidates_(consumes<l1sc::PFCandidateHostCollection>(iConfig.getParameter<edm::InputTag>("srcCandidates"))),
    srcClustersCandsMap_(consumes<l1sc::AssociationMapHost>(iConfig.getParameter<edm::InputTag>("srcClustersCandsMap"))),
    name_candidates_(iConfig.getParameter<std::string>("nameCandidates")),
    name_clusters_(iConfig.getParameter<std::string>("nameClusters")),
    doc_(iConfig.getParameter<std::string>("doc")) {
        produces<nanoaod::FlatTable>("cands");
        produces<nanoaod::FlatTable>("clusters");
    }
// -----------------------------------------------------------------------------

// ----------------------- method called for each orbit  -----------------------
void ClusterToFlatTable::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
    edm::Handle<l1sc::PFCandidateHostCollection> srcCandidates;
    iEvent.getByToken(srcCandidates_, srcCandidates);
    edm::Handle<l1sc::AssociationMapHost> srcClustersCandsMap;
    iEvent.getByToken(srcClustersCandsMap_, srcClustersCandsMap);

    const auto npoints = static_cast<uint32_t>(srcCandidates->const_view().metadata().size());
    const auto nclusters = srcClustersCandsMap->const_view<l1sc::OffsetsSoA>().metadata().size() - 1;
    
    const auto *cc_offsets = srcClustersCandsMap->const_view<l1sc::OffsetsSoA>().offsets().data();
    const auto *cc_indexes = srcClustersCandsMap->const_view<l1sc::IndexSoA>().indexes().data();

    // assign cluster indexes to each pf candidate (non-clustered points are assigned -1)
    std::vector<int32_t> point_to_cluster(npoints, -1);
    for (auto ii = 0; ii < nclusters; ++ii) {
        auto begin = cc_offsets[ii];
        auto end = cc_offsets[ii + 1];

        for (auto jj = begin; jj < end; ++jj) {
            const auto p = cc_indexes[jj];
            if (p < npoints) {
                point_to_cluster[p] = ii;
            } else {
                throw std::out_of_range("Point index out of range"); 
            }
        }
    }

    const auto *pt = srcCandidates->const_view().pt().data();
    const auto *eta = srcCandidates->const_view().eta().data();
    const auto *phi = srcCandidates->const_view().phi().data();
    const auto *z0 = srcCandidates->const_view().z0().data();
    const auto *dxy = srcCandidates->const_view().dxy().data();
    const auto *puppiw = srcCandidates->const_view().puppiw().data();
    const auto *quality = srcCandidates->const_view().quality().data();
    const auto *pdgid = srcCandidates->const_view().pdgid().data();

    std::vector<float> pf_pt{pt, pt + npoints};
    std::vector<float> pf_eta{eta, eta + npoints};
    std::vector<float> pf_phi{phi, phi + npoints};
    std::vector<float> pf_z0{z0, z0 + npoints};
    std::vector<float> pf_dxy{dxy, dxy + npoints};
    std::vector<float> pf_puppiw{puppiw, puppiw + npoints};
    std::vector<uint8_t> pf_quality{quality, quality + npoints};
    std::vector<int16_t> pf_pdgid{pdgid, pdgid + npoints};

    // table with pf candidates and cluster they belong to
    auto pf_table = std::make_unique<nanoaod::FlatTable>(npoints, name_candidates_, /*singleton=*/false, /*extension=*/false);
    pf_table->setDoc(doc_);
    pf_table->addColumn<float>("pt", pf_pt, "L1PF pt");
    pf_table->addColumn<float>("eta", pf_eta, "L1PF eta");
    pf_table->addColumn<float>("phi", pf_phi, "L1PF phi");
    pf_table->addColumn<float>("z0", pf_z0, "L1PF z0");
    pf_table->addColumn<float>("dxy", pf_dxy, "L1PF dxy");
    pf_table->addColumn<float>("puppiw", pf_puppiw, "L1PF puppiw");
    pf_table->addColumn<uint8_t>("quality", pf_quality, "L1PF quality");
    pf_table->addColumn<int16_t>("pdgid", pf_pdgid, "L1PF pdgid");
    pf_table->addColumn<int32_t>("cluster", point_to_cluster, "Cluster Index");
    iEvent.put(std::move(pf_table), "cands");

    // clusters constituents
    std::vector<float> pt_clu, eta_clu, phi_clu, mass_clu;

    for (auto ii = 0; ii < nclusters; ++ii) {
        auto clu_start = cc_offsets[ii];
        auto clu_end = cc_offsets[ii + 1];

        LorentzVector sum(0.0f, 0.0f, 0.0f, 0.0f);
        for (auto jj = clu_start; jj < clu_end; ++jj) {
            auto cand_idx = cc_indexes[jj]; // fix a candidate index
            LorentzVector cand(pf_pt[cand_idx], pf_eta[cand_idx], pf_phi[cand_idx], 0.13957f); // assume Pion mass
            sum += cand;
        }

        pt_clu.push_back(sum.Pt());
        eta_clu.push_back(sum.Eta());
        phi_clu.push_back(sum.Phi());
        mass_clu.push_back(sum.M());
    }

    // table with cluster
    auto clu_table = std::make_unique<nanoaod::FlatTable>(nclusters, name_clusters_, /*singleton=*/false, /*extension=*/false);
    clu_table->setDoc(doc_);
    clu_table->addColumn<float>("pt", pt_clu, "CLUECluster pt");
    clu_table->addColumn<float>("eta", eta_clu, "CLUECluster eta");
    clu_table->addColumn<float>("phi", phi_clu, "CLUECluster phi");
    clu_table->addColumn<float>("mass", mass_clu, "CLUECluster mass");
    iEvent.put(std::move(clu_table), "clusters");
}

void ClusterToFlatTable::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("srcCandidates");
    desc.add<edm::InputTag>("srcClustersCandsMap");
    desc.add<std::string>("nameCandidates");
    desc.add<std::string>("nameClusters");
    desc.add<std::string>("doc");
    descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ClusterToFlatTable);
