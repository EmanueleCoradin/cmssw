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
#include "DataFormats/L1ScoutingSoA/interface/BxLookupHost.h"
#include "DataFormats/L1ScoutingSoA/interface/ClustersHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/AssociationMapHost.h"

class ClusterSoAToNanoAODFlatTable : public edm::global::EDProducer<> {
public:
  // constructor and destructor
  explicit ClusterSoAToNanoAODFlatTable(const edm::ParameterSet&);
  ~ClusterSoAToNanoAODFlatTable() override {};

  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // the tokens to access the data
  edm::EDGetTokenT<l1sc::ClustersHostCollection> srcClusters_;

  std::string name_, clustering_name_, doc_;
};
// -----------------------------------------------------------------------------

// -------------------------------- constructor  -------------------------------

ClusterSoAToNanoAODFlatTable::ClusterSoAToNanoAODFlatTable(const edm::ParameterSet& iConfig)
    : srcClusters_(consumes<l1sc::ClustersHostCollection>(iConfig.getParameter<edm::InputTag>("srcClusters"))),
      name_(iConfig.getParameter<std::string>("name")),
      clustering_name_(iConfig.getParameter<std::string>("clustering_name")),
      doc_(iConfig.getParameter<std::string>("doc")) {
  produces<nanoaod::FlatTable>();
}
// -----------------------------------------------------------------------------

// ----------------------- method called for each orbit  -----------------------
void ClusterSoAToNanoAODFlatTable::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  edm::Handle<l1sc::ClustersHostCollection> srcClusters;
  iEvent.getByToken(srcClusters_, srcClusters);

  const auto* cluster = srcClusters->const_view().cluster().data();
  const auto* seed = srcClusters->const_view().is_seed().data();
  const unsigned int nclusters = srcClusters->const_view().metadata().size();
  std::vector<int32_t> clusters{cluster, cluster + nclusters};
  std::vector<int32_t> is_seed{seed, seed + nclusters};

  auto out = std::make_unique<nanoaod::FlatTable>(clusters.size(), name_, false, true);
  out->setDoc(doc_);
  out->addColumn<int32_t>("clusterIndex" + clustering_name_, clusters, "associated cluster index for " + clustering_name_);
  out->addColumn<int32_t>("isSeed" + clustering_name_, is_seed, "cluster used as seed for " + clustering_name_);
  iEvent.put(std::move(out));
}

void ClusterSoAToNanoAODFlatTable::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcClusters");
  desc.add<std::string>("name");
  desc.add<std::string>("clustering_name");
  desc.add<std::string>("doc");
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ClusterSoAToNanoAODFlatTable);