#include "FWCore/Framework/interface/MakerMacros.h"

#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <cmath>
#include <numeric>

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
#include "DataFormats/L1ScoutingSoA/interface/BxLookupHost.h"
#include "DataFormats/L1ScoutingSoA/interface/SoftTauHostTensor.h"
#include "DataFormats/L1ScoutingSoA/interface/ClustersHostCollection.h"

class SoftTauOutputTensorToOrbitFlatTable : public edm::global::EDProducer<> {
public:
  // constructor and destructor
  explicit SoftTauOutputTensorToOrbitFlatTable(const edm::ParameterSet&);
  ~SoftTauOutputTensorToOrbitFlatTable() override {};

  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // the tokens to access the data
  edm::EDGetTokenT<l1sc::BxLookupHost> srcBxClustersMap_;
  edm::EDGetTokenT<l1sc::SoftTauOutputHostTensor> srcOutputs_;

  std::string name_, doc_;
};
// -----------------------------------------------------------------------------

// -------------------------------- constructor  -------------------------------

SoftTauOutputTensorToOrbitFlatTable::SoftTauOutputTensorToOrbitFlatTable(const edm::ParameterSet& iConfig) :
      srcBxClustersMap_(consumes<l1sc::BxLookupHost>(iConfig.getParameter<edm::InputTag>("srcBxClustersMap"))),
      srcOutputs_(consumes<l1sc::SoftTauOutputHostTensor>(iConfig.getParameter<edm::InputTag>("srcOutputs"))),
      name_(iConfig.getParameter<std::string>("name")),
      doc_(iConfig.getParameter<std::string>("doc")) {
  produces<l1ScoutingRun3::OrbitFlatTable>("output");
}
// -----------------------------------------------------------------------------

// ----------------------- method called for each orbit  -----------------------
void SoftTauOutputTensorToOrbitFlatTable::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  edm::Handle<l1sc::BxLookupHost> srcBxClustersMap;
  iEvent.getByToken(srcBxClustersMap_, srcBxClustersMap);
  edm::Handle<l1sc::SoftTauOutputHostTensor> srcOutputs;
  iEvent.getByToken(srcOutputs_, srcOutputs);

  // get output tensor view
  const auto outputs_view = srcOutputs->const_view();

  // num bx and offsets
  const auto nbx = srcBxClustersMap->const_view().bx().metadata().size();
  const auto *bxc_offsets = srcBxClustersMap->const_view().offset().offset().data();
  std::vector<unsigned int> bxOffsets;
  bxOffsets.push_back(0);
  bxOffsets.insert(bxOffsets.end(), bxc_offsets, bxc_offsets + nbx + 1);

  // get num output tensors
  const auto num_outputs = outputs_view.metadata().size();

  // soft tau output tensor
  const auto *cls = outputs_view.cls().data();
  const auto *vz = outputs_view.vz().data();
  const auto *pt = outputs_view.pt().data();
  const auto *charge = outputs_view.charge().data();

  // table with output logits and outputTensorIndex
  std::vector<float> cls_vec{cls, cls + num_outputs};
  std::vector<float> vz_vec{vz, vz + num_outputs};
  std::vector<float> pt_vec{pt, pt + num_outputs};
  std::vector<float> charge_vec{charge, charge + num_outputs};
  std::vector<int> tensorIndex(num_outputs);
  iota(tensorIndex.begin(), tensorIndex.end(), 0);

  auto out_table = std::make_unique<l1ScoutingRun3::OrbitFlatTable>(bxOffsets, name_); // singleton and extension set to false by default
  out_table->setDoc(doc_);
  out_table->addColumn<float>("cls", cls_vec, "Classification score");
  out_table->addColumn<float>("vz", vz_vec, "vZ regression");
  out_table->addColumn<float>("pt", pt_vec, "pT regression");
  out_table->addColumn<float>("charge", charge_vec, "Charge score");
  out_table->addColumn<int>("outputTensorIndex", tensorIndex, "Index of the output tensor (corresponds to the cluster index)");
  
  iEvent.put(std::move(out_table), "output");
}

void SoftTauOutputTensorToOrbitFlatTable::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcBxClustersMap");
  desc.add<edm::InputTag>("srcOutputs");
  desc.add<std::string>("name");
  desc.add<std::string>("doc", "");
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(SoftTauOutputTensorToOrbitFlatTable);
