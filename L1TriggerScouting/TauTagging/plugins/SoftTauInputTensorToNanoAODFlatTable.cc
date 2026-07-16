#include <Eigen/Core>
#include <Eigen/Dense>

#include "FWCore/Framework/interface/MakerMacros.h"

#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <cmath>
#include <array>

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
#include "DataFormats/L1ScoutingSoA/interface/SoftTauHostTensor.h"

class SoftTauInputTensorToNanoAODFlatTable : public edm::global::EDProducer<> {
public:
  // constructor and destructor
  explicit SoftTauInputTensorToNanoAODFlatTable(const edm::ParameterSet&);
  ~SoftTauInputTensorToNanoAODFlatTable() override {};

  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // the tokens to access the data
  edm::EDGetTokenT<l1sc::SoftTauInputHostTensor> srcInputs_;

  std::string name_, doc_;
};
// -----------------------------------------------------------------------------

// -------------------------------- constructor  -------------------------------

SoftTauInputTensorToNanoAODFlatTable::SoftTauInputTensorToNanoAODFlatTable(const edm::ParameterSet& iConfig) :
      srcInputs_(consumes<l1sc::SoftTauInputHostTensor>(iConfig.getParameter<edm::InputTag>("srcInputs"))),
      name_(iConfig.getParameter<std::string>("name")),
      doc_(iConfig.getParameter<std::string>("doc")) {
  produces<nanoaod::FlatTable>("output");
}
// -----------------------------------------------------------------------------

// ----------------------- method called for each orbit  -----------------------
void SoftTauInputTensorToNanoAODFlatTable::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  edm::Handle<l1sc::SoftTauInputHostTensor> srcInputs;
  iEvent.getByToken(srcInputs_, srcInputs);

  // get input tensors view
  const auto tensors_view = srcInputs->const_view();

  // get num input tensors
  const auto num_tensors = tensors_view.metadata().size();

  // intermediate containers for soft tau input tensor
  std::array<std::vector<float>, 10> featureColumns; // 10 feature columns 
  for (auto& col : featureColumns) {
    col.reserve(16 * num_tensors);
  }
  std::vector<int> constituentIndex; // index of te constituent within the tensor
  constituentIndex.reserve(16 * num_tensors);
  std::vector<int> tensorIndex; // index of the input tensor (corresponds to cluster index)
  tensorIndex.reserve(16 * num_tensors);

  // copy features from tensor to columns and update offsets
  for (auto tensor = 0; tensor < num_tensors; ++tensor) {
    auto const& tensor_current = tensors_view[tensor];

    for (auto row = 0; row < 16; ++row) {
      // 1 means valid row
      if (tensor_current.pad_mask()(row) != 1.f) {
        continue;
      }

      for (int col = 0; col < 10; ++col) {
        featureColumns[col].push_back(tensor_current.features()(row, col));
      }

      constituentIndex.push_back(static_cast<int>(row));
      tensorIndex.push_back(static_cast<int>(tensor));
    }
  }

  auto out_table = std::make_unique<nanoaod::FlatTable>(constituentIndex.size(), name_, /* singleton */ false, /* extension */ false);
  out_table->setDoc(doc_);
  out_table->addColumn<float>("pt", featureColumns[0], "Constituent pT");
  out_table->addColumn<float>("deta", featureColumns[1], "Constituent Delta Eta wrt axis");
  out_table->addColumn<float>("dphi", featureColumns[2], "Constituent Delta Phi wrt axis");
  out_table->addColumn<float>("z0", featureColumns[3], "Constituent z0");
  out_table->addColumn<float>("charge", featureColumns[4], "Constituent Charge");
  out_table->addColumn<float>("oh0", featureColumns[5], "One-hot encoding 0");
  out_table->addColumn<float>("oh1", featureColumns[6], "One-hot encoding 1");
  out_table->addColumn<float>("oh2", featureColumns[7], "One-hot encoding 2");
  out_table->addColumn<float>("oh3", featureColumns[8], "One-hot encoding 3");
  out_table->addColumn<float>("oh4", featureColumns[9], "One-hot encoding 4");
  out_table->addColumn<int>("inputConstituentIndex", constituentIndex, "Index of the constituent within the input tensor");
  out_table->addColumn<int>("inputTensorIndex", tensorIndex, "Index of the input tensor (corresponds to cluster index)");

  iEvent.put(std::move(out_table), "output");
}

void SoftTauInputTensorToNanoAODFlatTable::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcInputs");
  desc.add<std::string>("name");
  desc.add<std::string>("doc", "");
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(SoftTauInputTensorToNanoAODFlatTable);
