#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <vector>
#include <set>

/*
 * Filter orbits that don't contain at least one selected BX
 * from a BxSelector module and produce a vector of selected BXs
 */
class FinalBxSelector : public edm::global::EDFilter<> {
public:
  explicit FinalBxSelector(const edm::ParameterSet&);
  ~FinalBxSelector() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  bool filter(edm::StreamID id, edm::Event&, const edm::EventSetup&) const override;
  void endJob() override { printReport(); }

  // tokens for BX selected by each analysis
  struct Analysis {
    std::string name;
    edm::EDGetTokenT<std::vector<unsigned>> token;
    mutable std::atomic<unsigned long long> nSelected;
    Analysis(const std::string& n, const edm::EDGetTokenT<std::vector<unsigned>>& t)
        : name(n), token(t), nSelected(0) {};
    // note that copy and assignment don't copy nSelected atomically!
    Analysis(const Analysis& other) : name(other.name), token(other.token), nSelected(other.nSelected.load()) {};
    Analysis& operator=(const Analysis& other) {
      name = other.name;
      token = other.token;
      nSelected = other.nSelected.load();
      return *this;
    }
  };
  std::vector<Analysis> analyes_;
  mutable std::atomic<unsigned long long> seenOrbits_, selectedBXs_;
  unsigned long long nPrint_;
  void printReport() const;
};

FinalBxSelector::FinalBxSelector(const edm::ParameterSet& iPSet)
    : seenOrbits_(0), selectedBXs_(0), nPrint_(iPSet.getUntrackedParameter<unsigned int>("nPrint")) {
  // get the list of selected BXs
  std::vector<edm::InputTag> bxLabels = iPSet.getParameter<std::vector<edm::InputTag>>("analysisLabels");
  for (const auto& bxLabel : bxLabels) {
    analyes_.emplace_back(bxLabel.encode(), consumes<std::vector<unsigned>>(bxLabel));
  }

  produces<std::vector<unsigned>>("SelBx").setBranchAlias("SelectedBxs");
}

// ------------ method called for each ORBIT  ------------
bool FinalBxSelector::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  bool noBxSelected = true;
  std::set<unsigned> uniqueBxs;

  edm::Handle<std::vector<unsigned>> bxList;
  for (const auto& analysis : analyes_) {
    iEvent.getByToken(analysis.token, bxList);
    analysis.nSelected += bxList->size();

    for (const unsigned& bx : *bxList) {
      uniqueBxs.insert(bx);
      noBxSelected = false;
    }
  }

  auto selectedBxs = std::make_unique<std::vector<unsigned>>(uniqueBxs.begin(), uniqueBxs.end());
  selectedBXs_ += selectedBxs->size();
  seenOrbits_++;
  iEvent.put(std::move(selectedBxs), "SelBx");

  if (nPrint_ != 0 && (seenOrbits_ % nPrint_ == 0)) {
    printReport();
  }
  return !noBxSelected;
}

void FinalBxSelector::printReport() const {
  std::ostringstream oss;
  oss << "Processed " << seenOrbits_.load() << " orbits.\n";
  for (const auto& analysis : analyes_) {
    oss << "Analysis " << analysis.name << " selected " << analysis.nSelected.load() << " BXs.\n";
  }
  oss << "FinalOR selected " << selectedBXs_.load() << " BXs.\n";
  edm::LogImportant("FinalBxSelector") << oss.str();
}

void FinalBxSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("analysisLabels");
  desc.addUntracked<unsigned int>("nPrint", 0);  // Number of orbits between printouts, 0 = no printout
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(FinalBxSelector);
