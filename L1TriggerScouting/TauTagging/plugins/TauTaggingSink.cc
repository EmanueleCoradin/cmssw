#include <cassert>

#include <fmt/format.h>

#include "DataFormats/L1ScoutingSoA/interface/ClustersHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PFCandidateHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "L1TriggerScouting/Phase2/interface/L1TScPhase2Common.h"

namespace l1sc {

  using namespace cms::alpakatools;

  inline edm::InputTag getBackendTag(edm::InputTag const& tag) {
    return edm::InputTag(tag.label(), "backend", tag.process());
  }

  class TauTaggingSink : public edm::stream::EDAnalyzer<> {
  public:
    TauTaggingSink(const edm::ParameterSet& params)
        : pf_token_{consumes(params.getUntrackedParameter<edm::InputTag>("pf"))},
          clusters_token_{consumes(params.getUntrackedParameter<edm::InputTag>("clusters"))},
          pf_backend_{consumes(getBackendTag(params.getUntrackedParameter<edm::InputTag>("pf")))},
          clusters_backend_{consumes(getBackendTag(params.getUntrackedParameter<edm::InputTag>("clusters")))},
          environment_{static_cast<Environment>(params.getUntrackedParameter<int>("environment"))} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<int>("environment", static_cast<int>(Environment::kDevelopment));
      desc.addUntracked<edm::InputTag>("pf");
      desc.addUntracked<edm::InputTag>("clusters");
      descriptions.addWithDefaultLabel(desc);
    }

    void analyze(edm::Event const& event, edm::EventSetup const&) override {
      if (environment_ >= Environment::kDevelopment) {
        constexpr int total_len = 72;
        auto label = fmt::format("EVENT: {}", event.id().event());
        int pad = total_len - static_cast<int>(label.size());
        int pad_left = pad / 2;
        int pad_right = pad - pad_left - 1;
        fmt::print("\n{0} {1} {2}\n\n", std::string(pad_left, '-'), label, std::string(pad_right, '-'));
      }

      // unpacking/decoding step debugging
      const auto pf_handle = event.getHandle(pf_token_);
      const auto clusters_handle = event.getHandle(clusters_token_);
      
      if (pf_handle.isValid()) {
        // pf
        auto const& pf = *pf_handle;
        auto const pf_backend = static_cast<Backend>(event.get(pf_backend_));
        if (!clusters_handle.isValid()) {
          print(pf.const_view(), toString(pf_backend));
        } else {
          // clusters
          auto const& clusters = *clusters_handle;
          // backends
          auto const clusters_backend = static_cast<Backend>(event.get(clusters_backend_));
          print(pf.const_view(), clusters.const_view(), toString(pf_backend), toString(clusters_backend));
        }
        
      }
    }

    void beginStream(edm::StreamID) override {
      if (environment_ >= Environment::kDevelopment) {
        fmt::print("=========================================================================\n");
      }
    }

    void endStream() override {
      fmt::print("=========================================================================\n");
      fmt::print("[INFO] OK - TauTagging\n");
      fmt::print("=========================================================================\n");
    }

    void print(const PFCandidateHostCollection::ConstView& pf, const ClustersHostCollection::ConstView& clusters, const std::string_view pf_backend, const std::string_view clusters_backend) {
      const auto size = pf.metadata().size();
      if (size == 0)
        return;

      // Header
      assert(pf_backend == clusters_backend);
      int clusters_num = 0;
      for (int i = 0; i < clusters.metadata().size(); ++i) {
        if (clusters.cluster()[i] > clusters_num)
          clusters_num = clusters.cluster()[i];
      }
      fmt::print("[DEBUG] CLUETaus[{}] ({}) found {} clusters:\n", size, pf_backend, clusters_num);

      constexpr auto sep = "+---------+---------+---------+---------+---------+---------+-------+---------+";
      auto printHeader = [&] {
        fmt::print("{}\n", sep);
        fmt::print("| {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>5} | {:>7} |\n",
                   "index",
                   "is_seed",
                   "cluster",
                   "pt",
                   "eta",
                   "phi",
                   "pdgid",
                   "z0");
        fmt::print("{}\n", sep);
      };

      auto printRow = [&](int index, const auto& pf_view, const auto& clusters_view) {
        fmt::print("| {:7d} | {:7d} | {:7d} | {:7.2f} | {:7.2f} | {:7.2f} | {:5d} | {:7.2f} |\n",
                   index,
                   clusters_view.is_seed(),
                   clusters_view.cluster(),
                   pf_view.pt(),
                   pf_view.eta(),
                   pf_view.phi(),
                   pf_view.pdgid(),
                   pf_view.z0());
      };

      printHeader();

      const int max_entries = (environment_ > Environment::kTest) ? size : 5;

      for (int i = 0; i < pf.metadata().size() && i < max_entries; ++i) {
        if (i >= size)
          break;
        printRow(i, pf[i], clusters[i]);
      }

      if (max_entries < size) {
        fmt::print("| {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>5} | {:>7} |\n",
                   "...",
                   "...",
                   "...",
                   "...",
                   "...",
                   "...",
                   "...",
                   "...");
        for (int j = pf.metadata().size() - 5; j < pf.metadata().size(); ++j) {
          printRow(j, pf[j], clusters[j]);
        }
      }

      fmt::print("{}\n", sep);
    }

    void print(const PFCandidateHostCollection::ConstView& pf, const std::string_view pf_backend) {
      const auto size = pf.metadata().size();
      if (size == 0)
        return;

      // Header
      fmt::print("[DEBUG] PFCandidateCollection[{}] ({}):\n", size, pf_backend);

      constexpr auto sep = "+---------+---------+---------+---------+-------+---------+";
      auto printHeader = [&] {
        fmt::print("{}\n", sep);
        fmt::print("| {:>7} | {:>7} | {:>7} | {:>7} | {:>5} | {:>7} |\n",
                   "index",
                   "pt",
                   "eta",
                   "phi",
                   "pdgid",
                   "z0");
        fmt::print("{}\n", sep);
      };

      auto printRow = [&](int index, const auto& view) {
        fmt::print("| {:7d} | {:7.2f} | {:7.2f} | {:7.2f} | {:5d} | {:7.2f} |\n",
                   index,
                   view.pt(),
                   view.eta(),
                   view.phi(),
                   view.pdgid(),
                   view.z0());
      };

      printHeader();

      const int max_entries = (environment_ > Environment::kTest) ? size : 5;

      for (int i = 0; i < pf.metadata().size() && i < max_entries; ++i) {
        if (i >= size)
          break;
        printRow(i, pf[i]);
      }

      if (max_entries < size) {
        fmt::print("| {:>7} | {:>7} | {:>7} | {:>7} | {:>5} | {:>7} |\n",
                   "...",
                   "...",
                   "...",
                   "...",
                   "...",
                   "...");
        for (int j = pf.metadata().size() - 5; j < pf.metadata().size(); ++j) {
          printRow(j, pf[j]);
        }
      }

      fmt::print("{}\n", sep);
    }

  private:
    // get products
    const edm::EDGetTokenT<PFCandidateHostCollection> pf_token_;
    const edm::EDGetTokenT<ClustersHostCollection> clusters_token_;
    // backend query
    const edm::EDGetTokenT<unsigned short> pf_backend_;
    const edm::EDGetTokenT<unsigned short> clusters_backend_;
    // debug
    const Environment environment_;
  };

}  // namespace l1sc

DEFINE_FWK_MODULE(l1sc::TauTaggingSink);