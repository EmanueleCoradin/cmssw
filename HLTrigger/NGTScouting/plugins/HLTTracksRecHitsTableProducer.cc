#include <memory>

// user include files
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

//
// class declaration
//

class HLTTracksRecHitsTableProducer : public edm::stream::EDProducer<> {
public:
  explicit HLTTracksRecHitsTableProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  const bool skipNonExistingSrc_;
  const std::string tableName_;
  const unsigned int precision_;
  const edm::EDGetTokenT<std::vector<reco::Track>> tracks_;
};

//
// constructors
//

HLTTracksRecHitsTableProducer::HLTTracksRecHitsTableProducer(const edm::ParameterSet& params)
    : skipNonExistingSrc_(params.getParameter<bool>("skipNonExistingSrc")),
      tableName_(params.getParameter<std::string>("tableName")),
      precision_(params.getParameter<int>("precision")),
      tracks_(consumes<std::vector<reco::Track>>(params.getParameter<edm::InputTag>("tracksSrc"))) {
  produces<nanoaod::FlatTable>(tableName_);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void HLTTracksRecHitsTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  //vertex collection
  auto tracksIn = iEvent.getHandle(tracks_);
  const size_t nTracks = tracksIn.isValid() ? (*tracksIn).size() : 0;

  static constexpr float default_value = std::numeric_limits<float>::quiet_NaN();
  static constexpr int maxRecHits = 16;

  std::vector<float> globalX(maxRecHits * nTracks, default_value);
  std::vector<float> globalY(maxRecHits * nTracks, default_value);
  std::vector<float> globalZ(maxRecHits * nTracks, default_value);
  std::vector<float> globalErrX(maxRecHits * nTracks, default_value);
  std::vector<float> globalErrY(maxRecHits * nTracks, default_value);
  std::vector<float> globalErrZ(maxRecHits * nTracks, default_value);

  std::vector<float> globalR(maxRecHits * nTracks, default_value);
  std::vector<float> globalEta(maxRecHits * nTracks, default_value);
  std::vector<float> globalPhi(maxRecHits * nTracks, default_value);

  if (tracksIn.isValid() || !(this->skipNonExistingSrc_)) {
    const auto& tracks = *tracksIn;
    for (size_t tkIndex = 0; tkIndex < nTracks; ++tkIndex) {
      const auto& track = tracks[tkIndex];
      for (auto it = track.recHitsBegin(); it != track.recHitsEnd(); ++it) {
        auto hit = *it;
        auto globalPoint = hit->globalPosition();
        auto globalError = hit->globalPositionError();
        auto hitIndex = std::distance(track.recHitsBegin(), it);
        if (hitIndex >= maxRecHits) {
          edm::LogWarning("HLTTracksRecHitsTableProducer")
              << " Track " << tkIndex << " has more (" << track.recHitsSize() << ") than " << maxRecHits
              << " recHits, skipping the rest.";
          break;
        }
        globalX[tkIndex * maxRecHits + hitIndex] = globalPoint.x();
        globalY[tkIndex * maxRecHits + hitIndex] = globalPoint.y();
        globalZ[tkIndex * maxRecHits + hitIndex] = globalPoint.z();
        globalErrX[tkIndex * maxRecHits + hitIndex] = globalError.cxx();
        globalErrY[tkIndex * maxRecHits + hitIndex] = globalError.cyy();
        globalErrZ[tkIndex * maxRecHits + hitIndex] = globalError.czz();
        
        globalR[tkIndex * maxRecHits + hitIndex] = globalPoint.perp();
        globalEta[tkIndex * maxRecHits + hitIndex] = globalPoint.eta();
        globalPhi[tkIndex * maxRecHits + hitIndex] = globalPoint.phi();
      }
    }
  } else {
    edm::LogWarning("HLTTracksRecHitsTableProducer")
        << " Invalid handle for " << tableName_ << " in tracks input collection";
  }

  assert(globalX.size() == globalY.size() && globalX.size() == globalZ.size() && 
         globalX.size() == globalErrX.size() && globalX.size() == globalErrY.size() && globalX.size() == globalErrZ.size() && 
         globalX.size() == globalR.size() && globalX.size() == globalEta.size() && globalX.size() == globalPhi.size());

  //table for all primary vertices
  auto tracksTable =
      std::make_unique<nanoaod::FlatTable>(nTracks * maxRecHits, tableName_, /*singleton*/ false, /*extension*/ false);
  tracksTable->addColumn<float>("globalX", globalX, "RecHits global x coordinate", precision_);
  tracksTable->addColumn<float>("globalY", globalY, "RecHits global y coordinate", precision_);
  tracksTable->addColumn<float>("globalZ", globalZ, "RecHits global z coordinate", precision_);
  tracksTable->addColumn<float>("globalErrX", globalErrX, "RecHits global x error", precision_);
  tracksTable->addColumn<float>("globalErrY", globalErrY, "RecHits global y error", precision_);
  tracksTable->addColumn<float>("globalErrZ", globalErrZ, "RecHits global z error", precision_);
  tracksTable->addColumn<float>("globalR", globalR, "RecHits global r coordinate", precision_);
  tracksTable->addColumn<float>("globalEta", globalEta, "RecHits global eta coordinate", precision_);
  tracksTable->addColumn<float>("globalPhi", globalPhi, "RecHits global phi coordinate", precision_);
  iEvent.put(std::move(tracksTable), tableName_);
}

// ------------ fill 'descriptions' with the allowed parameters for the module ------------
void HLTTracksRecHitsTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("skipNonExistingSrc", false)
      ->setComment("whether or not to skip producing the table on absent input product");
  desc.add<std::string>("tableName")->setComment("name of the flat table ouput");
  desc.add<edm::InputTag>("tracksSrc")->setComment("std::vector<reco::Track> input collections");
  desc.add<int>("precision", 7);
  descriptions.addWithDefaultLabel(desc);
}

// ------------ define this as a plug-in ------------
DEFINE_FWK_MODULE(HLTTracksRecHitsTableProducer);
