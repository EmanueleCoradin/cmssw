#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"

#include "RecoTracker/FinalTrackSelectors/interface/PixelRecHitFeaturesSoA.h"
#include "RecoTracker/FinalTrackSelectors/interface/PixelTrackFeaturesSoA.h"
#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackFeaturesDeviceCollection.h"
#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackFeaturesExtractorKernels.h"

#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"


#ifndef PIXEL_TRACK_HP_DEBUG
#define PIXEL_TRACK_HP_DEBUG
#endif

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class PixelTrackFeaturesExtractor : public stream::EDProducer<> {
    using TkSoADevice  = reco::TracksSoACollection;
    using HitsOnDevice = reco::TrackingRecHitsSoACollection;
    using TrackHitSoA  = ::reco::TrackHitSoA;

  public:
    explicit PixelTrackFeaturesExtractor(const edm::ParameterSet&);
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void produce(device::Event&, const device::EventSetup&) override;

    const device::EDGetToken<HitsOnDevice> recHitToken_;
    const device::EDGetToken<TkSoADevice> pixelTrackToken_;
    const int32_t maxNumberOfTracks_;
    const int32_t maxPreselectedTracks_;
    const int32_t maxHitsPerTrack_;
    const int32_t minNumberOfHits_;
    const int32_t avgHitsPerTrack_;
    const pixelTrack::Quality minimumTrackQuality_;

    torch::AlpakaModel model_;
    const double scoreThreshold_;

    const device::EDPutToken<TkSoADevice> tokenTrackOut_;
  };

  PixelTrackFeaturesExtractor::PixelTrackFeaturesExtractor(
      const edm::ParameterSet& iConfig)
      : EDProducer(iConfig),
        recHitToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
        pixelTrackToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"))),
        maxNumberOfTracks_(iConfig.getParameter<int>("maxNumberOfTracks")),
        maxPreselectedTracks_(iConfig.getParameter<int>("maxPreselectedTracks")),
        maxHitsPerTrack_(RecHitFeatures::MaxHitsPerTrack),
        minNumberOfHits_(iConfig.getParameter<int>("minNumberOfHits")),
        avgHitsPerTrack_(iConfig.getParameter<int>("avgHitsPerTrack")),
        minimumTrackQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minimumTrackQuality"))),
        model_(iConfig.getParameter<edm::FileInPath>("model").fullPath()),
        scoreThreshold_(iConfig.getParameter<double>("scoreThreshold")),
        tokenTrackOut_(produces())
  {
    if (minimumTrackQuality_ == pixelTrack::Quality::notQuality) {
      throw cms::Exception("PixelTrackConfiguration")
        << iConfig.getParameter<std::string>("minimumTrackQuality") + " is not a pixelTrack::Quality";
    }
    if (minimumTrackQuality_ < pixelTrack::Quality::dup) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minimumTrackQuality") + " not supported";
    }
    if (maxPreselectedTracks_ > maxNumberOfTracks_) {
      throw cms::Exception("PixelTrackConfiguration")
          << "maxPreselectedTracks must be <= maxNumberOfTracks";
    }
  }

  void PixelTrackFeaturesExtractor::produce(
      device::Event& iEvent,
      const device::EventSetup&) 
  {
/* 
    Processing steps:
      1. CA-based preselection of tracks
      2. Feature extraction (track + hit SoA)
      3. DNN inference
      4. Score-based filtering
      5. Track compaction and output production
*/

    // Retrieve tokens
    auto&       queue  = iEvent.queue();
    const auto& hits   = iEvent.get(recHitToken_);
    const auto& tracks = iEvent.get(pixelTrackToken_);

    // Instantiate the necessary objects in memory
    //  - Temporary storage for filtering
    auto d_nKeptTracks = cms::alpakatools::make_device_buffer<int>(queue);
    auto d_nKeptHits   = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_originalTrackIndex    = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    alpaka::memset(queue, d_nKeptTracks, 0);

    //  - Features and scores containers
    PixelTrackFeaturesOnDevice  trackFeatures(maxPreselectedTracks_, queue);
    PixelRecHitFeaturesOnDevice hitFeatures(maxPreselectedTracks_, queue);
    PixelTrackScoresOnDevice    trackScoresOnDevice(maxPreselectedTracks_, queue);

    // - Tensor collections for DNN inference
    cms::torch::alpakatools::TensorCollection<Queue> inputs(maxPreselectedTracks_);
    cms::torch::alpakatools::TensorCollection<Queue> outputs(maxPreselectedTracks_);

    // Optional debug definitions
#ifdef PIXEL_TRACK_HP_DEBUG
    auto h_nKeptTracks = cms::alpakatools::make_host_buffer<int>(queue);
    int nKeptTracks = 0;
    // Helper to copy the number of kept tracks back to host (debug only)
    auto fetchNKeptTracks = [&]() {
      alpaka::memcpy(queue, h_nKeptTracks, d_nKeptTracks);
      alpaka::wait(queue);
      return *h_nKeptTracks;
    };
#endif

    // 1. CA-based preselection of tracks
    //  Launch first kernel to look which tracks need to be filtered out
    //  based on quality criteria from the CA

    launchCAPreselectionKernel(
      queue,
      maxNumberOfTracks_,
      maxPreselectedTracks_,
      minNumberOfHits_,
      minimumTrackQuality_,
      tracks.view(),
      alpaka::getPtrNative(d_nKeptTracks),
      alpaka::getPtrNative(d_nKeptHits),
      alpaka::getPtrNative(d_originalTrackIndex)
    );

#ifdef PIXEL_TRACK_HP_DEBUG
    nKeptTracks = fetchNKeptTracks();
    std::cout << "PixelTrackFeaturesExtractor::Prefiltered tracks=" << nKeptTracks << "\n";
#endif

    // 2. Feature extraction (track + hit SoA)
    launchFeaturesExtractorKernel(
      queue, 
      maxPreselectedTracks_, 
      tracks.view(),
      tracks.view<TrackHitSoA>(), 
      hits.view(),
      trackFeatures.view(),
      hitFeatures.view(),
      alpaka::getPtrNative(d_nKeptTracks),
      alpaka::getPtrNative(d_originalTrackIndex)
    );

    // 3. DNN inference
    //  Prepare TensorCollection inputs and outputs for the model
    auto track_record = trackFeatures.view().records();
    auto hit_record = hitFeatures.view().records();
    auto score_record = trackScoresOnDevice.view().records();

    inputs.add<::RecHitFeatures::PixelRecHitFeaturesSoA>("hit_features",
      hit_record.hits()
    );

    inputs.add<PixelTrackFeaturesSoA>("track_features",
      track_record.chi2(),
      track_record.dzError(),
      track_record.dxyError(),
      track_record.eta(),
      track_record.ndof(),
      track_record.phi(),
      track_record.phiError(),
      track_record.pt(),
      track_record.ptError(),
      track_record.qoverp(),
      track_record.dzBS(),
      track_record.dxyBS()
    );

    outputs.add<PixelTrackScoresSoA>("track_scores",
      score_record.score()
    );

    //  Launch inference
    model_.forward(queue, inputs, outputs);

    // 4. Score-based filtering
    launchScoreFilterKernel(
      queue,
      maxPreselectedTracks_,
      scoreThreshold_,
      alpaka::getPtrNative(d_originalTrackIndex),
      trackScoresOnDevice.view()
    );

    // 5. Track compaction and output production
    launchHitOffsetCompactKernel(
      queue,
      maxPreselectedTracks_,
      alpaka::getPtrNative(d_originalTrackIndex),
      alpaka::getPtrNative(d_nKeptTracks),
      alpaka::getPtrNative(d_nKeptHits)
    );

#ifdef PIXEL_TRACK_HP_DEBUG    
    nKeptTracks = fetchNKeptTracks();
    std::cout << "PixelTrackFeaturesExtractor::Filtered tracks=" << nKeptTracks << "\n";
#endif

    auto tracks_out = launchProduceOutputTracks (
        queue,
        maxPreselectedTracks_,
        avgHitsPerTrack_,
        tracks.view(),
        tracks.view<TrackHitSoA>(), 
        alpaka::getPtrNative(d_originalTrackIndex),
        alpaka::getPtrNative(d_nKeptTracks),
        alpaka::getPtrNative(d_nKeptHits)
      );

    iEvent.emplace(tokenTrackOut_, std::move(tracks_out));
  }

  void PixelTrackFeaturesExtractor::fillDescriptions(
      edm::ConfigurationDescriptions& descriptions) 
  {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("pixelRecHitSrc", {"hltPhase2SiPixelRecHitsSoA"});
    desc.add<edm::InputTag>("pixelTrackSrc", {"hltPhase2PixelTracksSoA"});
    desc.add<int>("maxNumberOfTracks", 100000);
    desc.add<int>("maxPreselectedTracks", 10000);
    desc.add<int>("minNumberOfHits", 0);
    desc.add<int>("avgHitsPerTrack", 8);
    desc.add<std::string>("minimumTrackQuality", "tight");
    desc.add<edm::FileInPath>("model");
    desc.add<double>("scoreThreshold", 0.5);
    descriptions.addWithDefaultLabel(desc);
  }
};

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PixelTrackFeaturesExtractor);