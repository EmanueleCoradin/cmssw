#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"

#include "RecoTracker/FinalTrackSelectors/interface/PixelTrackFeaturesSoA.h"
#include "RecoTracker/FinalTrackSelectors/interface/PixelRecHitFeaturesSoA.h"
#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackFeaturesDeviceCollection.h"
#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackFeaturesExtractorKernels.h"

#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"


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

    const device::EDGetToken<HitsOnDevice> tokenHit_;
    const device::EDGetToken<TkSoADevice> tokenTrackIn_;
    const int32_t maxTracks_;
    const int32_t maxTracksPreselection_;
    const int32_t maxHitsPerTrack_;
    const int32_t minNumberOfHits_;
    const int32_t avgHitsPerTrack_;
    const pixelTrack::Quality minQuality_;

    torch::AlpakaModel model_;
    const double scoreThreshold_;

    const device::EDPutToken<TkSoADevice> tokenTrackOut_;
  };

  PixelTrackFeaturesExtractor::PixelTrackFeaturesExtractor(
      const edm::ParameterSet& iConfig)
      : EDProducer(iConfig),
        tokenHit_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
        tokenTrackIn_(consumes(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"))),
        maxTracks_(iConfig.getParameter<int>("maxTracks")),
        maxTracksPreselection_(iConfig.getParameter<int>("maxTracksPreselection")),
        maxHitsPerTrack_(RecHitFeatures::MaxHitsPerTrack),
        minNumberOfHits_(iConfig.getParameter<int>("minNumberOfHits")),
        avgHitsPerTrack_(iConfig.getParameter<int>("avgHitsPerTrack")),
        minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
        model_(iConfig.getParameter<edm::FileInPath>("model").fullPath()),
        scoreThreshold_(iConfig.getParameter<double>("scoreThreshold")),
        tokenTrackOut_(produces())
  {
    if (minQuality_ == pixelTrack::Quality::notQuality) {
      throw cms::Exception("PixelTrackConfiguration")
        << iConfig.getParameter<std::string>("minQuality") + " is not a pixelTrack::Quality";
    }
    if (minQuality_ < pixelTrack::Quality::dup) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " not supported";
    }
  }

  void PixelTrackFeaturesExtractor::produce(
      device::Event& iEvent,
      const device::EventSetup&) 
  {
    std::cout << "PixelTrackFeaturesExtractor::produce" << std::endl;

    // Retrieve tokens
    auto& queue = iEvent.queue();

    const auto& hits   = iEvent.get(tokenHit_);
    const auto& tracks = iEvent.get(tokenTrackIn_);

    //Instanciate the necessary objects in memory
    std::cout << "PixelTrackFeaturesExtractor::Producing features collection" << std::endl;

    PixelTrackFeaturesOnDevice features(maxTracksPreselection_, queue);
    PixelRecHitFeaturesOnDevice features_hit(maxTracksPreselection_, queue);

    auto d_nKeptTracks = cms::alpakatools::make_device_buffer<int>(queue);
    auto h_nKeptTracks = cms::alpakatools::make_host_buffer<int>(queue);
    auto d_nKeptHits   = cms::alpakatools::make_device_buffer<int[]>(queue, maxTracksPreselection_);
    auto d_oldIndex    = cms::alpakatools::make_device_buffer<int[]>(queue, maxTracksPreselection_);

    alpaka::memset(queue, d_nKeptTracks, 0);

    //Launch first kernel to look which tracks need to be filtered out
    // based on quality criteria from the CA
    std::cout << "PixelTrackFeaturesExtractor::Launching Preselection kernel" << std::endl;
    launchCAPreselectionKernel(
      queue,
      maxTracks_,
      maxTracksPreselection_,
      minNumberOfHits_,
      minQuality_,
      tracks.view(),
      alpaka::getPtrNative(d_nKeptTracks),
      alpaka::getPtrNative(d_nKeptHits),
      alpaka::getPtrNative(d_oldIndex)
    );
    std::cout << "PixelTrackFeaturesExtractor::done" << std::endl;

    alpaka::memcpy(queue, h_nKeptTracks, d_nKeptTracks);
    alpaka::wait(queue);
    int nKeptTracks = *h_nKeptTracks;

    std::cout << "PixelTrackFeaturesExtractor::Prefiltered tracks=" << nKeptTracks << "\n";

    //Launch the kernel to extract Features SoA
    std::cout << "PixelTrackFeaturesExtractor::Launching Features Extractor kernel" << std::endl;
    launchFeaturesExtractorKernel(
      queue, 
      maxTracksPreselection_, 
      tracks.view(),
      tracks.view<TrackHitSoA>(), 
      hits.view(),
      features.view(),
      features_hit.view(),
      alpaka::getPtrNative(d_nKeptTracks),
      alpaka::getPtrNative(d_oldIndex)
    );

    //Perform the DNN inference
    
    // Combined input TensorCollection
    cms::torch::alpakatools::TensorCollection<Queue> inputs(maxTracksPreselection_);
    cms::torch::alpakatools::TensorCollection<Queue> outputs(maxTracksPreselection_);

    PixelTrackScoresOnDevice trackScoresOnDevice(maxTracksPreselection_, queue);
    
    auto track_record = features.view().records();
    auto hit_record = features_hit.view().records();
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

    model_.forward(queue, inputs, outputs);

    std::cout << "PixelTrackFeaturesExtractor::DNN inference done" << std::endl;
    //Filter tracks based on score threshold
    launchScoreFilterKernel(
      queue,
      maxTracksPreselection_,
      scoreThreshold_,
      alpaka::getPtrNative(d_nKeptTracks),
      alpaka::getPtrNative(d_nKeptHits),
      alpaka::getPtrNative(d_oldIndex),
      trackScoresOnDevice.view()
    );
    std::cout << "PixelTrackFeaturesExtractor::Filtering done" << std::endl;


    //Compact the d_nKeptHits to prepare to fill the new hit Offset
    launchHitOffsetCompactKernel(
      queue,
      maxTracksPreselection_,
      alpaka::getPtrNative(d_oldIndex),
      alpaka::getPtrNative(d_nKeptTracks),
      alpaka::getPtrNative(d_nKeptHits)
    );

    alpaka::memcpy(queue, h_nKeptTracks, d_nKeptTracks);
    alpaka::wait(queue);
    nKeptTracks = *h_nKeptTracks;
    std::cout << "PixelTrackFeaturesExtractor::Filtered tracks=" << nKeptTracks << "\n";

    auto tracks_out = launchProduceOutputTracks (
        queue,
        maxTracksPreselection_,
        avgHitsPerTrack_,
        tracks.view(),
        tracks.view<TrackHitSoA>(), 
        alpaka::getPtrNative(d_oldIndex),
        alpaka::getPtrNative(d_nKeptTracks),
        alpaka::getPtrNative(d_nKeptHits)
      );

    iEvent.emplace(tokenTrackOut_, std::move(tracks_out));
    //iEvent.emplace(tokenTrackOut_, std::move(outTracks));
  }

  void PixelTrackFeaturesExtractor::fillDescriptions(
      edm::ConfigurationDescriptions& descriptions) 
  {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("pixelRecHitSrc", {"hltPhase2SiPixelRecHitsSoA"});
    desc.add<edm::InputTag>("pixelTrackSrc", {"hltPhase2PixelTracksSoA"});
    desc.add<int>("maxTracks", 100000);
    desc.add<int>("maxTracksPreselection", 10000);
    desc.add<int>("minNumberOfHits", 0);
    desc.add<int>("avgHitsPerTrack", 8);
    desc.add<std::string>("minQuality", "tight");
    desc.add<edm::FileInPath>("model");
    desc.add<double>("scoreThreshold", 0.5);
    descriptions.addWithDefaultLabel(desc);
  }
};

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PixelTrackFeaturesExtractor);