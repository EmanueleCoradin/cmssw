/**
 * PixelTrackTorchHighPuritySelector
 * =================================
 *
 * GPU/Accelerator module performing HighPurity pixel-track selection composed of:
 *
 *   1. CA-based quality preselection
 *   2. Feature extraction
 *   3. TorchScript DNN inference
 *   4. Score-based filtering
 *   5. Track/hit compaction and output production
 *
 * ------------------------------------------------------------------
 * Pipeline Overview
 * ------------------------------------------------------------------
 *
 *   Input:
 *       TracksSoA (pixel tracks + hit associations)
 *       TrackingRecHitsSoA
 *
 *   Transformations:
 *
 *       TracksSoA
 *          │
 *          v
 *       CA preselection
 *          │  Produces compacted preselected track index list 
 *          v
 *       Feature extraction
 *          │  Produces fixed-size features tensors
 *          v
 *       Torch inference
 *          │  Produces per-track classification score
 *          v
 *       Score filtering
 *          │  Filters tracks based on their classification scores
 *          v
 *       Output TrackSoA compaction
 *
 * ------------------------------------------------------------------
 * Torch Inference
 * ------------------------------------------------------------------
 *
 * The Torch model expects fixed-size tensors:
 *
 *     Track tensor:  [maxPreselectedTracks, N_track_features]
 *     Hit tensor:    [maxPreselectedTracks, MaxHitsPerTrack, N_hit_features]
 *
 * Padding slots are filled with NaNs.
 * ------------------------------------------------------------------
*/

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"

#include <deque>
#include <memory>
#include <mutex>
#include <optional>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#endif

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
#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackTorchHighPuritySelectorKernels.h"

#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {


  class PixelTrackTorchHighPuritySelector : public stream::EDProducer<> {
    using TkSoADevice  = reco::TracksSoACollection;
    using HitsOnDevice = reco::TrackingRecHitsSoACollection;
    using TrackHitSoA  = ::reco::TrackHitSoA;

  public:
    explicit PixelTrackTorchHighPuritySelector(const edm::ParameterSet&);
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
    const double scoreThreshold_;
    torch::AlpakaModel model_;
    const device::EDPutToken<TkSoADevice> tokenTrackOut_;

    #ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      std::optional<Queue> torchQueue_;
      std::optional<Event> featuresReadyEvent_;
      std::optional<Event> inferenceDoneEvent_;
    #endif  
  };


  PixelTrackTorchHighPuritySelector::PixelTrackTorchHighPuritySelector(
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
        scoreThreshold_(iConfig.getParameter<double>("scoreThreshold")),
        model_(iConfig.getParameter<edm::FileInPath>("model").fullPath()),
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

  void PixelTrackTorchHighPuritySelector::produce(
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

    // If not create yet, create an alpaka queue for Torch and associate it with the current device
    #ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      if (!torchQueue_) {
        torchQueue_.emplace(alpaka::getDev(queue));
        featuresReadyEvent_.emplace(alpaka::getDev(queue));
        inferenceDoneEvent_.emplace(alpaka::getDev(queue));
      }
    #endif  

    // Instantiate the necessary objects in memory
    //  - Temporary storage for filtering
    auto d_nPreselectedTracks      = cms::alpakatools::make_device_buffer<int>(queue);
    auto d_nSelectedTracks         = cms::alpakatools::make_device_buffer<int>(queue);
    auto d_preselectedTrackIndices = cms::alpakatools::make_device_buffer<int[]>(queue, maxNumberOfTracks_);
    auto d_selectedTrackIndices    = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_nKeptHits               = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_preselectionOffsets     = cms::alpakatools::make_device_buffer<int[]>(queue, maxNumberOfTracks_);
    
    alpaka::memset(queue, d_nPreselectedTracks, 0);
    alpaka::memset(queue, d_nSelectedTracks, 0);
    alpaka::memset(queue, d_nKeptHits, 0);
    alpaka::memset(queue, d_preselectedTrackIndices, 0xFF);
    alpaka::memset(queue, d_selectedTrackIndices, 0xFF);
    alpaka::memset(queue, d_preselectionOffsets, 0);

    //  - Features and scores containers
    PixelTrackFeaturesOnDevice  trackFeatures(maxPreselectedTracks_, queue);
    PixelRecHitFeaturesOnDevice hitFeatures(maxPreselectedTracks_, queue);
    PixelTrackScoresOnDevice    trackScoresOnDevice(maxPreselectedTracks_, queue);

    // - Tensor collections for DNN inference
    cms::torch::alpakatools::TensorCollection<Queue> inputs(maxPreselectedTracks_);
    cms::torch::alpakatools::TensorCollection<Queue> outputs(maxPreselectedTracks_);

    // Optional debug definitions
#ifdef PIXEL_TRACK_HP_DEBUG
    auto h_nPreselectedTracks  = cms::alpakatools::make_host_buffer<int>(queue);
    auto h_nSelectedTracks     = cms::alpakatools::make_host_buffer<int>(queue);
    int nPreselectedTracks     = 0;
    int nSelectedTracks        = 0;
    // Helper to copy the number of kept tracks back to host (debug only)
    auto fetchnPreselectedTracks = [&]() {
      alpaka::memcpy(queue, h_nPreselectedTracks, d_nPreselectedTracks);
      alpaka::wait(queue);
      return *h_nPreselectedTracks;
    };
    auto fetchnSelectedTracks = [&]() {
      alpaka::memcpy(queue, h_nSelectedTracks, d_nSelectedTracks);
      alpaka::wait(queue);
      return *h_nSelectedTracks;
    };
#endif

    // 1. CA-based preselection of tracks
    //  Launch first kernel to look which tracks need to be filtered out
    //  based on quality criteria from the CA
    launchCAPreselection(
      queue,
      maxNumberOfTracks_,
      minNumberOfHits_,
      minimumTrackQuality_,
      tracks.view(),
      alpaka::getPtrNative(d_preselectedTrackIndices),
      alpaka::getPtrNative(d_preselectionOffsets),
      alpaka::getPtrNative(d_nPreselectedTracks)
    );

#ifdef PIXEL_TRACK_HP_DEBUG
    nPreselectedTracks = fetchnPreselectedTracks();
    std::cout << "PixelTrackTorchHighPuritySelector::Prefiltered tracks=" << nPreselectedTracks << "\n";
#endif

    // 2. Feature extraction (track + hit SoA)
    launchFeaturesExtractor(
      queue, 
      maxPreselectedTracks_, 
      tracks.view(),
      tracks.view<TrackHitSoA>(), 
      hits.view(),
      alpaka::getPtrNative(d_preselectedTrackIndices),
      alpaka::getPtrNative(d_nPreselectedTracks),
      trackFeatures.view(),
      hitFeatures.view(),
      alpaka::getPtrNative(d_nKeptHits)
    );

    // 3. DNN inference
    //  Prepare TensorCollection inputs and outputs for the model
    auto track_record = trackFeatures.view().records();
    auto hit_record = hitFeatures.view().records();
    auto score_record = trackScoresOnDevice.view().records();

    inputs.add<::RecHitFeatures::PixelRecHitFeaturesSoA>("hit_features",
      hit_record.hits()
    );

    // Order must match the TorchScript model input schema
    inputs.add<PixelTrackFeaturesSoA>("track_features",
      track_record.chi2(),
      track_record.dzError(),
      track_record.dxyError(),
      track_record.eta(),
      track_record.nHits(),
      track_record.phi(),
      track_record.phiError(),
      track_record.pt(),
      track_record.qOverPtError(),
      track_record.dzBS(),
      track_record.dxyBS(),
      track_record.nLayers(),
      track_record.cotThetaError(),
      track_record.covCotThetaDz(),   
      track_record.covDxyQOverPt(),
      track_record.covPhiDxy(),
      track_record.covPhiQOverPt()
    );

    outputs.add<PixelTrackScoresSoA>("track_scores",
      score_record.score()
    );
    //
    //  Launch inference
    // 
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      
      // Record that features are ready on the Alpaka stream
      alpaka::enqueue(queue, *featuresReadyEvent_);
      // Make the Torch stream wait until features are ready
      alpaka::wait(*torchQueue_, *featuresReadyEvent_);

      // Run inference on the Torch stream
      model_.forward(*torchQueue_, inputs, outputs);

      // Record that inference is done on the Torch stream
      alpaka::enqueue(*torchQueue_, *inferenceDoneEvent_);
      // Make the Alpaka stream wait until inference is done
      alpaka::wait(queue, *inferenceDoneEvent_);
      
#else      
      model_.forward(queue, inputs, outputs);
#endif

    // 4. Score-based filtering
    launchScoreFilter(
      queue,
      maxPreselectedTracks_,
      scoreThreshold_,
      trackScoresOnDevice.view(),
      alpaka::getPtrNative(d_preselectedTrackIndices),
      alpaka::getPtrNative(d_nPreselectedTracks),
      alpaka::getPtrNative(d_selectedTrackIndices),
      alpaka::getPtrNative(d_nSelectedTracks),
      alpaka::getPtrNative(d_nKeptHits)
    );


#ifdef PIXEL_TRACK_HP_DEBUG    
    nSelectedTracks = fetchnSelectedTracks();
    std::cout << "PixelTrackTorchHighPuritySelector::Filtered tracks=" << nSelectedTracks << "\n";
#endif

    auto tracks_out = launchProduceOutputTracks (
        queue,
        maxPreselectedTracks_,
        avgHitsPerTrack_,
        tracks.view(),
        tracks.view<TrackHitSoA>(), 
        alpaka::getPtrNative(d_selectedTrackIndices),
        alpaka::getPtrNative(d_nSelectedTracks),
        alpaka::getPtrNative(d_nKeptHits)
      );

    iEvent.emplace(tokenTrackOut_, std::move(tracks_out));
  }

  void PixelTrackTorchHighPuritySelector::fillDescriptions(
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
DEFINE_FWK_ALPAKA_MODULE(PixelTrackTorchHighPuritySelector);