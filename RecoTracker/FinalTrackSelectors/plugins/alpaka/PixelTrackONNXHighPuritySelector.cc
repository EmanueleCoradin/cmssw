/**
 * PixelTrackONNXHighPuritySelector
 * =================================
 *
 * GPU/Accelerator module performing HighPurity pixel-track selection composed of:
 *
 *   1. CA-based quality preselection
 *   2. Feature extraction
 *   3. ONNX Runtime inference
 *   4. Score-based filtering
 *   5. Track/hit compaction and output production
 *
 * ------------------------------------------------------------------
 * Pipeline Overview
 * ------------------------------------------------------------------
 *
 *   Input:
 *       TracksSoA (pixel tracks + hit associations)
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
 *       ONNX inference
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
 *     Track tensor:  [maxPreselectedTracks, N_track_features]
 *
 * Padding slots are filled with 0s.
 * ------------------------------------------------------------------
*/

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include <deque>
#include <optional>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/PixelTrackFeaturesSoA.h"
#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackFeaturesDeviceCollection.h"
#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackTorchHighPuritySelectorKernels.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/alpaka/AlpakaSession.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoTrace.h"

// #define PIXEL_TRACK_HP_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  std::optional<cms::Ort::Backend> parseBackend(const std::string& name) {
    if (name == "auto")
      return std::nullopt;
    if (name == "cpu")
      return cms::Ort::Backend::cpu;
    if (name == "cuda")
      return cms::Ort::Backend::cuda;
    if (name == "tensorRT")
      return cms::Ort::Backend::tensorRT;
    if (name == "rocm")
      return cms::Ort::Backend::rocm;

    throw cms::Exception("Configuration") << "Unsupported ONNX Runtime backend: " << name;
  }

  /// Input/output tensors associated to a single inference batch.
  struct BatchIO {
    cms::Ort::alpakatools::TensorCollection<Queue> inputs;
    cms::Ort::alpakatools::TensorCollection<Queue> outputs;
  };

  class PixelTrackONNXHighPuritySelector : public stream::FixedQueueEDProducer<> {
    using TkSoADevice = reco::TracksSoACollection;
    using TrackHitSoA = ::reco::TrackHitSoA;

  public:
    explicit PixelTrackONNXHighPuritySelector(const edm::ParameterSet&);
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void produce(device::Event&, const device::EventSetup&) override;
    void beginStream(edm::StreamID /*sid*/, Queue queue) override;
    const device::EDGetToken<TkSoADevice> pixelTrackToken_;
    const int maxNumberOfTracks_;
    const int maxPreselectedTracks_;
    const int minNumberOfHits_;
    const int avgHitsPerTrack_;
    const pixelTrack::Quality minimumTrackQuality_;
    const double scoreThreshold_;

    ort::AlpakaSession model_;

    const bool featureMajor_;
    const int batchSize_;
    const device::EDPutToken<TkSoADevice> tokenTrackOut_;
  };

  PixelTrackONNXHighPuritySelector::PixelTrackONNXHighPuritySelector(const edm::ParameterSet& iConfig)

      : FixedQueueEDProducer(iConfig),
        pixelTrackToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"))),
        maxNumberOfTracks_(iConfig.getParameter<int>("maxNumberOfTracks")),
        maxPreselectedTracks_(iConfig.getParameter<int>("maxPreselectedTracks")),
        minNumberOfHits_(iConfig.getParameter<int>("minNumberOfHits")),
        avgHitsPerTrack_(iConfig.getParameter<int>("avgHitsPerTrack")),
        minimumTrackQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minimumTrackQuality"))),
        scoreThreshold_(iConfig.getParameter<double>("scoreThreshold")),
        model_(iConfig.getParameter<edm::FileInPath>("model").fullPath(),
               parseBackend(iConfig.getParameter<std::string>("backend"))),

        featureMajor_(iConfig.getParameter<bool>("featureMajor")),
        batchSize_(iConfig.getParameter<int>("batchSize")),
        tokenTrackOut_(produces()) {
    if (minimumTrackQuality_ == pixelTrack::Quality::notQuality) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minimumTrackQuality") + " is not a pixelTrack::Quality";
    }

    if (minimumTrackQuality_ < pixelTrack::Quality::dup) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minimumTrackQuality") + " not supported";
    }

    if (batchSize_ <= 0 || maxPreselectedTracks_ <= 0) {
      throw cms::Exception("PixelTrackConfiguration") << "batchSize and maxPreselectedTracks must be positive";
    }

    if (maxPreselectedTracks_ > maxNumberOfTracks_) {
      throw cms::Exception("PixelTrackConfiguration") << "maxPreselectedTracks must be <= maxNumberOfTracks";
    }
  }

  void PixelTrackONNXHighPuritySelector::beginStream(edm::StreamID /*sid*/, Queue queue) {
    // Bind ONNX Runtime to this EDM stream's fixed Alpaka queue.
    model_.bind(queue);
  }

  void PixelTrackONNXHighPuritySelector::produce(device::Event& iEvent, const device::EventSetup&) {
    /* 
    Processing steps:
      1. CA-based preselection of tracks
      2. Feature extraction (track SoA)
      3. DNN inference
      4. Score-based filtering
      5. Track compaction and output production
*/
    CMS_PERFETTO_SCOPE("PixelTrackONNXHighPuritySelector");
    // Retrieve tokens
    auto& queue = iEvent.queue();
    const auto& tracks = iEvent.get(pixelTrackToken_).view();

    // Instantiate the necessary objects in memory

    //  - Temporary storage for filtering

    auto d_nPreselectedTracks = cms::alpakatools::make_device_buffer<int>(queue);
    auto d_nSelectedTracks = cms::alpakatools::make_device_buffer<int>(queue);
    auto d_preselectedTrackIndices = cms::alpakatools::make_device_buffer<int[]>(queue, maxNumberOfTracks_);
    auto d_selectedTrackIndices = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_trackHitCounts = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_selectedTrackHitOffsets = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_preselectionOffsets = cms::alpakatools::make_device_buffer<int[]>(queue, maxNumberOfTracks_);

    alpaka::memset(queue, d_nPreselectedTracks, 0);
    alpaka::memset(queue, d_nSelectedTracks, 0);
    alpaka::memset(queue, d_trackHitCounts, 0);
    alpaka::memset(queue, d_selectedTrackHitOffsets, 0);
    alpaka::memset(queue, d_preselectedTrackIndices, 0xFF);
    alpaka::memset(queue, d_selectedTrackIndices, 0xFF);
    alpaka::memset(queue, d_preselectionOffsets, 0);

    //  - Features and scores containers

    PixelTrackFeaturesOnDevice trackFeatures(queue, maxPreselectedTracks_);
    PixelTrackScoresOnDevice trackScoresOnDevice(queue, maxPreselectedTracks_);

    // Optional debug definitions

#ifdef PIXEL_TRACK_HP_DEBUG

    auto h_nPreselectedTracks = cms::alpakatools::make_host_buffer<int>(queue);
    auto h_nSelectedTracks = cms::alpakatools::make_host_buffer<int>(queue);
    auto nPreselectedTracks = 0;
    auto nSelectedTracks = 0;
    // Helper to copy the number of kept tracks back to host (debug only)
    auto fetchNumPreselectedTracks = [&]() {
      alpaka::memcpy(queue, h_nPreselectedTracks, d_nPreselectedTracks);
      alpaka::wait(queue);
      return *h_nPreselectedTracks;
    };
    auto fetchNumSelectedTracks = [&]() {
      alpaka::memcpy(queue, h_nSelectedTracks, d_nSelectedTracks);
      alpaka::wait(queue);
      return *h_nSelectedTracks;
    };
#endif

    // 1. CA-based preselection of tracks
    //  Launch first kernel to look which tracks need to be filtered out
    //  based on quality criteria from the CA
    {
      CMS_PERFETTO_SCOPE("CA-Preselection");

      launchCAPreselection(queue,
                           maxNumberOfTracks_,
                           minNumberOfHits_,
                           minimumTrackQuality_,
                           tracks.tracks(),
                           alpaka::getPtrNative(d_preselectedTrackIndices),
                           alpaka::getPtrNative(d_preselectionOffsets),
                           alpaka::getPtrNative(d_nPreselectedTracks));
    }
#ifdef PIXEL_TRACK_HP_DEBUG

    nPreselectedTracks = fetchNumPreselectedTracks();
    std::cout << "PixelTrackONNXHighPuritySelector::Prefiltered tracks=" << nPreselectedTracks << "\n";

#endif
    {
      CMS_PERFETTO_SCOPE("FeaturesExtractor");

      launchFeaturesExtractor(queue,
                              maxPreselectedTracks_,
                              tracks.tracks(),
                              alpaka::getPtrNative(d_preselectedTrackIndices),
                              alpaka::getPtrNative(d_nPreselectedTracks),
                              trackFeatures.view(),
                              alpaka::getPtrNative(d_trackHitCounts));
    }
    // 3. DNN inference
    //  Prepare TensorCollection inputs and outputs for the model
    auto track_record = trackFeatures.view().records();
    auto score_record = trackScoresOnDevice.view().records();
    const auto n_batches = featureMajor_ ? 1 : (maxPreselectedTracks_ + batchSize_ - 1) / batchSize_;
    std::deque<BatchIO> batches;

    // - Tensor collections for DNN inference
    for (auto i_batch = 0; i_batch < n_batches; ++i_batch) {
      batches.emplace_back(BatchIO{cms::Ort::alpakatools::TensorCollection<Queue>(
                                       featureMajor_ ? maxPreselectedTracks_ : batchSize_, maxPreselectedTracks_),
                                   cms::Ort::alpakatools::TensorCollection<Queue>(
                                       featureMajor_ ? maxPreselectedTracks_ : batchSize_, maxPreselectedTracks_)});

      auto& batch = batches.back();
      // Order must match the ONNX model input schema
      batch.inputs.add<PixelTrackFeaturesSoA>("track_features",
                                              i_batch,
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
                                              track_record.covPhiQOverPt());

      batch.outputs.add<PixelTrackScoresSoA>("track_scores", i_batch, score_record.score());

      if (featureMajor_) {
        // Valid only for the complete collection, including its SoA alignment padding.
        // The corresponding ONNX graph must accept [17, padded_batch] and
        // produce [1, padded_batch], not sample-major tensors.
        batch.inputs.set_layout("track_features", cms::Ort::alpakatools::Layout::FeatureMajor);
        batch.outputs.set_layout("track_scores", cms::Ort::alpakatools::Layout::FeatureMajor);
      }
      {
        CMS_PERFETTO_SCOPE("Forward call");
        model_.forward(queue, batch.inputs, batch.outputs);
      }
    }

    {
      CMS_PERFETTO_SCOPE("Score Filter");
      launchScoreFilter(queue,
                        maxPreselectedTracks_,
                        scoreThreshold_,
                        trackScoresOnDevice.view(),
                        alpaka::getPtrNative(d_preselectedTrackIndices),
                        alpaka::getPtrNative(d_nPreselectedTracks),
                        alpaka::getPtrNative(d_trackHitCounts),
                        alpaka::getPtrNative(d_selectedTrackIndices),
                        alpaka::getPtrNative(d_nSelectedTracks),
                        alpaka::getPtrNative(d_selectedTrackHitOffsets));
    }
#ifdef PIXEL_TRACK_HP_DEBUG
    nSelectedTracks = fetchNumSelectedTracks();
    std::cout << "PixelTrackONNXHighPuritySelector::Filtered tracks=" << nSelectedTracks << "\n";
#endif
    TkSoADevice tracks_out(queue, int(maxPreselectedTracks_), int(maxPreselectedTracks_ * avgHitsPerTrack_));
    {
      CMS_PERFETTO_SCOPE("Output Tracks");
      tracks_out = launchProduceOutputTracks(queue,
                                             maxPreselectedTracks_,
                                             avgHitsPerTrack_,
                                             tracks.tracks(),
                                             tracks.trackHits(),
                                             alpaka::getPtrNative(d_selectedTrackIndices),
                                             alpaka::getPtrNative(d_nSelectedTracks),
                                             alpaka::getPtrNative(d_selectedTrackHitOffsets));
    }

    iEvent.emplace(tokenTrackOut_, std::move(tracks_out));
  }

  void PixelTrackONNXHighPuritySelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("pixelTrackSrc", {"hltPhase2PixelTracksSoA"});
    desc.add<int>("maxNumberOfTracks", 100000);
    desc.add<int>("maxPreselectedTracks", 10000);
    desc.add<int>("minNumberOfHits", 0);
    desc.add<int>("avgHitsPerTrack", 8);
    desc.add<std::string>("minimumTrackQuality", "tight");
    desc.add<edm::FileInPath>("model");
    desc.add<std::string>("backend", "auto");
    desc.add<double>("scoreThreshold", 0.5);
    desc.add<int>("batchSize", 10);
    desc.add<bool>("featureMajor", false);

    descriptions.addWithDefaultLabel(desc);
  }

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

DEFINE_FWK_ALPAKA_MODULE(PixelTrackONNXHighPuritySelector);
