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
#include <optional>
#include <nvtx3/nvtx3.hpp>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
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

//#define PIXEL_TRACK_HP_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct BatchIO {
    cms::torch::alpakatools::TensorCollection<Queue> inputs;
    cms::torch::alpakatools::TensorCollection<Queue> outputs;
  };

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
    const int32_t batchSize_;
    const bool to_half_;
    const bool use_cudaGraphs_; 
    bool graph_ready_ = false;  
    bool model_warmed_up_ = false;  
    const int warmup_iterations_ = 5; // Number of iterations to warm up the model
    int n_batches_;
    std::deque<BatchIO> batches_;
    std::optional<Queue> torchQueue_;
    std::optional<Event> featuresReadyEvent_;
    std::optional<Event> inferenceDoneEvent_;
    std::optional<cms::alpakatools::device_buffer<Device, int>> d_nPreselectedTracks_;
    std::optional<cms::alpakatools::device_buffer<Device, int>> d_nSelectedTracks_;
    std::optional<cms::alpakatools::device_buffer<Device, int[]>> d_preselectedTrackIndices_;
    std::optional<cms::alpakatools::device_buffer<Device, int[]>> d_selectedTrackIndices_;
    std::optional<cms::alpakatools::device_buffer<Device, int[]>> d_nKeptHits_;
    std::optional<cms::alpakatools::device_buffer<Device, int[]>> d_preselectionOffsets_;
    std::optional<PixelTrackFeaturesOnDevice>  trackFeatures_;
    std::optional<PixelRecHitFeaturesOnDevice> hitFeatures_;
    std::optional<PixelTrackScoresOnDevice>    trackScoresOnDevice_;
    #ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    ::at::cuda::CUDAGraph inference_graph_;
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
        tokenTrackOut_(produces()),
        batchSize_(iConfig.getParameter<int>("batchSize")),
        to_half_(iConfig.getParameter<bool>("toHalf")),
	use_cudaGraphs_(iConfig.getParameter<bool>("useCudaGraphs"))

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
    n_batches_ = maxPreselectedTracks_ / batchSize_;
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
    nvtx3::scoped_range range{"PixelTrackTorchHP Producer"};
    // Retrieve tokens
    auto&       queue  = iEvent.queue();
    const auto& hits   = iEvent.get(recHitToken_).view();
    const auto& tracks = iEvent.get(pixelTrackToken_).view();

    // If not create yet, create an alpaka queue for Torch and associate it with the current device
    if (!torchQueue_) {
      torchQueue_.emplace(alpaka::getDev(queue));
      featuresReadyEvent_.emplace(alpaka::getDev(queue));
      inferenceDoneEvent_.emplace(alpaka::getDev(queue));
    }
    alpaka::enqueue(queue, *featuresReadyEvent_);
    alpaka::wait(*torchQueue_, *featuresReadyEvent_);

    // Instantiate the necessary objects in memory
    //  - Temporary storage for filtering
    if (!d_nPreselectedTracks_) {
      d_nPreselectedTracks_.emplace(cms::alpakatools::make_device_buffer<int>(*torchQueue_));
      d_nSelectedTracks_.emplace(cms::alpakatools::make_device_buffer<int>(*torchQueue_));
      d_preselectedTrackIndices_.emplace(cms::alpakatools::make_device_buffer<int[]>(*torchQueue_, maxNumberOfTracks_));
      d_selectedTrackIndices_.emplace(cms::alpakatools::make_device_buffer<int[]>(*torchQueue_, maxPreselectedTracks_));
      d_nKeptHits_.emplace(cms::alpakatools::make_device_buffer<int[]>(*torchQueue_, maxPreselectedTracks_));
      d_preselectionOffsets_.emplace(cms::alpakatools::make_device_buffer<int[]>(*torchQueue_, maxNumberOfTracks_));
      trackFeatures_.emplace(*torchQueue_, maxPreselectedTracks_);
      hitFeatures_.emplace(*torchQueue_, maxPreselectedTracks_);
      trackScoresOnDevice_.emplace(*torchQueue_, maxPreselectedTracks_);
      // - Tensor collections for DNN inference  
      for(auto i=0; i<n_batches_; i++){
        batches_.emplace_back(BatchIO{
          cms::torch::alpakatools::TensorCollection<Queue>(batchSize_),
          cms::torch::alpakatools::TensorCollection<Queue>(batchSize_)});
        auto offset = i * batchSize_;
        auto track_record = trackFeatures_->view().records();
        auto hit_record = hitFeatures_->view().records();
        auto score_record = trackScoresOnDevice_->view().records();
        auto& batch = batches_[i];
        batch.inputs.add<::RecHitFeatures::PixelRecHitFeaturesSoA>(
                "hit_features", 
                batchSize_, 
                maxPreselectedTracks_, 
                offset,
                hit_record.hits()
              );
        // Order must match the TorchScript model input schema
        batch.inputs.add<PixelTrackFeaturesSoA>(
            "track_features",
            batchSize_,
            maxPreselectedTracks_,
            offset,
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

          batch.outputs.add<PixelTrackScoresSoA>(
            "track_scores",
            batchSize_,
            maxPreselectedTracks_,
            offset,
            score_record.score()
          );
      }
    }

    alpaka::memset(*torchQueue_, *d_nPreselectedTracks_, 0);
    alpaka::memset(*torchQueue_, *d_nSelectedTracks_, 0);
    alpaka::memset(*torchQueue_, *d_nKeptHits_, 0);
    alpaka::memset(*torchQueue_, *d_preselectedTrackIndices_, 0xFF);
    alpaka::memset(*torchQueue_, *d_selectedTrackIndices_, 0xFF);
    alpaka::memset(*torchQueue_, *d_preselectionOffsets_, 0);

    // Optional debug definitions
#ifdef PIXEL_TRACK_HP_DEBUG
    auto h_nPreselectedTracks  = cms::alpakatools::make_host_buffer<int>(*torchQueue_);
    auto h_nSelectedTracks     = cms::alpakatools::make_host_buffer<int>(*torchQueue_);
    int nPreselectedTracks     = 0;
    int nSelectedTracks        = 0;
    // Helper to copy the number of kept tracks back to host (debug only)
    auto fetchnPreselectedTracks = [&]() {
      alpaka::memcpy(*torchQueue_, h_nPreselectedTracks, *d_nPreselectedTracks_);
      alpaka::wait(*torchQueue_);
      return *h_nPreselectedTracks;
    };
    auto fetchnSelectedTracks = [&]() {
      alpaka::memcpy(*torchQueue_, h_nSelectedTracks, *d_nSelectedTracks_);
      alpaka::wait(*torchQueue_);
      return *h_nSelectedTracks;
    };
#endif

    // 1. CA-based preselection of tracks
    //  Launch first kernel to look which tracks need to be filtered out
    //  based on quality criteria from the CA
    {
      nvtx3::scoped_range range1{"CAPreselection"};
      launchCAPreselection(
        *torchQueue_,
        maxNumberOfTracks_,
        minNumberOfHits_,
        minimumTrackQuality_,
        tracks.tracks(),
        alpaka::getPtrNative(*d_preselectedTrackIndices_),
        alpaka::getPtrNative(*d_preselectionOffsets_),
        alpaka::getPtrNative(*d_nPreselectedTracks_)
      );
    }

#ifdef PIXEL_TRACK_HP_DEBUG
    nPreselectedTracks = fetchnPreselectedTracks();
    std::cout << "PixelTrackTorchHighPuritySelector::Prefiltered tracks=" << nPreselectedTracks << "\n";
#endif

    // 2. Feature extraction (track + hit SoA)
    {
      nvtx3::scoped_range range2{"FeaturesExtractor"};
      launchFeaturesExtractor(
        *torchQueue_,
        maxPreselectedTracks_,
        tracks.tracks(),
        tracks.trackHits(),
        hits.trackingHits(),
        alpaka::getPtrNative(*d_preselectedTrackIndices_),
        alpaka::getPtrNative(*d_nPreselectedTracks_),
        trackFeatures_->view(),
        hitFeatures_->view(),
        alpaka::getPtrNative(*d_nKeptHits_)
      );
    }

    // 3. DNN inference
    //  Prepare TensorCollection inputs and outputs for the model
    if(!model_warmed_up_){
      nvtx3::scoped_range range_warmup{"ModelWarmup"};
      // Run a dummy inference to warm up the model
      for(auto i=0; i<warmup_iterations_; i++){
        for(auto& batch : batches_){
          model_.forward(*torchQueue_, batch.inputs, batch.outputs, to_half_);
        }
      }
      alpaka::wait(*torchQueue_);
      model_warmed_up_ = true;  
    }

  #ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    if(!graph_ready_&&use_cudaGraphs_){
      // If using CUDA, we can capture the inference in a CUDA graph for faster subsequent execution
      auto stream = c10::cuda::getStreamFromExternal(
          torchQueue_->getNativeHandle(),
          cms::torch::alpakatools::getDevice(*torchQueue_).index()
      );

      ::at::cuda::setCurrentCUDAStream(stream);

      inference_graph_.capture_begin(/*pool=*/{}, cudaStreamCaptureModeThreadLocal);

      for (auto& batch : batches_) {
        model_.forward(*torchQueue_, batch.inputs, batch.outputs, to_half_);
      }

      inference_graph_.capture_end();

      graph_ready_ = true;
    }
  #endif
  
    // - Tensor collections for DNN inference
    #ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      if(graph_ready_&&use_cudaGraphs_){
          nvtx3::scoped_range range_graph{"CudaGraphReplay"};
          auto stream = c10::cuda::getStreamFromExternal(
          torchQueue_->getNativeHandle(),
              cms::torch::alpakatools::getDevice(*torchQueue_).index()
          );

          ::at::cuda::setCurrentCUDAStream(stream);
          inference_graph_.replay();
      }
      else{
          nvtx3::scoped_range range_no_graph{"NoCudaGraph"};
          for(auto& batch : batches_){
            model_.forward(*torchQueue_, batch.inputs, batch.outputs, to_half_);
          }
      }
    #else
      for(auto& batch : batches_){
        nvtx3::scoped_range range3{"DNNInference"};
        model_.forward(*torchQueue_, batch.inputs, batch.outputs, to_half_);
      }
    #endif
    // 4. Score-based filtering
    {
      nvtx3::scoped_range range4{"ScoreFilter"};
      launchScoreFilter(
        *torchQueue_,
        maxPreselectedTracks_,
        scoreThreshold_,
        trackScoresOnDevice_->view(),
        alpaka::getPtrNative(*d_preselectedTrackIndices_),
        alpaka::getPtrNative(*d_nPreselectedTracks_),
        alpaka::getPtrNative(*d_selectedTrackIndices_),
        alpaka::getPtrNative(*d_nSelectedTracks_),
        alpaka::getPtrNative(*d_nKeptHits_)
      );
    }

#ifdef PIXEL_TRACK_HP_DEBUG    
    nSelectedTracks = fetchnSelectedTracks();
    std::cout << "PixelTrackTorchHighPuritySelector::Filtered tracks=" << nSelectedTracks << "\n";
#endif
    {
      nvtx3::scoped_range range5{"OutputCompaction"};
      auto tracks_out = launchProduceOutputTracks (
          *torchQueue_,
          maxPreselectedTracks_,
          avgHitsPerTrack_,
          tracks.tracks(),
          tracks.trackHits(), 
          alpaka::getPtrNative(*d_selectedTrackIndices_),
          alpaka::getPtrNative(*d_nSelectedTracks_),
          alpaka::getPtrNative(*d_nKeptHits_)
        );
      // Record that inference is done on the Torch stream
      alpaka::enqueue(*torchQueue_, *inferenceDoneEvent_);
      // Make the Alpaka stream wait until inference is done
      alpaka::wait(queue, *inferenceDoneEvent_);
      iEvent.emplace(tokenTrackOut_, std::move(tracks_out));
    }
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
    desc.add<int>("batchSize", 10);
    desc.add<bool>("toHalf", false);
    desc.add<bool>("useCudaGraphs", false);
    descriptions.addWithDefaultLabel(desc);
  }
};

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PixelTrackTorchHighPuritySelector);
